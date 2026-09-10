import os
from check_deps import ensure_dependencies
ensure_dependencies()

import time
import json
import base64
import asyncio
from io import BytesIO
try:
    import config  # Автозагрузка .env файла (если присутствует)
except ImportError:
    pass

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import argparse
from model import EdgeFlowPhaseV2
from encoders import TextEncoder, SDVAE, SDXLVAE, TAESD
from rectified_flow import RectifiedFlow4D
from download_utils import (
    ensure_text_encoder,
    ensure_vae,
    ensure_checkpoint,
    get_checkpoints_dir,
    find_available_checkpoints,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

parser = argparse.ArgumentParser()
parser.add_argument("--vae", type=str, default="taesd", choices=["taesd", "sd", "sdxl"])
parser.add_argument("--ckpt-dir", type=str, default=None, help="Path to checkpoints directory")
parser.add_argument("--host", type=str, default=None, help="Host to bind server")
parser.add_argument("--port", type=int, default=None, help="Port to bind server")
args, _ = parser.parse_known_args()

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

app = FastAPI()

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelState:
    def __init__(self):
        self.device = "cpu"
        self.amp_dtype = torch.float32
        
        self.text_encoder = None
        self.vae = None
        self.rf = None
        
        self.current_model = None
        self.current_ckpt_path = None
        self.latent_mean = None
        self.latent_std = None
        self.loaded = False
        self._device_switched = True

    def get_amp_dtype(self, device):
        if str(device).startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def load_base(self):
        if not self.loaded:
            print("Loading base components (checking/downloading if needed)...")
            ensure_text_encoder()
            vae_type = args.vae.lower()
            ensure_vae(vae_type)

            self.text_encoder = TextEncoder(
                out_dim=768,
                include_encoder=True,
                hidden_size=512,
                max_length=128,
            )
            self.text_encoder.eval()

            if vae_type == "taesd":
                self.vae = TAESD()
            elif vae_type == "sdxl":
                self.vae = SDXLVAE()
            else:
                self.vae = SDVAE()
            self.vae.eval()

            self.rf = RectifiedFlow4D()
            self.loaded = True

    def load_checkpoint(self, checkpoint_path: str, target_device: str, use_ema: bool = True):
        self.load_base()
        
        if self.current_ckpt_path != checkpoint_path:
            import gc
            self.current_model = None
            gc.collect()
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"Loading checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            config = checkpoint.get("config", {})
            if not config:
                config = dict(
                    latent_channels=4,
                    patch_size=2,
                    dim=768,
                    depth=8,
                    heads=12,
                    kv_heads=2,
                    context_dim=768,
                    num_phases=4,
                    ffn_hidden=1024,
                    ada_rank=64,
                    cross_every=2,
                    expert_hidden_list=[768, 768, 768, 768]
                )
            model = EdgeFlowPhaseV2(**config).to(target_device)

            ema_key = None
            if use_ema:
                if "ema_model_state_dict" in checkpoint:
                    ema_key = "ema_model_state_dict"
                elif "ema_state_dict" in checkpoint:
                    ema_key = "ema_state_dict"

            if ema_key is not None:
                state_dict = checkpoint[ema_key]
                print(f"Loaded EMA weights ({ema_key})")
            else:
                state_dict = checkpoint.get("model_state_dict", {})
                if checkpoint.get("is_ema"):
                    print("Loaded EMA weights (from pure EMA checkpoint)")
                else:
                    print("Loaded model_state_dict")

            def clean_dict(d):
                cd = {}
                for k, v in d.items():
                    if k.endswith("n_averaged"): continue
                    ck = k
                    while ck.startswith("module."): ck = ck[7:]
                    while ck.startswith("_orig_mod."): ck = ck[10:]
                    cd[ck] = v
                return cd

            clean_state = clean_dict(state_dict)

            if "model_scales" in checkpoint and checkpoint["model_scales"]:
                scales = checkpoint["model_scales"]
                for k, v in clean_state.items():
                    if torch.is_tensor(v) and v.dtype == torch.int8:
                        sc = scales.get(k, 1.0)
                        clean_state[k] = v.to(torch.float16) * (sc.to(torch.float16) if torch.is_tensor(sc) else sc)
                print("Dequantized INT8 model weights to FP16")

            # Auto-cast model to fp16/bf16 if weights are 16-bit
            first_weight = next(iter(clean_state.values()), None)
            if first_weight is not None and str(target_device).startswith("cuda"):
                if first_weight.dtype in (torch.float16, torch.bfloat16):
                    model = model.to(first_weight.dtype)
                    print(f"Model placed in {first_weight.dtype}")
                elif "float8" in str(first_weight.dtype):
                    model = model.to(torch.float16)
                    print(f"Model placed in torch.float16 (unpacked from FP8)")

            model.load_state_dict(clean_state, strict=False)
            
            model.eval()
            self.current_model = model
            self.current_ckpt_path = checkpoint_path

            def dequant_sub(d, scales):
                if not scales:
                    return d
                return {
                    k: (v.to(torch.float16) * (scales.get(k, 1.0).to(torch.float16) if torch.is_tensor(scales.get(k, 1.0)) else scales.get(k, 1.0)))
                    if (torch.is_tensor(v) and v.dtype == torch.int8) else v
                    for k, v in d.items()
                }

            if use_ema and "ema_text_proj_state_dict" in checkpoint:
                self.text_encoder.proj.load_state_dict(clean_dict(checkpoint["ema_text_proj_state_dict"]), strict=False)
            elif "text_encoder_state_dict" in checkpoint and checkpoint["text_encoder_state_dict"] is not None:
                p_dict = clean_dict(checkpoint["text_encoder_state_dict"])
                p_dict = dequant_sub(p_dict, checkpoint.get("text_encoder_scales"))
                self.text_encoder.proj.load_state_dict(p_dict, strict=False)
                
            if use_ema and "ema_text_pooler_state_dict" in checkpoint:
                self.text_encoder.pooler.load_state_dict(clean_dict(checkpoint["ema_text_pooler_state_dict"]), strict=False)
            elif "text_encoder_pooler_state_dict" in checkpoint and checkpoint["text_encoder_pooler_state_dict"] is not None:
                pl_dict = clean_dict(checkpoint["text_encoder_pooler_state_dict"])
                pl_dict = dequant_sub(pl_dict, checkpoint.get("text_encoder_pooler_scales"))
                self.text_encoder.pooler.load_state_dict(pl_dict, strict=False)
            
            if "latent_mean" in checkpoint and checkpoint["latent_mean"] is not None:
                self.latent_mean = checkpoint["latent_mean"].to(self.device).view(1, 4, 1, 1).clone()
                self.latent_std = checkpoint["latent_std"].to(self.device).view(1, 4, 1, 1).clone()
            else:
                self.latent_mean = torch.tensor([0.0]*4, dtype=torch.float32, device=self.device).view(1, 4, 1, 1)
                self.latent_std = torch.tensor([1.0]*4, dtype=torch.float32, device=self.device).view(1, 4, 1, 1)

            del checkpoint
            if 'ema_state' in locals():
                del ema_state
            if 'ema_text_state' in locals():
                del ema_text_state
            if 'ema_pooler_state' in locals():
                del ema_pooler_state
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._device_switched = True

        if self.device != target_device or self._device_switched:
            print(f"Moving models to {target_device}...")
            self.text_encoder = self.text_encoder.to(target_device)
            self.vae = self.vae.to(target_device)
            if self.current_model is not None:
                self.current_model = self.current_model.to(target_device)
            self.device = target_device
            if self.latent_mean is not None:
                self.latent_mean = self.latent_mean.to(target_device)
                self.latent_std = self.latent_std.to(target_device)
            self.amp_dtype = self.get_amp_dtype(target_device)
            self._device_switched = False

state = ModelState()
import threading
gpu_lock = threading.Lock()

@app.get("/")
def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>PEDiT Server</h1><p>Static index.html not found.</p>")

@app.get("/models")
def get_models():
    cp_dir = get_checkpoints_dir(getattr(args, "ckpt_dir", None))
    models = find_available_checkpoints(cp_dir, always_include_standard=True)
    return {"models": models}

@app.websocket("/ws/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_str = await websocket.receive_text()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
                continue
            
            prompt = data.get("prompt", "")
            model_name = os.path.basename(data.get("model", ""))
            steps = int(data.get("steps", 8))
            cfg_scale = float(data.get("cfg", 4.0))
            raw_seed = data.get("seed", 42)
            try:
                seed = int(raw_seed) if raw_seed is not None and raw_seed != "" else None
            except (ValueError, TypeError):
                seed = 42
                
            fast_generation = data.get("fast_generation", False)
                
            width = int(data.get("width", 1024))
            height = int(data.get("height", 1024))
            
            width = (width // 32) * 32
            height = (height // 32) * 32
            
            target_device = data.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            if str(target_device).startswith("cuda"):
                if not torch.cuda.is_available():
                    target_device = "cpu"
                else:
                    try:
                        torch.zeros(1, device=target_device)
                    except Exception:
                        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            if not model_name:
                await websocket.send_json({"error": "No model selected"})
                continue
                
            cp_dir = get_checkpoints_dir(getattr(args, "ckpt_dir", None))
            checkpoint_path = os.path.join(cp_dir, model_name)
            if not os.path.exists(checkpoint_path):
                await websocket.send_json({
                    "status": "downloading",
                    "message": f"Downloading {model_name} from Hugging Face..."
                })
                resolved = await asyncio.to_thread(ensure_checkpoint, model_name, cp_dir)
                if resolved and os.path.exists(resolved):
                    checkpoint_path = resolved
                else:
                    await websocket.send_json({"error": f"Model '{model_name}' could not be downloaded from Hugging Face."})
                    continue
            import random
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
                
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            if True:
                try:
                    def _locked_load():
                        with gpu_lock:
                            state.load_checkpoint(checkpoint_path, target_device)
                    await asyncio.shield(asyncio.to_thread(_locked_load))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"error": f"Error loading model: {str(e)}. The file might be corrupted."})
                    continue

                await websocket.send_json({"status": "starting"})
                
                total_start_time = time.time()
                loop = asyncio.get_running_loop()
                queue = asyncio.Queue(maxsize=4)
                import threading
                cancel_event = threading.Event()

                def _sync_generation_inner():
                    try:
                        def send_msg(msg):
                            if cancel_event.is_set():
                                return
                            def _put():
                                try:
                                    queue.put_nowait(msg)
                                except Exception:
                                    pass
                            loop.call_soon_threadsafe(_put)

                        total_model_time_ms = 0
                        total_vae_time_ms = 0
                        
                        dev_type = "cuda" if str(state.device).startswith("cuda") else "cpu"
                        with torch.autocast(device_type=dev_type, dtype=state.amp_dtype, enabled=(dev_type == "cuda")), torch.inference_mode():
                            prompts = ["", prompt]
                            cond_pooled, cond_seq, mask = state.text_encoder(prompts, state.device)
                            text_kvs = state.current_model.encode_text(cond_seq)
                            
                            latent_h = height // 8
                            latent_w = width // 8
                            shape = (1, 4, latent_h, latent_w)
                            
                            gen = torch.Generator(device=state.device).manual_seed(seed) if seed is not None else None
                            
                            def step_callback(step, x_1_pred, step_time_ms):
                                nonlocal total_model_time_ms, total_vae_time_ms
                                if cancel_event.is_set():
                                    raise InterruptedError("Cancelled")
                                
                                total_model_time_ms += step_time_ms
                                total_time_ms = int((time.time() - total_start_time) * 1000)

                                if fast_generation and step < steps - 1:
                                    send_msg({
                                        "step": step + 1,
                                        "model_time_ms": step_time_ms,
                                        "vae_time_ms": 0,
                                        "step_time_ms": step_time_ms,
                                        "total_time_ms": total_time_ms
                                    })
                                    return

                                vae_start = time.perf_counter()
                                x_1_pred = (x_1_pred * state.latent_std + state.latent_mean).float()
                                with torch.autocast(device_type=dev_type, enabled=False):
                                    image_tensor = state.vae.decode(x_1_pred)
                                image_tensor = (image_tensor.clamp(0.0, 1.0) * 255).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
                                img = Image.fromarray(image_tensor)
                                vae_time_ms = int((time.perf_counter() - vae_start) * 1000)
                                total_vae_time_ms += vae_time_ms
                                
                                with BytesIO() as buffered:
                                    img.save(buffered, format="JPEG")
                                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                    buffered.close()
                                
                                send_msg({
                                    "step": step + 1,
                                    "image": img_str,
                                    "model_time_ms": step_time_ms,
                                    "vae_time_ms": vae_time_ms,
                                    "step_time_ms": step_time_ms + vae_time_ms,
                                    "total_time_ms": int((time.time() - total_start_time) * 1000)
                                })
                                del buffered, img_str

                            state.rf.phase_euler_sample(
                                model=state.current_model,
                                cond_pooled=cond_pooled,
                                cond_seq=cond_seq,
                                shape=shape,
                                device=state.device,
                                steps=steps,
                                mask=mask,
                                text_kvs=text_kvs,
                                cfg_scale=cfg_scale,
                                generator=gen,
                                callback=step_callback
                            )

                        send_msg({
                            "status": "done",
                            "total_model_time_ms": total_model_time_ms,
                            "total_vae_time_ms": total_vae_time_ms,
                            "total_time_ms": int((time.time() - total_start_time) * 1000)
                        })

                    except InterruptedError:
                        pass
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        send_msg({"error": str(e)})
                    finally:
                        send_msg(None)

                def _sync_generation():
                    with gpu_lock:
                        _sync_generation_inner()

                gen_task = asyncio.create_task(asyncio.to_thread(_sync_generation))

                try:
                    has_error = False
                    while True:
                        msg = await queue.get()
                        if msg is None:
                            break
                        if isinstance(msg, dict) and "error" in msg:
                            has_error = True
                        await websocket.send_json(msg)
                finally:
                    cancel_event.set()
                    await gen_task

    except (WebSocketDisconnect, RuntimeError, ConnectionResetError):
        print("Client disconnected")

if __name__ == "__main__":
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="A highly detailed portrait", help="Prompt")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="outputs")
    
    args, _ = parser.parse_known_args()

    if args.cli:
        resolved_ckpt = ensure_checkpoint(args.ckpt, ckpt_dir=args.ckpt_dir)
        if not resolved_ckpt or not os.path.exists(resolved_ckpt):
            print("Error: No checkpoint found. Please specify --ckpt <path/url> or place a checkpoint in the checkpoints folder.")
            exit(1)
            
        import random
        import time
        os.makedirs(args.outdir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading checkpoint {resolved_ckpt}...")
        state.load_checkpoint(resolved_ckpt, device)
        
        print(f"Generating image for prompt: '{args.prompt}'...")
        dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
        
        with torch.autocast(device_type=dev_type, dtype=state.amp_dtype, enabled=(dev_type == "cuda")), torch.inference_mode():
            prompts = ["", args.prompt]
            cond_pooled, cond_seq, mask = state.text_encoder(prompts, state.device)
            text_kvs = state.current_model.encode_text(cond_seq)
            
            latent_h = args.height // 8
            latent_w = args.width // 8
            shape = (1, 4, latent_h, latent_w)
            
            gen = torch.Generator(device=state.device).manual_seed(args.seed) if args.seed is not None else None
            
            def step_callback(step, x_1_pred, step_time_ms):
                print(f"Step {step+1}/{args.steps} ({step_time_ms:.1f}ms)")
                
                if step == args.steps - 1:
                    vae_start = time.perf_counter()
                    x_1_p = (x_1_pred * state.latent_std + state.latent_mean).float()
                    with torch.autocast(device_type=dev_type, enabled=False):
                        image_tensor = state.vae.decode(x_1_p)
                    image_tensor = (image_tensor.clamp(0.0, 1.0) * 255).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
                    img = Image.fromarray(image_tensor)
                    
                    filename = f"out_{int(time.time())}_seed{args.seed}.jpg"
                    out_path = os.path.join(args.outdir, filename)
                    img.save(out_path, format="JPEG", quality=95)
                    print(f"Saved to {out_path}")

            print("Starting sampling...")
            t0 = time.time()
            state.rf.phase_euler_sample(
                model=state.current_model,
                cond_pooled=cond_pooled,
                cond_seq=cond_seq,
                shape=shape,
                device=state.device,
                steps=args.steps,
                mask=mask,
                text_kvs=text_kvs,
                cfg_scale=args.cfg,
                generator=gen,
                callback=step_callback
            )
            print(f"Generation finished in {time.time()-t0:.2f}s")
    else:
        import uvicorn
        import logging

        host = args.host or os.environ.get("HOST", "0.0.0.0")
        port = int(args.port or os.environ.get("PORT", 8000))
        display_host = "localhost" if host in ("0.0.0.0", "::") else host

        class LocalhostFilter(logging.Filter):
            def filter(self, record):
                if hasattr(record, "msg") and isinstance(record.msg, str):
                    record.msg = record.msg.replace("0.0.0.0", "localhost")
                if hasattr(record, "args") and record.args:
                    record.args = tuple("localhost" if a == "0.0.0.0" else a for a in record.args)
                return True

        for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(logger_name).addFilter(LocalhostFilter())

        print(f"Starting PEDiT server on http://{display_host}:{port}")
        uvicorn.run(app, host=host, port=port, reload=False)
