import os
import sys
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.abspath(os.path.join(_current_dir, ".."))
_model_dir = os.path.abspath(os.path.join(_repo_dir, "model"))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import config
import argparse
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_scalar_outputs = True

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import torchvision
from PIL import Image

from model import EdgeFlowPhaseV2
from streamer import ChunkedWeightStreamer, ChunkedResidencyManager, DEFAULT_CHUNK_SIZE_MB
from encoders import TextEncoder, DCAE, tiled_decode_latents
from rectified_flow import RectifiedFlow4D
from sampler import PhaseAwareSampler
from config import get_model_config, get_desktop_hardware_profile

HW_PROFILE = get_desktop_hardware_profile()


def reconcile_model_config_with_state_dict(
    config: Dict[str, Any],
    state_dict: Dict[str, Any],
    scale: str = "4b",
    steps: int = 12,
    experts_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Согласует конфигурацию модели с фактическими тензорами в state_dict,
    обеспечивая 100% поддержку архитектуры V4/V5 (и обратную совместимость с 24L/30L/100M).
    """
    base_cfg = dict(get_model_config(scale, steps=steps))
    if config and "dim" in config:
        base_cfg.update(config)

    if not state_dict:
        return base_cfg

    # Очистка префиксов module. / _orig_mod. для инспекции ключей
    norm_sd = {}
    for k, v in state_dict.items():
        ck = k
        while ck.startswith("module."):
            ck = ck[7:]
        while ck.startswith("_orig_mod."):
            ck = ck[10:]
        norm_sd[ck] = v

    if "norm_final.weight" in norm_sd:
        base_cfg["dim"] = int(norm_sd["norm_final.weight"].shape[0])
    elif "time_norm.weight" in norm_sd:
        base_cfg["dim"] = int(norm_sd["time_norm.weight"].shape[0])

    block_indices = [
        int(k.split(".")[1])
        for k in norm_sd.keys()
        if k.startswith("blocks.") and k.split(".")[1].isdigit()
    ]
    if block_indices:
        base_cfg["depth"] = max(block_indices) + 1

    if "blocks.0.attn.q_norm.weight" in norm_sd:
        head_dim = int(norm_sd["blocks.0.attn.q_norm.weight"].shape[0])
        base_cfg["head_dim"] = head_dim
        if "blocks.0.attn.q_proj.weight" in norm_sd:
            base_cfg["heads"] = int(norm_sd["blocks.0.attn.q_proj.weight"].shape[0]) // head_dim
        if "blocks.0.attn.k_proj.weight" in norm_sd:
            base_cfg["kv_heads"] = int(norm_sd["blocks.0.attn.k_proj.weight"].shape[0]) // head_dim

    if "blocks.0.ff.w1.weight" in norm_sd:
        base_cfg["ffn_hidden"] = int(norm_sd["blocks.0.ff.w1.weight"].shape[0])

    # Определение ada_rank по размерности первого линейного слоя adaLN_modulation
    dim = base_cfg.get("dim", 1536)
    for cand_k in ("blocks.0.adaLN_modulation.1.weight", "blocks.0.adaLN_modulation.0.weight"):
        if cand_k in norm_sd:
            r_cand = int(norm_sd[cand_k].shape[0])
            base_cfg["ada_rank"] = r_cand if r_cand < 8 * dim else None
            break

    # Cross-LoRA и Phase-LoRA
    has_cross_lora = any(".cross_lora." in k for k in norm_sd.keys())
    base_cfg["use_cross_lora"] = has_cross_lora
    if has_cross_lora and "blocks.0.cross.cross_lora.k_a.weight" in norm_sd:
        base_cfg["cross_lora_rank"] = int(norm_sd["blocks.0.cross.cross_lora.k_a.weight"].shape[0])

    has_phase_lora = any("phase_attn_lora." in k for k in norm_sd.keys())
    if has_phase_lora:
        base_cfg["use_phase_lora"] = True
        for k, v in norm_sd.items():
            if "phase_attn_lora." in k and k.endswith(".q_a.weight"):
                base_cfg["phase_lora_rank"] = int(v.shape[0])
                break
        base_cfg["use_phase_k_lora"] = any("phase_attn_lora." in k and ".k_a." in k for k in norm_sd.keys())

    # Архитектурные флаги V4/V5
    base_cfg["num_registers"] = int(norm_sd["registers"].shape[1]) if "registers" in norm_sd else 0
    base_cfg["use_dw_conv"] = any(".dw_conv." in k for k in norm_sd.keys())
    base_cfg["use_skip_connections"] = any("skip_projs." in k for k in norm_sd.keys())
    base_cfg["learnable_head_scale"] = any(".head_scale" in k for k in norm_sd.keys())
    base_cfg["use_value_residual"] = any(".v_res_lambda" in k for k in norm_sd.keys())
    base_cfg["use_cross_sink"] = any(".sink_k" in k for k in norm_sd.keys())
    base_cfg["use_head_gate"] = any(".head_gate." in k for k in norm_sd.keys())
    base_cfg["use_hf_booster"] = any(".hf_gain" in k for k in norm_sd.keys())
    base_cfg["use_refinement_head"] = any("refinement_head." in k for k in norm_sd.keys())

    if "proj_in.weight" in norm_sd:
        in_ch = int(norm_sd["proj_in.weight"].shape[1])
        lat_ch = base_cfg.get("latent_channels", 32)
        base_cfg["use_coord_conv"] = (in_ch == lat_ch + 3)

    # Размерности и число активных слоев каскадных фазовых экспертов
    num_phases = int(base_cfg.get("num_phases", 3))
    depth = int(base_cfg.get("depth", 12))
    exp_hiddens = []
    exp_layer_counts = []
    for p in range(num_phases):
        active_blocks_for_p = [
            b_i for b_i in range(depth)
            if f"blocks.{b_i}.phase_experts.{p}.w1.weight" in norm_sd
        ]
        if active_blocks_for_p:
            first_b = active_blocks_for_p[0]
            exp_hiddens.append(int(norm_sd[f"blocks.{first_b}.phase_experts.{p}.w1.weight"].shape[0]))
            exp_layer_counts.append(len(active_blocks_for_p))
    if len(exp_hiddens) == num_phases:
        base_cfg["expert_hidden_list"] = exp_hiddens
        base_cfg["phase_expert_layers"] = tuple(exp_layer_counts)

    return base_cfg


_CACHED_DCAE: Optional[DCAE] = None

def get_dcae(
    device: Union[str, torch.device] = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
    decoder_only: bool = True,
    compile_decoder: bool = False,
    warmup: bool = False,
) -> DCAE:
    """
    Returns persistent cached DCAE instance to eliminate 2.5-3.5s disk reloads across inference runs.
    """
    global _CACHED_DCAE
    target_dev = torch.device(device)
    target_dtype = torch_dtype or (torch.float16 if str(target_dev).startswith("cuda") else torch.float32)

    if _CACHED_DCAE is None:
        _CACHED_DCAE = DCAE(
            torch_dtype=target_dtype,
            decoder_only=decoder_only,
        )
    cur_param = next(_CACHED_DCAE.parameters(), None)
    if cur_param is not None and (cur_param.device != target_dev or cur_param.dtype != target_dtype):
        _CACHED_DCAE = _CACHED_DCAE.to(device=target_dev, dtype=target_dtype)
    if compile_decoder and not getattr(_CACHED_DCAE, "_compiled", False):
        _CACHED_DCAE.compile()
    if warmup and str(target_dev).startswith("cuda"):
        try:
            with torch.inference_mode():
                dummy_32 = torch.zeros(1, 32, 32, 32, device=target_dev, dtype=target_dtype)
                _ = _CACHED_DCAE.decode(dummy_32)
                dummy_8 = torch.zeros(1, 32, 8, 8, device=target_dev, dtype=target_dtype)
                _ = _CACHED_DCAE.decode(dummy_8)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception:
            pass
    return _CACHED_DCAE


@torch.inference_mode()
def generate(
    checkpoint_path: str,
    prompt: str,
    output_path: str = "output.png",
    steps: int = 24,
    cfg_scale: float = 1.0,
    cfg_rescale: float = 0.5,
    image_size: int = 512,
    vae_type: str = "dcae",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    use_ema: bool = True,
    solver: str = "phase_ab2",
    compile_model: bool = False,
    compile_threads: Optional[int] = None,
    compile_warmup: bool = True,
    rescale_norm: bool = True,
    shift: float = 1.0,
    model_scale: str = "4b",
    text_8bit: bool = True,
    text_quant: str = "8bit",
    sequential_loading: Optional[bool] = None,
    tiled_decode: Optional[bool] = None,
    tile_size: int = 48,
    tile_overlap: int = 12,
    max_experts_vram: Optional[int] = None,
    max_experts_ram: Optional[int] = None,
    offload_text_encoder: Optional[bool] = None,
    fp8_latents: bool = True,
):
    hw_prof = get_desktop_hardware_profile(device=device)
    if sequential_loading is None:
        sequential_loading = hw_prof["is_desktop_gpu"]
    if sequential_loading:
        if max_experts_vram is None:
            max_experts_vram = 1
        if max_experts_ram is None:
            max_experts_ram = 2
        if offload_text_encoder is None:
            offload_text_encoder = True
    else:
        if max_experts_vram is None:
            max_experts_vram = hw_prof["default_max_experts_vram"]
        if max_experts_ram is None:
            max_experts_ram = hw_prof["default_max_experts_ram"]
        if offload_text_encoder is None:
            offload_text_encoder = False

    max_vram = max_experts_vram
    max_ram = max_experts_ram

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})
    ema_key = "ema_state_dict" if "ema_state_dict" in checkpoint else "ema_model_state_dict"
    if use_ema and ema_key in checkpoint:
        raw_sd = checkpoint[ema_key]
        print("Using EMA model weights.")
    else:
        raw_sd = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
        print("Using base model weights.")

    quant_mode = config.get("quantization", None)
    if quant_mode is None:
        has_fp8_tensors = any("float8" in str(v.dtype) for v in raw_sd.values())
        if has_fp8_tensors or "fp8" in checkpoint_path.lower():
            quant_mode = "fp8"
        elif any(k.endswith(".scale") for k in raw_sd.keys()) or "int8" in checkpoint_path.lower():
            quant_mode = "int8"

    quantize_fp8 = (quant_mode == "fp8")
    quantize_int8 = (quant_mode == "int8")

    model_cfg = reconcile_model_config_with_state_dict(
        config=config,
        state_dict=raw_sd,
        scale=model_scale,
        steps=steps,
    )
    model_cfg.pop("quantize_fp8", None)
    model_cfg.pop("quantize_int8", None)
    model_cfg.pop("quantization", None)
    cross_every = model_cfg.pop("cross_every", 1)

    if max_vram < model_cfg.get("num_phases", 3):
        # Constrained desktop VRAM mode (<= 8GB):
        # Clear expert parameter allocations on CPU BEFORE moving model to target GPU,
        # ensuring GPU memory is never spiked with unused non-resident weights!
        model = EdgeFlowPhaseV2(
            **model_cfg,
            cross_every=cross_every,
            quantize_fp8=quantize_fp8,
            quantize_int8=quantize_int8,
            quantization=quant_mode,
        )
        for p in range(model.num_phases):
            model.clear_active_expert(p)
        model = model.to(device)
    else:
        model = EdgeFlowPhaseV2(
            **model_cfg,
            cross_every=cross_every,
            quantize_fp8=quantize_fp8,
            quantize_int8=quantize_int8,
            quantization=quant_mode,
        ).to(device)

    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    model_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    specific_experts_dir = os.path.join(ckpt_dir, f"experts_{model_stem}")
    default_experts_dir = os.path.join(ckpt_dir, "experts")

    if os.path.exists(specific_experts_dir) and any(f.startswith("expert_phase_") for f in os.listdir(specific_experts_dir)):
        experts_dir = specific_experts_dir
    elif os.path.exists(default_experts_dir) and any(f.startswith("expert_phase_") for f in os.listdir(default_experts_dir)) and "4b" not in model_stem.lower():
        experts_dir = default_experts_dir
    elif os.path.exists("cp/experts") and any(f.startswith("expert_phase_") for f in os.listdir("cp/experts")) and "4b" not in model_stem.lower():
        experts_dir = "cp/experts"
    else:
        experts_dir = specific_experts_dir

    has_experts_on_disk = os.path.exists(experts_dir) and any(
        f.startswith("expert_phase_") for f in os.listdir(experts_dir)
    )
    has_experts_in_sd = any("phase_experts." in k for k in raw_sd.keys())
    has_lora_in_sd = any("phase_attn_lora." in k for k in raw_sd.keys())

    lora_in_experts = False
    if has_experts_on_disk:
        cand = os.path.join(experts_dir, "expert_phase_0.pt")
        if os.path.exists(cand):
            try:
                sd_cand = torch.load(cand, map_location="cpu", weights_only=False)
                lora_in_experts = any("phase_attn_lora." in k for k in sd_cand.get("state_dict", sd_cand).keys())
            except Exception:
                pass
    elif has_experts_in_sd:
        lora_in_experts = has_lora_in_sd

    model._lora_in_experts = lora_in_experts

    max_vram = max_experts_vram
    max_ram = max_experts_ram

    clean_sd = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in raw_sd.items()
        if not k.endswith("n_averaged")
    }

    if has_experts_in_sd and max_vram >= model.num_phases:
        print(f"High-capacity VRAM mode: max_experts_vram ({max_vram}) >= num_phases ({model.num_phases}). Loading entire model directly to {device}...")
        model.load_state_dict(clean_sd, strict=False)
        del clean_sd

        res_mgr = ChunkedResidencyManager(
            model=model,
            experts_dir=experts_dir,
            device=torch.device(device),
            max_experts_vram=max_vram,
            max_experts_ram=max_ram,
            lora_in_experts=lora_in_experts,
        )
        for p in range(model.num_phases):
            res_mgr.vram_resident_experts[p] = None
        res_mgr._active_phase = 0
        model.expert_streamer = res_mgr
        print(f"All {model.num_phases} phase experts and backbone resident in VRAM. Direct fast-path active (0 ms switch).")
    else:
        # Constrained VRAM fallback: only backbone initially resident
        for p in range(model.num_phases):
            model.clear_active_expert(p)

        print(f"Loading shared backbone layers (SwiGLU FFN, low-rank AdaLN, GQA, Norms) in 128MB chunks to {device}...")
        backbone_sd = {
            k: v for k, v in raw_sd.items()
            if "phase_experts." not in k
            and (not lora_in_experts or "phase_attn_lora." not in k)
            and not k.endswith("n_averaged")
        }

        streamer = ChunkedWeightStreamer(chunk_size_mb=DEFAULT_CHUNK_SIZE_MB)
        streamer.stream_state_dict_to_model(
            model=model,
            state_dict=backbone_sd,
            device=torch.device(device),
            tag="Inference Backbone",
            allow_delete_keys=True,
        )
        print(f"Shared backbone successfully loaded into VRAM ({device}). Shared layers permanently resident.")

        if not has_experts_on_disk and has_experts_in_sd:
            print(f"Extracting modular phase experts from checkpoint to {experts_dir} for 128MB chunked streaming...")
            os.makedirs(experts_dir, exist_ok=True)
            for p in range(model.num_phases):
                prefix = f"phase_experts.{p}."
                p_sd = {k: v for k, v in raw_sd.items() if prefix in k}
                cross_prefix = f"phase_cross_kv.{p}."
                if not any(k.startswith(cross_prefix) for k in raw_sd.keys()) and p == 1:
                    cross_prefix = "phase_cross_kv.0."
                for k, v in raw_sd.items():
                    if cross_prefix in k:
                        p_sd[k] = v
                if lora_in_experts:
                    lora_prefix = f"phase_attn_lora.{p}."
                    for k, v in raw_sd.items():
                        if lora_prefix in k:
                            p_sd[k] = v
                if p_sd:
                    exp_file = os.path.join(experts_dir, f"expert_phase_{p}.pt")
                    if not os.path.exists(exp_file):
                        torch.save({"phase_idx": p, "state_dict": p_sd}, exp_file)
                        print(f"Saved Phase {p} expert to {exp_file} ({os.path.getsize(exp_file)/(1024**2):.1f} MB)")
            has_experts_on_disk = True

        if has_experts_on_disk:
            print(f"Enabling 128MB chunked expert streaming from {experts_dir} (VRAM: {max_vram}, RAM: {max_ram})...")
            model.enable_expert_streaming(
                experts_dir=experts_dir,
                device=device,
                max_experts_vram=max_vram,
                max_experts_ram=max_ram,
                lora_in_experts=lora_in_experts,
            )
            if max_vram >= model.num_phases:
                for p in range(model.num_phases):
                    model.expert_streamer.stream_phase(p)
        else:
            # Fallback: stream Phase 0 expert directly if present
            exp0_sd = {k: v for k, v in raw_sd.items() if "phase_experts.0." in k}
            if exp0_sd:
                print("Streaming Phase 0 expert from checkpoint in 128MB chunks...")
                streamer.stream_state_dict_to_model(
                    model=model,
                    state_dict=exp0_sd,
                    device=torch.device(device),
                    tag="Inference Expert 0",
                    allow_delete_keys=True,
                )

    model.eval()

    if compile_model:
        import compile_optimizer
        threads = compile_optimizer.setup_fast_compile(
            num_threads=compile_threads,
            enable_caching=True,
            verbose=True,
        )
        print(f"Compiling EdgeFlowPhaseV2 with torch.compile(mode='default', dynamic=True, {threads} threads)...")
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="default", dynamic=True)
            print(f"Model successfully wrapped with torch.compile ({threads} threads, suppress_errors=True)!")
            if compile_warmup and str(device).startswith("cuda"):
                c_dim = model_cfg.get("context_dim", 2560)
                compile_optimizer.warmup_compiled_model(
                    model=model,
                    device=device,
                    latent_channels=model_cfg.get("latent_channels", 32),
                    latent_h=max(1, image_size // 32),
                    latent_w=max(1, image_size // 32),
                    context_dim=c_dim,
                    num_phases=model_cfg.get("num_phases", 3),
                    batch_sizes=(1, 2) if cfg_scale > 1.0 else (1,),
                    verbose=True,
                )
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), falling back to eager mode.")

    latent_channels = model_cfg.get("latent_channels", 32)
    if "latent_mean" in checkpoint:
        latent_mean = checkpoint["latent_mean"].to(device).float().view(1, -1, 1, 1)
        latent_std = checkpoint["latent_std"].to(device).float().view(1, -1, 1, 1)
    else:
        latent_mean = torch.zeros((1, latent_channels, 1, 1), device=device)
        latent_std = torch.ones((1, latent_channels, 1, 1), device=device)

    context_dim = model_cfg.get("context_dim", 2560)
    max_txt_len = int(os.environ.get("MAX_TEXT_LEN", "512"))
    eff_quant = text_quant or ("8bit" if text_8bit else "none")
    try:
        if context_dim == 2560:
            from encoders import Qwen35TextEncoder
            text_encoder = Qwen35TextEncoder(
                out_dim=context_dim,
                include_encoder=True,
                load_in_4bit=(eff_quant == "4bit"),
                load_in_8bit=(eff_quant == "8bit"),
                max_length=max_txt_len,
            ).to(device)
        else:
            text_encoder = TextEncoder(
                out_dim=context_dim,
                include_encoder=True,
                hidden_size=512,
                max_length=max_txt_len,
            ).to(device)
    except Exception as e:
        print(f"Warning: Could not load full text encoder ({e}), falling back to projection-only...")
        if context_dim == 2560:
            from encoders import Qwen35TextEncoder
            text_encoder = Qwen35TextEncoder(
                out_dim=context_dim,
                include_encoder=False,
                load_in_4bit=False,
                load_in_8bit=False,
                max_length=max_txt_len,
            ).to(device)
        else:
            text_encoder = TextEncoder(
                out_dim=context_dim,
                include_encoder=False,
                hidden_size=512,
                max_length=max_txt_len,
            ).to(device)

    if use_ema and "ema_text_proj_state_dict" in checkpoint:
        ema_text_state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in checkpoint["ema_text_proj_state_dict"].items()
            if not k.endswith("n_averaged")
        }
        if hasattr(text_encoder, "proj"):
            text_encoder.proj.load_state_dict(ema_text_state, strict=False)
            print("Loaded EMA text projector weights.")
    elif "text_encoder_state_dict" in checkpoint:
        if hasattr(text_encoder, "proj"):
            text_encoder.proj.load_state_dict(checkpoint["text_encoder_state_dict"], strict=False)

    model_api_raw = getattr(model, "_orig_mod", model)
    if use_ema and "ema_text_pooler_state_dict" in checkpoint:
        ema_pooler_state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in checkpoint["ema_text_pooler_state_dict"].items()
            if not k.endswith("n_averaged")
        }
        if hasattr(text_encoder, "pooler"):
            text_encoder.pooler.load_state_dict(ema_pooler_state, strict=False)
            print("Loaded EMA text pooler weights.")
        if hasattr(model_api_raw, "pooler") and model_api_raw.pooler is not None:
            model_api_raw.pooler.load_state_dict(ema_pooler_state, strict=False)
    elif "text_encoder_pooler_state_dict" in checkpoint:
        pooler_state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in checkpoint["text_encoder_pooler_state_dict"].items()
            if not k.endswith("n_averaged")
        }
        if hasattr(text_encoder, "pooler"):
            text_encoder.pooler.load_state_dict(pooler_state, strict=False)
        if hasattr(model_api_raw, "pooler") and model_api_raw.pooler is not None:
            model_api_raw.pooler.load_state_dict(pooler_state, strict=False)
    elif hasattr(model_api_raw, "pooler") and model_api_raw.pooler is not None and hasattr(text_encoder, "pooler"):
        text_encoder.pooler.load_state_dict(model_api_raw.pooler.state_dict(), strict=False)

    text_encoder.eval()

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

    rf = RectifiedFlow4D()

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=(dev_type == "cuda")):
        # Encode conditional / unconditional text representations
        prompts = ["", prompt] if cfg_scale > 1.0 else [prompt]
        cond_pooled, cond_seq, mask = text_encoder(prompts, device)

        # Offload text encoder to CPU immediately after prompt encoding to free ~4.5 GB of VRAM
        if offload_text_encoder and text_encoder is not None:
            text_encoder.to("cpu")
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Acquire cached DC-AE decoder without disk reload
        vae = get_dcae(
            device=device,
            torch_dtype=amp_dtype,
            decoder_only=True,
            compile_decoder=compile_model,
        )
        vae.eval()

        # cond_pooled: [1, 2560] (or [2, 2560] with CFG), cond_seq: [B, 3, seq_len, 2560] (Layer 8, 16, 28)
        model_api = getattr(model, "_orig_mod", model)
        cache_kvs = {}
        orig_encode_text = getattr(model_api, "encode_text", None)
        if orig_encode_text is not None:
            def cached_encode_text(seq, phase_idx=0, **kwargs):
                p = phase_idx if isinstance(phase_idx, int) else (int(phase_idx[0].item()) if phase_idx is not None and phase_idx.numel() > 0 else 0)
                apply_k_norm = kwargs.get("apply_k_norm", True)
                key = (id(seq), p, apply_k_norm)
                if key not in cache_kvs:
                    cache_kvs[key] = orig_encode_text(seq, phase_idx=phase_idx, **kwargs)
                return cache_kvs[key]
            model_api.encode_text = cached_encode_text

        try:
            num_phases = getattr(model_api, "num_phases", 3)
            if orig_encode_text is not None:
                for p in range(num_phases):
                    cached_encode_text(cond_seq, phase_idx=p)
                text_kvs = cached_encode_text(cond_seq, phase_idx=0)
            else:
                text_kvs = None

            spatial_factor = getattr(vae, "spatial_scale_factor", 32)
            latent_h = image_size // spatial_factor
            latent_w = image_size // spatial_factor
            patch_size = getattr(model_api, "patch_size", 1)
            if latent_h % patch_size != 0:
                latent_h = max(patch_size, (latent_h // patch_size) * patch_size)
            if latent_w % patch_size != 0:
                latent_w = max(patch_size, (latent_w // patch_size) * patch_size)
            shape = (1, latent_channels, latent_h, latent_w)

            initial_noise = None
            if fp8_latents and hasattr(torch, "float8_e4m3fn") and str(device).startswith("cuda"):
                raw_noise = torch.randn(shape, device=device)
                initial_noise = raw_noise.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)

            if solver in ("unipc", "dpm++_2m", "phase_ab2", "ab2"):
                if initial_noise is None:
                    initial_noise = torch.randn(shape, device=device, dtype=amp_dtype)
                else:
                    initial_noise = initial_noise.to(device=device, dtype=amp_dtype)
                phase_sampler = PhaseAwareSampler(
                    model=model,
                    solver_type="unipc" if solver == "unipc" else "dpm++_2m",
                    time_shift=shift if shift != 1.0 else 1.9,
                )
                x_0_pred = phase_sampler.sample(
                    z=initial_noise,
                    cond_seq=cond_seq,
                    cond_pooled=cond_pooled,
                    cond_mask=mask,
                    cfg_scale=cfg_scale,
                    steps=steps,
                    dynamic_cfg=True,
                    use_block_cache=True,
                )
            elif solver == "multistep":
                x_0_pred = rf.phase_multistep_sample(
                    model=model,
                    cond_pooled=cond_pooled,
                    cond_seq=cond_seq,
                    shape=shape,
                    device=device,
                    steps=steps,
                    mask=mask,
                    text_kvs=text_kvs,
                    cfg_scale=cfg_scale,
                    cfg_rescale=cfg_rescale,
                    attn_temperature=0.8 if cfg_scale <= 1.0 else 1.0,
                    initial_noise=initial_noise,
                )
            elif solver == "heun":
                x_0_pred = rf.phase_heun_sample(
                    model=model,
                    cond_pooled=cond_pooled,
                    cond_seq=cond_seq,
                    shape=shape,
                    device=device,
                    steps=steps,
                    mask=mask,
                    text_kvs=text_kvs,
                    cfg_scale=cfg_scale,
                    cfg_rescale=cfg_rescale,
                    shift=shift,
                    rescale_norm=rescale_norm,
                    initial_noise=initial_noise,
                )
            else:
                x_0_pred = rf.phase_euler_sample(
                    model=model,
                    cond_pooled=cond_pooled,
                    cond_seq=cond_seq,
                    shape=shape,
                    device=device,
                    steps=steps,
                    mask=mask,
                    text_kvs=text_kvs,
                    cfg_scale=cfg_scale,
                    cfg_rescale=cfg_rescale,
                    shift=shift,
                    rescale_norm=rescale_norm,
                    initial_noise=initial_noise,
                )
        finally:
            if orig_encode_text is not None:
                model_api.encode_text = orig_encode_text

        l_std = latent_std
        l_mean = latent_mean
        if l_std.shape[1] != x_0_pred.shape[1]:
            if x_0_pred.shape[1] > l_std.shape[1]:
                reps = (x_0_pred.shape[1] + l_std.shape[1] - 1) // l_std.shape[1]
                l_std = l_std.repeat(1, reps, 1, 1)[:, :x_0_pred.shape[1]]
                l_mean = l_mean.repeat(1, reps, 1, 1)[:, :x_0_pred.shape[1]]
            else:
                l_std = l_std[:, :x_0_pred.shape[1]]
                l_mean = l_mean[:, :x_0_pred.shape[1]]
        x_0_pred = x_0_pred * l_std + l_mean
        
    # For DC-AE, 1024 tokens = 32x32 latent. Peak VRAM during direct decode is only ~1.97 GB in FP16.
    # Desktop cards (>= 8GB) can safely direct-decode up to 48x48 latents without tiling.
    lat_h, lat_w = x_0_pred.shape[-2], x_0_pred.shape[-1]
    act_tile_size = 48 if tile_size < 48 else tile_size
    act_overlap = 12 if tile_overlap < 12 else tile_overlap

    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            free_gb = free_bytes / (1024**3)
            # Evict expert if sequential loading is enabled or free VRAM is critically low (< 0.8 GB)
            if (sequential_loading or free_gb < 0.8) and hasattr(model_api, "expert_streamer") and model_api.expert_streamer is not None:
                model_api.expert_streamer.evict_current_expert()
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Auto-decide tiling if not explicitly set
    force_tiling = tiled_decode
    if force_tiling is None:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            try:
                free_b, tot_b = torch.cuda.mem_get_info(device)
                free_gb = free_b / (1024**3)
                # Direct decode is safe on desktop GPUs for 1024 tokens (32x32) unless free VRAM < 1.0 GB
                force_tiling = (lat_h > 48 or lat_w > 48) and (free_gb < 1.0)
            except Exception:
                force_tiling = False
        else:
            force_tiling = False

    # Decode in half-precision (amp_dtype) to leverage Tensor Cores, NOT forced float32
    decode_latent = x_0_pred.to(dtype=amp_dtype) if str(device).startswith("cuda") else x_0_pred.float()
    if force_tiling:
        image_tensor = tiled_decode_latents(
            vae=vae,
            latent=decode_latent,
            tile_size=act_tile_size,
            tile_overlap=act_overlap,
        )
    else:
        try:
            image_tensor = vae.decode(decode_latent)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            image_tensor = tiled_decode_latents(
                vae=vae,
                latent=decode_latent,
                tile_size=act_tile_size,
                tile_overlap=act_overlap,
            )
    image_tensor = image_tensor.clamp(0.0, 1.0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torchvision.utils.save_image(
        image_tensor,
        output_path,
        normalize=False,
    )
    print(f"Generated image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate image using trained EdgeFlow-Phase DiT v3")
    default_ckpt = ""
    cand_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "EdgeFlow-Phase-Experts-20k_Checkpoints"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cp"),
    ]
    for d in cand_dirs:
        if os.path.exists(d):
            pts = [f for f in os.listdir(d) if f.endswith(".pt")]
            if pts:
                import re
                def _step_num(fn):
                    m = re.search(r'_step_(\d+)', fn)
                    return int(m.group(1)) if m else 0
                pts.sort(key=lambda fn: (_step_num(fn), not fn.endswith("_weights.pt")), reverse=True)
                default_ckpt = os.path.join(d, pts[0])
                break

    parser.add_argument("--checkpoint", type=str, default=default_ckpt or None, required=(not default_ckpt), help="Path to checkpoint .pt")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="sample.png", help="Output path")
    parser.add_argument("--scale", type=str, default="4b", choices=["4b", "100m"], help="Model preset scale (default: 4b)")
    parser.add_argument("--steps", type=int, default=24, help="Sampling steps (Euler default: 24, recommended 24-30)")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale (default: 1.0 for CFG-free generation)")
    parser.add_argument("--cfg_rescale", type=float, default=0.5, help="CFG rescale (default: 0.5)")
    parser.add_argument("--shift", type=float, default=1.0, help="Noise time schedule shift (default: 1.0)")
    try:
        parser.add_argument("--rescale-norm", "--rescale_norm", action=argparse.BooleanOptionalAction, default=True, dest="rescale_norm", help="Velocity norm rescaling to counteract shrinkage")
    except AttributeError:
        parser.add_argument("--rescale-norm", "--rescale_norm", action="store_true", default=True, dest="rescale_norm", help="Velocity norm rescaling")
        parser.add_argument("--no-rescale-norm", "--no_rescale_norm", action="store_false", dest="rescale_norm")
    parser.add_argument("--tokens", type=int, default=None, choices=[256, 1024, 4096], help="Token budget: 256 (512x512 with DC-AE), 1024 (1024x1024 with DC-AE), 4096 (2048x2048 with DC-AE)")
    parser.add_argument("--solver", type=str, default="phase_ab2", choices=["phase_ab2", "unipc", "dpm++_2m", "ab2", "multistep", "euler", "heun"], help="ODE Solver (default: phase_ab2)")
    parser.add_argument("--image_size", "--image-size", type=int, default=1024, dest="image_size", help="Output image size (default: 1024 for DC-AE 1024 tokens)")
    parser.add_argument("--vae", type=str.lower, default="dcae", choices=["dcae"], help="VAE type (strictly DC-AE)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_ema", "--no-ema", action="store_true", dest="no_ema", help="Do not use EMA weights")
    try:
        parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="Compile model using torch.compile (Inductor backend)")
    except AttributeError:
        parser.add_argument("--compile", action="store_true", default=False, help="Compile model using torch.compile (Inductor backend)")
        parser.add_argument("--no-compile", action="store_false", dest="compile", help="Disable model compilation")
    parser.add_argument("--compile-threads", "--compile_threads", type=int, default=None, dest="compile_threads", help="Number of worker threads for parallel compilation (default: auto)")
    try:
        parser.add_argument("--compile-warmup", "--compile_warmup", action=argparse.BooleanOptionalAction, default=True, dest="compile_warmup", help="Enable automatic multi-phase warmup (default: True)")
    except AttributeError:
        parser.add_argument("--compile-warmup", "--compile_warmup", action="store_true", default=True, dest="compile_warmup", help="Enable automatic multi-phase warmup")
        parser.add_argument("--no-compile-warmup", "--no_compile_warmup", action="store_false", dest="compile_warmup")
    try:
        parser.add_argument("--text-8bit", "--text_8bit", action=argparse.BooleanOptionalAction, default=True, dest="text_8bit", help="Load Qwen 3.5 4B text encoder in 8-bit (default: True)")
    except AttributeError:
        parser.add_argument("--text-8bit", "--text_8bit", action="store_true", default=True, dest="text_8bit", help="Load Qwen in 8-bit")
        parser.add_argument("--no-text-8bit", "--no_text_8bit", action="store_false", dest="text_8bit")
    parser.add_argument("--text-quant", "--text_quant", type=str.lower, default="8bit", choices=["8bit", "4bit", "none", "fp16", "bf16"], dest="text_quant", help="Text encoder quantization mode: '8bit' (BitsAndBytes), '4bit' (BitsAndBytes NF4), or 'none' (BF16)")
    try:
        parser.add_argument("--sequential-loading", "--sequential_loading", action=argparse.BooleanOptionalAction, default=None, dest="sequential_loading", help="Enable sequential loading of text encoder, DiT experts, and VAE (default: auto)")
    except AttributeError:
        parser.add_argument("--sequential-loading", "--sequential_loading", action="store_true", default=None, dest="sequential_loading", help="Enable sequential loading")
        parser.add_argument("--no-sequential-loading", "--no_sequential_loading", action="store_false", dest="sequential_loading")
    try:
        parser.add_argument("--tiled-decode", "--tiled_decode", action=argparse.BooleanOptionalAction, default=HW_PROFILE["default_tiled_decode"], dest="tiled_decode", help=f"Use tiled/chunked VAE decoding to save VRAM (default: {HW_PROFILE['default_tiled_decode']})")
    except AttributeError:
        parser.add_argument("--tiled-decode", "--tiled_decode", action="store_true", default=HW_PROFILE["default_tiled_decode"], dest="tiled_decode", help="Use tiled VAE decoding")
        parser.add_argument("--no-tiled-decode", "--no_tiled_decode", action="store_false", dest="tiled_decode")
    try:
        parser.add_argument("--offload-text-encoder", "--offload_text_encoder", action=argparse.BooleanOptionalAction, default=HW_PROFILE["is_desktop_gpu"], dest="offload_text_encoder", help=f"Offload text encoder to CPU after prompt encoding (default: {HW_PROFILE['is_desktop_gpu']})")
    except AttributeError:
        parser.add_argument("--offload-text-encoder", "--offload_text_encoder", action="store_true", default=HW_PROFILE["is_desktop_gpu"], dest="offload_text_encoder", help="Offload text encoder to CPU")
        parser.add_argument("--no-offload-text-encoder", "--no_offload_text_encoder", action="store_false", dest="offload_text_encoder")
    try:
        parser.add_argument("--fp8-latents", "--fp8_latents", action=argparse.BooleanOptionalAction, default=True, dest="fp8_latents", help="Use FP8 initial noisy latents (default: True)")
    except AttributeError:
        parser.add_argument("--fp8-latents", "--fp8_latents", action="store_true", default=True, dest="fp8_latents", help="Use FP8 noisy latents")
        parser.add_argument("--no-fp8-latents", "--no_fp8_latents", action="store_false", dest="fp8_latents")
    parser.add_argument("--tile_size", "--tile-size", type=int, default=48, dest="tile_size", help="Tile size in latent space for VAE decoding (default: 48)")
    parser.add_argument("--tile_overlap", "--tile-overlap", type=int, default=12, dest="tile_overlap", help="Tile overlap in latent space for VAE decoding (default: 12)")
    parser.add_argument("--max_experts_vram", "--max-experts-vram", type=int, default=HW_PROFILE["default_max_experts_vram"], help=f"Max experts resident in VRAM (default: {HW_PROFILE['default_max_experts_vram']})")
    parser.add_argument("--max_experts_ram", "--max-experts-ram", type=int, default=HW_PROFILE["default_max_experts_ram"], help=f"Max experts resident in host RAM (default: {HW_PROFILE['default_max_experts_ram']})")

    args = parser.parse_args()

    image_size = args.image_size
    if args.tokens == 256:
        image_size = 512
    elif args.tokens == 1024:
        image_size = 1024
    elif args.tokens == 4096:
        image_size = 2048

    generate(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        steps=args.steps,
        cfg_scale=args.cfg,
        cfg_rescale=args.cfg_rescale,
        image_size=image_size,
        vae_type=args.vae,
        seed=args.seed,
        use_ema=not args.no_ema,
        solver=args.solver,
        compile_model=args.compile,
        compile_threads=args.compile_threads,
        compile_warmup=args.compile_warmup,
        rescale_norm=args.rescale_norm,
        shift=args.shift,
        model_scale=args.scale,
        text_8bit=args.text_8bit,
        text_quant=args.text_quant,
        sequential_loading=args.sequential_loading,
        tiled_decode=args.tiled_decode,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        max_experts_vram=args.max_experts_vram,
        max_experts_ram=args.max_experts_ram,
        offload_text_encoder=args.offload_text_encoder,
        fp8_latents=args.fp8_latents,
    )


if __name__ == "__main__":
    main()
