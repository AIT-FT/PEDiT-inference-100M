import os
import argparse
from typing import Optional
from check_deps import ensure_dependencies
ensure_dependencies()

import config  # Автозагрузка .env файла
import torch
import torchvision
from PIL import Image

from model import EdgeFlowPhaseV2
from encoders import TextEncoder, TAESD, SDVAE, SDXLVAE
from rectified_flow import RectifiedFlow4D
from download_utils import ensure_checkpoint, ensure_text_encoder, ensure_vae, get_checkpoints_dir


@torch.inference_mode()
def generate(
    checkpoint_path: Optional[str] = None,
    prompt: str = "",
    output_path: str = "outputs/sample.png",
    steps: int = 8,
    cfg_scale: float = 4.0,
    cfg_rescale: float = 0.0,
    image_size: int = 512,
    vae_type: str = "taesd",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    use_ema: bool = True,
    ckpt_dir: Optional[str] = None,
    precision: Optional[str] = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if not checkpoint_path and precision:
        checkpoint_path = f"PEDiT-100M-{precision.upper()}.pt"

    # Автоматическая проверка/загрузка чекпоинта, текстового энкодера и VAE
    resolved_ckpt = ensure_checkpoint(checkpoint_path, ckpt_dir=ckpt_dir)
    if not resolved_ckpt or not os.path.exists(resolved_ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found. Please specify --checkpoint <path/url> or place a .pt file in '{get_checkpoints_dir(ckpt_dir)}'."
        )

    ensure_text_encoder()
    ensure_vae(vae_type)

    print(f"Loading checkpoint: {resolved_ckpt}")
    try:
        checkpoint = torch.load(resolved_ckpt, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        checkpoint = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})

    model = EdgeFlowPhaseV2(
        latent_channels=config.get("latent_channels", 4),
        patch_size=config.get("patch_size", 2),
        dim=config.get("dim", 768),
        depth=config.get("depth", 8),
        heads=config.get("heads", 12),
        kv_heads=config.get("kv_heads", 2),
        context_dim=config.get("context_dim", 768),
        num_phases=config.get("num_phases", 4),
        ffn_hidden=config.get("ffn_hidden", 1024),
        ada_rank=config.get("ada_rank", 64),
        cross_every=config.get("cross_every", 2),
        expert_hidden_list=config.get("expert_hidden_list", [768, 768, 768, 768]),
    ).to(device)

    def clean_dict(d):
        cd = {}
        for k, v in d.items():
            if k.endswith("n_averaged"):
                continue
            ck = k
            while ck.startswith("module."):
                ck = ck[7:]
            while ck.startswith("_orig_mod."):
                ck = ck[10:]
            cd[ck] = v
        return cd

    ema_key = None
    if use_ema:
        if "ema_model_state_dict" in checkpoint:
            ema_key = "ema_model_state_dict"
        elif "ema_state_dict" in checkpoint:
            ema_key = "ema_state_dict"

    if ema_key is not None:
        state_dict = checkpoint[ema_key]
        print(f"Loaded EMA weights ({ema_key}).")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if checkpoint.get("is_ema"):
            print("Loaded EMA weights (from pure EMA checkpoint).")
        else:
            print("Loaded base model weights.")
    else:
        raise ValueError("No valid model weights found in checkpoint.")

    clean_state = clean_dict(state_dict)

    if "model_scales" in checkpoint and checkpoint["model_scales"]:
        scales = checkpoint["model_scales"]
        for k, v in clean_state.items():
            if torch.is_tensor(v) and v.dtype == torch.int8:
                sc = scales.get(k, 1.0)
                clean_state[k] = v.to(torch.float16) * (sc.to(torch.float16) if torch.is_tensor(sc) else sc)
        print("Dequantized INT8 model weights to FP16.")

    first_val = next(iter(clean_state.values()), None)
    if first_val is not None and str(device).startswith("cuda"):
        if first_val.dtype in (torch.float16, torch.bfloat16):
            model = model.to(first_val.dtype)
            print(f"Model placed in {first_val.dtype}.")
        elif "float8" in str(first_val.dtype):
            model = model.to(torch.float16)
            print(f"Model placed in torch.float16 (unpacked from FP8).")

    model.load_state_dict(clean_state, strict=False)
    model.eval()

    if "latent_mean" in checkpoint and checkpoint["latent_mean"] is not None:
        latent_mean = checkpoint["latent_mean"].to(device).view(1, 4, 1, 1)
        latent_std = checkpoint["latent_std"].to(device).view(1, 4, 1, 1)
    else:
        latent_mean = torch.tensor([0.0]*4, device=device).view(1, 4, 1, 1)
        latent_std = torch.tensor([1.0]*4, device=device).view(1, 4, 1, 1)

    text_encoder = TextEncoder(
        out_dim=config.get("context_dim", 768),
        include_encoder=True,
        hidden_size=512,
        max_length=128,
    ).to(device)

    def dequant_sub(d, scales):
        if not scales:
            return d
        return {
            k: (v.to(torch.float16) * (scales.get(k, 1.0).to(torch.float16) if torch.is_tensor(scales.get(k, 1.0)) else scales.get(k, 1.0)))
            if (torch.is_tensor(v) and v.dtype == torch.int8) else v
            for k, v in d.items()
        }

    if use_ema and "ema_text_proj_state_dict" in checkpoint:
        text_encoder.proj.load_state_dict(clean_dict(checkpoint["ema_text_proj_state_dict"]), strict=False)
        print("Loaded EMA text projector weights.")
    elif "text_encoder_state_dict" in checkpoint and checkpoint["text_encoder_state_dict"] is not None:
        proj_dict = clean_dict(checkpoint["text_encoder_state_dict"])
        proj_dict = dequant_sub(proj_dict, checkpoint.get("text_encoder_scales"))
        text_encoder.proj.load_state_dict(proj_dict, strict=False)
        print("Loaded text projector weights.")
        
    if use_ema and "ema_text_pooler_state_dict" in checkpoint:
        text_encoder.pooler.load_state_dict(clean_dict(checkpoint["ema_text_pooler_state_dict"]), strict=False)
        print("Loaded EMA text pooler weights.")
    elif "text_encoder_pooler_state_dict" in checkpoint and checkpoint["text_encoder_pooler_state_dict"] is not None:
        pooler_dict = clean_dict(checkpoint["text_encoder_pooler_state_dict"])
        pooler_dict = dequant_sub(pooler_dict, checkpoint.get("text_encoder_pooler_scales"))
        text_encoder.pooler.load_state_dict(pooler_dict, strict=False)
        print("Loaded text pooler weights.")

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

    vae_type = vae_type.lower()
    if vae_type == "taesd":
        vae = TAESD().to(device)
    elif vae_type == "sd":
        vae = SDVAE().to(device)
    elif vae_type == "sdxl":
        vae = SDXLVAE().to(device)
    else:
        vae = TAESD().to(device)
    vae.eval()

    rf = RectifiedFlow4D()

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=(dev_type == "cuda")):
        # Кодирование unconditional ("") и prompt
        prompts = ["", prompt]
        cond_pooled, cond_seq, mask = text_encoder(prompts, device)

        # cond_pooled: [2, 1024], cond_seq: [2, 128, 1024], mask: [2, 128]
        text_kvs = model.encode_text(cond_seq)

        latent_h = image_size // 8
        latent_w = image_size // 8
        shape = (1, 4, latent_h, latent_w)

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
        )

        x_0_pred = x_0_pred * latent_std + latent_mean
        
    image_tensor = vae.decode(x_0_pred.float())
    image_tensor = image_tensor.clamp(0.0, 1.0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torchvision.utils.save_image(
        image_tensor,
        output_path,
        normalize=False,
    )
    print(f"Generated image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate image using trained EdgeFlow-Phase DiT v2")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path or URL to checkpoint .pt (default: auto-detect latest)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directory with checkpoints (default: auto-detected)")
    parser.add_argument("--prompt", type=str, default="A detailed photographic portrait of a person, high quality", help="Text prompt")
    parser.add_argument("--output", type=str, default="outputs/sample.png", help="Output path")
    parser.add_argument("--steps", type=int, default=8, help="Sampling steps (Euler)")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--cfg_rescale", type=float, default=0.0, help="CFG rescale")
    parser.add_argument("--image_size", type=int, default=512, help="Output image size")
    parser.add_argument("--vae", type=str, default="taesd", choices=["taesd", "sd", "sdxl"], help="VAE type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--precision", type=str, default=None, choices=["fp16", "fp8", "int8"], help="Model precision (fp16, fp8, or int8; auto-downloads from Hugging Face if missing)")
    parser.add_argument("--no_ema", action="store_true", help="Do not use EMA weights")

    args = parser.parse_args()

    generate(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        output_path=args.output,
        steps=args.steps,
        cfg_scale=args.cfg,
        cfg_rescale=args.cfg_rescale,
        image_size=args.image_size,
        vae_type=args.vae,
        seed=args.seed,
        use_ema=not args.no_ema,
        ckpt_dir=args.ckpt_dir,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
