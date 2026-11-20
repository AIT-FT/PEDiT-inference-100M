import copy
import os
import sys
from typing import Any, Dict, Optional, Union


def load_dotenv(dotenv_path: str = ".env") -> None:
    if not dotenv_path or not os.path.exists(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8-sig") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("export ") and (len(line) > 7 and line[6].isspace()):
                    line = line[7:].lstrip()

                if "=" not in line:
                    continue

                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()

                if not k:
                    continue

                if len(v) >= 2 and v[0] in ("'", '"') and v[-1] == v[0]:
                    quote_char = v[0]
                    v = v[1:-1]
                    if quote_char == '"':
                        v = v.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t").replace('\\"', '"')
                else:
                    if v and v[0] in ("'", '"'):
                        quote_char = v[0]
                        end_idx = v.find(quote_char, 1)
                        if end_idx != -1:
                            v = v[1:end_idx]
                        else:
                            v = v[1:]
                    else:
                        for i, ch in enumerate(v):
                            if ch == "#" and (i == 0 or v[i - 1].isspace()):
                                v = v[:i].rstrip()
                                break

                if k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print(f"[config] Warning: Failed to load {dotenv_path}: {e}", file=sys.stderr)


load_dotenv()

_cur_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if not _cur_alloc:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
elif "expandable_segments" not in _cur_alloc:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{_cur_alloc},expandable_segments:True"

EDGEFLOW_100M_CONFIG = {
    "architecture_version": "v5.4_anatomy_refinement",
    "attention_version": "full_mha_cross48_rope2d_exact",
    "sampler_default": "unipc",
    "latent_channels": 32,
    "patch_size": 1,
    "dim": 640,
    "depth": 12,
    "heads": 10,
    "kv_heads": 10,
    "head_dim": 64,
    "context_dim": 2560,
    "num_phases": 3,
    "ffn_hidden": 1024,
    "ada_rank": 128,
    "gate_init": 0.12,
    "cross_every": 1,
    "expert_hidden_list": [768, 768, 768],
    "phase_expert_layers": (12, 12, 12),
    "per_block_adaln": True,
    "phase_layers": (8, 16, 28),
    "use_phase_lora": True,
    "phase_lora_rank": 48,
    "use_phase_k_lora": True,
    "use_cross_lora": True,
    "cross_lora_rank": 48,
    "num_registers": 8,
    "num_sinks": 8,
    "use_dw_conv": True,
    "use_skip_connections": True,
    "learnable_head_scale": True,
    "use_coord_conv": True,
    "rope_base": 16.0,
    "use_value_residual": True,
    "use_cross_sink": True,
    "use_head_gate": True,
    "use_hf_booster": False,
    "use_refinement_head": True,
    "text_cache_schema_version": 1,
}

EDGEFLOW_4B_CONFIG_24L: Dict[str, Any] = {
    "architecture_version": "v2_24l",
    "attention_version": "gqa",
    "sampler_default": "ab2",
    "latent_channels": 32,
    "patch_size": 1,
    "dim": 1536,
    "depth": 24,
    "heads": 24,
    "kv_heads": 6,
    "head_dim": 64,
    "context_dim": 2560,
    "num_phases": 3,
    "ffn_hidden": 2048,
    "ada_rank": None,
    "gate_init": 0.1,
    "cross_every": 1,
    "expert_hidden_list": [9216, 9216, 9216],
    "phase_expert_layers": (24, 24, 24),
    "per_block_adaln": True,
    "phase_layers": (8, 16, 28),
    "use_phase_lora": False,
    "phase_lora_rank": 32,
    "use_phase_k_lora": False,
    "use_cross_lora": False,
    "cross_lora_rank": 32,
    "num_registers": 0,
    "num_sinks": 0,
    "use_dw_conv": False,
    "use_skip_connections": False,
    "learnable_head_scale": False,
    "use_coord_conv": False,
    "rope_base": 10000.0,
    "use_value_residual": False,
    "use_cross_sink": False,
    "use_head_gate": False,
    "use_hf_booster": False,
    "use_refinement_head": False,
    "text_cache_schema_version": 1,
}

EDGEFLOW_4B_CONFIG_30L: Dict[str, Any] = {
    **EDGEFLOW_4B_CONFIG_24L,
    "architecture_version": "v3_30l",
    "depth": 30,
    "ffn_hidden": 1024,
    "expert_hidden_list": [7168, 7168, 7168],
    "phase_expert_layers": (30, 30, 30),
    "use_phase_lora": True,
    "phase_lora_rank": 32,
}

EDGEFLOW_4B_CONFIG_V4: Dict[str, Any] = {
    **EDGEFLOW_4B_CONFIG_30L,
    "architecture_version": "v4_full_experts",
    "heads": 16,
    "kv_heads": 8,
    "head_dim": 128,
    "ada_rank": 256,
    "ffn_hidden": 2560,
    "expert_hidden_list": [6656, 6656, 6656],
    "phase_expert_layers": (30, 30, 30),
    "phase_lora_rank": 64,
    "cross_lora_rank": 64,
    "num_registers": 8,
    "num_sinks": 0,
    "use_dw_conv": True,
    "use_skip_connections": True,
    "use_cross_lora": True,
    "learnable_head_scale": True,
    "use_coord_conv": True,
    "rope_base": 256.0,
    "use_value_residual": True,
    "use_cross_sink": True,
    "use_phase_k_lora": True,
    "use_head_gate": True,
    "use_hf_booster": True,
    "use_refinement_head": True,
    "text_cache_schema_version": 1,
}

EDGEFLOW_4B_CONFIG_V5: Dict[str, Any] = {
    **EDGEFLOW_4B_CONFIG_V4,
    "architecture_version": "v5.1_cascaded_experts",
    "attention_version": "full_mha_cross128",
    "sampler_default": "unipc",
    "heads": 16,
    "kv_heads": 16,
    "expert_hidden_list": [6656, 6656, 6656],
    "phase_expert_layers": (16, 22, 30),
    "phase_lora_rank": 64,
    "cross_lora_rank": 128,
    "num_registers": 0,
    "num_sinks": 4,
    "use_phase_k_lora": True,
    "use_hf_booster": False,
    "text_cache_schema_version": 1,
}


def _finalize_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    dim = int(out["dim"])
    heads = int(out["heads"])
    kv_heads = int(out.get("kv_heads", heads))
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads != 0:
        raise ValueError(f"Invalid attention heads configuration: heads={heads}, kv_heads={kv_heads}")

    if out.get("head_dim") is None:
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}) when head_dim is not set")
        out["head_dim"] = dim // heads

    head_dim = int(out["head_dim"])
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for 2D RoPE")

    out["attn_inner_dim"] = heads * head_dim

    depth = int(out["depth"])
    num_phases = int(out.get("num_phases", 3))
    if "phase_expert_layers" not in out or out["phase_expert_layers"] is None:
        out["phase_expert_layers"] = tuple(depth for _ in range(num_phases))
    else:
        out["phase_expert_layers"] = tuple(int(x) for x in out["phase_expert_layers"])
        if len(out["phase_expert_layers"]) != num_phases:
            raise ValueError(
                f"phase_expert_layers length ({len(out['phase_expert_layers'])}) != num_phases ({num_phases})"
            )
        if any(l < 1 or l > depth for l in out["phase_expert_layers"]):
            raise ValueError(
                f"phase_expert_layers {out['phase_expert_layers']} out of valid range [1, {depth}]"
            )

    out.setdefault("text_cache_schema_version", 1)
    out.setdefault("num_registers", 0)
    out.setdefault("num_sinks", 0)
    return out


def get_model_config(
    scale: str = "4b",
    steps: int = 12,
    version: Optional[str] = "v5",
) -> Dict[str, Any]:
    if not isinstance(scale, str) or not scale.strip():
        raise ValueError(f"Invalid model scale: {scale!r}. Choose '100m', '4b', '4b_v4', or '4b_v5'.")

    s = scale.lower().strip()
    ver = version.lower().strip() if (isinstance(version, str) and version.strip()) else "v5"

    if s in ("100m", "small", "base"):
        return _finalize_model_config(EDGEFLOW_100M_CONFIG)

    if s in ("4b_v5", "4b-v5", "v5"):
        return _finalize_model_config(EDGEFLOW_4B_CONFIG_V5)

    if s in ("4b_v4", "4b-v4", "v4"):
        return _finalize_model_config(EDGEFLOW_4B_CONFIG_V4)

    if s in ("4b", "4-billion", "4_billion", "large"):
        if ver in ("v5", "5", "full_attention", "full-expert-attention"):
            return _finalize_model_config(EDGEFLOW_4B_CONFIG_V5)
        if ver in ("v4", "4"):
            return _finalize_model_config(EDGEFLOW_4B_CONFIG_V4)
        if ver == "30l" or (ver != "24l" and steps >= 30):
            return _finalize_model_config(EDGEFLOW_4B_CONFIG_30L)
        if ver == "24l" or steps < 30:
            return _finalize_model_config(EDGEFLOW_4B_CONFIG_24L)
        raise ValueError(f"Unknown 4B model version: {version!r}. Choose 'v5', 'v4', '30l', or '24l'.")

    raise ValueError(f"Unknown model scale: {scale!r}. Choose '100m', '4b', '4b_v4', or '4b_v5'.")


def get_desktop_hardware_profile(
    device: Optional[Union[str, int, Any]] = None,
    scale: str = "4b",
    resolution: int = 512,
) -> Dict[str, Any]:
    is_small_model = isinstance(scale, str) and scale.lower().strip() in ("100m", "small", "base")

    profile: Dict[str, Any] = {
        "is_cuda": False,
        "device_name": "CPU",
        "total_vram_gb": 0.0,
        "is_desktop_gpu": True,
        "is_8gb_vram": True,
        "default_max_experts_vram": 3 if is_small_model else 1,
        "default_max_experts_ram": 0 if is_small_model else 2,
        "default_tiled_decode": False,
        "default_text_8bit": True,
        "default_fp8_latents": True,
    }

    try:
        import torch

        dev_idx = 0
        if device is not None:
            if isinstance(device, int):
                if device < 0:
                    return profile
                dev_idx = device
            else:
                dev_obj = torch.device(device)
                if dev_obj.type != "cuda":
                    profile["device_name"] = dev_obj.type.upper()
                    return profile
                if dev_obj.index is not None:
                    dev_idx = dev_obj.index

        if not torch.cuda.is_available():
            return profile

        device_count = torch.cuda.device_count()
        if dev_idx < 0 or dev_idx >= device_count:
            print(
                f"[config] Warning: Requested CUDA device index {dev_idx} out of range "
                f"(available: {device_count}). Falling back to cuda:0.",
                file=sys.stderr,
            )
            dev_idx = 0

        props = torch.cuda.get_device_properties(dev_idx)
        total_gb = props.total_memory / (1024 ** 3)
        dev_name = torch.cuda.get_device_name(dev_idx)

        is_server = (total_gb >= 23.0) or any(tag in dev_name for tag in ("A100", "H100", "H200", "L40", "A6000"))
        is_desktop = not is_server
        is_8gb = total_gb <= 8.5

        profile["is_cuda"] = True
        profile["device_name"] = dev_name
        profile["total_vram_gb"] = round(total_gb, 2)
        profile["is_desktop_gpu"] = is_desktop
        profile["is_8gb_vram"] = is_8gb

        if is_small_model:
            profile["default_max_experts_vram"] = 3
            profile["default_max_experts_ram"] = 0
        elif is_desktop:
            if total_gb >= 15.5:
                profile["default_max_experts_vram"] = 3
                profile["default_max_experts_ram"] = 0
            else:
                profile["default_max_experts_vram"] = 1
                profile["default_max_experts_ram"] = 2
        else:
            profile["default_max_experts_vram"] = 3
            profile["default_max_experts_ram"] = 0

        profile["default_tiled_decode"] = bool(is_8gb and resolution > 512)

    except Exception as e:
        print(f"[config] Warning: Hardware profile detection failed: {e}", file=sys.stderr)

    return profile
