import gc
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("EdgeFlow.Streamer")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_CHUNK_SIZE_MB = int(os.environ.get("WEIGHT_CHUNK_SIZE_MB", "128"))
DEFAULT_CHUNK_SIZE_BYTES = DEFAULT_CHUNK_SIZE_MB * 1024 * 1024


def clean_key(key: str) -> str:
    k = key
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod.", "_fsdp_wrapped_module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                changed = True
    return k

def calculate_tensor_bytes(tensor: Any) -> int:
    if isinstance(tensor, torch.Tensor):
        return tensor.numel() * tensor.element_size()
    return 0


def unwrap_model_deep(model: nn.Module) -> nn.Module:
    m = model
    visited = set()
    while id(m) not in visited:
        visited.add(id(m))
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif hasattr(m, "_fsdp_wrapped_module"):
            m = m._fsdp_wrapped_module
        elif hasattr(m, "module"):
            m = m.module
        else:
            break
    return m


def partition_state_dict_into_chunks(
    state_dict: Dict[str, Any],
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> List[List[str]]:
    groups: List[List[str]] = []
    visited = set()

    for k in state_dict.keys():
        if k in visited or k.endswith("n_averaged"):
            continue

        if k.endswith(".scale"):
            weight_k = k[:-6] + ".weight"
            if weight_k in state_dict and weight_k not in visited:
                groups.append([weight_k, k])
                visited.add(weight_k)
                visited.add(k)
                continue

        group_keys = [k]
        visited.add(k)

        if k.endswith(".weight"):
            scale_k = k[:-7] + ".scale"
            if scale_k in state_dict and scale_k not in visited:
                group_keys.append(scale_k)
                visited.add(scale_k)

        groups.append(group_keys)

    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_bytes = 0

    for group in groups:
        group_bytes = sum(calculate_tensor_bytes(state_dict.get(k)) for k in group)
        if current_bytes + group_bytes > chunk_size_bytes and len(current_chunk) > 0:
            chunks.append(current_chunk)
            current_chunk = list(group)
            current_bytes = group_bytes
        else:
            current_chunk.extend(group)
            current_bytes += group_bytes

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def apply_tensor_to_model(
    model: nn.Module,
    key: str,
    tensor: torch.Tensor,
    device: torch.device,
    scale_tensor: Optional[torch.Tensor] = None,
) -> bool:
    target_model = unwrap_model_deep(model)
    k = clean_key(key)
    if k == "proj_out.weight" and getattr(target_model, "refinement_head", None) is not None:
        k = "refinement_head.proj_base.weight"
    elif k == "refinement_head.proj_base.weight" and getattr(target_model, "refinement_head", None) is None:
        k = "proj_out.weight"

    model_quant_mode = getattr(target_model, "quantization", None)
    cache = getattr(target_model, "_tensor_target_cache", None)
    if cache is None or getattr(target_model, "_tensor_target_cache_qmode", None) != model_quant_mode:
        cache = {}
        target_model._tensor_target_cache = cache
        target_model._tensor_target_cache_qmode = model_quant_mode

    resolved = cache.get(k)
    if resolved is not None:
        submod, attr_name, is_fp8_linear, is_int8_linear, fp8_dtype = resolved
    else:
        parts = k.split(".")
        submod: Any = target_model
        try:
            for p in parts[:-1]:
                if p.isdigit():
                    submod = submod[int(p)]
                else:
                    submod = getattr(submod, p)
        except (AttributeError, IndexError, KeyError):
            return False
        attr_name = parts[-1]
        mod_type_name = type(submod).__name__
        is_fp8_linear = mod_type_name == "FP8Linear" or hasattr(submod, "fp8_dtype")
        is_int8_linear = mod_type_name == "Int8Linear"
        fp8_dtype = getattr(submod, "fp8_dtype", torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.int8) if is_fp8_linear else None
        cache[k] = (submod, attr_name, is_fp8_linear, is_int8_linear, fp8_dtype)

    if k == "proj_in.weight" or k.endswith(".proj_in.weight"):
        use_coord = getattr(target_model, "use_coord_conv", False)
        latent_ch = getattr(target_model, "latent_channels", 32)
        if use_coord and tensor.shape[1] == latent_ch:
            expanded = torch.zeros(
                (tensor.shape[0], latent_ch + 3, tensor.shape[2], tensor.shape[3]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            expanded[:, :latent_ch] = tensor
            tensor = expanded
        elif not use_coord and tensor.shape[1] == latent_ch + 3:
            tensor = tensor[:, :latent_ch]
        if tensor.shape[-1] == 1 and hasattr(target_model, "proj_in") and target_model.proj_in.weight.shape[-1] == 3:
            expanded_3x3 = torch.zeros(
                (tensor.shape[0], tensor.shape[1], 3, 3),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            expanded_3x3[:, :, 1:2, 1:2] = tensor
            tensor = expanded_3x3

    if is_fp8_linear:
        fp8_dtype = getattr(submod, "fp8_dtype", torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.int8)
        if attr_name == "weight":
            val = tensor
            if "float8" not in str(val.dtype) and val.dtype not in (torch.int8, torch.uint8):
                w_float = val.float()
                max_val = w_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                fp8_max = 448.0 if "e4m3" in str(fp8_dtype) else (57344.0 if "e5m2" in str(fp8_dtype) else 127.0)
                scale = max_val / fp8_max
                if fp8_dtype == torch.int8:
                    val = (w_float / scale).round().clamp(-128, 127).to(torch.int8)
                else:
                    val = (w_float / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
                if hasattr(submod, "_buffers") and "scale" in submod._buffers and scale_tensor is None:
                    submod._buffers["scale"].data = scale.to(device=device, dtype=torch.float16, non_blocking=True)

            if hasattr(submod, "_buffers") and "weight" in submod._buffers:
                target_buf = submod._buffers["weight"]
                if target_buf is not None and target_buf.shape == val.shape and target_buf.device == device and target_buf.dtype == val.dtype:
                    target_buf.copy_(val, non_blocking=True)
                elif target_buf is not None:
                    target_buf.data = val.to(device=device, dtype=fp8_dtype, non_blocking=True)
                else:
                    submod._buffers["weight"] = val.to(device=device, dtype=fp8_dtype, non_blocking=True)
            else:
                submod.register_buffer("weight", val.to(device=device, dtype=fp8_dtype, non_blocking=True))
            return True

        elif attr_name == "scale":
            val = tensor.to(dtype=torch.float16)
            if hasattr(submod, "_buffers") and "scale" in submod._buffers:
                target_buf = submod._buffers["scale"]
                if target_buf is not None and target_buf.shape == val.shape and target_buf.device == device:
                    target_buf.copy_(val, non_blocking=True)
                elif target_buf is not None:
                    target_buf.data = val.to(device=device, dtype=torch.float16, non_blocking=True)
                else:
                    submod._buffers["scale"] = val.to(device=device, dtype=torch.float16, non_blocking=True)
            else:
                submod.register_buffer("scale", val.to(device=device, dtype=torch.float16, non_blocking=True))
            return True

    if is_int8_linear:
        if attr_name == "weight":
            val = tensor
            if val.dtype != torch.int8:
                w_float = val.float()
                max_val = w_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                scale = max_val / 127.0
                val = (w_float / scale).round().clamp(-128, 127).to(torch.int8)
                if hasattr(submod, "_buffers") and "scale" in submod._buffers and scale_tensor is None:
                    submod._buffers["scale"].data = scale.to(device=device, dtype=torch.float16, non_blocking=True)

            if hasattr(submod, "_buffers") and "weight" in submod._buffers:
                target_buf = submod._buffers["weight"]
                if target_buf is not None and target_buf.shape == val.shape and target_buf.device == device:
                    target_buf.copy_(val, non_blocking=True)
                elif target_buf is not None:
                    target_buf.data = val.to(device=device, dtype=torch.int8, non_blocking=True)
                else:
                    submod._buffers["weight"] = val.to(device=device, dtype=torch.int8, non_blocking=True)
            else:
                submod.register_buffer("weight", val.to(device=device, dtype=torch.int8, non_blocking=True))
            return True

        elif attr_name == "scale":
            val = tensor.to(dtype=torch.float16)
            if hasattr(submod, "_buffers") and "scale" in submod._buffers:
                target_buf = submod._buffers["scale"]
                if target_buf is not None and target_buf.shape == val.shape and target_buf.device == device:
                    target_buf.copy_(val, non_blocking=True)
                elif target_buf is not None:
                    target_buf.data = val.to(device=device, dtype=torch.float16, non_blocking=True)
                else:
                    submod._buffers["scale"] = val.to(device=device, dtype=torch.float16, non_blocking=True)
            else:
                submod.register_buffer("scale", val.to(device=device, dtype=torch.float16, non_blocking=True))
            return True

    if hasattr(submod, "_parameters") and attr_name in submod._parameters:
        target_param = submod._parameters[attr_name]
        if target_param is not None and k == "refinement_head.refine.0.weight" and target_param.shape != tensor.shape:
            return True

    if hasattr(submod, "_parameters") and attr_name in submod._parameters:
        target_param = submod._parameters[attr_name]
        val = tensor
        if scale_tensor is not None and (val.dtype in (torch.int8, torch.uint8) or "float8" in str(val.dtype)):
            val = val.to(torch.float32) * scale_tensor.to(torch.float32)

        target_dtype = target_param.dtype if target_param is not None else (
            val.dtype if val.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
        )
        if val.dtype != target_dtype:
            val = val.to(dtype=target_dtype)

        req_grad = target_param.requires_grad if target_param is not None else False
        if target_param is not None and target_param.shape == val.shape and target_param.device == device:
            target_param.data.copy_(val, non_blocking=True)
        elif target_param is not None and isinstance(target_param, nn.Parameter):
            target_param.data = val.to(device=device, non_blocking=True)
        else:
            submod._parameters[attr_name] = nn.Parameter(
                val.to(device=device, non_blocking=True),
                requires_grad=req_grad,
            )
        return True

    if hasattr(submod, "_buffers") and attr_name in submod._buffers:
        target_buf = submod._buffers[attr_name]
        val = tensor
        target_dtype = target_buf.dtype if target_buf is not None else val.dtype
        if val.dtype != target_dtype:
            val = val.to(dtype=target_dtype)

        if target_buf is not None and target_buf.shape == val.shape and target_buf.device == device:
            target_buf.data.copy_(val, non_blocking=True)
        elif target_buf is not None:
            target_buf.data = val.to(device=device, non_blocking=True)
        else:
            submod._buffers[attr_name] = val.to(device=device, non_blocking=True)
        return True

    setattr(submod, attr_name, tensor.to(device=device, non_blocking=True))
    return True


class ChunkedWeightStreamer:

    def __init__(self, chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB):
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        self.chunk_size_mb = chunk_size_mb

    def stream_state_dict_to_model(
        self,
        model: nn.Module,
        state_dict: Dict[str, Any],
        device: torch.device,
        tag: str = "Weights",
        allow_delete_keys: bool = True,
        precomputed_chunks: Optional[List[List[str]]] = None,
        precomputed_scales: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        target_model = unwrap_model_deep(model)
        per_block = getattr(target_model, "per_block_adaln", None)
        if per_block is not None:
            has_old_ada = any(k.startswith("ada_proj.") or k == "ada_proj.weight" for k in state_dict.keys())
            has_new_ada = any(".adaLN_modulation." in k for k in state_dict.keys())
            if per_block and has_old_ada and not has_new_ada:
                w0 = state_dict.get("ada_proj.0.weight")
                w2 = state_dict.get("ada_proj.2.weight")
                w1 = state_dict.get("ada_proj.1.weight", state_dict.get("ada_proj.weight"))
                dim = getattr(target_model, "dim", 1536)
                if w1 is not None:
                    state_dict = dict(state_dict)
                    for i, block in enumerate(getattr(target_model, "blocks", [])):
                        slice_w1 = w1[i * 8 * dim : (i + 1) * 8 * dim]
                        if hasattr(block, "adaLN_modulation"):
                            n_ada = len(block.adaLN_modulation)
                            out_idx = 3 if n_ada == 4 else (2 if n_ada == 3 else 1)
                            state_dict[f"blocks.{i}.adaLN_modulation.{out_idx}.weight"] = slice_w1
                    state_dict.pop("ada_proj.1.weight", None)
                    state_dict.pop("ada_proj.weight", None)
                    state_dict.pop("ada_proj.1.bias", None)
                    state_dict.pop("ada_proj.bias", None)
                elif w0 is not None and w2 is not None:
                    state_dict = dict(state_dict)
                    for i, block in enumerate(getattr(target_model, "blocks", [])):
                        slice_w2 = w2[i * 8 * dim : (i + 1) * 8 * dim]
                        if hasattr(block, "adaLN_modulation"):
                            n_ada = len(block.adaLN_modulation)
                            if n_ada == 4:
                                state_dict[f"blocks.{i}.adaLN_modulation.1.weight"] = w0
                                state_dict[f"blocks.{i}.adaLN_modulation.3.weight"] = slice_w2
                            elif n_ada == 3:
                                state_dict[f"blocks.{i}.adaLN_modulation.0.weight"] = w0
                                state_dict[f"blocks.{i}.adaLN_modulation.2.weight"] = slice_w2
                            else:
                                slice_f = slice_w2.float() if hasattr(slice_w2, "float") else slice_w2
                                w0_f = w0.float() if hasattr(w0, "float") else w0
                                state_dict[f"blocks.{i}.adaLN_modulation.1.weight"] = (slice_f @ w0_f).to(dtype=slice_w2.dtype)
                    state_dict.pop("ada_proj.0.weight", None)
                    state_dict.pop("ada_proj.2.weight", None)
                    state_dict.pop("ada_proj.0.bias", None)
                    state_dict.pop("ada_proj.2.bias", None)
            elif not per_block and has_new_ada and not has_old_ada:
                state_dict = dict(state_dict)
                w2_slices = []
                w0 = None
                depth = getattr(target_model, "depth", len(getattr(target_model, "blocks", [])))
                for i in range(depth):
                    k0 = f"blocks.{i}.adaLN_modulation.0.weight"
                    k2 = f"blocks.{i}.adaLN_modulation.2.weight"
                    k1 = f"blocks.{i}.adaLN_modulation.1.weight"
                    if k0 in state_dict and w0 is None:
                        w0 = state_dict.pop(k0)
                    else:
                        state_dict.pop(k0, None)
                    if k2 in state_dict:
                        w2_slices.append(state_dict.pop(k2))
                    elif k1 in state_dict:
                        w2_slices.append(state_dict.pop(k1))
                if w0 is not None and w2_slices:
                    state_dict["ada_proj.0.weight"] = w0
                    state_dict["ada_proj.2.weight"] = torch.cat(w2_slices, dim=0)
                elif w2_slices:
                    state_dict["ada_proj.1.weight"] = torch.cat(w2_slices, dim=0)

        if "proj_in.weight" in state_dict:
            pw = state_dict["proj_in.weight"]
            use_coord = getattr(target_model, "use_coord_conv", False)
            latent_ch = getattr(target_model, "latent_channels", 32)
            if use_coord and pw.shape[1] == latent_ch:
                expanded = torch.zeros(
                    (pw.shape[0], latent_ch + 3, pw.shape[2], pw.shape[3]),
                    dtype=pw.dtype,
                    device=pw.device,
                )
                expanded[:, :latent_ch] = pw
                state_dict["proj_in.weight"] = expanded
            elif not use_coord and pw.shape[1] == latent_ch + 3:
                state_dict["proj_in.weight"] = pw[:, :latent_ch]

        if precomputed_scales is not None:
            scales_map = precomputed_scales
        else:
            scales_map: Dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.endswith(".scale") and isinstance(v, torch.Tensor):
                    scales_map[k] = v

        if precomputed_chunks is not None:
            chunks = precomputed_chunks
        else:
            chunks = partition_state_dict_into_chunks(state_dict, self.chunk_size_bytes)
        total_chunks = len(chunks)
        total_bytes = sum(calculate_tensor_bytes(v) for v in state_dict.values())
        total_mb = total_bytes / (1024**2)

        logger.info(
            f"[{tag}] Starting 128MB chunked weight transfer to {device}: "
            f"{total_mb:.2f} MB across {total_chunks} chunk(s) (chunk_size={self.chunk_size_mb} MB)."
        )

        total_ssd_time = 0.0
        total_vram_time = 0.0
        total_free_time = 0.0

        for idx, chunk_keys in enumerate(chunks):
            chunk_num = idx + 1
            chunk_bytes = sum(calculate_tensor_bytes(state_dict.get(k)) for k in chunk_keys)
            chunk_mb = chunk_bytes / (1024**2)

            t0 = time.perf_counter()
            chunk_ram: Dict[str, Any] = {}
            for k in chunk_keys:
                val = state_dict.get(k)
                if isinstance(val, torch.Tensor):
                    chunk_ram[k] = val.clone() if (val.is_shared() or val.device.type != "cpu") else val
                else:
                    chunk_ram[k] = val
            t_ssd_read = time.perf_counter() - t0
            total_ssd_time += t_ssd_read

            t1 = time.perf_counter()
            for k in chunk_keys:
                tensor = chunk_ram[k]
                if not isinstance(tensor, torch.Tensor):
                    continue

                scale_tensor = None
                if k.endswith(".weight"):
                    scale_key = k[:-7] + ".scale"
                    if scale_key in chunk_ram:
                        scale_tensor = chunk_ram[scale_key]
                    elif scale_key in scales_map:
                        scale_tensor = scales_map[scale_key]
                    elif scale_key in state_dict:
                        scale_tensor = state_dict[scale_key]

                apply_tensor_to_model(model, k, tensor, device, scale_tensor=scale_tensor)

            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t_ram_to_vram = time.perf_counter() - t1
            total_vram_time += t_ram_to_vram

            t2 = time.perf_counter()
            del chunk_ram
            if allow_delete_keys:
                for k in chunk_keys:
                    state_dict.pop(k, None)
            t_ram_free = time.perf_counter() - t2
            total_free_time += t_ram_free

            ssd_mb_s = chunk_mb / max(t_ssd_read, 1e-6)
            vram_mb_s = chunk_mb / max(t_ram_to_vram, 1e-6)
            total_chunk_time = (t_ssd_read + t_ram_to_vram + t_ram_free) * 1000

            vram_alloc = ""
            if str(device).startswith("cuda") and torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(device) / (1024**2)
                vram_res = torch.cuda.memory_reserved(device) / (1024**2)
                vram_alloc = f" | VRAM: {vram_used:.1f}MB (res {vram_res:.1f}MB)"

            logger.info(
                f"[{tag}] Chunk {chunk_num}/{total_chunks} ({chunk_mb:.2f} MB, {len(chunk_keys)} tensors): "
                f"SSD->RAM: {t_ssd_read*1000:.2f} ms ({ssd_mb_s:.1f} MB/s) | "
                f"RAM->VRAM: {t_ram_to_vram*1000:.2f} ms ({vram_mb_s:.1f} MB/s) | "
                f"RAM cleanup: {t_ram_free*1000:.2f} ms | "
                f"Chunk total: {total_chunk_time:.2f} ms{vram_alloc}"
            )

        if allow_delete_keys:
            gc.collect()

        overall_time = total_ssd_time + total_vram_time + total_free_time
        overall_mb_s = total_mb / max(overall_time, 1e-6)
        logger.info(
            f"[{tag}] 128MB Chunked transfer complete! Total: {total_mb:.2f} MB in {overall_time*1000:.2f} ms "
            f"({overall_mb_s:.1f} MB/s) | SSD: {total_ssd_time*1000:.1f}ms, VRAM: {total_vram_time*1000:.1f}ms, Free: {total_free_time*1000:.1f}ms."
        )

        return {
            "total_bytes": total_bytes,
            "total_ssd_time": total_ssd_time,
            "total_vram_time": total_vram_time,
            "total_free_time": total_free_time,
            "overall_time": overall_time,
        }

    def stream_sharded_chunks_to_model(
        self,
        model: nn.Module,
        chunk_dir: str,
        device: torch.device,
        tag: str = "Shards",
        chunk_files: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        if chunk_files is None:
            if not os.path.exists(chunk_dir):
                raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

            manifest_path = os.path.join(chunk_dir, "manifest.json")
            chunk_files = []
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for item in manifest.get("chunks", []):
                    chunk_files.append(os.path.join(chunk_dir, item["filename"]))
            else:
                files = [f for f in os.listdir(chunk_dir) if f.endswith(".pt")]
                files.sort()
                chunk_files = [os.path.join(chunk_dir, f) for f in files]

        total_chunks = len(chunk_files)
        total_bytes = sum(os.path.getsize(p) for p in chunk_files)
        total_mb = total_bytes / (1024**2)

        logger.info(
            f"[{tag}] Streaming {total_chunks} physical 128MB chunk files from {chunk_dir} to {device} ({total_mb:.2f} MB total)..."
        )

        total_ssd_time = 0.0
        total_vram_time = 0.0
        total_free_time = 0.0

        for idx, chunk_file in enumerate(chunk_files):
            chunk_num = idx + 1
            file_bytes = os.path.getsize(chunk_file)
            file_mb = file_bytes / (1024**2)

            t0 = time.perf_counter()
            chunk_sd = torch.load(chunk_file, map_location="cpu", weights_only=False)
            t_ssd_read = time.perf_counter() - t0
            total_ssd_time += t_ssd_read

            scales_map = {k: v for k, v in chunk_sd.items() if k.endswith(".scale") and isinstance(v, torch.Tensor)}

            t1 = time.perf_counter()
            for k, tensor in chunk_sd.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                scale_tensor = None
                if k.endswith(".weight"):
                    scale_key = k[:-7] + ".scale"
                    scale_tensor = scales_map.get(scale_key, chunk_sd.get(scale_key))

                apply_tensor_to_model(model, k, tensor, device, scale_tensor=scale_tensor)

            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t_ram_to_vram = time.perf_counter() - t1
            total_vram_time += t_ram_to_vram

            t2 = time.perf_counter()
            del chunk_sd
            del scales_map
            t_ram_free = time.perf_counter() - t2
            total_free_time += t_ram_free

            ssd_mb_s = file_mb / max(t_ssd_read, 1e-6)
            vram_mb_s = file_mb / max(t_ram_to_vram, 1e-6)
            total_chunk_time = (t_ssd_read + t_ram_to_vram + t_ram_free) * 1000

            vram_alloc = ""
            if str(device).startswith("cuda") and torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(device) / (1024**2)
                vram_res = torch.cuda.memory_reserved(device) / (1024**2)
                vram_alloc = f" | VRAM: {vram_used:.1f}MB (res {vram_res:.1f}MB)"

            logger.info(
                f"[{tag}] Chunk {chunk_num}/{total_chunks} ({file_mb:.2f} MB, {os.path.basename(chunk_file)}): "
                f"SSD->RAM: {t_ssd_read*1000:.2f} ms ({ssd_mb_s:.1f} MB/s) | "
                f"RAM->VRAM: {t_ram_to_vram*1000:.2f} ms ({vram_mb_s:.1f} MB/s) | "
                f"RAM cleanup: {t_ram_free*1000:.2f} ms | "
                f"Chunk total: {total_chunk_time:.2f} ms{vram_alloc}"
            )

        overall_time = total_ssd_time + total_vram_time + total_free_time
        overall_mb_s = total_mb / max(overall_time, 1e-6)
        logger.info(
            f"[{tag}] Direct SSD 128MB chunked streaming complete! Total: {total_mb:.2f} MB in {overall_time*1000:.2f} ms "
            f"({overall_mb_s:.1f} MB/s) | SSD: {total_ssd_time*1000:.1f}ms, VRAM: {total_vram_time*1000:.1f}ms, Free: {total_free_time*1000:.1f}ms."
        )

        return {
            "total_bytes": total_bytes,
            "total_ssd_time": total_ssd_time,
            "total_vram_time": total_vram_time,
            "total_free_time": total_free_time,
            "overall_time": overall_time,
        }

    def load_checkpoint_chunked(
        self,
        model: nn.Module,
        checkpoint_path: str,
        device: torch.device,
        use_ema: bool = True,
        is_backbone_only: bool = True,
        load_lora: bool = True,
    ) -> Dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        chunk_dir_candidates = [
            f"{os.path.splitext(checkpoint_path)[0]}_chunks",
            f"{checkpoint_path}_chunks",
            checkpoint_path if os.path.isdir(checkpoint_path) else "",
        ]
        chunk_dir = next((d for d in chunk_dir_candidates if d and os.path.isdir(d)), None)

        if chunk_dir is not None:
            logger.info(f"[Loader] Found physical 128MB chunk directory: {chunk_dir}")
            manifest_file = os.path.join(chunk_dir, "manifest.json")
            metadata: Dict[str, Any] = {}
            if os.path.exists(manifest_file):
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    metadata = manifest_data.get("metadata", {})
            self.stream_sharded_chunks_to_model(
                model=model,
                chunk_dir=chunk_dir,
                device=device,
                tag="Shared Backbone" if is_backbone_only else "Full Model",
            )
            return metadata

        logger.info(f"[Loader] Opening checkpoint with mmap (zero full-weights copy): {checkpoint_path}")
        t_open = time.perf_counter()
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        logger.info(f"[Loader] Checkpoint opened in {(time.perf_counter() - t_open)*1000:.1f} ms.")

        config = checkpoint.get("config", {})
        metadata = {
            "config": config,
            "latent_mean": checkpoint.get("latent_mean"),
            "latent_std": checkpoint.get("latent_std"),
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "text_encoder_state_dict": checkpoint.get("text_encoder_state_dict"),
            "text_encoder_pooler_state_dict": checkpoint.get("text_encoder_pooler_state_dict"),
            "ema_text_proj_state_dict": checkpoint.get("ema_text_proj_state_dict"),
            "ema_text_pooler_state_dict": checkpoint.get("ema_text_pooler_state_dict"),
        }

        ema_key = "ema_state_dict" if "ema_state_dict" in checkpoint else "ema_model_state_dict"
        if use_ema and ema_key in checkpoint:
            raw_sd = checkpoint[ema_key]
        else:
            raw_sd = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))

        if is_backbone_only:
            target_sd = {
                clean_key(k): v for k, v in raw_sd.items()
                if "phase_experts." not in k
                and (load_lora or "phase_attn_lora." not in k)
                and not k.endswith("n_averaged")
            }
            tag = "Shared Backbone"
            if load_lora and any("phase_attn_lora." in k for k in raw_sd.keys()):
                setattr(unwrap_model_deep(model), "_lora_in_experts", False)
        else:
            target_sd = {clean_key(k): v for k, v in raw_sd.items() if not k.endswith("n_averaged")}
            tag = "Full Model"

        self.stream_state_dict_to_model(
            model=model,
            state_dict=target_sd,
            device=device,
            tag=tag,
            allow_delete_keys=True,
        )

        del checkpoint
        gc.collect()
        return metadata


class ChunkedResidencyManager:

    def __init__(
        self,
        model: nn.Module,
        experts_dir: str = "cp/experts",
        device: Optional[torch.device] = None,
        chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
        max_experts_vram: int = 1,
        max_experts_ram: int = 0,
        lora_in_experts: Optional[bool] = None,
    ):
        self.model = model
        self.experts_dir = experts_dir
        self.device = device or next(model.parameters()).device
        self.chunk_size_mb = chunk_size_mb
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        self.streamer = ChunkedWeightStreamer(chunk_size_mb=chunk_size_mb)

        self.max_experts_vram = max(1, int(max_experts_vram))
        self.max_experts_ram = max(0, int(max_experts_ram))

        self.vram_resident_experts: OrderedDict[int, None] = OrderedDict()

        self.ram_resident_experts: OrderedDict[int, Dict[str, Any]] = OrderedDict()

        self._active_phase: Optional[int] = None
        target_model = unwrap_model_deep(model)
        self.num_phases = getattr(target_model, "num_phases", 3)
        self._expert_cache: Dict[int, Dict[str, Any]] = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_threads: Dict[int, threading.Thread] = {}

        if lora_in_experts is not None:
            self._lora_in_experts: Optional[bool] = lora_in_experts
            setattr(target_model, "_lora_in_experts", lora_in_experts)
        else:
            self._lora_in_experts = getattr(target_model, "_lora_in_experts", None)

    @property
    def current_vram_phase(self) -> Optional[int]:
        return self._active_phase

    @current_vram_phase.setter
    def current_vram_phase(self, phase: Optional[int]):
        self._active_phase = phase
        if phase is not None and phase not in self.vram_resident_experts:
            self.vram_resident_experts[phase] = None

    def _find_expert_source(self, phase_idx: int) -> Tuple[Optional[str], bool]:
        if not os.path.exists(self.experts_dir):
            return None, False

        is_fp8 = getattr(self.model, "quantized_fp8", False) or getattr(self.model, "quantization", None) == "fp8"
        is_int8 = getattr(self.model, "quantized_int8", False) or getattr(self.model, "quantization", None) == "int8"

        if is_fp8:
            dir_candidates = [
                os.path.join(self.experts_dir, f"expert_phase_{phase_idx}_fp8_chunks"),
                os.path.join(self.experts_dir, f"expert_phase_{phase_idx}_int8_chunks"),
                os.path.join(self.experts_dir, f"expert_phase_{phase_idx}_chunks"),
            ]
            candidates = [
                f"expert_phase_{phase_idx}_fp8.pt",
                f"expert_phase_{phase_idx}_int8.pt",
                f"expert_phase_{phase_idx}.pt",
            ]
        elif is_int8:
            dir_candidates = [
                os.path.join(self.experts_dir, f"expert_phase_{phase_idx}_int8_chunks"),
                os.path.join(self.experts_dir, f"expert_phase_{phase_idx}_fp8_chunks"),
                os.path.join(self.experts_dir, f"expert_phase_{phase_idx}_chunks"),
            ]
            candidates = [
                f"expert_phase_{phase_idx}_int8.pt",
                f"expert_phase_{phase_idx}_fp8.pt",
                f"expert_phase_{phase_idx}.pt",
            ]
        else:
            dir_candidates = [
                os.path.join(self.experts_dir, f"expert_phase_{phase_idx}_chunks"),
                os.path.join(self.experts_dir, f"expert_phase_{phase_idx}_fp8_chunks"),
                os.path.join(self.experts_dir, f"expert_phase_{phase_idx}_int8_chunks"),
            ]
            candidates = [
                f"expert_phase_{phase_idx}.pt",
                f"expert_phase_{phase_idx}_fp8.pt",
                f"expert_phase_{phase_idx}_int8.pt",
            ]

        for d in dir_candidates:
            if os.path.isdir(d):
                return d, True

        for cand in candidates:
            p = os.path.join(self.experts_dir, cand)
            if os.path.exists(p):
                return p, False

        for f in os.listdir(self.experts_dir):
            full_p = os.path.join(self.experts_dir, f)
            if f.startswith(f"expert_phase_{phase_idx}") and os.path.isdir(full_p):
                return full_p, True
            if f.startswith(f"expert_phase_{phase_idx}") and f.endswith(".pt"):
                return full_p, False

        return None, False

    def _get_or_load_expert(self, phase_idx: int) -> Optional[Dict[str, Any]]:
        with self._prefetch_lock:
            if phase_idx in self._expert_cache:
                return self._expert_cache[phase_idx]

        source_path, is_dir = self._find_expert_source(phase_idx)
        if source_path is None or not os.path.exists(source_path):
            return None

        if is_dir:
            manifest_path = os.path.join(source_path, "manifest.json")
            chunk_files = []
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    for item in manifest.get("chunks", []):
                        chunk_files.append(os.path.join(source_path, item["filename"]))
                except Exception:
                    pass
            if not chunk_files:
                files = [f for f in os.listdir(source_path) if f.endswith(".pt")]
                files.sort()
                chunk_files = [os.path.join(source_path, f) for f in files]

            entry = {
                "source_path": source_path,
                "is_dir": True,
                "chunk_files": chunk_files,
            }
        else:
            try:
                data = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                data = torch.load(source_path, map_location="cpu", weights_only=False)

            expert_sd = data.get("state_dict", data)
            if self._lora_in_experts is None:
                has_lora = any("phase_attn_lora." in k for k in expert_sd.keys())
                self._lora_in_experts = has_lora
                setattr(unwrap_model_deep(self.model), "_lora_in_experts", has_lora)
            scales_map = {k: v for k, v in expert_sd.items() if k.endswith(".scale") and isinstance(v, torch.Tensor)}
            chunks = partition_state_dict_into_chunks(expert_sd, self.chunk_size_bytes)
            entry = {
                "source_path": source_path,
                "is_dir": False,
                "data": data,
                "state_dict": expert_sd,
                "scales_map": scales_map,
                "chunks": chunks,
            }

        with self._prefetch_lock:
            self._expert_cache[phase_idx] = entry
        return entry

    def _warm_expert(self, entry: Dict[str, Any]):
        if entry.get("is_dir"):
            for cf in entry.get("chunk_files", []):
                try:
                    with open(cf, "rb") as f:
                        while f.read(16 * 1024 * 1024):
                            pass
                except Exception:
                    pass
        else:
            sd = entry.get("state_dict", {})
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    try:
                        _ = v.untyped_storage()
                    except Exception:
                        pass

    def pin_expert_in_ram(self, phase_idx: int) -> Optional[Dict[str, Any]]:
        if self.max_experts_ram <= 0:
            return None

        with self._prefetch_lock:
            if phase_idx in self.vram_resident_experts:
                return None
            if phase_idx in self.ram_resident_experts:
                return self.ram_resident_experts[phase_idx]

        entry = self._get_or_load_expert(phase_idx)
        if entry is None or "state_dict" not in entry:
            return None

        sd = entry["state_dict"]
        pinned_sd = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                try:
                    pinned_sd[k] = v.pin_memory() if not v.is_pinned() else v
                except Exception:
                    pinned_sd[k] = v.clone()
            else:
                pinned_sd[k] = v

        with self._prefetch_lock:
            if phase_idx in self.vram_resident_experts:
                del pinned_sd
                return None
            while len(self.ram_resident_experts) >= self.max_experts_ram:
                self.ram_resident_experts.popitem(last=False)
            self.ram_resident_experts[phase_idx] = pinned_sd
        return pinned_sd

    def fast_stream_expert_to_model(self, phase_idx: int, expert_sd: Dict[str, Any]) -> float:
        t0 = time.perf_counter()
        target_model = unwrap_model_deep(self.model)

        scales_map = {k: v for k, v in expert_sd.items() if k.endswith(".scale")}

        for k, tensor in expert_sd.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            scale_tensor = None
            if k.endswith(".weight"):
                scale_key = k[:-7] + ".scale"
                if scale_key in expert_sd:
                    scale_tensor = expert_sd[scale_key]
                elif scale_key in scales_map:
                    scale_tensor = scales_map[scale_key]

            apply_tensor_to_model(
                target_model,
                k,
                tensor,
                self.device,
                scale_tensor=scale_tensor,
            )

        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.current_stream(self.device).synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return elapsed_ms

    def evict_expert_from_vram(self, phase_idx: int, move_to_ram: bool = False):
        if phase_idx in self.vram_resident_experts:
            del self.vram_resident_experts[phase_idx]
        if self._active_phase == phase_idx:
            self._active_phase = None

        t0 = time.perf_counter()
        target_model = unwrap_model_deep(self.model)

        if move_to_ram and self.max_experts_ram > 0:
            while len(self.ram_resident_experts) >= self.max_experts_ram:
                lru_ram_phase, _ = self.ram_resident_experts.popitem(last=False)
                logger.info(
                    f"[ResidencyManager] RAM capacity ({self.max_experts_ram}) reached: "
                    f"purged Phase {lru_ram_phase} from RAM cache."
                )

            if phase_idx not in self.ram_resident_experts:
                self.pin_expert_in_ram(phase_idx)

        target_clear_lora = getattr(self, "_lora_in_experts", None)
        if target_clear_lora is None:
            target_clear_lora = getattr(target_model, "_lora_in_experts", False)

        if hasattr(target_model, "clear_active_expert"):
            target_model.clear_active_expert(phase_idx, clear_lora=target_clear_lora)
        else:
            clear_lora = target_clear_lora
            for block in getattr(target_model, "blocks", []):
                if hasattr(block, "phase_experts") and block.phase_experts is not None:
                    if phase_idx < len(block.phase_experts):
                        exp = block.phase_experts[phase_idx]
                        for param in exp.parameters():
                            param.data = torch.empty((0, 0), dtype=param.dtype, device="cpu")
                        for buf in exp.buffers():
                            buf.data = torch.empty((0, 0), dtype=buf.dtype, device="cpu")
                if clear_lora and hasattr(block, "phase_attn_lora") and block.phase_attn_lora is not None:
                    if phase_idx < len(block.phase_attn_lora):
                        lora_m = block.phase_attn_lora[phase_idx]
                        for param in lora_m.parameters():
                            param.data = torch.empty((0, 0), dtype=param.dtype, device="cpu")
                        for buf in lora_m.buffers():
                            buf.data = torch.empty((0, 0), dtype=buf.dtype, device="cpu")

        overlap = set(self.vram_resident_experts.keys()) & set(self.ram_resident_experts.keys())
        assert len(overlap) == 0, f"Critical invariant violation: expert(s) {overlap} present in both RAM and VRAM!"

        t_evict = time.perf_counter() - t0
        vram_alloc = ""
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
            res_mb = torch.cuda.memory_reserved(self.device) / (1024**2)
            vram_alloc = f" | VRAM alloc: {alloc_mb:.1f} MB (res: {res_mb:.1f} MB)"
        logger.info(
            f"[ResidencyManager] Phase {phase_idx} expert evicted from VRAM in {t_evict*1000:.2f} ms.{vram_alloc}"
        )

    def evict_current_expert(self):
        if self._active_phase is not None:
            self.evict_expert_from_vram(self._active_phase, move_to_ram=(self.max_experts_ram > 0))
        elif self.vram_resident_experts:
            oldest = next(iter(self.vram_resident_experts))
            self.evict_expert_from_vram(oldest, move_to_ram=(self.max_experts_ram > 0))

    def configure_residency(self, max_experts_vram: int = 1, max_experts_ram: int = 0) -> Dict[str, Any]:
        self.max_experts_vram = max(1, int(max_experts_vram))
        self.max_experts_ram = max(0, int(max_experts_ram))

        logger.info(
            f"[ResidencyManager] Reconfigured residency limits: max_experts_vram={self.max_experts_vram}, "
            f"max_experts_ram={self.max_experts_ram}"
        )

        while len(self.vram_resident_experts) > self.max_experts_vram:
            candidates = [p for p in self.vram_resident_experts if p != self._active_phase]
            evict_p = candidates[0] if candidates else next(iter(self.vram_resident_experts))
            self.evict_expert_from_vram(evict_p, move_to_ram=(self.max_experts_ram > 0))

        while len(self.ram_resident_experts) > self.max_experts_ram:
            lru_ram_phase, _ = self.ram_resident_experts.popitem(last=False)
            logger.info(f"[ResidencyManager] Trimmed excess Phase {lru_ram_phase} from RAM cache.")

        if self.max_experts_ram > 0:
            for p in range(self.num_phases):
                if p not in self.vram_resident_experts and len(self.ram_resident_experts) < self.max_experts_ram:
                    self.pin_expert_in_ram(p)

        return self.get_residency_status()

    def sanitize_non_resident_experts(self):
        target_model = unwrap_model_deep(self.model)
        target_clear_lora = getattr(self, "_lora_in_experts", None)
        if target_clear_lora is None:
            target_clear_lora = getattr(target_model, "_lora_in_experts", False)
        clear_lora = target_clear_lora
        for block in getattr(target_model, "blocks", []):
            if hasattr(block, "phase_experts") and block.phase_experts is not None:
                for p, exp in enumerate(block.phase_experts):
                    if p not in self.vram_resident_experts:
                        for param in exp.parameters():
                            if param.device.type == "cuda" or param.numel() > 0:
                                param.data = torch.empty((0, 0), dtype=param.dtype, device="cpu")
                        for buf in exp.buffers():
                            if buf.device.type == "cuda" or buf.numel() > 0:
                                buf.data = torch.empty((0, 0), dtype=buf.dtype, device="cpu")
            if clear_lora and hasattr(block, "phase_attn_lora") and block.phase_attn_lora is not None:
                for p, lora_m in enumerate(block.phase_attn_lora):
                    if p not in self.vram_resident_experts:
                        for param in lora_m.parameters():
                            if param.device.type == "cuda" or param.numel() > 0:
                                param.data = torch.empty((0, 0), dtype=param.dtype, device="cpu")
                        for buf in lora_m.buffers():
                            if buf.device.type == "cuda" or buf.numel() > 0:
                                buf.data = torch.empty((0, 0), dtype=buf.dtype, device="cpu")

    def get_residency_status(self) -> Dict[str, Any]:
        vram_list = list(self.vram_resident_experts.keys())
        ram_list = list(self.ram_resident_experts.keys())
        all_phases = set(range(self.num_phases))
        ssd_list = sorted(list(all_phases - set(vram_list) - set(ram_list)))

        return {
            "max_experts_vram": self.max_experts_vram,
            "max_experts_ram": self.max_experts_ram,
            "vram_resident_experts": vram_list,
            "ram_resident_experts": ram_list,
            "ssd_resident_experts": ssd_list,
            "active_phase": self._active_phase,
            "mutual_exclusion_verified": len(set(vram_list) & set(ram_list)) == 0,
        }

    def prefetch_next(self, next_phase: int):
        if next_phase >= self.num_phases:
            return

        with self._prefetch_lock:
            if (
                next_phase in self.ram_resident_experts
                or next_phase in self.vram_resident_experts
            ):
                return
            if next_phase in self._prefetch_threads and self._prefetch_threads[next_phase].is_alive():
                return

            def _prefetch_worker():
                try:
                    entry = self._get_or_load_expert(next_phase)
                    if entry is not None:
                        self._warm_expert(entry)
                        if self.max_experts_ram > 0 and next_phase not in self.vram_resident_experts:
                            self.pin_expert_in_ram(next_phase)
                except Exception as e:
                    logger.debug(f"[ResidencyManager] Prefetch background task error: {e}")

            t = threading.Thread(
                target=_prefetch_worker,
                daemon=True,
                name=f"Prefetch-Expert-{next_phase}",
            )
            self._prefetch_threads[next_phase] = t
            t.start()
    def stream_expert(self, phase_idx: int) -> bool:
        if phase_idx in self.vram_resident_experts:
            self.vram_resident_experts.move_to_end(phase_idx)
            self._active_phase = phase_idx
            self.ram_resident_experts.pop(phase_idx, None)
            logger.info(
                f"[ResidencyManager] Phase {phase_idx} expert is already RESIDENT in VRAM (0 ms switch). "
                f"VRAM: {list(self.vram_resident_experts.keys())}, RAM: {list(self.ram_resident_experts.keys())}."
            )
            return True

        logger.info(
            f"[ResidencyManager] === Phase Switch: activating Phase {phase_idx} "
            f"(VRAM: {list(self.vram_resident_experts.keys())} / {self.max_experts_vram}, "
            f"RAM: {list(self.ram_resident_experts.keys())} / {self.max_experts_ram}) ==="
        )

        prefetch_t = self._prefetch_threads.get(phase_idx)
        if prefetch_t is not None and prefetch_t.is_alive():
            prefetch_t.join()

        cached_ram_sd = None
        if phase_idx in self.ram_resident_experts:
            cached_ram_sd = self.ram_resident_experts.pop(phase_idx)
            logger.info(
                f"[ResidencyManager] Extracted Phase {phase_idx} from RAM cache for transfer to VRAM. "
                f"RAM slots now: {len(self.ram_resident_experts)}/{self.max_experts_ram}."
            )

        while len(self.vram_resident_experts) >= self.max_experts_vram:
            candidates = [p for p in self.vram_resident_experts if p != phase_idx]
            evict_p = candidates[0] if candidates else next(iter(self.vram_resident_experts))
            self.evict_expert_from_vram(evict_p, move_to_ram=(self.max_experts_ram > 0))

        tag = f"Expert Phase {phase_idx}"

        if cached_ram_sd is not None:
            t_ram_start = time.perf_counter()
            logger.info(f"[ResidencyManager] Streaming Phase {phase_idx} directly from pinned RAM cache to VRAM...")
            self.fast_stream_expert_to_model(phase_idx, cached_ram_sd)
            logger.info(
                f"[ResidencyManager] Phase {phase_idx} transferred from RAM to VRAM in "
                f"{(time.perf_counter() - t_ram_start)*1000:.2f} ms."
            )
        else:
            expert_entry = self._get_or_load_expert(phase_idx)
            if expert_entry is None:
                logger.warning(
                    f"[ResidencyManager] Expert source for Phase {phase_idx} not found in {self.experts_dir}."
                )
                return False

            if expert_entry.get("is_dir", False):
                self.streamer.stream_sharded_chunks_to_model(
                    model=self.model,
                    chunk_dir=expert_entry["source_path"],
                    device=self.device,
                    tag=tag,
                    chunk_files=expert_entry.get("chunk_files"),
                )
            else:
                expert_sd = expert_entry["state_dict"]
                self.fast_stream_expert_to_model(phase_idx, expert_sd)

        self.vram_resident_experts[phase_idx] = None
        self.ram_resident_experts.pop(phase_idx, None)
        self._active_phase = phase_idx

        self.sanitize_non_resident_experts()

        overlap = set(self.vram_resident_experts.keys()) & set(self.ram_resident_experts.keys())
        assert len(overlap) == 0, f"Critical invariant violation: expert(s) {overlap} present in BOTH RAM and VRAM!"

        next_p = phase_idx + 1
        if (
            next_p < self.num_phases
            and next_p not in self.vram_resident_experts
            and next_p not in self.ram_resident_experts
        ):
            self.prefetch_next(next_p)

        vram_alloc = ""
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
            res_mb = torch.cuda.memory_reserved(self.device) / (1024**2)
            vram_alloc = f" (VRAM alloc: {alloc_mb:.1f} MB, res: {res_mb:.1f} MB)"

        logger.info(
            f"[ResidencyManager] Phase {phase_idx} is now RESIDENT in VRAM{vram_alloc}. "
            f"VRAM experts: {list(self.vram_resident_experts.keys())}, "
            f"RAM experts: {list(self.ram_resident_experts.keys())}. "
            f"Shared SwiGLU FFN, Per-Block AdaLN & Backbone intact."
        )
        return True

    def stream_phase(self, phase_idx: int):
        return self.stream_expert(phase_idx)

    def clear_cache(self):
        with self._prefetch_lock:
            self._expert_cache.clear()
            self._prefetch_threads.clear()
        self.ram_resident_experts.clear()
        gc.collect()


def shard_state_dict_to_128mb_chunks(
    state_dict: Dict[str, Any],
    out_dir: str,
    prefix: str = "chunk",
    chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    chunks = partition_state_dict_into_chunks(state_dict, chunk_size_bytes=chunk_size_bytes)

    saved_files = []
    manifest = {
        "chunk_size_mb": chunk_size_mb,
        "total_chunks": len(chunks),
        "metadata": metadata or {},
        "chunks": [],
    }

    for idx, keys in enumerate(chunks):
        chunk_fname = f"{prefix}_{idx:03d}.pt"
        chunk_path = os.path.join(out_dir, chunk_fname)
        chunk_sd = {k: state_dict[k] for k in keys}
        torch.save(chunk_sd, chunk_path)
        saved_files.append(chunk_path)
        manifest["chunks"].append({
            "index": idx,
            "filename": chunk_fname,
            "keys": keys,
            "size_bytes": os.path.getsize(chunk_path),
        })
        del chunk_sd
        gc.collect()

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"[Sharder] Successfully saved {len(saved_files)} chunks ({chunk_size_mb}MB each) to {out_dir}.")
    return saved_files
