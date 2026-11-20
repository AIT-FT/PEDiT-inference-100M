import inspect
import math
import os
import sys
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

if sys.platform == "win32":
    os.environ["TORCH_WARM_POOL"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch._dynamo


from streamer import (
    ChunkedWeightStreamer,
    ChunkedResidencyManager,
    apply_tensor_to_model,
    partition_state_dict_into_chunks,
    DEFAULT_CHUNK_SIZE_MB,
    DEFAULT_CHUNK_SIZE_BYTES,
)


_ROPE_CACHE = OrderedDict()
_ROPE_CACHE_MAX = 128
_ROPE_LOCK = threading.Lock()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (int(dim),)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            w = self.weight if self.weight.dtype == x.dtype else self.weight.to(dtype=x.dtype)
            return F.rms_norm(x, self.normalized_shape, w, self.eps)
        dtype = x.dtype
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        w = self.weight if self.weight.dtype == torch.float32 else self.weight.float()
        return (x_f * w).to(dtype)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, heads: int = 8, inner_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim if inner_dim is not None else min(dim, 512)
        if self.inner_dim % heads == 0:
            self.heads = heads
        else:
            self.heads = math.gcd(self.inner_dim, max(1, heads))
        self.head_dim = self.inner_dim // self.heads

        self.probe = nn.Parameter(torch.randn(1, 1, self.inner_dim) * 0.02)
        self.null_embedding = nn.Parameter(torch.randn(dim) * 0.02)

        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        target_device = self.k_proj.weight.device
        if x.device != target_device:
            x = x.to(device=target_device)
        if not torch.is_autocast_enabled():
            target_dtype = self.k_proj.weight.dtype
            if x.dtype != target_dtype:
                x = x.to(dtype=target_dtype)

        B, N, _ = x.shape
        probe_in = self.probe.to(device=x.device, dtype=x.dtype).expand(B, -1, -1)

        q = self.q_proj(probe_in).view(B, 1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        has_valid_2d = None
        if mask is not None:
            bool_mask = mask.view(B, 1, 1, N).bool()
            has_valid = bool_mask.any(dim=-1, keepdim=True)
            has_valid_2d = has_valid.view(B, 1)
            attn_mask = torch.where(has_valid, bool_mask, torch.ones_like(bool_mask))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, self.inner_dim)
        out = self.out_proj(out)

        if has_valid_2d is not None:
            out = torch.where(
                has_valid_2d,
                out,
                self.null_embedding.view(1, -1).to(device=out.device, dtype=out.dtype),
            )

        return out

class Int8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer("scale", torch.ones((out_features, 1), dtype=torch.float16, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear: nn.Linear, scale_dtype: torch.dtype = torch.float16) -> "Int8Linear":
        dev = linear.weight.device
        l8 = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, device=dev)
        w = linear.weight.detach().float()
        max_val = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = max_val / 127.0
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
        l8.weight.copy_(w_int8)
        l8.scale.copy_(scale.to(scale_dtype))
        if linear.bias is not None:
            l8.bias.data.copy_(linear.bias.detach().to(scale_dtype))
        return l8

    def to_float(self, dtype: torch.dtype = torch.float32) -> nn.Linear:
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=dtype,
        )
        w = self.weight.to(dtype) * self.scale.to(dtype)
        linear.weight.data.copy_(w)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.to(dtype))
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        target_dtype = x.dtype
        x_2d = x.reshape(-1, self.in_features)

        if (
            x.is_cuda
            and x_2d.shape[0] > 16
            and self.in_features % 16 == 0
            and self.out_features % 8 == 0
            and hasattr(torch, "_int_mm")
        ):
            try:
                x_scale = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
                x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)
                out_int32 = torch._int_mm(x_int8, self.weight.t())
                combined_scale = (x_scale.float() * self.scale.t().float()).to(target_dtype)
                out = out_int32.to(target_dtype) * combined_scale
                if self.bias is not None:
                    out = out + self.bias.to(target_dtype)
                return out.view(*orig_shape[:-1], self.out_features)
            except Exception:
                pass

        if x.device.type == "cpu" and target_dtype == torch.float16:
            w = self.weight.to(torch.float32)
            out = F.linear(x.float(), w, None) * self.scale.to(torch.float32).t()
            if self.bias is not None:
                out = out + self.bias.to(torch.float32)
            return out.to(torch.float16)

        w = self.weight.to(target_dtype)
        out = F.linear(x, w, None) * self.scale.to(target_dtype).t()
        if self.bias is not None:
            out = out + self.bias.to(target_dtype)
        return out


def quantize_module_to_int8(module: nn.Module, exclude_types: tuple = ()) -> nn.Module:
    for name, child in module.named_children():
        if exclude_types and isinstance(child, exclude_types):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear.from_float(child))
        elif isinstance(child, FP8Linear):
            float_lin = child.to_float(dtype=torch.float32)
            setattr(module, name, Int8Linear.from_float(float_lin))
        else:
            quantize_module_to_int8(child, exclude_types=exclude_types)
    return module


FP8_E4M3_AVAILABLE = hasattr(torch, "float8_e4m3fn")
FP8_E5M2_AVAILABLE = hasattr(torch, "float8_e5m2")

def get_default_fp8_dtype():
    if torch.cuda.is_available():
        try:
            cc = torch.cuda.get_device_capability()
            if cc >= (8, 9) and FP8_E4M3_AVAILABLE:
                return torch.float8_e4m3fn, 448.0
            elif FP8_E5M2_AVAILABLE:
                return torch.float8_e5m2, 57344.0
        except Exception:
            pass
    if FP8_E5M2_AVAILABLE:
        return torch.float8_e5m2, 57344.0
    elif FP8_E4M3_AVAILABLE:
        return torch.float8_e4m3fn, 448.0
    return torch.int8, 127.0

FP8_DTYPE, FP8_MAX = get_default_fp8_dtype()


class FP8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        fp8_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if fp8_dtype is None:
            fp8_dtype = FP8_DTYPE
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_dtype = fp8_dtype
        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=fp8_dtype, device=device))
        self.register_buffer("scale", torch.ones((out_features, 1), dtype=torch.float16, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter("bias", None)

    def _apply(self, fn, recurse=True):
        orig_weight = self._buffers.get("weight", None)
        if orig_weight is not None:
            w_dtype = orig_weight.dtype
            probe = fn(torch.empty(0, device=orig_weight.device, dtype=torch.float32))
            if probe.device != orig_weight.device:
                self._buffers["weight"] = orig_weight.to(device=probe.device, dtype=w_dtype)
            self._buffers.pop("weight", None)
            try:
                ret = super()._apply(fn, recurse=recurse)
            finally:
                self._buffers["weight"] = orig_weight if probe.device == orig_weight.device else orig_weight.to(device=probe.device, dtype=w_dtype)
            return ret
        return super()._apply(fn, recurse=recurse)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        scale_dtype: torch.dtype = torch.float16,
        fp8_dtype: Optional[torch.dtype] = None,
    ) -> "FP8Linear":
        if fp8_dtype is None:
            fp8_dtype = FP8_DTYPE
        dev = linear.weight.device
        l8 = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, fp8_dtype=fp8_dtype, device=dev)
        w = linear.weight.detach().float()
        max_val = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)

        fp8_max = FP8_MAX
        if hasattr(torch, "finfo") and fp8_dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
            try:
                fp8_max = float(torch.finfo(fp8_dtype).max)
            except Exception:
                fp8_max = FP8_MAX

        scale = max_val / fp8_max
        w_scaled = (w / scale).clamp(-fp8_max, fp8_max)
        if fp8_dtype == torch.int8:
            w_fp8 = w_scaled.round().to(torch.int8)
        else:
            w_fp8 = w_scaled.to(fp8_dtype)

        l8.weight.copy_(w_fp8)
        l8.scale.copy_(scale.to(scale_dtype))
        if linear.bias is not None:
            l8.bias.data.copy_(linear.bias.detach().to(scale_dtype))
        return l8

    def to_float(self, dtype: torch.dtype = torch.float32) -> nn.Linear:
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=dtype,
        )
        w = self.weight.to(dtype) * self.scale.to(dtype)
        linear.weight.data.copy_(w)
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.to(dtype))
        return linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        if "float8" in str(target_dtype):
            target_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            x = x.to(target_dtype)

        if x.device.type == "cpu" and target_dtype == torch.float16:
            w = self.weight.to(torch.float32)
            out = F.linear(x.float(), w, None) * self.scale.to(torch.float32).t()
            if self.bias is not None:
                out = out + self.bias.to(torch.float32)
            return out.to(torch.float16)

        w = self.weight.to(target_dtype)
        out = F.linear(x, w, None) * self.scale.to(target_dtype).t()
        if self.bias is not None:
            out = out + self.bias.to(target_dtype)
        return out
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, dtype={self.weight.dtype}"


def quantize_module_to_fp8(module: nn.Module, exclude_types: tuple = (), fp8_dtype: Optional[torch.dtype] = None) -> nn.Module:
    if fp8_dtype is None:
        fp8_dtype = FP8_DTYPE
    for name, child in module.named_children():
        if exclude_types and isinstance(child, exclude_types):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, FP8Linear.from_float(child, fp8_dtype=fp8_dtype))
        elif isinstance(child, Int8Linear):
            float_lin = child.to_float(dtype=torch.float32)
            setattr(module, name, FP8Linear.from_float(float_lin, fp8_dtype=fp8_dtype))
        else:
            quantize_module_to_fp8(child, exclude_types=exclude_types, fp8_dtype=fp8_dtype)
    return module


ExpertPipelineStreamer = ChunkedResidencyManager


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "TimestepEmbedding dim must be even"

        self.dim = dim
        half_dim = dim // 2
        emb_scale = math.log(10000.0) / max(1, half_dim - 1)
        freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32) * emb_scale)
        self.register_buffer("freqs", freqs, persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_f = t.flatten().float() * 1000.0
        freqs = self.freqs if self.freqs.device == t_f.device else self.freqs.to(device=t_f.device)

        args = t_f[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if not torch.is_autocast_enabled():
            target_dtype = self.mlp[0].weight.dtype
            if target_dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64) or "float8" in str(target_dtype):
                target_dtype = torch.float32
            if emb.dtype != target_dtype:
                emb = emb.to(dtype=target_dtype)

        return self.mlp(emb)


def _get_rope_2d(
    grid_h: int,
    grid_w: int,
    head_dim: int,
    device: torch.device,
    base: float = 16.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for symmetric 2D RoPE, got {head_dim}"
    eff_base = max(float(base), 1.0)
    device_key = str(device)
    key = (grid_h, grid_w, head_dim, eff_base, device_key)

    is_compiling = False
    try:
        is_compiling = torch.compiler.is_compiling()
    except Exception:
        pass

    if not is_compiling:
        with _ROPE_LOCK:
            if key in _ROPE_CACHE:
                val = _ROPE_CACHE.pop(key)
                _ROPE_CACHE[key] = val
                return val

    half_dim = head_dim // 2
    dim_y = half_dim // 2
    dim_x = half_dim - dim_y

    freqs_y = 1.0 / (
        eff_base ** (torch.arange(0, dim_y, device=device, dtype=torch.float32) / max(dim_y, 1))
    )
    freqs_x = 1.0 / (
        eff_base ** (torch.arange(0, dim_x, device=device, dtype=torch.float32) / max(dim_x, 1))
    )

    coord_scale = 16.0 / max(float(max(grid_h, grid_w)), 16.0)
    y = (torch.arange(grid_h, device=device, dtype=torch.float32) - (grid_h - 1) / 2.0) * coord_scale
    x = (torch.arange(grid_w, device=device, dtype=torch.float32) - (grid_w - 1) / 2.0) * coord_scale

    angles_y = (
        torch.outer(y, freqs_y)
        .view(grid_h, 1, dim_y)
        .expand(grid_h, grid_w, dim_y)
        .reshape(-1, dim_y)
    )
    angles_x = (
        torch.outer(x, freqs_x)
        .view(1, grid_w, dim_x)
        .expand(grid_h, grid_w, dim_x)
        .reshape(-1, dim_x)
    )

    angles = torch.cat([angles_y, angles_x], dim=-1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    if not is_compiling:
        with _ROPE_LOCK:
            _ROPE_CACHE[key] = (cos, sin)
            if len(_ROPE_CACHE) > _ROPE_CACHE_MAX:
                _ROPE_CACHE.popitem(last=False)

    return cos, sin


def apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_h: int,
    grid_w: int,
    base: float = 16.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial_len = grid_h * grid_w
    N = q.shape[2]
    cos, sin = _get_rope_2d(grid_h, grid_w, q.shape[-1], q.device, base=base)

    cos_q = cos.to(q.dtype).unsqueeze(0).unsqueeze(0)
    sin_q = sin.to(q.dtype).unsqueeze(0).unsqueeze(0)
    cos_k = cos_q if k.dtype == q.dtype else cos.to(k.dtype).unsqueeze(0).unsqueeze(0)
    sin_k = sin_q if k.dtype == q.dtype else sin.to(k.dtype).unsqueeze(0).unsqueeze(0)

    if N > spatial_len:
        q_sp, q_reg = q[:, :, :spatial_len], q[:, :, spatial_len:]
        k_sp, k_reg = k[:, :, :spatial_len], k[:, :, spatial_len:]
        q1, q2 = q_sp.chunk(2, dim=-1)
        k1, k2 = k_sp.chunk(2, dim=-1)
        q_sp_out = torch.cat([q1 * cos_q - q2 * sin_q, q2 * cos_q + q1 * sin_q], dim=-1)
        k_sp_out = torch.cat([k1 * cos_k - k2 * sin_k, k2 * cos_k + k1 * sin_k], dim=-1)
        return torch.cat([q_sp_out, q_reg], dim=2), torch.cat([k_sp_out, k_reg], dim=2)

    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)

    q_out = torch.cat([q1 * cos_q - q2 * sin_q, q2 * cos_q + q1 * sin_q], dim=-1)
    k_out = torch.cat([k1 * cos_k - k2 * sin_k, k2 * cos_k + k1 * sin_k], dim=-1)

    return q_out, k_out


try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except Exception:
    flash_attn_func = None
    _HAS_FLASH_ATTN = False

try:
    _SDPA_PARAMS = inspect.signature(F.scaled_dot_product_attention).parameters
    _SDPA_SUPPORTS_GQA = "enable_gqa" in _SDPA_PARAMS
except Exception:
    _SDPA_SUPPORTS_GQA = False


def sdpa_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]

    kwargs = {}
    if scale is not None:
        kwargs["scale"] = scale

    is_compiling = False
    try:
        is_compiling = torch.compiler.is_compiling()
    except Exception:
        pass

    if (
        _HAS_FLASH_ATTN
        and not is_compiling
        and q.is_cuda
        and attn_mask is None
        and q.dtype in (torch.float16, torch.bfloat16)
    ):
        try:
            out_fa = flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                softmax_scale=scale,
            )
            return out_fa.transpose(1, 2)
        except Exception:
            pass

    if num_heads == num_kv_heads:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, **kwargs)

    if _SDPA_SUPPORTS_GQA:
        try:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=True, **kwargs
            )
        except (TypeError, RuntimeError):
            pass

    rep = num_heads // num_kv_heads
    k_rep = k.repeat_interleave(rep, dim=1)
    v_rep = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask, **kwargs)


class PhaseAttnLoRA(nn.Module):
    def __init__(
        self,
        dim: int,
        kv_dim: int,
        rank: int = 64,
        alpha: Optional[float] = None,
        use_k_lora: bool = False,
        q_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.q_dim = q_dim if q_dim is not None else dim
        self.kv_dim = kv_dim
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / float(rank)
        self.use_k_lora = use_k_lora

        self.q_a = nn.Linear(dim, rank, bias=False)
        self.q_b = nn.Linear(rank, self.q_dim, bias=False)
        self.v_a = nn.Linear(dim, rank, bias=False)
        self.v_b = nn.Linear(rank, kv_dim, bias=False)

        if use_k_lora:
            self.k_a = nn.Linear(dim, rank, bias=False)
            self.k_b = nn.Linear(rank, kv_dim, bias=False)
        else:
            self.k_a = None
            self.k_b = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.q_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.q_b.weight)
        nn.init.kaiming_uniform_(self.v_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.v_b.weight)
        if self.k_a is not None and self.k_b is not None:
            nn.init.kaiming_uniform_(self.k_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.k_b.weight)

    def forward(self, x: torch.Tensor):
        if (
            self.q_a.weight.numel() == 0
            or self.q_b.weight.numel() == 0
            or self.v_a.weight.numel() == 0
            or self.v_b.weight.numel() == 0
        ):
            zq = torch.zeros((x.shape[0], x.shape[1], self.q_dim), dtype=x.dtype, device=x.device)
            zkv = torch.zeros((x.shape[0], x.shape[1], self.kv_dim), dtype=x.dtype, device=x.device)
            if self.use_k_lora and self.k_a is not None:
                return zq, zkv, zkv
            return zq, zkv

        orig_dtype = x.dtype
        if (
            type(self.q_a) is nn.Linear
            and type(self.v_a) is nn.Linear
            and (not self.use_k_lora or type(self.k_a) is nn.Linear)
        ):
            w_dtype = self.q_a.weight.dtype
            x_in = x if (torch.is_autocast_enabled() or x.dtype == w_dtype) else x.to(dtype=w_dtype)
            if self.use_k_lora and self.k_a is not None and self.k_b is not None and self.k_a.weight.numel() > 0:
                w_down = torch.cat([self.q_a.weight, self.k_a.weight, self.v_a.weight], dim=0)
                q_low, k_low, v_low = F.linear(x_in, w_down).split(self.rank, dim=-1)
                delta_q = self.q_b(q_low).to(dtype=orig_dtype)
                delta_k = self.k_b(k_low).to(dtype=orig_dtype)
                delta_v = self.v_b(v_low).to(dtype=orig_dtype)
                if self.scaling != 1.0:
                    delta_q = delta_q * self.scaling
                    delta_k = delta_k * self.scaling
                    delta_v = delta_v * self.scaling
                return delta_q, delta_k, delta_v
            else:
                w_down = torch.cat([self.q_a.weight, self.v_a.weight], dim=0)
                q_low, v_low = F.linear(x_in, w_down).split(self.rank, dim=-1)
                delta_q = self.q_b(q_low).to(dtype=orig_dtype)
                delta_v = self.v_b(v_low).to(dtype=orig_dtype)
                if self.scaling != 1.0:
                    delta_q = delta_q * self.scaling
                    delta_v = delta_v * self.scaling
                return delta_q, delta_v

        delta_q = self.q_b(self.q_a(x)).to(dtype=orig_dtype)
        delta_v = self.v_b(self.v_a(x)).to(dtype=orig_dtype)
        if self.scaling != 1.0:
            delta_q = delta_q * self.scaling
            delta_v = delta_v * self.scaling

        if self.use_k_lora and self.k_a is not None and self.k_b is not None and self.k_a.weight.numel() > 0:
            delta_k = self.k_b(self.k_a(x)).to(dtype=orig_dtype)
            if self.scaling != 1.0:
                delta_k = delta_k * self.scaling
            return delta_q, delta_k, delta_v

        return delta_q, delta_v


_COORD_CACHE = OrderedDict()
_COORD_CACHE_MAX = 64


def get_phase_expert_block_mask(
    depth: int,
    phase_expert_layers: Optional[Tuple[int, ...]] = None,
    num_phases: int = 3,
) -> List[List[bool]]:
    if phase_expert_layers is None:
        return [[True] * num_phases for _ in range(depth)]

    center_outward_order: List[int] = []
    if depth % 2 == 1:
        mid = depth // 2
        center_outward_order.append(mid)
        for offset in range(1, mid + 1):
            center_outward_order.append(mid - offset)
            center_outward_order.append(mid + offset)
    else:
        half = depth // 2
        for offset in range(half):
            center_outward_order.append(half - 1 - offset)
            center_outward_order.append(half + offset)

    mask_per_phase: List[List[bool]] = [[False] * num_phases for _ in range(depth)]

    for p in range(num_phases):
        target_k = int(phase_expert_layers[min(p, len(phase_expert_layers) - 1)])
        if target_k >= depth:
            for b_i in range(depth):
                mask_per_phase[b_i][p] = True
            continue
        if target_k <= 0:
            continue

        for b_i in center_outward_order[:target_k]:
            mask_per_phase[b_i][p] = True

    return mask_per_phase

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        head_dim: Optional[int] = None,
        rope_base: float = 256.0,
        use_value_residual: bool = False,
        use_head_gate: bool = True,
    ):
        super().__init__()

        assert heads % kv_heads == 0, "heads must be divisible by kv_heads"

        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim if head_dim is not None else (dim // heads)
        self.q_dim = self.heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.rope_base = rope_base
        self.use_value_residual = use_value_residual
        self.use_head_gate = use_head_gate

        assert self.head_dim % 2 == 0, f"head_dim must be even for 2D RoPE, got {self.head_dim}"

        self.q_proj = nn.Linear(dim, self.q_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        if use_head_gate:
            self.head_gate = nn.Linear(dim, heads, bias=False)
            nn.init.zeros_(self.head_gate.weight)
        else:
            self.head_gate = None

        self.out_proj = nn.Linear(self.q_dim, dim, bias=False)
        self.track_attention: bool = False
        self.last_attn_weights: Optional[torch.Tensor] = None

        if use_value_residual:
            self.v_res_lambda = nn.Parameter(torch.zeros(1, kv_heads, 1, 1))
        else:
            self.v_res_lambda = None

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
        delta_q: Optional[torch.Tensor] = None,
        delta_v: Optional[torch.Tensor] = None,
        delta_k: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
        return_v: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, _ = x.shape

        if (
            type(self.q_proj) is nn.Linear
            and type(self.k_proj) is nn.Linear
            and type(self.v_proj) is nn.Linear
        ):
            w_qkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
            q, k, v = F.linear(x, w_qkv).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        if delta_q is not None:
            q = q + delta_q.to(dtype=q.dtype)
        q = q.view(B, N, self.heads, self.head_dim)

        if delta_k is not None:
            k = k + delta_k.to(dtype=k.dtype)
        k = k.view(B, N, self.kv_heads, self.head_dim)

        if delta_v is not None:
            v = v + delta_v.to(dtype=v.dtype)
        v = v.view(B, N, self.kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        v_raw = v

        if v_first is not None and self.v_res_lambda is not None:
            alpha = 0.5 * torch.tanh(self.v_res_lambda.to(dtype=v.dtype))
            v_f = v_first.to(dtype=v.dtype)
            v_rms = v.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            vf_rms = v_f.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            v_f_scaled = v_f * (v_rms / vf_rms).clamp(0.5, 2.0).to(dtype=v.dtype)
            norm_scale = torch.rsqrt((1.0 - alpha).square() + alpha.square() + 1e-6)
            v = ((1.0 - alpha) * v + alpha * v_f_scaled) * norm_scale

        spatial_len = grid_h * grid_w
        if N > spatial_len:
            q_sp, q_reg = q[:, :, :spatial_len], q[:, :, spatial_len:]
            k_sp, k_reg = k[:, :, :spatial_len], k[:, :, spatial_len:]
            q_sp, k_sp = apply_2d_rope(q_sp, k_sp, grid_h, grid_w, base=self.rope_base)
            q = torch.cat([q_sp, q_reg], dim=2)
            k = torch.cat([k_sp, k_reg], dim=2)
        else:
            q, k = apply_2d_rope(q, k, grid_h, grid_w, base=self.rope_base)

        if self.track_attention:
            if self.kv_heads != self.heads:
                rep = self.heads // self.kv_heads
                k_track = k.repeat_interleave(rep, dim=1)
                v_track = v.repeat_interleave(rep, dim=1)
            else:
                k_track, v_track = k, v
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q.float(), k_track.float().transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            self.last_attn_weights = attn_weights.detach()
            out = torch.matmul(attn_weights.to(v.dtype), v_track)
        else:
            out = sdpa_gqa(q, k, v)

        if self.head_gate is not None:
            g = 2.0 * torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
            out = out * g.to(out.dtype)

        out = out.transpose(1, 2).contiguous().view(B, N, self.q_dim)
        out = self.out_proj(out)

        if return_v:
            return out, v_raw
        return out


class CrossAttnLoRA(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int = 32,
        alpha: Optional[float] = None,
        kv_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.kv_dim = kv_dim if kv_dim is not None else dim
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / float(rank)

        self.k_a = nn.Linear(self.kv_dim, rank, bias=False)
        self.k_b = nn.Linear(rank, self.kv_dim, bias=False)
        self.v_a = nn.Linear(self.kv_dim, rank, bias=False)
        self.v_b = nn.Linear(rank, self.kv_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.k_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.k_b.weight)
        nn.init.kaiming_uniform_(self.v_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.v_b.weight)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self.k_a.weight.numel() == 0
            or self.k_b.weight.numel() == 0
            or self.v_a.weight.numel() == 0
            or self.v_b.weight.numel() == 0
        ):
            return k, v

        B, H, M, D = k.shape
        target_dtype = k.dtype
        in_dim = H * D

        k_flat = k.transpose(1, 2).reshape(B, M, in_dim)
        v_flat = v.transpose(1, 2).reshape(B, M, in_dim)

        if not torch.is_autocast_enabled():
            if isinstance(self.k_a, nn.Linear) and self.k_a.weight.is_floating_point() and not str(self.k_a.weight.dtype).startswith("torch.float8"):
                if self.k_a.weight.dtype != k_flat.dtype:
                    k_flat = k_flat.to(self.k_a.weight.dtype)
            if isinstance(self.v_a, nn.Linear) and self.v_a.weight.is_floating_point() and not str(self.v_a.weight.dtype).startswith("torch.float8"):
                if self.v_a.weight.dtype != v_flat.dtype:
                    v_flat = v_flat.to(self.v_a.weight.dtype)

        delta_k = self.k_b(self.k_a(k_flat))
        delta_v = self.v_b(self.v_a(v_flat))

        if self.scaling != 1.0:
            delta_k = delta_k * self.scaling
            delta_v = delta_v * self.scaling

        delta_k = delta_k.view(B, M, H, D).transpose(1, 2).to(target_dtype)
        delta_v = delta_v.view(B, M, H, D).transpose(1, 2).to(target_dtype)

        return k + delta_k, v + delta_v


class CrossAttentionLite(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_cross_lora: bool = False,
        cross_lora_rank: int = 64,
        learnable_head_scale: bool = False,
        use_sink: bool = False,
        num_sinks: int = 8,
        use_head_gate: bool = True,
        rope_base: float = 256.0,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim if head_dim is not None else (dim // heads)
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.q_dim = self.heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.use_sink = use_sink
        self.num_sinks = num_sinks if use_sink else 0
        self.rope_base = rope_base

        self.q = nn.Linear(dim, self.q_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(self.q_dim, dim, bias=False)
        self.track_attention: bool = False
        self.last_attn_weights: Optional[torch.Tensor] = None

        if use_head_gate:
            self.head_gate = nn.Linear(dim, self.heads, bias=False)
            nn.init.zeros_(self.head_gate.weight)
        else:
            self.head_gate = None

        self.use_cross_lora = use_cross_lora
        if use_cross_lora and cross_lora_rank > 0:
            self.cross_lora = CrossAttnLoRA(self.q_dim, rank=cross_lora_rank, kv_dim=self.kv_dim)
        else:
            self.cross_lora = None

        self.learnable_head_scale = learnable_head_scale
        if learnable_head_scale:
            self.head_scale = nn.Parameter(torch.ones(1, heads, 1, 1))
        else:
            self.head_scale = None

        if use_sink and self.num_sinks > 0:
            self.sink_k = nn.Parameter(torch.randn(1, self.kv_heads, self.num_sinks, self.head_dim) * 0.5)
            self.sink_v = nn.Parameter(torch.zeros(1, self.kv_heads, self.num_sinks, self.head_dim))
        else:
            self.sink_k = None
            self.sink_v = None

    def append_sinks(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_sink or self.sink_k is None or self.sink_v is None:
            return k, v
        B = k.shape[0]
        s_k = F.rms_norm(self.sink_k, (self.head_dim,), eps=1e-6).to(dtype=k.dtype, device=k.device)
        s_k = s_k.expand(B, -1, -1, -1)
        s_v = self.sink_v.expand(B, -1, -1, -1).to(dtype=v.dtype, device=v.device)
        return torch.cat([k, s_k], dim=2), torch.cat([v, s_v], dim=2)

    def _apply_query_rope(self, q: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        spatial_len = grid_h * grid_w
        if q.shape[2] < spatial_len or grid_h <= 0 or grid_w <= 0:
            return q
        cos, sin = _get_rope_2d(grid_h, grid_w, q.shape[-1], q.device, base=self.rope_base)
        cos = cos.to(q.dtype).unsqueeze(0).unsqueeze(0)
        sin = sin.to(q.dtype).unsqueeze(0).unsqueeze(0)
        if q.shape[2] > spatial_len:
            q_sp, q_reg = q[:, :, :spatial_len], q[:, :, spatial_len:]
            q1, q2 = q_sp.chunk(2, dim=-1)
            q_sp = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
            return torch.cat([q_sp, q_reg], dim=2)
        q1, q2 = q.chunk(2, dim=-1)
        return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)

    def _forward_core(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        sinks_already_attached: bool = False,
        grid_h: int = 16,
        grid_w: int = 16,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.q(x).view(B, N, self.heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        inv_temp = 1.0 / max(float(temperature), 1e-4)
        if self.head_scale is not None:
            q = q * (self.head_scale.to(dtype=q.dtype) * inv_temp)
        elif inv_temp != 1.0:
            q = q * inv_temp

        has_sinks = bool(self.use_sink and self.sink_k is not None and self.sink_v is not None)
        if has_sinks and not sinks_already_attached:
            k, v = self.append_sinks(k, v)

        attn_mask = None
        if mask is not None:
            if mask.dim() == 4 and mask.dtype == torch.bool and mask.shape[-1] == k.shape[2]:
                attn_mask = mask
            else:
                bool_mask = mask.view(B, 1, 1, -1).bool()
                if has_sinks and bool_mask.shape[-1] < k.shape[2]:
                    n_sinks = k.shape[2] - bool_mask.shape[-1]
                    sink_pad = torch.ones((B, 1, 1, n_sinks), dtype=torch.bool, device=mask.device)
                    bool_mask = torch.cat([bool_mask, sink_pad], dim=-1)
                has_valid = bool_mask.any(dim=-1, keepdim=True)
                attn_mask = torch.where(has_valid, bool_mask, torch.ones_like(bool_mask))

        if k.dtype != q.dtype:
            k = k.to(q.dtype)
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        if self.track_attention:
            if k.shape[1] != q.shape[1]:
                rep = q.shape[1] // k.shape[1]
                k_track = k.repeat_interleave(rep, dim=1)
                v_track = v.repeat_interleave(rep, dim=1)
            else:
                k_track, v_track = k, v
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q.float(), k_track.float().transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            self.last_attn_weights = attn_weights.detach()
            out = torch.matmul(attn_weights.to(v.dtype), v_track)
        else:
            out = sdpa_gqa(q, k, v, attn_mask=attn_mask, scale=None)

        if self.head_gate is not None:
            g = 2.0 * torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
            out = out * g.to(out.dtype)

        out = out.transpose(1, 2).contiguous().view(B, N, self.q_dim)
        out = self.proj(out)

        if mask is not None and not has_sinks:
            valid = mask.view(B, -1).bool().any(dim=1).view(B, 1, 1).to(out.dtype)
            out = out * valid

        return out

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        kv_already_adapted: bool = False,
        k_norm_fn: Optional[nn.Module] = None,
        grid_h: int = 16,
        grid_w: int = 16,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        if not kv_already_adapted and k is not None and v is not None:
            if self.cross_lora is not None:
                k, v = self.cross_lora(k, v)
            if k_norm_fn is not None:
                k = k_norm_fn(k)

        if k is not None and B % 2 == 0 and k.shape[0] == B // 2:
            half = B // 2
            mask_cond = mask[half:] if (mask is not None and mask.shape[0] == B) else mask
            out_cond = self._forward_core(
                x[half:],
                k,
                v,
                mask=mask_cond,
                temperature=temperature,
                sinks_already_attached=kv_already_adapted,
                grid_h=grid_h,
                grid_w=grid_w,
            )
            if self.use_sink and self.sink_k is not None and self.sink_v is not None:
                s_k = F.rms_norm(self.sink_k.float(), (self.head_dim,), eps=1e-6).to(dtype=x.dtype, device=x.device).expand(half, -1, -1, -1)
                s_v = self.sink_v.to(dtype=x.dtype, device=x.device).expand(half, -1, -1, -1)
                out_uncond = self._forward_core(
                    x[:half],
                    s_k,
                    s_v,
                    mask=None,
                    temperature=temperature,
                    sinks_already_attached=True,
                    grid_h=grid_h,
                    grid_w=grid_w,
                )
            else:
                out_uncond = torch.zeros_like(out_cond)
            return torch.cat([out_uncond, out_cond], dim=0)

        return self._forward_core(
            x,
            k,
            v,
            mask=mask,
            temperature=temperature,
            sinks_already_attached=kv_already_adapted,
            grid_h=grid_h,
            grid_w=grid_w,
        )


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.w12 = None

    def _compute_gate_up(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "w12", None) is not None:
            return self.w12(x).chunk(2, dim=-1)
        if type(self.w1) is nn.Linear and type(self.w2) is nn.Linear:
            w12_weight = torch.cat([self.w1.weight, self.w2.weight], dim=0)
            return F.linear(x, w12_weight).chunk(2, dim=-1)
        return self.w1(x), self.w2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self._compute_gate_up(x)
        return self.w3(F.silu(x1) * x2)

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw_dilated = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.dw.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.dw_dilated.weight, mean=0.0, std=0.008)

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        if not torch.is_autocast_enabled() and self.dw.weight.dtype != x_2d.dtype:
            x_2d = x_2d.to(self.dw.weight.dtype)
        return 0.45 * self.dw(x_2d) + 0.25 * self.dw_dilated(x_2d)

class SpatialRefinementHead(nn.Module):
    def __init__(self, dim: int, out_channels: int, mid_channels: int = 256):
        super().__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.proj_base = nn.Linear(dim, out_channels, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(dim, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.proj_base.weight)
        nn.init.kaiming_uniform_(self.refine[0].weight, a=math.sqrt(5))
        nn.init.normal_(self.refine[2].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.refine[4].weight)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        B = x.shape[0]
        if not torch.is_autocast_enabled() and hasattr(self.proj_base, "weight"):
            w_dtype = self.proj_base.weight.dtype
            if w_dtype.is_floating_point and "float8" not in str(w_dtype) and x.dtype != w_dtype:
                x = x.to(w_dtype)

        base = self.proj_base(x)
        base_2d = base.view(B, grid_h, grid_w, self.out_channels).permute(0, 3, 1, 2).contiguous()

        x_2d = x.view(B, grid_h, grid_w, self.dim).permute(0, 3, 1, 2)
        if not torch.is_autocast_enabled() and self.refine[0].weight.dtype != x_2d.dtype:
            x_2d = x_2d.to(self.refine[0].weight.dtype)

        delta = self.refine(x_2d).to(base_2d.dtype)
        return base_2d + delta


class SpatialMixSwiGLU(SwiGLU):
    def __init__(self, dim: int, hidden_dim: int, use_dw_conv: bool = True):
        super().__init__(dim, hidden_dim)
        self.dw_conv = MultiScaleDWConv(hidden_dim) if use_dw_conv else None

    def forward(self, x: torch.Tensor, grid_h: int = 16, grid_w: int = 16) -> torch.Tensor:
        x1, x2 = self._compute_gate_up(x)
        hidden = F.silu(x1) * x2

        spatial_len = grid_h * grid_w
        if self.dw_conv is not None and hidden.shape[1] >= spatial_len:
            B = hidden.shape[0]
            if hidden.shape[1] == spatial_len:
                h_2d = hidden.view(B, grid_h, grid_w, -1).permute(0, 3, 1, 2)
                conv_out = self.dw_conv(h_2d).to(hidden.dtype).permute(0, 2, 3, 1).reshape(B, spatial_len, -1)
                hidden = hidden + conv_out
            else:
                num_reg = hidden.shape[1] - spatial_len
                h_sp = hidden[:, :spatial_len].contiguous()
                h_2d = h_sp.view(B, grid_h, grid_w, -1).permute(0, 3, 1, 2)
                conv_out = self.dw_conv(h_2d).to(hidden.dtype).permute(0, 2, 3, 1).reshape(B, spatial_len, -1)
                hidden = hidden + F.pad(conv_out, (0, 0, 0, num_reg))

        return self.w3(hidden)


class PhaseExpert(SpatialMixSwiGLU):
    def __init__(self, dim: int, hidden_dim: int, use_dw_conv: bool = True):
        super().__init__(dim, hidden_dim, use_dw_conv=use_dw_conv)


class EdgeFlowPhaseBlockV2(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        use_cross: bool,
        ffn_hidden: int,
        head_dim: Optional[int] = None,
        num_phases: int = 0,
        expert_hidden_list: Optional[List[int]] = None,
        active_experts_mask: Optional[List[bool]] = None,
        gate_init: float = 1e-2,
        ada_rank: Optional[int] = None,
        use_phase_lora: bool = True,
        phase_lora_rank: Optional[int] = 64,
        use_dw_conv: bool = True,
        use_cross_lora: bool = True,
        cross_lora_rank: Optional[int] = 64,
        learnable_head_scale: bool = True,
        rope_base: float = 256.0,
        use_value_residual: bool = True,
        use_cross_sink: bool = True,
        num_sinks: int = 8,
        use_phase_k_lora: bool = True,
        use_head_gate: bool = True,
        use_hf_booster: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.use_cross = use_cross
        self.use_phase_lora = use_phase_lora
        self.phase_lora_rank = phase_lora_rank
        self.use_dw_conv = use_dw_conv
        self.use_cross_lora = use_cross_lora
        self.cross_lora_rank = cross_lora_rank
        self.learnable_head_scale = learnable_head_scale
        self.rope_base = rope_base
        self.use_value_residual = use_value_residual
        self.use_cross_sink = use_cross_sink
        self.use_phase_k_lora = use_phase_k_lora
        self.head_dim = head_dim if head_dim is not None else (dim // heads)
        self.q_dim = heads * self.head_dim
        self.kv_dim = kv_heads * self.head_dim

        self.register_buffer(
            "gate_init",
            torch.tensor(float(gate_init), dtype=torch.float32),
            persistent=False,
        )

        if ada_rank is not None and ada_rank > 0:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, ada_rank, bias=False),
                nn.SiLU(),
                nn.Linear(ada_rank, 8 * dim, bias=False),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 8 * dim, bias=False),
            )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)

        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(
            dim,
            heads,
            kv_heads,
            head_dim=self.head_dim,
            rope_base=rope_base,
            use_value_residual=use_value_residual,
            use_head_gate=use_head_gate,
        )

        self.dw_conv = None
        self.hf_gain = None

        if use_cross:
            self.norm_cross = RMSNorm(dim)
            self.cross = CrossAttentionLite(
                dim,
                heads,
                kv_heads=kv_heads,
                head_dim=self.head_dim,
                use_cross_lora=use_cross_lora,
                cross_lora_rank=cross_lora_rank or 64,
                learnable_head_scale=learnable_head_scale,
                use_sink=use_cross_sink,
                num_sinks=num_sinks,
                use_head_gate=use_head_gate,
            )

        self.norm2 = RMSNorm(dim)
        self.ff = SpatialMixSwiGLU(dim, ffn_hidden, use_dw_conv=use_dw_conv)

        self.phase_experts = None
        self.has_expert: List[bool] = []
        if num_phases > 0 and expert_hidden_list is not None:
            assert len(expert_hidden_list) == num_phases
            if active_experts_mask is None:
                active_experts_mask = [bool(h is not None and h > 0) for h in expert_hidden_list]
            self.has_expert = list(active_experts_mask)
            self.phase_experts = nn.ModuleList([
                PhaseExpert(dim, hidden, use_dw_conv=use_dw_conv) if (self.has_expert[p] and hidden > 0) else nn.Identity()
                for p, hidden in enumerate(expert_hidden_list)
            ])

        if use_phase_lora and phase_lora_rank is not None and phase_lora_rank > 0 and num_phases > 0:
            self.phase_attn_lora = nn.ModuleList([
                PhaseAttnLoRA(
                    dim=dim,
                    q_dim=self.q_dim,
                    kv_dim=self.kv_dim,
                    rank=phase_lora_rank,
                    use_k_lora=use_phase_k_lora,
                )
                for _ in range(num_phases)
            ])
        else:
            self.phase_attn_lora = None

        self.sparse_experts = os.environ.get("SPARSE_EXPERTS", "1") == "1"

    def _is_expert_active(self, p: int) -> bool:
        if self.phase_experts is None or not (0 <= p < len(self.phase_experts)):
            return False
        if self.has_expert and not self.has_expert[p]:
            return False
        exp = self.phase_experts[p]
        if isinstance(exp, nn.Identity):
            return False
        if hasattr(exp, "w1") and hasattr(exp.w1, "weight"):
            return exp.w1.weight.numel() > 0
        return True

    @torch._dynamo.disable
    def _apply_sparse_experts(
        self,
        h: torch.Tensor,
        phase_idx: torch.Tensor,
        grid_h: int = 16,
        grid_w: int = 16,
    ) -> torch.Tensor:
        out = torch.zeros_like(h)
        for p, expert in enumerate(self.phase_experts):
            if not self._is_expert_active(p):
                continue
            idx = torch.nonzero(phase_idx == p, as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            h_selected = h.index_select(0, idx)
            expert_out = expert(h_selected, grid_h=grid_h, grid_w=grid_w).to(dtype=out.dtype)
            out = torch.index_put(out, (idx,), expert_out)
        return out

    def _apply_phase_experts(
        self,
        h: torch.Tensor,
        phase_idx: Union[int, torch.Tensor],
        grid_h: int = 16,
        grid_w: int = 16,
    ) -> Optional[torch.Tensor]:
        if self.phase_experts is None or phase_idx is None:
            return None

        if isinstance(phase_idx, int):
            if self._is_expert_active(phase_idx):
                return self.phase_experts[phase_idx](h, grid_h=grid_h, grid_w=grid_w)
            return None

        if isinstance(phase_idx, torch.Tensor):
            if phase_idx.device.type == "meta":
                return self.phase_experts[0](h, grid_h=grid_h, grid_w=grid_w) if self._is_expert_active(0) else None
            if phase_idx.numel() == 1:
                p = int(phase_idx.item())
                return self.phase_experts[p](h, grid_h=grid_h, grid_w=grid_w) if self._is_expert_active(p) else None
            elif phase_idx.device.type == "cpu" and phase_idx.numel() > 0 and (phase_idx == phase_idx[0]).all():
                p = int(phase_idx[0].item())
                return self.phase_experts[p](h, grid_h=grid_h, grid_w=grid_w) if self._is_expert_active(p) else None

        if torch.compiler.is_compiling() or not getattr(self, "sparse_experts", False):
            out = None
            for p, expert in enumerate(self.phase_experts):
                if not self._is_expert_active(p):
                    continue
                mask_p = (phase_idx == p).to(h.dtype).view(-1, 1, 1)
                term = expert(h, grid_h=grid_h, grid_w=grid_w) * mask_p
                out = term if out is None else (out + term)
            return out

        return self._apply_sparse_experts(h, phase_idx, grid_h=grid_h, grid_w=grid_w)

    @torch._dynamo.disable
    def _apply_sparse_attn_lora(
        self,
        h: torch.Tensor,
        phase_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        delta_q = torch.zeros((h.shape[0], h.shape[1], self.q_dim), dtype=h.dtype, device=h.device)
        delta_v = torch.zeros((h.shape[0], h.shape[1], self.kv_dim), dtype=h.dtype, device=h.device)
        has_k = getattr(self.phase_attn_lora[0], "use_k_lora", False)
        delta_k = torch.zeros((h.shape[0], h.shape[1], self.kv_dim), dtype=h.dtype, device=h.device) if has_k else None
        for p, lora in enumerate(self.phase_attn_lora):
            if hasattr(lora, "q_a") and hasattr(lora.q_a, "weight") and lora.q_a.weight.numel() == 0:
                continue
            idx = torch.nonzero(phase_idx == p, as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            h_sel = h.index_select(0, idx)
            res = lora(h_sel)
            if len(res) == 3:
                dq, dk, dv = res
                if delta_k is not None and dk is not None:
                    delta_k = torch.index_put(delta_k, (idx,), dk.to(dtype=delta_k.dtype))
            else:
                dq, dv = res
            delta_q = torch.index_put(delta_q, (idx,), dq.to(dtype=delta_q.dtype))
            delta_v = torch.index_put(delta_v, (idx,), dv.to(dtype=delta_v.dtype))
        return delta_q, delta_k, delta_v

    def _apply_phase_attn_lora(
        self,
        h: torch.Tensor,
        phase_idx: Optional[Union[int, torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.phase_attn_lora is None or phase_idx is None:
            return None, None, None

        def _unpack(res):
            if res is None:
                return None, None, None
            if len(res) == 3:
                return res[0], res[1], res[2]
            return res[0], None, res[1]

        if isinstance(phase_idx, int):
            p = phase_idx
            if 0 <= p < len(self.phase_attn_lora):
                lora = self.phase_attn_lora[p]
                if hasattr(lora, "q_a") and hasattr(lora.q_a, "weight") and lora.q_a.weight.numel() > 0:
                    return _unpack(lora(h))
                elif hasattr(lora, "weight") and lora.weight.numel() > 0:
                    return _unpack(lora(h))
            return None, None, None

        if isinstance(phase_idx, torch.Tensor):
            if phase_idx.device.type == "meta":
                p = 0
                if 0 <= p < len(self.phase_attn_lora):
                    return _unpack(self.phase_attn_lora[p](h))
                return None, None, None
            if phase_idx.numel() == 1:
                p = int(phase_idx.item())
                if 0 <= p < len(self.phase_attn_lora):
                    lora = self.phase_attn_lora[p]
                    if hasattr(lora, "q_a") and hasattr(lora.q_a, "weight") and lora.q_a.weight.numel() > 0:
                        return _unpack(lora(h))
                return None, None, None
            elif phase_idx.device.type == "cpu" and phase_idx.numel() > 0 and (phase_idx == phase_idx[0]).all():
                p = int(phase_idx[0].item())
                if 0 <= p < len(self.phase_attn_lora):
                    lora = self.phase_attn_lora[p]
                    if hasattr(lora, "q_a") and hasattr(lora.q_a, "weight") and lora.q_a.weight.numel() > 0:
                        return _unpack(lora(h))
                return None, None, None

        if torch.compiler.is_compiling() or not getattr(self, "sparse_experts", False):
            delta_q = torch.zeros((h.shape[0], h.shape[1], self.q_dim), dtype=h.dtype, device=h.device)
            delta_v = torch.zeros((h.shape[0], h.shape[1], self.kv_dim), dtype=h.dtype, device=h.device)
            has_k = getattr(self.phase_attn_lora[0], "use_k_lora", False)
            delta_k = torch.zeros((h.shape[0], h.shape[1], self.kv_dim), dtype=h.dtype, device=h.device) if has_k else None
            for p, lora in enumerate(self.phase_attn_lora):
                if hasattr(lora, "q_a") and hasattr(lora.q_a, "weight") and lora.q_a.weight.numel() == 0:
                    continue
                mask_p = (phase_idx == p).to(h.dtype).view(-1, 1, 1)
                dq, dk, dv = _unpack(lora(h))
                delta_q = delta_q + dq * mask_p
                delta_v = delta_v + dv * mask_p
                if delta_k is not None and dk is not None:
                    delta_k = delta_k + dk * mask_p
            return delta_q, delta_k, delta_v

        return self._apply_sparse_attn_lora(h, phase_idx)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        scale1: Optional[torch.Tensor] = None,
        shift1: Optional[torch.Tensor] = None,
        gate1: Optional[torch.Tensor] = None,
        scale2: Optional[torch.Tensor] = None,
        shift2: Optional[torch.Tensor] = None,
        gate2: Optional[torch.Tensor] = None,
        gate_exp: Optional[torch.Tensor] = None,
        gate_cross: Optional[torch.Tensor] = None,
        grid_h: int = 16,
        grid_w: int = 16,
        phase_idx: Optional[Union[int, torch.Tensor]] = None,
        text_k: Optional[torch.Tensor] = None,
        text_v: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        v_first: Optional[torch.Tensor] = None,
        kv_already_adapted: bool = False,
        k_norm_fn: Optional[nn.Module] = None,
        return_v: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if c is not None:
            mod = self.adaLN_modulation(c).view(-1, 8, self.dim)
            scale1, shift1, gate1, scale2, shift2, gate2, gate_exp, gate_cross = mod.unbind(dim=1)
            gi = self.gate_init.to(dtype=gate1.dtype)
            gate1 = gate1 + gi
            gate2 = gate2 + gi
            gate_cross = gate_cross + gi
            gate_exp = gate_exp + gi

        h = self.norm1(x)
        h = h * (1.0 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        dq, dk, dv = self._apply_phase_attn_lora(h, phase_idx)
        attn_res = self.attn(
            h,
            grid_h,
            grid_w,
            delta_q=dq,
            delta_v=dv,
            delta_k=dk,
            v_first=v_first,
            return_v=return_v,
        )
        if return_v:
            attn_out, v_raw = attn_res
        else:
            attn_out, v_raw = attn_res, None

        x = x + attn_out * torch.tanh(gate1).unsqueeze(1)

        if self.use_cross and text_k is not None:
            x = x + self.cross(
                self.norm_cross(x),
                k=text_k,
                v=text_v,
                mask=mask,
                temperature=temperature,
                kv_already_adapted=kv_already_adapted,
                k_norm_fn=k_norm_fn,
                grid_h=grid_h,
                grid_w=grid_w,
            ) * torch.tanh(gate_cross).unsqueeze(1)

        h = self.norm2(x)
        h = h * (1.0 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)

        x = x + self.ff(h, grid_h=grid_h, grid_w=grid_w) * torch.tanh(gate2).unsqueeze(1)

        if self.phase_experts is not None and phase_idx is not None:
            expert_out = self._apply_phase_experts(h, phase_idx, grid_h=grid_h, grid_w=grid_w)
            if expert_out is not None:
                x = x + expert_out * torch.tanh(gate_exp).unsqueeze(1)

        if return_v:
            return x, v_raw
        return x
class _VirtualWeightTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        res = super().__torch_function__(func, types, args, kwargs)
        if getattr(func, "__name__", "").endswith("_") and len(args) > 0 and isinstance(args[0], _VirtualWeightTensor):
            args[0]._sync_back()
        return res

    def _sync_back(self):
        if getattr(self, "_in_sync", False) or not hasattr(self, "_blocks") or self._blocks is None:
            return
        self._in_sync = True
        try:
            with torch.no_grad():
                base = self.as_subclass(torch.Tensor)
                depth = len(self._blocks)
                if getattr(self, "_sub_idx", -1) == 0:
                    for b in self._blocks:
                        if hasattr(b, "adaLN_modulation"):
                            if len(b.adaLN_modulation) == 4:
                                b.adaLN_modulation[1].weight.data.copy_(base)
                            elif len(b.adaLN_modulation) == 3:
                                b.adaLN_modulation[0].weight.data.copy_(base)
                            elif base.shape == b.adaLN_modulation[-1].weight.shape:
                                b.adaLN_modulation[-1].weight.data.copy_(base)
                else:
                    slices = base.chunk(depth, dim=0)
                    for b, s in zip(self._blocks, slices):
                        if hasattr(b, "adaLN_modulation"):
                            b.adaLN_modulation[-1].weight.data.copy_(s)
        finally:
            self._in_sync = False

class _VirtualAdaProj:
    def __init__(self, blocks: nn.ModuleList):
        self._blocks = blocks

    def __call__(self, c: torch.Tensor) -> torch.Tensor:
        mods = [b.adaLN_modulation(c) for b in self._blocks]
        return torch.cat(mods, dim=-1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self(c)

    def __getitem__(self, idx: int):
        class _SliceProxy:
            def __init__(self, blocks, sub_idx):
                self._blocks = blocks
                self._sub_idx = sub_idx

            @property
            def weight(self):
                if self._sub_idx == 0:
                    b0 = self._blocks[0]
                    if hasattr(b0, "adaLN_modulation"):
                        if len(b0.adaLN_modulation) == 4:
                            w = b0.adaLN_modulation[1].weight
                        elif len(b0.adaLN_modulation) == 3:
                            w = b0.adaLN_modulation[0].weight
                        else:
                            w = b0.adaLN_modulation[-1].weight
                    else:
                        w = b0.adaLN_modulation[-1].weight
                    vw = w.as_subclass(_VirtualWeightTensor)
                    vw._blocks = self._blocks
                    vw._sub_idx = 0
                    return vw
                else:
                    cat_w = torch.cat([b.adaLN_modulation[-1].weight for b in self._blocks], dim=0)
                    vw = cat_w.as_subclass(_VirtualWeightTensor)
                    vw._blocks = self._blocks
                    vw._sub_idx = self._sub_idx
                    return vw

            @weight.setter
            def weight(self, val):
                if not isinstance(val, torch.Tensor):
                    return
                if self._sub_idx == 0:
                    for b in self._blocks:
                        if hasattr(b, "adaLN_modulation"):
                            target_layer = (
                                b.adaLN_modulation[1]
                                if len(b.adaLN_modulation) == 4
                                else (b.adaLN_modulation[0] if len(b.adaLN_modulation) == 3 else b.adaLN_modulation[-1])
                            )
                            target_layer.weight.data.copy_(
                                val.to(device=target_layer.weight.device, dtype=target_layer.weight.dtype)
                            )
                else:
                    depth = len(self._blocks)
                    slices = val.chunk(depth, dim=0)
                    for b, s in zip(self._blocks, slices):
                        if hasattr(b, "adaLN_modulation"):
                            b.adaLN_modulation[-1].weight.data.copy_(
                                s.to(device=b.adaLN_modulation[-1].weight.device, dtype=b.adaLN_modulation[-1].weight.dtype)
                            )

            @property
            def in_features(self):
                return self.weight.shape[1]

            @property
            def out_features(self):
                return self.weight.shape[0]

            def __getattr__(self, name):
                if self._sub_idx == 0:
                    b0 = self._blocks[0]
                    if hasattr(b0, "adaLN_modulation"):
                        if len(b0.adaLN_modulation) == 4:
                            target = b0.adaLN_modulation[1]
                        elif len(b0.adaLN_modulation) == 3:
                            target = b0.adaLN_modulation[0]
                        else:
                            target = b0.adaLN_modulation[-1]
                    else:
                        target = b0.adaLN_modulation[-1]
                else:
                    target = self._blocks[0].adaLN_modulation[-1]
                return getattr(target, name)

        return _SliceProxy(self._blocks, idx)

    def __repr__(self) -> str:
        return f"_VirtualAdaProj(blocks={len(self._blocks)}, per_block_adaln=True)"


class EdgeFlowPhaseV2(nn.Module):

    def __init__(
        self,
        latent_channels: int = 32,
        patch_size: int = 1,
        dim: int = 1536,
        depth: int = 30,
        heads: int = 16,
        kv_heads: int = 8,
        head_dim: Optional[int] = 128,
        context_dim: int = 2560,
        num_phases: int = 3,
        ffn_hidden: int = 2560,
        ada_rank: Optional[int] = 256,
        cross_every: int = 1,
        expert_hidden_list: Optional[List[int]] = None,
        gate_init: float = 1e-2,
        use_guidance: bool = False,
        quantize_int8: bool = False,
        quantize_fp8: bool = False,
        quantization: Optional[str] = None,
        per_block_adaln: bool = True,
        phase_layers: Tuple[int, ...] = (8, 16, 28),
        phase_expert_layers: Optional[Tuple[int, ...]] = None,
        use_phase_lora: bool = True,
        phase_lora_rank: Optional[int] = 64,
        use_dw_conv: bool = True,
        use_skip_connections: bool = True,
        use_cross_lora: bool = True,
        cross_lora_rank: Optional[int] = 64,
        learnable_head_scale: bool = True,
        use_coord_conv: bool = True,
        rope_base: float = 256.0,
        use_value_residual: bool = True,
        use_cross_sink: bool = True,
        num_sinks: int = 4,
        use_phase_k_lora: bool = True,
        num_registers: int = 8,
        use_head_gate: bool = True,
        use_hf_booster: bool = True,
        use_refinement_head: bool = True,
        v4: bool = False,
        architecture_version: Optional[str] = None,
        attention_version: Optional[str] = None,
        sampler_default: Optional[str] = None,
        text_cache_schema_version: int = 0,
        **kwargs,
    ):
        super().__init__()

        if v4:
            use_dw_conv = True
            use_skip_connections = True
            use_cross_lora = True
            learnable_head_scale = True
            use_coord_conv = True
            use_value_residual = True
            use_cross_sink = True
            use_phase_k_lora = True

        if quantization == "fp8":
            quantize_fp8 = True
        elif quantization == "int8":
            quantize_int8 = True

        self.context_dim = context_dim
        self.use_guidance = use_guidance
        self.quantized_fp8 = quantize_fp8
        self.quantized_int8 = quantize_int8
        self.quantization = "fp8" if quantize_fp8 else ("int8" if quantize_int8 else None)
        self.per_block_adaln = per_block_adaln
        self.phase_layers = phase_layers
        self.phase_expert_layers = phase_expert_layers
        self.use_phase_lora = use_phase_lora
        self.phase_lora_rank = phase_lora_rank
        self.use_dw_conv = use_dw_conv
        self.use_skip_connections = use_skip_connections
        self.use_cross_lora = use_cross_lora
        self.cross_lora_rank = cross_lora_rank
        self.learnable_head_scale = learnable_head_scale
        self.use_coord_conv = use_coord_conv
        self.rope_base = rope_base
        self.use_value_residual = use_value_residual
        self.use_cross_sink = use_cross_sink
        self.num_sinks = num_sinks
        self.use_phase_k_lora = use_phase_k_lora
        self.num_registers = num_registers
        self.use_head_gate = use_head_gate
        self.use_hf_booster = use_hf_booster
        self.use_refinement_head = use_refinement_head
        self.architecture_version = architecture_version
        self.attention_version = attention_version
        self.sampler_default = sampler_default
        self.text_cache_schema_version = text_cache_schema_version
        self._lora_in_experts: Optional[bool] = None

        if expert_hidden_list is None:
            if num_phases == 3:
                expert_hidden_list = [6656, 6656, 6656]
            elif num_phases > 0:
                expert_hidden_list = [ffn_hidden] * num_phases
            else:
                expert_hidden_list = []

        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_phases = num_phases
        self.use_phase = num_phases > 1
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim if head_dim is not None else (dim // heads)
        self.q_dim = self.heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.gradient_checkpointing = False

        patch_dim = latent_channels * patch_size * patch_size
        in_channels = (latent_channels + 3) if self.use_coord_conv else latent_channels

        self.proj_in = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=3 if patch_size == 1 else patch_size,
            stride=patch_size,
            padding=1 if patch_size == 1 else 0,
            bias=False,
        )

        if self.num_registers > 0:
            self.registers = nn.Parameter(torch.randn(1, self.num_registers, dim) * 0.02)
        else:
            self.registers = None

        num_skips = max(0, (depth // 2) - 1) if (self.use_skip_connections and depth > 2) else 0
        if num_skips > 0:
            self.skip_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_skips)])
            self.skip_norms = nn.ModuleList([RMSNorm(dim) for _ in range(num_skips)])
            if num_phases == 3:
                init_skip_gate = torch.tensor([0.25, 0.65, 1.0], dtype=torch.float32).view(3, 1, 1).expand(3, num_skips, dim).clone()
            else:
                init_skip_gate = torch.ones(max(1, num_phases), num_skips, dim)
            self.skip_phase_gate = nn.Parameter(init_skip_gate)
            for proj in self.skip_projs:
                nn.init.normal_(proj.weight, mean=0.0, std=0.01)
        else:
            self.skip_projs = None
            self.skip_norms = None
            self.skip_phase_gate = None

        self.time_embed = TimestepEmbedding(dim)
        self.time_norm = RMSNorm(dim)
        if context_dim != dim:
            self.text_proj = nn.Linear(context_dim, dim, bias=False)
        else:
            self.text_proj = nn.Identity()
        self.text_norm = RMSNorm(dim)

        self.pooler = AttentionPooling(context_dim, heads=min(8, max(1, context_dim // 128)))

        if self.use_phase:
            self.phase_table = nn.Parameter(torch.randn(num_phases, dim) * 0.02)
            self.phase_norm = RMSNorm(dim)
            cond_in_dim = dim * 3
            if num_phases == 3:
                init_logits = torch.tensor([
                    [0.0, 0.5, 1.5],
                    [0.2, 1.5, 0.8],
                    [1.2, 0.8, 1.0],
                ], dtype=torch.float32).unsqueeze(-1).expand(3, 3, context_dim).clone()
            else:
                init_logits = torch.eye(num_phases).unsqueeze(-1).expand(num_phases, num_phases, context_dim) * 1.5
            self.phase_layer_mixer = nn.Parameter(init_logits)
        else:
            self.phase_table = None
            self.phase_layer_mixer = None
            cond_in_dim = dim * 2

        if self.use_guidance:
            self.guidance_embed = TimestepEmbedding(dim)
            self.guidance_norm = RMSNorm(dim)
            cond_in_dim += dim

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

        if not per_block_adaln:
            self._ada_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, ada_rank or 256, bias=False),
                nn.SiLU(),
                nn.Linear(ada_rank or 256, depth * dim * 8, bias=False),
            )
        else:
            self._ada_proj = None

        num_cross_kv = num_phases if self.use_phase else 1
        self.phase_cross_kv = nn.ModuleList([
            nn.Linear(context_dim, self.kv_dim * 2, bias=False) for _ in range(num_cross_kv)
        ])
        self.shared_cross_kv = self.phase_cross_kv[0]
        self.shared_k_norm = RMSNorm(self.head_dim)
        self.shared_v_norm = RMSNorm(self.head_dim)
        self.shared_v_norm.requires_grad_(False)

        self.norm_final = RMSNorm(dim)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=False),
        )
        if self.use_refinement_head:
            self.refinement_head = SpatialRefinementHead(dim, patch_dim, mid_channels=256)
            self.proj_out = self.refinement_head.proj_base
        else:
            self.refinement_head = None
            self.proj_out = nn.Linear(dim, patch_dim, bias=False)

        self.register_buffer(
            "gate_init",
            torch.tensor(float(gate_init), dtype=torch.float32),
            persistent=False,
        )

        self.apply(self._init_weights)

        if self._ada_proj is not None:
            nn.init.zeros_(self._ada_proj[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].weight)
        if self.refinement_head is not None:
            self.refinement_head.reset_parameters()
        else:
            nn.init.zeros_(self.proj_out.weight)
        if self.use_coord_conv:
            nn.init.normal_(self.proj_in.weight.data[:, latent_channels:, :, :], mean=0.0, std=0.01)
        if self.skip_projs is not None:
            for proj in self.skip_projs:
                nn.init.normal_(proj.weight, mean=0.0, std=0.01)

        ce = 1 if cross_every is None else int(cross_every)
        self.phase_expert_layers = phase_expert_layers
        self.num_sinks = num_sinks
        expert_block_mask = get_phase_expert_block_mask(
            depth=depth,
            phase_expert_layers=phase_expert_layers,
            num_phases=num_phases if self.use_phase else 0,
        )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = EdgeFlowPhaseBlockV2(
                dim=dim,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=self.head_dim,
                use_cross=((i + 1) % ce == 0) if ce > 0 else False,
                ffn_hidden=ffn_hidden,
                num_phases=num_phases if self.use_phase else 0,
                expert_hidden_list=expert_hidden_list if self.use_phase else None,
                active_experts_mask=expert_block_mask[i] if self.use_phase else None,
                gate_init=gate_init,
                ada_rank=ada_rank,
                use_phase_lora=use_phase_lora if self.use_phase else False,
                phase_lora_rank=phase_lora_rank,
                use_dw_conv=self.use_dw_conv,
                use_cross_lora=self.use_cross_lora,
                cross_lora_rank=self.cross_lora_rank,
                learnable_head_scale=self.learnable_head_scale,
                rope_base=self.rope_base,
                use_value_residual=self.use_value_residual,
                use_cross_sink=self.use_cross_sink,
                num_sinks=self.num_sinks,
                use_phase_k_lora=self.use_phase_k_lora,
                use_head_gate=self.use_head_gate,
                use_hf_booster=self.use_hf_booster,
            )
            block.apply(self._init_weights)
            if hasattr(block, "adaLN_modulation"):
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
            if block.phase_experts is not None:
                for expert in block.phase_experts:
                    if hasattr(expert, "w3") and hasattr(expert.w3, "weight"):
                        nn.init.normal_(expert.w3.weight, mean=0.0, std=0.015)
                    if getattr(expert, "dw_conv", None) is not None and hasattr(expert.dw_conv, "reset_parameters"):
                        expert.dw_conv.reset_parameters()
            if getattr(block, "phase_attn_lora", None) is not None:
                for lora in block.phase_attn_lora:
                    nn.init.zeros_(lora.q_b.weight)
                    nn.init.zeros_(lora.v_b.weight)
                    if getattr(lora, "k_b", None) is not None:
                        nn.init.zeros_(lora.k_b.weight)
            if getattr(block.attn, "v_res_lambda", None) is not None:
                nn.init.zeros_(block.attn.v_res_lambda)
            if getattr(block.attn, "head_gate", None) is not None:
                nn.init.zeros_(block.attn.head_gate.weight)
            if getattr(block, "use_cross", False) and getattr(block.cross, "head_gate", None) is not None:
                nn.init.zeros_(block.cross.head_gate.weight)
            if getattr(block.ff, "dw_conv", None) is not None and hasattr(block.ff.dw_conv, "reset_parameters"):
                block.ff.dw_conv.reset_parameters()

            if hasattr(block.ff, "fuse_w12") and not (quantize_fp8 or quantize_int8):
                block.ff.fuse_w12()

            if quantize_fp8:
                quantize_module_to_fp8(block)
            elif quantize_int8:
                quantize_module_to_int8(block)

            self.blocks.append(block)

        if len(self.blocks) > 0 and getattr(self.blocks[0].attn, "v_res_lambda", None) is not None:
            self.blocks[0].attn.v_res_lambda.requires_grad_(False)

        self._virtual_ada_proj = _VirtualAdaProj(self.blocks)

        if self.skip_projs is not None:
            if quantize_fp8:
                quantize_module_to_fp8(self.skip_projs)
            elif quantize_int8:
                quantize_module_to_int8(self.skip_projs)

        if quantize_fp8:
            quantize_module_to_fp8(self.cond_mlp)
            if self._ada_proj is not None:
                quantize_module_to_fp8(self._ada_proj)
            quantize_module_to_fp8(self.phase_cross_kv)
            if isinstance(self.text_proj, nn.Linear):
                self.text_proj = FP8Linear.from_float(self.text_proj)
            self.shared_cross_kv = self.phase_cross_kv[0]
            if self.refinement_head is not None:
                self.proj_out = self.refinement_head.proj_base
            elif isinstance(self.proj_out, nn.Linear):
                self.proj_out = FP8Linear.from_float(self.proj_out)
        elif quantize_int8:
            quantize_module_to_int8(self.cond_mlp)
            if self._ada_proj is not None:
                quantize_module_to_int8(self._ada_proj)
            quantize_module_to_int8(self.phase_cross_kv)
            if isinstance(self.text_proj, nn.Linear):
                self.text_proj = Int8Linear.from_float(self.text_proj)
            self.shared_cross_kv = self.phase_cross_kv[0]
            if self.refinement_head is not None:
                self.proj_out = self.refinement_head.proj_base
            elif isinstance(self.proj_out, nn.Linear):
                self.proj_out = Int8Linear.from_float(self.proj_out)

    @property
    def ada_proj(self):
        if self._ada_proj is not None:
            return self._ada_proj
        return self._virtual_ada_proj

    def to_fp8(self, fp8_dtype: Optional[torch.dtype] = None):
        if fp8_dtype is None:
            fp8_dtype = FP8_DTYPE
        quantize_module_to_fp8(self, exclude_types=(TimestepEmbedding,), fp8_dtype=fp8_dtype)
        self.shared_cross_kv = self.phase_cross_kv[0]
        if self.refinement_head is not None:
            self.proj_out = self.refinement_head.proj_base
        self.quantized_fp8 = True
        self.quantized_int8 = False
        self.quantization = "fp8"
        return self

    def to_int8(self):
        quantize_module_to_int8(self, exclude_types=(TimestepEmbedding,))
        self.shared_cross_kv = self.phase_cross_kv[0]
        if self.refinement_head is not None:
            self.proj_out = self.refinement_head.proj_base
        self.quantized_int8 = True
        self.quantized_fp8 = False
        self.quantization = "int8"
        return self

    def to_float(self, dtype: torch.dtype = torch.float32):
        def _convert(mod):
            for name, child in mod.named_children():
                if isinstance(child, (Int8Linear, FP8Linear)):
                    setattr(mod, name, child.to_float(dtype=dtype))
                else:
                    _convert(child)
        _convert(self)
        self.shared_cross_kv = self.phase_cross_kv[0]
        if self.refinement_head is not None:
            self.proj_out = self.refinement_head.proj_base
        self.quantized_int8 = False
        self.quantized_fp8 = False
        self.quantization = None
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        has_scales = any(k.endswith(".scale") for k in state_dict.keys())
        if has_scales:
            is_fp8 = False
            for k, v in state_dict.items():
                if k.endswith(".weight"):
                    scale_k = k[:-7] + ".scale"
                    if scale_k in state_dict:
                        if "float8" in str(v.dtype):
                            is_fp8 = True
                            break
            if not is_fp8 and any("float8" in str(v.dtype) for v in state_dict.values()):
                is_fp8 = True
            if not is_fp8 and state_dict.get("config", {}).get("quantization") == "fp8":
                is_fp8 = True

            if is_fp8:
                if not getattr(self, "quantized_fp8", False):
                    self.to_fp8()
            else:
                if not getattr(self, "quantized_int8", False):
                    self.to_int8()
        else:
            if getattr(self, "quantized_fp8", False) or getattr(self, "quantized_int8", False):
                self.to_float()

        if self.per_block_adaln:
            has_old_ada = any(k.startswith("ada_proj.") or k == "ada_proj.weight" for k in state_dict.keys())
            has_new_ada = any(".adaLN_modulation." in k for k in state_dict.keys())
            if has_old_ada and not has_new_ada:
                w0 = state_dict.get("ada_proj.0.weight")
                w2 = state_dict.get("ada_proj.2.weight")
                w1 = state_dict.get("ada_proj.1.weight", state_dict.get("ada_proj.weight"))
                if w1 is not None:
                    state_dict = dict(state_dict)
                    for i, block in enumerate(self.blocks):
                        slice_w1 = w1[i * 8 * self.dim : (i + 1) * 8 * self.dim]
                        if hasattr(block, "adaLN_modulation"):
                            if len(block.adaLN_modulation) == 4:
                                state_dict[f"blocks.{i}.adaLN_modulation.3.weight"] = slice_w1
                            elif len(block.adaLN_modulation) == 3:
                                state_dict[f"blocks.{i}.adaLN_modulation.2.weight"] = slice_w1
                            else:
                                state_dict[f"blocks.{i}.adaLN_modulation.1.weight"] = slice_w1
                    state_dict.pop("ada_proj.1.weight", None)
                    state_dict.pop("ada_proj.weight", None)
                    state_dict.pop("ada_proj.1.bias", None)
                    state_dict.pop("ada_proj.bias", None)
                elif w0 is not None and w2 is not None:
                    state_dict = dict(state_dict)
                    for i, block in enumerate(self.blocks):
                        slice_w2 = w2[i * 8 * self.dim : (i + 1) * 8 * self.dim]
                        if len(block.adaLN_modulation) == 4:
                            state_dict[f"blocks.{i}.adaLN_modulation.1.weight"] = w0
                            state_dict[f"blocks.{i}.adaLN_modulation.3.weight"] = slice_w2
                        elif len(block.adaLN_modulation) == 3:
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

            state_dict = dict(state_dict)
            for i, block in enumerate(self.blocks):
                if hasattr(block, "adaLN_modulation") and len(block.adaLN_modulation) == 4:
                    k0 = f"blocks.{i}.adaLN_modulation.0.weight"
                    k2 = f"blocks.{i}.adaLN_modulation.2.weight"
                    k1 = f"blocks.{i}.adaLN_modulation.1.weight"
                    k3 = f"blocks.{i}.adaLN_modulation.3.weight"
                    if k0 in state_dict and k1 not in state_dict:
                        state_dict[k1] = state_dict.pop(k0)
                    if k2 in state_dict and k3 not in state_dict:
                        state_dict[k3] = state_dict.pop(k2)

        state_dict = dict(state_dict)
        for i, block in enumerate(self.blocks):
            dense_k = f"blocks.{i}.adaLN_modulation.1.weight"
            if (
                dense_k in state_dict
                and hasattr(block, "adaLN_modulation")
                and len(block.adaLN_modulation) in (3, 4)
            ):
                w_dense = state_dict[dense_k]
                if w_dense.ndim == 2 and w_dense.shape[0] == 8 * self.dim:
                    state_dict.pop(dense_k)
                    target_lin1 = block.adaLN_modulation[1] if len(block.adaLN_modulation) == 4 else block.adaLN_modulation[0]
                    target_lin2 = block.adaLN_modulation[3] if len(block.adaLN_modulation) == 4 else block.adaLN_modulation[2]
                    if w_dense.abs().max().item() > 1e-6:
                        r = target_lin1.weight.shape[0]
                        try:
                            U, S, Vh = torch.linalg.svd(w_dense.float(), full_matrices=False)
                            sqrt_S = torch.sqrt(S[:r].clamp(min=1e-8))
                            w0 = (sqrt_S.unsqueeze(1) * Vh[:r, :]).to(target_lin1.weight.dtype)
                            w2 = (U[:, :r] * sqrt_S.unsqueeze(0)).to(target_lin2.weight.dtype)
                            if len(block.adaLN_modulation) == 4:
                                state_dict[f"blocks.{i}.adaLN_modulation.1.weight"] = w0
                                state_dict[f"blocks.{i}.adaLN_modulation.3.weight"] = w2
                            else:
                                state_dict[f"blocks.{i}.adaLN_modulation.0.weight"] = w0
                                state_dict[f"blocks.{i}.adaLN_modulation.2.weight"] = w2
                        except Exception:
                            pass
                    else:
                        if len(block.adaLN_modulation) == 4:
                            state_dict[f"blocks.{i}.adaLN_modulation.1.weight"] = target_lin1.weight.data
                            state_dict[f"blocks.{i}.adaLN_modulation.3.weight"] = torch.zeros_like(target_lin2.weight.data)
                        else:
                            state_dict[f"blocks.{i}.adaLN_modulation.0.weight"] = target_lin1.weight.data
                            state_dict[f"blocks.{i}.adaLN_modulation.2.weight"] = torch.zeros_like(target_lin2.weight.data)

        if "proj_in.weight" in state_dict:
            pw = state_dict["proj_in.weight"]
            state_dict = dict(state_dict)
            if self.use_coord_conv and pw.shape[1] == self.latent_channels:
                expanded = torch.zeros(
                    (self.dim, self.latent_channels + 3, pw.shape[2], pw.shape[3]),
                    dtype=pw.dtype,
                    device=pw.device,
                )
                expanded[:, :self.latent_channels] = pw
                pw = expanded
            elif not self.use_coord_conv and pw.shape[1] == self.latent_channels + 3:
                pw = pw[:, :self.latent_channels]

            if pw.shape[-1] == 1 and self.proj_in.weight.shape[-1] == 3:
                expanded_3x3 = torch.zeros(
                    (pw.shape[0], pw.shape[1], 3, 3),
                    dtype=pw.dtype,
                    device=pw.device,
                )
                expanded_3x3[:, :, 1:2, 1:2] = pw
                pw = expanded_3x3
            state_dict["proj_in.weight"] = pw

        if self.refinement_head is not None:
            state_dict = dict(state_dict)
            if "proj_out.weight" in state_dict and "refinement_head.proj_base.weight" not in state_dict:
                state_dict["refinement_head.proj_base.weight"] = state_dict["proj_out.weight"]
            elif "refinement_head.proj_base.weight" in state_dict and "proj_out.weight" not in state_dict:
                state_dict["proj_out.weight"] = state_dict["refinement_head.proj_base.weight"]
            if "proj_out.scale" in state_dict and "refinement_head.proj_base.scale" not in state_dict:
                state_dict["refinement_head.proj_base.scale"] = state_dict["proj_out.scale"]
            elif "refinement_head.proj_base.scale" in state_dict and "proj_out.scale" not in state_dict:
                state_dict["proj_out.scale"] = state_dict["refinement_head.proj_base.scale"]

            ref0_k = "refinement_head.refine.0.weight"
            if ref0_k in state_dict and state_dict[ref0_k].shape != self.refinement_head.refine[0].weight.shape:
                state_dict[ref0_k] = self.refinement_head.refine[0].weight.data
        else:
            if "refinement_head.proj_base.weight" in state_dict and "proj_out.weight" not in state_dict:
                state_dict = dict(state_dict)
                state_dict["proj_out.weight"] = state_dict.pop("refinement_head.proj_base.weight")
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("refinement_head.")}

        has_lora_in_sd = any("phase_attn_lora." in k for k in state_dict.keys())
        has_lora_in_model = any(getattr(b, "phase_attn_lora", None) is not None for b in self.blocks)
        if has_lora_in_model and not has_lora_in_sd:
            state_dict = dict(state_dict)
            for k, v in self.state_dict().items():
                if "phase_attn_lora." in k:
                    state_dict[k] = v
        elif not has_lora_in_model and has_lora_in_sd:
            state_dict = {k: v for k, v in state_dict.items() if "phase_attn_lora." not in k}

        has_cross_lora_in_sd = any("cross_lora." in k for k in state_dict.keys())
        has_cross_lora_in_model = any(
            getattr(getattr(b, "cross", None), "cross_lora", None) is not None for b in self.blocks
        )
        if has_cross_lora_in_model and not has_cross_lora_in_sd:
            state_dict = dict(state_dict)
            for k, v in self.state_dict().items():
                if "cross_lora." in k:
                    state_dict[k] = v
        elif not has_cross_lora_in_model and has_cross_lora_in_sd:
            state_dict = {k: v for k, v in state_dict.items() if "cross_lora." not in k}

        has_head_scale_in_sd = any("head_scale" in k for k in state_dict.keys())
        has_head_scale_in_model = any(
            getattr(getattr(b, "cross", None), "head_scale", None) is not None for b in self.blocks
        )
        if has_head_scale_in_model and not has_head_scale_in_sd:
            state_dict = dict(state_dict)
            for k, v in self.state_dict().items():
                if "head_scale" in k:
                    state_dict[k] = v
        elif not has_head_scale_in_model and has_head_scale_in_sd:
            state_dict = {k: v for k, v in state_dict.items() if "head_scale" not in k}

        has_dw_conv_in_sd = any("dw_conv." in k for k in state_dict.keys())
        has_dw_conv_in_model = any("dw_conv." in k for k in self.state_dict().keys())
        if has_dw_conv_in_model and not has_dw_conv_in_sd:
            state_dict = dict(state_dict)
            for k, v in self.state_dict().items():
                if "dw_conv." in k:
                    state_dict[k] = v
        elif not has_dw_conv_in_model and has_dw_conv_in_sd:
            state_dict = {k: v for k, v in state_dict.items() if "dw_conv." not in k}

        has_skip_in_sd = any("skip_projs." in k for k in state_dict.keys())
        has_skip_in_model = self.skip_projs is not None
        if has_skip_in_model and not has_skip_in_sd:
            state_dict = dict(state_dict)
            for k, v in self.state_dict().items():
                if "skip_projs." in k:
                    state_dict[k] = v
        elif not has_skip_in_model and has_skip_in_sd:
            state_dict = {k: v for k, v in state_dict.items() if "skip_projs." not in k}

        model_sd = self.state_dict()
        for k, v in model_sd.items():
            if k not in state_dict and any(
                tag in k for tag in (
                    "skip_norms.", "skip_phase_gate", "v_res_lambda", "sink_k", ".k_a.", ".k_b.",
                    "registers", "phase_layer_mixer", "final_modulation.", "refinement_head.",
                    "pooler.", "head_gate.", "hf_gain"
                )
            ):
                state_dict[k] = v

        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def get_expert_weights(self, phase_idx: int, include_lora: bool = True) -> dict:
        prefix = f"phase_experts.{phase_idx}."
        expert_sd = {}
        for k, v in self.state_dict().items():
            if prefix in k:
                expert_sd[k] = v
        cross_idx = min(phase_idx, len(self.phase_cross_kv) - 1)
        cross_prefix = f"phase_cross_kv.{cross_idx}."
        for k, v in self.state_dict().items():
            if cross_prefix in k:
                expert_sd[k] = v
        if include_lora:
            lora_prefix = f"phase_attn_lora.{phase_idx}."
            for k, v in self.state_dict().items():
                if lora_prefix in k:
                    expert_sd[k] = v
        return expert_sd

    def get_backbone_weights(self, include_lora: bool = False) -> dict:
        backbone_sd = {}
        for k, v in self.state_dict().items():
            if "phase_experts." in k:
                continue
            if not include_lora and "phase_attn_lora." in k:
                continue
            backbone_sd[k] = v
        return backbone_sd

    def set_active_expert(self, phase_idx: int, expert_sd: dict, device: torch.device):
        for i, block in enumerate(self.blocks):
            if block.phase_experts is not None and phase_idx < len(block.phase_experts):
                exp = block.phase_experts[phase_idx]
                prefix = f"blocks.{i}.phase_experts.{phase_idx}."
                for name, param in exp.named_parameters():
                    k = prefix + name
                    if k in expert_sd:
                        val = expert_sd[k]
                        scale_k = prefix + (name[:-7] if name.endswith(".weight") else name) + ".scale"
                        if scale_k in expert_sd:
                            val = val.to(torch.float32) * expert_sd[scale_k].to(torch.float32)
                        param.data = val.to(device=device, dtype=param.dtype)

                for name, buf in exp.named_buffers():
                    k = prefix + name
                    if name.endswith(".weight") and k in expert_sd and (prefix + name[:-7] + ".scale") not in expert_sd:
                        w_raw = expert_sd[k].float()
                        if w_raw.dtype != buf.dtype and not str(expert_sd[k].dtype).startswith("torch.float8") and expert_sd[k].dtype != torch.int8:
                            max_val = w_raw.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                            fp8_max = FP8_MAX if getattr(self, "quantized_fp8", False) else 127.0
                            scale = max_val / fp8_max
                            w_quant = (w_raw / scale).round().clamp(-128, 127).to(buf.dtype) if buf.dtype == torch.int8 else (w_raw / scale).clamp(-fp8_max, fp8_max).to(buf.dtype)
                            buf.data = w_quant.to(device=device)
                        else:
                            buf.data = expert_sd[k].to(device=device, dtype=buf.dtype)
                    elif k in expert_sd:
                        buf.data = expert_sd[k].to(device=device, dtype=buf.dtype)
                    elif name.endswith(".scale") and (prefix + name[:-6] + ".weight") in expert_sd:
                        w_raw = expert_sd[prefix + name[:-6] + ".weight"].float()
                        max_val = w_raw.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                        fp8_max = FP8_MAX if getattr(self, "quantized_fp8", False) else 127.0
                        scale = (max_val / fp8_max).to(device=device, dtype=buf.dtype)
                        buf.data = scale

            if hasattr(block, "phase_attn_lora") and block.phase_attn_lora is not None and phase_idx < len(block.phase_attn_lora):
                lora_mod = block.phase_attn_lora[phase_idx]
                lora_prefix = f"blocks.{i}.phase_attn_lora.{phase_idx}."
                for name, param in lora_mod.named_parameters():
                    k = lora_prefix + name
                    if k in expert_sd:
                        self._lora_in_experts = True
                        val = expert_sd[k]
                        scale_k = lora_prefix + (name[:-7] if name.endswith(".weight") else name) + ".scale"
                        if scale_k in expert_sd:
                            val = val.to(torch.float32) * expert_sd[scale_k].to(torch.float32)
                        param.data = val.to(device=device, dtype=param.dtype)
                for name, buf in lora_mod.named_buffers():
                    k = lora_prefix + name
                    if name.endswith(".weight") and k in expert_sd and (lora_prefix + name[:-7] + ".scale") not in expert_sd:
                        self._lora_in_experts = True
                        w_raw = expert_sd[k].float()
                        if w_raw.dtype != buf.dtype and not str(expert_sd[k].dtype).startswith("torch.float8") and expert_sd[k].dtype != torch.int8:
                            max_val = w_raw.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                            fp8_max = FP8_MAX if getattr(self, "quantized_fp8", False) else 127.0
                            scale = max_val / fp8_max
                            w_quant = (w_raw / scale).round().clamp(-128, 127).to(buf.dtype) if buf.dtype == torch.int8 else (w_raw / scale).clamp(-fp8_max, fp8_max).to(buf.dtype)
                            buf.data = w_quant.to(device=device)
                        else:
                            buf.data = expert_sd[k].to(device=device, dtype=buf.dtype)
                    elif k in expert_sd:
                        self._lora_in_experts = True
                        buf.data = expert_sd[k].to(device=device, dtype=buf.dtype)
                    elif name.endswith(".scale") and (lora_prefix + name[:-6] + ".weight") in expert_sd:
                        self._lora_in_experts = True
                        w_raw = expert_sd[lora_prefix + name[:-6] + ".weight"].float()
                        max_val = w_raw.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                        fp8_max = FP8_MAX if getattr(self, "quantized_fp8", False) else 127.0
                        scale = (max_val / fp8_max).to(device=device, dtype=buf.dtype)
                        buf.data = scale

            if hasattr(block, "phase_attn_lora") and block.phase_attn_lora is not None and phase_idx < len(block.phase_attn_lora):
                lora_mod = block.phase_attn_lora[phase_idx]
                lora_prefix = f"blocks.{i}.phase_attn_lora.{phase_idx}."
                for name, param in lora_mod.named_parameters():
                    k = lora_prefix + name
                    if k in expert_sd:
                        self._lora_in_experts = True
                        val = expert_sd[k]
                        scale_k = lora_prefix + (name[:-7] if name.endswith(".weight") else name) + ".scale"
                        if scale_k in expert_sd:
                            val = val.to(torch.float32) * expert_sd[scale_k].to(torch.float32)
                        param.data = val.to(device=device, dtype=param.dtype)
                for name, buf in lora_mod.named_buffers():
                    k = lora_prefix + name
                    if k in expert_sd:
                        self._lora_in_experts = True
                        buf.data = expert_sd[k].to(device=device, dtype=buf.dtype)
                    elif name.endswith(".scale") and (lora_prefix + name[:-6] + ".weight") in expert_sd:
                        self._lora_in_experts = True
                        w_raw = expert_sd[lora_prefix + name[:-6] + ".weight"].float()
                        max_val = w_raw.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                        fp8_max = FP8_MAX if getattr(self, "quantized_fp8", False) else 127.0
                        scale = (max_val / fp8_max).to(device=device, dtype=buf.dtype)
                        buf.data = scale
                    elif name.endswith(".weight") and k in expert_sd and (lora_prefix + name[:-7] + ".scale") not in expert_sd:
                        self._lora_in_experts = True
                        w_raw = expert_sd[k].float()
                        max_val = w_raw.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                        fp8_max = FP8_MAX if getattr(self, "quantized_fp8", False) else 127.0
                        scale = max_val / fp8_max
                        w_quant = (w_raw / scale).clamp(-fp8_max, fp8_max).to(buf.dtype)
                        buf.data = w_quant.to(device=device)

        cross_idx = min(phase_idx, len(self.phase_cross_kv) - 1)
        if cross_idx < len(self.phase_cross_kv):
            kv_mod = self.phase_cross_kv[cross_idx]
            cross_prefix = f"phase_cross_kv.{cross_idx}."
            for name, param in kv_mod.named_parameters():
                k = cross_prefix + name
                if k in expert_sd:
                    param.data = expert_sd[k].to(device=device, dtype=param.dtype)
            for name, buf in kv_mod.named_buffers():
                k = cross_prefix + name
                if k in expert_sd:
                    buf.data = expert_sd[k].to(device=device, dtype=buf.dtype)

    def clear_active_expert(self, phase_idx: int, clear_lora: Optional[bool] = None):
        if clear_lora is None:
            clear_lora = getattr(self, "_lora_in_experts", False)
        for block in self.blocks:
            if block.phase_experts is not None and phase_idx < len(block.phase_experts):
                exp = block.phase_experts[phase_idx]
                for param in exp.parameters():
                    try:
                        param.data = torch.empty((0, 0), dtype=param.dtype, device="cpu")
                    except Exception:
                        param.data = torch.empty((0, 0), dtype=torch.float32, device="cpu")
                for buf in exp.buffers():
                    try:
                        buf.data = torch.empty((0, 0), dtype=buf.dtype, device="cpu")
                    except Exception:
                        buf.data = torch.empty((0, 0), dtype=torch.uint8, device="cpu")

            if clear_lora and getattr(block, "phase_attn_lora", None) is not None and phase_idx < len(block.phase_attn_lora):
                lora_mod = block.phase_attn_lora[phase_idx]
                for param in lora_mod.parameters():
                    try:
                        param.data = torch.empty((0, 0), dtype=param.dtype, device="cpu")
                    except Exception:
                        param.data = torch.empty((0, 0), dtype=torch.float32, device="cpu")
                for buf in lora_mod.buffers():
                    try:
                        buf.data = torch.empty((0, 0), dtype=buf.dtype, device="cpu")
                    except Exception:
                        buf.data = torch.empty((0, 0), dtype=torch.uint8, device="cpu")

    def get_expert_cuda_memory_mb(self) -> Dict[int, float]:
        res: Dict[int, float] = {}
        for p in range(self.num_phases):
            total_bytes = 0
            for block in self.blocks:
                if block.phase_experts is not None and p < len(block.phase_experts):
                    exp = block.phase_experts[p]
                    for param in exp.parameters():
                        if str(param.device).startswith("cuda") and param.numel() > 0:
                            total_bytes += param.numel() * param.element_size()
                    for buf in exp.buffers():
                        if str(buf.device).startswith("cuda") and buf.numel() > 0:
                            total_bytes += buf.numel() * buf.element_size()
                if getattr(block, "phase_attn_lora", None) is not None and p < len(block.phase_attn_lora):
                    lora_mod = block.phase_attn_lora[p]
                    for param in lora_mod.parameters():
                        if str(param.device).startswith("cuda") and param.numel() > 0:
                            total_bytes += param.numel() * param.element_size()
                    for buf in lora_mod.buffers():
                        if str(buf.device).startswith("cuda") and buf.numel() > 0:
                            total_bytes += buf.numel() * buf.element_size()
            res[p] = total_bytes / (1024**2)
        return res

    def enable_expert_streaming(
        self,
        experts_dir: str = "cp/experts",
        device: Optional[torch.device] = None,
        chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
        max_experts_vram: int = 1,
        max_experts_ram: int = 0,
        lora_in_experts: Optional[bool] = None,
    ):
        target_dev = device or next(self.parameters()).device
        if lora_in_experts is not None:
            self._lora_in_experts = lora_in_experts
        self.expert_streamer = ChunkedResidencyManager(
            self,
            experts_dir=experts_dir,
            device=target_dev,
            chunk_size_mb=chunk_size_mb,
            max_experts_vram=max_experts_vram,
            max_experts_ram=max_experts_ram,
            lora_in_experts=lora_in_experts,
        )
        for p in range(self.num_phases):
            self.clear_active_expert(p)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        self.expert_streamer.stream_phase(0)
        return self

    def configure_expert_residency(self, max_experts_vram: int = 1, max_experts_ram: int = 0) -> Dict[str, Any]:
        if hasattr(self, "expert_streamer") and self.expert_streamer is not None:
            return self.expert_streamer.configure_residency(max_experts_vram=max_experts_vram, max_experts_ram=max_experts_ram)
        return {"max_experts_vram": max_experts_vram, "max_experts_ram": max_experts_ram}

    def get_expert_residency_status(self) -> Dict[str, Any]:
        if hasattr(self, "expert_streamer") and self.expert_streamer is not None:
            return self.expert_streamer.get_residency_status()
        return {"error": "expert_streamer_not_initialized"}

    def stream_phase_expert(self, phase_idx: int):
        if hasattr(self, "expert_streamer") and self.expert_streamer is not None:
            self.expert_streamer.stream_phase(phase_idx)

    def prefetch_next_phase(self, phase_idx: int):
        if hasattr(self, "expert_streamer") and self.expert_streamer is not None:
            self.expert_streamer.prefetch_next(phase_idx)

    def load_backbone_chunked(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        use_ema: bool = True,
        chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
        load_lora: bool = True,
    ) -> Dict[str, Any]:
        target_dev = device or next(self.parameters()).device
        streamer = ChunkedWeightStreamer(chunk_size_mb=chunk_size_mb)
        res = streamer.load_checkpoint_chunked(
            model=self,
            checkpoint_path=checkpoint_path,
            device=target_dev,
            use_ema=use_ema,
            is_backbone_only=True,
            load_lora=load_lora,
        )
        if load_lora and self._lora_in_experts is None:
            self._lora_in_experts = False
        return res

    def _init_weights(self_or_module, module: Optional[nn.Module] = None):
        if module is None:
            mod = self_or_module
            self_inst = None
        else:
            self_inst = self_or_module
            mod = module

        if isinstance(mod, PhaseAttnLoRA):
            mod.reset_parameters()
        elif isinstance(mod, CrossAttnLoRA):
            mod.reset_parameters()
        elif isinstance(mod, PhaseExpert):
            nn.init.normal_(mod.w3.weight, mean=0.0, std=0.015)
        elif isinstance(mod, nn.Linear):
            if self_inst is not None and hasattr(self_inst, "skip_projs") and self_inst.skip_projs is not None:
                if any(mod is p for p in self_inst.skip_projs):
                    nn.init.normal_(mod.weight, mean=0.0, std=0.01)
                    return
            nn.init.xavier_uniform_(mod.weight)
        elif isinstance(mod, nn.Conv2d):
            if mod.groups == mod.in_channels and mod.groups > 1:
                nn.init.normal_(mod.weight, mean=0.0, std=0.01)
            else:
                nn.init.xavier_uniform_(mod.weight)
                if self_inst is not None and hasattr(self_inst, "proj_in") and mod is self_inst.proj_in and getattr(self_inst, "use_coord_conv", False):
                    nn.init.normal_(self_inst.proj_in.weight.data[:, self_inst.latent_channels:, :, :], mean=0.0, std=0.01)

    @staticmethod
    def _get_coord_maps(
        B: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (H, W, str(device), dtype)
        is_compiling = False
        try:
            is_compiling = torch.compiler.is_compiling()
        except Exception:
            pass

        if not is_compiling and key in _COORD_CACHE:
            base_map = _COORD_CACHE[key]
            return base_map.expand(B, -1, -1, -1)

        y = torch.zeros(1, device=device, dtype=dtype) if H == 1 else torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
        x = torch.zeros(1, device=device, dtype=dtype) if W == 1 else torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        r = (grid_x.pow(2) + grid_y.pow(2)) - (2.0 / 3.0)
        base_map = (0.25 * torch.stack([grid_x, grid_y, r], dim=0)).unsqueeze(0)

        if not is_compiling:
            _COORD_CACHE[key] = base_map
            if len(_COORD_CACHE) > _COORD_CACHE_MAX:
                _COORD_CACHE.popitem(last=False)

        return base_map.expand(B, -1, -1, -1)

    @staticmethod
    def _get_2d_pos_embed(
        H: int,
        W: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = ("pos2d_v2", H, W, dim, str(device), dtype)
        is_compiling = False
        try:
            is_compiling = torch.compiler.is_compiling()
        except Exception:
            pass

        if not is_compiling and key in _COORD_CACHE:
            return _COORD_CACHE[key]

        half_d = dim // 2
        q_d = max(1, half_d // 2)
        omega = torch.exp(
            torch.linspace(math.log(0.25), math.log(2.0 * math.pi), steps=q_d, device=device, dtype=torch.float32)
        )
        y = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        ang_y = gy.reshape(-1, 1) * omega.unsqueeze(0)
        ang_x = gx.reshape(-1, 1) * omega.unsqueeze(0)
        emb = torch.cat([torch.sin(ang_y), torch.cos(ang_y), torch.sin(ang_x), torch.cos(ang_x)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        else:
            emb = emb[:, :dim]
        pos = (0.20 * emb).unsqueeze(0).to(dtype=dtype)

        if not is_compiling:
            _COORD_CACHE[key] = pos
            if len(_COORD_CACHE) > _COORD_CACHE_MAX:
                _COORD_CACHE.popitem(last=False)
        return pos

    def _mix_phase_text_layers(self, cond_seq: torch.Tensor, p: int) -> torch.Tensor:
        if cond_seq.dim() != 4:
            return cond_seq
        num_avail = cond_seq.shape[1]
        if self.phase_layer_mixer is not None and num_avail == self.phase_layer_mixer.shape[1]:
            w = F.softmax(self.phase_layer_mixer[p].float(), dim=0).to(cond_seq.dtype)
            return (cond_seq * w.unsqueeze(0).unsqueeze(2)).sum(dim=1)
        p_slice = min(p, num_avail - 1)
        return cond_seq[:, p_slice]

    def pool_text(self, text_seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hasattr(self, "pooler") and self.pooler is not None:
            return self.pooler(text_seq, mask=mask)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(text_seq.dtype)
            return (text_seq * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        return text_seq.mean(dim=1)

    def encode_text(
        self,
        cond_seq: torch.Tensor,
        phase_idx: Optional[Union[int, torch.Tensor]] = None,
        apply_k_norm: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = cond_seq.shape[0]
        if isinstance(phase_idx, torch.Tensor) and phase_idx.numel() == 2 * B:
            phase_idx = phase_idx[B:]

        is_uniform = True
        if isinstance(phase_idx, torch.Tensor) and phase_idx.numel() > 1:
            if phase_idx.device.type == "cpu" and not (phase_idx == phase_idx[0]).all():
                is_uniform = False
            elif phase_idx.device.type != "cpu":
                is_uniform = False

        if is_uniform:
            if phase_idx is None:
                p = 0
            elif isinstance(phase_idx, int):
                p = min(max(0, phase_idx), len(self.phase_cross_kv) - 1)
            elif isinstance(phase_idx, torch.Tensor):
                p = 0 if phase_idx.device.type == "meta" else min(max(0, int(phase_idx[0].item())), len(self.phase_cross_kv) - 1)
            else:
                p = 0

            seq_p = self._mix_phase_text_layers(cond_seq, p)
            B, M, _ = seq_p.shape
            proj = self.phase_cross_kv[p]
            proj_w = getattr(proj, "weight", None)
            if not torch.is_autocast_enabled():
                if (
                    proj_w is not None
                    and proj_w.dtype.is_floating_point
                    and "float8" not in str(proj_w.dtype)
                    and seq_p.dtype != proj_w.dtype
                ):
                    seq_p = seq_p.to(proj_w.dtype)
                elif "float8" in str(seq_p.dtype):
                    seq_p = seq_p.to(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
            elif "float8" in str(seq_p.dtype):
                seq_p = seq_p.to(torch.bfloat16)
            kv = proj(seq_p)
        else:
            p_indices = phase_idx.to(device=cond_seq.device, dtype=torch.long)
            seq_dim4 = (cond_seq.dim() == 4)
            M = cond_seq.shape[2] if seq_dim4 else cond_seq.shape[1]
            out_features = self.kv_heads * self.head_dim * 2
            target_dtype = cond_seq.dtype if cond_seq.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
            kv = torch.empty((B, M, out_features), dtype=target_dtype, device=cond_seq.device)
            for p_idx in range(len(self.phase_cross_kv)):
                mask_p = (p_indices == p_idx)
                if mask_p.any():
                    seq_subset = self._mix_phase_text_layers(cond_seq[mask_p], p_idx) if seq_dim4 else cond_seq[mask_p]
                    proj_out = self.phase_cross_kv[p_idx](seq_subset.to(target_dtype))
                    kv[mask_p] = proj_out.to(target_dtype)

        kv = kv.view(B, M, 2, self.kv_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        if apply_k_norm:
            k = self.shared_k_norm(k)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        return k, v

    @torch.inference_mode()
    def prepare_phase_cross_kvs(
        self,
        cond_seq: torch.Tensor,
        phase_idx: int,
    ) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        base_k, base_v = self.encode_text(cond_seq, phase_idx=phase_idx, apply_k_norm=False)
        per_block_kvs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for block in self.blocks:
            if getattr(block, "use_cross", False) and hasattr(block, "cross"):
                if getattr(block.cross, "cross_lora", None) is not None:
                    k_b, v_b = block.cross.cross_lora(base_k, base_v)
                else:
                    k_b, v_b = base_k, base_v
                k_b = self.shared_k_norm(k_b).contiguous()
                k_b, v_b = block.cross.append_sinks(k_b, v_b.contiguous())
                per_block_kvs.append((k_b.contiguous(), v_b.contiguous()))
            else:
                per_block_kvs.append(None)
        return per_block_kvs

    def forward(
        self,
        z: torch.Tensor,
        cond_pooled: torch.Tensor,
        cond_seq: Optional[torch.Tensor] = None,
        t_phase: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        text_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        per_block_text_kvs: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        temperature: float = 1.0,
        guidance: Optional[torch.Tensor] = None,
        phase_idx: Optional[Union[int, torch.Tensor]] = None,
        use_block_cache: bool = False,
        cached_mid_residual: Optional[torch.Tensor] = None,
        return_mid_residual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, C, H, W = z.shape
        p = self.patch_size
        Hp = H // p
        Wp = W // p
        spatial_len = Hp * Wp

        if self.use_coord_conv:
            coords = self._get_coord_maps(B, H, W, z.device, z.dtype)
            z = torch.cat([z, coords], dim=1)

        if not torch.is_autocast_enabled() and self.proj_in.weight.dtype != z.dtype:
            z = z.to(self.proj_in.weight.dtype)
        x = self.proj_in(z)
        x = x.flatten(2).transpose(1, 2)
        x = x + self._get_2d_pos_embed(Hp, Wp, self.dim, x.device, x.dtype)

        if self.registers is not None and self.num_registers > 0:
            regs = self.registers.expand(B, -1, -1).to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, regs], dim=1)

        if cond_pooled.shape[0] != B and cond_pooled.shape[0] == 2 * B:
            cond_pooled = cond_pooled[B:]
            if cond_seq is not None and cond_seq.shape[0] == 2 * B:
                cond_seq = cond_seq[B:]
            if mask is not None and mask.shape[0] == 2 * B:
                mask = mask[B:]
            if text_kvs is not None and isinstance(text_kvs, (tuple, list)) and len(text_kvs) == 2:
                if text_kvs[0].shape[0] == 2 * B and text_kvs[1].shape[0] == 2 * B:
                    text_kvs = (text_kvs[0][B:], text_kvs[1][B:])

        time_emb = self.time_norm(self.time_embed(t_phase))
        if cond_pooled.shape[-1] == self.dim:
            text_emb = self.text_norm(cond_pooled)
        else:
            text_emb = self.text_norm(self.text_proj(cond_pooled))

        phase_idx_tensor = None
        if phase_idx is not None:
            if isinstance(phase_idx, int):
                phase_idx_tensor = torch.full((B,), phase_idx, device=z.device, dtype=torch.long)
            elif phase_idx.dim() == 0:
                phase_idx_tensor = phase_idx.unsqueeze(0).expand(B).to(dtype=torch.long, device=z.device)
            else:
                phase_idx_tensor = phase_idx.to(dtype=torch.long, device=z.device)

        if self.use_phase:
            t01 = t_phase.flatten().float().clamp(0.0, 1.0)
            u = t01 * float(self.num_phases - 1)
            idx_low = u.floor().long().clamp(0, self.num_phases - 2)
            idx_high = idx_low + 1
            alpha = (u - idx_low.float()).unsqueeze(-1)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)

            phase_emb = (1.0 - alpha) * self.phase_table[idx_low] + alpha * self.phase_table[idx_high]
            phase_emb = self.phase_norm(phase_emb).to(time_emb.dtype)

            if phase_idx_tensor is None:
                phase_idx_tensor = (t01 * self.num_phases).floor().long().clamp(0, self.num_phases - 1)

            cond_combined = torch.cat([time_emb, text_emb, phase_emb], dim=-1)
        else:
            cond_combined = torch.cat([time_emb, text_emb], dim=-1)

        if self.use_guidance and guidance is not None:
            g_emb = self.guidance_norm(self.guidance_embed(guidance))
            cond_combined = torch.cat([cond_combined, g_emb], dim=-1)

        cond_vec = self.cond_mlp(cond_combined)
        phase_idx_block = phase_idx if isinstance(phase_idx, int) else phase_idx_tensor

        text_k, text_v = (None, None)
        has_any_cross_lora = any(
            getattr(getattr(b, "cross", None), "cross_lora", None) is not None for b in self.blocks
        )
        if per_block_text_kvs is None:
            if text_kvs is None:
                if cond_seq is None:
                    raise ValueError("text_kvs is None and cond_seq is None")
                text_k, text_v = self.encode_text(
                    cond_seq,
                    phase_idx=phase_idx_block,
                    apply_k_norm=not has_any_cross_lora,
                )
            else:
                text_k, text_v = text_kvs

        precomputed_mask = mask
        if mask is not None and mask.dim() == 2:
            b_m = mask.shape[0]
            bool_m = mask.view(b_m, 1, 1, -1).bool()
            if self.use_cross_sink and self.num_sinks > 0:
                sink_pad = torch.ones((b_m, 1, 1, self.num_sinks), dtype=torch.bool, device=mask.device)
                bool_m = torch.cat([bool_m, sink_pad], dim=-1)
            has_val = bool_m.any(dim=-1, keepdim=True)
            precomputed_mask = torch.where(has_val, bool_m, torch.ones_like(bool_m))

        actual_temp = float(temperature)
        saved_skips: Dict[int, torch.Tensor] = {}
        num_skips = len(self.skip_projs) if (self.use_skip_connections and self.skip_projs is not None) else 0

        v_anchor: Optional[torch.Tensor] = None
        bottleneck_idx = self.depth // 2

        mid_start = self.depth // 3
        mid_end = self.depth - (self.depth // 3)
        x_before_mid: Optional[torch.Tensor] = None
        new_mid_cache = None

        for i, block in enumerate(self.blocks):
            if use_block_cache and cached_mid_residual is not None:
                if i == mid_start:
                    mid_delta = cached_mid_residual[0] if isinstance(cached_mid_residual, tuple) else cached_mid_residual
                    x = x + mid_delta.to(x.dtype)
                    continue
                elif mid_start < i < mid_end:
                    continue

            if return_mid_residual and i == mid_start:
                x_before_mid = x

            if self.use_skip_connections and self.skip_projs is not None and i >= self.depth - num_skips:
                skip_idx = self.depth - 1 - i
                if skip_idx in saved_skips:
                    proj = self.skip_projs[skip_idx]
                    skip_val = saved_skips[skip_idx]
                    if self.skip_norms is not None and skip_idx < len(self.skip_norms):
                        skip_val = self.skip_norms[skip_idx](skip_val)
                    if not torch.is_autocast_enabled():
                        if isinstance(proj, nn.Linear) and proj.weight.is_floating_point() and not str(proj.weight.dtype).startswith("torch.float8"):
                            if skip_val.dtype != proj.weight.dtype:
                                skip_val = skip_val.to(proj.weight.dtype)
                    skip_out = proj(skip_val).to(x.dtype)
                    if self.skip_phase_gate is not None and (phase_idx is not None or phase_idx_tensor is not None):
                        if isinstance(phase_idx, int):
                            p_g = min(max(0, phase_idx), self.skip_phase_gate.shape[0] - 1)
                            gate_s = self.skip_phase_gate[p_g, skip_idx].view(1, 1, -1).to(x.dtype)
                        elif phase_idx_tensor is not None:
                            p_g = phase_idx_tensor.long().clamp(0, self.skip_phase_gate.shape[0] - 1)
                            gate_s = self.skip_phase_gate[p_g, skip_idx].unsqueeze(1).to(x.dtype)
                        else:
                            gate_s = self.skip_phase_gate[0, skip_idx].view(1, 1, -1).to(x.dtype)
                        skip_out = skip_out * (0.35 * torch.tanh(gate_s))
                    else:
                        skip_out = skip_out * 0.35
                    x = x + skip_out

            if per_block_text_kvs is not None and i < len(per_block_text_kvs):
                kv_pair = per_block_text_kvs[i]
                k, v = kv_pair if kv_pair is not None else (None, None)
                kv_adapted = True
                k_norm_fn = None
            else:
                k = text_k if block.use_cross else None
                v = text_v if block.use_cross else None
                kv_adapted = False
                k_norm_fn = self.shared_k_norm if has_any_cross_lora else None

            need_v = self.use_value_residual and (i == 0)
            v_in = None if need_v else v_anchor

            if self.training and self.gradient_checkpointing:
                res_b = cp.checkpoint(
                    block,
                    x, cond_vec, None, None, None, None, None, None, None, None,
                    Hp, Wp, phase_idx_block, k, v, precomputed_mask, actual_temp, v_in, kv_adapted, k_norm_fn, need_v,
                    use_reentrant=False,
                )
            else:
                res_b = block(
                    x, c=cond_vec, grid_h=Hp, grid_w=Wp, phase_idx=phase_idx_block,
                    text_k=k, text_v=v, mask=precomputed_mask, temperature=actual_temp,
                    v_first=v_in, kv_already_adapted=kv_adapted, k_norm_fn=k_norm_fn, return_v=need_v,
                )

            if need_v and isinstance(res_b, tuple):
                x, v_anchor = res_b
            else:
                x = res_b

            if self.use_skip_connections and self.skip_projs is not None and i < num_skips:
                saved_skips[i] = x

            if return_mid_residual and i == mid_end - 1 and x_before_mid is not None:
                new_mid_cache = x - x_before_mid

        if self.num_registers > 0 and x.shape[1] > spatial_len:
            x = x[:, :spatial_len]

        scale_f, shift_f = self.final_modulation(cond_vec).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale_f.unsqueeze(1)) + shift_f.unsqueeze(1)

        if self.refinement_head is not None:
            x = self.refinement_head(x, Hp, Wp)
        else:
            x = self.proj_out(x)
            patch_dim = self.latent_channels * self.patch_size * self.patch_size
            x = x.transpose(1, 2).contiguous().reshape(B, patch_dim, Hp, Wp)

        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)

        if return_mid_residual and new_mid_cache is not None:
            return x, new_mid_cache
        return x

    def compile_model(self, mode: str = "default", dynamic: bool = True):
        if hasattr(torch, "compile"):
            return torch.compile(self, mode=mode, dynamic=dynamic)
        return self

    def set_attention_tracking(self, enabled: bool = True):
        for block in self.blocks:
            if hasattr(block, "attn"):
                block.attn.track_attention = enabled
                if not enabled:
                    block.attn.last_attn_weights = None
            if hasattr(block, "cross") and getattr(block, "use_cross", False):
                block.cross.track_attention = enabled
                if not enabled:
                    block.cross.last_attn_weights = None

    def get_attention_maps(self) -> Dict[str, Dict[str, torch.Tensor]]:
        maps = {}
        for i, block in enumerate(self.blocks):
            block_maps = {}
            if hasattr(block, "attn") and block.attn.last_attn_weights is not None:
                block_maps["self_attn"] = block.attn.last_attn_weights
            if hasattr(block, "cross") and getattr(block, "use_cross", False) and block.cross.last_attn_weights is not None:
                block_maps["cross_attn"] = block.cross.last_attn_weights
            if block_maps:
                maps[f"block_{i}"] = block_maps
        return maps

    def get_attention_statistics(self) -> Dict[str, float]:
        stats = {}
        cross_entropies = []
        cross_maxes = []
        self_entropies = []
        for i, block in enumerate(self.blocks):
            if hasattr(block, "cross") and getattr(block, "use_cross", False) and block.cross.last_attn_weights is not None:
                w = block.cross.last_attn_weights.float()
                eps = 1e-12
                entropy = -(w * torch.log(w + eps)).sum(dim=-1).mean().item()
                max_p = w.max(dim=-1).values.mean().item()
                cross_entropies.append(entropy)
                cross_maxes.append(max_p)
                stats[f"layer_{i}_cross_entropy"] = entropy
                stats[f"layer_{i}_cross_max_prob"] = max_p
            if hasattr(block, "attn") and block.attn.last_attn_weights is not None:
                w = block.attn.last_attn_weights.float()
                eps = 1e-12
                entropy = -(w * torch.log(w + eps)).sum(dim=-1).mean().item()
                self_entropies.append(entropy)
                stats[f"layer_{i}_self_entropy"] = entropy

        if cross_entropies:
            stats["mean_cross_entropy"] = float(sum(cross_entropies) / len(cross_entropies))
            stats["mean_cross_max_prob"] = float(sum(cross_maxes) / len(cross_maxes))
        if self_entropies:
            stats["mean_self_entropy"] = float(sum(self_entropies) / len(self_entropies))
        return stats


from config import (
    EDGEFLOW_100M_CONFIG,
    EDGEFLOW_4B_CONFIG_24L,
    EDGEFLOW_4B_CONFIG_30L,
    EDGEFLOW_4B_CONFIG_V4,
    EDGEFLOW_4B_CONFIG_V5,
    get_model_config,
)
