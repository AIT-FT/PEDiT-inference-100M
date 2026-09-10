import math
import os
import threading
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


_ROPE_CACHE = OrderedDict()
_ROPE_CACHE_MAX = 128
_ROPE_LOCK = threading.Lock()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "TimestepEmbedding dim must be even"

        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        t = t.flatten().float() * 1000.0

        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / max(1, half_dim - 1)
        freqs = torch.exp(
            -torch.arange(half_dim, device=device, dtype=torch.float32) * emb_scale
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


def _get_rope_2d(
    grid_h: int,
    grid_w: int,
    head_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device_key = str(device)
    key = (grid_h, grid_w, head_dim, device_key)

    with _ROPE_LOCK:
        if key in _ROPE_CACHE:
            val = _ROPE_CACHE.pop(key)
            _ROPE_CACHE[key] = val
            return val

    half_dim = head_dim // 2
    dim_y = half_dim // 2
    dim_x = half_dim - dim_y

    freqs_y = 1.0 / (
        10000.0 ** (torch.arange(0, dim_y, device=device).float() / max(dim_y, 1))
    )
    freqs_x = 1.0 / (
        10000.0 ** (torch.arange(0, dim_x, device=device).float() / max(dim_x, 1))
    )

    # Centered coordinates improve resolution extrapolation.
    y = torch.arange(grid_h, device=device).float() - (grid_h - 1) / 2.0
    x = torch.arange(grid_w, device=device).float() - (grid_w - 1) / 2.0

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

    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _get_rope_2d(grid_h, grid_w, q.shape[-1], q.device)

    cos = cos.to(q.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(q.dtype).unsqueeze(0).unsqueeze(0)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int):
        super().__init__()

        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        assert heads % kv_heads == 0, "heads must be divisible by kv_heads"

        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads

        assert self.head_dim % 2 == 0, (
            f"head_dim must be even for 2D RoPE, got {self.head_dim}"
        )

        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x).view(B, N, self.heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_2d_rope(q, k, grid_h, grid_w)

        if self.kv_heads != self.heads:
            try:
                # Новые версии PyTorch могут использовать GQA без физического расширения K/V.
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    enable_gqa=True,
                )
            except TypeError:
                rep = self.heads // self.kv_heads

                k_gqa = (
                    k.unsqueeze(2)
                    .expand(B, self.kv_heads, rep, N, self.head_dim)
                    .reshape(B, self.heads, N, self.head_dim)
                    .contiguous()
                )
                v_gqa = (
                    v.unsqueeze(2)
                    .expand(B, self.kv_heads, rep, N, self.head_dim)
                    .reshape(B, self.heads, N, self.head_dim)
                    .contiguous()
                )

                out = F.scaled_dot_product_attention(q, k_gqa, v_gqa)
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)


class CrossAttentionLite(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()

        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"

        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q(x).view(B, N, self.heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        attn_mask = None

        if mask is not None:
            bool_mask = mask.view(B, 1, 1, -1).bool()
            has_valid = bool_mask.any(dim=-1, keepdim=True)
            attn_mask = torch.where(has_valid, bool_mask, torch.ones_like(bool_mask))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)

        # Если у сэмпла нет валидных текстовых токенов, полностью обнуляем вклад.
        if mask is not None:
            valid = mask.view(B, -1).bool().any(dim=1).view(B, 1, 1).to(out.dtype)
            out = out * valid

        return out


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class EdgeFlowPhaseBlockV2(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        use_cross: bool,
        ffn_hidden: int,
        num_phases: int = 0,
        expert_hidden_list: Optional[List[int]] = None,
    ):
        super().__init__()

        self.use_cross = use_cross

        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, heads, kv_heads)

        if use_cross:
            self.norm_cross = RMSNorm(dim)
            self.cross = CrossAttentionLite(dim, heads)

        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ffn_hidden)

        self.phase_experts = None

        if num_phases > 0 and expert_hidden_list is not None:
            assert len(expert_hidden_list) == num_phases, (
                "expert_hidden_list length must equal num_phases"
            )

            self.phase_experts = nn.ModuleList(
                [SwiGLU(dim, hidden) for hidden in expert_hidden_list]
            )

        # Разреженные эксперты считаются только для активных фаз.
        # Если torch.compile плохо работает с динамическими nonzero/index_put,
        # можно отключить: SPARSE_EXPERTS=0.
        self.sparse_experts = os.environ.get("SPARSE_EXPERTS", "1") == "1"

    def _apply_phase_experts(
        self,
        h: torch.Tensor,
        phase_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.phase_experts is None or phase_idx is None:
            return torch.zeros_like(h)

        # Плотный fallback.
        # Считает все эксперты для всех сэмплов, затем маскирует.
        # Тяжелее, но иногда стабильнее с torch.compile.
        if not self.sparse_experts:
            out = torch.zeros_like(h)

            for p, expert in enumerate(self.phase_experts):
                mask_p = (phase_idx == p).to(h.dtype).view(-1, 1, 1)
                out = out + expert(h) * mask_p

            return out

        # Разреженный routing:
        # каждый сэмпл использует только один эксперт.
        out = torch.zeros_like(h)

        for p, expert in enumerate(self.phase_experts):
            idx = torch.nonzero(phase_idx == p, as_tuple=True)[0]

            if idx.numel() == 0:
                continue

            h_selected = h.index_select(0, idx)
            expert_out = expert(h_selected)

            try:
                out = torch.index_put(out, (idx,), expert_out)
            except Exception:
                # Запасной вариант для старых PyTorch.
                out = out.clone()
                out[idx] = expert_out

        return out

    def forward(
        self,
        x: torch.Tensor,
        scale1: torch.Tensor,
        shift1: torch.Tensor,
        gate1: torch.Tensor,
        scale2: torch.Tensor,
        shift2: torch.Tensor,
        gate2: torch.Tensor,
        gate_exp: torch.Tensor,
        gate_cross: torch.Tensor,
        grid_h: int,
        grid_w: int,
        phase_idx: Optional[torch.Tensor] = None,
        text_k: Optional[torch.Tensor] = None,
        text_v: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = h * (1.0 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + self.attn(h, grid_h, grid_w) * gate1.unsqueeze(1)

        if self.use_cross:
            x = x + self.cross(
                self.norm_cross(x),
                k=text_k,
                v=text_v,
                mask=mask,
            ) * gate_cross.unsqueeze(1)

        h = self.norm2(x)
        h = h * (1.0 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.ff(h) * gate2.unsqueeze(1)

        if self.phase_experts is not None and phase_idx is not None:
            expert_out = self._apply_phase_experts(h, phase_idx)
            x = x + expert_out * gate_exp.unsqueeze(1)

        return x


class EdgeFlowPhaseV2(nn.Module):
    """
    Phase-DiT with hard phase experts.

    Default config is tuned for 256x256 and ~250M total trainable params
    together with the cached-text adapter:

        latent_channels=4
        patch_size=2
        dim=896
        depth=12
        heads=14
        kv_heads=2
        context_dim=896
        num_phases=4
        ffn_hidden=768
        ada_rank=48
        cross_every=3
        expert_hidden_list=[768, 1216, 1600, 2048]

    Expert share is high by design.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        patch_size: int = 2,
        dim: int = 768,
        depth: int = 8,
        heads: int = 12,
        kv_heads: int = 2,
        context_dim: int = 768,
        num_phases: int = 4,
        ffn_hidden: int = 1024,
        ada_rank: int = 64,
        cross_every: int = 2,
        expert_hidden_list: Optional[List[int]] = None,
        gate_init: float = 0.01,
    ):
        super().__init__()

        assert context_dim == dim, (
            f"EdgeFlowPhaseV2 requires context_dim == dim. "
            f"Got context_dim={context_dim}, dim={dim}."
        )

        if expert_hidden_list is None:
            expert_hidden_list = [768, 768, 768, 768]

        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_phases = num_phases
        self.use_phase = num_phases > 1
        self.heads = heads
        self.head_dim = dim // heads
        self.gradient_checkpointing = False

        patch_dim = latent_channels * patch_size * patch_size

        self.proj_in = nn.Conv2d(
            latent_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.time_embed = TimestepEmbedding(dim)
        self.time_norm = RMSNorm(dim)
        self.text_norm = RMSNorm(dim)

        if self.use_phase:
            self.phase_table = nn.Parameter(torch.randn(num_phases, dim) * 0.02)
            self.phase_norm = RMSNorm(dim)
            cond_in_dim = dim * 3
        else:
            self.phase_table = None
            cond_in_dim = dim * 2

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

        # 8 modulation slots:
        # scale1, shift1, gate1,
        # scale2, shift2, gate_ff,
        # gate_expert, gate_cross
        self.ada_proj = nn.Sequential(
            nn.Linear(dim, ada_rank, bias=False),
            nn.SiLU(),
            nn.Linear(ada_rank, depth * dim * 8, bias=False),
        )

        self.shared_cross_kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.shared_k_norm = RMSNorm(self.head_dim)
        self.shared_v_norm = RMSNorm(self.head_dim)

        self.blocks = nn.ModuleList(
            [
                EdgeFlowPhaseBlockV2(
                    dim=dim,
                    heads=heads,
                    kv_heads=kv_heads,
                    use_cross=((i + 1) % cross_every == 0),
                    ffn_hidden=ffn_hidden,
                    num_phases=num_phases if self.use_phase else 0,
                    expert_hidden_list=expert_hidden_list if self.use_phase else None,
                )
                for i in range(depth)
            ]
        )

        self.norm_final = RMSNorm(dim)
        self.proj_out = nn.Linear(dim, patch_dim, bias=False)

        self.register_buffer(
            "gate_init",
            torch.tensor(float(gate_init), dtype=torch.float32),
            persistent=False,
        )

        self.apply(self._init_weights)

        # Специальная нулевая инициализация после общей инициализации.
        nn.init.zeros_(self.ada_proj[-1].weight)
        nn.init.zeros_(self.proj_out.weight)

        for block in self.blocks:
            if block.phase_experts is not None:
                for expert in block.phase_experts:
                    nn.init.zeros_(expert.w3.weight)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)

    def encode_text(self, cond_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.shared_cross_kv.weight.dtype
        if cond_seq.dtype != target_dtype:
            cond_seq = cond_seq.to(target_dtype)

        B, M, _ = cond_seq.shape

        kv = self.shared_cross_kv(cond_seq).view(B, M, 2, self.heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        k = self.shared_k_norm(k).transpose(1, 2).contiguous()
        v = self.shared_v_norm(v).transpose(1, 2).contiguous()

        return k, v

    def forward(
        self,
        z: torch.Tensor,
        cond_pooled: torch.Tensor,
        cond_seq: Optional[torch.Tensor],
        t_phase: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        text_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        target_dtype = self.proj_in.weight.dtype
        if z.dtype != target_dtype:
            z = z.to(target_dtype)
        if cond_pooled.dtype != target_dtype:
            cond_pooled = cond_pooled.to(target_dtype)
        if cond_seq is not None and cond_seq.dtype != target_dtype:
            cond_seq = cond_seq.to(target_dtype)
        if t_phase.dtype != target_dtype:
            t_phase = t_phase.to(target_dtype)

        B, C, H, W = z.shape

        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Latent H/W ({H}, {W}) must be divisible by patch_size {self.patch_size}"
        )

        p = self.patch_size
        Hp = H // p
        Wp = W // p

        x = self.proj_in(z)
        x = x.flatten(2).transpose(1, 2)

        time_emb = self.time_norm(self.time_embed(t_phase))
        text_emb = self.text_norm(cond_pooled)

        phase_idx = None

        if self.use_phase:
            t01 = t_phase.flatten().float().clamp(0.0, 1.0)

            # Интерполяция phase embedding по таблице фаз.
            u = t01 * float(self.num_phases - 1)
            idx_low = u.floor().long().clamp(0, self.num_phases - 2)
            idx_high = idx_low + 1
            alpha = (u - idx_low.float()).unsqueeze(-1)

            # smoothstep s-curve
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)

            phase_emb = (
                1.0 - alpha
            ) * self.phase_table[idx_low] + alpha * self.phase_table[idx_high]

            phase_emb = self.phase_norm(phase_emb)
            phase_emb = phase_emb.to(time_emb.dtype)

            # Жёсткий routing экспертов по равным интервалам t.
            # Для num_phases=4:
            #   [0.00, 0.25) -> expert 0
            #   [0.25, 0.50) -> expert 1
            #   [0.50, 0.75) -> expert 2
            #   [0.75, 1.00] -> expert 3
            phase_idx = (t01 * self.num_phases).floor().long().clamp(
                0, self.num_phases - 1
            )

            cond_combined = torch.cat([time_emb, text_emb, phase_emb], dim=-1)
        else:
            cond_combined = torch.cat([time_emb, text_emb], dim=-1)

        cond_vec = self.cond_mlp(cond_combined)
        ada = self.ada_proj(cond_vec)
        ada = ada.view(B, self.depth, 8, self.dim)

        scales1 = ada[:, :, 0]
        shifts1 = ada[:, :, 1]
        gates1 = ada[:, :, 2]

        scales2 = ada[:, :, 3]
        shifts2 = ada[:, :, 4]
        gates2 = ada[:, :, 5]

        gates_exp = ada[:, :, 6]
        gates_cross = ada[:, :, 7]

        gi = self.gate_init.to(dtype=gates1.dtype)

        gates1 = gates1 + gi
        gates2 = gates2 + gi
        gates_cross = gates_cross + gi

        # Expert output zero-initialized, поэтому expert gate можно держать около 1.
        gates_exp = gates_exp + 1.0

        if text_kvs is None:
            if cond_seq is None:
                raise ValueError("text_kvs is None and cond_seq is None")
            text_k, text_v = self.encode_text(cond_seq)
        else:
            text_k, text_v = text_kvs

        for i, block in enumerate(self.blocks):
            k = text_k if block.use_cross else None
            v = text_v if block.use_cross else None

            scale1_val = scales1[:, i]
            shift1_val = shifts1[:, i]
            gate1_val = gates1[:, i]

            scale2_val = scales2[:, i]
            shift2_val = shifts2[:, i]
            gate2_val = gates2[:, i]

            gate_exp_val = gates_exp[:, i]
            gate_cross_val = gates_cross[:, i]

            if self.training and self.gradient_checkpointing:
                x = cp.checkpoint(
                    block,
                    x,
                    scale1_val,
                    shift1_val,
                    gate1_val,
                    scale2_val,
                    shift2_val,
                    gate2_val,
                    gate_exp_val,
                    gate_cross_val,
                    Hp,
                    Wp,
                    phase_idx,
                    k,
                    v,
                    mask,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    scale1=scale1_val,
                    shift1=shift1_val,
                    gate1=gate1_val,
                    scale2=scale2_val,
                    shift2=shift2_val,
                    gate2=gate2_val,
                    gate_exp=gate_exp_val,
                    gate_cross=gate_cross_val,
                    grid_h=Hp,
                    grid_w=Wp,
                    phase_idx=phase_idx,
                    text_k=k,
                    text_v=v,
                    mask=mask,
                )

        x = self.norm_final(x)
        x = self.proj_out(x)

        patch_dim = self.latent_channels * self.patch_size * self.patch_size
        x = x.transpose(1, 2).contiguous().reshape(B, patch_dim, Hp, Wp)
        x = F.pixel_shuffle(x, self.patch_size)

        return x