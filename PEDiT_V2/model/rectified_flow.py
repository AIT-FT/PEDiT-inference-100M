import os
import time
import logging
from typing import Any, Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

RF_CLAMP_RANGE = float(os.environ.get("RF_CLAMP_RANGE", "10.0"))
RF_T_EPS = float(os.environ.get("RF_T_EPS", "0.0"))


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
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

def _is_fsdp_wrapped(model: torch.nn.Module) -> bool:
    m = model
    visited = set()
    while id(m) not in visited:
        visited.add(id(m))
        if hasattr(m, "_fsdp_wrapped_module") or type(m).__name__ == "FullyShardedDataParallel":
            return True
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif hasattr(m, "module"):
            m = m.module
        else:
            break
    return False

class RectifiedFlow4D:

    @staticmethod
    def _steps_to_per_phase(steps: int) -> Tuple[int, int, int]:
        if steps <= 3:
            return (1, 1, max(1, steps - 2)) if steps == 3 else (steps, 0, 0)
        if steps == 9:
            return (3, 3, 3)
        if steps == 10:
            return (3, 4, 3)
        if steps == 11:
            return (4, 4, 3)
        if steps == 12:
            return (4, 4, 4)
        if steps == 14:
            return (5, 6, 3)
        if steps == 15:
            return (6, 6, 3)
        if steps == 28:
            return (9, 10, 9)
        base = steps // 3
        rem = steps % 3
        return (
            base + (1 if rem > 0 else 0),
            base + (1 if rem > 1 else 0),
            base,
        )

    @staticmethod
    def _time_schedule(steps: int, shift: float, device: torch.device) -> torch.Tensor:
        calc_device = (
            torch.device("cpu")
            if ((isinstance(device, torch.device) and device.type == "meta") or str(device) == "meta")
            else device
        )
        if steps < 3:
            t = torch.linspace(0.0, 1.0, steps + 1, device=calc_device, dtype=torch.float32)
            if abs(shift - 1.0) > 1e-6:
                mu = max(float(shift), 1e-4)
                t = t / (mu - (mu - 1.0) * t)
        else:
            steps_per_phase = RectifiedFlow4D._steps_to_per_phase(steps)
            boundaries = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
            mu_base = max(float(shift), 1e-4)
            phase_shifts = (
                mu_base,
                1.0 + 0.45 * (mu_base - 1.0),
                1.0 + 0.15 * (mu_base - 1.0),
            )
            segments = []
            for p_idx, n_s in enumerate(steps_per_phase):
                if n_s <= 0:
                    continue
                t_start, t_end = boundaries[p_idx], boundaries[p_idx + 1]
                u = torch.linspace(0.0, 1.0, n_s + 1, device=calc_device, dtype=torch.float32)
                mu = phase_shifts[p_idx]
                if abs(mu - 1.0) > 1e-6:
                    u = u / (mu - (mu - 1.0) * u)
                seg = t_start + (t_end - t_start) * u
                segments.append(seg[:-1] if p_idx < 2 else seg)
            t = torch.cat(segments, dim=0)

        if RF_T_EPS > 0.0 and t.numel() > 2:
            t = t.clone()
            t[1:-1] = t[1:-1].clamp(RF_T_EPS, 1.0 - RF_T_EPS)
            t[0] = 0.0
            t[-1] = 1.0

        return t

    @staticmethod
    def _prepare_sampling_conditioning(
        model: torch.nn.Module,
        model_api: torch.nn.Module,
        B: int,
        cfg_scale: float,
        cond_pooled: torch.Tensor,
        cond_seq: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        text_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]],
        uncond_pooled: Optional[torch.Tensor],
        uncond_seq: Optional[torch.Tensor],
        uncond_mask: Optional[torch.Tensor],
    ):
        do_cfg = cfg_scale > 1.0
        has_null_emb = hasattr(model_api, "pooler") and hasattr(model_api.pooler, "null_embedding")

        if do_cfg:
            if cond_pooled.shape[0] == 2 * B:
                c_pooled = cond_pooled
                c_seq = cond_seq
                c_mask = mask
            elif uncond_pooled is not None or has_null_emb:
                if uncond_pooled is None:
                    u_pool = model_api.pooler.null_embedding.view(1, -1).expand(B, -1).to(
                        device=cond_pooled.device, dtype=cond_pooled.dtype
                    )
                else:
                    u_pool = uncond_pooled
                c_pooled = torch.cat([u_pool, cond_pooled], dim=0)
                if uncond_seq is not None and cond_seq is not None:
                    c_seq = torch.cat([uncond_seq, cond_seq], dim=0)
                else:
                    c_seq = cond_seq
                if uncond_mask is not None and mask is not None and c_seq is not None and c_seq.shape[0] == 2 * B:
                    c_mask = torch.cat([uncond_mask, mask], dim=0)
                else:
                    c_mask = mask
            else:
                logger.warning(
                    "cfg_scale > 1.0, but no unconditional prompt or null_embedding available. Running without CFG."
                )
                do_cfg = False
                c_pooled = cond_pooled
                c_seq = cond_seq
                c_mask = mask
        else:
            if cond_pooled.shape[0] == 2 * B:
                c_pooled = cond_pooled[B:]
                c_seq = cond_seq[B:] if (cond_seq is not None and cond_seq.shape[0] == 2 * B) else cond_seq
                c_mask = mask[B:] if (mask is not None and mask.shape[0] == 2 * B) else mask
                if text_kvs is not None and text_kvs[0].shape[0] == 2 * B:
                    text_kvs = (text_kvs[0][B:], text_kvs[1][B:])
            else:
                c_pooled = cond_pooled
                c_seq = cond_seq
                c_mask = mask

        if do_cfg:
            if has_null_emb and c_pooled.shape[0] == 2 * B:
                null_emb = model_api.pooler.null_embedding.view(1, -1).expand(B, -1).to(
                    device=c_pooled.device, dtype=c_pooled.dtype
                )
                c_pooled = torch.cat([null_emb, c_pooled[B:]], dim=0)
            if c_seq is not None and c_seq.shape[0] == 2 * B:
                c_seq = c_seq[B:]
            if c_mask is not None and c_mask.shape[0] == 2 * B:
                c_mask = c_mask[B:]
            if text_kvs is not None and text_kvs[0].shape[0] == 2 * B:
                text_kvs = (text_kvs[0][B:], text_kvs[1][B:])

        if c_mask is not None and c_mask.shape[0] not in (B, c_pooled.shape[0]):
            raise ValueError(
                f"mask batch size {c_mask.shape[0]} does not match conditioning batch size {c_pooled.shape[0]}"
            )

        if c_mask is not None:
            seq_cap = c_mask.shape[-1]
            active_any = (c_mask != 0).any(dim=0)
            if active_any.any():
                max_pos = int(torch.nonzero(active_any, as_tuple=False)[-1].item()) + 1
            else:
                max_pos = 16
            real_len = min(seq_cap, max(16, ((max_pos + 7) // 8) * 8))

            if c_seq is not None:
                if c_seq.dim() == 4:
                    real_len = min(real_len, c_seq.shape[2])
                    if c_seq.shape[2] > real_len:
                        c_seq = c_seq[:, :, :real_len].contiguous()
                else:
                    real_len = min(real_len, c_seq.shape[1])
                    if c_seq.shape[1] > real_len:
                        c_seq = c_seq[:, :real_len].contiguous()
            elif text_kvs is not None:
                real_len = min(real_len, text_kvs[0].shape[2])

            if c_mask.shape[1] > real_len:
                c_mask = c_mask[:, :real_len].contiguous()
            if text_kvs is not None and text_kvs[0].shape[2] > real_len:
                text_kvs = (
                    text_kvs[0][:, :, :real_len, :].contiguous(),
                    text_kvs[1][:, :, :real_len, :].contiguous(),
                )

        if text_kvs is None and c_seq is None:
            raise ValueError("Both text_kvs and cond_seq are None.")

        return do_cfg, c_pooled, c_seq, c_mask, text_kvs

    @staticmethod
    def _resolve_phase_idx(step: int, steps: int, t0: float, num_phases: int) -> Tuple[int, bool]:
        if num_phases == 3 and steps >= 3:
            s0, s1, s2 = RectifiedFlow4D._steps_to_per_phase(steps)
            if step < s0:
                return 0, (step == s0 - 1)
            elif step < s0 + s1:
                return 1, (step == s0 + s1 - 1)
            else:
                return 2, (step == steps - 1)
        p_idx = min(int(t0 * float(num_phases)), num_phases - 1)
        return p_idx, (step == steps - 1)

    def add_noise(
        self,
        x_1: torch.Tensor,
        t_phase: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_dtype = x_1.dtype
        x_1_f = x_1.detach().float()

        if generator is not None and generator.device.type != x_1.device.type:
            x_0_f = torch.randn(
                x_1.shape,
                device=generator.device,
                dtype=torch.float32,
                generator=generator,
            ).to(device=x_1.device, non_blocking=True)
        else:
            x_0_f = torch.randn(
                x_1.shape,
                device=x_1.device,
                dtype=torch.float32,
                generator=generator,
            )

        t = t_phase.to(device=x_1.device, dtype=torch.float32).reshape(
            -1, *([1] * (x_1.dim() - 1))
        )

        x_t = (t * x_1_f + (1.0 - t) * x_0_f).to(dtype=orig_dtype)
        target_v = (x_1_f - x_0_f).to(dtype=orig_dtype)
        x_0 = x_0_f.to(dtype=orig_dtype)

        return x_t, x_0, target_v

    @torch.inference_mode()
    def phase_euler_sample(
        self,
        model: torch.nn.Module,
        cond_pooled: torch.Tensor,
        cond_seq: Optional[torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        steps: int = 24,
        mask: Optional[torch.Tensor] = None,
        text_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 0.5,
        shift: float = 1.0,
        rescale_norm: bool = False,
        return_history: bool = False,
        uncond_pooled: Optional[torch.Tensor] = None,
        uncond_seq: Optional[torch.Tensor] = None,
        uncond_mask: Optional[torch.Tensor] = None,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, torch.Tensor, int], Any]] = None,
        attn_temperature: Optional[float] = None,
        **kwargs,
    ) -> Any:
        model.eval()

        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")

        if isinstance(device, str):
            device = torch.device(device)

        B = shape[0]

        if initial_noise is not None:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError(
                    f"initial_noise shape {tuple(initial_noise.shape)} != shape {tuple(shape)}"
                )
            x_t = initial_noise.clone().to(device=device, dtype=torch.float32)
        else:
            if generator is not None and generator.device.type != device.type:
                x_t = torch.randn(
                    shape,
                    device=generator.device,
                    dtype=torch.float32,
                    generator=generator,
                ).to(device)
            else:
                x_t = torch.randn(
                    shape,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )

        times = self._time_schedule(steps=steps, shift=shift, device=device)
        model_api = _unwrap_model(model)
        is_fsdp = _is_fsdp_wrapped(model)
        num_phases = getattr(model_api, "num_phases", 3)

        do_classifier_free_guidance, c_pooled, c_seq, c_mask, text_kvs = self._prepare_sampling_conditioning(
            model=model,
            model_api=model_api,
            B=B,
            cfg_scale=cfg_scale,
            cond_pooled=cond_pooled,
            cond_seq=cond_seq,
            mask=mask,
            text_kvs=text_kvs,
            uncond_pooled=uncond_pooled,
            uncond_seq=uncond_seq,
            uncond_mask=uncond_mask,
        )

        per_block_kvs = None

        history = [x_t.detach().cpu().half()] if return_history else None
        step_times = [] if return_history else None
        v_prev = None
        prev_phase = -1
        if hasattr(model_api, "stream_phase_expert"):
            model_api.stream_phase_expert(0)
        if hasattr(model_api, "prefetch_next_phase") and num_phases > 1:
            model_api.prefetch_next_phase(1)

        sync_per_step = (os.environ.get("SYNC_PER_STEP", "0") == "1")

        for step in range(steps):
            need_timing = return_history or (callback is not None and sync_per_step)

            if (
                need_timing
                and torch.cuda.is_available()
                and x_t.device.type == "cuda"
            ):
                torch.cuda.synchronize()

            step_start = time.perf_counter()

            t0 = float(times[step].item())
            t1 = float(times[step + 1].item())
            dt = t1 - t0

            phase_idx, is_last_in_phase = self._resolve_phase_idx(step, steps, t0, num_phases)

            if phase_idx != prev_phase:
                v_prev = None
                prev_phase = phase_idx
                if hasattr(model_api, "stream_phase_expert"):
                    model_api.stream_phase_expert(phase_idx)
                if phase_idx + 1 < num_phases and hasattr(model_api, "prefetch_next_phase"):
                    model_api.prefetch_next_phase(phase_idx + 1)
                if not is_fsdp:
                    if c_seq is not None and hasattr(model_api, "prepare_phase_cross_kvs"):
                        per_block_kvs = model_api.prepare_phase_cross_kvs(c_seq, phase_idx)
                        text_kvs = None
                    elif c_seq is not None and hasattr(model_api, "encode_text"):
                        text_kvs = model_api.encode_text(c_seq, phase_idx=phase_idx, apply_k_norm=False)
                        per_block_kvs = None
            elif step == 1 and num_phases > 2 and hasattr(model_api, "prefetch_next_phase"):
                model_api.prefetch_next_phase(2)

            phase_temps = (0.90, 0.80, 0.70)
            if attn_temperature is not None:
                temp = float(attn_temperature)
            elif not do_classifier_free_guidance and 0 <= phase_idx < len(phase_temps):
                temp = phase_temps[phase_idx]
            else:
                temp = 1.0

            if do_classifier_free_guidance:
                latent_model_input = torch.cat([x_t, x_t], dim=0)
                t_phase = torch.full(
                    (2 * B,),
                    t0,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                latent_model_input = x_t
                t_phase = torch.full(
                    (B,),
                    t0,
                    device=device,
                    dtype=torch.float32,
                )

            v_pred = model(
                z=latent_model_input,
                cond_pooled=c_pooled,
                cond_seq=c_seq if per_block_kvs is None else None,
                t_phase=t_phase,
                mask=c_mask,
                text_kvs=text_kvs,
                per_block_text_kvs=per_block_kvs,
                temperature=temp,
                phase_idx=phase_idx,
            )

            v_pred = v_pred.float()
            if do_classifier_free_guidance:
                v_uncond, v_text = v_pred.chunk(2, dim=0)
                v_cfg = v_uncond + cfg_scale * (v_text - v_uncond)

                if cfg_rescale > 0.0:
                    dims = (1, 2, 3) if v_text.dim() == 4 else tuple(range(1, v_text.dim()))

                    std_text = v_text.std(dim=dims, keepdim=True, unbiased=False)
                    std_cfg = v_cfg.std(dim=dims, keepdim=True, unbiased=False)

                    factor = std_text / (std_cfg + 1e-6)
                    factor = factor.clamp(0.5, 2.0)

                    v_rescaled = v_cfg * factor
                    v_pred = (1.0 - cfg_rescale) * v_cfg + cfg_rescale * v_rescaled
                else:
                    v_pred = v_cfg

            if rescale_norm:
                target_rms = 1.4142
                dims = (1, 2, 3) if v_pred.dim() == 4 else tuple(range(1, v_pred.dim()))
                v_rms = torch.sqrt(torch.mean(v_pred * v_pred, dim=dims, keepdim=True) + 1e-8)
                scale_factor = torch.clamp(target_rms / v_rms, min=0.8, max=1.05)
                v_pred = v_pred * scale_factor

            if not do_classifier_free_guidance and v_prev is not None:
                v_boost = v_pred + 0.20 * (v_pred - v_prev)
            else:
                v_boost = v_pred

            x_t = x_t.add(v_boost, alpha=dt)
            v_prev = v_pred

            if not torch.isfinite(x_t).all():
                logger.warning("NaN/Inf detected in phase_euler_sample, applying nan_to_num")
                x_t = torch.nan_to_num(
                    x_t,
                    nan=0.0,
                    posinf=RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else 10.0,
                    neginf=-RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else -10.0,
                )

            if RF_CLAMP_RANGE > 0.0:
                x_t = x_t.clamp(-RF_CLAMP_RANGE, RF_CLAMP_RANGE)

            if return_history or callback is not None:
                x_1_pred = x_t + (1.0 - t1) * v_boost
                x_1_pred = torch.nan_to_num(
                    x_1_pred,
                    nan=0.0,
                    posinf=RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else 10.0,
                    neginf=-RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else -10.0,
                )
                if RF_CLAMP_RANGE > 0.0:
                    x_1_pred = x_1_pred.clamp(-RF_CLAMP_RANGE, RF_CLAMP_RANGE)

            if return_history:
                history.append(x_1_pred.detach().cpu().half())
                step_times.append(round((time.perf_counter() - step_start) * 1000))

            if callback is not None:
                callback(step, x_1_pred, round((time.perf_counter() - step_start) * 1000))

        if not do_classifier_free_guidance and kwargs.get("restore_contrast", False):
            dims = (-3, -2, -1) if x_t.dim() == 4 else -1
            std = x_t.std(dim=dims, keepdim=True)
            mean = x_t.mean(dim=dims, keepdim=True)
            x_t = (x_t - mean) / (std + 1e-6) * 0.95 + mean

        if return_history:
            return x_t, history, step_times

        return x_t

    @torch.inference_mode()
    def phase_heun_sample(
        self,
        model: torch.nn.Module,
        cond_pooled: torch.Tensor,
        cond_seq: Optional[torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        steps: int = 24,
        mask: Optional[torch.Tensor] = None,
        text_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 0.5,
        shift: float = 1.0,
        rescale_norm: bool = False,
        return_history: bool = False,
        uncond_pooled: Optional[torch.Tensor] = None,
        uncond_seq: Optional[torch.Tensor] = None,
        uncond_mask: Optional[torch.Tensor] = None,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, torch.Tensor, int], Any]] = None,
        attn_temperature: Optional[float] = None,
        **kwargs,
    ) -> Any:
        model.eval()
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")
        if isinstance(device, str):
            device = torch.device(device)

        B = shape[0]
        if initial_noise is not None:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError(f"initial_noise shape {tuple(initial_noise.shape)} != shape {tuple(shape)}")
            x_t = initial_noise.clone().to(device=device, dtype=torch.float32)
        elif generator is not None and generator.device.type != device.type:
            x_t = torch.randn(shape, device=generator.device, dtype=torch.float32, generator=generator).to(device)
        else:
            x_t = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)

        times = self._time_schedule(steps=steps, shift=shift, device=device)
        model_api = _unwrap_model(model)
        is_fsdp = _is_fsdp_wrapped(model)
        num_phases = getattr(model_api, "num_phases", 3)

        do_classifier_free_guidance, c_pooled, c_seq, c_mask, text_kvs = self._prepare_sampling_conditioning(
            model=model,
            model_api=model_api,
            B=B,
            cfg_scale=cfg_scale,
            cond_pooled=cond_pooled,
            cond_seq=cond_seq,
            mask=mask,
            text_kvs=text_kvs,
            uncond_pooled=uncond_pooled,
            uncond_seq=uncond_seq,
            uncond_mask=uncond_mask,
        )

        per_block_kvs = None
        history = [x_t.detach().cpu().half()] if return_history else None
        step_times = [] if return_history else None
        v_prev = None
        prev_phase = -1

        if hasattr(model_api, "stream_phase_expert"):
            model_api.stream_phase_expert(0)
        if hasattr(model_api, "prefetch_next_phase") and num_phases > 1:
            model_api.prefetch_next_phase(1)

        phase_temps = (0.90, 0.80, 0.70)
        sync_per_step = (os.environ.get("SYNC_PER_STEP", "0") == "1")

        for step in range(steps):
            need_timing = return_history or (callback is not None and sync_per_step)
            if need_timing and torch.cuda.is_available() and x_t.device.type == "cuda":
                torch.cuda.synchronize()

            step_start = time.perf_counter()
            t0 = float(times[step].item())
            t1 = float(times[step + 1].item())
            dt = t1 - t0

            phase_idx, is_last_in_phase = self._resolve_phase_idx(step, steps, t0, num_phases)

            if phase_idx != prev_phase:
                v_prev = None
                prev_phase = phase_idx
                if hasattr(model_api, "stream_phase_expert"):
                    model_api.stream_phase_expert(phase_idx)
                if phase_idx + 1 < num_phases and hasattr(model_api, "prefetch_next_phase"):
                    model_api.prefetch_next_phase(phase_idx + 1)
                if not is_fsdp:
                    if c_seq is not None and hasattr(model_api, "prepare_phase_cross_kvs"):
                        per_block_kvs = model_api.prepare_phase_cross_kvs(c_seq, phase_idx)
                        text_kvs = None
                    elif c_seq is not None and hasattr(model_api, "encode_text"):
                        text_kvs = model_api.encode_text(c_seq, phase_idx=phase_idx, apply_k_norm=False)
                        per_block_kvs = None
            elif step == 1 and num_phases > 2 and hasattr(model_api, "prefetch_next_phase"):
                model_api.prefetch_next_phase(2)

            if attn_temperature is not None:
                temp = float(attn_temperature)
            elif not do_classifier_free_guidance and 0 <= phase_idx < len(phase_temps):
                temp = phase_temps[phase_idx]
            else:
                temp = 1.0

            if do_classifier_free_guidance:
                latent_model_input = torch.cat([x_t, x_t], dim=0)
                t_phase = torch.full((2 * B,), t0, device=device, dtype=torch.float32)
            else:
                latent_model_input = x_t
                t_phase = torch.full((B,), t0, device=device, dtype=torch.float32)

            v0 = model(
                z=latent_model_input,
                cond_pooled=c_pooled,
                cond_seq=c_seq if per_block_kvs is None else None,
                t_phase=t_phase,
                mask=c_mask,
                text_kvs=text_kvs,
                per_block_text_kvs=per_block_kvs,
                temperature=temp,
                phase_idx=phase_idx,
            ).float()

            if do_classifier_free_guidance:
                v_uncond, v_text = v0.chunk(2, dim=0)
                v_cfg = v_uncond + cfg_scale * (v_text - v_uncond)
                if cfg_rescale > 0.0:
                    dims = (1, 2, 3) if v_text.dim() == 4 else tuple(range(1, v_text.dim()))
                    std_text = v_text.std(dim=dims, keepdim=True, unbiased=False)
                    std_cfg = v_cfg.std(dim=dims, keepdim=True, unbiased=False)
                    factor = (std_text / (std_cfg + 1e-6)).clamp(0.5, 2.0)
                    v_rescaled = v_cfg * factor
                    v0 = (1.0 - cfg_rescale) * v_cfg + cfg_rescale * v_rescaled
                else:
                    v0 = v_cfg

            if rescale_norm:
                target_rms = 1.4142
                dims = (1, 2, 3) if v0.dim() == 4 else tuple(range(1, v0.dim()))
                v_rms = torch.sqrt(torch.mean(v0 * v0, dim=dims, keepdim=True) + 1e-8)
                v0 = v0 * torch.clamp(target_rms / v_rms, min=0.8, max=1.05)

            if not do_classifier_free_guidance and v_prev is not None:
                v0_boost = v0 + 0.20 * (v0 - v_prev)
            else:
                v0_boost = v0

            x_pred = x_t.add(v0_boost, alpha=dt)

            phase_upper = float(phase_idx + 1) / float(num_phases)
            t1_eff = min(t1, phase_upper - 1e-5) if (is_last_in_phase and phase_idx < num_phases - 1) else t1

            if do_classifier_free_guidance:
                latent_model_input_1 = torch.cat([x_pred, x_pred], dim=0)
                t_phase_1 = torch.full((2 * B,), t1_eff, device=device, dtype=torch.float32)
            else:
                latent_model_input_1 = x_pred
                t_phase_1 = torch.full((B,), t1_eff, device=device, dtype=torch.float32)

            v1 = model(
                z=latent_model_input_1,
                cond_pooled=c_pooled,
                cond_seq=c_seq if per_block_kvs is None else None,
                t_phase=t_phase_1,
                mask=c_mask,
                text_kvs=text_kvs,
                per_block_text_kvs=per_block_kvs,
                temperature=temp,
                phase_idx=phase_idx,
            ).float()

            if do_classifier_free_guidance:
                v_uncond_1, v_text_1 = v1.chunk(2, dim=0)
                v_cfg_1 = v_uncond_1 + cfg_scale * (v_text_1 - v_uncond_1)
                if cfg_rescale > 0.0:
                    dims = (1, 2, 3) if v_text_1.dim() == 4 else tuple(range(1, v_text_1.dim()))
                    std_text_1 = v_text_1.std(dim=dims, keepdim=True, unbiased=False)
                    std_cfg_1 = v_cfg_1.std(dim=dims, keepdim=True, unbiased=False)
                    factor_1 = (std_text_1 / (std_cfg_1 + 1e-6)).clamp(0.5, 2.0)
                    v_rescaled_1 = v_cfg_1 * factor_1
                    v1 = (1.0 - cfg_rescale) * v_cfg_1 + cfg_rescale * v_rescaled_1
                else:
                    v1 = v_cfg_1

            if rescale_norm:
                target_rms = 1.4142
                dims = (1, 2, 3) if v1.dim() == 4 else tuple(range(1, v1.dim()))
                v_rms = torch.sqrt(torch.mean(v1 * v1, dim=dims, keepdim=True) + 1e-8)
                v1 = v1 * torch.clamp(target_rms / v_rms, min=0.8, max=1.05)

            v_avg = 0.5 * (v0_boost + v1)
            x_t = x_t.add(v_avg, alpha=dt)
            v_prev = v0

            if not torch.isfinite(x_t).all():
                logger.warning("NaN/Inf detected in phase_heun_sample, applying nan_to_num")
                x_t = torch.nan_to_num(
                    x_t,
                    nan=0.0,
                    posinf=RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else 10.0,
                    neginf=-RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else -10.0,
                )

            if RF_CLAMP_RANGE > 0.0:
                x_t = x_t.clamp(-RF_CLAMP_RANGE, RF_CLAMP_RANGE)

            if return_history or callback is not None:
                x_1_pred = x_t + (1.0 - t1) * v_avg
                x_1_pred = torch.nan_to_num(
                    x_1_pred,
                    nan=0.0,
                    posinf=RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else 10.0,
                    neginf=-RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else -10.0,
                )
                if RF_CLAMP_RANGE > 0.0:
                    x_1_pred = x_1_pred.clamp(-RF_CLAMP_RANGE, RF_CLAMP_RANGE)

            if return_history:
                history.append(x_1_pred.detach().cpu().half())
                step_times.append(round((time.perf_counter() - step_start) * 1000))

            if callback is not None:
                callback(step, x_1_pred, round((time.perf_counter() - step_start) * 1000))

        if not do_classifier_free_guidance and kwargs.get("restore_contrast", False):
            dims = (-3, -2, -1) if x_t.dim() == 4 else -1
            std = x_t.std(dim=dims, keepdim=True)
            mean = x_t.mean(dim=dims, keepdim=True)
            x_t = (x_t - mean) / (std + 1e-6) * 0.95 + mean

        if return_history:
            return x_t, history, step_times
        return x_t

    @torch.inference_mode()
    def phase_multistep_sample(
        self,
        model: torch.nn.Module,
        cond_pooled: torch.Tensor,
        cond_seq: Optional[torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        steps: int = 24,
        mask: Optional[torch.Tensor] = None,
        text_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 0.0,
        uncond_pooled: Optional[torch.Tensor] = None,
        uncond_seq: Optional[torch.Tensor] = None,
        uncond_mask: Optional[torch.Tensor] = None,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, torch.Tensor, int], Any]] = None,
        return_history: bool = False,
        shift: float = 1.0,
        attn_temperature: Optional[float] = None,
        num_phases: int = 3,
        rescale_norm: bool = False,
        **kwargs,
    ) -> Any:
        model.eval()
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")
        if isinstance(device, str):
            device = torch.device(device)

        B = shape[0]
        if initial_noise is not None:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError(f"initial_noise shape {tuple(initial_noise.shape)} != shape {tuple(shape)}")
            x_t = initial_noise.clone().to(device=device, dtype=torch.float32)
        elif generator is not None and generator.device.type != device.type:
            x_t = torch.randn(shape, device=generator.device, dtype=torch.float32, generator=generator).to(device)
        else:
            x_t = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)

        times = self._time_schedule(steps=steps, shift=shift, device=device)
        model_api = _unwrap_model(model)
        is_fsdp = _is_fsdp_wrapped(model)
        num_phases = getattr(model_api, "num_phases", num_phases)

        do_classifier_free_guidance, c_pooled, c_seq, c_mask, text_kvs = self._prepare_sampling_conditioning(
            model=model,
            model_api=model_api,
            B=B,
            cfg_scale=cfg_scale,
            cond_pooled=cond_pooled,
            cond_seq=cond_seq,
            mask=mask,
            text_kvs=text_kvs,
            uncond_pooled=uncond_pooled,
            uncond_seq=uncond_seq,
            uncond_mask=uncond_mask,
        )

        per_block_kvs = None
        history = [x_t.detach().cpu().half()] if return_history else None
        step_times = [] if return_history else None

        v_prev: Optional[torch.Tensor] = None
        prev_dt: Optional[float] = None
        prev_phase = -1

        if hasattr(model_api, "stream_phase_expert"):
            model_api.stream_phase_expert(0)
        if hasattr(model_api, "prefetch_next_phase") and num_phases > 1:
            model_api.prefetch_next_phase(1)

        phase_temps = (0.90, 0.80, 0.70)
        sync_per_step = (os.environ.get("SYNC_PER_STEP", "0") == "1")
        total_steps = len(times) - 1

        for step in range(total_steps):
            need_timing = return_history or (callback is not None and sync_per_step)
            if need_timing and torch.cuda.is_available() and x_t.device.type == "cuda":
                torch.cuda.synchronize()

            step_start = time.perf_counter()
            t0 = float(times[step].item())
            t1 = float(times[step + 1].item())
            dt = t1 - t0

            phase_idx, is_last_in_phase = self._resolve_phase_idx(step, total_steps, t0, num_phases)

            if phase_idx != prev_phase:
                v_prev = None
                prev_dt = None
                prev_phase = phase_idx
                if hasattr(model_api, "stream_phase_expert"):
                    model_api.stream_phase_expert(phase_idx)
                if phase_idx + 1 < num_phases and hasattr(model_api, "prefetch_next_phase"):
                    model_api.prefetch_next_phase(phase_idx + 1)
                if not is_fsdp:
                    if c_seq is not None and hasattr(model_api, "prepare_phase_cross_kvs"):
                        per_block_kvs = model_api.prepare_phase_cross_kvs(c_seq, phase_idx)
                        text_kvs = None
                    elif c_seq is not None and hasattr(model_api, "encode_text"):
                        text_kvs = model_api.encode_text(c_seq, phase_idx=phase_idx, apply_k_norm=False)
                        per_block_kvs = None
            elif step == 1 and num_phases > 2 and hasattr(model_api, "prefetch_next_phase"):
                model_api.prefetch_next_phase(2)

            if attn_temperature is not None:
                temp = float(attn_temperature)
            elif not do_classifier_free_guidance and 0 <= phase_idx < len(phase_temps):
                temp = phase_temps[phase_idx]
            else:
                temp = 1.0

            if do_classifier_free_guidance:
                latent_input = torch.cat([x_t, x_t], dim=0)
                t_tensor = torch.full((2 * B,), t0, device=device, dtype=torch.float32)
            else:
                latent_input = x_t
                t_tensor = torch.full((B,), t0, device=device, dtype=torch.float32)

            v_pred = model(
                z=latent_input,
                cond_pooled=c_pooled,
                cond_seq=c_seq if per_block_kvs is None else None,
                t_phase=t_tensor,
                mask=c_mask,
                text_kvs=text_kvs,
                per_block_text_kvs=per_block_kvs,
                temperature=temp,
                phase_idx=phase_idx,
            ).float()

            if do_classifier_free_guidance:
                v_uncond, v_text = v_pred.chunk(2, dim=0)
                v_cfg = v_uncond + cfg_scale * (v_text - v_uncond)
                if cfg_rescale > 0.0:
                    dims = (1, 2, 3) if v_text.dim() == 4 else tuple(range(1, v_text.dim()))
                    std_text = v_text.std(dim=dims, keepdim=True, unbiased=False)
                    std_cfg = v_cfg.std(dim=dims, keepdim=True, unbiased=False)
                    factor = (std_text / (std_cfg + 1e-6)).clamp(0.5, 2.0)
                    v_rescaled = v_cfg * factor
                    v_pred = (1.0 - cfg_rescale) * v_cfg + cfg_rescale * v_rescaled
                else:
                    v_pred = v_cfg

            if rescale_norm:
                target_rms = 1.4142
                dims = (1, 2, 3) if v_pred.dim() == 4 else tuple(range(1, v_pred.dim()))
                v_rms = torch.sqrt(torch.mean(v_pred * v_pred, dim=dims, keepdim=True) + 1e-8)
                v_pred = v_pred * torch.clamp(target_rms / v_rms, min=0.8, max=1.05)

            if v_prev is None or prev_dt is None:
                x_t = x_t.add(v_pred, alpha=dt)
            else:
                step_ratio = min(dt / max(prev_dt, 1e-6), 1.5)
                ab_coeff = 0.25 if (is_last_in_phase or step == total_steps - 1) else 0.50
                v_ab2 = v_pred + (ab_coeff * step_ratio) * (v_pred - v_prev)
                x_t = x_t.add(v_ab2, alpha=dt)

            v_prev = v_pred
            prev_dt = dt

            if not torch.isfinite(x_t).all():
                logger.warning("NaN/Inf detected in phase_multistep_sample, applying nan_to_num")
                x_t = torch.nan_to_num(
                    x_t,
                    nan=0.0,
                    posinf=RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else 10.0,
                    neginf=-RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else -10.0,
                )

            if RF_CLAMP_RANGE > 0.0:
                x_t = x_t.clamp(-RF_CLAMP_RANGE, RF_CLAMP_RANGE)

            if return_history or callback is not None:
                x_1_pred = x_t + (1.0 - t1) * v_pred
                x_1_pred = torch.nan_to_num(
                    x_1_pred,
                    nan=0.0,
                    posinf=RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else 10.0,
                    neginf=-RF_CLAMP_RANGE if RF_CLAMP_RANGE > 0.0 else -10.0,
                )
                if RF_CLAMP_RANGE > 0.0:
                    x_1_pred = x_1_pred.clamp(-RF_CLAMP_RANGE, RF_CLAMP_RANGE)

            if return_history:
                history.append(x_1_pred.detach().cpu().half())
                step_times.append(round((time.perf_counter() - step_start) * 1000))

            if callback is not None:
                callback(step, x_1_pred, round((time.perf_counter() - step_start) * 1000))

        if not do_classifier_free_guidance and kwargs.get("restore_contrast", False):
            dims = (-3, -2, -1) if x_t.dim() == 4 else -1
            std = x_t.std(dim=dims, keepdim=True)
            mean = x_t.mean(dim=dims, keepdim=True)
            x_t = (x_t - mean) / (std + 1e-6) * 0.95 + mean

        if return_history:
            return x_t, history, step_times
        return x_t

    sample_adams_bashforth = phase_multistep_sample
