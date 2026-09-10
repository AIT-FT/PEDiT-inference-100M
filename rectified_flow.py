import os
import time
import logging
from typing import Any, Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

RF_CLAMP_RANGE = float(os.environ.get("RF_CLAMP_RANGE", "10.0"))
RF_T_EPS = float(os.environ.get("RF_T_EPS", "0.0"))


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Разворачивает DDP / torch.compile обертки.
    """
    m = model

    for _ in range(10):
        if hasattr(m, "module"):
            m = m.module
        elif hasattr(m, "_orig_mod"):
            m = m._orig_mod
        else:
            break

    return m


class RectifiedFlow4D:
    """
    Rectified Flow:
        x_t = t * x_1 + (1 - t) * x_0
        target_v = x_1 - x_0

    Sampling:
        ODE dx/dt = v(x, t), t: 0 -> 1
    """

    @staticmethod
    def _time_schedule(steps: int, shift: float, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)

        if abs(shift - 1.0) > 1e-6:
            t = shift * t / (1.0 + (shift - 1.0) * t)

        if RF_T_EPS > 0.0:
            t = t.clamp(RF_T_EPS, 1.0 - RF_T_EPS)

        return t

    def add_noise(
        self,
        x_1: torch.Tensor,
        t_phase: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if generator is not None and generator.device.type != x_1.device.type:
            x_0 = torch.randn(
                x_1.shape,
                device=generator.device,
                dtype=x_1.dtype,
                generator=generator,
            ).to(x_1.device)
        else:
            x_0 = torch.randn(
                x_1.shape,
                device=x_1.device,
                dtype=x_1.dtype,
                generator=generator,
            )

        if x_1.dim() == 4:
            t = t_phase.reshape(-1, 1, 1, 1)
        else:
            t = t_phase.reshape(-1, 1)

        x_t = t * x_1 + (1.0 - t) * x_0
        target_v = x_1 - x_0

        return x_t, x_0, target_v

    @torch.inference_mode()
    def phase_euler_sample(
        self,
        model: torch.nn.Module,
        cond_pooled: torch.Tensor,
        cond_seq: Optional[torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        steps: int = 8,
        mask: Optional[torch.Tensor] = None,
        text_kvs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 0.0,
        shift: float = 1.0,
        return_history: bool = False,
        uncond_pooled: Optional[torch.Tensor] = None,
        uncond_seq: Optional[torch.Tensor] = None,
        uncond_mask: Optional[torch.Tensor] = None,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        callback: Optional[Callable[[int, torch.Tensor, int], Any]] = None,
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

        do_classifier_free_guidance = cfg_scale > 1.0

        if do_classifier_free_guidance:
            if cond_pooled.shape[0] == 2 * B:
                c_pooled = cond_pooled
                c_seq = cond_seq
                c_mask = mask
            elif uncond_pooled is not None:
                c_pooled = torch.cat([uncond_pooled, cond_pooled], dim=0)

                c_seq = (
                    torch.cat([uncond_seq, cond_seq], dim=0)
                    if uncond_seq is not None and cond_seq is not None
                    else None
                )

                if uncond_mask is not None and mask is not None:
                    c_mask = torch.cat([uncond_mask, mask], dim=0)
                elif uncond_mask is not None and mask is None:
                    c_mask = torch.cat([uncond_mask, torch.ones_like(uncond_mask)], dim=0)
                elif uncond_mask is None and mask is not None:
                    c_mask = torch.cat([torch.ones_like(mask), mask], dim=0)
                else:
                    c_mask = None
            else:
                logger.warning(
                    "cfg_scale > 1.0, but no unconditional prompt provided and cond is not 2*B. "
                    "Running without CFG."
                )
                do_classifier_free_guidance = False
                c_pooled = cond_pooled
                c_seq = cond_seq
                c_mask = mask
        else:
            if cond_pooled.shape[0] == 2 * B:
                c_pooled = cond_pooled[B:]
                c_seq = cond_seq[B:] if cond_seq is not None else None
                c_mask = mask[B:] if mask is not None else None

                if text_kvs is not None:
                    text_kvs = (text_kvs[0][B:], text_kvs[1][B:])
            else:
                c_pooled = cond_pooled
                c_seq = cond_seq
                c_mask = mask

        if c_mask is not None and c_mask.shape[0] != c_pooled.shape[0]:
            raise ValueError(
                f"mask batch size {c_mask.shape[0]} does not match "
                f"conditioning batch size {c_pooled.shape[0]}"
            )

        model_api = _unwrap_model(model)

        if text_kvs is None:
            if c_seq is None:
                raise ValueError("text_kvs is None and cond_seq is None")
            text_kvs = model_api.encode_text(c_seq)
        else:
            if text_kvs[0].shape[0] != c_pooled.shape[0]:
                if c_seq is not None:
                    text_kvs = model_api.encode_text(c_seq)
                else:
                    raise ValueError(
                        f"text_kvs batch size {text_kvs[0].shape[0]} "
                        f"does not match conditioning batch size {c_pooled.shape[0]}, "
                        "and cond_seq is None, so text_kvs cannot be recomputed."
                    )

        history = [x_t.detach().cpu().half()] if return_history else None
        step_times = [] if return_history else None

        for step in range(steps):
            need_timing = return_history or callback is not None

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
                cond_seq=c_seq,
                t_phase=t_phase,
                mask=c_mask,
                text_kvs=text_kvs,
            )

            v_pred = v_pred.float()
            if do_classifier_free_guidance:
                v_uncond, v_text = v_pred.chunk(2, dim=0)
                v_cfg = v_uncond + cfg_scale * (v_text - v_uncond)

                if cfg_rescale > 0.0:
                    dims = tuple(range(1, v_text.dim()))

                    std_text = v_text.std(dim=dims, keepdim=True, unbiased=False)
                    std_cfg = v_cfg.std(dim=dims, keepdim=True, unbiased=False)

                    factor = std_text / (std_cfg + 1e-6)
                    factor = factor.clamp(0.5, 2.0)

                    v_rescaled = v_cfg * factor
                    v_pred = (1.0 - cfg_rescale) * v_cfg + cfg_rescale * v_rescaled
                else:
                    v_pred = v_cfg

            x_t = x_t + v_pred * dt

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

        if return_history:
            return x_t, history, step_times

        return x_t