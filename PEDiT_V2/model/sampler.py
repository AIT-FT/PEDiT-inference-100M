import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class PhaseAwareSampler:
    """
    Фазово-синхронизированный сэмплер 2-го порядка (UniPC-bh2 / Phase-AB2) для EdgeFlowPhaseV2:
    - Автоматический сброс истории производных на границах фаз (0 -> 1 -> 2).
    - Честное выполнение всех блоков сети на малом числе шагов (9-12 шагов).
    - Полная совместимость сигнатур вызовов между train.py, inference.py и server.py.
    """
    def __init__(
        self,
        model: nn.Module,
        solver_type: str = "unipc",
        time_shift: float = 1.5,
        rescaled_phi: float = 0.0,
        phase_handoff_gamma: float = 0.15,
        phase_temperatures: Tuple[float, float, float] = (0.95, 0.85, 0.72),
        **kwargs,
    ):
        self.model = model
        if "solver" in kwargs and kwargs["solver"] is not None:
            solver_type = str(kwargs["solver"])
        if "shift" in kwargs and kwargs["shift"] is not None:
            time_shift = float(kwargs["shift"])
        if "cfg_rescale" in kwargs and kwargs["cfg_rescale"] is not None:
            rescaled_phi = float(kwargs["cfg_rescale"])
        self.solver_type = solver_type.lower().strip()
        self.time_shift = float(time_shift)
        self.rescaled_phi = float(rescaled_phi)
        self.phase_handoff_gamma = float(phase_handoff_gamma)
        self.phase_temperatures = phase_temperatures
        self.default_device = kwargs.get("device", None)
        self.default_dtype = kwargs.get("dtype", None)
        self.default_use_block_cache = bool(kwargs.get("use_block_cache", False))

    @staticmethod
    def steps_to_per_phase(steps: int) -> Tuple[int, int, int]:
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")
        if steps <= 3:
            return (1, 1, max(1, steps - 2)) if steps == 3 else (steps, 0, 0)
        if steps == 9:
            return (3, 3, 3)
        if steps == 10:
            return (3, 4, 3)   # Больше шагов Эксперту 1 (было 3, 3, 4)
        if steps == 11:
            return (4, 4, 3)
        if steps == 12:
            return (4, 4, 4)
        if steps == 14:
            return (5, 6, 3)   # Усиленный режим для анатомии (5 + 6 шагов на Фазы 0-1, 3 шага на Фазу 2)
        if steps == 15:
            return (6, 6, 3)   # 15 шагов: максимум точности для Экспертов 0 и 1
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
    def apply_apg_momentum(
        v_curr: torch.Tensor,
        v_prev: torch.Tensor,
        eta: float = 0.0,
        norm_thr: float = 2.5,
        boost_scale: float = 0.20,
    ) -> torch.Tensor:
        orig_dtype = v_curr.dtype
        v_c = v_curr.float()
        v_p = v_prev.float()
        diff = v_c - v_p
        dims = tuple(range(1, v_c.dim()))

        v_norm = torch.linalg.vector_norm(v_c, dim=dims, keepdim=True).clamp(min=1e-6)
        v_unit = v_c / v_norm
        diff_parallel = (diff * v_unit).sum(dim=dims, keepdim=True) * v_unit
        diff_orth = diff - diff_parallel

        orth_norm = torch.linalg.vector_norm(diff_orth, dim=dims, keepdim=True).clamp(min=1e-6)
        clip_factor = torch.clamp(norm_thr / orth_norm, max=1.0)
        diff_orth = diff_orth * clip_factor

        out = v_c + boost_scale * (eta * diff_parallel + diff_orth)
        return out.to(dtype=orig_dtype)

    def get_phase_schedule(
        self,
        steps_per_phase: Tuple[int, int, int] = (4, 4, 4),
    ) -> List[dict]:
        schedule = []
        total_req = sum(steps_per_phase)
        mu_base = max(float(self.time_shift), 1e-4)

        # Корректная обработка малого числа шагов (steps < 3) без разрыва интервала [0, 1]
        if total_req < 3 or any(n <= 0 for n in steps_per_phase):
            n_total = max(1, total_req)
            u = torch.linspace(0.0, 1.0, n_total + 1)
            if abs(mu_base - 1.0) > 1e-6:
                u = u / (mu_base - (mu_base - 1.0) * u)
            for s_idx in range(n_total):
                t_curr = float(u[s_idx].item())
                t_next = float(u[s_idx + 1].item())
                p_idx = min(int(t_curr * 3.0), 2)
                schedule.append({
                    "step_idx": s_idx,
                    "phase_idx": p_idx,
                    "step_in_phase": 0 if s_idx == 0 else s_idx,
                    "t_curr": t_curr,
                    "t_next": t_next,
                    "is_last_in_phase": True,
                })
            return schedule

        num_phases = len(steps_per_phase)
        boundaries = [float(i) / float(num_phases) for i in range(num_phases + 1)]
        phase_shifts = (
            mu_base,
            1.0 + 0.45 * (mu_base - 1.0),
            1.0 + 0.15 * (mu_base - 1.0),
        )

        curr_idx = 0
        for p_idx, n_steps in enumerate(steps_per_phase):
            t_start = boundaries[p_idx]
            t_end = boundaries[p_idx + 1]
            u = torch.linspace(0.0, 1.0, n_steps + 1)
            mu = phase_shifts[min(p_idx, len(phase_shifts) - 1)]
            if abs(mu - 1.0) > 1e-6:
                u = u / (mu - (mu - 1.0) * u)
            t_grid = t_start + (t_end - t_start) * u

            for s_in_p in range(n_steps):
                t_curr = float(t_grid[s_in_p].item())
                t_next = float(t_grid[s_in_p + 1].item())
                schedule.append({
                    "step_idx": curr_idx,
                    "phase_idx": p_idx,
                    "step_in_phase": s_in_p,
                    "t_curr": t_curr,
                    "t_next": t_next,
                    "is_last_in_phase": (s_in_p == n_steps - 1),
                })
                curr_idx += 1
        return schedule

    @torch.inference_mode()
    def sample(
        self,
        z: Optional[torch.Tensor] = None,
        cond_seq: Optional[torch.Tensor] = None,
        cond_pooled: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        uncond_pooled: Optional[torch.Tensor] = None,
        uncond_seq: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        steps: Optional[int] = None,
        steps_per_phase: Optional[Tuple[int, int, int]] = None,
        dynamic_cfg: bool = True,
        use_block_cache: Optional[bool] = None,
        guidance: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, torch.Tensor, int], Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if cond_mask is None and "mask" in kwargs:
            cond_mask = kwargs["mask"]
        if use_block_cache is None:
            use_block_cache = self.default_use_block_cache
        eff_rescaled_phi = float(kwargs["cfg_rescale"]) if ("cfg_rescale" in kwargs and kwargs["cfg_rescale"] is not None) else self.rescaled_phi
        phase_handoff_gamma = float(kwargs.get("phase_handoff_gamma", self.phase_handoff_gamma))
        gen = kwargs.get("generator", None)

        if z is None:
            initial_noise = kwargs.get("initial_noise", None)
            shape = kwargs.get("shape", None)
            model_param = next(self.model.parameters(), None)
            fallback_dev = model_param.device if model_param is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dev = self.default_device or (cond_pooled.device if cond_pooled is not None else fallback_dev)
            dt = self.default_dtype or (cond_pooled.dtype if cond_pooled is not None else torch.float32)
            if initial_noise is not None:
                z = initial_noise.to(device=dev, dtype=dt)
            elif shape is not None:
                z = torch.randn(shape, device=dev, dtype=dt, generator=gen)
            else:
                raise ValueError("Either `z`, `initial_noise`, or `shape` must be provided to PhaseAwareSampler.sample()")

        if cond_pooled is None:
            raise ValueError("`cond_pooled` must not be None in PhaseAwareSampler.sample()")

        B, C, H, W = z.shape
        device = z.device
        dtype = z.dtype

        # Корректное рекурсивное разворачивание любых комбинаций DDP / FSDP / AveragedModel / torch.compile
        model_api = self.model
        visited = set()
        while id(model_api) not in visited:
            visited.add(id(model_api))
            if hasattr(model_api, "_orig_mod"):
                model_api = model_api._orig_mod
            elif hasattr(model_api, "_fsdp_wrapped_module"):
                model_api = model_api._fsdp_wrapped_module
            elif hasattr(model_api, "module"):
                model_api = model_api.module
            else:
                break

        explicit_uncond = uncond_pooled is not None
        if cond_pooled.shape[0] == 2 * B:
            if uncond_pooled is None:
                uncond_pooled = cond_pooled[:B]
            cond_pooled = cond_pooled[B:]
        if cond_seq is not None and cond_seq.shape[0] == 2 * B:
            cond_seq = cond_seq[B:]
        if cond_mask is not None and cond_mask.shape[0] == 2 * B:
            cond_mask = cond_mask[B:]

        # Используем null_embedding только если пользователь НЕ передал явный uncond_pooled (негативный промпт)
        if not explicit_uncond and hasattr(model_api, "pooler") and hasattr(model_api.pooler, "null_embedding"):
            uncond_pooled = model_api.pooler.null_embedding.view(1, -1).expand(B, -1).to(
                device=cond_pooled.device, dtype=cond_pooled.dtype
            )
        elif uncond_pooled is None:
            uncond_pooled = torch.zeros_like(cond_pooled)

        eff_cond_mask = cond_mask
        if eff_cond_mask is not None:
            seq_cap = eff_cond_mask.shape[-1]
            active_any = (eff_cond_mask != 0).any(dim=0)
            if active_any.any():
                max_pos = int(torch.nonzero(active_any, as_tuple=False)[-1].item()) + 1
            else:
                max_pos = 16
            real_len = min(seq_cap, max(16, ((max_pos + 7) // 8) * 8))

            if cond_seq is not None:
                if cond_seq.dim() == 4 and cond_seq.shape[2] > real_len:
                    cond_seq = cond_seq[:, :, :real_len].contiguous()
                elif cond_seq.dim() == 3 and cond_seq.shape[1] > real_len:
                    cond_seq = cond_seq[:, :real_len].contiguous()
            if eff_cond_mask.shape[1] > real_len:
                eff_cond_mask = eff_cond_mask[:, :real_len].contiguous()
            if eff_cond_mask.bool().all():
                eff_cond_mask = None
            elif eff_cond_mask.device != device:
                eff_cond_mask = eff_cond_mask.to(device=device, non_blocking=True)

        if steps_per_phase is None:
            steps_per_phase = self.steps_to_per_phase(steps if steps is not None else 12)

        schedule = self.get_phase_schedule(steps_per_phase=steps_per_phase)
        total_steps = len(schedule)
        is_euler = self.solver_type == "euler"
        is_unipc = self.solver_type == "unipc"

        prev_v: Optional[torch.Tensor] = None
        prev_h: Optional[float] = None
        z_prev: Optional[torch.Tensor] = None
        prev_was_cfg: bool = False
        phase_kvs = None
        current_phase = -1
        mid_residual = None
        orig_dtype = z.dtype
        z = z.float()

        for item in schedule:
            step_start = time.perf_counter()
            step_idx = item["step_idx"]
            p_idx = item["phase_idx"]
            s_in_p = item["step_in_phase"]
            t_curr = item["t_curr"]
            t_next = item["t_next"]
            is_last = item["is_last_in_phase"]
            h_i = t_next - t_curr

            if p_idx != current_phase:
                current_phase = p_idx
                if hasattr(model_api, "stream_phase_expert"):
                    model_api.stream_phase_expert(p_idx)
                mid_residual = None
                if cond_seq is not None and hasattr(model_api, "prepare_phase_cross_kvs"):
                    phase_kvs = model_api.prepare_phase_cross_kvs(cond_seq, p_idx)

            if s_in_p == 0 and p_idx + 1 < 3 and hasattr(model_api, "prefetch_next_phase"):
                model_api.prefetch_next_phase(p_idx + 1)

            t_tensor = torch.full((B,), t_curr, device=device, dtype=dtype)
            is_cfg_step = (cfg_scale > 1.0)
            eff_cfg_scale = (1.0 + 0.65 * (cfg_scale - 1.0)) if (p_idx == 2 and dynamic_cfg) else cfg_scale

            if is_cfg_step:
                temp_p = 1.0
            else:
                temp_p = self.phase_temperatures[min(p_idx, len(self.phase_temperatures) - 1)]

            allow_cache = use_block_cache and (p_idx < len(steps_per_phase) and steps_per_phase[p_idx] >= 4) and (total_steps >= 18)
            capture_mid = allow_cache and (s_in_p == 1)
            apply_mid = allow_cache and (s_in_p == 2) and (mid_residual is not None)

            eff_mid_residual = None
            if apply_mid and mid_residual is not None:
                mid_delta = mid_residual[0] if isinstance(mid_residual, tuple) else mid_residual
                target_b = 2 * B if is_cfg_step else B
                eff_mid_residual = mid_delta[target_b:] if mid_delta.shape[0] == 2 * target_b else mid_delta

            z_f = z.float()

            if is_cfg_step:
                z_in = torch.cat([z, z], dim=0)
                t_in = torch.cat([t_tensor, t_tensor], dim=0)
                pooled_in = torch.cat([uncond_pooled, cond_pooled], dim=0)
                g_in = torch.cat([guidance, guidance], dim=0) if guidance is not None else None

                res = self.model(
                    z_in,
                    pooled_in,
                    cond_seq=cond_seq if phase_kvs is None else None,
                    t_phase=t_in,
                    phase_idx=p_idx,
                    per_block_text_kvs=phase_kvs,
                    mask=eff_cond_mask,
                    temperature=temp_p,
                    guidance=g_in,
                    use_block_cache=apply_mid,
                    cached_mid_residual=eff_mid_residual,
                    return_mid_residual=capture_mid,
                )
                if capture_mid and isinstance(res, tuple):
                    v_raw, mid_residual = res
                else:
                    v_raw = res

                v_uncond, v_cond = v_raw.float().chunk(2, dim=0)
                v_cfg = v_uncond + eff_cfg_scale * (v_cond - v_uncond)

                rem_t = max(1.0 - t_curr, 1e-3)
                if eff_rescaled_phi > 0.0 and rem_t > 0.05:
                    x1_cond = z_f + rem_t * v_cond
                    x1_cfg = z_f + rem_t * v_cfg
                    dims = tuple(range(1, x1_cfg.dim()))
                    mean_cond = x1_cond.mean(dim=dims, keepdim=True)
                    mean_cfg = x1_cfg.mean(dim=dims, keepdim=True)
                    std_cond = x1_cond.std(dim=dims, keepdim=True, unbiased=False).clamp(min=1e-5)
                    std_cfg = x1_cfg.std(dim=dims, keepdim=True, unbiased=False).clamp(min=1e-5)
                    factor = (std_cond / std_cfg).clamp(0.65, 1.25)
                    x1_rescaled = mean_cond + (x1_cfg - mean_cfg) * factor
                    x1_final = eff_rescaled_phi * x1_rescaled + (1.0 - eff_rescaled_phi) * x1_cfg
                    v_i = (x1_final - z_f) / rem_t
                else:
                    v_i = v_cfg
            else:
                res = self.model(
                    z,
                    cond_pooled,
                    cond_seq=cond_seq if phase_kvs is None else None,
                    t_phase=t_tensor,
                    phase_idx=p_idx,
                    per_block_text_kvs=phase_kvs,
                    mask=eff_cond_mask,
                    temperature=temp_p,
                    guidance=guidance,
                    use_block_cache=apply_mid,
                    cached_mid_residual=eff_mid_residual,
                    return_mid_residual=capture_mid,
                )
                if capture_mid and isinstance(res, tuple):
                    v_i, mid_residual = res
                else:
                    v_i = res
                v_i = v_i.float()

            # 1. На границе смены фазы ИЛИ смены режима CFG строго сбрасываем историю ДО корректора UniC-2,
            # чтобы не интерполировать скорости между двумя разными экспертами и слоями Qwen!
            if s_in_p == 0 or (is_cfg_step != prev_was_cfg):
                prev_v = None
                prev_h = None
                z_prev = None
            prev_was_cfg = is_cfg_step

            # 2. Трапецеидальный корректор UniC-2 (строго внутри текущей фазы на гладком векторном поле одного эксперта)
            if is_unipc and z_prev is not None and prev_v is not None and prev_h is not None:
                z_f = z_prev + prev_h * (0.5 * prev_v + 0.5 * v_i)

            # 3. Предиктор (2-го порядка AB2 для unipc/ab2/dpm++_2m, либо 1-го порядка для euler)
            if not is_euler and prev_v is not None and prev_h is not None:
                d_i = (v_i - prev_v) * min(h_i / max(prev_h, 1e-6), 1.5)
                ab_coeff = 0.25 if (is_last or step_idx == total_steps - 1) else 0.50
                z_next = z_f + h_i * v_i + ab_coeff * h_i * d_i
            else:
                z_next = z_f + h_i * v_i

            z_next = torch.nan_to_num(z_next, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

            z_prev = z_f
            prev_v = v_i
            prev_h = h_i
            z = z_next

            # В момент перехода с Фазы 0 на Фазу 1 (при t = 1/3):
            # Слегка "размягчаем" высокочастотную зажатость болванки Эксперта 0, сохраняя 100% композиции:
            if p_idx == 0 and is_last and (len(steps_per_phase) > 1):
                if phase_handoff_gamma > 0.0:
                    t_b = t_next
                    x_1_pred_exp0 = z + (1.0 - t_b) * v_i
                    x_0_est = z - t_b * v_i
                    x_0_noise = (
                        torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=gen)
                        if (gen is not None and getattr(gen, "device", None) == z.device)
                        else torch.randn_like(z)
                    )
                    x_0_fresh = ((1.0 - phase_handoff_gamma**2) ** 0.5) * x_0_est + phase_handoff_gamma * x_0_noise
                    z = t_b * x_1_pred_exp0 + (1.0 - t_b) * x_0_fresh
                    z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

            if callback is not None:
                x_1_pred = (z + max(1.0 - t_next, 0.0) * v_i).clamp(-10.0, 10.0)
                step_ms = round((time.perf_counter() - step_start) * 1000)
                callback(step_idx, x_1_pred, step_ms)

        return z.to(dtype=orig_dtype)