import json
import os
import time
from collections import deque
from typing import Any, Dict


class EfficiencyTracker:
    def __init__(
        self,
        model_config: Dict[str, Any],
        batch_size: int,
        image_tokens: int,
        text_tokens: int,
        gradient_checkpointing: bool = False,
    ):
        self.model_config = model_config
        self.batch_size = int(batch_size)
        self.image_tokens = int(image_tokens)
        self.text_tokens = int(text_tokens)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        window = int(os.environ.get("EFF_WINDOW", "100"))
        self.data_times = deque(maxlen=window)
        self.compute_times = deque(maxlen=window)

        default_peak = 71.0
        try:
            import torch
            if torch.cuda.is_available():
                dev_idx = torch.cuda.current_device()
                dev_name = torch.cuda.get_device_name(dev_idx)
                if "3060" in dev_name:
                    default_peak = 25.5
                elif "3070" in dev_name:
                    default_peak = 40.6
                elif "3080" in dev_name:
                    default_peak = 59.5
                elif "3090" in dev_name:
                    default_peak = 71.0
                elif "4060" in dev_name:
                    default_peak = 30.2
                elif "4070" in dev_name:
                    default_peak = 58.2
                elif "4080" in dev_name:
                    default_peak = 97.5
                elif "4090" in dev_name:
                    default_peak = 165.2
                elif "A100" in dev_name:
                    default_peak = 312.0
                elif "H100" in dev_name:
                    default_peak = 989.0

                use_fp8 = (
                    os.environ.get("FP8_MATMUL", "0") == "1"
                    or bool(self.model_config.get("quantize_fp8", False))
                    or self.model_config.get("quantization") == "fp8"
                )
                if use_fp8 and any(k in dev_name for k in ("4060", "4070", "4080", "4090", "H100", "H200", "L40")):
                    default_peak *= 2.0
        except Exception:
            pass
        self.peak_tflops = float(os.environ.get("PEAK_TFLOPS", str(default_peak)))
        self.flops_per_sample = self._estimate_flops_per_sample()
    def _estimate_flops_per_sample(self) -> float:
        cfg = self.model_config

        dim = int(cfg.get("dim", 1536))
        depth = int(cfg.get("depth", 30))
        heads = int(cfg.get("heads", 16))
        kv_heads = int(cfg.get("kv_heads", heads))
        head_dim = int(cfg.get("head_dim") or (dim // max(1, heads)))
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim

        ffn_hidden = int(cfg.get("ffn_hidden", 2560))
        cross_every = int(cfg.get("cross_every", 1))
        expert_hidden_list = cfg.get("expert_hidden_list", [])
        phase_expert_layers = cfg.get("phase_expert_layers", None)
        num_registers = int(cfg.get("num_registers", 0))
        num_sinks = int(cfg.get("num_sinks", 4)) if cfg.get("use_cross_sink", False) else 0
        ada_rank = cfg.get("ada_rank", None)
        phase_lora_rank = int(cfg.get("phase_lora_rank") or 0) if cfg.get("use_phase_lora", False) else 0
        cross_lora_rank = int(cfg.get("cross_lora_rank") or 0) if cfg.get("use_cross_lora", False) else 0

        if expert_hidden_list:
            active_expert_hidden = float(sum(expert_hidden_list) / len(expert_hidden_list))
        else:
            active_expert_hidden = 0.0

        # Учитываем каскадное число слоев экспертов (например, (6, 9, 12) из 12 -> в среднем 9/12 = 75% слоев)
        if phase_expert_layers and len(phase_expert_layers) > 0:
            avg_expert_layers = sum(min(int(k), depth) for k in phase_expert_layers) / float(len(phase_expert_layers))
            expert_layer_ratio = avg_expert_layers / float(max(1, depth))
        else:
            expert_layer_ratio = 1.0

        tokens = max(1, int(self.image_tokens) + num_registers)
        text_tokens = max(1, int(self.text_tokens) + num_sinks)
        cross_layers = depth // cross_every if cross_every > 0 else 0

        # Self-Attention (Q, K, V, Out + SDPA + Phase-LoRA)
        sa_proj_flops = 2.0 * tokens * dim * (2 * q_dim + 2 * kv_dim)
        if phase_lora_rank > 0:
            k_mult = 2 if cfg.get("use_phase_k_lora", False) else 1
            sa_proj_flops += 2.0 * tokens * phase_lora_rank * ((1 + k_mult) * dim + q_dim + k_mult * kv_dim)
        sa_sdpa_flops = 4.0 * tokens * tokens * q_dim

        # Shared SpatialMixSwiGLU + Каскадный Phase Expert + Per-Block AdaLN
        ffn_flops = 6.0 * tokens * dim * (ffn_hidden + active_expert_hidden * expert_layer_ratio)
        if ada_rank:
            ada_flops = 2.0 * (dim * ada_rank + ada_rank * 8 * dim)
        else:
            ada_flops = 2.0 * dim * (8 * dim)

        forward_flops = depth * (sa_proj_flops + sa_sdpa_flops + ffn_flops + ada_flops)

        # Cross-Attention (Q, Out + Cross-LoRA + SDPA)
        ca_proj_flops = 4.0 * tokens * dim * q_dim
        if cross_lora_rank > 0:
            ca_proj_flops += 8.0 * text_tokens * kv_dim * cross_lora_rank
        ca_sdpa_flops = 4.0 * tokens * text_tokens * q_dim
        forward_flops += cross_layers * (ca_proj_flops + ca_sdpa_flops)

        backward_multiplier = 4.0 if self.gradient_checkpointing else 3.0
        return float(forward_flops * backward_multiplier)

    def update(self, data_time: float, compute_time: float):
        self.data_times.append(float(data_time))
        self.compute_times.append(float(compute_time))

    def get_metrics(self) -> Dict[str, Any]:
        if len(self.data_times) > 0:
            avg_data_time = sum(self.data_times) / len(self.data_times)
        else:
            avg_data_time = 0.001

        if len(self.compute_times) > 0:
            avg_compute_time = sum(self.compute_times) / len(self.compute_times)
        else:
            avg_compute_time = 0.001

        step_time = max(avg_data_time + avg_compute_time, 1e-9)
        samples_per_sec = self.batch_size / step_time

        achieved_tflops = self.flops_per_sample * samples_per_sec / 1e12
        mfu = 100.0 * achieved_tflops / max(self.peak_tflops, 1e-6)

        data_bottleneck = avg_data_time > (0.3 * avg_compute_time)

        return {
            "avg_data_time": avg_data_time,
            "avg_compute_time": avg_compute_time,
            "samples_per_sec": samples_per_sec,
            "achieved_tflops": achieved_tflops,
            "mfu": mfu,
            "data_bottleneck": data_bottleneck,
        }

    def save_metrics(self, save_dir: str, step: int, metrics: Dict[str, Any]):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "metrics.jsonl")

        payload = {
            "step": int(step),
            "time": time.time(),
            **metrics,
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")