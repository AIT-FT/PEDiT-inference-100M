import contextlib
import faulthandler
import gc
import glob
import json
import logging
import math
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import config
except Exception:
    pass

if sys.platform == "win32":
    _torch_cache_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".torch_cache"))
    os.makedirs(os.path.join(_torch_cache_root, "inductor"), exist_ok=True)
    os.makedirs(os.path.join(_torch_cache_root, "triton"), exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(_torch_cache_root, "inductor")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(_torch_cache_root, "triton")

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision

from torch.optim.swa_utils import AveragedModel

try:
    from torch.optim.swa_utils import get_ema_multi_avg_fn
except Exception:
    def get_ema_multi_avg_fn(decay: float):
        def multi_avg_fn(averaged, model, num_averaged=None):
            return decay * averaged + (1.0 - decay) * model

        return multi_avg_fn

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.nn.parallel import DistributedDataParallel as DDP
except Exception:
    DDP = None

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    FSDP_AVAILABLE = True
except Exception:
    FSDP = None
    FSDP_AVAILABLE = False

from dataset_cached import (
    CachedLatentTextDataset,
    cache_is_ready,
    get_cached_dataloader,
    read_meta,
    shutdown_dataloader,
    wait_for_cache,
)
from efficiency import EfficiencyTracker
from encoders import SDVAE, SDXLVAE, TAESD, FluxVAE, DCAE, TextEncoder, Qwen35TextEncoder
from config import get_model_config
from model import EdgeFlowPhaseV2, EdgeFlowPhaseBlockV2, PhaseExpert
from rectified_flow import RectifiedFlow4D
from sampler import PhaseAwareSampler


faulthandler.enable()

torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except Exception:
    pass


USE_COMPILE = os.environ.get("USE_COMPILE", "1") == "1"
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1"


def safe_replace(src: str, dst: str, retries: int = 5, delay: float = 0.1) -> None:
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except OSError as e:
            if attempt == retries - 1:
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.replace(src, dst)
                    return
                except Exception:
                    raise e
            time.sleep(delay)


def get_rclone_cmd() -> str:
    rc = shutil.which("rclone")
    if rc:
        return rc
    candidates = [
        "rclone",
        "rclone.exe",
        "./rclone-extracted/rclone-v1.65.2-linux-amd64/rclone",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return "rclone"


def setup_logger(save_dir: str, is_main_process: bool = True) -> logging.Logger:
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("EdgeFlowPhaseExperts")
    logger.setLevel(logging.INFO if is_main_process else logging.WARNING)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if is_main_process:
        file_handler = logging.FileHandler(
            os.path.join(save_dir, "training.log"),
            mode="a",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_memory_usage() -> str:
    ram = psutil.virtual_memory()

    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)
    ram_free_gb = ram.available / (1024**3)

    if torch.cuda.is_available():
        vram_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        vram_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        vram_total_gb = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory / (1024**3)
        vram_free_gb = vram_total_gb - vram_reserved_gb

        return (
            f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB "
            f"(Free: {ram_free_gb:.1f}GB) | "
            f"VRAM: {vram_allocated_gb:.1f}/{vram_reserved_gb:.1f}GB "
            f"(Free: {vram_free_gb:.1f}GB)"
        )

    return (
        f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB "
        f"(Free: {ram_free_gb:.1f}GB)"
    )


def parse_checkpoint_step(path: str) -> int:
    filename = os.path.splitext(os.path.basename(path))[0]

    try:
        if filename.endswith(("_weights", "_ema")):
            return int(filename.split("_")[-2])

        return int(filename.split("_")[-1])
    except Exception:
        return -1


def ensure_free_disk_space(
    save_dir: str,
    logger: logging.Logger,
    min_free_gb: float = float(os.environ.get("MIN_FREE_DISK_GB", "4.0")),
):
    import shutil

    def get_free_gb():
        try:
            return shutil.disk_usage(save_dir).free / (1024**3)
        except Exception:
            return shutil.disk_usage(".").free / (1024**3)

    while get_free_gb() < min_free_gb:
        deleted_something = False

        attn_dir = os.path.join(save_dir, "attention_analysis")
        if os.path.exists(attn_dir):
            png_files = sorted(glob.glob(os.path.join(attn_dir, "*.png")), key=os.path.getmtime)
            if len(png_files) > 180:
                to_delete = png_files[: min(len(png_files) - 180, 72)]
                for p in to_delete:
                    try:
                        os.remove(p)
                        deleted_something = True
                    except Exception:
                        pass

        full_ckpts = [
            p
            for p in glob.glob(os.path.join(save_dir, "edgeflow_phase_experts_step_*.pt"))
            if not p.endswith("_weights.pt") and not p.endswith("_ema.pt")
        ]
        full_ckpts = sorted(full_ckpts, key=parse_checkpoint_step)

        weights_ckpts = glob.glob(
            os.path.join(save_dir, "edgeflow_phase_experts_step_*_weights.pt")
        )
        weights_ckpts = sorted(weights_ckpts, key=parse_checkpoint_step)

        ema_ckpts = glob.glob(
            os.path.join(save_dir, "edgeflow_phase_experts_step_*_ema.pt")
        )
        ema_ckpts = sorted(ema_ckpts, key=parse_checkpoint_step)

        if len(full_ckpts) > 1:
            try:
                logger.info(
                    f"[Disk Space] Less than {min_free_gb}GB free. "
                    f"Deleting oldest full checkpoint: {full_ckpts[0]}"
                )
                os.remove(full_ckpts[0])
                deleted_something = True
            except Exception as e:
                logger.warning(f"[Disk Space] Failed to delete {full_ckpts[0]}: {e}")

        if len(weights_ckpts) > 1:
            try:
                logger.info(
                    f"[Disk Space] Less than {min_free_gb}GB free. "
                    f"Deleting oldest weights checkpoint: {weights_ckpts[0]}"
                )
                os.remove(weights_ckpts[0])
                deleted_something = True
            except Exception as e:
                logger.warning(
                    f"[Disk Space] Failed to delete {weights_ckpts[0]}: {e}"
                )

        if len(ema_ckpts) > 1:
            try:
                logger.info(
                    f"[Disk Space] Less than {min_free_gb}GB free. "
                    f"Deleting oldest EMA checkpoint: {ema_ckpts[0]}"
                )
                os.remove(ema_ckpts[0])
                deleted_something = True
            except Exception:
                pass

        if not deleted_something:
            logger.warning("[Disk Space] Could not free up space by deleting checkpoints!")
            break


def cleanup_checkpoints(save_dir: str, keep: int = 2):
    if keep <= 0:
        return

    full_ckpts = [
        p
        for p in glob.glob(os.path.join(save_dir, "edgeflow_phase_experts_step_*.pt"))
        if not p.endswith("_weights.pt") and not p.endswith("_ema.pt")
    ]
    full_ckpts = sorted(full_ckpts, key=parse_checkpoint_step)

    if len(full_ckpts) > keep:
        for p in full_ckpts[:-keep]:
            try:
                os.remove(p)
            except Exception:
                pass

    weights_ckpts = glob.glob(
        os.path.join(save_dir, "edgeflow_phase_experts_step_*_weights.pt")
    )
    weights_ckpts = sorted(weights_ckpts, key=parse_checkpoint_step)

    if len(weights_ckpts) > keep:
        for p in weights_ckpts[:-keep]:
            try:
                os.remove(p)
            except Exception:
                pass

    ema_ckpts = glob.glob(
        os.path.join(save_dir, "edgeflow_phase_experts_step_*_ema.pt")
    )
    ema_ckpts = sorted(ema_ckpts, key=parse_checkpoint_step)

    if len(ema_ckpts) > keep:
        for p in ema_ckpts[:-keep]:
            try:
                os.remove(p)
            except Exception:
                pass

    tmp_files = glob.glob(os.path.join(save_dir, "*.tmp"))

    for tmp_file in tmp_files:
        try:
            os.remove(tmp_file)
        except Exception:
            pass


def unwrap_model(model):
    while True:
        if isinstance(model, AveragedModel):
            model = model.module
        elif hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            break
    return model


def save_checkpoint(
    model,
    text_encoder,
    optimizer,
    scheduler,
    step,
    save_dir,
    logger,
    ema_model=None,
    ema_text_proj=None,
    ema_text_pooler=None,
    scaler=None,
    epoch=0,
    config=None,
    latent_mean=None,
    latent_std=None,
):
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"edgeflow_phase_experts_step_{step}.pt")

    if FSDP_AVAILABLE and isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_sd = model.state_dict()
    else:
        raw_model = unwrap_model(model)
        model_sd = raw_model.state_dict()

    is_main_proc = (not dist.is_initialized()) or (dist.get_rank() == 0)

    if is_main_proc:
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "config": config if config is not None else {},
            "model_state_dict": model_sd,
            "text_encoder_state_dict": text_encoder.proj.state_dict(),
            "text_encoder_pooler_state_dict": text_encoder.pooler.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
        }

        if ema_model is not None:
            checkpoint["ema_state_dict"] = ema_model.state_dict()

        if ema_text_proj is not None:
            checkpoint["ema_text_proj_state_dict"] = ema_text_proj.state_dict()

        if ema_text_pooler is not None:
            checkpoint["ema_text_pooler_state_dict"] = ema_text_pooler.state_dict()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        if latent_mean is not None:
            checkpoint["latent_mean"] = latent_mean.cpu()

        if latent_std is not None:
            checkpoint["latent_std"] = latent_std.cpu()

        temp_path = save_path + ".tmp"
        torch.save(checkpoint, temp_path)

        weights_path = os.path.join(
            save_dir,
            f"edgeflow_phase_experts_step_{step}_weights.pt",
        )

        weights_checkpoint = {
            "step": step,
            "epoch": epoch,
            "config": config if config is not None else {},
            "model_state_dict": model_sd,
            "text_encoder_state_dict": text_encoder.proj.state_dict(),
            "text_encoder_pooler_state_dict": text_encoder.pooler.state_dict(),
        }

        if ema_model is not None:
            weights_checkpoint["ema_model_state_dict"] = ema_model.state_dict()

        if ema_text_proj is not None:
            weights_checkpoint["ema_text_proj_state_dict"] = ema_text_proj.state_dict()

        if ema_text_pooler is not None:
            weights_checkpoint["ema_text_pooler_state_dict"] = ema_text_pooler.state_dict()

        if latent_mean is not None:
            weights_checkpoint["latent_mean"] = latent_mean.cpu()

        if latent_std is not None:
            weights_checkpoint["latent_std"] = latent_std.cpu()

        torch.save(weights_checkpoint, weights_path + ".tmp")
        safe_replace(weights_path + ".tmp", weights_path)

        if os.environ.get("CHECKPOINT_INTEGRITY_CHECK", "0") == "1":
            try:
                torch.load(temp_path, map_location="cpu")
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError(f"Checkpoint integrity check failed: {e}")

        safe_replace(temp_path, save_path)
        logger.info(f"Checkpoint saved: {save_path}")

    if dist.is_initialized():
        dist.barrier()


def load_latest_checkpoint(
    model,
    text_encoder,
    optimizer,
    scheduler,
    save_dir,
    logger,
    device,
    ema_model=None,
    ema_text_proj=None,
    ema_text_pooler=None,
    scaler=None,
):
    if not os.path.exists(save_dir):
        return 0, 0, None, None

    checkpoints = [
        p
        for p in glob.glob(os.path.join(save_dir, "edgeflow_phase_experts_step_*.pt"))
        if not p.endswith("_weights.pt") and not p.endswith("_ema.pt")
    ]

    if not checkpoints:
        return 0, 0, None, None

    latest_ckpt = max(checkpoints, key=parse_checkpoint_step)
    logger.info(f"Found checkpoint: {latest_ckpt}")

    try:
        checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(latest_ckpt, map_location="cpu")

    raw_sd = checkpoint["model_state_dict"]
    cleaned_sd = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(cleaned_sd)

    if ema_model is not None:
        ema_model.module.load_state_dict(cleaned_sd)

    if "text_encoder_state_dict" in checkpoint:
        text_encoder.proj.load_state_dict(checkpoint["text_encoder_state_dict"])

        if ema_text_proj is not None and "ema_text_proj_state_dict" not in checkpoint:
            ema_text_proj.module.load_state_dict(checkpoint["text_encoder_state_dict"])

    if "text_encoder_pooler_state_dict" in checkpoint:
        text_encoder.pooler.load_state_dict(
            checkpoint["text_encoder_pooler_state_dict"]
        )

        if (
            ema_text_pooler is not None
            and "ema_text_pooler_state_dict" not in checkpoint
        ):
            ema_text_pooler.module.load_state_dict(
                checkpoint["text_encoder_pooler_state_dict"]
            )

    if "optimizer_state_dict" in checkpoint:
        try:
            desired_group_meta = [
                {
                    "lr_scale": g.get("lr_scale", 1.0),
                    "weight_decay": g.get("weight_decay", 0.0),
                    "name": g.get("name", ""),
                }
                for g in optimizer.param_groups
            ]
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for g, meta in zip(optimizer.param_groups, desired_group_meta):
                g["lr_scale"] = meta["lr_scale"]
                g["weight_decay"] = meta["weight_decay"]
                if meta["name"]:
                    g["name"] = meta["name"]
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")

    if "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")

    if ema_model is not None:
        if "ema_state_dict" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
        else:
            if hasattr(ema_model, "n_averaged"):
                ema_model.n_averaged.fill_(0)

    if ema_text_proj is not None:
        if "ema_text_proj_state_dict" in checkpoint:
            ema_text_proj.load_state_dict(checkpoint["ema_text_proj_state_dict"])
        else:
            if hasattr(ema_text_proj, "n_averaged"):
                ema_text_proj.n_averaged.fill_(0)

    if ema_text_pooler is not None:
        if "ema_text_pooler_state_dict" in checkpoint:
            ema_text_pooler.load_state_dict(checkpoint["ema_text_pooler_state_dict"])
        else:
            if hasattr(ema_text_pooler, "n_averaged"):
                ema_text_pooler.n_averaged.fill_(0)

    if scaler is not None and "scaler_state_dict" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except Exception as e:
            logger.warning(f"Failed to load scaler state: {e}")

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    if (
        "cuda_rng_state" in checkpoint
        and checkpoint["cuda_rng_state"] is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    step = checkpoint.get("step", 0)
    epoch = checkpoint.get("epoch", 0)

    latent_mean = checkpoint.get("latent_mean", None)
    latent_std = checkpoint.get("latent_std", None)

    logger.info(f"Resuming from step {step}, epoch {epoch}")

    del checkpoint, raw_sd, cleaned_sd
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return step, epoch, latent_mean, latent_std


def apply_fixed_lr(optimizer, scheduler, fixed_lr: float, lr_text: Optional[float] = None):
    if lr_text is None:
        lr_text = fixed_lr * 0.5
    for group in optimizer.param_groups:
        gname = group.get("name", "")
        lr_mult = float(group.get("lr_scale", 1.0))
        target_lr = (lr_text if "text" in gname else fixed_lr) * lr_mult
        group["initial_lr"] = target_lr
        group["lr"] = target_lr
    if scheduler is not None and hasattr(scheduler, "set_fixed_lr"):
        scheduler.set_fixed_lr(fixed_lr, lr_text=lr_text)


class CustomLossScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr=1e-4, min_lr=5e-5, fixed_lr=None, lr_text=None):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.base_lr = float(base_lr)
        if "FIXED_LR" in os.environ and os.environ["FIXED_LR"].strip():
            self.fixed_lr = float(os.environ["FIXED_LR"])
        else:
            self.fixed_lr = float(fixed_lr) if fixed_lr is not None else None

        self.lr_text = float(lr_text) if lr_text is not None else (self.fixed_lr * 0.5 if self.fixed_lr is not None else None)
        self.min_lr = float(os.environ.get("MIN_LR", str(min_lr)))
        self.current_step = 0
        decay_start_default = str(min(15000, self.total_steps // 2))
        self.decay_start_step = int(os.environ.get("DECAY_START_STEP", decay_start_default))

        for group in self.optimizer.param_groups:
            lr_mult = float(group.get("lr_scale", 1.0))
            if self.fixed_lr is not None:
                gname = group.get("name", "")
                target_lr = (self.lr_text if "text" in gname else self.fixed_lr) * lr_mult
                group["initial_lr"] = target_lr
                group["lr"] = target_lr if self.warmup_steps == 0 else 0.0
            else:
                group.setdefault("initial_lr", group["lr"])

    def set_fixed_lr(self, fixed_lr: float, lr_text: Optional[float] = None):
        self.fixed_lr = float(fixed_lr)
        self.lr_text = float(lr_text) if lr_text is not None else self.fixed_lr * 0.5
        for group in self.optimizer.param_groups:
            gname = group.get("name", "")
            lr_mult = float(group.get("lr_scale", 1.0))
            target_lr = (self.lr_text if "text" in gname else self.fixed_lr) * lr_mult
            group["initial_lr"] = target_lr
            scale = min(1.0, self.current_step / float(self.warmup_steps)) if self.warmup_steps > 0 and self.current_step > 0 else 1.0
            group["lr"] = target_lr * scale

    def configure_cosine_decay(
        self,
        start_step: int,
        total_steps: int,
        base_lr: float,
        min_lr: float,
        lr_text: Optional[float] = None,
    ):
        self.fixed_lr = None
        if start_step > 0:
            self.warmup_steps = min(self.warmup_steps, start_step)
        self.decay_start_step = start_step
        self.total_steps = total_steps
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        if lr_text is None:
            lr_text = base_lr * 0.5
        self.lr_text = float(lr_text)

        eff_step = max(1, self.current_step)
        ratio = min(1.0, max(0.0, self.min_lr / max(1e-12, self.base_lr)))

        for group in self.optimizer.param_groups:
            gname = group.get("name", "")
            lr_mult = float(group.get("lr_scale", 1.0))
            target_lr = (self.lr_text if "text" in gname else self.base_lr) * lr_mult
            group["initial_lr"] = target_lr

            if self.warmup_steps > 0 and eff_step <= self.warmup_steps:
                group["lr"] = target_lr * (eff_step / float(self.warmup_steps))
            elif eff_step < self.decay_start_step:
                group["lr"] = target_lr
            else:
                decay_start = max(self.warmup_steps, self.decay_start_step)
                decay_total = max(1, self.total_steps - decay_start)
                progress = min(1.0, max(0.0, (eff_step - decay_start) / decay_total))
                scale = 0.5 * (1.0 + math.cos(math.pi * progress))
                p_min = target_lr * ratio
                group["lr"] = p_min + (target_lr - p_min) * scale

    def step(self, loss_val=None):
        self.current_step += 1

        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
            scale = self.current_step / float(self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * scale
            return

        if self.fixed_lr is not None:
            for group in self.optimizer.param_groups:
                gname = group.get("name", "")
                lr_mult = float(group.get("lr_scale", 1.0))
                target_lr = (self.lr_text if "text" in gname else self.fixed_lr) * lr_mult
                group["lr"] = target_lr
            return

        if self.current_step < self.decay_start_step:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"]
        else:
            decay_start = max(self.warmup_steps, self.decay_start_step)
            decay_total = max(1, self.total_steps - decay_start)
            progress = min(1.0, max(0.0, (self.current_step - decay_start) / decay_total))
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            base_lr_ref = getattr(self, "base_lr", 1e-4)
            min_lr = getattr(self, "min_lr", 5e-5)
            ratio = min(1.0, max(0.0, min_lr / max(1e-12, base_lr_ref)))
            for param_group in self.optimizer.param_groups:
                p_min = param_group["initial_lr"] * ratio
                param_group["lr"] = p_min + (param_group["initial_lr"] - p_min) * scale

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "fixed_lr": self.fixed_lr,
            "lr_text": self.lr_text,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict.get("current_step", 0)
        if self.fixed_lr is not None:
            for group in self.optimizer.param_groups:
                gname = group.get("name", "")
                lr_mult = float(group.get("lr_scale", 1.0))
                target_lr = (self.lr_text if "text" in gname else self.fixed_lr) * lr_mult
                group["initial_lr"] = target_lr
                scale = min(1.0, self.current_step / float(self.warmup_steps)) if self.warmup_steps > 0 else 1.0
                group["lr"] = target_lr * scale

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
    fixed_lr: Optional[float] = None,
    lr_text: Optional[float] = None,
    base_lr: Optional[float] = None,
    min_lr: Optional[float] = None,
):
    cosine_steps = int(os.environ.get("COSINE_STEPS", str(num_training_steps)))
    if base_lr is None:
        base_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 2e-4
    if min_lr is None:
        min_lr = float(os.environ.get("MIN_LR", "5e-5"))
    return CustomLossScheduler(
        optimizer,
        num_warmup_steps,
        cosine_steps,
        base_lr=base_lr,
        min_lr=min_lr,
        fixed_lr=fixed_lr,
        lr_text=lr_text,
    )

def make_ema(module: torch.nn.Module, decay: float) -> Optional[AveragedModel]:
    if module is None or sum(p.numel() for p in module.parameters()) == 0:
        return None

    step_counter = [-1]

    def ema_multi_avg_with_warmup(averaged_params, current_params, num_averaged):
        if step_counter[0] < 0:
            step_counter[0] = int(num_averaged.item()) if isinstance(num_averaged, torch.Tensor) else int(num_averaged or 0)
        step_counter[0] += 1
        n = step_counter[0]
        eff_decay = min(float(decay), float(1 + n) / float(2 + n))
        torch._foreach_lerp_(averaged_params, current_params, 1.0 - eff_decay)

    return AveragedModel(module, multi_avg_fn=ema_multi_avg_with_warmup)

def create_vae(device: torch.device):
    vae_type = os.environ.get("VAE_TYPE", "dcae").lower()
    for i, a in enumerate(sys.argv):
        if a == "--vae" and i + 1 < len(sys.argv):
            vae_type = sys.argv[i + 1].lower()
        elif a.startswith("--vae="):
            vae_type = a.split("=", 1)[1].lower()

    vae_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    if vae_type in ("dcae", "dc-ae", "dc_ae"):
        vae = DCAE(local_files_only=True, decoder_only=True, torch_dtype=vae_dtype)
    elif vae_type == "taesd":
        vae = TAESD()
    elif vae_type == "sd":
        vae = SDVAE()
    elif vae_type in ("sdxl", "sd_xl", "sd-xl"):
        vae = SDXLVAE()
    elif vae_type in ("flux", "flux1", "flux.1"):
        vae = FluxVAE()
    else:
        raise ValueError(f"Unknown VAE_TYPE: {vae_type}")

    return vae.to(device).eval()


def load_latent_stats(meta: Dict, device: torch.device):
    latent_shape = meta.get("latent_shape", [4])
    channels = latent_shape[0] if isinstance(latent_shape, (list, tuple)) and len(latent_shape) > 0 else 4

    mean = meta.get("latent_mean", [0.0] * channels)
    std = meta.get("latent_std", [1.0] * channels)

    if isinstance(mean, (int, float)):
        mean = [float(mean)] * channels

    if isinstance(std, (int, float)):
        std = [float(std)] * channels

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    std_t = torch.where(std_t == 0.0, torch.ones_like(std_t), std_t)

    return mean_t, std_t


def load_optional_npy(
    path: str,
    shape,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    _logger = logging.getLogger(__name__)

    if os.path.exists(path):
        arr = np.load(path)

        while arr.ndim > len(shape):
            arr = arr[0]

        tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)

        if tensor.shape == torch.Size(shape):
            return tensor

        _logger.warning(
            f"Optional file {path} has shape {tuple(tensor.shape)}, "
            f"expected {tuple(shape)}. Falling back to zeros."
        )

    return torch.zeros(shape, device=device, dtype=dtype)


def get_param_groups(
    model_raw,
    text_encoder,
    weight_decay: float,
    lr: float,
    lr_text: float,
    expert_weight_decay: Optional[float] = 0.0,
):
    groups = []

    base_decay = []
    base_no_decay = []
    shared_lora_adaln = []
    phase_routed_lora = []
    conv_refine_decay = []

    expert_decay = []
    expert_no_decay = []

    no_decay_keywords = (
        ".bias",
        "norm",
        "phase_table",
        "gate_init",
        "head_scale",
        "skip_phase_gate",
        "phase_layer_mixer",
        "v_res_lambda",
        "registers",
        "sink_k",
        "sink_v",
        "hf_gain",
        "probe",
        "null_embedding",
    )
    shared_low_rank_keywords = (
        "cross_lora",
        "adaLN_modulation",
        "final_modulation",
    )
    phase_routed_keywords = (
        "phase_attn_lora",
        "phase_cross_kv",
    )
    conv_shortcut_keywords = (
        "dw_conv",
        "refinement_head.refine",
        "skip_projs",
    )

    pooler_param_ids = (
        {id(p) for p in text_encoder.pooler.parameters()}
        if hasattr(text_encoder, "pooler") and text_encoder.pooler is not None
        else set()
    )

    eff_expert_wd = float(expert_weight_decay) if (expert_weight_decay is not None and expert_weight_decay >= 0.0) else weight_decay

    for name, param in model_raw.named_parameters():
        if not param.requires_grad or id(param) in pooler_param_ids:
            continue

        is_conv_shortcut = any(kw in name for kw in conv_shortcut_keywords)
        is_expert = "phase_experts" in name
        is_no_decay = (param.ndim <= 1) or any(kw in name for kw in no_decay_keywords)
        is_phase_routed = any(kw in name for kw in phase_routed_keywords)
        is_shared_low_rank = any(kw in name for kw in shared_low_rank_keywords)

        if is_expert:
            if is_no_decay:
                expert_no_decay.append(param)
            else:
                expert_decay.append(param)
        elif is_conv_shortcut:
            if is_no_decay:
                base_no_decay.append(param)
            else:
                conv_refine_decay.append(param)
        elif is_phase_routed:
            phase_routed_lora.append(param)
        elif is_shared_low_rank:
            shared_lora_adaln.append(param)
        else:
            if is_no_decay:
                base_no_decay.append(param)
            else:
                base_decay.append(param)

    if base_decay:
        groups.append({"params": base_decay, "weight_decay": weight_decay, "lr": lr, "lr_scale": 1.0, "name": "base_decay"})
    if base_no_decay:
        groups.append({"params": base_no_decay, "weight_decay": 0.0, "lr": lr, "lr_scale": 1.0, "name": "base_no_decay"})
    if shared_lora_adaln:
        groups.append({"params": shared_lora_adaln, "weight_decay": weight_decay * 0.25, "lr": lr, "lr_scale": 1.0, "name": "lora_adaln"})
    if phase_routed_lora:
        groups.append({"params": phase_routed_lora, "weight_decay": weight_decay * 0.25, "lr": lr, "lr_scale": 1.0, "name": "phase_routed_lora"})
    if conv_refine_decay:
        groups.append({"params": conv_refine_decay, "weight_decay": weight_decay, "lr": lr, "lr_scale": 1.0, "name": "conv_refine_decay"})
    if expert_decay:
        groups.append({"params": expert_decay, "weight_decay": eff_expert_wd, "lr": lr, "lr_scale": 1.0, "name": "expert_decay"})
    if expert_no_decay:
        groups.append({"params": expert_no_decay, "weight_decay": 0.0, "lr": lr, "lr_scale": 1.0, "name": "expert_no_decay"})

    for prefix, module in [("text_proj", text_encoder.proj), ("text_pooler", text_encoder.pooler)]:
        if module is None:
            continue
        decay, no_decay = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or any(kw in name for kw in no_decay_keywords):
                no_decay.append(param)
            else:
                decay.append(param)
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay, "lr": lr_text, "lr_scale": 1.0, "name": f"{prefix}_decay"})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0, "lr": lr_text, "lr_scale": 1.0, "name": f"{prefix}_no_decay"})

    return groups


def compute_phase_aware_loss(
    v_pred: torch.Tensor,
    target_v: torch.Tensor,
    t_phase: torch.Tensor,
    x_t: Optional[torch.Tensor] = None,
    x_1: Optional[torch.Tensor] = None,
    huber_c: float = 1.0,
    cos_weight: float = 0.10,
) -> torch.Tensor:
    v_p = v_pred.float()
    v_t = target_v.float()
    delta_v = v_p - v_t
    t_1d = t_phase.float().view(-1)

    diff_sq = delta_v.square()
    c_sq = huber_c * huber_c
    loss_huber = (c_sq * (torch.sqrt(1.0 + diff_sq / c_sq) - 1.0)).mean(dim=(1, 2, 3))

    loss_cos_global = 1.0 - F.cosine_similarity(v_p.flatten(1), v_t.flatten(1), dim=-1, eps=1e-6)

    loss_cos_patch = (1.0 - F.cosine_similarity(v_p, v_t, dim=1, eps=0.15)).mean(dim=(1, 2))

    total_loss = loss_huber + cos_weight * (0.25 * loss_cos_global + 0.75 * t_1d * loss_cos_patch)

    if x_t is not None and x_1 is not None:
        low_2x2_err = F.avg_pool2d(delta_v, kernel_size=2, stride=1)
        low_3x3_err = F.avg_pool2d(delta_v, kernel_size=3, stride=1, padding=1, count_include_pad=False)

        loss_low = (
            low_2x2_err.square().mean(dim=(1, 2, 3))
            + low_3x3_err.square().mean(dim=(1, 2, 3))
        )
        w_low = 0.25 * ((1.0 - t_1d).clamp(min=0.0) ** 3)

        dx_err = delta_v[:, :, :, 1:] - delta_v[:, :, :, :-1]
        dy_err = delta_v[:, :, 1:, :] - delta_v[:, :, :-1, :]
        d_diag1 = delta_v[:, :, 1:, 1:] - delta_v[:, :, :-1, :-1]
        d_diag2 = delta_v[:, :, 1:, :-1] - delta_v[:, :, :-1, 1:]
        loss_edge = 0.35 * (
            dx_err.square().mean(dim=(1, 2, 3))
            + dy_err.square().mean(dim=(1, 2, 3))
        ) + 0.15 * (
            d_diag1.square().mean(dim=(1, 2, 3))
            + d_diag2.square().mean(dim=(1, 2, 3))
        )
        w_edge = 0.20 * (t_1d * torch.sqrt(t_1d.clamp(min=1e-6)))

        total_loss = total_loss + w_low * loss_low + w_edge * loss_edge

    return total_loss.mean()

def sample_continuous_t(
    batch_size: int,
    device: torch.device,
    uniform_prob: float = 1.0,
    shift: float = 1.0,
) -> torch.Tensor:
    if uniform_prob >= 1.0:
        t = torch.rand(batch_size, device=device)
    elif uniform_prob <= 0.0:
        t = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        t_uniform = torch.rand(batch_size, device=device)
        t_logit = torch.sigmoid(torch.randn(batch_size, device=device))
        use_uniform = torch.rand(batch_size, device=device) < uniform_prob
        t = torch.where(use_uniform, t_uniform, t_logit)

    if abs(shift - 1.0) > 1e-6:
        mu = max(float(shift), 1e-4)
        t = t / (mu - (mu - 1.0) * t)

    return t.clamp(0.001, 0.999)


def sample_phase_stratified_t(
    batch_size: int,
    device: torch.device,
    phase_idx: int,
    num_phases: int = 3,
    shift: float = 1.0,
    overlap: float = 0.03,
) -> Tuple[torch.Tensor, int]:
    p_low = (phase_idx / float(num_phases)) - (overlap if phase_idx > 0 else 0.0)
    p_high = ((phase_idx + 1) / float(num_phases)) + (overlap if phase_idx < num_phases - 1 else 0.0)

    p_low = max(p_low, 0.001)
    p_high = min(p_high, 0.999)

    u = torch.rand(batch_size, device=device)
    eff_shift = (
        shift
        if phase_idx == 0
        else (1.0 + 0.45 * (shift - 1.0) if phase_idx == 1 else 1.0 + 0.15 * (shift - 1.0))
    )
    if abs(eff_shift - 1.0) > 1e-6:
        mu = max(float(eff_shift), 1e-4)
        u = u / (mu - (mu - 1.0) * u)

    t = p_low + u * (p_high - p_low)
    return t.clamp(p_low, p_high), phase_idx


def scaler_is_enabled(scaler) -> bool:
    if scaler is None:
        return False

    if hasattr(scaler, "is_enabled"):
        return bool(scaler.is_enabled())

    return bool(getattr(scaler, "_enabled", False))


def train_step(
    model,
    ema_source_model,
    text_encoder,
    rf,
    batch_latents,
    batch_text_features,
    batch_masks,
    empty_text,
    empty_mask,
    optimizer,
    lr_scheduler,
    device,
    amp_dtype,
    logger,
    latent_mean,
    latent_std,
    ema_model=None,
    ema_text_proj=None,
    ema_text_pooler=None,
    scaler=None,
    accumulate_steps: int = 1,
    sync_gradients: bool = True,
    t_shift: float = 1.5,
    t_uniform_prob: float = 1.0,
    cfg_drop_prob: float = 0.15,
    clip_norm: float = 1.0,
    target_phase: Optional[int] = None,
    is_distributed: bool = False,
    world_size: int = 1,
):
    latents = batch_latents.to(device, non_blocking=True).float()
    latents.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    latents = (latents - latent_mean) / latent_std

    B = batch_latents.size(0)
    seq_cap = batch_masks.shape[-1]

    if not USE_COMPILE and os.environ.get("DYNAMIC_SEQ_BUCKETS", "0") == "1":
        max_tok_cpu = int(batch_masks.sum(dim=-1).max().item())
        real_len = 64 if (max_tok_cpu <= 64 and seq_cap >= 64) else seq_cap
    else:
        real_len = seq_cap

    if real_len < seq_cap:
        batch_masks = batch_masks[:, :real_len].contiguous()
        if batch_text_features.dim() == 4:
            batch_text_features = batch_text_features[:, :, :real_len, :].contiguous()
        else:
            batch_text_features = batch_text_features[:, :real_len, :].contiguous()

    masks = batch_masks.to(device, non_blocking=True).long()
    target_text_dtype = amp_dtype if device.type == "cuda" else torch.float32
    text_features = batch_text_features.to(device=device, non_blocking=True).to(dtype=target_text_dtype)
    text_features.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    if cfg_drop_prob > 0.0:
        drop = torch.rand(B, device=device) < cfg_drop_prob
        if text_features.dim() == 4 and empty_text.dim() == 3:
            empty_view = empty_text[:, :real_len, :].to(device=device, dtype=text_features.dtype).unsqueeze(0)
        elif text_features.dim() == 4 and empty_text.dim() == 2:
            n_p = text_features.shape[1]
            empty_view = empty_text[:real_len, :].to(device=device, dtype=text_features.dtype).unsqueeze(0).unsqueeze(0).expand(1, n_p, -1, -1)
        else:
            empty_view = empty_text[:real_len, :].to(device=device, dtype=text_features.dtype).unsqueeze(0)
        drop_mask_4d = drop.view(B, *([1] * (text_features.dim() - 1)))
        text_features = torch.where(drop_mask_4d, empty_view, text_features)
        masks = masks.masked_fill(drop.unsqueeze(1), 0)

    if target_phase is not None:
        num_phases = getattr(ema_source_model, "num_phases", 3)
        t_phase, phase_idx_val = sample_phase_stratified_t(
            batch_size=B,
            device=device,
            phase_idx=target_phase,
            num_phases=num_phases,
            shift=t_shift,
        )
    else:
        t_phase = sample_continuous_t(
            batch_size=B,
            device=device,
            uniform_prob=t_uniform_prob,
            shift=t_shift,
        )
        phase_idx_val = None

    x_t, _, target_v = rf.add_noise(latents, t_phase)

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda"),
    ):
        cond_pooled, cond_seq, mask = text_encoder.forward_cached(
            text_features,
            masks,
        )

        v_pred = model(
            z=x_t,
            cond_pooled=cond_pooled,
            cond_seq=cond_seq,
            t_phase=t_phase,
            mask=mask,
            phase_idx=phase_idx_val,
        )

        loss = compute_phase_aware_loss(v_pred, target_v, t_phase, x_t=x_t, x_1=latents)
        loss = loss / float(accumulate_steps)

    if scaler_is_enabled(scaler):
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if not sync_gradients:
        return loss.detach() * float(accumulate_steps), 0.0, False

    if scaler_is_enabled(scaler):
        scaler.unscale_(optimizer)

    params_to_clip = list(
        {
            id(p): p
            for p in (
                list(ema_source_model.parameters())
                + list(text_encoder.proj.parameters())
                + list(text_encoder.pooler.parameters())
            )
            if p.requires_grad and p.grad is not None
        }.values()
    )

    if is_distributed and dist.is_initialized() and world_size > 1:
        model_param_ids = {id(p) for p in ema_source_model.parameters()}
        text_params = [
            p for p in list(text_encoder.proj.parameters()) + list(text_encoder.pooler.parameters())
            if p.requires_grad and p.grad is not None and id(p) not in model_param_ids
        ]
        for p in text_params:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)

        is_fsdp = FSDP_AVAILABLE and isinstance(model, FSDP)
        is_ddp = DDP is not None and isinstance(model, DDP)
        if is_ddp and accumulate_steps > 1 and target_phase is not None:
            for name, p in ema_source_model.named_parameters():
                if p.requires_grad and p.grad is not None and any(
                    tag in name for tag in ("phase_experts.", "phase_attn_lora.", "phase_cross_kv.")
                ):
                    if f".{target_phase}." not in name:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)
        elif not is_fsdp and not is_ddp:
            for p in ema_source_model.parameters():
                if p.requires_grad and p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        params_to_clip,
        max_norm=clip_norm,
    )

    loss_val = loss.item() * float(accumulate_steps)
    grad_norm_val = grad_norm.item() if torch.isfinite(grad_norm) else float("nan")
    is_valid_step = math.isfinite(loss_val) and math.isfinite(grad_norm_val)

    if not is_valid_step:
        logger.warning(f"Non-finite loss ({loss_val}) or grad_norm ({grad_norm_val}) detected. Resetting gradients.")
        if scaler_is_enabled(scaler):
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return float("nan"), 0.0, True

    if scaler_is_enabled(scaler):
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        step_was_skipped = scale_after < scale_before
    else:
        step_was_skipped = False
        optimizer.step()

    if step_was_skipped:
        optimizer.zero_grad(set_to_none=True)
    else:
        if lr_scheduler is not None:
            try:
                lr_scheduler.step(loss_val)
            except TypeError:
                lr_scheduler.step()

        if ema_model is not None:
            ema_model.update_parameters(ema_source_model)
        if ema_text_proj is not None:
            ema_text_proj.update_parameters(text_encoder.proj)
        if ema_text_pooler is not None:
            ema_text_pooler.update_parameters(text_encoder.pooler)

        optimizer.zero_grad(set_to_none=True)

    return (
        loss_val,
        grad_norm_val,
        step_was_skipped,
    )


@torch.inference_mode()
def generate_validation_samples(
    ema_model,
    text_encoder,
    vae,
    rf,
    device,
    amp_dtype,
    step,
    save_dir,
    val_text_features,
    val_masks,
    fixed_noise,
    logger,
    latent_mean,
    latent_std,
    ema_text_proj=None,
    ema_text_pooler=None,
    val_latents=None,
    validation_steps: int = 12,
    cfg_scale: float = 1.0,
    cfg_rescale: float = 0.0,
    shift: float = 1.0,
    rescale_norm: bool = False,
    val_solver: str = "unipc",
):
    logger.info(f"Generating validation samples at step {step}...")

    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)

    base_model = unwrap_model(ema_model)
    base_model.eval()

    text_encoder.eval()

    eff_val_masks = val_masks.clone()
    if eff_val_masks.shape[0] % 2 == 0 and fixed_noise.shape[0] == eff_val_masks.shape[0] // 2:
        eff_val_masks[: fixed_noise.shape[0]] = 0

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda"),
    ):
        orig_proj = None
        orig_pooler = None
        orig_base_pooler = getattr(base_model, "pooler", None)

        try:
            if ema_text_proj is not None:
                orig_proj = text_encoder.proj
                text_encoder.proj = unwrap_model(ema_text_proj)
                text_encoder.proj.eval()

            if ema_text_pooler is not None:
                orig_pooler = text_encoder.pooler
                ema_pooler_unwrapped = unwrap_model(ema_text_pooler)
                ema_pooler_unwrapped.eval()
                text_encoder.pooler = ema_pooler_unwrapped
                base_model.pooler = ema_pooler_unwrapped

            cond_pooled, cond_seq, mask = text_encoder.forward_cached(
                val_text_features,
                eff_val_masks,
            )

            if val_latents is not None:
                B_val = val_latents.size(0)

                t_phase = sample_continuous_t(
                    batch_size=B_val,
                    device=device,
                    uniform_prob=1.0,
                    shift=shift,
                )

                val_latents_norm = (val_latents - latent_mean) / latent_std
                val_gen = torch.Generator(device=device).manual_seed(1337)

                x_t, _, target_v = rf.add_noise(
                    val_latents_norm,
                    t_phase,
                    generator=val_gen,
                )

                cond_idx_start = B_val if cond_pooled.shape[0] == 2 * B_val else 0

                v_pred = base_model(
                    z=x_t,
                    cond_pooled=cond_pooled[cond_idx_start : cond_idx_start + B_val],
                    cond_seq=cond_seq[cond_idx_start : cond_idx_start + B_val],
                    t_phase=t_phase,
                    mask=mask[cond_idx_start : cond_idx_start + B_val],
                )

                val_loss = F.mse_loss(v_pred.float(), target_v.float()).item()
                logger.info(f"Val Loss: {val_loss:.4f}")

            is_cfg = cfg_scale > 1.0
            if is_cfg:
                B = cond_pooled.shape[0] // 2
            else:
                if cond_pooled.shape[0] % 2 == 0 and fixed_noise.shape[0] == cond_pooled.shape[0] // 2:
                    B = cond_pooled.shape[0] // 2
                else:
                    B = cond_pooled.shape[0]

            if B <= 0:
                logger.warning("Validation batch size is zero. Skipping.")
                return

            shape = (
                B,
                fixed_noise.shape[1],
                fixed_noise.shape[2],
                fixed_noise.shape[3],
            )

            init_noise = fixed_noise[:B]
            if init_noise.shape[0] < B:
                extra_noise = torch.randn(
                    (B - init_noise.shape[0], *fixed_noise.shape[1:]),
                    device=device,
                    dtype=fixed_noise.dtype,
                    generator=torch.Generator(device=device).manual_seed(42),
                )
                init_noise = torch.cat([init_noise, extra_noise], dim=0)

            val_solver_clean = str(val_solver).lower().strip()
            if val_solver_clean in ("unipc", "phase_ab2", "dpm++_2m", "ab2"):
                sampler = PhaseAwareSampler(
                    model=base_model,
                    solver_type="unipc" if val_solver_clean == "unipc" else "dpm++_2m",
                    time_shift=shift,
                    rescaled_phi=cfg_rescale,
                )
                u_pool = cond_pooled[:B] if cond_pooled.shape[0] == 2 * B else None
                c_pool = cond_pooled[B:] if cond_pooled.shape[0] == 2 * B else cond_pooled
                u_seq = cond_seq[:B] if (cond_seq is not None and cond_seq.shape[0] == 2 * B) else None
                c_seq = cond_seq[B:] if (cond_seq is not None and cond_seq.shape[0] == 2 * B) else cond_seq
                c_mask = mask[B:] if (mask is not None and mask.shape[0] == 2 * B) else mask

                x_0_pred = sampler.sample(
                    z=init_noise.to(device=device, dtype=torch.float32),
                    cond_seq=c_seq,
                    cond_pooled=c_pool,
                    cond_mask=c_mask,
                    uncond_pooled=u_pool,
                    uncond_seq=u_seq,
                    cfg_scale=cfg_scale,
                    cfg_rescale=cfg_rescale,
                    steps=validation_steps,
                    dynamic_cfg=True,
                    use_block_cache=False,
                )
            else:
                text_kvs = base_model.encode_text(cond_seq)
                x_0_pred = rf.phase_euler_sample(
                    model=base_model,
                    cond_pooled=cond_pooled,
                    cond_seq=cond_seq,
                    shape=shape,
                    device=device,
                    steps=validation_steps,
                    mask=mask,
                    text_kvs=text_kvs,
                    cfg_scale=cfg_scale,
                    cfg_rescale=cfg_rescale,
                    shift=shift,
                    rescale_norm=rescale_norm,
                    initial_noise=init_noise,
                )

            x_0_pred = x_0_pred * latent_std + latent_mean
        finally:
            if orig_proj is not None:
                text_encoder.proj = orig_proj
            if orig_pooler is not None:
                text_encoder.pooler = orig_pooler
            if orig_base_pooler is not None:
                base_model.pooler = orig_base_pooler

    if hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
        except Exception:
            pass
    chunk_size = max(1, int(os.environ.get("VAL_DECODE_CHUNK_SIZE", "1")))
    decoded_chunks = []
    for chunk in torch.split(x_0_pred.float(), chunk_size, dim=0):
        dec = vae.decode(chunk)
        decoded_chunks.append(dec.cpu())
    images_tensor = torch.cat(decoded_chunks, dim=0)

    save_path = os.path.join(save_dir, "samples", f"sample_step_{step}.jpg")

    torchvision.utils.save_image(
        images_tensor,
        save_path,
        nrow=3,
        normalize=False,
    )

    logger.info(f"Samples saved: {save_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_time(seconds: float) -> str:
    seconds = int(seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"

    return f"{m}m {s}s"


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_reload_dataloader(
    dataloader,
    cache_dir: str,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    logger,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
):
    try:
        new_meta = read_meta(cache_dir)
        new_count = int(new_meta.get("count", 0))
        old_count = len(dataloader.dataset)

        if new_count > old_count:
            logger.info(
                f"New data detected in cache: {old_count} -> {new_count}. "
                "Reloading dataset..."
            )

            try:
                dataloader.dataset.close()
            except Exception:
                pass

            shutdown_dataloader(dataloader)

            dataloader = get_cached_dataloader(
                cache_dir=cache_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                split="train",
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                is_distributed=is_distributed,
                world_size=world_size,
                rank=rank,
            )

            return dataloader, True

    except Exception as e:
        logger.warning(f"Failed to check meta.json for reload: {e}")

    return dataloader, False

class AttentionMonitor:
    def __init__(self, save_dir: str, logger: logging.Logger):
        self.save_dir = os.path.join(save_dir, "attention_analysis")
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logger
        self.history: List[Dict[str, Any]] = []
        self.executor = ThreadPoolExecutor(max_workers=1)

    def close(self):
        try:
            self.executor.shutdown(wait=True)
        except Exception:
            pass

    def __del__(self):
        self.close()

    def record_step(self, model: torch.nn.Module, step: int) -> Dict[str, float]:
        base_mod = unwrap_model(model)
        if not hasattr(base_mod, "get_attention_statistics"):
            return {}

        stats = base_mod.get_attention_statistics()
        if not stats:
            return {}

        stats["step"] = step
        self.history.append(stats)
        self.logger.info(
            f"[AttentionMonitor @ step {step}] "
            f"Mean Cross-Entropy: {stats.get('mean_cross_entropy', 0.0):.4f} | "
            f"Mean Max Prob: {stats.get('mean_cross_max_prob', 0.0):.4f} | "
            f"Mean Self-Entropy: {stats.get('mean_self_entropy', 0.0):.4f}"
        )

        json_path = os.path.join(self.save_dir, f"attention_stats_step_{step}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save attention JSON: {e}")

        try:
            if hasattr(base_mod, "get_attention_maps"):
                maps = base_mod.get_attention_maps()
                if maps:
                    cpu_maps = {}
                    for blk_name, blk_maps in maps.items():
                        blk_cpu = {}
                        if "cross_attn" in blk_maps:
                            blk_cpu["cross_attn"] = (
                                blk_maps["cross_attn"].detach().float().mean(dim=(0, 1)).cpu().numpy()
                            )
                        if "self_attn" in blk_maps:
                            blk_cpu["self_attn"] = (
                                blk_maps["self_attn"].detach().float().mean(dim=(0, 1)).cpu().numpy()
                            )
                        if blk_cpu:
                            cpu_maps[blk_name] = blk_cpu

                    if cpu_maps:
                        self.executor.submit(self._async_render_heatmaps, cpu_maps, step, getattr(base_mod, "num_sinks", 8))
        except Exception as e:
            self.logger.warning(f"Failed to offload attention heatmaps: {e}")

        return stats

    def _async_render_heatmaps(self, cpu_maps: Dict[str, Dict[str, np.ndarray]], step: int, num_sinks: int = 8) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            self.logger.warning(f"matplotlib unavailable for attention heatmaps: {e}")
            return

        try:
            for blk_name, blk_maps in cpu_maps.items():
                if "cross_attn" in blk_maps:
                    ca = blk_maps["cross_attn"]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(ca, aspect="auto", cmap="viridis")
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"Cross-Attention Matrix ({blk_name}) @ Step {step}")
                    ax.set_xlabel("Text Token Index (S)")
                    ax.set_ylabel("Spatial Image Tokens (N=256, 16x16 grid)")
                    fig_path = os.path.join(self.save_dir, f"{blk_name}_cross_heatmap_step_{step}.png")
                    plt.tight_layout()
                    plt.savefig(fig_path, dpi=120)
                    plt.close(fig)

                    spatial_side = int(math.isqrt(ca.shape[0]))
                    spatial_len = spatial_side * spatial_side
                    if spatial_side > 0 and ca.shape[1] > max(1, num_sinks):
                        ca_sp = ca[:spatial_len, :-num_sinks] if num_sinks > 0 else ca[:spatial_len]
                        top_token_idx = int(ca_sp.mean(axis=0).argmax())
                        spatial_2d = ca_sp[:, top_token_idx].reshape(spatial_side, spatial_side)
                        fig_sp, ax_sp = plt.subplots(figsize=(6, 5))
                        im_sp = ax_sp.imshow(spatial_2d, cmap="inferno", interpolation="bilinear")
                        plt.colorbar(im_sp, ax=ax_sp)
                        ax_sp.set_title(f"2D Spatial Cross-Attention ({blk_name}) for Token {top_token_idx} @ Step {step}")
                        ax_sp.set_xlabel(f"Latent Width ({spatial_side}x{spatial_side} grid)")
                        ax_sp.set_ylabel(f"Latent Height ({spatial_side}x{spatial_side} grid)")
                        fig_sp_path = os.path.join(self.save_dir, f"{blk_name}_spatial_token{top_token_idx}_step_{step}.png")
                        plt.tight_layout()
                        plt.savefig(fig_sp_path, dpi=120)
                        plt.close(fig_sp)

                if "self_attn" in blk_maps:
                    sa = blk_maps["self_attn"]
                    fig, ax = plt.subplots(figsize=(8, 7))
                    im = ax.imshow(sa, aspect="auto", cmap="magma")
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"Self-Attention Map ({blk_name}) @ Step {step}")
                    ax.set_xlabel("Spatial Token (Key)")
                    ax.set_ylabel("Spatial Token (Query)")
                    fig_path = os.path.join(self.save_dir, f"{blk_name}_self_heatmap_step_{step}.png")
                    plt.tight_layout()
                    plt.savefig(fig_path, dpi=120)
                    plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Error in async heatmap rendering at step {step}: {e}")


def setup_compiled_model(
    model: torch.nn.Module,
    logger: logging.Logger,
    device: torch.device,
    use_compile: bool = True,
    backend: str = "auto",
    mode: str = "default",
    dynamic: bool = False,
    latent_channels: int = 32,
    latent_size: int = 16,
    context_dim: int = 2560,
    seq_len: int = 128,
    batch_size: int = 2,
    amp_dtype: Optional[torch.dtype] = None,
    text_shape: Optional[Tuple[int, ...]] = None,
) -> torch.nn.Module:
    if not use_compile or not hasattr(torch, "compile"):
        logger.info("[torch.compile] Compilation disabled or not supported in this torch build.")
        return model

    if device.type != "cuda":
        logger.info("[torch.compile] CUDA is not available. Skipping compilation on CPU.")
        return model

    logger.info(f"[torch.compile] Configuring compilation (backend='{backend}', mode='{mode}', dynamic={dynamic})...")

    try:
        import compile_optimizer
        compile_optimizer.setup_fast_compile(enable_caching=True, verbose=False)
    except Exception:
        pass

    candidates = ["inductor", "aot_eager"] if backend == "auto" else [backend]

    b_warmup = max(1, batch_size)
    for cand_backend in candidates:
        try:
            logger.info(f"[torch.compile] Attempting compile with backend='{cand_backend}'...")
            compile_kwargs = {"backend": cand_backend, "dynamic": dynamic}
            if cand_backend == "inductor" and mode:
                compile_kwargs["mode"] = mode

            compiled = torch.compile(model, **compile_kwargs)

            seq_dtype = amp_dtype if (amp_dtype is not None and device.type == "cuda") else torch.float32
            dummy_z = torch.zeros(b_warmup, latent_channels, latent_size, latent_size, device=device, dtype=torch.float32, requires_grad=False)
            dummy_pooled = torch.zeros(b_warmup, context_dim, device=device, dtype=seq_dtype, requires_grad=True)

            raw_m = getattr(model, "_orig_mod", model)
            raw_m = getattr(raw_m, "module", raw_m)
            has_mixer = getattr(raw_m, "phase_layer_mixer", None) is not None

            if text_shape is not None and len(text_shape) in (2, 3):
                dummy_seq = torch.zeros(b_warmup, *text_shape, device=device, dtype=seq_dtype, requires_grad=False)
            elif has_mixer:
                dummy_seq = torch.zeros(b_warmup, 3, seq_len, context_dim, device=device, dtype=seq_dtype, requires_grad=False)
            else:
                dummy_seq = torch.zeros(b_warmup, seq_len, context_dim, device=device, dtype=seq_dtype, requires_grad=False)

            dummy_mask = torch.ones(b_warmup, seq_len, dtype=torch.long, device=device)

            compiled.train()
            n_p = getattr(raw_m, "num_phases", 3)
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if (amp_dtype is not None and device.type == "cuda")
                else contextlib.nullcontext()
            )
            for p in range(n_p):
                dummy_t = torch.full((b_warmup,), float(p) / float(max(1, n_p - 1)), device=device, dtype=torch.float32)
                with autocast_ctx:
                    out_warmup = compiled(
                        z=dummy_z,
                        cond_pooled=dummy_pooled,
                        cond_seq=dummy_seq,
                        t_phase=dummy_t,
                        mask=dummy_mask,
                        phase_idx=p,
                    )
                loss_warmup = out_warmup.float().sum()
                loss_warmup.backward()
                model.zero_grad(set_to_none=True)

            logger.info(f"[torch.compile] Successfully verified compiled model with backend='{cand_backend}'!")
            return compiled
        except Exception as e:
            logger.warning(
                f"[torch.compile] Backend '{cand_backend}' verification failed on Windows: {e}. "
                f"Falling back..."
            )

    logger.warning("[torch.compile] All compilation attempts failed or skipped. Continuing in eager mode.")
    return model


def run_attention_test_training(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rf: RectifiedFlow4D,
    save_dir: str,
    logger: logging.Logger,
    device: torch.device,
    amp_dtype: torch.dtype,
    steps: int = 5,
    latent_channels: int = 32,
    latent_size: int = 16,
    context_dim: int = 2560,
    seq_len: int = 128,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info(f"STARTING ATTENTION MECHANISM ANALYSIS TEST RUN ({steps} STEPS)")
    logger.info(f"Resolution: 512x512 | Latent: [B, {latent_channels}, {latent_size}, {latent_size}] ({latent_size*latent_size} tokens)")
    logger.info(f"Device: {device} | Precision: {amp_dtype}")
    logger.info("=" * 70)

    monitor = AttentionMonitor(save_dir, logger)
    base_model = unwrap_model(model)
    base_model.set_attention_tracking(True)

    B = 2
    last_stats = {}
    for step in range(1, steps + 1):
        optimizer.zero_grad()

        latents = torch.randn(B, latent_channels, latent_size, latent_size, device=device)
        text_features = torch.randn(B, seq_len, context_dim, device=device)
        masks = torch.ones(B, seq_len, dtype=torch.long, device=device)
        text_features[:, :4, :] += 2.5

        t_phase = torch.tensor([0.25, 0.75], device=device)
        x_t, _, target_v = rf.add_noise(latents, t_phase)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            cond_pooled, cond_seq, mask = text_encoder.forward_cached(text_features, masks)
            v_pred = model(
                z=x_t,
                cond_pooled=cond_pooled,
                cond_seq=cond_seq,
                t_phase=t_phase,
                mask=mask,
            )
            loss = F.mse_loss(v_pred.float(), target_v.float())

        loss.backward()

        grad_norms = {}
        for i, blk in enumerate(base_model.blocks):
            if hasattr(blk, "cross") and getattr(blk, "use_cross", False):
                if blk.cross.q.weight.grad is not None:
                    grad_norms[f"blk_{i}_cross_q"] = blk.cross.q.weight.grad.norm().item()
                if blk.cross.proj.weight.grad is not None:
                    grad_norms[f"blk_{i}_cross_proj"] = blk.cross.proj.weight.grad.norm().item()
        if hasattr(base_model, "phase_cross_kv"):
            for p_i, p_kv in enumerate(base_model.phase_cross_kv):
                if hasattr(p_kv, "weight") and p_kv.weight.grad is not None:
                    grad_norms[f"phase_cross_kv_{p_i}"] = p_kv.weight.grad.norm().item()

        optimizer.step()

        stats = monitor.record_step(model, step)
        stats["loss"] = loss.item()
        stats["cross_grad_norms"] = grad_norms
        last_stats = stats

        logger.info(
            f"[Attention Test Step {step}/{steps}] Loss: {loss.item():.4f} | "
            f"Cross-Entropy: {stats.get('mean_cross_entropy', 0.0):.4f} | "
            f"Cross-MaxProb: {stats.get('mean_cross_max_prob', 0.0):.4f} | "
            f"Self-Entropy: {stats.get('mean_self_entropy', 0.0):.4f} | "
            f"Active Cross Grads: {len(grad_norms)}"
        )

    base_model.set_attention_tracking(False)
    logger.info("=" * 70)
    logger.info("ATTENTION MECHANISM ANALYSIS TEST COMPLETED SUCCESSFULLY!")
    logger.info(f"Attention heatmaps and statistics saved to: {monitor.save_dir}")
    logger.info("=" * 70)
    monitor.close()
    return last_stats


def main():
    parser = argparse.ArgumentParser(description="EdgeFlow-Phase DiT Training & Attention Mechanism Analysis")
    parser.add_argument("--test-attention", action="store_true", help="Run attention tracking test pass and exit")
    parser.add_argument("--attention-steps", type=int, default=int(os.environ.get("ATTENTION_STEPS", "5")), help="Steps for attention test")
    parser.add_argument("--track-attention-every", type=int, default=int(os.environ.get("TRACK_ATTENTION_EVERY", "0")), help="Track attention every N training steps (0=disabled)")
    parser.add_argument("--vae", type=str.lower, default=os.environ.get("VAE_TYPE", "dcae"), choices=["taesd", "sd", "sdxl", "flux", "dcae"])
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("EXPECTED_IMAGE_SIZE", "512")))
    parser.add_argument("--compile", action="store_true", default=(os.environ.get("USE_COMPILE", "1") == "1"), help="Enable torch.compile")
    parser.add_argument("--no-compile", action="store_false", dest="compile", help="Disable torch.compile")
    parser.add_argument("--compile-backend", type=str, default=os.environ.get("COMPILE_BACKEND", "auto"))
    parser.add_argument("--compile-mode", type=str, default=os.environ.get("COMPILE_MODE", "default"))
    parser.add_argument("--compile-dynamic", action="store_true", default=(os.environ.get("COMPILE_DYNAMIC", "0") == "1"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "32")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "0" if sys.platform == "win32" else "4")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("TOTAL_STEPS", "500000")))
    parser.add_argument("--save-dir", type=str, default=os.environ.get("SAVE_DIR", "EdgeFlow-Phase-Experts-512-100M"))
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("CACHE_DIR", "hf_cache_512"))
    parser.add_argument("--dim", type=int, default=int(os.environ["DIM"]) if "DIM" in os.environ else None)
    parser.add_argument("--depth", type=int, default=int(os.environ["DEPTH"]) if "DEPTH" in os.environ else None)
    parser.add_argument("--heads", type=int, default=int(os.environ["HEADS"]) if "HEADS" in os.environ else None)
    parser.add_argument("--kv-heads", type=int, default=int(os.environ["KV_HEADS"]) if "KV_HEADS" in os.environ else None)
    parser.add_argument("--context-dim", type=int, default=int(os.environ["CONTEXT_DIM"]) if "CONTEXT_DIM" in os.environ else None)
    parser.add_argument("--ffn-hidden", type=int, default=int(os.environ["FFN_HIDDEN"]) if "FFN_HIDDEN" in os.environ else None)
    parser.add_argument("--ada-rank", type=int, default=int(os.environ["ADA_RANK"]) if "ADA_RANK" in os.environ else None)
    parser.add_argument("--expert-hidden-list", type=str, default=os.environ.get("EXPERT_HIDDEN_LIST", None))
    parser.add_argument("--val-steps", type=int, default=int(os.environ.get("VAL_STEPS", "24")))
    parser.add_argument("--val-cfg-scale", type=float, default=float(os.environ.get("VAL_CFG_SCALE", "2.5")))
    parser.add_argument("--val-cfg-rescale", type=float, default=float(os.environ.get("VAL_CFG_RESCALE", "0.0")))
    parser.add_argument("--val-shift", type=float, default=float(os.environ.get("VAL_SHIFT", os.environ.get("T_SHIFT", "1.5"))))
    parser.add_argument("--val-rescale-norm", action="store_true", default=(os.environ.get("VAL_RESCALE_NORM", "0") == "1"))
    parser.add_argument("--cfg-drop-prob", type=float, default=float(os.environ.get("CFG_DROP_PROB", "0.08")))
    parser.add_argument("--fixed-lr", type=float, default=None, help="Fix learning rate at a constant value (disables warmup/decay)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides default/env)")
    parser.add_argument("--lr-text", type=float, default=None, help="Text adapter learning rate")
    parser.add_argument("--min-lr", type=float, default=float(os.environ.get("MIN_LR", "5e-5")), help="Minimum learning rate floor")
    parser.add_argument("--decay-start-step", type=int, default=None, help="Step at which cosine decay begins")
    parser.add_argument("--reset-lr", action="store_true", default=(os.environ.get("RESET_LR", "0") == "1"), help="Force reset optimizer and scheduler LR upon checkpoint resume")
    parser.add_argument("--from-scratch", action="store_true", default=(os.environ.get("FROM_SCRATCH", "0") == "1"), help="Force training from scratch (step 0, epoch 0) without resuming from any checkpoint")
    parser.add_argument("--total-steps", type=int, default=None, dest="total_steps_alias", help="Total steps (alias for --steps)")
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint every N steps")
    parser.add_argument("--val-every", type=int, default=None, help="Generate validation samples every N steps")
    parser.add_argument("--val-solver", type=str.lower, default=os.environ.get("VAL_SOLVER", "unipc"), choices=["unipc", "phase_ab2", "dpm++_2m", "euler"], help="ODE Solver for validation generation ('unipc', 'phase_ab2', or 'euler')")
    parser.add_argument("--distributed", action="store_true", default=False, help="Force enable distributed training")
    parser.add_argument("--dist-backend", type=str, default=os.environ.get("DIST_BACKEND", "nccl"), help="Distributed backend (nccl/gloo)")
    parser.add_argument("--strategy", type=str, default=os.environ.get("DIST_STRATEGY", "ddp"), choices=["ddp", "fsdp", "none"], help="Distributed parallel strategy")
    parser.add_argument("--fsdp-sharding", type=str, default=os.environ.get("FSDP_SHARDING", "full_shard"), choices=["full_shard", "shard_grad_op", "hybrid_shard"], help="FSDP sharding strategy")
    parser.add_argument("--phase-stratified", action="store_true", default=(os.environ.get("PHASE_STRATIFIED", "1") == "1"), help="Enable Phase-Stratified Batching")
    parser.add_argument("--no-phase-stratified", action="store_false", dest="phase_stratified", help="Disable Phase-Stratified Batching")
    parser.add_argument("--phase-strategy", type=str, default=os.environ.get("PHASE_STRATEGY", "round_robin"), choices=["round_robin", "random"], help="Phase scheduling strategy across microbatches")
    parser.add_argument("--phase-granularity", type=str, default=os.environ.get("PHASE_GRANULARITY", "microbatch"), choices=["microbatch", "step"], help="Granularity for phase stratification ('microbatch' or 'step')")
    parser.add_argument("--model-scale", type=str, default=os.environ.get("MODEL_SCALE", "100m"), choices=["100m", "4b"], help="Architecture scale preset: '100m' or '4b' (EDGEFLOW_4B_CONFIG_24L/30L)")
    parser.add_argument("--cfg-less", action="store_true", default=(os.environ.get("CFG_LESS", "0") == "1"), help="Enable CFG-less training (forces cfg_drop_prob=0.0 and val_cfg_scale=1.0)")
    parser.add_argument("--no-cfg-less", action="store_false", dest="cfg_less", help="Disable CFG-less training")
    parser.add_argument("--accumulate-steps", type=int, default=int(os.environ.get("ACCUMULATE_STEPS", "3")), help="Gradient accumulation microbatch steps (multiple of 3 recommended for 3-phase stratification)")
    parser.add_argument("--pin-memory", action="store_true", default=(os.environ.get("PIN_MEMORY", "1" if sys.platform != "win32" else "0") == "1"), help="Enable DataLoader memory pinning")
    parser.add_argument("--fp8-matmul", action="store_true", default=(os.environ.get("FP8_MATMUL", "0") == "1"), help="Enable FP8 matmul acceleration on H100 Hopper SXM5")
    parser.add_argument("--use-phase-lora", action="store_true", default=None, help="Enable Phase-LoRA in Self-Attention")
    parser.add_argument("--no-use-phase-lora", action="store_false", dest="use_phase_lora", help="Disable Phase-LoRA in Self-Attention")
    parser.add_argument("--phase-lora-rank", type=int, default=None, help="Low-rank dimension for Phase-LoRA (default: 32)")
    parser.add_argument(
        "--ram-chunk-size",
        type=int,
        default=int(os.environ.get("CACHE_RAM_CHUNK_SIZE", "960")),
        help="Samples per RAM chunk (must be a multiple of batch_size * accumulate_steps, e.g. 96)",
    )

    args, unknown = parser.parse_known_args()

    os.environ["CACHE_RAM_CHUNK_SIZE"] = str(args.ram_chunk_size)
    os.environ["ACCUMULATE_STEPS"] = str(args.accumulate_steps)

    seed = int(os.environ.get("SEED", "42"))

    torch.manual_seed(seed)
    np.random.seed(seed)

    is_distributed = False
    rank = 0
    local_rank = 0
    world_size = 1

    if "WORLD_SIZE" in os.environ or "RANK" in os.environ or args.distributed:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = args.dist_backend if args.dist_backend else "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        is_distributed = dist.is_initialized()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = (rank == 0)

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

    save_dir = args.save_dir
    logger = setup_logger(save_dir, is_main_process=is_main_process)

    if getattr(args, "cfg_less", False):
        args.cfg_drop_prob = 0.0
        args.val_cfg_scale = 1.0

    if getattr(args, "fp8_matmul", False) or os.environ.get("FP8_MATMUL", "0") == "1":
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        try:
            torch.backends.cuda.matmul.allow_fp8_matmul = True
        except Exception:
            pass
        if is_main_process:
            logger.info("[Hopper H100] Enabled FP8 matmul and reduced precision tensor core acceleration.")

    if is_main_process:
        logger.info("=" * 70)
        logger.info(f"Distributed: {is_distributed} | Strategy: {args.strategy} | World Size: {world_size}")
        logger.info(f"Model Scale: {args.model_scale} | CFG-less: {args.cfg_less} (drop_prob={args.cfg_drop_prob})")
        logger.info(f"Device: {device} | Precision: {amp_dtype}")
        logger.info(f"Save Directory: {save_dir}")
        logger.info(f"OS: {sys.platform} | Num Workers: {args.num_workers}")
        logger.info(f"Phase-Stratified Batching: {args.phase_stratified} (granularity={args.phase_granularity}, strategy={args.phase_strategy})")
        logger.info("=" * 70)

    if args.test_attention or os.environ.get("TEST_ATTENTION", "0") == "1":
        logger.info("[Test Mode] Initiating standalone attention mechanism verification pass...")
        scale = getattr(args, "model_scale", "100m").lower().strip()
        preset_cfg = get_model_config(scale)
        is_dcae = args.vae in ("dcae", "dc-ae", "dc_ae")
        latent_channels = 32 if is_dcae else 4
        patch_size = 1 if is_dcae else 2
        latent_size = args.image_size // (32 if is_dcae else 8)

        expert_hidden_list = (
            [int(x.strip()) for x in args.expert_hidden_list.split(",") if x.strip()]
            if args.expert_hidden_list
            else preset_cfg["expert_hidden_list"]
        )
        model_config = dict(preset_cfg)
        model_config.update(
            latent_channels=latent_channels,
            patch_size=patch_size,
            dim=args.dim or preset_cfg["dim"],
            depth=args.depth or preset_cfg["depth"],
            heads=args.heads or preset_cfg["heads"],
            kv_heads=args.kv_heads or preset_cfg["kv_heads"],
            context_dim=args.context_dim or preset_cfg["context_dim"],
            num_phases=len(expert_hidden_list),
            ffn_hidden=args.ffn_hidden or preset_cfg["ffn_hidden"],
            expert_hidden_list=expert_hidden_list,
        )
        eff_ctx_dim = model_config["context_dim"]
        model_raw = EdgeFlowPhaseV2(**model_config).to(device)
        text_encoder = Qwen35TextEncoder(
            out_dim=eff_ctx_dim,
            include_encoder=False,
            max_length=128,
        ).to(device)

        if hasattr(text_encoder, "pooler") and text_encoder.pooler is not None and hasattr(model_raw, "pooler"):
            model_raw.pooler = text_encoder.pooler

        if args.compile:
            model = setup_compiled_model(
                model=model_raw,
                logger=logger,
                device=device,
                use_compile=True,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
                latent_channels=latent_channels,
                latent_size=latent_size,
                context_dim=eff_ctx_dim,
                seq_len=128,
                batch_size=2,
                amp_dtype=amp_dtype,
            )
        else:
            model = model_raw

        rf = RectifiedFlow4D()
        param_groups = get_param_groups(
            model_raw=model_raw,
            text_encoder=text_encoder,
            weight_decay=0.01,
            lr=2e-4,
            lr_text=1e-4,
            expert_weight_decay=0.0,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=2e-4)

        run_attention_test_training(
            model=model,
            text_encoder=text_encoder,
            optimizer=optimizer,
            rf=rf,
            save_dir=save_dir,
            logger=logger,
            device=device,
            amp_dtype=amp_dtype,
            steps=args.attention_steps,
            latent_channels=latent_channels,
            latent_size=latent_size,
            context_dim=eff_ctx_dim,
            seq_len=128,
        )
        logger.info("Attention analysis test run completed successfully.")
        return

    cache_dir = args.cache_dir

    logger.info(f"Waiting for ready cache in {cache_dir} ...")

    meta_data = wait_for_cache(
        cache_dir,
        timeout=3600.0,
        sleep=5.0,
        min_count=1,
    )

    logger.info("Cache is ready.")

    required_validation_files = [
        "validation_text_features.npy",
        "validation_masks.npy",
    ]

    for filename in required_validation_files:
        path = os.path.join(cache_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Required validation file not found: {path}")

    expected_image_size = args.image_size

    if expected_image_size <= 0:
        raise ValueError("EXPECTED_IMAGE_SIZE must be positive")

    vae_type_env = args.vae
    spatial_ratio = 32 if vae_type_env in ("dcae", "dc-ae", "dc_ae") else 8
    expected_latent_size = expected_image_size // spatial_ratio

    if "latent_shape" not in meta_data:
        default_ch = 32 if vae_type_env in ("dcae", "dc-ae", "dc_ae") else (16 if vae_type_env in ("flux", "flux1", "flux.1") else 4)
        meta_data["latent_shape"] = [
            default_ch,
            expected_latent_size,
            expected_latent_size,
        ]

    cache_image_size = meta_data.get("image_size", expected_image_size)
    cache_seq_len = meta_data.get("seq_len", 128)
    cache_hidden_size = meta_data.get("hidden_size", 512)
    text_shape_meta = meta_data.get("text_shape", None)

    latent_shape = meta_data.get(
        "latent_shape",
        [32 if vae_type_env in ("dcae", "dc-ae", "dc_ae") else 4, expected_latent_size, expected_latent_size],
    )

    if int(cache_image_size) != expected_image_size:
        raise ValueError(
            f"This run is configured for EXPECTED_IMAGE_SIZE={expected_image_size}, "
            f"but cache meta says image_size={cache_image_size}. "
            f"Use a {expected_image_size}x{expected_image_size} cache or change EXPECTED_IMAGE_SIZE."
        )

    if (
        latent_shape[1] != expected_latent_size
        or latent_shape[2] != expected_latent_size
    ):
        raise ValueError(
            f"Latent shape {latent_shape} does not match expected "
            f"latent size {expected_latent_size} for image size {expected_image_size}."
        )

    scale = getattr(args, "model_scale", "4b").lower().strip()
    val_steps_cnt = getattr(args, "val_steps", int(os.environ.get("VAL_STEPS", "24")))
    preset_cfg = get_model_config(scale, steps=val_steps_cnt)

    model_config = dict(preset_cfg)

    if args.dim is not None:
        model_config["dim"] = args.dim
    if args.depth is not None:
        model_config["depth"] = args.depth
    if args.heads is not None:
        model_config["heads"] = args.heads
    if args.kv_heads is not None:
        model_config["kv_heads"] = args.kv_heads
    if args.context_dim is not None:
        model_config["context_dim"] = args.context_dim
    if args.ffn_hidden is not None:
        model_config["ffn_hidden"] = args.ffn_hidden
    if args.expert_hidden_list is not None:
        model_config["expert_hidden_list"] = [int(x.strip()) for x in args.expert_hidden_list.split(",") if x.strip()]
    if args.ada_rank is not None:
        model_config["ada_rank"] = args.ada_rank
    if args.use_phase_lora is not None:
        model_config["use_phase_lora"] = args.use_phase_lora
    if args.phase_lora_rank is not None:
        model_config["phase_lora_rank"] = args.phase_lora_rank

    model_config["latent_channels"] = int(os.environ.get("LATENT_CHANNELS", str(latent_shape[0])))
    existing_ckpts = [
        p
        for p in glob.glob(os.path.join(save_dir, "edgeflow_phase_experts_step_*.pt"))
        if not p.endswith("_weights.pt") and not p.endswith("_ema.pt")
    ]
    if existing_ckpts and not getattr(args, "from_scratch", False):
        latest_existing_ckpt = max(existing_ckpts, key=parse_checkpoint_step)
        try:
            ckpt_meta = torch.load(latest_existing_ckpt, map_location="cpu", weights_only=False)
            if "config" in ckpt_meta and isinstance(ckpt_meta["config"], dict):
                logger.info(f"Using architecture config from existing checkpoint: {latest_existing_ckpt}")
                model_config.update(ckpt_meta["config"])
        except Exception as e:
            logger.warning(f"Could not inspect checkpoint config from {latest_existing_ckpt}: {e}")

    dim = model_config["dim"]
    depth = model_config["depth"]
    heads = model_config["heads"]
    kv_heads = model_config["kv_heads"]
    context_dim = model_config["context_dim"]
    ffn_hidden = model_config["ffn_hidden"]
    expert_hidden_list = model_config["expert_hidden_list"]
    num_phases = len(expert_hidden_list)

    model_raw = EdgeFlowPhaseV2(**model_config).to(device)
    model_raw.gradient_checkpointing = GRADIENT_CHECKPOINTING

    text_encoder_type = os.environ.get("TEXT_ENCODER_TYPE", "qwen35" if cache_hidden_size == 2560 else "text_encoder")
    if text_encoder_type == "qwen35":
        text_encoder = Qwen35TextEncoder(
            out_dim=model_config["context_dim"],
            include_encoder=False,
            max_length=cache_seq_len,
            target_layers=model_config.get("phase_layers", (8, 16, 28)),
            load_in_8bit=True,
        ).to(device)
    else:
        text_encoder = TextEncoder(
            out_dim=model_config["context_dim"],
            include_encoder=False,
            hidden_size=cache_hidden_size,
            max_length=cache_seq_len,
        ).to(device)

    if hasattr(text_encoder, "pooler") and text_encoder.pooler is not None and hasattr(model_raw, "pooler"):
        model_raw.pooler = text_encoder.pooler

    total_params, trainable_params = count_parameters(model_raw)

    expert_params = sum(
        p.numel()
        for name, p in model_raw.named_parameters()
        if "phase_experts" in name
    )

    base_params = total_params - expert_params

    text_total_params, text_trainable_params = count_parameters(text_encoder)

    total_trainable_all = trainable_params + text_trainable_params

    logger.info(f"Total model parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable model parameters: {trainable_params / 1e6:.2f}M")
    logger.info(f"Base parameters: {base_params / 1e6:.2f}M")
    logger.info(f"Phase expert parameters: {expert_params / 1e6:.2f}M")
    logger.info(f"Text adapter trainable parameters: {text_trainable_params / 1e6:.2f}M")
    logger.info(f"Total trainable parameters incl. text adapter: {total_trainable_all / 1e6:.2f}M")

    vae = create_vae(device)
    rf = RectifiedFlow4D()

    fixed_lr = getattr(args, "fixed_lr", None)
    if fixed_lr is None and os.environ.get("FIXED_LR"):
        fixed_lr = float(os.environ["FIXED_LR"])

    lr_arg = getattr(args, "lr", None)
    if fixed_lr is not None:
        lr = fixed_lr
    elif lr_arg is not None:
        lr = float(lr_arg)
    else:
        lr = float(os.environ.get("LR", "2e-4"))

    lr_text_arg = getattr(args, "lr_text", None)
    if lr_text_arg is not None:
        lr_text = float(lr_text_arg)
    elif fixed_lr is not None:
        lr_text = fixed_lr * 0.5
    else:
        lr_text_env = os.environ.get("LR_TEXT")
        lr_text = float(lr_text_env) if lr_text_env is not None else lr * 0.5

    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    expert_weight_decay = float(os.environ.get("EXPERT_WEIGHT_DECAY", "0.0"))

    param_groups = get_param_groups(
        model_raw=model_raw,
        text_encoder=text_encoder,
        weight_decay=weight_decay,
        lr=lr,
        lr_text=lr_text,
        expert_weight_decay=expert_weight_decay,
    )

    use_fused = (device.type == "cuda")

    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-8,
            fused=use_fused,
        )
        logger.info(f"Using AdamW optimizer (fused={use_fused})")
    except Exception:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-8,
        )
        logger.info("Using standard AdamW optimizer")

    batch_size = max(1, int(args.batch_size))
    accumulate_steps = max(1, getattr(args, "accumulate_steps", int(os.environ.get("ACCUMULATE_STEPS", "3"))))
    num_workers = args.num_workers

    if getattr(args, "total_steps_alias", None) is not None:
        total_steps = max(1, int(args.total_steps_alias))
    else:
        total_steps = max(1, int(args.steps))
    epochs = max(1, int(os.environ.get("EPOCHS", "100")))

    warmup_steps = int(os.environ.get("WARMUP_STEPS", "2000"))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.01,
        fixed_lr=fixed_lr,
        lr_text=lr_text,
    )

    ema_decay = float(os.environ.get("EMA_DECAY", "0.9995"))

    ema_model = make_ema(model_raw, ema_decay)
    ema_text_proj = make_ema(text_encoder.proj, ema_decay)
    ema_text_pooler = make_ema(text_encoder.pooler, ema_decay)

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(amp_dtype == torch.float16),
        )
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))


    if getattr(args, "from_scratch", False):
        logger.info("[Training] --from-scratch active: Starting training from scratch (step 0, epoch 0).")
        global_step, epoch_from_ckpt, ckpt_latent_mean, ckpt_latent_std = 0, 0, None, None
    else:
        global_step, epoch_from_ckpt, ckpt_latent_mean, ckpt_latent_std = load_latest_checkpoint(
            model=model_raw,
            text_encoder=text_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=save_dir,
            logger=logger,
            device=device,
            ema_model=ema_model,
            ema_text_proj=ema_text_proj,
            ema_text_pooler=ema_text_pooler,
            scaler=scaler,
        )

    if fixed_lr is not None:
        apply_fixed_lr(optimizer, scheduler, fixed_lr, lr_text=lr_text)
        logger.info(f"Enforced fixed learning rate on resumed checkpoint: base/experts={fixed_lr:.2e}, text={lr_text:.2e}")
    else:
        decay_start_arg = getattr(args, "decay_start_step", None)
        if decay_start_arg is None and os.environ.get("DECAY_START_STEP"):
            decay_start_arg = int(os.environ["DECAY_START_STEP"])

        should_reset_lr = (
            getattr(args, "reset_lr", False)
            or os.environ.get("RESET_LR", "0") == "1"
            or decay_start_arg is not None
            or args.lr is not None
        )

        if should_reset_lr:
            target_base_lr = lr if lr is not None else float(os.environ.get("LR", "1e-4"))
            target_min_lr = getattr(args, "min_lr", float(os.environ.get("MIN_LR", "5e-5")))
            target_decay_start = decay_start_arg if decay_start_arg is not None else global_step
            target_lr_text = lr_text if lr_text is not None else float(os.environ.get("LR_TEXT", str(target_base_lr * 0.5)))
            if hasattr(scheduler, "configure_cosine_decay"):
                scheduler.configure_cosine_decay(
                    start_step=target_decay_start,
                    total_steps=total_steps,
                    base_lr=target_base_lr,
                    min_lr=target_min_lr,
                    lr_text=target_lr_text,
                )
                logger.info(
                    f"Configured cosine LR decay from step {target_decay_start} to {total_steps}: "
                    f"base_lr={target_base_lr:.2e} -> {target_min_lr:.2e}, text_lr={target_lr_text:.2e} -> {target_min_lr * 0.5:.2e}"
                )

    if is_distributed and args.strategy == "fsdp":
        if not FSDP_AVAILABLE:
            raise RuntimeError("FSDP requested but FullyShardedDataParallel is not available.")
        import functools
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={EdgeFlowPhaseBlockV2, PhaseExpert},
        )
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
        )
        strategy_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        }
        sharding = strategy_map.get(args.fsdp_sharding, ShardingStrategy.FULL_SHARD)
        ignored_fsdp = [model_raw.pooler] if getattr(model_raw, "pooler", None) is not None else None
        model = FSDP(
            model_raw,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=sharding,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device() if device.type == "cuda" else None,
            limit_all_gathers=True,
            use_orig_params=True,
            ignored_modules=ignored_fsdp,
        )
        if args.compile:
            model = setup_compiled_model(
                model=model,
                logger=logger,
                device=device,
                use_compile=True,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
                latent_channels=model_config["latent_channels"],
                latent_size=expected_latent_size,
                context_dim=context_dim,
                seq_len=cache_seq_len,
                batch_size=batch_size,
                amp_dtype=amp_dtype,
                text_shape=tuple(text_shape_meta) if text_shape_meta else None,
            )
        if is_main_process:
            logger.info(f"[FSDP] Wrapped model in FSDP (rank {rank}/{world_size}, sharding={sharding})")
    else:
        model = setup_compiled_model(
            model=model_raw,
            logger=logger,
            device=device,
            use_compile=args.compile,
            backend=args.compile_backend,
            mode=args.compile_mode,
            dynamic=args.compile_dynamic,
            latent_channels=model_config["latent_channels"],
            latent_size=expected_latent_size,
            context_dim=context_dim,
            seq_len=cache_seq_len,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
            text_shape=tuple(text_shape_meta) if text_shape_meta else None,
        )
        if is_distributed and args.strategy == "ddp":
            if DDP is None:
                raise RuntimeError("DDP requested but DistributedDataParallel is not available.")
            model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
            if is_main_process:
                logger.info(f"[DDP] Wrapped model in DistributedDataParallel (rank {rank}/{world_size}, find_unused_parameters=True)")

    latent_mean, latent_std = load_latent_stats(meta_data, device)

    use_ckpt_latent_stats = os.environ.get("USE_CKPT_LATENT_STATS", "1") == "1"

    if use_ckpt_latent_stats:
        if ckpt_latent_mean is not None:
            latent_mean = ckpt_latent_mean.to(device)
            logger.info("Using latent_mean from checkpoint.")

        if ckpt_latent_std is not None:
            latent_std = ckpt_latent_std.to(device)
            logger.info("Using latent_std from checkpoint.")
    else:
        logger.info("Ignoring checkpoint latent stats and using current cache meta stats.")

    val_text_features = (
        torch.from_numpy(
            np.load(os.path.join(cache_dir, "validation_text_features.npy"))
        )
        .float()
        .to(device)
    )

    val_masks = (
        torch.from_numpy(np.load(os.path.join(cache_dir, "validation_masks.npy")))
        .long()
        .to(device)
    )

    val_latents_path = os.path.join(cache_dir, "validation_latents.npy")

    if os.path.exists(val_latents_path):
        val_latents = torch.from_numpy(np.load(val_latents_path)).float().to(device)
    else:
        val_latents = None

    empty_text_shape = tuple(text_shape_meta) if text_shape_meta else (cache_seq_len, cache_hidden_size)
    empty_text = load_optional_npy(
        os.path.join(cache_dir, "empty_text_features.npy"),
        shape=empty_text_shape,
        device=device,
        dtype=torch.float32,
    )

    empty_mask = load_optional_npy(
        os.path.join(cache_dir, "empty_masks.npy"),
        shape=(cache_seq_len,),
        device=device,
        dtype=torch.long,
    )

    if val_text_features.shape[-1] != cache_hidden_size:
        raise ValueError(
            f"validation_text_features has hidden_size={val_text_features.shape[-1]}, "
            f"but meta.json has {cache_hidden_size}"
        )

    if val_masks.shape[-1] != cache_seq_len:
        raise ValueError(
            f"validation_masks has seq_len={val_masks.shape[-1]}, "
            f"but meta.json has {cache_seq_len}"
        )

    validation_steps = getattr(args, "val_steps", int(os.environ.get("VAL_STEPS", "12")))
    val_cfg_scale = getattr(args, "val_cfg_scale", float(os.environ.get("VAL_CFG_SCALE", "1.0")))
    val_cfg_rescale = getattr(args, "val_cfg_rescale", float(os.environ.get("VAL_CFG_RESCALE", "0.0")))
    val_shift = getattr(args, "val_shift", float(os.environ.get("VAL_SHIFT", "1.0")))
    val_rescale_norm = getattr(args, "val_rescale_norm", (os.environ.get("VAL_RESCALE_NORM", "0") == "1"))
    val_solver = getattr(args, "val_solver", os.environ.get("VAL_SOLVER", "unipc")).lower()

    if val_text_features.shape[0] == 0:
        raise ValueError("validation_text_features.npy must not be empty")

    if val_cfg_scale > 1.0 and val_text_features.shape[0] % 2 != 0:
        raise ValueError(
            "validation_text_features must contain an even number of embeddings "
            "for [uncond, cond] CFG pairs when VAL_CFG_SCALE > 1.0."
        )

    if val_latents is not None:
        if tuple(val_latents.shape[1:]) != tuple(latent_shape):
            raise ValueError(
                f"validation_latents.npy has shape {tuple(val_latents.shape)}, "
                f"but expected latent shape is {tuple(latent_shape)}."
            )

        expected_val_text_count = 2 * val_latents.shape[0] if val_cfg_scale > 1.0 else val_latents.shape[0]

        if val_cfg_scale > 1.0 and val_text_features.shape[0] != expected_val_text_count:
            raise ValueError(
                "validation_text_features.npy must contain [uncond, cond] pairs "
                "for each validation latent when VAL_CFG_SCALE > 1.0. "
                f"Expected {expected_val_text_count} text embeddings, "
                f"got {val_text_features.shape[0]}."
            )

    assert latent_shape[0] == model_config.get("latent_channels", 4)

    if getattr(args, "cfg_less", False) or val_cfg_scale <= 1.0:
        val_sample_count = (
            val_text_features.shape[0] // 2
            if val_text_features.shape[0] % 2 == 0
            else max(1, val_text_features.shape[0])
        )
    elif (val_text_features.shape[0] % 2 == 0 and val_latents is not None and val_text_features.shape[0] == 2 * val_latents.shape[0]):
        val_sample_count = val_text_features.shape[0] // 2
    elif val_text_features.shape[0] % 2 == 0 and val_latents is None and val_text_features.shape[0] >= 2:
        val_sample_count = val_text_features.shape[0] // 2
    else:
        val_sample_count = val_text_features.shape[0]

    val_sample_count = max(1, val_sample_count)

    fixed_val_noise = torch.randn(
        val_sample_count,
        latent_shape[0],
        latent_shape[1],
        latent_shape[2],
        device=device,
        generator=torch.Generator(device=device).manual_seed(42),
    )

    if getattr(args, "save_every", None) is not None:
        save_every_steps = max(1, int(args.save_every))
    else:
        save_every_steps = max(1, int(os.environ.get("SAVE_EVERY", "1000")))

    if getattr(args, "val_every", None) is not None:
        val_every_steps = max(1, int(args.val_every))
    else:
        val_every_steps = max(1, int(os.environ.get("VAL_EVERY", "1000")))

    clip_norm = float(os.environ.get("CLIP_NORM", "1.0"))

    t_shift = float(os.environ.get("T_SHIFT", "1.5"))
    t_uniform_prob = float(os.environ.get("T_UNIFORM_PROB", "1.0"))

    cfg_drop_prob = getattr(args, "cfg_drop_prob", float(os.environ.get("CFG_DROP_PROB", "0.05")))

    keep_checkpoints = int(os.environ.get("KEEP_CHECKPOINTS", "2"))
    reload_check_steps = max(0, int(os.environ.get("RELOAD_CHECK_STEPS", "0")))
    prefetch_factor = max(1, int(os.environ.get("PREFETCH_FACTOR", "2")))

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Accumulate steps: {accumulate_steps}")
    logger.info(f"Effective batch size: {batch_size * accumulate_steps}")
    logger.info(f"Total target optimizer steps: {total_steps}")
    logger.info(f"Starting at global step: {global_step}")
    logger.info(f"Expected image size: {expected_image_size}")
    logger.info(f"Latent shape: {latent_shape}")
    logger.info(f"T_SHIFT: {t_shift}")
    logger.info(f"T_UNIFORM_PROB: {t_uniform_prob}")

    patch_size = model_config.get("patch_size", 2)

    effective_latent_size = (latent_shape[1] // patch_size) * (
        latent_shape[2] // patch_size
    )

    eff_tracker = EfficiencyTracker(
        model_config=model_config,
        batch_size=batch_size,
        image_tokens=effective_latent_size,
        text_tokens=cache_seq_len,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )

    persistent_workers = (num_workers > 0) and os.environ.get(
        "PERSISTENT_WORKERS",
        "1",
    ) == "1"

    pin_memory = getattr(args, "pin_memory", False)
    dataloader = get_cached_dataloader(
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        split="train",
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        is_distributed=is_distributed,
        world_size=world_size,
        rank=rank,
    )
    sampler = dataloader.sampler

    epoch = epoch_from_ckpt
    attention_monitor = AttentionMonitor(save_dir, logger)
    total_microbatch_counter = global_step * accumulate_steps
    phase_visit_counts = [0 for _ in range(max(1, num_phases))]

    wait_attempts = 0
    consecutive_skips = 0
    step_was_skipped = False

    compute_end_time = time.perf_counter()

    try:
        while global_step < total_steps:
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            if reload_check_steps > 0:
                dataloader, _ = try_reload_dataloader(
                    dataloader=dataloader,
                    cache_dir=cache_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    persistent_workers=persistent_workers,
                    logger=logger,
                    prefetch_factor=prefetch_factor,
                    pin_memory=pin_memory,
                    is_distributed=is_distributed,
                    world_size=world_size,
                    rank=rank,
                )

            total_images = len(dataloader.dataset)

            if total_images < batch_size:
                if wait_attempts >= 360:
                    raise RuntimeError("Dataset cache did not grow enough in time.")

                logger.info(
                    f"Waiting for dataset to cache more samples "
                    f"(current: {total_images}/{batch_size})..."
                )

                time.sleep(10)
                wait_attempts += 1

                continue

            wait_attempts = 0

            logger.info(
                f"Epoch {epoch} | Dataset samples: {total_images} | "
                f"Batches: {len(dataloader)}"
            )

            model.train()
            text_encoder.train()

            optimizer.zero_grad(set_to_none=True)

            batches = len(dataloader)
            accumulation_counter = 0
            step_loss_accum = 0.0
            reload_requested = False
            stop_requested = False

            for batch_idx, (
                batch_latents,
                batch_text_features,
                batch_masks,
            ) in enumerate(dataloader):
                if global_step >= total_steps:
                    break

                data_end_time = time.perf_counter()

                if batch_idx == 0:
                    data_time = 0.001
                else:
                    data_time = max(0.0, data_end_time - compute_end_time)

                compute_start_time = time.perf_counter()

                accumulation_counter += 1
                total_microbatch_counter += 1

                if getattr(args, "phase_stratified", True):
                    n_phases = getattr(model_raw, "num_phases", 3)
                    if getattr(args, "phase_granularity", "microbatch") == "step":
                        target_phase = global_step % n_phases
                    elif getattr(args, "phase_strategy", "round_robin") == "random":
                        if is_distributed:
                            if is_main_process:
                                rnd_p = torch.randint(0, n_phases, (1,), device=device)
                            else:
                                rnd_p = torch.zeros(1, dtype=torch.long, device=device)
                            dist.broadcast(rnd_p, src=0)
                            target_phase = int(rnd_p.item())
                        else:
                            target_phase = int(torch.randint(0, n_phases, (1,)).item())
                    else:
                        if n_phases == 3:
                            if global_step < 15000:
                                sched = (0, 1, 2, 0, 1)
                            elif global_step < 45000:
                                sched = (0, 1, 2, 0, 1, 1, 0, 1, 2)
                            else:
                                sched = (0, 1, 0, 1, 2, 1)
                            target_phase = sched[(total_microbatch_counter - 1) % len(sched)]
                        else:
                            target_phase = (total_microbatch_counter - 1) % n_phases
                    if 0 <= target_phase < len(phase_visit_counts):
                        phase_visit_counts[target_phase] += 1
                else:
                    target_phase = None

                sync_gradients = (accumulation_counter == accumulate_steps)

                use_no_sync = hasattr(model, "no_sync") and not sync_gradients
                sync_ctx = model.no_sync() if use_no_sync else contextlib.nullcontext()

                with sync_ctx:
                    loss, grad_norm, step_was_skipped = train_step(
                        model=model,
                        ema_source_model=model_raw,
                        text_encoder=text_encoder,
                        rf=rf,
                        batch_latents=batch_latents,
                        batch_text_features=batch_text_features,
                        batch_masks=batch_masks,
                        empty_text=empty_text,
                        empty_mask=empty_mask,
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        device=device,
                        amp_dtype=amp_dtype,
                        logger=logger,
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                        ema_model=ema_model,
                        ema_text_proj=ema_text_proj,
                        ema_text_pooler=ema_text_pooler,
                        scaler=scaler,
                        accumulate_steps=accumulate_steps,
                        sync_gradients=sync_gradients,
                        t_shift=t_shift,
                        t_uniform_prob=t_uniform_prob,
                        cfg_drop_prob=cfg_drop_prob,
                        clip_norm=clip_norm,
                        target_phase=target_phase,
                        is_distributed=is_distributed,
                        world_size=world_size,
                    )

                compute_end_time = time.perf_counter()
                compute_time = compute_end_time - compute_start_time

                eff_tracker.update(data_time, compute_time)

                if step_was_skipped and not sync_gradients:
                    accumulation_counter = 0
                    step_loss_accum = 0.0
                    optimizer.zero_grad(set_to_none=True)
                    continue

                step_loss_accum += loss / float(accumulate_steps)

                if sync_gradients:
                    if not step_was_skipped:
                        global_step += 1
                        consecutive_skips = 0
                    else:
                        consecutive_skips += 1

                        logger.info(
                            f"DEBUG: skip! grad_norm={grad_norm}, loss={loss}, "
                            f"batch_idx={batch_idx}, len(dataloader)={len(dataloader)}"
                        )

                        if consecutive_skips > 50:
                            emergency_path = os.path.join(
                                save_dir,
                                f"emergency_diverged_step_{global_step}.pt",
                            )

                            torch.save(model_raw.state_dict(), emergency_path)

                            raise RuntimeError("Training diverged!")

                    if is_main_process:
                        if global_step % 10 == 0 or batch_idx == 0:
                            metrics = eff_tracker.get_metrics()
                            eff_tracker.save_metrics(save_dir, global_step, metrics)

                            avg_step_time = (
                                metrics["avg_data_time"] + metrics["avg_compute_time"]
                            )

                            steps_to_ckpt = save_every_steps - (
                                global_step % save_every_steps
                            )

                            if steps_to_ckpt == save_every_steps:
                                steps_to_ckpt = 0

                            eta_ckpt = steps_to_ckpt * avg_step_time * accumulate_steps
                            eta_total = (total_steps - global_step) * avg_step_time * accumulate_steps

                            current_lr = scheduler.get_last_lr()[0]

                            mem_stats = get_memory_usage()

                            bottleneck_warn = (
                                " [Data Bottleneck]"
                                if metrics["data_bottleneck"]
                                else ""
                            )

                            logger.info(
                                f"Step {global_step:05d} | "
                                f"Loss: {step_loss_accum:.4f} | "
                                f"LR: {current_lr:.2e} | "
                                f"Grad: {grad_norm:.2f} | "
                                f"ETA Ckpt: {format_time(eta_ckpt)} | "
                                f"ETA Total: {format_time(eta_total)} | "
                                f"Throughput: {metrics['samples_per_sec']:.1f} smp/s | "
                                f"{metrics['achieved_tflops']:.1f} TFLOPS | "
                                f"MFU: {metrics['mfu']:.1f}% | "
                                f"Data/Compute: "
                                f"{metrics['avg_data_time']*1000:.0f}ms/"
                                f"{metrics['avg_compute_time']*1000:.0f}ms | "
                                f"{mem_stats}{bottleneck_warn}"
                            )

                        if global_step % 100 == 0 and num_phases > 1:
                            total_visits = sum(phase_visit_counts)
                            if total_visits > 0:
                                phase_str = " | ".join(
                                    f"p{i}={phase_visit_counts[i] / total_visits:.3f}"
                                    for i in range(num_phases)
                                )
                                logger.info(f"Phase distribution (actual microbatches): {phase_str}")
                                phase_visit_counts = [0 for _ in range(num_phases)]
                            else:
                                with torch.no_grad():
                                    t_debug = sample_continuous_t(
                                        batch_size=4096,
                                        device=device,
                                        uniform_prob=t_uniform_prob,
                                        shift=t_shift,
                                    )
                                    phase_debug = (t_debug * num_phases).floor().long().clamp(0, num_phases - 1)
                                    phase_counts = torch.bincount(phase_debug, minlength=num_phases)
                                    phase_probs = phase_counts.float() / phase_counts.sum()
                                    phase_str = " | ".join(f"p{i}={phase_probs[i]:.3f}" for i in range(num_phases))
                                    logger.info(f"Phase distribution (continuous): {phase_str}")

                        if (
                            args.track_attention_every > 0
                            and global_step > 0
                            and not step_was_skipped
                            and global_step % args.track_attention_every == 0
                        ):
                            base_model = unwrap_model(model)
                            base_model.set_attention_tracking(True)
                            b_diag = min(2, batch_latents.shape[0])
                            t_diag = torch.linspace(0.25, 0.75, b_diag, device=device) if b_diag > 1 else torch.tensor([0.5], device=device)
                            b_lat = (batch_latents[:b_diag].to(device, non_blocking=True).float() - latent_mean) / latent_std
                            b_txt = batch_text_features[:b_diag].to(device, non_blocking=True).float()
                            b_mask = batch_masks[:b_diag].to(device, non_blocking=True).long()
                            with torch.no_grad():
                                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                                    cond_p_diag, cond_s_diag, mask_diag = text_encoder.forward_cached(
                                        b_txt, b_mask
                                    )
                                    _ = base_model(
                                        z=b_lat,
                                        cond_pooled=cond_p_diag,
                                        cond_seq=cond_s_diag,
                                        t_phase=t_diag,
                                        mask=mask_diag,
                                    )
                            attention_monitor.record_step(base_model, global_step)
                            base_model.set_attention_tracking(False)

                        if (
                            global_step > 0
                            and not step_was_skipped
                            and global_step % save_every_steps == 0
                        ):
                            ensure_free_disk_space(save_dir, logger)

                            save_checkpoint(
                                model=model,
                                text_encoder=text_encoder,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                step=global_step,
                                save_dir=save_dir,
                                logger=logger,
                                ema_model=ema_model,
                                ema_text_proj=ema_text_proj,
                                ema_text_pooler=ema_text_pooler,
                                scaler=scaler,
                                epoch=epoch,
                                config=model_config,
                                latent_mean=latent_mean,
                                latent_std=latent_std,
                            )

                            cleanup_checkpoints(save_dir, keep=keep_checkpoints)

                        if (
                            global_step > 0
                            and not step_was_skipped
                            and global_step % val_every_steps == 0
                        ):
                            try:
                                v_feat_path = os.path.join(cache_dir, "validation_text_features.npy")
                                v_mask_path = os.path.join(cache_dir, "validation_masks.npy")
                                if os.path.exists(v_feat_path) and os.path.exists(v_mask_path):
                                    val_text_features = torch.from_numpy(np.load(v_feat_path)).float().to(device)
                                    val_masks = torch.from_numpy(np.load(v_mask_path)).long().to(device)
                            except Exception as _e:
                                logger.warning(f"Could not reload validation features from disk: {_e}")

                            generate_validation_samples(
                                ema_model=ema_model,
                                text_encoder=text_encoder,
                                vae=vae,
                                rf=rf,
                                device=device,
                                amp_dtype=amp_dtype,
                                step=global_step,
                                save_dir=save_dir,
                                val_text_features=val_text_features,
                                val_masks=val_masks,
                                fixed_noise=fixed_val_noise,
                                logger=logger,
                                latent_mean=latent_mean,
                                latent_std=latent_std,
                                ema_text_proj=ema_text_proj,
                                ema_text_pooler=ema_text_pooler,
                                val_latents=val_latents,
                                validation_steps=validation_steps,
                                cfg_scale=val_cfg_scale,
                                cfg_rescale=val_cfg_rescale,
                                shift=val_shift,
                                rescale_norm=val_rescale_norm,
                                val_solver=val_solver,
                            )

                            model.train()
                            text_encoder.train()

                    if is_distributed and not step_was_skipped:
                        if global_step % save_every_steps == 0 or global_step % val_every_steps == 0:
                            dist.barrier()

                    compute_end_time = time.perf_counter()

                    accumulation_counter = 0
                    step_loss_accum = 0.0

                    if (
                        reload_check_steps > 0
                        and global_step > 0
                        and global_step % reload_check_steps == 0
                    ):
                        try:
                            new_meta = read_meta(cache_dir)
                            new_count = int(new_meta.get("count", 0))
                            old_count = len(dataloader.dataset)

                            if new_count > old_count:
                                logger.info(
                                    f"New data detected: {old_count} -> {new_count}. "
                                    "Dataset will be reloaded after this step."
                                )

                                reload_requested = True

                        except Exception as e:
                            logger.warning(f"Failed to check meta.json for reload: {e}")

                    if os.path.exists("stop_training.flag"):
                        logger.info("stop_training.flag detected! Gracefully stopping training loop...")
                        if is_distributed and dist.is_initialized():
                            dist.barrier()
                        if is_main_process:
                            try:
                                os.remove("stop_training.flag")
                            except Exception:
                                pass
                        stop_requested = True
                        break
                    if reload_requested and global_step < total_steps:
                        break

                compute_end_time = time.perf_counter()

            if stop_requested:
                break

            if reload_requested and global_step < total_steps:
                logger.info("Reloading dataloader due to dataset growth.")

                try:
                    dataloader.dataset.close()
                except Exception:
                    pass

                shutdown_dataloader(dataloader)

                dataloader = get_cached_dataloader(
                    cache_dir=cache_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    persistent_workers=persistent_workers,
                    split="train",
                    prefetch_factor=prefetch_factor,
                    pin_memory=pin_memory,
                    is_distributed=is_distributed,
                    world_size=world_size,
                    rank=rank,
                )
                sampler = dataloader.sampler

                compute_end_time = time.perf_counter()

                continue

            epoch += 1

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving emergency checkpoint...")

    finally:
        if is_main_process:
            if (
                global_step % save_every_steps != 0
                or global_step == 0
                or step_was_skipped
            ):
                ensure_free_disk_space(save_dir, logger)

                save_checkpoint(
                    model=model,
                    text_encoder=text_encoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    save_dir=save_dir,
                    logger=logger,
                    ema_model=ema_model,
                    ema_text_proj=ema_text_proj,
                    ema_text_pooler=ema_text_pooler,
                    scaler=scaler,
                    epoch=epoch,
                    config=model_config,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                )

                cleanup_checkpoints(save_dir, keep=keep_checkpoints)

            try:
                attention_monitor.close()
            except Exception:
                pass

            logger.info(f"Final checkpoint saved at step {global_step}.")
            logger.info("Training session finished.")

        if is_distributed and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
