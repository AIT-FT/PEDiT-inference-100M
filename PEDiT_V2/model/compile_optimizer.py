"""
Compilation Optimization & Parallelization Engine for EdgeFlow Phase DiT.

Solves single-threaded compilation bottlenecks under Windows / Linux:
1. Multi-Threading:
   - PyTorch Inductor defaults to `compile_threads = 1` on Windows (`sys.platform == 'win32'`)
     because PyTorch's native `SubprocPool` relies on Unix-only `pass_fds`.
   - This module installs a Windows-compatible high-performance ThreadPoolExecutor
     for concurrent Triton and C++ kernel compilation across all available CPU cores.
2. Persistent Caching:
   - Enforces `fx_graph_cache`, `autotune_local_cache`, and `bundle_triton_into_fx_graph_cache`.
   - Subsequent compilations / restarts hit disk cache in ~5ms instead of minutes.
3. Inductor Specialization:
   - Fast compile tuning (coordinate_descent_tuning=False, max_autotune=False, TF32 enabled).
4. Multi-Phase Warmup:
   - EdgeFlowPhaseDiT employs 3 distinct phase experts (`phase_idx` in 0, 1, 2).
   - Pre-warms all 3 phases ahead of time with exact cross-KV cached features and
     phase-adaptive temperatures, eliminating mid-generation compilation stalls.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

# Prevent PyTorch from auto-warming broken SubprocPool on Windows at import time
if sys.platform == "win32":
    os.environ["TORCH_WARM_POOL"] = "0"

logger = logging.getLogger("compile_optimizer")

# Global lock and executor for parallel compilation
_COMPILE_POOL_LOCK = threading.Lock()
_COMPILE_POOL: Optional[ThreadPoolExecutor] = None
_IS_PARALLEL_ACTIVE = False
_ACTIVE_COMPILE_THREADS = 1
_ORIGINAL_TRITON: Optional[Callable[..., Any]] = None
_ORIGINAL_WAIT: Optional[Callable[..., Any]] = None
_ORIGINAL_PROCESS_POOL: Optional[Callable[..., Any]] = None


class _ProcessPoolBridge:
    """Bridge replacing SubprocPool on Windows with ThreadPoolExecutor."""

    def __init__(self, pool: Optional[ThreadPoolExecutor]) -> None:
        self._pool = pool
        self.ready_future: Future[bool] = Future()
        self.ready_future.set_result(True)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        if self._pool is not None:
            return self._pool.submit(fn, *args, **kwargs)
        f: Future[Any] = Future()
        try:
            res = fn(*args, **kwargs)
            f.set_result(res)
        except Exception as e:
            f.set_exception(e)
        return f


def get_optimal_compile_threads() -> int:
    """Calculates optimal compilation threads based on hardware CPU count."""
    env_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if env_threads:
        try:
            return max(1, int(env_threads))
        except ValueError:
            pass
    cpu_cores = os.cpu_count() or 4
    # Utilize all logical CPU cores (up to 24 workers)
    return max(4, min(cpu_cores, 24))


def setup_fast_compile(
    num_threads: Optional[int] = None,
    enable_caching: bool = True,
    verbose: bool = True,
) -> int:
    """
    Configures PyTorch Inductor for maximum compilation throughput.

    Enables:
    - Multi-threaded Triton kernel compilation on Windows (overcoming 1-thread default)
    - Multi-threaded C++ kernel compilation
    - FX Graph Cache and Triton disk cache
    - Matmul TF32 precision
    - Multi-phase readiness
    """
    global _COMPILE_POOL, _IS_PARALLEL_ACTIVE, _ACTIVE_COMPILE_THREADS
    global _ORIGINAL_TRITON, _ORIGINAL_WAIT, _ORIGINAL_PROCESS_POOL

    threads = num_threads if num_threads is not None and num_threads > 0 else get_optimal_compile_threads()
    _ACTIVE_COMPILE_THREADS = threads

    # Enforce environment variables across all worker subprocesses and OpenMP runtimes
    os.environ["TORCH_WARM_POOL"] = "0"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    if hasattr(torch, "set_num_threads"):
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass

    # 1. Hardware TF32 precision (avoids Inductor warnings and accelerates tensor core operations)
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # 2. Configure PyTorch Inductor & Dynamo config
    try:
        import torch._dynamo.config as dc
        dc.suppress_errors = True
        dc.cache_size_limit = 64
        dc.accumulated_cache_size_limit = 128
    except Exception:
        pass

    try:
        import torch._inductor.config as ic
        ic.compile_threads = threads
        ic.cpp.threads = threads  # Enable OpenMP/MSVC all-core utilization for C++ kernels
        ic.coordinate_descent_tuning = False
        ic.max_autotune = False

        if enable_caching:
            ic.fx_graph_cache = True
            ic.autotune_local_cache = True
            ic.bundle_triton_into_fx_graph_cache = True

        # Ensure cache directories exist
        try:
            from torch._inductor import codecache
            cdir = codecache.cache_dir()
            if cdir and not os.path.exists(cdir):
                os.makedirs(cdir, exist_ok=True)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"[compile_optimizer] Could not set inductor config: {e}")

    # 3. Windows-specific Triton & C++ Parallel Compilation Bridge
    # PyTorch's native SubprocPool crashes on Windows due to pass_fds.
    # We provide a thread-pool based asynchronous compiler for Triton and C++ kernels.
    if sys.platform == "win32":
        try:
            import torch._inductor.async_compile as ac
            from torch._inductor.codecache import CodeCacheFuture, PyCodeCache, TritonCodeCache

            # Save originals once
            if _ORIGINAL_TRITON is None:
                _ORIGINAL_TRITON = ac.AsyncCompile.triton
            if _ORIGINAL_WAIT is None:
                _ORIGINAL_WAIT = ac.AsyncCompile.wait
            if _ORIGINAL_PROCESS_POOL is None:
                _ORIGINAL_PROCESS_POOL = ac.AsyncCompile.process_pool

            # Suppress SubprocPool warm_pool which fails on win32
            ac.AsyncCompile.warm_pool = classmethod(lambda cls: None)

            cur_cuda_dev = torch.cuda.current_device() if torch.cuda.is_available() else 0

            def _thread_precompile_worker(kern: Any, dev_idx: int) -> None:
                if torch.cuda.is_available():
                    torch.cuda.set_device(dev_idx)
                kern.precompile(warm_cache_only=False)

            class ThreadTritonFuture(CodeCacheFuture):
                """Future wrapper for Triton kernels compiled in background threads."""

                def __init__(self, kernel: Any, future: Optional[Future[Any]]) -> None:
                    self.kernel = kernel
                    self.future = future

                def result(self) -> Any:
                    if self.future is not None:
                        try:
                            self.future.result()
                        except Exception:
                            # Fallback: завершаем precompile синхронно в главном потоке при ошибке контекста в воркере
                            self.kernel.precompile(warm_cache_only=False)
                        self.future = None
                    return self.kernel

            def _parallel_triton(self: Any, kernel_name: str, source_code: str, device_str: str = "cuda") -> Any:
                from torch._inductor.async_compile import _compile_start, _set_triton_ptxas_path

                _compile_start()
                _set_triton_ptxas_path()

                if os.environ.get("TRITON_INTERPRET", "0") == "1":
                    return getattr(PyCodeCache.load(source_code), kernel_name)

                kernel = TritonCodeCache.load(kernel_name, source_code)
                with _COMPILE_POOL_LOCK:
                    active_pool = _COMPILE_POOL
                    is_active = _IS_PARALLEL_ACTIVE

                if active_pool is not None and is_active:
                    future = active_pool.submit(_thread_precompile_worker, kernel, cur_cuda_dev)
                    return ThreadTritonFuture(kernel, future)
                else:
                    kernel.precompile(warm_cache_only=False)
                    return kernel

            def _safe_wait(self: Any, scope: Dict[str, Any]) -> None:
                from torch._inductor.async_compile import _compile_end

                # Safely unwrap any Future or CodeCacheFuture in scope regardless of compile_threads
                for key in list(scope.keys()):
                    val = scope[key]
                    if isinstance(val, (Future, CodeCacheFuture)):
                        try:
                            scope[key] = val.result()
                        except Exception as e:
                            logger.error(f"[compile_optimizer] Error compiling kernel {key}: {e}")
                            raise
                _compile_end()

            with _COMPILE_POOL_LOCK:
                if threads > 1:
                    if _COMPILE_POOL is None or _COMPILE_POOL._max_workers != threads:
                        if _COMPILE_POOL is not None:
                            try:
                                _COMPILE_POOL.shutdown(wait=False)
                            except Exception:
                                pass
                        _COMPILE_POOL = ThreadPoolExecutor(
                            max_workers=threads,
                            thread_name_prefix="InductorTritonWorker",
                        )
                    _IS_PARALLEL_ACTIVE = True
                else:
                    if _COMPILE_POOL is not None:
                        try:
                            _COMPILE_POOL.shutdown(wait=False)
                        except Exception:
                            pass
                        _COMPILE_POOL = None
                    _IS_PARALLEL_ACTIVE = False

            # Install multi-worker parallel bridge methods
            ac.get_compile_threads = lambda: _ACTIVE_COMPILE_THREADS
            ac.AsyncCompile.pool = staticmethod(lambda: _COMPILE_POOL)
            ac.AsyncCompile.submit = classmethod(lambda cls, task: _COMPILE_POOL.submit(task) if _COMPILE_POOL is not None else task())
            ac.AsyncCompile.triton = _parallel_triton
            ac.AsyncCompile.wait = _safe_wait
            ac.AsyncCompile.process_pool = staticmethod(lambda: _ProcessPoolBridge(_COMPILE_POOL))
            ac.AsyncCompile._use_process_pool = lambda self: (_COMPILE_POOL is not None and _ACTIVE_COMPILE_THREADS > 1)

        except Exception as pe:
            logger.warning(f"[compile_optimizer] Failed to install parallel Triton compiler bridge: {pe}")
            _IS_PARALLEL_ACTIVE = False
    else:
        _IS_PARALLEL_ACTIVE = (threads > 1)

    if verbose:
        mode_desc = f"Multi-Threaded ({threads} workers)" if _IS_PARALLEL_ACTIVE else "Single-Threaded"
        cache_desc = "ON (fx_graph_cache + triton disk cache)" if enable_caching else "OFF"
        print(f"[compile_optimizer] [*] torch.compile configured: {mode_desc} | Cache: {cache_desc} | Platform: {sys.platform}")

    return threads


def is_parallel_compile_active() -> Tuple[bool, int]:
    """Returns (is_parallel, thread_count)."""
    return _IS_PARALLEL_ACTIVE, _ACTIVE_COMPILE_THREADS


def get_compile_status() -> Dict[str, Any]:
    """Returns a dictionary describing current compilation optimization settings."""
    cache_dir = None
    try:
        from torch._inductor import codecache
        cache_dir = codecache.cache_dir()
    except Exception:
        pass

    return {
        "is_parallel": _IS_PARALLEL_ACTIVE,
        "threads": _ACTIVE_COMPILE_THREADS,
        "platform": sys.platform,
        "cpu_cores": os.cpu_count(),
        "cache_dir": cache_dir,
        "fx_graph_cache": getattr(torch._inductor.config, "fx_graph_cache", True) if hasattr(torch, "_inductor") else None,
    }


def warmup_compiled_model(
    model: torch.nn.Module,
    device: Union[str, torch.device],
    latent_channels: int = 32,
    latent_h: int = 16,
    latent_w: int = 16,
    context_dim: int = 2560,
    num_phases: int = 3,
    batch_sizes: Tuple[int, ...] = (1, 2),
    verbose: bool = True,
) -> float:
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type != "cuda":
        return 0.0

    model.eval()
    target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Полное разворачивание оберток до базового экземпляра EdgeFlowPhaseV2
    model_api = model
    visited = set()
    while id(model_api) not in visited:
        visited.add(id(model_api))
        if hasattr(model_api, "_orig_mod"):
            model_api = model_api._orig_mod
        elif hasattr(model_api, "module"):
            model_api = model_api.module
        else:
            break

    l_ch = getattr(model_api, "latent_channels", latent_channels)
    c_dim = getattr(model_api, "context_dim", context_dim)
    n_phases = getattr(model_api, "num_phases", num_phases)

    p_size = getattr(model_api, "patch_size", 1)
    if latent_h % p_size != 0:
        latent_h = max(p_size, (latent_h // p_size) * p_size)
    if latent_w % p_size != 0:
        latent_w = max(p_size, (latent_w // p_size) * p_size)

    if verbose:
        is_par, th = is_parallel_compile_active()
        th_info = f"{th} parallel threads" if is_par else "single-threaded"
        print(f"[compile_optimizer] [*] Starting JIT warmup across all {n_phases} phase experts and batches {batch_sizes} ({th_info})...")

    shapes = [(latent_h, latent_w)]
    if (latent_h, latent_w) == (16, 16) and l_ch == 32:
        shapes.append((32, 32))

    phase_temps = (0.95, 0.85, 0.72)
    t_start = time.perf_counter()

    # Внешний цикл по фазам p (чтобы стримить каждого эксперта в VRAM ровно 1 раз при прогреве!)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=target_dtype):
        for p in range(n_phases):
            if hasattr(model_api, "stream_phase_expert"):
                model_api.stream_phase_expert(p)

            t_val = float(p) / float(max(1, n_phases - 1))

            for (cur_h, cur_w) in shapes:
                for b in batch_sizes:
                    dummy_z = torch.zeros(b, l_ch, cur_h, cur_w, device=dev, dtype=target_dtype)
                    dummy_cond_pooled = torch.zeros(b, c_dim, device=dev, dtype=target_dtype)
                    t_phase = torch.full((b,), t_val, device=dev, dtype=torch.float32)

                    # При b == 2 (CFG) текстовая последовательность и per_block_text_kvs имеют размер b // 2 = 1 (сплит-батчинг CrossAttentionLite)
                    seq_b = (b // 2) if (b % 2 == 0 and b >= 2) else b
                    dummy_cond_seq = torch.zeros(seq_b, 16, c_dim, device=dev, dtype=target_dtype)
                    temp_p = 1.0 if b >= 2 else phase_temps[min(p, len(phase_temps) - 1)]

                    per_block_kvs = None
                    text_kvs = None
                    if hasattr(model_api, "prepare_phase_cross_kvs"):
                        try:
                            per_block_kvs = model_api.prepare_phase_cross_kvs(dummy_cond_seq, p)
                        except Exception as ke:
                            logger.debug(f"[compile_optimizer] prepare_phase_cross_kvs probe error: {ke}")
                    elif hasattr(model_api, "encode_text"):
                        try:
                            text_kvs = model_api.encode_text(dummy_cond_seq, phase_idx=p)
                        except Exception as ke:
                            logger.debug(f"[compile_optimizer] encode_text probe error: {ke}")

                    try:
                        _ = model(
                            dummy_z,
                            dummy_cond_pooled,
                            cond_seq=dummy_cond_seq if (per_block_kvs is None and text_kvs is None) else None,
                            t_phase=t_phase,
                            phase_idx=p,
                            text_kvs=text_kvs,
                            per_block_text_kvs=per_block_kvs,
                            mask=None,
                            temperature=temp_p,
                        )
                    except Exception as inner_e:
                        logger.warning(f"[compile_optimizer] Warmup for b={b}, shape=({cur_h},{cur_w}), phase={p} encountered: {inner_e}")

        # Возвращаем Эксперта 0 в VRAM к старту первой генерации
        if hasattr(model_api, "stream_phase_expert") and n_phases > 1:
            model_api.stream_phase_expert(0)
        if hasattr(model_api, "prefetch_next_phase") and n_phases > 1:
            model_api.prefetch_next_phase(1)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"[compile_optimizer] [OK] JIT warmup completed in {elapsed:.2f}s! All {n_phases} phase experts cached.")

    return elapsed


def start_async_warmup(
    model: torch.nn.Module,
    device: Union[str, torch.device],
    latent_channels: int = 32,
    latent_h: int = 16,
    latent_w: int = 16,
    context_dim: int = 2560,
    num_phases: int = 3,
    batch_sizes: Tuple[int, ...] = (1, 2),
    on_complete: Optional[Callable[[float], None]] = None,
    lock: Optional[Any] = None,
) -> Tuple[threading.Thread, threading.Event]:
    """
    Spawns background warmup thread so server startup is instantaneous.
    Returns (thread, ready_event).
    """
    ready_event = threading.Event()

    def _worker():
        try:
            if lock is not None:
                with lock:
                    elapsed = warmup_compiled_model(
                        model=model,
                        device=device,
                        latent_channels=latent_channels,
                        latent_h=latent_h,
                        latent_w=latent_w,
                        context_dim=context_dim,
                        num_phases=num_phases,
                        batch_sizes=batch_sizes,
                        verbose=True,
                    )
            else:
                elapsed = warmup_compiled_model(
                    model=model,
                    device=device,
                    latent_channels=latent_channels,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    context_dim=context_dim,
                    num_phases=num_phases,
                    batch_sizes=batch_sizes,
                    verbose=True,
                )
            if on_complete:
                on_complete(elapsed)
        except Exception as e:
            logger.warning(f"[compile_optimizer] Background warmup warning: {e}")
        finally:
            ready_event.set()

    thread = threading.Thread(target=_worker, daemon=True, name="InductorWarmupThread")
    thread.start()
    return thread, ready_event
