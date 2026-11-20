import gc
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def read_meta(cache_dir: str) -> dict:
    meta_path = os.path.join(cache_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _expected_bytes(count: int, shape, dtype) -> int:
    full_shape = (int(count), *shape)
    return int(np.prod(full_shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def cache_is_ready(cache_dir: str, min_count: int = 1) -> bool:
    meta_path = os.path.join(cache_dir, "meta.json")
    ready_path = os.path.join(cache_dir, "cache.ready")

    if not os.path.exists(meta_path):
        return False

    require_ready = os.environ.get("CACHE_REQUIRE_READY", "0") == "1"
    if require_ready and not os.path.exists(ready_path):
        return False

    try:
        meta = read_meta(cache_dir)
        count = int(meta.get("count", 0))
        if count < min_count:
            return False

        latent_shape = tuple(meta["latent_shape"])
        text_shape = tuple(meta["text_shape"])
        mask_shape = tuple(meta["mask_shape"])

        lat_path = os.path.join(cache_dir, "latents.fp16")
        if not os.path.exists(lat_path) or os.path.getsize(lat_path) < _expected_bytes(count, latent_shape, np.float16):
            return False

        mask_path = os.path.join(cache_dir, "masks.uint8")
        if not os.path.exists(mask_path) or os.path.getsize(mask_path) < _expected_bytes(count, mask_shape, np.uint8):
            return False

        fp16_path = os.path.join(cache_dir, "text_features.fp16")
        fp8_path = os.path.join(cache_dir, "text_features.fp8")
        has_fp16 = os.path.exists(fp16_path) and os.path.getsize(fp16_path) >= _expected_bytes(count, text_shape, np.float16)
        has_fp8 = os.path.exists(fp8_path) and os.path.getsize(fp8_path) >= _expected_bytes(count, text_shape, np.uint8)

        return bool(has_fp16 or has_fp8)
    except Exception:
        return False


def wait_for_cache(
    cache_dir: str,
    timeout: float = 3600.0,
    sleep: float = 5.0,
    min_count: int = 1,
) -> dict:
    start = time.time()
    while True:
        if cache_is_ready(cache_dir, min_count=min_count):
            return read_meta(cache_dir)
        if time.time() - start > timeout:
            raise RuntimeError(f"Cache in {cache_dir} did not become ready within {timeout:.0f}s.")
        time.sleep(sleep)


class _RAMChunkSlot:
    def __init__(
        self,
        slot_id: int,
        ram_chunk_size: int,
        latent_shape: tuple,
        text_shape: tuple,
        mask_shape: tuple,
        text_torch_dtype: torch.dtype,
    ):
        self.slot_id = slot_id
        self.latents = torch.empty((ram_chunk_size, *latent_shape), dtype=torch.float16)
        self.texts = torch.empty((ram_chunk_size, *text_shape), dtype=text_torch_dtype)
        self.masks = torch.empty((ram_chunk_size, *mask_shape), dtype=torch.uint8)

        self.chunk_id: int = -1
        self.valid_count: int = 0
        self.ready_event = threading.Event()
        self.ready_event.set()

    def invalidate(self) -> None:
        self.chunk_id = -1
        self.valid_count = 0


class DoubleBufferedChunkSampler(Sampler[int]):
    def __init__(
        self,
        dataset: "CachedLatentTextDataset",
        step_multiple: int = 96,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.ram_chunk_size = dataset.ram_chunk_size
        self.num_chunks = dataset.num_chunks
        self.step_multiple = max(1, int(step_multiple))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_epoch_plan(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            chunk_order = torch.randperm(self.num_chunks, generator=g).tolist()
        else:
            chunk_order = list(range(self.num_chunks))

        valid_chunk_order: List[int] = []
        indices: List[int] = []
        eff_multiple = self.step_multiple * self.world_size

        for c_id in chunk_order:
            length = self.dataset.get_chunk_valid_count(c_id)
            if length <= 0:
                continue

            base_virtual = c_id * self.ram_chunk_size
            if self.shuffle:
                local_perm = (torch.randperm(length, generator=g) + base_virtual).tolist()
            else:
                local_perm = list(range(base_virtual, base_virtual + length))

            usable_len = (length // eff_multiple) * eff_multiple
            if usable_len == 0 and length > 0:
                reps = math.ceil(eff_multiple / length)
                local_perm = (local_perm * reps)[:eff_multiple]
                usable_len = eff_multiple
            else:
                local_perm = local_perm[:usable_len]

            if usable_len > 0:
                valid_chunk_order.append(c_id)
                indices.extend(local_perm)

        if self.world_size > 1:
            total_size = (len(indices) // eff_multiple) * eff_multiple
            indices = indices[:total_size]
            indices = indices[self.rank : total_size : self.world_size]

        return valid_chunk_order, indices

    def __iter__(self) -> Iterator[int]:
        chunk_order, indices = self._build_epoch_plan()
        self.dataset.prepare_epoch_schedule(chunk_order)
        return iter(indices)

    def __len__(self) -> int:
        _, indices = self._build_epoch_plan()
        return len(indices)


class CachedLatentTextDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        split: Optional[str] = None,
        pin_memory: Optional[bool] = None,
        ram_chunk_size: Optional[int] = None,
        step_multiple: int = 96,
    ):
        self.cache_dir = cache_dir
        self.split = split
        self.step_multiple = max(1, int(step_multiple))

        if not cache_is_ready(cache_dir, min_count=1):
            raise RuntimeError(f"Cache in {cache_dir} is not ready. Run prepare_cache.py first.")

        self.meta = read_meta(cache_dir)
        self.length = int(self.meta["count"])

        self.latent_shape = tuple(self.meta["latent_shape"])
        self.text_shape = tuple(self.meta["text_shape"])
        self.mask_shape = tuple(self.meta["mask_shape"])

        if ram_chunk_size is None:
            ram_chunk_size = int(os.environ.get("CACHE_RAM_CHUNK_SIZE", "960"))
        raw_chunk_size = max(self.step_multiple, int(ram_chunk_size))

        if raw_chunk_size % self.step_multiple != 0:
            raw_chunk_size = max(self.step_multiple, round(raw_chunk_size / self.step_multiple) * self.step_multiple)

        self.ram_chunk_size = min(raw_chunk_size, max(self.step_multiple, self.length))

        self.stripe_size = min(64, max(16, self.ram_chunk_size // 16))
        while self.ram_chunk_size % self.stripe_size != 0 and self.stripe_size > 1:
            self.stripe_size -= 1
        self.stripes_per_chunk = max(1, self.ram_chunk_size // self.stripe_size)

        total_stripes = math.ceil(self.length / self.stripe_size)
        g_map = torch.Generator()
        g_map.manual_seed(42)
        all_stripes = torch.randperm(total_stripes, generator=g_map).tolist()

        self.chunk_stripes_map: List[List[tuple]] = []
        for i in range(0, total_stripes, self.stripes_per_chunk):
            stripe_ids = all_stripes[i : i + self.stripes_per_chunk]
            spans = []
            for s_id in stripe_ids:
                s_start = s_id * self.stripe_size
                s_end = min(s_start + self.stripe_size, self.length)
                if s_end > s_start:
                    spans.append((s_start, s_end - s_start))
            if spans:
                self.chunk_stripes_map.append(spans)

        self.num_chunks = len(self.chunk_stripes_map)

        self.lat_path = os.path.join(self.cache_dir, "latents.fp16")
        self.mask_path = os.path.join(self.cache_dir, "masks.uint8")
        fp16_path = os.path.join(self.cache_dir, "text_features.fp16")
        fp8_path = os.path.join(self.cache_dir, "text_features.fp8")

        if os.path.exists(fp16_path) and os.path.getsize(fp16_path) >= _expected_bytes(self.length, self.text_shape, np.float16):
            self.text_path = fp16_path
            self.text_is_fp16 = True
            self.text_torch_dtype = torch.float16
            self.text_elem_bytes = 2
        elif os.path.exists(fp8_path) and os.path.getsize(fp8_path) >= _expected_bytes(self.length, self.text_shape, np.uint8):
            self.text_path = fp8_path
            self.text_is_fp16 = False
            self.text_torch_dtype = torch.uint8
            self.text_elem_bytes = 1
        else:
            raise FileNotFoundError(f"Neither {fp16_path} nor {fp8_path} is complete in {self.cache_dir}")

        self.lat_sample_bytes = int(np.prod(self.latent_shape, dtype=np.int64)) * 2
        self.text_sample_bytes = int(np.prod(self.text_shape, dtype=np.int64)) * self.text_elem_bytes
        self.mask_sample_bytes = int(np.prod(self.mask_shape, dtype=np.int64)) * 1

        self.slots: List[_RAMChunkSlot] = [
            _RAMChunkSlot(0, self.ram_chunk_size, self.latent_shape, self.text_shape, self.mask_shape, self.text_torch_dtype),
            _RAMChunkSlot(1, self.ram_chunk_size, self.latent_shape, self.text_shape, self.mask_shape, self.text_torch_dtype),
        ]

        self.active_slot_idx: int = 0
        self.chunk_schedule: List[int] = []
        self.schedule_pos: int = 0
        self._bg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ram_chunk_prefetch")
        self._io_lock = threading.Lock()

        single_slot_mb = (
            self.slots[0].latents.numel() * 2
            + self.slots[0].texts.numel() * self.text_elem_bytes
            + self.slots[0].masks.numel()
        ) / (1024 ** 2)
        dtype_str = "FP16 (16-bit)" if self.text_is_fp16 else "FP8 (8-bit)"
        print(
            f"[Dataset] Striped Double-Buffered RAM Loader Ready: total={self.length} samples | "
            f"chunk={self.ram_chunk_size} ({self.stripes_per_chunk} stripes x {self.stripe_size}) | "
            f"text={dtype_str} | RAM: 2 chunks x {single_slot_mb:.1f} MB = {single_slot_mb * 2:.1f} MB",
            flush=True,
        )

    def get_chunk_valid_count(self, chunk_id: int) -> int:
        if 0 <= chunk_id < len(self.chunk_stripes_map):
            return sum(cnt for _, cnt in self.chunk_stripes_map[chunk_id])
        return 0

    @staticmethod
    def _drop_os_page_cache(fd: int, offset: int, length: int) -> None:
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass

    def _read_disk_chunk_into_slot(self, slot: _RAMChunkSlot, chunk_id: int) -> None:
        try:
            with self._io_lock:
                if not (0 <= chunk_id < len(self.chunk_stripes_map)):
                    return
                spans = self.chunk_stripes_map[chunk_id]

                with open(self.lat_path, "rb") as f_lat, \
                     open(self.text_path, "rb") as f_txt, \
                     open(self.mask_path, "rb") as f_msk:

                    lat_fd = f_lat.fileno()
                    txt_fd = f_txt.fileno()
                    msk_fd = f_msk.fileno()

                    dest_pos = 0
                    for disk_start, count in spans:
                        dest_end = dest_pos + count

                        lat_off = disk_start * self.lat_sample_bytes
                        lat_len = count * self.lat_sample_bytes
                        f_lat.seek(lat_off)
                        f_lat.readinto(slot.latents[dest_pos:dest_end].numpy())
                        self._drop_os_page_cache(lat_fd, lat_off, lat_len)

                        txt_off = disk_start * self.text_sample_bytes
                        txt_len = count * self.text_sample_bytes
                        f_txt.seek(txt_off)
                        f_txt.readinto(slot.texts[dest_pos:dest_end].numpy())
                        self._drop_os_page_cache(txt_fd, txt_off, txt_len)

                        msk_off = disk_start * self.mask_sample_bytes
                        msk_len = count * self.mask_sample_bytes
                        f_msk.seek(msk_off)
                        f_msk.readinto(slot.masks[dest_pos:dest_end].numpy())
                        self._drop_os_page_cache(msk_fd, msk_off, msk_len)

                        dest_pos = dest_end

                slot.valid_count = dest_pos
                slot.chunk_id = chunk_id
        finally:
            slot.ready_event.set()

    def _start_async_load(self, slot_idx: int, chunk_id: int) -> None:
        slot = self.slots[slot_idx]
        slot.ready_event.clear()
        slot.invalidate()
        self._bg_executor.submit(self._read_disk_chunk_into_slot, slot, chunk_id)

    def prepare_epoch_schedule(self, chunk_order: List[int]) -> None:
        if not chunk_order:
            return
        for s in self.slots:
            s.ready_event.wait()

        self.chunk_schedule = list(chunk_order)
        self.schedule_pos = 0
        first_chunk = self.chunk_schedule[0]

        if self.slots[0].chunk_id == first_chunk:
            self.active_slot_idx = 0
        elif self.slots[1].chunk_id == first_chunk:
            self.active_slot_idx = 1
        else:
            self.active_slot_idx = 0
            self.slots[0].ready_event.clear()
            self._read_disk_chunk_into_slot(self.slots[0], first_chunk)

        inactive_slot_idx = 1 - self.active_slot_idx
        if len(self.chunk_schedule) > 1:
            second_chunk = self.chunk_schedule[1]
            if self.slots[inactive_slot_idx].chunk_id != second_chunk:
                self._start_async_load(inactive_slot_idx, second_chunk)

    def _switch_to_chunk(self, target_chunk_id: int) -> _RAMChunkSlot:
        other_slot_idx = 1 - self.active_slot_idx
        other_slot = self.slots[other_slot_idx]
        other_slot.ready_event.wait()

        if other_slot.chunk_id == target_chunk_id:
            old_slot_idx = self.active_slot_idx
            self.active_slot_idx = other_slot_idx
            self.schedule_pos += 1

            next_pos = self.schedule_pos + 1
            if next_pos < len(self.chunk_schedule):
                next_chunk_id = self.chunk_schedule[next_pos]
                self._start_async_load(old_slot_idx, next_chunk_id)
            else:
                self.slots[old_slot_idx].invalidate()

            return other_slot

        active_slot = self.slots[self.active_slot_idx]
        active_slot.ready_event.wait()
        active_slot.ready_event.clear()
        self._read_disk_chunk_into_slot(active_slot, target_chunk_id)
        return active_slot

    def close(self):
        try:
            self._bg_executor.shutdown(wait=True)
        except Exception:
            pass
        self.slots.clear()
        gc.collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, virtual_idx: int):
        if virtual_idx < 0:
            virtual_idx += self.length
        target_chunk_id = virtual_idx // self.ram_chunk_size
        local_idx = virtual_idx % self.ram_chunk_size

        active_slot = self.slots[self.active_slot_idx]
        if active_slot.chunk_id != target_chunk_id:
            active_slot = self._switch_to_chunk(target_chunk_id)

        local_idx = min(local_idx, max(0, active_slot.valid_count - 1))
        latent = active_slot.latents[local_idx]
        raw_text = active_slot.texts[local_idx]
        text = raw_text if self.text_is_fp16 else raw_text.view(torch.float8_e4m3fn)
        mask = active_slot.masks[local_idx]
        return latent, text, mask

    def __getitems__(self, virtual_indices: List[int]):
        if not virtual_indices:
            return []

        norm_indices = [idx if idx >= 0 else idx + self.length for idx in virtual_indices]
        first_chunk_id = norm_indices[0] // self.ram_chunk_size

        if all((idx // self.ram_chunk_size) == first_chunk_id for idx in norm_indices):
            active_slot = self.slots[self.active_slot_idx]
            if active_slot.chunk_id != first_chunk_id:
                active_slot = self._switch_to_chunk(first_chunk_id)

            max_valid = max(0, active_slot.valid_count - 1)
            local_indices = [min(idx % self.ram_chunk_size, max_valid) for idx in norm_indices]
            idx_tensor = torch.tensor(local_indices, dtype=torch.long)

            latents_batch = active_slot.latents.index_select(0, idx_tensor)
            raw_texts_batch = active_slot.texts.index_select(0, idx_tensor)
            texts_batch = raw_texts_batch if self.text_is_fp16 else raw_texts_batch.view(torch.float8_e4m3fn)
            masks_batch = active_slot.masks.index_select(0, idx_tensor)
            return latents_batch, texts_batch, masks_batch

        lats, txts, msks = [], [], []
        for idx in norm_indices:
            l_i, t_i, m_i = self.__getitem__(idx)
            lats.append(l_i.clone())
            txts.append(t_i.clone())
            msks.append(m_i.clone())
        return torch.stack(lats, dim=0), torch.stack(txts, dim=0), torch.stack(msks, dim=0)


def _fast_passthrough_collate(batch):
    if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[0], torch.Tensor):
        return batch
    return torch.utils.data.default_collate(batch)


def get_cached_dataloader(
    cache_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    persistent_workers: bool = False,
    split: Optional[str] = None,
    prefetch_factor: Optional[int] = None,
    pin_memory: bool = False,
    sampler: Optional[Any] = None,
    shuffle: Optional[bool] = None,
    dataset: Optional[CachedLatentTextDataset] = None,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    ram_chunk_size: Optional[int] = None,
    accumulate_steps: Optional[int] = None,
):
    if accumulate_steps is None:
        accumulate_steps = int(os.environ.get("ACCUMULATE_STEPS", "3"))
    step_multiple = max(1, int(batch_size) * max(1, int(accumulate_steps)))

    if dataset is None:
        dataset = CachedLatentTextDataset(
            cache_dir,
            split=split,
            pin_memory=pin_memory,
            ram_chunk_size=ram_chunk_size,
            step_multiple=step_multiple,
        )

    do_shuffle = True if shuffle is None else bool(shuffle)
    if sampler is None:
        sampler = DoubleBufferedChunkSampler(
            dataset=dataset,
            step_multiple=step_multiple,
            shuffle=do_shuffle,
            rank=rank if is_distributed else 0,
            world_size=world_size if is_distributed else 1,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        collate_fn=_fast_passthrough_collate,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=True,
        prefetch_factor=None,
    )


def shutdown_dataloader(dataloader: Optional[DataLoader]) -> None:
    if dataloader is None:
        return
    try:
        del dataloader
    except Exception:
        pass
    gc.collect()
