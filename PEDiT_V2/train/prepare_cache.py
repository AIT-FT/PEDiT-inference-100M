import os
import sys
import gc
import itertools
import io
import json
import time
import shutil
import logging
from typing import List, Optional, Tuple

import config
import torch
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from encoders import TextEncoder, TAESD, SDVAE, SDXLVAE, FluxVAE, DCAE, Qwen35TextEncoder

Image.MAX_IMAGE_PIXELS = 100_000_000


VALID_PROMPTS = [
    "A fair-skinned female with a slender build and visible zygomatic arches is captured in a close-up, frontal portrait. Her hair is long, blonde with lighter highlights, and partially obscures the left side of her face. She is smiling broadly, revealing upper and lower teeth. Her eyes are brown with visible irises and pupils. She wears dangling, turquoise earrings and a sleeveless top of an indiscernible color.",
    "A middle-aged man with a robust build and visible facial hair is positioned in a close-up, head-and-shoulders portrait. He has a full, dark brown beard and mustache, with graying interspersed throughout. His skin exhibits moderate wrinkles and texture. He wears rectangular-framed eyeglasses. The man is smiling, revealing upper teeth. He is dressed in a dark navy blue polo shirt.",
    "A close-up, frontal portrait of an infant with light brown skin and dark hair. The infant has a rounded face, visible cheek fullness, and dark eyes. The nose is small with rounded nostrils. Lips are closed and neutral in expression. The infant is wearing a short-sleeved, lime green shirt with horizontal white and yellow stripes across the chest.",
    "A fair-skinned male with short, dark brown curly hair is positioned in a close-up, frontal portrait. Facial features include brown eyes, a slightly asymmetrical nose, and a visible smile exposing upper teeth. He wears rectangular, dark-rimmed glasses. The subject’s shoulders and upper chest are visible, covered by a dark green, textured knit sweater with a ribbed collar.",
    "A close-up, frontal portrait of a fair-skinned woman with short, wavy brown hair. She wears rectangular tortoiseshell-framed eyeglasses reflecting blurred light sources. Her facial features include a prominent nose, thin lips slightly parted, and visible nasolabial folds. Lighting is bright and focused on her face, creating highlights and shadows, and revealing subtle skin texture.",
    "A front-facing headshot depicts a woman with smooth, light-toned skin and visible cheekbone definition. Her dark eyebrows are neatly shaped above dark eyes with visible eyelashes. Lips are full and colored a muted rose tone. She wears a knitted, magenta-colored cap embellished with a pale pink floral detail and a sheer, ruffled white fabric overlay.",
    "A close-up portrait of a cat with yellow eyes, set against a plain indoor background.",
    "A cat with blue eyes is photographed in a portrait style, indoors, against a warm background of red and orange hues.",
    "A black cat with yellow eyes is photographed in a studio setting against a white background in a portrait style.",
    "A cat with green eyes is photographed in a portrait style, standing in front of a white background, with a sunny setting.",
    "A black and white cat is seen in a portrait-style image, standing outside with a collar around its neck, positioned in front of a fan.",
    "A smiling dog with a husky breed, set against a natural backdrop of trees and grass, in a portrait view.",
    "A close-up portrait of a happy dog in a green field.",
    "A happy dog with a tongue hanging out, posed in a portrait against a backdrop of trees in an outdoor setting.",
    "A brown and white dog is depicted outdoors in a portrait, surrounded by trees, with a red collar around its neck."
]


def safe_replace(src: str, dst: str, max_retries: int = 5, delay: float = 0.2):
    for attempt in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError) as e:
            if attempt == max_retries - 1:
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.rename(src, dst)
                    return
                except Exception:
                    raise e
            time.sleep(delay)


def compute_latent_stats(
    cache_dir: str,
    count: int,
    latent_shape: Tuple[int, ...],
    max_samples: int = 20000,
) -> Optional[Tuple[List[float], List[float]]]:
    if count <= 0:
        return None
    latents_path = os.path.join(cache_dir, "latents.fp16")
    if not os.path.exists(latents_path):
        return None

    file_size = os.path.getsize(latents_path)
    sample_bytes = int(np.prod(latent_shape)) * 2
    if sample_bytes == 0:
        return None
    capacity = file_size // sample_bytes
    if capacity < count:
        logger.warning(f"latents.fp16 capacity {capacity} is less than count {count}")
        return None

    latents = np.memmap(
        latents_path,
        dtype=np.float16,
        mode="r",
        shape=(capacity, *latent_shape),
    )

    n = min(max_samples, count)
    rng = np.random.default_rng(0)
    idx = rng.choice(count, size=n, replace=False)
    idx = np.sort(idx)

    chunk_size = 500
    channels = latent_shape[0]
    sum_x = np.zeros(channels, dtype=np.float64)
    sum_sq = np.zeros(channels, dtype=np.float64)
    total_elements = 0

    try:
        for i in range(0, n, chunk_size):
            end_idx = min(i + chunk_size, n)
            chunk_idx = idx[i:end_idx]
            chunk_data = latents[chunk_idx].astype(np.float32)
            chunk_data = np.nan_to_num(chunk_data, nan=0.0, posinf=0.0, neginf=0.0)
            sum_x += chunk_data.sum(axis=(0, 2, 3))
            sum_sq += np.square(chunk_data).sum(axis=(0, 2, 3))
            total_elements += chunk_data.shape[0] * chunk_data.shape[2] * chunk_data.shape[3]
    finally:
        if hasattr(latents, "_mmap") and latents._mmap is not None:
            try:
                latents._mmap.close()
            except Exception:
                pass
        del latents

    if total_elements == 0:
        return None

    mean = sum_x / total_elements
    var = (sum_sq / total_elements) - (mean ** 2)
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var)

    return mean.tolist(), std.tolist()


def write_meta(
    cache_dir: str,
    dataset_name: str,
    count: int,
    limit: int,
    image_size: int,
    seq_len: int,
    hidden_size: int,
    latent_shape: Tuple[int, ...],
    latent_mean: List[float],
    latent_std: List[float],
    text_shape: Optional[Tuple[int, ...]] = None,
    phase_layers: Optional[Tuple[int, ...]] = (8, 16, 28),
    latent_count: Optional[int] = None,
    text_count: Optional[int] = None,
):
    if text_shape is None:
        text_shape = (len(phase_layers), seq_len, hidden_size) if phase_layers and len(phase_layers) > 1 else (seq_len, hidden_size)
    eff_lat_count = count if latent_count is None else latent_count
    eff_txt_count = count if text_count is None else text_count
    meta = {
        "dataset_name": dataset_name,
        "count": min(eff_lat_count, eff_txt_count),
        "latent_count": eff_lat_count,
        "text_count": eff_txt_count,
        "limit": limit,
        "image_size": image_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "latent_shape": list(latent_shape),
        "text_shape": list(text_shape),
        "phase_layers": list(phase_layers) if phase_layers else [28],
        "text_cache_schema_version": 1 if len(text_shape) == 3 else 0,
        "mask_shape": [seq_len],
        "empty_text_features": "empty_text_features.npy",
        "empty_masks": "empty_masks.npy",
        "latent_mean": latent_mean,
        "latent_std": latent_std,
    }
    tmp_meta = os.path.join(cache_dir, "meta.json.tmp")
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    safe_replace(tmp_meta, os.path.join(cache_dir, "meta.json"))


def get_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def extract_image_and_prompt(item, ds_name=""):
    img = None
    image_keys = ["image", "image_path", "img", "file", "jpg", "png", "conditioning_image", "target", "source"]
    for key in image_keys:
        if key in item:
            val = item[key]
            if isinstance(val, Image.Image):
                img = val.copy()
                break
            elif isinstance(val, dict) and "bytes" in val and val["bytes"] is not None:
                try:
                    with Image.open(io.BytesIO(val["bytes"])) as tmp_img:
                        img = tmp_img.copy()
                    break
                except Exception:
                    pass
            elif isinstance(val, bytes):
                try:
                    with Image.open(io.BytesIO(val)) as tmp_img:
                        img = tmp_img.copy()
                    break
                except Exception:
                    pass
            elif isinstance(val, str):
                try:
                    with Image.open(val) as tmp_img:
                        img = tmp_img.copy()
                    break
                except Exception:
                    pass

    prompt = ""
    prompt_keys = ["caption", "text", "prompt", "description", "label", "label_name", "class_name", "category", "tag", "tags", "title", "conversations"]
    ds_name_lower = ds_name.lower()

    for key in prompt_keys:
        if key in item:
            val = item[key]
            if isinstance(val, str) and len(val.strip()) > 0:
                prompt = val.strip()
                break
            elif isinstance(val, dict):
                for subkey in ["caption", "text", "value", "label"]:
                    if subkey in val and isinstance(val[subkey], str) and len(str(val[subkey]).strip()) > 0:
                        prompt = str(val[subkey]).strip()
                        break
                if prompt:
                    break
            elif isinstance(val, list) and len(val) > 0:
                strings = []
                for v in val:
                    if isinstance(v, str):
                        strings.append(v.strip())
                    elif isinstance(v, dict) and "value" in v and isinstance(v["value"], str):
                        strings.append(v["value"].strip())
                if strings:
                    prompt = ", ".join(strings)
                    break
            elif isinstance(val, int):
                if "cub" in ds_name_lower or "bird" in ds_name_lower:
                    prompt = f"a high quality photo of a bird of class {val}"
                elif "animal" in ds_name_lower or "afhq" in ds_name_lower or "dog" in ds_name_lower or "cat" in ds_name_lower:
                    prompt = f"a high quality photo of an animal of class {val}"
                elif "car" in ds_name_lower:
                    prompt = f"a high quality photo of a car of class {val}"
                elif "city" in ds_name_lower:
                    prompt = f"a high quality cityscape photo of class {val}"
                else:
                    prompt = f"a high quality photograph, class {val}"
                break

    return img, prompt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

    token = os.environ.get("HF_TOKEN", None)
    if token in ("", "your_hf_token_here", "None", "null"):
        token = None

    dataset_name = os.environ.get(
        "DATASET_NAME",
        "jxie/coco_captions",
    )

    cache_dir = os.environ.get("CACHE_DIR", "hf_cache_10k_coco")
    os.makedirs(cache_dir, exist_ok=True)

    limit = int(os.environ.get("CACHE_LIMIT", "10000"))
    image_size = int(os.environ.get("IMAGE_SIZE", "512"))
    batch_size = int(os.environ.get("CACHE_BATCH_SIZE", "32"))

    seq_len = int(os.environ.get("SEQ_LEN", "128"))
    hidden_size = int(os.environ.get("TEXT_HIDDEN_SIZE", "2560"))
    vae_type = os.environ.get("VAE_TYPE", "dcae").lower()
    for i, a in enumerate(sys.argv):
        if a == "--vae" and i + 1 < len(sys.argv):
            vae_type = sys.argv[i + 1].lower()
        elif a.startswith("--vae="):
            vae_type = a.split("=", 1)[1].lower()
    spatial_scale = 32 if vae_type in ("dcae", "dc-ae", "dc_ae") else 8
    latent_channels = int(
        os.environ.get(
            "LATENT_CHANNELS",
            "32" if vae_type in ("dcae", "dc-ae", "dc_ae") else ("16" if vae_type in ("flux", "flux1", "flux.1") else "4"),
        )
    )
    latent_h = image_size // spatial_scale
    latent_w = image_size // spatial_scale
    latent_shape = (latent_channels, latent_h, latent_w)

    print(f"Device: {device} | AMP: {amp_dtype}")
    print(f"Dataset: {dataset_name}")
    print(f"Cache dir: {cache_dir}")
    print(f"Limit: {limit}")
    print(f"Batch size: {batch_size}")
    print(f"VAE type: {vae_type} | Latent channels: {latent_channels}")

    def get_dataset_iterator(skip_count=0):
        names = [n.strip() for n in dataset_name.split(",")]
        skip_per_dataset = skip_count // len(names) if names else 0
        extra_skip = skip_count % len(names) if names else 0
        iterators = []
        for idx, name in enumerate(names):
            clean_name = name.split("/")[-1]
            short_aliases = {
                "cub200_dataset": "cub200",
                "cub200": "cub200",
                "MMFace-DiT-Datasets": "mmface",
                "mmface": "mmface",
                "Caption3o-Opt-v3-Tiny": "caption3o",
                "caption3o": "caption3o",
            }
            alias_name = short_aliases.get(clean_name, clean_name.lower())
            roots = [
                "datasets_512",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets_512"),
                "datasets",
                "dataset",
                "data",
                r"C:\ai_proj\dataset",
                r"C:\ai_proj\datasets",
                r"C:\ai_proj\dataset_test",
            ]
            local_candidates = [name]
            for r in roots:
                local_candidates.append(os.path.join(r, alias_name))
                local_candidates.append(os.path.join(r, clean_name))
                local_candidates.append(os.path.join(r, clean_name.lower()))

            local_found = None
            for cand in local_candidates:
                if os.path.isdir(cand):
                    local_found = cand
                    break

            skip_this = skip_per_dataset + (1 if idx < extra_skip else 0)

            if local_found:
                print(f"Loading local dataset from: {local_found} (skipping first {skip_this} samples)")
                def local_generator(path, ds_label, to_skip=0):
                    skipped = 0
                    meta_path = os.path.join(path, "metadata.jsonl")
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    if not line.strip():
                                        continue
                                    try:
                                        rec = json.loads(line)
                                    except Exception:
                                        continue
                                    img_p = rec.get("image_path") or rec.get("image")
                                    if not img_p:
                                        continue
                                    if not os.path.isabs(img_p):
                                        if os.path.exists(img_p):
                                            img_p = os.path.abspath(img_p)
                                        elif os.path.exists(os.path.join(path, img_p)):
                                            img_p = os.path.join(path, img_p)
                                        elif os.path.exists(os.path.join(path, "images", os.path.basename(img_p))):
                                            img_p = os.path.join(path, "images", os.path.basename(img_p))
                                    caption = rec.get("caption") or rec.get("text") or ""
                                    if os.path.exists(img_p):
                                        if skipped < to_skip:
                                            skipped += 1
                                            continue
                                        yield {"image": img_p, "caption": caption, "id": rec.get("id")}
                            return
                        except Exception as e_meta:
                            print(f"[prepare_cache] Notice: reading metadata.jsonl in {path} failed ({e_meta}), scanning files...")

                    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                    img_dir = os.path.join(path, "images") if os.path.isdir(os.path.join(path, "images")) else path
                    txt_dir = os.path.join(path, "captions") if os.path.isdir(os.path.join(path, "captions")) else img_dir

                    files = sorted(os.listdir(img_dir))
                    img_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
                    for img_f in img_files:
                        if skipped < to_skip:
                            skipped += 1
                            continue
                        img_p = os.path.join(img_dir, img_f)
                        base_stem = os.path.splitext(img_f)[0]
                        txt_candidates = [
                            os.path.join(txt_dir, base_stem + ".txt"),
                            os.path.join(img_dir, base_stem + ".txt"),
                            os.path.join(path, base_stem + ".txt"),
                        ]
                        prompt = ""
                        for txt_p in txt_candidates:
                            if os.path.exists(txt_p):
                                try:
                                    with open(txt_p, "r", encoding="utf-8") as tf:
                                        prompt = tf.read().strip()
                                        if prompt:
                                            break
                                except Exception:
                                    pass
                        yield {"image": img_p, "caption": prompt, "id": base_stem}

                iterators.append((name, local_generator(local_found, name, to_skip=skip_this)))
                continue

            print(f"Loading HF streaming dataset: {name}")
            for attempt in range(3):
                try:
                    ds = load_dataset(name, split="train", streaming=True, token=token.strip() if token else None)
                    if skip_this > 0:
                        ds = ds.skip(skip_this)
                    iterators.append((name, iter(ds)))
                    break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed to load {name}: {e}")
                    time.sleep(2)

        seen_ids = set()
        while iterators:
            active_iterators = []
            for name, it in iterators:
                try:
                    while True:
                        item = next(it)
                        unique_id = item.get("cocoid") or item.get("image_id") or item.get("filename") or item.get("id")
                        if unique_id is not None:
                            if unique_id in seen_ids:
                                continue
                            seen_ids.add(unique_id)
                        yield item, name
                        break
                    active_iterators.append((name, it))
                except StopIteration:
                    pass
                except Exception as e:
                    print(f"Error reading from {name}: {e}")
            iterators = active_iterators

    text_encoder_type = os.environ.get("TEXT_ENCODER_TYPE", "qwen35")
    use_multi_layer_text = (text_encoder_type == "qwen35") and (os.environ.get("PHASE_TEXT_LAYERS", "1") == "1")
    phase_layers = (8, 16, 28) if use_multi_layer_text else (28,)
    text_shape = (len(phase_layers), seq_len, hidden_size) if use_multi_layer_text else (seq_len, hidden_size)

    cache_meta_path = os.path.join(cache_dir, "meta.json")
    start_count = 0
    start_text_count = 0
    memmap_mode = "w+"
    latent_mean = [0.0] * latent_channels
    latent_std = [1.0] * latent_channels

    if os.path.exists(cache_meta_path):
        try:
            with open(cache_meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                start_count = meta_data.get("latent_count", meta_data.get("count", 0))
                start_text_count = meta_data.get("text_count", meta_data.get("count", 0))
                cached_limit = meta_data.get("limit", None)
                if cached_limit is not None and cached_limit != limit:
                    print(f"[prepare_cache] Notice: CACHE_LIMIT={limit} differs from meta.json limit={cached_limit}. Using cached limit {cached_limit} to preserve memmap capacity.")
                    limit = cached_limit
                latent_mean = meta_data.get("latent_mean", latent_mean)
                latent_std = meta_data.get("latent_std", latent_std)
                if "text_shape" in meta_data:
                    text_shape = tuple(meta_data["text_shape"])
                if "phase_layers" in meta_data:
                    phase_layers = tuple(meta_data["phase_layers"])
            if start_count > 0:
                memmap_mode = "r+"
        except Exception:
            pass
    else:
        write_meta(
            cache_dir=cache_dir,
            dataset_name="mixed",
            count=0,
            limit=limit,
            image_size=image_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            latent_shape=latent_shape,
            latent_mean=latent_mean,
            latent_std=latent_std,
            text_shape=text_shape,
            phase_layers=phase_layers,
        )

    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def check_disk_space(required_bytes):
        total, used, free = shutil.disk_usage(cache_dir)
        if free < required_bytes:
            raise RuntimeError(f"Not enough disk space. Required: {required_bytes}, Free: {free}")

    if memmap_mode == "w+":
        check_disk_space(
            limit * int(np.prod(latent_shape)) * 2
            + limit * int(np.prod(text_shape)) * 2
            + limit * seq_len * 1
        )

    lat_path = os.path.join(cache_dir, "latents.fp16")
    lat_mode = "r+" if (os.path.exists(lat_path) and os.path.getsize(lat_path) > 0) else "w+"
    latents = np.memmap(
        lat_path,
        dtype=np.float16,
        mode=lat_mode,
        shape=(limit, *latent_shape),
    )

    fp16_text_path = os.path.join(cache_dir, "text_features.fp16")
    fp16_mode = "r+" if (os.path.exists(fp16_text_path) and os.path.getsize(fp16_text_path) > 0) else "w+"
    text_features = np.memmap(
        fp16_text_path,
        dtype=np.float16,
        mode=fp16_mode,
        shape=(limit, *text_shape),
    )

    mask_path = os.path.join(cache_dir, "masks.uint8")
    mask_mode = "r+" if (os.path.exists(mask_path) and os.path.getsize(mask_path) > 0) else "w+"
    masks = np.memmap(
        mask_path,
        dtype=np.uint8,
        mode=mask_mode,
        shape=(limit, seq_len),
    )

    prompts_cache_file = os.path.join(cache_dir, "prompts_cache.json")
    all_prompts = []
    if os.path.exists(prompts_cache_file):
        try:
            with open(prompts_cache_file, "r", encoding="utf-8") as pf:
                all_prompts = json.load(pf)
        except Exception:
            all_prompts = []

    transform = get_transform(image_size)
    vae_batch_size = int(os.environ.get("VAE_BATCH_SIZE", "16"))
    text_encoder_type = os.environ.get("TEXT_ENCODER_TYPE", "qwen35")

    count = len(all_prompts)
    empty_prompt_count = 0
    corrupted_samples_skipped = 0

    if count < limit:
        print(f"\n==================================================================")
        print(f"[Phase 1/2] Loading VAE ({vae_type}) - VRAM usage ~1.2 GB")
        print(f"==================================================================")
        if vae_type in ("dcae", "dc-ae", "dc_ae"):
            vae = DCAE(decoder_only=False)
        elif vae_type == "taesd":
            vae = TAESD()
        elif vae_type == "sd":
            vae = SDVAE()
        elif vae_type in ("sdxl", "sd_xl", "sd-xl"):
            vae = SDXLVAE()
        elif vae_type in ("flux", "flux1", "flux.1"):
            vae = FluxVAE()
        else:
            vae = DCAE(decoder_only=False)
        vae = vae.to(device).eval()

        batch_images = []
        batch_new_prompts = []

        def flush_vae(force_sync=False):
            nonlocal count
            if len(batch_images) == 0:
                return
            n_imgs = len(batch_images)
            chunk_tensors = torch.stack(batch_images).to(device)
            with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=(dev_type == "cuda")):
                latent_batch = vae.encode(chunk_tensors)
            del chunk_tensors
            latents[count : count + n_imgs] = latent_batch.float().cpu().numpy().astype(np.float16)
            del latent_batch
            all_prompts.extend(batch_new_prompts)
            count += n_imgs
            batch_images.clear()
            batch_new_prompts.clear()
            if force_sync or count % 256 == 0 or count >= limit:
                latents.flush()
                tmp_prompts = prompts_cache_file + ".tmp"
                with open(tmp_prompts, "w", encoding="utf-8") as pf:
                    json.dump(all_prompts, pf, ensure_ascii=False)
                safe_replace(tmp_prompts, prompts_cache_file)
                write_meta(
                    cache_dir=cache_dir,
                    dataset_name=dataset_name,
                    count=min(count, start_text_count),
                    latent_count=count,
                    text_count=start_text_count,
                    limit=limit,
                    image_size=image_size,
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    latent_shape=latent_shape,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    text_shape=text_shape,
                    phase_layers=phase_layers,
                )
        dataset_iterator = get_dataset_iterator(count)
        pbar1 = tqdm(total=limit, initial=count, desc="Phase 1: VAE Latents")
        for item_tuple in dataset_iterator:
            if count >= limit:
                break
            try:
                item, current_ds_name = item_tuple
                img, prompt = extract_image_and_prompt(item, current_ds_name)
                if img is None:
                    continue
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                img = img.convert("RGB")
                tensor = transform(img)

                if not prompt:
                    empty_prompt_count += 1
                    fallback_prompts = [
                        "A high quality realistic photograph with natural lighting.",
                        "A detailed studio photograph with soft lighting.",
                        "A clear realistic portrait with natural shadows.",
                        "A sharp, high quality photograph with cinematic light.",
                    ]
                    prompt = fallback_prompts[(count + len(batch_images)) % len(fallback_prompts)]

                batch_images.append(tensor)
                batch_new_prompts.append(prompt)
            except Exception:
                corrupted_samples_skipped += 1
                continue

            if len(batch_images) >= vae_batch_size:
                flush_vae()
                pbar1.update(count - pbar1.n)
                if count >= limit:
                    break

        flush_vae(force_sync=True)
        pbar1.update(count - pbar1.n)
        pbar1.close()

        print(f"\n[Phase 1 Complete] Cached {count} image latents. Unloading VAE to clear VRAM...")
        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    else:
        print(f"\n[Phase 1 Skipped] Already have {count} cached latents & prompts.")

    if count == 0:
        raise RuntimeError("Cache failed: no samples were processed in Phase 1")

    if os.environ.get("COMPUTE_LATENT_STATS", "1") == "1":
        try:
            stat_samples = int(os.environ.get("STAT_SAMPLES", "20000"))
            logger.info(f"Computing empirical latent statistics using up to {stat_samples} samples...")
            stats = compute_latent_stats(
                cache_dir=cache_dir,
                count=count,
                latent_shape=latent_shape,
                max_samples=stat_samples,
            )
            if stats is not None:
                latent_mean, latent_std = stats
                logger.info(f"Computed empirical latent_mean: {latent_mean[:4]}...")
                logger.info(f"Computed empirical latent_std: {latent_std[:4]}...")
                write_meta(
                    cache_dir=cache_dir,
                    dataset_name=dataset_name,
                    count=count,
                    limit=limit,
                    image_size=image_size,
                    seq_len=seq_len,
                    hidden_size=hidden_size,
                    latent_shape=latent_shape,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    text_shape=text_shape,
                    phase_layers=phase_layers,
                )
        except Exception as e:
            logger.warning(f"Failed to compute latent stats: {e}")

    print(f"\n==================================================================")
    print(f"[Phase 2/2] Loading Text Encoder ({text_encoder_type}) in 16-bit into clean VRAM (~5.6 GB)")
    print(f"==================================================================")
    if text_encoder_type == "qwen35":
        text_encoder = Qwen35TextEncoder(
            out_dim=hidden_size,
            include_encoder=True,
            load_in_4bit=False,
            load_in_8bit=False,
            max_length=seq_len,
            target_layers=phase_layers,
        ).to(device)
    else:
        text_encoder = TextEncoder(
            out_dim=1024,
            include_encoder=True,
            hidden_size=hidden_size,
            max_length=seq_len,
        ).to(device)
    text_encoder.eval()

    def encode_text_batch(prompts_batch):
        if text_encoder_type == "qwen35":
            if use_multi_layer_text:
                return text_encoder.encode_phase_layers(
                    prompts_batch, device=device, layers=phase_layers, pad_to_max=True
                )
            return text_encoder.encode_raw(prompts_batch, device=device, pad_to_max=True)
        return text_encoder.encode_raw(prompts_batch, device=device)

    validation_prompts = [""] * len(VALID_PROMPTS) + VALID_PROMPTS
    with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=(dev_type == "cuda")):
        val_features, val_masks = encode_text_batch(validation_prompts)
    np.save(os.path.join(cache_dir, "validation_text_features.npy"), val_features.float().cpu().numpy().astype(np.float16))
    np.save(os.path.join(cache_dir, "validation_masks.npy"), val_masks.cpu().numpy().astype(np.uint8))

    empty_prompt = [""]
    with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=(dev_type == "cuda")):
        empty_features, empty_masks = encode_text_batch(empty_prompt)
    np.save(os.path.join(cache_dir, "empty_text_features.npy"), empty_features.float().cpu().numpy().astype(np.float16))
    np.save(os.path.join(cache_dir, "empty_masks.npy"), empty_masks.cpu().numpy().astype(np.uint8))

    text_batch_size = int(os.environ.get("TEXT_BATCH_SIZE", "32"))
    target_count = min(len(all_prompts), limit)
    resume_txt_idx = min(start_text_count, target_count)
    pbar2 = tqdm(total=target_count, initial=resume_txt_idx, desc="Phase 2: Text encoding (Qwen 16-bit -> FP16)")
    for i in range(resume_txt_idx, target_count, text_batch_size):
        chunk = all_prompts[i : i + text_batch_size]
        with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=(dev_type == "cuda")):
            feat, mask = encode_text_batch(chunk)
        cur_end = i + len(chunk)
        text_features[i:cur_end] = feat.to(torch.float16).cpu().numpy()
        masks[i:cur_end] = mask.cpu().numpy().astype(np.uint8)
        pbar2.update(len(chunk))

        if ((i - resume_txt_idx) // text_batch_size + 1) % 32 == 0 or cur_end >= target_count:
            text_features.flush()
            masks.flush()
            write_meta(
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                count=cur_end,
                latent_count=count,
                text_count=cur_end,
                limit=limit,
                image_size=image_size,
                seq_len=seq_len,
                hidden_size=hidden_size,
                latent_shape=latent_shape,
                latent_mean=latent_mean,
                latent_std=latent_std,
                text_shape=text_shape,
                phase_layers=phase_layers,
            )
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                try:
                    with open(fp16_text_path, "rb") as f_adv:
                        os.posix_fadvise(f_adv.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
    pbar2.close()

    text_features.flush()
    masks.flush()

    print(f"\n[Phase 2 Complete] Encoded 16-bit text features for {target_count} samples. Unloading Text Encoder...")
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    write_meta(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        count=count,
        limit=limit,
        image_size=image_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        latent_shape=latent_shape,
        latent_mean=latent_mean,
        latent_std=latent_std,
        text_shape=text_shape,
        phase_layers=phase_layers,
    )

    ready_path = os.path.join(cache_dir, "cache.ready")
    with open(ready_path, "w", encoding="utf-8") as f:
        f.write(str(count))
    logger.info(f"Marker file created: {ready_path}")
    print(f"Cache finished. Processed {count} samples in 16-bit.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
