import os
import re
import sys
import glob
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Hugging Face models
DEFAULT_TEXT_ENCODER = "google/mt5-small"
VAE_MODELS = {
    "taesd": "madebyollin/taesd",
    "sd": "stabilityai/sd-vae-ft-mse",
    "sdxl": "madebyollin/sdxl-vae-fp16-fix",
}
DEFAULT_HF_REPO = os.environ.get("PEDIT_HF_REPO", "AIT-FT/PEDiT-100M")
STANDARD_MODELS = [
    "PEDiT-100M-FP16.pt",
    "PEDiT-100M-FP8.pt",
    "PEDiT-100M-INT8.pt",
]


def get_project_root() -> str:
    """Возвращает абсолютный путь к папке проекта (final_tren)."""
    return os.path.dirname(os.path.abspath(__file__))


def get_checkpoints_dir(custom_dir: Optional[str] = None) -> str:
    """
    Динамически определяет директорию чекпоинтов без жестких личных путей:
    1. Переданный custom_dir
    2. Переменная окружения CHECKPOINTS_DIR
    3. Относительные папки внутри проекта: checkpoints_100M, checkpoints, cp
    4. По умолчанию: ./checkpoints_100M
    """
    if custom_dir:
        abs_custom = os.path.abspath(custom_dir)
        os.makedirs(abs_custom, exist_ok=True)
        return abs_custom

    env_dir = os.environ.get("CHECKPOINTS_DIR")
    if env_dir:
        abs_env = os.path.abspath(env_dir) if not os.path.isabs(env_dir) else env_dir
        if os.path.isdir(abs_env):
            return abs_env

    root = get_project_root()
    for candidate in ["checkpoints_100M", "checkpoints", "cp"]:
        candidate_path = os.path.join(root, candidate)
        if os.path.isdir(candidate_path):
            return candidate_path

    # Fallback: создаем дефолтную папку checkpoints_100M в корне проекта
    default_dir = os.path.join(root, "checkpoints_100M")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


def download_from_hf(
    filename: str,
    target_dir: str,
    repo_id: str = DEFAULT_HF_REPO
) -> str:
    """
    Скачивает модель напрямую из репозитория Hugging Face в target_dir.
    """
    from huggingface_hub import hf_hub_download
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Downloading '{filename}' from Hugging Face repo '{repo_id}' into '{target_dir}'...")
    dest = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
    )
    logger.info(f"Successfully downloaded: {dest}")
    return dest


def find_available_checkpoints(ckpt_dir: Optional[str] = None, always_include_standard: bool = False) -> List[str]:
    """
    Возвращает список доступных чекпоинтов (.pt / .safetensors), отсортированных по приоритету.
    Если always_include_standard=True, то 3 официальные версии (FP16, FP8, INT8) всегда присутствуют в начале списка.
    """
    target_dir = get_checkpoints_dir(ckpt_dir)
    existing_files = []
    if os.path.exists(target_dir):
        existing_files = [f for f in os.listdir(target_dir) if f.endswith((".pt", ".safetensors", ".ckpt"))]

    def ckpt_priority(filename: str) -> tuple:
        if "PEDiT-100M-FP16" in filename or "_ema_fp16" in filename:
            return (1000000, 10)
        elif "PEDiT-100M-FP8" in filename or "_ema_fp8" in filename:
            return (1000000, 9)
        elif "PEDiT-100M-INT8" in filename or "_ema_int8" in filename:
            return (1000000, 8)

        match = re.search(r'_step_(\d+)', filename)
        if match:
            step = int(match.group(1))
        else:
            digits = re.findall(r'\d+', filename)
            step = int(digits[-1]) if digits else 0
            
        if "_weights" in filename:
            pref = 2
        elif "_ema" in filename:
            pref = 2.5
        else:
            pref = 1
        return (step, pref)

    existing_files.sort(key=ckpt_priority, reverse=True)

    if always_include_standard:
        result = list(STANDARD_MODELS)
        for f in existing_files:
            if f not in result:
                result.append(f)
        return result

    return existing_files


def find_latest_checkpoint(ckpt_dir: Optional[str] = None) -> Optional[str]:
    """Находит последний (с максимальным шагом) чекпоинт в папке."""
    target_dir = get_checkpoints_dir(ckpt_dir)
    ckpts = find_available_checkpoints(target_dir)
    if not ckpts:
        return None
    return os.path.join(target_dir, ckpts[0])


def download_file(url: str, dest_path: str, desc: str = "Downloading file") -> str:
    """Скачивает файл по прямой ссылке HTTP/HTTPS с отображением прогресса."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        import urllib.request
        logger.info(f"{desc}: {url} -> {dest_path}")
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        return dest_path

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = dest_path + ".part"

    logger.info(f"{desc} from {url} to {dest_path}...")
    headers = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"

    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1MB

    with open(temp_path, "wb") as file, tqdm(
        desc=desc,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    logger.info(f"Successfully downloaded to {dest_path}")
    return dest_path


def ensure_text_encoder(model_name: str = DEFAULT_TEXT_ENCODER) -> bool:
    """
    Проверяет наличие и скачивает токенизатор и модель mT5 с Hugging Face Hub.
    """
    logger.info(f"Checking/downloading text encoder '{model_name}'...")
    try:
        from transformers import AutoTokenizer, MT5EncoderModel

        # Hugging Face автоматически скачивает и кэширует компоненты
        AutoTokenizer.from_pretrained(model_name)
        MT5EncoderModel.from_pretrained(model_name)
        logger.info(f"Text encoder '{model_name}' is ready.")
        return True
    except Exception as e:
        logger.error(f"Failed to load/download text encoder '{model_name}': {e}")
        return False


def ensure_vae(vae_type: str = "taesd") -> bool:
    """
    Проверяет наличие и скачивает выбранный VAE с Hugging Face Hub.
    """
    vae_type = vae_type.lower()
    model_name = VAE_MODELS.get(vae_type, VAE_MODELS["taesd"])
    logger.info(f"Checking/downloading VAE ({vae_type}): '{model_name}'...")

    try:
        if vae_type == "taesd":
            from diffusers import AutoencoderTiny
            AutoencoderTiny.from_pretrained(model_name)
        else:
            from diffusers import AutoencoderKL
            AutoencoderKL.from_pretrained(model_name)

        logger.info(f"VAE '{model_name}' is ready.")
        return True
    except Exception as e:
        logger.error(f"Failed to load/download VAE '{model_name}': {e}")
        return False


def ensure_checkpoint(
    checkpoint_path_or_url: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    repo_id: str = DEFAULT_HF_REPO,
) -> Optional[str]:
    """
    Проверяет и при необходимости автоматически скачивает чекпоинт:
    1. Поддерживает алиасы (fp16, fp8, int8, pedit-fp16, etc.)
    2. Если файл существует локально — возвращает его путь.
    3. Если передан HTTP/HTTPS или hf:// — скачивает соответственно.
    4. Если чекпоинт не найден локально — скачивает из Hugging Face репозитория (по умолчанию AIT-FT/PEDiT-100M).
    5. Если имя не указано:
       - Ищет локальные чекпоинты (приоритет у PEDiT-100M-FP16.pt).
       - Если локально ничего нет — скачивает PEDiT-100M-FP16.pt из Hugging Face.
    """
    target_dir = get_checkpoints_dir(ckpt_dir)

    alias_map = {
        "fp16": "PEDiT-100M-FP16.pt",
        "pedit-fp16": "PEDiT-100M-FP16.pt",
        "pedit-100m-fp16": "PEDiT-100M-FP16.pt",
        "pedit-100m-fp16.pt": "PEDiT-100M-FP16.pt",
        "fp8": "PEDiT-100M-FP8.pt",
        "pedit-fp8": "PEDiT-100M-FP8.pt",
        "pedit-100m-fp8": "PEDiT-100M-FP8.pt",
        "pedit-100m-fp8.pt": "PEDiT-100M-FP8.pt",
        "int8": "PEDiT-100M-INT8.pt",
        "pedit-int8": "PEDiT-100M-INT8.pt",
        "pedit-100m-int8": "PEDiT-100M-INT8.pt",
        "pedit-100m-int8.pt": "PEDiT-100M-INT8.pt",
    }

    # 1. Если аргумент передан
    if checkpoint_path_or_url:
        cleaned = str(checkpoint_path_or_url).strip()
        lower_cleaned = cleaned.lower()
        if lower_cleaned in alias_map:
            cleaned = alias_map[lower_cleaned]

        # Прямая ссылка на файл (HTTP/HTTPS)
        if cleaned.startswith(("http://", "https://")):
            filename = cleaned.split("?")[0].split("/")[-1] or "checkpoint.pt"
            dest = os.path.join(target_dir, filename)
            if not os.path.exists(dest):
                download_file(cleaned, dest, desc=f"Downloading {filename}")
            return dest

        # HF hub URI format: hf://repo_id/filename
        if cleaned.startswith("hf://"):
            parts = cleaned[5:].split("/", 1)
            if len(parts) == 2:
                r_id, filename = parts
                return download_from_hf(filename=filename, target_dir=target_dir, repo_id=r_id)

        # Локальный файл по прямому пути
        if os.path.isabs(cleaned) and os.path.exists(cleaned):
            return cleaned

        # Возможно относительное имя внутри ckpt_dir
        candidate = os.path.join(target_dir, os.path.basename(cleaned))
        if os.path.exists(candidate):
            return candidate

        if os.path.exists(cleaned):
            return os.path.abspath(cleaned)

        # Если файл не найден локально — скачиваем с Hugging Face Hub!
        filename_to_dl = os.path.basename(cleaned)
        logger.info(f"Checkpoint '{filename_to_dl}' not found locally. Auto-downloading from Hugging Face ({repo_id})...")
        try:
            return download_from_hf(filename=filename_to_dl, target_dir=target_dir, repo_id=repo_id)
        except Exception as e:
            logger.warning(f"Failed to auto-download '{filename_to_dl}' from Hugging Face ({repo_id}): {e}")

    # 2. Поиск существующего локального чекпоинта
    latest = find_latest_checkpoint(target_dir)
    if latest and os.path.exists(latest):
        logger.info(f"Found local checkpoint: {latest}")
        return latest

    # 3. Попытка скачать по DEFAULT_CHECKPOINT_URL из окружения
    default_url = os.environ.get("DEFAULT_CHECKPOINT_URL") or os.environ.get("CHECKPOINT_URL")
    if default_url:
        logger.info(f"No local checkpoints found. Downloading from DEFAULT_CHECKPOINT_URL: {default_url}")
        filename = default_url.split("?")[0].split("/")[-1] or "PEDiT-100M-FP16.pt"
        dest = os.path.join(target_dir, filename)
        if not os.path.exists(dest):
            download_file(default_url, dest, desc=f"Downloading {filename}")
        return dest

    # 4. Если ничего не найдено локально — скачиваем стандартную модель (FP16) с Hugging Face
    default_model = STANDARD_MODELS[0]
    logger.info(f"No checkpoints found in '{target_dir}'. Auto-downloading default '{default_model}' from Hugging Face ({repo_id})...")
    try:
        return download_from_hf(filename=default_model, target_dir=target_dir, repo_id=repo_id)
    except Exception as e:
        logger.error(f"Failed to auto-download default model from Hugging Face ({repo_id}): {e}")

    logger.warning(
        f"No checkpoints available in '{target_dir}' and failed to download from Hugging Face. "
        f"Please place your .pt checkpoint file in '{target_dir}' or specify --checkpoint <path/url>."
    )
    return None


def download_all(vae_type: str = "taesd", download_ckpt_url: Optional[str] = None, models: Optional[List[str]] = None):
    """Скачивает все базовые модели и зависимости для полной готовности к оффлайн-работе."""
    print("=" * 60)
    print("PEDiT-100M: Checking and Downloading Model Assets")
    print("=" * 60)

    print("\n[1/3] Text Encoder (google/mt5-small)...")
    ensure_text_encoder(DEFAULT_TEXT_ENCODER)

    print(f"\n[2/3] VAE ({vae_type})...")
    ensure_vae(vae_type)

    print("\n[3/3] PEDiT Checkpoints...")
    if models:
        for m in models:
            ckpt = ensure_checkpoint(m)
            if ckpt:
                print(f"Ready: {ckpt}")
    else:
        ckpt = ensure_checkpoint(download_ckpt_url)
        if ckpt:
            print(f"Checkpoint ready at: {ckpt}")

    print("\n[Done] All required components checked and ready.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and verify PEDiT model assets from Hugging Face")
    parser.add_argument("--model", type=str, default=None, choices=["fp16", "fp8", "int8", "all"], help="Specific model precision to download from Hugging Face")
    parser.add_argument("--all", action="store_true", help="Download all base models (encoder, VAE, and all 3 checkpoints)")
    parser.add_argument("--vae", type=str, default="taesd", choices=["taesd", "sd", "sdxl"], help="VAE type to download")
    parser.add_argument("--checkpoint-url", type=str, default=None, help="Optional direct URL to download checkpoint .pt")
    parser.add_argument("--check", action="store_true", help="Check status of assets")

    args = parser.parse_args()

    if args.check:
        print(f"Project root: {get_project_root()}")
        print(f"Checkpoints dir: {get_checkpoints_dir()}")
        ckpts = find_available_checkpoints(always_include_standard=False)
        print(f"Available checkpoints ({len(ckpts)}): {ckpts}")
    elif args.model:
        if args.model == "all":
            download_all(vae_type=args.vae, models=STANDARD_MODELS)
        else:
            ensure_checkpoint(args.model)
    elif args.all:
        download_all(vae_type=args.vae, models=STANDARD_MODELS)
    else:
        download_all(vae_type=args.vae, download_ckpt_url=args.checkpoint_url)
