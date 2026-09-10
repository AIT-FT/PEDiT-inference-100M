#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "          PEDiT-100M - Hugging Face Model Downloader"
echo "================================================================"
echo "Repository: https://huggingface.co/AIT-FT/PEDiT-100M"
echo ""

if [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found in PATH!"
    exit 1
fi

PY_BIN=$(command -v python3 || command -v python)

echo "Select which models to download from Hugging Face:"
echo "  1) FP16 (Recommended, 206 MB)"
echo "  2) FP8  (Fast & Light, 103 MB)"
echo "  3) INT8 (Maximum Speed, 103 MB)"
echo "  4) ALL  (FP16 + FP8 + INT8 + VAE + Text Encoder)"
echo "  5) Check local checkpoints status"
echo ""

read -p "Enter choice [1-5, default: 1]: " CHOICE
CHOICE=${CHOICE:-1}

case "$CHOICE" in
    1)
        "$PY_BIN" download_utils.py --model fp16
        ;;
    2)
        "$PY_BIN" download_utils.py --model fp8
        ;;
    3)
        "$PY_BIN" download_utils.py --model int8
        ;;
    4)
        "$PY_BIN" download_utils.py --all
        ;;
    5)
        "$PY_BIN" download_utils.py --check
        ;;
    *)
        echo "[INFO] Invalid choice, downloading default FP16..."
        "$PY_BIN" download_utils.py --model fp16
        ;;
esac

echo ""
echo "[SUCCESS] Done!"
