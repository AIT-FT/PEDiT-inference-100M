#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "                  PEDiT - CLI Inference"
echo "================================================================"
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

read -p "Enter prompt (press Enter for default): " PROMPT_TEXT
PROMPT_TEXT=${PROMPT_TEXT:-"A cinematic portrait of a cybernetic cat, neon lighting, 8k"}

echo ""
echo "[INFO] Generating image..."
"$PY_BIN" inference.py --prompt "$PROMPT_TEXT" --steps 8 --cfg 4.0 --output outputs/sample.png "$@"

if [ -f "outputs/sample.png" ]; then
    echo "[SUCCESS] Saved to outputs/sample.png"
fi
