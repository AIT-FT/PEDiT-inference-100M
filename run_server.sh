#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "                   PEDiT - Web Server"
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

echo "[INFO] Starting server (checking dependencies and launching)..."
echo "[INFO] Web UI will be available at: http://localhost:8000"
echo ""

"$PY_BIN" server.py "$@"
