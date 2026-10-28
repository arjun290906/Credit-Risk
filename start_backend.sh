#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "==================================================="
echo "  Explainable Credit Risk API - Backend Launcher"
echo "==================================================="

VENV_PY="./Model/.venv/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo "[INFO] Virtual environment not found. Setting up Model/.venv..."
    python3 -m venv ./Model/.venv
    ./Model/.venv/bin/pip install -r requirements.txt
fi

echo "[INFO] Starting FastAPI Backend on http://127.0.0.1:8000..."
./Model/.venv/bin/python -m uvicorn Backend.main:app --host 127.0.0.1 --port 8000 --reload
