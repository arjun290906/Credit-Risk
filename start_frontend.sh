#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/Frontend"

echo "==================================================="
echo "  Credit Risk Web App - Frontend Launcher"
echo "==================================================="

if [ ! -d "node_modules" ]; then
    echo "[INFO] node_modules not found. Running npm install..."
    npm install
fi

echo "[INFO] Starting Vite Frontend on http://localhost:5173..."
npm run dev
