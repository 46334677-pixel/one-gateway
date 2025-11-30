#!/usr/bin/env bash
set -euo pipefail

# Quickstart: install deps, set .env, run uvicorn, and start a Cloudflare Quick Tunnel.
# Usage:
#   bash scripts/quickstart_tunnel.sh [-k YOUR_GATEWAY_API_KEY]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

KEY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key)
      KEY="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "[1/5] Installing system dependencies..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3 python3-venv python3-pip curl
else
  echo "Please install Python 3.10+, venv, pip, curl manually." >&2
fi

echo "[2/5] Creating virtualenv and installing Python packages..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/5] Preparing .env ..."
if [[ ! -f .env ]]; then
  cp .env.example .env
fi
if [[ -n "${KEY}" ]]; then
  sed -i "s/^GATEWAY_API_KEY=.*/GATEWAY_API_KEY=${KEY}/" .env || true
fi
if ! grep -q "^GATEWAY_API_KEY=" .env || grep -q "GATEWAY_API_KEY=change_me" .env; then
  echo "GATEWAY_API_KEY not set or default. Please input a strong key:" >&2
  read -r -p "Enter GATEWAY_API_KEY: " KEY_INPUT
  sed -i "s/^GATEWAY_API_KEY=.*/GATEWAY_API_KEY=${KEY_INPUT}/" .env
fi

echo "[4/5] Starting uvicorn (background)..."
if pgrep -f "uvicorn app.main:app" >/dev/null; then
  echo "Found running uvicorn. Skipping start.";
else
  nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 >/tmp/uvicorn.log 2>&1 &
  sleep 2
fi
curl -fsS http://127.0.0.1:8000/healthz || { echo "Uvicorn not responding" >&2; exit 1; }
echo "Uvicorn running. Log: /tmp/uvicorn.log"

echo "[5/5] Installing cloudflared and starting Quick Tunnel..."
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL https://pkg.cloudflare.com/install.sh | sudo bash
  sudo apt-get install -y cloudflared
fi
echo "Opening a Quick Tunnel. Keep this process running to keep the URL alive."
echo "When the URL appears like https://xxxx.trycloudflare.com, copy it."
exec cloudflared tunnel --url http://localhost:8000

