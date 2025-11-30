#!/usr/bin/env bash
set -euo pipefail

echo "Stopping uvicorn and cloudflared if running..."
pkill -f "uvicorn app.main:app" || true
pkill -f "cloudflared tunnel --url" || true
echo "Done."

