#!/usr/bin/env bash
# Stop the gateway stack
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[stop] Stopping Smart Router..."
pkill -f "smart_router.py" 2>/dev/null || true

echo "[stop] Stopping LiteLLM..."
pkill -f "litellm --config" 2>/dev/null || true

echo "[stop] Stopping watchdog..."
pkill -f "watchdog.sh" 2>/dev/null || true

echo "[stop] Stopping Redis..."
docker stop llm-cache 2>/dev/null || true

rm -f "$ROOT_DIR/.litellm.pid" "$ROOT_DIR/.router.pid" "$ROOT_DIR/.watchdog.pid"
echo "[stop] Done."
