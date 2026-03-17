#!/usr/bin/env bash
# Run smart router locally.
# Logs go to logs/router/ (rotated automatically by the router).
# Usage: ./scripts/run_router.sh

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export LITELLM_BASE="${LITELLM_BASE:-http://localhost:4002}"
export ROUTER_PORT="${ROUTER_PORT:-4000}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

echo "=== Smart Router ==="
echo "  Backend: $LITELLM_BASE"
echo "  Port:    $ROUTER_PORT (public-facing)"
echo "  Logs:    $LOG_DIR/router/router.log"
echo "  Training: $LOG_DIR/training/routing.jsonl"
echo ""

uv run python "$ROOT_DIR/smart_router.py"
