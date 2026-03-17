#!/usr/bin/env bash
# Run LiteLLM gateway locally with logs to terminal AND file.
# Logs go to logs/litellm/ with rotation managed by rotatelogs.
# Usage: ./scripts/run_gateway.sh

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load env vars
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"

LOG_DIR="$ROOT_DIR/logs/litellm"
mkdir -p "$LOG_DIR"

LITELLM_LOG="$LOG_DIR/gateway.log"

echo "=== LiteLLM Gateway ==="
echo "  Config:  $ROOT_DIR/litellm_config.yaml"
echo "  Logs:    $LOG_DIR/gateway.log"
echo "  Port:    4002 (internal — router on 4000 is public-facing)"
echo ""

# Run litellm — tee to both terminal and dated log file
uv run litellm \
  --config "$ROOT_DIR/litellm_config.yaml" \
  --port 4002 \
  2>&1 | tee -a "$LITELLM_LOG"
