#!/usr/bin/env bash
# =============================================================================
# Start the full gateway stack: Redis + LiteLLM + Smart Router
# =============================================================================
# Usage: bash scripts/start.sh
#        bash scripts/start.sh --bg    (run in background, no terminal output)
#
# Stops any existing instances first, then starts fresh.
# For auto-restart on crash, use: bash scripts/watchdog.sh
# =============================================================================

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BG="${1:-}"

# ── Load environment ──
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export LITELLM_BASE="${LITELLM_BASE:-http://localhost:4002}"
export ROUTER_PORT="${ROUTER_PORT:-4000}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
export CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/litellm_config.yaml}"
export GATEWAY_URL="${GATEWAY_URL:-http://localhost:4000}"
export OLLAMA_HOST_1="${OLLAMA_HOST_1:-}"
export OLLAMA_HOST_2="${OLLAMA_HOST_2:-}"
export OLLAMA_HOST_3="${OLLAMA_HOST_3:-}"

mkdir -p "$LOG_DIR/litellm" "$LOG_DIR/router" "$LOG_DIR/training/conversations"

# ── Stop existing instances ──
echo "[start] Stopping existing processes..."
pkill -f "litellm --config" 2>/dev/null || true
pkill -f "smart_router.py" 2>/dev/null || true
sleep 2

# ── Redis ──
echo "[start] Starting Redis..."
docker start llm-cache 2>/dev/null || \
    docker run -d --name llm-cache -p 6379:6379 \
        redis:7-alpine redis-server \
        --maxmemory 2gb --maxmemory-policy allkeys-lru \
        --save "" --appendonly no 2>&1
sleep 2

# ── LiteLLM ──
echo "[start] Starting LiteLLM on port 4002..."
uv run litellm --config "$CONFIG_PATH" --port 4002 >> "$LOG_DIR/litellm/gateway.log" 2>&1 &
LITELLM_PID=$!
echo "[start] LiteLLM PID: $LITELLM_PID"

# Wait for LiteLLM to be ready (up to 60s)
echo -n "[start] Waiting for LiteLLM..."
for i in $(seq 1 30); do
    if curl -s --max-time 2 http://localhost:4002/health/readiness | grep -q "healthy" 2>/dev/null; then
        echo " ready (${i}s)"
        break
    fi
    echo -n "."
    sleep 2
done

if ! curl -s --max-time 2 http://localhost:4002/health/readiness | grep -q "healthy" 2>/dev/null; then
    echo " FAILED — check logs/litellm/gateway.log"
    exit 1
fi

# ── Smart Router ──
echo "[start] Starting Smart Router on port $ROUTER_PORT..."
uv run python "$ROOT_DIR/smart_router.py" >> "$LOG_DIR/router/startup.log" 2>&1 &
ROUTER_PID=$!
echo "[start] Router PID: $ROUTER_PID"

# Wait for router to be ready
echo -n "[start] Waiting for Router..."
for i in $(seq 1 10); do
    if curl -s --max-time 2 http://localhost:$ROUTER_PORT/router/health | grep -q "healthy" 2>/dev/null; then
        echo " ready (${i}s)"
        break
    fi
    echo -n "."
    sleep 2
done

if ! curl -s --max-time 2 http://localhost:$ROUTER_PORT/router/health | grep -q "healthy" 2>/dev/null; then
    echo " FAILED — check logs/router/startup.log"
    exit 1
fi

# ── Status ──
echo ""
echo "=== Gateway Started ==="
echo "  Router:  http://localhost:$ROUTER_PORT (public)"
echo "  LiteLLM: http://localhost:4002 (internal)"
echo "  Redis:   localhost:$REDIS_PORT"
echo ""

HEALTH=$(curl -s http://localhost:$ROUTER_PORT/router/health 2>/dev/null)
echo "$HEALTH" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for name, info in d.get('ollama_hosts', {}).items():
        print(f'  {name}: {info.get(\"healthy\", \"?\")}')
except Exception:
    print('  Ollama health: ?')
" 2>/dev/null
echo ""
echo "  PIDs: LiteLLM=$LITELLM_PID Router=$ROUTER_PID"
echo "  Stop: bash scripts/stop.sh"
echo "  Logs: tail -f logs/router/router.log"
echo ""

# Save PIDs for stop script
echo "$LITELLM_PID" > "$ROOT_DIR/.litellm.pid"
echo "$ROUTER_PID" > "$ROOT_DIR/.router.pid"
