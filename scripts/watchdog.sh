#!/usr/bin/env bash
# =============================================================================
# Watchdog — keeps the gateway alive without Claude
# =============================================================================
# Checks every 60s. Restarts crashed services automatically.
# Logs health checks to logs/watchdog.log.
#
# Usage:
#   bash scripts/watchdog.sh &            # run in background
#   nohup bash scripts/watchdog.sh &      # survive terminal close
#   bash scripts/watchdog.sh --once       # single check, no loop
#
# Stop: bash scripts/stop.sh (also kills watchdog)
#       or: kill $(cat .watchdog.pid)
# =============================================================================

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ONCE="${1:-}"
CHECK_INTERVAL=60
LOG="$ROOT_DIR/logs/watchdog.log"
mkdir -p "$(dirname "$LOG")"

# Save PID
echo $$ > "$ROOT_DIR/.watchdog.pid"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] $*" | tee -a "$LOG"; }

check_and_fix() {
    local issues=0

    # ── Check Redis ──
    if ! docker ps | grep -q llm-cache 2>/dev/null; then
        log "Redis DOWN — restarting"
        docker start llm-cache 2>/dev/null || \
            docker run -d --name llm-cache -p 6379:6379 \
                redis:7-alpine redis-server \
                --maxmemory 2gb --maxmemory-policy allkeys-lru \
                --save "" --appendonly no 2>/dev/null
        sleep 2
        ((issues++))
    fi

    # ── Check LiteLLM ──
    if ! curl -s --max-time 5 http://localhost:4002/health/readiness | grep -q "healthy" 2>/dev/null; then
        log "LiteLLM DOWN — restarting"
        pkill -f "litellm --config" 2>/dev/null || true
        sleep 2
        source "$ROOT_DIR/.env" 2>/dev/null || true
        export REDIS_HOST="${REDIS_HOST:-localhost}" REDIS_PORT="${REDIS_PORT:-6379}"
        uv run litellm --config "$ROOT_DIR/litellm_config.yaml" --port 4002 \
            >> "$ROOT_DIR/logs/litellm/gateway.log" 2>&1 &
        log "LiteLLM restarted (PID: $!)"
        echo "$!" > "$ROOT_DIR/.litellm.pid"

        # Wait for it
        for i in $(seq 1 30); do
            if curl -s --max-time 2 http://localhost:4002/health/readiness | grep -q "healthy" 2>/dev/null; then
                log "LiteLLM ready after ${i}s"
                break
            fi
            sleep 2
        done
        ((issues++))
    fi

    # ── Check Smart Router ──
    if ! curl -s --max-time 5 http://localhost:4000/router/health | grep -q "healthy" 2>/dev/null; then
        log "Router DOWN — restarting"
        pkill -f "smart_router.py" 2>/dev/null || true
        sleep 2
        source "$ROOT_DIR/.env" 2>/dev/null || true
        export LITELLM_BASE="http://localhost:4002" \
               ROUTER_PORT=4000 LOG_LEVEL=INFO \
               LOG_DIR="$ROOT_DIR/logs" \
               CONFIG_PATH="$ROOT_DIR/litellm_config.yaml" \
               GATEWAY_URL="http://localhost:4000" \
               OLLAMA_HOST_1="${OLLAMA_HOST_1:-}" \
               OLLAMA_HOST_2="${OLLAMA_HOST_2:-}"
        uv run python "$ROOT_DIR/smart_router.py" >> "$ROOT_DIR/logs/router/startup.log" 2>&1 &
        log "Router restarted (PID: $!)"
        echo "$!" > "$ROOT_DIR/.router.pid"
        sleep 5
        ((issues++))
    fi

    # ── Health summary ──
    if [[ $issues -eq 0 ]]; then
        # Quiet log — only log health every 10 minutes to avoid noise
        MINUTE=$(date +%M)
        if [[ "$MINUTE" == *0 ]]; then
            HEALTH=$(curl -s --max-time 3 http://localhost:4000/router/health 2>/dev/null || echo "{}")
            log "OK — $(echo "$HEALTH" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    h1=d['ollama_hosts']['OLLAMA_HOST_1']['healthy']
    h2=d['ollama_hosts']['OLLAMA_HOST_2']['healthy']
    print(f'H1:{h1} H2:{h2}')
except: print('health parse error')
" 2>/dev/null)"
        fi
    else
        log "Fixed $issues issue(s)"
    fi
}

# ── Main loop ──
log "Started (PID: $$, interval: ${CHECK_INTERVAL}s)"

if [[ "$ONCE" == "--once" ]]; then
    check_and_fix
    exit 0
fi

while true; do
    check_and_fix
    sleep "$CHECK_INTERVAL"
done
