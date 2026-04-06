#!/bin/bash
# =============================================================================
# Provider Auto-Probe — queries each provider's /v1/models endpoint, compares
# against litellm_config.yaml, and optionally auto-fixes bad model IDs.
#
# Usage:
#   bash scripts/probe_providers.sh              # dry-run (report only)
#   bash scripts/probe_providers.sh --fix        # auto-fix config + restart
#   bash scripts/probe_providers.sh --provider groq  # probe one provider only
#
# Requires: curl, python3, pyyaml
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/litellm_config.yaml"
BACKUP="$PROJECT_DIR/litellm_config.yaml.bak"
ENV_FILE="$PROJECT_DIR/.env"

AUTO_FIX=false
REMOVE_BROKEN=false
FILTER_PROVIDER=""

for arg in "$@"; do
    case "$arg" in
        --fix) AUTO_FIX=true ;;
        --remove-broken) REMOVE_BROKEN=true ;;
        --provider) : ;; # next arg handled below
        *) if [[ "${prev:-}" == "--provider" ]]; then FILTER_PROVIDER="$arg"; fi ;;
    esac
    prev="$arg"
done

# Load env
set -a; source "$ENV_FILE"; set +a

FIXES=0
BROKEN=0
OK=0
SKIPPED=0

echo "============================================================"
echo "Provider Auto-Probe $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config:  $CONFIG"
echo "Mode:    $([ "$AUTO_FIX" = true ] && echo 'AUTO-FIX' || echo 'DRY-RUN')"
[[ -n "$FILTER_PROVIDER" ]] && echo "Filter:  $FILTER_PROVIDER"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────────
# Helper: fetch available models from a provider API
# Returns one model ID per line
# ─────────────────────────────────────────────────────────────────
fetch_models() {
    local provider="$1"
    local url="" headers=""

    case "$provider" in
        groq)
            url="https://api.groq.com/openai/v1/models"
            headers="-H \"Authorization: Bearer $GROQ_API_KEY\""
            ;;
        nvidia_nim)
            url="https://integrate.api.nvidia.com/v1/models"
            headers="-H \"Authorization: Bearer $NVIDIA_NIM_API_KEY\""
            ;;
        gemini)
            url="https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY"
            ;;
        mistral)
            url="https://api.mistral.ai/v1/models"
            headers="-H \"Authorization: Bearer $MISTRAL_API_KEY\""
            ;;
        github)
            url="https://models.inference.ai.azure.com/models"
            headers="-H \"Authorization: Bearer $GITHUB_TOKEN\""
            ;;
        cohere)
            url="https://api.cohere.com/v2/models"
            headers="-H \"Authorization: Bearer $COHERE_API_KEY\""
            ;;
        xai)
            # xAI credits exhausted, skip
            echo "__SKIP__"
            return
            ;;
        cerebras)
            url="https://api.cerebras.ai/v1/models"
            headers="-H \"Authorization: Bearer $CEREBRAS_API_KEY\""
            ;;
        openrouter)
            url="https://openrouter.ai/api/v1/models"
            headers=""
            ;;
        cloudflare)
            # Cloudflare Workers AI has no /models — use catalog API
            url="https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID:-unknown}/ai/models/search"
            headers="-H \"Authorization: Bearer $CLOUDFLARE_API_TOKEN\""
            ;;
        ollama)
            echo "__SKIP__"
            return
            ;;
        openai)
            # Local Ollama endpoints pretending to be OpenAI
            echo "__SKIP__"
            return
            ;;
        *)
            echo "__SKIP__"
            return
            ;;
    esac

    local resp
    resp=$(eval curl -sf --max-time 15 "$url" $headers 2>/dev/null) || { echo "__ERROR__"; return; }

    # Extract model IDs based on provider response format
    echo "$resp" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    # GitHub format: [{name: ..., friendly_name: ...}] (top-level list)
    if isinstance(d, list):
        for m in d:
            name = m.get('name') or m.get('id') or ''
            if name: print(name)
    # Standard OpenAI format: {data: [{id: ...}]}
    elif 'data' in d:
        for m in d['data']:
            mid = m.get('id') or m.get('name') or ''
            if mid: print(mid)
    # Gemini format: {models: [{name: 'models/xxx'}]}
    elif 'models' in d:
        for m in d['models']:
            name = m.get('name', '')
            # Strip 'models/' prefix
            if name.startswith('models/'):
                name = name[7:]
            print(name)
    # Cloudflare format: {result: [{name: ...}]}
    elif 'result' in d:
        for m in d.get('result', []):
            print(m.get('name', ''))
except Exception as e:
    print(f'__PARSE_ERROR__:{e}', file=sys.stderr)
" 2>/dev/null
}

# ─────────────────────────────────────────────────────────────────
# Helper: extract model suffix (strip provider prefix)
# e.g. "groq/meta-llama/llama-4-scout" -> "meta-llama/llama-4-scout"
# ─────────────────────────────────────────────────────────────────
strip_provider_prefix() {
    local model="$1"
    local provider="$2"
    echo "$model" | sed "s|^${provider}/||"
}

# ─────────────────────────────────────────────────────────────────
# Helper: find closest match for a broken model ID
# ─────────────────────────────────────────────────────────────────
find_closest_match() {
    local broken_id="$1"
    local available_file="$2"
    local provider="$3"

    python3 -c "
import sys
from difflib import SequenceMatcher

broken = '$broken_id'
provider = '$provider'
available = open('$available_file').read().strip().split('\n')
available = [a for a in available if a]

if not available:
    sys.exit(1)

# Preserve suffix like :free for openrouter
suffix = ''
base_broken = broken
if ':' in broken:
    parts = broken.rsplit(':', 1)
    base_broken = parts[0]
    suffix = ':' + parts[1]

# Score each available model
scores = []
for a in available:
    base_a = a.rsplit(':', 1)[0] if ':' in a else a

    # Direct substring match (strongest signal)
    if base_broken in base_a or base_a in base_broken:
        scores.append((0.95, a))
        continue

    # Key parts matching
    broken_parts = set(base_broken.lower().replace('-', ' ').replace('_', ' ').split())
    a_parts = set(base_a.lower().replace('-', ' ').replace('_', ' ').split())
    overlap = len(broken_parts & a_parts) / max(len(broken_parts | a_parts), 1)

    # Sequence similarity
    seq = SequenceMatcher(None, base_broken.lower(), base_a.lower()).ratio()

    # Combined score
    combined = 0.6 * seq + 0.4 * overlap
    scores.append((combined, a))

scores.sort(reverse=True)
best_score, best_match = scores[0]

if best_score >= 0.70:
    # If original had a suffix like :free, ensure suggestion keeps it
    if suffix and ':' not in best_match:
        best_match = best_match + suffix
    print(best_match)
else:
    sys.exit(1)
" 2>/dev/null
}

# ─────────────────────────────────────────────────────────────────
# Main: extract config models, probe providers, compare
# ─────────────────────────────────────────────────────────────────

# Get unique (provider, model_id) pairs from config
CONFIG_MODELS=$(python3 -c "
import yaml, json
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
seen = set()
for entry in cfg.get('model_list', []):
    model = entry.get('litellm_params', {}).get('model', '')
    if '/' in model:
        provider = model.split('/')[0]
        model_id = '/'.join(model.split('/')[1:])
        key = f'{provider}|{model_id}'
        if key not in seen:
            seen.add(key)
            print(f'{provider}\t{model_id}\t{model}')
")

# Group by provider
PROVIDERS=$(echo "$CONFIG_MODELS" | cut -f1 | sort -u)

for provider in $PROVIDERS; do
    if [[ -n "$FILTER_PROVIDER" && "$provider" != "$FILTER_PROVIDER" ]]; then continue; fi

    echo "--- $provider ---"

    # Fetch available models
    TMPFILE=$(mktemp)
    AVAILABLE=$(fetch_models "$provider")

    if [[ "$AVAILABLE" == "__SKIP__" ]]; then
        echo "  SKIP: no model listing API (local/custom)"
        ((SKIPPED++)) || true
        rm -f "$TMPFILE"
        echo ""
        continue
    fi

    if [[ "$AVAILABLE" == "__ERROR__" || -z "$AVAILABLE" ]]; then
        echo "  ERROR: could not reach $provider API"
        ((BROKEN++)) || true
        rm -f "$TMPFILE"
        echo ""
        continue
    fi

    echo "$AVAILABLE" > "$TMPFILE"
    AVAIL_COUNT=$(wc -l < "$TMPFILE")
    echo "  API returned $AVAIL_COUNT models"

    # Check each config model against available models
    echo "$CONFIG_MODELS" | grep "^${provider}	" | while IFS=$'\t' read -r prov model_id full_model; do
        # Check if model_id exists in available list
        if grep -qxF "$model_id" "$TMPFILE" 2>/dev/null; then
            echo "  OK:    $full_model"
            # Can't increment OK in subshell, use temp file
            echo "OK" >> "${TMPFILE}.results"
        else
            # Try partial match (some APIs return short names)
            SHORT_NAME=$(basename "$model_id")
            if grep -qiF "$SHORT_NAME" "$TMPFILE" 2>/dev/null; then
                echo "  OK:    $full_model (matched via short name)"
                echo "OK" >> "${TMPFILE}.results"
            else
                echo "  BROKEN: $full_model"

                # Try to find closest match
                MATCH=$(find_closest_match "$model_id" "$TMPFILE" "$provider" 2>/dev/null) || true

                if [[ -n "$MATCH" ]]; then
                    echo "         -> suggestion: ${provider}/${MATCH}"

                    if [[ "$AUTO_FIX" == true ]]; then
                        # Apply fix to config
                        OLD_ESCAPED=$(echo "$full_model" | sed 's|/|\\/|g')
                        NEW_MODEL="${provider}/${MATCH}"
                        NEW_ESCAPED=$(echo "$NEW_MODEL" | sed 's|/|\\/|g')

                        # Only fix if old != new
                        if [[ "$full_model" != "$NEW_MODEL" ]]; then
                            sed -i "s|model: ${full_model}$|model: ${NEW_MODEL}|g" "$CONFIG"
                            echo "         -> FIXED: $full_model => $NEW_MODEL"
                            echo "FIX" >> "${TMPFILE}.results"
                        fi
                    else
                        echo "FIX" >> "${TMPFILE}.results"
                    fi
                else
                    echo "         -> no close match found (may be removed from provider)"
                    if [[ "$REMOVE_BROKEN" == true ]]; then
                        # Remove all config entries with this model
                        python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
before = len(cfg['model_list'])
cfg['model_list'] = [e for e in cfg['model_list'] if e.get('litellm_params',{}).get('model','') != '$full_model']
after = len(cfg['model_list'])
with open('$CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
print(f'removed {before - after} entries')
"
                        echo "         -> REMOVED from config"
                        echo "FIX" >> "${TMPFILE}.results"
                    else
                        echo "BROKEN" >> "${TMPFILE}.results"
                    fi
                fi
            fi
        fi
    done

    # Count results from subshell
    if [[ -f "${TMPFILE}.results" ]]; then
        _ok=$(grep -c "^OK$" "${TMPFILE}.results" 2>/dev/null || true)
        _fix=$(grep -c "^FIX$" "${TMPFILE}.results" 2>/dev/null || true)
        _brk=$(grep -c "^BROKEN$" "${TMPFILE}.results" 2>/dev/null || true)
        OK=$((OK + ${_ok:-0}))
        FIXES=$((FIXES + ${_fix:-0}))
        BROKEN=$((BROKEN + ${_brk:-0}))
    fi

    rm -f "$TMPFILE" "${TMPFILE}.results"
    echo ""
done

# ─────────────────────────────────────────────────────────────────
# Validate YAML after fixes
# ─────────────────────────────────────────────────────────────────
if [[ "$AUTO_FIX" == true && $FIXES -gt 0 ]]; then
    echo "--- Validating config ---"
    if python3 -c "import yaml; yaml.safe_load(open('$CONFIG')); print('YAML OK')"; then
        echo ""
        echo "Restarting services..."

        # Kill existing
        pkill -f "litellm --config" 2>/dev/null || true
        pkill -f "smart_router" 2>/dev/null || true
        sleep 3

        # Restart LiteLLM
        export REDIS_HOST=localhost REDIS_PORT=6379
        nohup uv run litellm --config "$CONFIG" --port 4002 >> "$PROJECT_DIR/logs/litellm/gateway.log" 2>&1 &
        for i in $(seq 1 15); do
            curl -s --max-time 2 http://localhost:4002/health/readiness | grep -q "healthy" 2>/dev/null && echo "LiteLLM ready (${i}s)" && break
            sleep 2
        done

        # Restart Router
        export LITELLM_BASE=http://localhost:4002 ROUTER_PORT=4000 LOG_LEVEL=INFO LOG_DIR="$PROJECT_DIR/logs" CONFIG_PATH="$CONFIG" GATEWAY_URL=http://localhost:4000
        nohup uv run python "$PROJECT_DIR/smart_router.py" > /tmp/router_startup.log 2>&1 &
        sleep 5

        if curl -s http://localhost:4000/router/health | grep -q healthy; then
            echo "All services UP after config fix"
        else
            echo "WARNING: Services may not have restarted cleanly — check manually"
        fi
    else
        echo "YAML BROKEN after fixes — restoring backup"
        cp "$BACKUP" "$CONFIG"
        echo "Restored from backup"
    fi
fi

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "RESULTS: $OK ok, $FIXES fixable, $BROKEN unfixable, $SKIPPED skipped"
if [[ "$AUTO_FIX" == true && $FIXES -gt 0 ]]; then
    echo "Applied $FIXES fixes and restarted services"
elif [[ $FIXES -gt 0 ]]; then
    echo "Run with --fix to auto-apply $FIXES fixes"
fi
echo "============================================================"

[[ $BROKEN -eq 0 ]] && exit 0 || exit 1
