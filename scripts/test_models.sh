#!/bin/bash
# =============================================================================
# Model Suitability Test — tests each model against its alias requirements
#
# For each alias, verifies that every model in the alias can:
#   - Respond within the expected latency
#   - Handle the request type (tools, vision, reasoning, etc.)
#   - Return proper format (content field, not just reasoning_content)
#
# Usage: bash scripts/test_models.sh [gateway_url] [api_key]
#        bash scripts/test_models.sh --alias tools   # test only one alias
#
# Requires: curl, python3, jq (optional)
# =============================================================================

GATEWAY_BASE="${1:-http://localhost:4000}"
API_KEY="${2:-sk-my-secret-gateway-key}"

# Parse --alias flag
FILTER_ALIAS=""
for arg in "$@"; do
    if [[ "$prev" == "--alias" ]]; then FILTER_ALIAS="$arg"; fi
    prev="$arg"
done

PASS=0; FAIL=0; SKIP=0; WARN=0
pass() { echo "  PASS: $1"; ((PASS++)); }
fail() { echo "  FAIL: $1"; ((FAIL++)); }
warn() { echo "  WARN: $1"; ((WARN++)); }
skip() { echo "  SKIP: $1"; ((SKIP++)); }

# Helper: send a request directly to LiteLLM (port 4002) with a specific model
probe() {
    local model="$1"
    local payload="$2"
    local timeout="${3:-30}"
    curl -sf --max-time "$timeout" \
        http://localhost:4002/v1/chat/completions \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null
}

echo "============================================================"
echo "Model Suitability Test Suite"
echo "Gateway: $GATEWAY_BASE"
echo "Time:    $(date)"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────────
# Load all models from config
# ─────────────────────────────────────────────────────────────────
MODELS_BY_ALIAS=$(python3 -c "
import yaml, json
with open('litellm_config.yaml') as f:
    cfg = yaml.safe_load(f)
aliases = {}
for entry in cfg.get('model_list', []):
    alias = entry.get('model_name', '')
    model = entry.get('litellm_params', {}).get('model', '')
    if alias and model:
        aliases.setdefault(alias, [])
        if model not in aliases[alias]:
            aliases[alias].append(model)
print(json.dumps(aliases))
" 2>/dev/null)

# ─────────────────────────────────────────────────────────────────
# Test: Tool calling aliases
# ─────────────────────────────────────────────────────────────────
TOOL_ALIASES="tools tools_stable tools_large tools_local tools_cloud bench bench_stable swebench coding"
TOOL_PAYLOAD='{"model":"MODEL_PLACEHOLDER","messages":[{"role":"user","content":"What time is it in London?"}],"tools":[{"type":"function","function":{"name":"get_time","description":"Get current time in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"stream":false,"max_tokens":300}'

for alias in $TOOL_ALIASES; do
    if [[ -n "$FILTER_ALIAS" && "$alias" != "$FILTER_ALIAS" ]]; then continue; fi

    models=$(echo "$MODELS_BY_ALIAS" | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(d.get('$alias',[])))" 2>/dev/null)
    if [[ -z "$models" ]]; then continue; fi

    echo "--- $alias (tool calling) ---"
    while IFS= read -r model; do
        [[ -z "$model" ]] && continue
        payload="${TOOL_PAYLOAD//MODEL_PLACEHOLDER/$model}"
        START=$(date +%s%N)
        RESP=$(probe "$model" "$payload" 45)
        END=$(date +%s%N)
        MS=$(( (END - START) / 1000000 ))

        if [[ -z "$RESP" ]]; then
            fail "$model — timeout/error (${MS}ms)"
            continue
        fi

        VERDICT=$(echo "$RESP" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    if 'error' in d:
        print(f'ERROR:{d[\"error\"].get(\"message\",\"\")[:80]}')
        sys.exit()
    msg=d.get('choices',[{}])[0].get('message',{})
    tc=msg.get('tool_calls')
    c=msg.get('content','') or ''
    rc=msg.get('reasoning_content','') or ''
    fr=d.get('choices',[{}])[0].get('finish_reason','')
    if tc:
        print(f'TOOL_OK:{fr}')
    elif c and len(c) > 5:
        print(f'CONTENT_ONLY:{fr}:no tool_calls but has content')
    elif rc and not c:
        print(f'REASONING_ONLY:{fr}:reasoning_content only, empty content')
    else:
        print(f'EMPTY:{fr}:no content or tool_calls')
except Exception as e:
    print(f'PARSE_ERROR:{e}')
" 2>/dev/null)

        TYPE=$(echo "$VERDICT" | cut -d: -f1)
        DETAIL=$(echo "$VERDICT" | cut -d: -f2-)

        case "$TYPE" in
            TOOL_OK)       pass "$model — tool_calls OK (${MS}ms)" ;;
            CONTENT_ONLY)  warn "$model — responded with content, no tool_calls (${MS}ms)" ;;
            REASONING_ONLY) fail "$model — reasoning_content only, empty content (${MS}ms)" ;;
            EMPTY)         fail "$model — empty response (${MS}ms)" ;;
            ERROR)         fail "$model — $DETAIL (${MS}ms)" ;;
            *)             fail "$model — $VERDICT (${MS}ms)" ;;
        esac
    done <<< "$models"
    echo ""
done

# ─────────────────────────────────────────────────────────────────
# Test: Chat aliases (no tools required)
# ─────────────────────────────────────────────────────────────────
CHAT_ALIASES="default fast thinking big"
CHAT_PAYLOAD='{"model":"MODEL_PLACEHOLDER","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"stream":false,"max_tokens":100}'

for alias in $CHAT_ALIASES; do
    if [[ -n "$FILTER_ALIAS" && "$alias" != "$FILTER_ALIAS" ]]; then continue; fi

    models=$(echo "$MODELS_BY_ALIAS" | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(d.get('$alias',[])))" 2>/dev/null)
    if [[ -z "$models" ]]; then continue; fi

    echo "--- $alias (chat) ---"
    while IFS= read -r model; do
        [[ -z "$model" ]] && continue
        payload="${CHAT_PAYLOAD//MODEL_PLACEHOLDER/$model}"
        START=$(date +%s%N)
        RESP=$(probe "$model" "$payload" 30)
        END=$(date +%s%N)
        MS=$(( (END - START) / 1000000 ))

        if [[ -z "$RESP" ]]; then
            fail "$model — timeout/error (${MS}ms)"
            continue
        fi

        VERDICT=$(echo "$RESP" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    if 'error' in d:
        print(f'ERROR:{d[\"error\"].get(\"message\",\"\")[:80]}')
        sys.exit()
    msg=d.get('choices',[{}])[0].get('message',{})
    c=msg.get('content','') or ''
    rc=msg.get('reasoning_content','') or ''
    if c and len(c.strip()) > 0:
        print(f'OK:{c.strip()[:50]}')
    elif rc and not c:
        print(f'REASONING_ONLY:reasoning_content only')
    else:
        print(f'EMPTY:no content')
except Exception as e:
    print(f'PARSE_ERROR:{e}')
" 2>/dev/null)

        TYPE=$(echo "$VERDICT" | cut -d: -f1)
        DETAIL=$(echo "$VERDICT" | cut -d: -f2-)

        case "$TYPE" in
            OK)             pass "$model — OK (${MS}ms)" ;;
            REASONING_ONLY) fail "$model — reasoning only, no content (${MS}ms)" ;;
            EMPTY)          fail "$model — empty response (${MS}ms)" ;;
            ERROR)          fail "$model — $DETAIL (${MS}ms)" ;;
            *)              fail "$model — $VERDICT (${MS}ms)" ;;
        esac
    done <<< "$models"
    echo ""
done

# ─────────────────────────────────────────────────────────────────
# Test: Vision alias
# ─────────────────────────────────────────────────────────────────
if [[ -z "$FILTER_ALIAS" || "$FILTER_ALIAS" == "vision" ]]; then
    VISION_PAYLOAD='{"model":"MODEL_PLACEHOLDER","messages":[{"role":"user","content":[{"type":"text","text":"What color is the sky? One word."},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPj/HwADBwIAMCbHYQAAAABJRU5ErkJggg=="}}]}],"stream":false,"max_tokens":50}'

    models=$(echo "$MODELS_BY_ALIAS" | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(d.get('vision',[])))" 2>/dev/null)
    if [[ -n "$models" ]]; then
        echo "--- vision ---"
        while IFS= read -r model; do
            [[ -z "$model" ]] && continue
            payload="${VISION_PAYLOAD//MODEL_PLACEHOLDER/$model}"
            START=$(date +%s%N)
            RESP=$(probe "$model" "$payload" 30)
            END=$(date +%s%N)
            MS=$(( (END - START) / 1000000 ))

            if [[ -z "$RESP" ]]; then
                fail "$model — timeout/error (${MS}ms)"
                continue
            fi

            HAS_ERROR=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print('yes' if 'error' in d else 'no')" 2>/dev/null)
            if [[ "$HAS_ERROR" == "yes" ]]; then
                fail "$model — $(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['error'].get('message','')[:80])" 2>/dev/null) (${MS}ms)"
            else
                pass "$model — vision OK (${MS}ms)"
            fi
        done <<< "$models"
        echo ""
    fi
fi

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed, $SKIP skipped, $WARN warnings"
echo "============================================================"

if [ $FAIL -eq 0 ]; then
    echo "ALL MODELS SUITABLE — no misplaced models detected"
    exit 0
else
    echo "FAILURES DETECTED — review model placement in litellm_config.yaml"
    exit 1
fi
