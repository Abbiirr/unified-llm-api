#!/usr/bin/env bash
# =============================================================================
# Ollama Model Smoke Test
# =============================================================================
# Tests each Ollama model one at a time (waits for model to load before testing).
# Checks: chat, speed, tool calling, context window, thinking/reasoning.
# Outputs a placement recommendation for each model.
#
# Usage:
#   bash scripts/test_ollama_models.sh [HOST_URL] [MODEL_FILTER]
#   bash scripts/test_ollama_models.sh                          # test all on H1
#   bash scripts/test_ollama_models.sh http://192.168.0.73:11434
#   bash scripts/test_ollama_models.sh http://10.112.30.10:11434 gurubot
# =============================================================================

set -euo pipefail

HOST="${1:-http://10.112.30.10:11434}"
FILTER="${2:-}"
TIMEOUT=300  # Max seconds to wait for model load

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

pass=0
fail=0
skip=0
results=()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

log_pass() { echo -e "  ${GREEN}✓ PASS${NC} $1"; ((pass++)); }
log_fail() { echo -e "  ${RED}✗ FAIL${NC} $1"; ((fail++)); }
log_skip() { echo -e "  ${YELLOW}⊘ SKIP${NC} $1"; ((skip++)); }
log_info() { echo -e "  ${CYAN}ℹ${NC} $1"; }

# Wait for a model to be loaded and responsive
wait_for_model() {
    local model="$1"
    local start=$SECONDS
    echo -e "\n${BOLD}Loading model: ${model}${NC}"
    echo -n "  Waiting for model to respond"

    while true; do
        elapsed=$((SECONDS - start))
        if [ $elapsed -ge $TIMEOUT ]; then
            echo ""
            log_fail "Model did not load within ${TIMEOUT}s"
            return 1
        fi

        # Send a tiny request to trigger model load
        resp=$(curl -s --max-time 60 "${HOST}/api/chat" \
            -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false,\"options\":{\"num_ctx\":2048,\"num_predict\":5}}" 2>/dev/null || true)

        if echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('message',{}).get('content') else 1)" 2>/dev/null; then
            echo ""
            log_info "Model loaded in ${elapsed}s"
            return 0
        fi

        echo -n "."
        sleep 3
    done
}

# Get model info
get_model_info() {
    local model="$1"
    curl -s "${HOST}/api/show" -d "{\"name\":\"${model}\"}" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
info = d.get('details', {})
mi = d.get('model_info', {})
params = d.get('parameters', '')
tmpl = d.get('template', '')
ctx = 0
for k, v in mi.items():
    if 'context_length' in k:
        ctx = v
has_builtin_ctx = 'num_ctx' in params
has_tool_tmpl = '<tool_call>' in tmpl or 'tools' in tmpl.lower()
print(f'family={info.get(\"family\",\"?\")}')
print(f'params={info.get(\"parameter_size\",\"?\")}')
print(f'quant={info.get(\"quantization_level\",\"?\")}')
print(f'context={ctx}')
print(f'has_builtin_ctx={has_builtin_ctx}')
print(f'has_tool_template={has_tool_tmpl}')
" 2>/dev/null
}

# Run a chat test and return speed
test_chat() {
    local model="$1"
    local prompt="$2"
    local num_ctx="${3:-8192}"

    curl -s --max-time 120 "${HOST}/api/chat" \
        -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"stream\":false,\"options\":{\"num_ctx\":${num_ctx}}}" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
msg = d.get('message', {})
content = msg.get('content', '')
eval_count = d.get('eval_count', 0)
eval_dur = d.get('eval_duration', 1)
tok_s = round(eval_count / (eval_dur / 1e9), 1) if eval_dur > 0 else 0
total_dur = d.get('total_duration', 0) / 1e9
print(f'content_len={len(content)}')
print(f'eval_tokens={eval_count}')
print(f'tok_per_sec={tok_s}')
print(f'total_sec={total_dur:.1f}')
print(f'content_preview={content[:150]}')
" 2>/dev/null
}

# Test tool calling via OpenAI-compatible endpoint
test_tool_calling() {
    local model="$1"

    curl -s --max-time 120 "${HOST}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Get the current weather in Paris\"}],
            \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}],
            \"stream\": false
        }" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'choices' not in d:
    print('tool_call=error')
    print(f'error={json.dumps(d)[:200]}')
    sys.exit(0)
msg = d['choices'][0]['message']
tc = msg.get('tool_calls', [])
if tc:
    fn = tc[0].get('function', {})
    print(f'tool_call=yes')
    print(f'function_name={fn.get(\"name\",\"?\")}')
    print(f'arguments={fn.get(\"arguments\",\"?\")}')
else:
    content = msg.get('content', '')
    print(f'tool_call=no')
    print(f'content_preview={content[:100]}')
" 2>/dev/null
}

# Test multi-turn tool calling
test_multi_turn_tools() {
    local model="$1"

    curl -s --max-time 120 "${HOST}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${model}"'",
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temp\": 18, \"condition\": \"sunny\"}"},
                {"role": "user", "content": "Is it warm enough for a walk?"}
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
            "stream": false
        }' 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'choices' not in d:
    print('multi_turn=error')
    sys.exit(0)
msg = d['choices'][0]['message']
content = msg.get('content', '') or ''
# Should reference the 18°C/sunny weather from tool result
has_context = any(w in content.lower() for w in ['18', 'sunny', 'warm', 'yes', 'walk', 'enjoy', 'pleasant'])
print(f'multi_turn={"yes" if has_context else "partial"}')
print(f'content_preview={content[:150]}')
" 2>/dev/null
}

# Test context window with increasing sizes
test_context_window() {
    local model="$1"
    local max_ctx="$2"
    local sizes=(8192 32768 65536 131072)

    for sz in "${sizes[@]}"; do
        if [ "$sz" -gt "$max_ctx" ]; then
            log_skip "Context ${sz} (exceeds model max ${max_ctx})"
            continue
        fi

        # Generate a prompt that fills ~80% of the context
        fill_tokens=$((sz * 3 / 4))
        fill_chars=$((fill_tokens * 4))  # ~4 chars per token
        # Generate filler text
        filler=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * ($fill_chars // 47 + 1))")
        prompt="Read this text and tell me the LAST word: ${filler}"
        prompt_len=${#prompt}

        start=$SECONDS
        resp=$(curl -s --max-time 180 "${HOST}/api/chat" \
            -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Respond with just OK.\"}],\"stream\":false,\"options\":{\"num_ctx\":${sz},\"num_predict\":10}}" 2>/dev/null || echo '{}')
        elapsed=$((SECONDS - start))

        has_response=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print('yes' if d.get('message',{}).get('content') else 'no')" 2>/dev/null || echo "no")

        if [ "$has_response" = "yes" ]; then
            tok_s=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); e=d.get('eval_duration',1); print(round(d.get('eval_count',0)/(e/1e9),1) if e>0 else 0)" 2>/dev/null || echo "?")
            log_pass "Context ${sz}: responded in ${elapsed}s (${tok_s} tok/s)"
        else
            log_fail "Context ${sz}: no response in ${elapsed}s"
            break  # No point testing larger contexts
        fi
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${BOLD}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         Ollama Model Smoke Test Suite                 ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════╝${NC}"
echo -e "Host: ${HOST}"
echo -e "Filter: ${FILTER:-all models}"
echo ""

# Check host reachability
if ! curl -s --max-time 5 "${HOST}/api/tags" >/dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot reach Ollama at ${HOST}${NC}"
    exit 1
fi

# Get model list
models=$(curl -s "${HOST}/api/tags" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for m in d.get('models', []):
    name = m['name']
    # Skip embedding and OCR models
    fam = m.get('details', {}).get('family', '')
    if fam in ('nomic-bert', 'paddleocr', 'glmocr'):
        continue
    print(name)
")

echo -e "${BOLD}Models to test:${NC}"
for m in $models; do
    if [ -n "$FILTER" ] && [[ "$m" != *"$FILTER"* ]]; then
        continue
    fi
    echo "  - $m"
done
echo ""

for model in $models; do
    # Apply filter
    if [ -n "$FILTER" ] && [[ "$model" != *"$FILTER"* ]]; then
        continue
    fi

    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Model: ${model}${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # ── Step 1: Get model info ──
    echo -e "\n${CYAN}[1/6] Model Info${NC}"
    info_output=$(get_model_info "$model")
    eval "$info_output" 2>/dev/null || true

    log_info "Family: ${family:-?}, Params: ${params:-?}, Quant: ${quant:-?}"
    log_info "Max context: ${context:-?}, Built-in num_ctx: ${has_builtin_ctx:-?}"
    log_info "Template tool support: ${has_tool_template:-?}"

    # ── Step 2: Wait for model to load ──
    echo -e "\n${CYAN}[2/6] Loading Model${NC}"
    if ! wait_for_model "$model"; then
        results+=("${model}|FAILED|Could not load")
        continue
    fi

    # ── Step 3: Chat speed test ──
    echo -e "\n${CYAN}[3/6] Chat Speed Test${NC}"
    chat_output=$(test_chat "$model" "What is 2+2? Answer with just the number." 8192)
    eval "$chat_output" 2>/dev/null || true

    if [ -n "${tok_per_sec:-}" ] && [ "${tok_per_sec}" != "0" ]; then
        log_pass "Chat: ${tok_per_sec} tok/s, ${eval_tokens:-?} tokens in ${total_sec:-?}s"
        log_info "Preview: ${content_preview:-}"
        speed="${tok_per_sec}"
    else
        log_fail "Chat: no response or zero speed"
        speed="0"
    fi

    # ── Step 4: Tool calling test ──
    echo -e "\n${CYAN}[4/6] Tool Calling Test${NC}"
    tool_output=$(test_tool_calling "$model")
    eval "$tool_output" 2>/dev/null || true

    has_tools="no"
    if [ "${tool_call:-no}" = "yes" ]; then
        log_pass "Tool calling: ${function_name:-?}(${arguments:-?})"
        has_tools="yes"

        # Also test multi-turn
        mt_output=$(test_multi_turn_tools "$model")
        eval "$mt_output" 2>/dev/null || true
        if [ "${multi_turn:-no}" = "yes" ] || [ "${multi_turn:-no}" = "partial" ]; then
            log_pass "Multi-turn tools: ${multi_turn} — ${content_preview:-}"
        else
            log_fail "Multi-turn tools: failed"
            has_tools="partial"
        fi
    elif [ "${tool_call:-no}" = "error" ]; then
        log_fail "Tool calling: error — ${error:-}"
    else
        log_skip "Tool calling: model does not emit tool_calls"
        log_info "Response instead: ${content_preview:-}"
    fi

    # ── Step 5: Context window test ──
    echo -e "\n${CYAN}[5/6] Context Window Test${NC}"
    max_ctx="${context:-8192}"
    test_context_window "$model" "$max_ctx"

    # ── Step 6: Reasoning/thinking test ──
    echo -e "\n${CYAN}[6/6] Reasoning Test${NC}"
    reason_output=$(test_chat "$model" "What is 15 factorial? Show your reasoning step by step. Be brief." 8192)
    eval "$reason_output" 2>/dev/null || true

    has_reasoning="no"
    if echo "${content_preview:-}" | grep -qi -E "reasoning|step|think|1307674368000|15!"; then
        log_pass "Reasoning: model shows step-by-step thinking"
        has_reasoning="yes"
    else
        log_info "Reasoning: ${content_preview:-}"
        has_reasoning="minimal"
    fi

    # ── Placement recommendation ──
    echo -e "\n${CYAN}Placement Recommendation:${NC}"
    placements=""
    if [ "$has_tools" = "yes" ]; then
        placements="tools_local, swebench, tools_large"
    fi
    if [ "$has_reasoning" = "yes" ]; then
        placements="${placements:+${placements}, }thinking"
    fi
    # All models can do coding and default
    placements="${placements:+${placements}, }coding, default, bench, local"

    # Speed-based placement
    speed_int=$(echo "$speed" | cut -d. -f1)
    if [ "${speed_int:-0}" -ge 20 ]; then
        placements="${placements}, fast"
    fi

    log_info "Speed: ${speed} tok/s | Tools: ${has_tools} | Reasoning: ${has_reasoning} | Context: ${max_ctx}"
    log_info "Suggested aliases: ${placements}"

    results+=("${model}|${speed}tok/s|tools=${has_tools}|reason=${has_reasoning}|ctx=${max_ctx}|→ ${placements}")
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                    SUMMARY                            ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════╝${NC}"
echo -e "  ${GREEN}Pass: ${pass}${NC}  ${RED}Fail: ${fail}${NC}  ${YELLOW}Skip: ${skip}${NC}"
echo ""

echo -e "${BOLD}Model Placement Matrix:${NC}"
printf "%-55s %s\n" "Model" "Result"
printf "%-55s %s\n" "$(printf '%.0s─' {1..55})" "$(printf '%.0s─' {1..60})"
for r in "${results[@]}"; do
    model=$(echo "$r" | cut -d'|' -f1)
    rest=$(echo "$r" | cut -d'|' -f2-)
    printf "%-55s %s\n" "$model" "$rest"
done
echo ""
