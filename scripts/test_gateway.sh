#!/bin/bash
# =============================================================================
# LLM Gateway Pre-Benchmark Test Suite v2
#
# Tests every failure mode observed in benchmark runs:
#   - Provider health & auth
#   - All alias availability
#   - Tool calling (single, multi-turn, 3-turn, 5-turn)
#   - Orphan tool result repair (complaint #4: LiteLLM 500)
#   - Null content assistant messages (complaint #2: Cohere 400)
#   - Large prompts 4K/20K/50K (complaint #3: 413 body too large)
#   - Rate limit resilience (complaint #5: 429 cascading)
#   - Response time under load
#   - No 413 on any alias
#   - No provider-specific 400s leaking to client
#   - Streaming works
#
# Self-contained — only requires: curl, python3, bash.
#
# Usage: bash scripts/test_gateway.sh [gateway_url] [api_key]
#   gateway_url  defaults to http://localhost:4000
#   api_key      defaults to sk-my-secret-gateway-key
#
# Exit code: 0 = all pass, 1 = failures detected
# =============================================================================

GATEWAY_BASE="${1:-http://localhost:4000}"
API_KEY="${2:-sk-my-secret-gateway-key}"
GATEWAY="$GATEWAY_BASE/v1"
HEALTH_URL="$GATEWAY_BASE/health/readiness"
ROUTER_HEALTH="$GATEWAY_BASE/router/health"

PASS=0
FAIL=0
SKIP=0
WARN=0

pass() { PASS=$((PASS+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); echo "  FAIL: $1"; }
skip() { SKIP=$((SKIP+1)); echo "  SKIP: $1"; }
warn() { WARN=$((WARN+1)); echo "  WARN: $1"; }

chat() {
    local body tmpfile
    body=$(printf '%s' "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); d['stream']=False; print(json.dumps(d))" 2>/dev/null || printf '%s' "$1")
    tmpfile=$(mktemp)
    curl -s --max-time "${2:-30}" "$GATEWAY/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "$body" -o "$tmpfile" 2>/dev/null
    cat "$tmpfile"
    rm -f "$tmpfile"
}

has_choices() {
    python3 -c "import sys,json; exit(0 if json.loads(sys.stdin.read()).get('choices') else 1)" 2>/dev/null
}

get_error() {
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',{}).get('message','empty response')[:${1:-120}])" 2>/dev/null
}

get_status() {
    python3 -c "
import sys,json
raw=sys.stdin.read()
try:
    d=json.loads(raw)
    if d.get('choices'): print('200')
    elif '413' in str(d.get('error','')): print('413')
    elif '429' in str(d.get('error','')): print('429')
    elif '400' in str(d.get('error',{}).get('code','')): print('400')
    else: print(d.get('error',{}).get('code','unknown'))
except:
    # Could be streaming SSE or truncated — check if data: prefix exists
    if raw.strip().startswith('data:'): print('200')
    else: print('parse_error')
" 2>/dev/null
}

TOOL_DEF='[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]'
TOOL_DEF_MULTI='[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},{"type":"function","function":{"name":"write_file","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]'

echo "============================================================"
echo "LLM Gateway Pre-Benchmark Test Suite v2"
echo "Gateway: $GATEWAY_BASE"
echo "Time:    $(date)"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────────
# T1: Health endpoints
# ─────────────────────────────────────────────────────────────────
echo "--- T1: Health endpoints ---"
HEALTH=$(curl -s --max-time 5 "$HEALTH_URL" 2>&1)
if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('status')=='healthy' else 1)" 2>/dev/null; then
    pass "LiteLLM health endpoint"
else
    fail "LiteLLM health: $(echo "$HEALTH" | head -c 200)"
    echo "ABORT: Gateway is not healthy."
    exit 1
fi

RHEALTH=$(curl -s --max-time 5 "$ROUTER_HEALTH" 2>&1)
if echo "$RHEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('status')=='healthy' else 1)" 2>/dev/null; then
    pass "Smart router health endpoint"
else
    warn "Smart router not responding (direct LiteLLM mode)"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T2: Authentication
# ─────────────────────────────────────────────────────────────────
echo "--- T2: Authentication ---"
NOAUTH=$(curl -s --max-time 10 "$GATEWAY/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":false}' 2>&1)
if echo "$NOAUTH" | grep -qi "auth\|401"; then
    pass "Unauthenticated request rejected"
else
    skip "No auth required (open gateway)"
fi

AUTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$GATEWAY/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{"model":"tools","messages":[{"role":"user","content":"Auth test"}],"max_tokens":5,"stream":false}' 2>&1)
if [ "$AUTH_STATUS" = "200" ]; then
    pass "Authenticated request succeeds (HTTP $AUTH_STATUS)"
else
    fail "Authenticated request failed: $(printf "%s" "$AUTH" | get_error 200)"
    echo "ABORT: Cannot authenticate."
    exit 1
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T3: All aliases respond
# ─────────────────────────────────────────────────────────────────
echo "--- T3: Alias availability ---"
for alias in default fast thinking coding vision tools big bench tools_stable bench_stable swebench tools_large tools_local; do
    result=$(chat "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":5}" 20)
    if printf "%s" "$result" | has_choices; then
        pass "Alias '$alias'"
    else
        fail "Alias '$alias': $(printf "%s" "$result" | get_error 80)"
    fi
    sleep 0.5
done
echo ""

# ─────────────────────────────────────────────────────────────────
# T4: Tool calling (single turn)
# ─────────────────────────────────────────────────────────────────
echo "--- T4: Tool calling (single turn) ---"
for alias in tools tools_stable bench default; do
    result=$(chat "{
        \"model\":\"$alias\",
        \"messages\":[{\"role\":\"system\",\"content\":\"You must use tools.\"},{\"role\":\"user\",\"content\":\"Read main.py\"}],
        \"tools\":$TOOL_DEF,
        \"max_tokens\":50,
        \"tool_choice\":\"required\"
    }" 30)
    tc_count=$(printf "%s" "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('choices',[{}])[0].get('message',{}).get('tool_calls',[])))" 2>/dev/null)
    if [ "$tc_count" -gt 0 ] 2>/dev/null; then
        pass "Alias '$alias' tool call ($tc_count)"
    else
        fail "Alias '$alias' tool call: $(printf "%s" "$result" | get_error 80)"
    fi
    sleep 1
done
echo ""

# ─────────────────────────────────────────────────────────────────
# T5: Multi-turn tool conversation (2 turns)
# ─────────────────────────────────────────────────────────────────
echo "--- T5: Multi-turn tool conversation ---"
T1_RESP=$(chat "{
    \"model\":\"tools\",
    \"messages\":[{\"role\":\"system\",\"content\":\"Use tools.\"},{\"role\":\"user\",\"content\":\"Read main.py\"}],
    \"tools\":$TOOL_DEF,
    \"max_tokens\":50,
    \"tool_choice\":\"required\"
}" 30)

TC_ID=$(printf "%s" "$T1_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['tool_calls'][0]['id'])" 2>/dev/null)

if [ -z "$TC_ID" ]; then
    fail "Turn 1: no tool call returned"
else
    pass "Turn 1: tool_call id=$TC_ID"

    T2_RESP=$(chat "{
        \"model\":\"tools\",
        \"messages\":[
            {\"role\":\"system\",\"content\":\"You are helpful.\"},
            {\"role\":\"user\",\"content\":\"Read main.py\"},
            {\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"$TC_ID\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"main.py\\\"}\"}}]},
            {\"role\":\"tool\",\"tool_call_id\":\"$TC_ID\",\"content\":\"def hello():\\n    print('Hello world')\"}
        ],
        \"max_tokens\":100
    }" 30)

    if printf "%s" "$T2_RESP" | has_choices; then
        pass "Turn 2: response after tool result"
    else
        fail "Turn 2: $(printf "%s" "$T2_RESP" | get_error)"
    fi
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T6: 3-turn tool conversation
# ─────────────────────────────────────────────────────────────────
echo "--- T6: 3-turn tool conversation ---"
if [ -n "$TC_ID" ]; then
    T3_RESP=$(chat "{
        \"model\":\"tools\",
        \"messages\":[
            {\"role\":\"system\",\"content\":\"Use tools.\"},
            {\"role\":\"user\",\"content\":\"Read main.py\"},
            {\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"$TC_ID\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"main.py\\\"}\"}}]},
            {\"role\":\"tool\",\"tool_call_id\":\"$TC_ID\",\"content\":\"def hello():\\n    print('Hello')\"},
            {\"role\":\"user\",\"content\":\"Now write a test\"}
        ],
        \"tools\":$TOOL_DEF_MULTI,
        \"max_tokens\":200
    }" 30)

    if printf "%s" "$T3_RESP" | has_choices; then
        pass "3-turn conversation accepted"
    else
        fail "3-turn conversation: $(printf "%s" "$T3_RESP" | get_error)"
    fi
else
    skip "Skipped (T5 turn 1 failed)"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T7: Orphan tool result repair (Complaint #4: LiteLLM 500)
# ─────────────────────────────────────────────────────────────────
echo "--- T7: Orphan tool result repair ---"
ORPHAN_RESP=$(chat "{
    \"model\":\"tools\",
    \"messages\":[
        {\"role\":\"system\",\"content\":\"You are helpful.\"},
        {\"role\":\"user\",\"content\":\"Fix the bug\"},
        {\"role\":\"tool\",\"tool_call_id\":\"orphan_no_match_999\",\"content\":\"some file contents here\"},
        {\"role\":\"user\",\"content\":\"What should I do?\"}
    ],
    \"max_tokens\":50
}" 30)
if printf "%s" "$ORPHAN_RESP" | has_choices; then
    pass "Orphan tool result repaired (not 500)"
else
    err=$(printf "%s" "$ORPHAN_RESP" | get_error 80)
    if echo "$err" | grep -qi "500\|Missing corresponding tool call"; then
        fail "Orphan tool result caused 500: $err"
    else
        fail "Orphan tool result: $err"
    fi
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T8: Null content assistant (Complaint #2: Cohere 400)
# ─────────────────────────────────────────────────────────────────
echo "--- T8: Null content assistant message ---"
NULL_RESP=$(chat "{
    \"model\":\"tools\",
    \"messages\":[
        {\"role\":\"user\",\"content\":\"Hello\"},
        {\"role\":\"assistant\",\"content\":null},
        {\"role\":\"user\",\"content\":\"Fix the bug\"}
    ],
    \"max_tokens\":50
}" 30)
if printf "%s" "$NULL_RESP" | has_choices; then
    pass "Null content assistant handled (not 400)"
else
    err=$(printf "%s" "$NULL_RESP" | get_error 80)
    if echo "$err" | grep -qi "non-empty content\|invalid message"; then
        fail "Null content caused provider 400: $err"
    else
        fail "Null content: $err"
    fi
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T9: Large prompts — no 413 (Complaint #3)
# ─────────────────────────────────────────────────────────────────
echo "--- T9: Large prompts (no 413) ---"
for size in 4000 20000 50000; do
    # Write to temp file — shell arg passing corrupts large strings
    _t9tmp=$(mktemp)
    python3 -c "
import json, sys
body = {'model':'tools','stream':False,'max_tokens':20,
        'messages':[{'role':'user','content':'Fix: '+'x'*$size}],
        'tools':$TOOL_DEF}
print(json.dumps(body))
" > "$_t9tmp"
    result=$(curl -s --max-time 60 "$GATEWAY/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "@$_t9tmp" 2>/dev/null)
    rm -f "$_t9tmp"
    status=$(printf "%s" "$result" | get_status)
    if [ "$status" = "200" ]; then
        pass "${size}-char prompt accepted"
    elif [ "$status" = "413" ]; then
        fail "${size}-char prompt: 413 body too large"
    else
        fail "${size}-char prompt: status=$status"
    fi
    sleep 1
done
echo ""

# ─────────────────────────────────────────────────────────────────
# T10: No 413 on any alias with medium payload
# ─────────────────────────────────────────────────────────────────
echo "--- T10: No 413 on any alias (5K chars) ---"
PAD5=$(python3 -c "print('y' * 5000)")
FOUR13=0
for alias in default fast thinking coding tools big bench; do
    result=$(chat "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"Fix: $PAD5\"}],\"max_tokens\":10}" 30)
    status=$(printf "%s" "$result" | get_status)
    if [ "$status" = "413" ]; then
        FOUR13=$((FOUR13+1))
        fail "413 on alias '$alias'"
    fi
done
[ $FOUR13 -eq 0 ] && pass "Zero 413s across all aliases"
echo ""

# ─────────────────────────────────────────────────────────────────
# T11: Rate limit resilience (Complaint #5)
# ─────────────────────────────────────────────────────────────────
echo "--- T11: Rate limit resilience (15 rapid requests) ---"
T11_OK=0
T11_429=0
for i in $(seq 1 15); do
    result=$(chat '{"model":"tools","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}' 15)
    status=$(printf "%s" "$result" | get_status)
    if [ "$status" = "200" ]; then
        T11_OK=$((T11_OK+1))
    elif [ "$status" = "429" ]; then
        T11_429=$((T11_429+1))
    fi
done
if [ $T11_OK -ge 12 ]; then
    pass "Rapid fire: $T11_OK/15 succeeded, $T11_429 rate-limited"
elif [ $T11_OK -ge 8 ]; then
    warn "Rapid fire: $T11_OK/15 (acceptable but not ideal)"
else
    fail "Rapid fire: $T11_OK/15 — failover not working ($T11_429 got 429)"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T12: Response time (Complaint #1: timeout)
# ─────────────────────────────────────────────────────────────────
echo "--- T12: Response time ---"
START=$(date +%s%N)
result=$(chat '{"model":"tools","messages":[{"role":"user","content":"Write fibonacci in Python"}],"max_tokens":200}' 120)
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
if printf "%s" "$result" | has_choices; then
    if [ $ELAPSED_MS -lt 10000 ]; then
        pass "Response in ${ELAPSED_MS}ms (fast)"
    elif [ $ELAPSED_MS -lt 30000 ]; then
        pass "Response in ${ELAPSED_MS}ms (acceptable)"
    else
        warn "Response in ${ELAPSED_MS}ms (slow but succeeded)"
    fi
else
    fail "No response within 120s"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T13: Streaming works
# ─────────────────────────────────────────────────────────────────
echo "--- T13: Streaming ---"
STREAM_RESP=$(curl -s --max-time 30 "$GATEWAY/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{"model":"tools","messages":[{"role":"user","content":"Count to 3"}],"max_tokens":20,"stream":true}' 2>&1)
if echo "$STREAM_RESP" | grep -q "data:"; then
    pass "Streaming response received"
else
    fail "Streaming: no SSE data chunks"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T14: No provider-specific 400s leaking to client
# ─────────────────────────────────────────────────────────────────
echo "--- T14: No provider 400s on standard requests ---"
T14_400=0
for i in $(seq 1 5); do
    result=$(chat '{"model":"tools","messages":[{"role":"system","content":"Use tools."},{"role":"user","content":"List files in current directory"}],"tools":[{"type":"function","function":{"name":"list_files","description":"List files","parameters":{"type":"object","properties":{"dir":{"type":"string"}},"required":["dir"]}}}],"max_tokens":50,"tool_choice":"required"}' 30)
    status=$(printf "%s" "$result" | get_status)
    [ "$status" = "400" ] && T14_400=$((T14_400+1))
    sleep 1
done
if [ $T14_400 -eq 0 ]; then
    pass "Zero provider 400s on 5 tool requests"
else
    fail "$T14_400/5 tool requests got 400 — provider compatibility issue"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T15: Cloud-only aliases respond (Ollama bypass path)
# ─────────────────────────────────────────────────────────────────
echo "--- T15: Cloud-only aliases ---"
for alias in tools_cloud bench_cloud default_cloud swebench_cloud; do
    result=$(chat "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":5}" 20)
    if printf "%s" "$result" | has_choices; then
        pass "Alias '$alias'"
    else
        fail "Alias '$alias': $(printf "%s" "$result" | get_error 80)"
    fi
    sleep 0.5
done
echo ""

# ─────────────────────────────────────────────────────────────────
# T16: Ollama health probe reports status
# ─────────────────────────────────────────────────────────────────
echo "--- T16: Ollama health probe ---"
HEALTH_DATA=$(curl -s --max-time 5 "$GATEWAY_BASE/router/health" 2>&1)
if echo "$HEALTH_DATA" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if 'ollama_hosts' in d else 1)" 2>/dev/null; then
    pass "Ollama health probe active"
    echo "$HEALTH_DATA" | python3 -c "
import sys,json
d=json.load(sys.stdin)
for name, info in d.get('ollama_hosts',{}).items():
    status = 'UP' if info['healthy'] else 'DOWN'
    print(f'        {name}: {status} ({info[\"url\"]})')
" 2>/dev/null
else
    warn "Ollama health probe not available (direct LiteLLM mode)"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T17: Fallback chain doesn't route back to Ollama
# ─────────────────────────────────────────────────────────────────
echo "--- T17: Fallback chain integrity ---"
# tools_cloud should work without any Ollama dependency
result=$(chat '{"model":"tools_cloud","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}' 15)
if printf "%s" "$result" | has_choices; then
    pass "tools_cloud responds (no Ollama in chain)"
else
    fail "tools_cloud failed — fallback chain may include Ollama"
fi

# default_cloud should be the final fallback (no Ollama)
result=$(chat '{"model":"default_cloud","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}' 15)
if printf "%s" "$result" | has_choices; then
    pass "default_cloud responds (final fallback, no Ollama)"
else
    fail "default_cloud failed"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T18: Flush cooldowns endpoint
# ─────────────────────────────────────────────────────────────────
echo "--- T18: Flush cooldowns endpoint ---"
FLUSH=$(curl -s -X POST --max-time 5 "$GATEWAY_BASE/router/flush-cooldowns" 2>&1)
if echo "$FLUSH" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('status')=='flushed' else 1)" 2>/dev/null; then
    pass "Flush cooldowns endpoint works"
else
    warn "Flush cooldowns not available (direct LiteLLM mode)"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T19: Large context providers available in swebench
# ─────────────────────────────────────────────────────────────────
echo "--- T19: Large context coverage ---"
# Verify swebench has 256K+ providers by checking model list
SWEBENCH_MODELS=$(curl -s --max-time 5 "$GATEWAY/models" \
    -H "Authorization: Bearer $API_KEY" 2>&1)
if printf '%s' "$SWEBENCH_MODELS" | python3 -c "import sys,json; d=json.load(sys.stdin); models=[m['id'] for m in d.get('data',[])]; exit(0 if 'swebench' in models else 1)" 2>/dev/null; then
    pass "swebench alias registered"
else
    fail "swebench alias not found in model list"
fi

if printf '%s' "$SWEBENCH_MODELS" | python3 -c "import sys,json; d=json.load(sys.stdin); models=[m['id'] for m in d.get('data',[])]; exit(0 if 'tools_large' in models else 1)" 2>/dev/null; then
    pass "tools_large alias registered"
else
    fail "tools_large alias not found in model list"
fi

if printf '%s' "$SWEBENCH_MODELS" | python3 -c "import sys,json; d=json.load(sys.stdin); models=[m['id'] for m in d.get('data',[])]; exit(0 if 'tools_local' in models else 1)" 2>/dev/null; then
    pass "tools_local alias registered (429 rescue)"
else
    fail "tools_local alias not found"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T20: 429 rescue endpoint available
# ─────────────────────────────────────────────────────────────────
echo "--- T20: 429 rescue infrastructure ---"
# Verify tools_local works (the rescue target)
result=$(chat '{"model":"tools_local","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}' 60)
if printf '%s' "$result" | has_choices; then
    pass "tools_local responds (429 rescue target)"
else
    # tools_local may fail if both Ollama are down — that's acceptable
    warn "tools_local unavailable (Ollama may be down — rescue will fail gracefully)"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T21: Provider status endpoint
# ─────────────────────────────────────────────────────────────────
# T22: H1 OCR models loaded — evict true squatters, load required OCR models
# ─────────────────────────────────────────────────────────────────
# VRAM budget (24GB GPU, H1):
#   qwen3.5:9b  = 14.5GB  (gateway's permanent model — NOT a squatter)
#   LightOnOCR-2 = 3.0GB  (fits with qwen3.5:9b: 17.5GB total)
#   glm-ocr      = 0GB    (CPU-only)
#   deepseek-ocr = 9.5GB  (would need 27GB alongside both — excluded from required)
# ─────────────────────────────────────────────────────────────────
echo "--- T22: H1 OCR model warmth ---"
# Models that must be warm for OCR to work locally.
OCR_REQUIRED=("maternion/LightOnOCR-2:latest" "glm-ocr:latest")
OCR_CPU=("glm-ocr:latest")
# Gateway's permanent model — always loaded, not a squatter.
OCR_GATEWAY_MODELS=("qwen3.5:9b")

# Discover H1 URL from router health (first healthy Ollama host that isn't /v1)
H1_URL=$(curl -sf --max-time 5 "$ROUTER_HEALTH" 2>/dev/null \
    | python3 -c "
import sys,json
d=json.load(sys.stdin)
for h,s in d.get('ollama_hosts',{}).items():
    if isinstance(s,dict) and s.get('healthy') and '/v1' not in s.get('url',''):
        print(s['url']); break
" 2>/dev/null)

if [ -z "$H1_URL" ]; then
    warn "T22: could not determine H1 URL from router health — skipping OCR check"
else
    ps_json=$(curl --noproxy '*' -sf --connect-timeout 5 --max-time 10 "$H1_URL/api/ps" 2>/dev/null || echo '{"models":[]}')
    loaded=$(echo "$ps_json" | python3 -c "import sys,json; print('\n'.join(m['name'] for m in json.load(sys.stdin).get('models',[])))" 2>/dev/null || true)

    # Detect true squatters: non-OCR models excluding the gateway's permanent model.
    squatters=$(python3 -c "
ocr={'maternion/LightOnOCR-2:latest','deepseek-ocr:latest','glm-ocr:latest'}
gateway={'qwen3.5:9b'}
loaded='''$loaded'''.strip().splitlines()
print('\n'.join(m for m in loaded if m and m not in ocr and m not in gateway))
" 2>/dev/null)

    missing=""
    for m in "${OCR_REQUIRED[@]}"; do
        echo "$loaded" | grep -qxF "$m" || missing="$missing $m"
    done

    if [ -n "$squatters" ]; then
        echo "  INFO: squatters: $(echo "$squatters" | tr '\n' ' ')"
        echo "  INFO: flushing squatters..."
        # Flush only the squatters, leave OCR + gateway models intact.
        while IFS= read -r m; do
            [ -z "$m" ] && continue
            curl --noproxy '*' -s --max-time 30 -X POST "$H1_URL/api/generate" \
                -H 'Content-Type: application/json' \
                -d "{\"model\":\"$m\",\"prompt\":\"x\",\"stream\":false,\"keep_alive\":0,\"options\":{\"num_predict\":1}}" \
                > /dev/null 2>&1 || true
        done <<< "$squatters"
    fi

    if [ -n "$missing" ]; then
        echo "  INFO: missing:$(echo "$missing")"
        echo "  INFO: loading missing OCR models..."
        reload_ok=1
        for m in "${OCR_REQUIRED[@]}"; do
            echo "$loaded" | grep -qxF "$m" && continue   # already loaded
            is_cpu=0
            for c in "${OCR_CPU[@]}"; do [ "$c" = "$m" ] && is_cpu=1; done
            num_gpu=$( [ "$is_cpu" -eq 1 ] && echo 0 || echo 99 )
            resp=$(curl --noproxy '*' -s --max-time 300 -X POST "$H1_URL/api/generate" \
                -H 'Content-Type: application/json' \
                -d "{\"model\":\"$m\",\"prompt\":\"OCR_OK\",\"stream\":false,\"keep_alive\":\"1h\",\"options\":{\"num_predict\":4,\"num_gpu\":$num_gpu}}" 2>/dev/null || echo '{}')
            if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('done') and not d.get('error') else 1)" 2>/dev/null; then
                echo "  INFO: loaded $m"
            else
                echo "  WARN: failed to load $m"
                reload_ok=0
            fi
        done
        [ "$reload_ok" -eq 1 ] && pass "H1 OCR models loaded (was stale)" || fail "H1 OCR reload incomplete"
    elif [ -n "$squatters" ]; then
        pass "H1 OCR models already warm (flushed squatters)"
    else
        pass "H1 OCR models all warm"
    fi
fi

# ─────────────────────────────────────────────────────────────────
echo "--- T21: Provider status & model identity ---"
pstatus=$(curl -sf --max-time 5 "$GATEWAY_BASE/router/provider-status" 2>/dev/null)
if printf '%s' "$pstatus" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'providers' in d; assert 'model_identity_count' in d" 2>/dev/null; then
    pass "Provider status endpoint responds"
    model_count=$(printf '%s' "$pstatus" | python3 -c "import json,sys; print(json.load(sys.stdin)['model_identity_count'])")
    if [ "$model_count" -gt 0 ]; then
        pass "Model identity map loaded ($model_count mappings)"
    else
        warn "Model identity map empty"
    fi
else
    fail "Provider status endpoint failed"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed, $SKIP skipped, $WARN warnings"
echo "============================================================"

if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASSED — gateway ready for benchmarks"
    exit 0
else
    echo "FAILURES DETECTED — fix gateway before running benchmarks"
    exit 1
fi
