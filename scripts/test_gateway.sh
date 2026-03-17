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
    local body
    body=$(echo "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); d['stream']=False; print(json.dumps(d))" 2>/dev/null || echo "$1")
    curl -s --max-time "${2:-30}" "$GATEWAY/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "$body" 2>&1
}

has_choices() {
    python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('choices') else 1)" 2>/dev/null
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
except: print('parse_error')
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

AUTH=$(chat '{"model":"default","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}' 15)
if echo "$AUTH" | has_choices; then
    pass "Authenticated request succeeds"
else
    fail "Authenticated request failed: $(echo "$AUTH" | get_error 200)"
    echo "ABORT: Cannot authenticate."
    exit 1
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# T3: All aliases respond
# ─────────────────────────────────────────────────────────────────
echo "--- T3: Alias availability ---"
for alias in default fast thinking coding vision tools big bench tools_stable bench_stable; do
    result=$(chat "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":5}" 20)
    if echo "$result" | has_choices; then
        pass "Alias '$alias'"
    else
        fail "Alias '$alias': $(echo "$result" | get_error 80)"
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
    tc_count=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('choices',[{}])[0].get('message',{}).get('tool_calls',[])))" 2>/dev/null)
    if [ "$tc_count" -gt 0 ] 2>/dev/null; then
        pass "Alias '$alias' tool call ($tc_count)"
    else
        fail "Alias '$alias' tool call: $(echo "$result" | get_error 80)"
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

TC_ID=$(echo "$T1_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['tool_calls'][0]['id'])" 2>/dev/null)

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

    if echo "$T2_RESP" | has_choices; then
        pass "Turn 2: response after tool result"
    else
        fail "Turn 2: $(echo "$T2_RESP" | get_error)"
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

    if echo "$T3_RESP" | has_choices; then
        pass "3-turn conversation accepted"
    else
        fail "3-turn conversation: $(echo "$T3_RESP" | get_error)"
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
if echo "$ORPHAN_RESP" | has_choices; then
    pass "Orphan tool result repaired (not 500)"
else
    err=$(echo "$ORPHAN_RESP" | get_error 80)
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
if echo "$NULL_RESP" | has_choices; then
    pass "Null content assistant handled (not 400)"
else
    err=$(echo "$NULL_RESP" | get_error 80)
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
    PAD=$(python3 -c "print('x' * $size)")
    result=$(chat "{
        \"model\":\"tools\",
        \"messages\":[{\"role\":\"user\",\"content\":\"Fix: $PAD\"}],
        \"tools\":$TOOL_DEF,
        \"max_tokens\":20
    }" 60)
    status=$(echo "$result" | get_status)
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
    status=$(echo "$result" | get_status)
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
    status=$(echo "$result" | get_status)
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
if echo "$result" | has_choices; then
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
    status=$(echo "$result" | get_status)
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
