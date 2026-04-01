#!/usr/bin/env bash
# Quick gateway status check
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .env 2>/dev/null || true

echo "=== Gateway Status ==="

# Processes
echo -n "  LiteLLM:  "
if curl -s --max-time 3 http://localhost:4002/health/readiness | grep -q "healthy" 2>/dev/null; then
    echo "UP (port 4002)"
else
    echo "DOWN"
fi

echo -n "  Router:   "
HEALTH=$(curl -s --max-time 3 http://localhost:4000/router/health 2>/dev/null || echo "")
if echo "$HEALTH" | grep -q "healthy" 2>/dev/null; then
    echo "UP (port 4000)"
else
    echo "DOWN"
fi

echo -n "  Redis:    "
if docker ps | grep -q llm-cache 2>/dev/null; then
    echo "UP (port 6379)"
else
    echo "DOWN"
fi

echo -n "  Watchdog: "
if [[ -f .watchdog.pid ]] && kill -0 "$(cat .watchdog.pid)" 2>/dev/null; then
    echo "RUNNING (PID $(cat .watchdog.pid))"
else
    echo "NOT RUNNING"
fi

# Ollama
if [[ -n "$HEALTH" ]]; then
    echo ""
    echo "=== Ollama Hosts ==="
    echo "$HEALTH" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    for name, info in d.get('ollama_hosts',{}).items():
        status = 'UP' if info['healthy'] else 'DOWN'
        print(f'  {name}: {status} ({info[\"url\"]})')
except: pass
" 2>/dev/null
fi

# Provider health (rolling 5-min window)
PSTATUS=$(curl -s --max-time 3 http://localhost:4000/router/provider-status 2>/dev/null || echo "")
if [[ -n "$PSTATUS" ]]; then
    echo ""
    echo "=== Provider Health (5m) ==="
    echo "$PSTATUS" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    for name, info in d.get('providers',{}).items():
        status = 'BROKEN' if info['circuit_broken'] else 'OK'
        print(f'  {name:12s} {status:6s}  reqs={info[\"requests_5m\"]:3d}  err={info[\"error_rate\"]*100:.0f}%  latency={info[\"avg_latency_ms\"]}ms')
    broken = d.get('circuit_broken', [])
    if broken: print(f'  ⚠ Circuit-broken: {broken}')
    print(f'  Model identity mappings: {d.get(\"model_identity_count\", 0)}')
except: pass
" 2>/dev/null
fi

# Today's traffic
echo ""
echo "=== Today's Traffic ==="
if [[ -f logs/router/router.log ]]; then
    TODAY=$(date +%Y-%m-%d)
    grep "$TODAY" logs/router/router.log | grep "status=" | python3 -c "
import sys, re, collections
total=0; statuses=collections.Counter()
for line in sys.stdin:
    m=re.search(r'status=(\d+)',line)
    if m: total+=1; statuses[int(m.group(1))]+=1
if total:
    s200=statuses.get(200,0)
    errs={k:v for k,v in sorted(statuses.items()) if k!=200}
    print(f'  Requests: {total} | Success: {s200}/{total} ({s200/total*100:.1f}%)')
    if errs: print(f'  Errors: {errs}')
else:
    print('  No traffic today')
" 2>/dev/null
fi

# Training data
echo ""
echo "=== Training Data ==="
if [[ -f logs/training/routing.jsonl ]]; then
    echo "  Router samples: $(wc -l < logs/training/routing.jsonl)"
fi
if [[ -f logs/training/conversations/convos.jsonl ]]; then
    echo "  Conversations:  $(wc -l < logs/training/conversations/convos.jsonl)"
fi
