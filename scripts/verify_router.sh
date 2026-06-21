#!/usr/bin/env bash
# =============================================================================
# verify_router.sh — one-shot health + performance check for the auto-router.
# Bundles: (1) liveness/mode, (2) free log-based feedback, (3) a small live
# probe, (4) a trend line in logs/training/reports/verifier_history.tsv, and
# (5) regression flags vs the previous run. Designed to be called on a loop.
#
# Usage:
#   bash scripts/verify_router.sh              # health + feedback + 20-req probe
#   VR_OFFLINE=1 bash scripts/verify_router.sh # health + feedback only (no quota)
#   VR_N=30 VR_DELAY=2.5 bash scripts/verify_router.sh
#
# Env knobs: VR_N(20) VR_DELAY(2.0) VR_TIMEOUT(12) VR_RETRIES(2) VR_OFFLINE(0)
# Exit code: 0 = OK, 2 = regression flagged, 3 = router down. Never mutates git.
# =============================================================================
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VR_N="${VR_N:-20}"
VR_DELAY="${VR_DELAY:-2.0}"
VR_TIMEOUT="${VR_TIMEOUT:-12}"
VR_RETRIES="${VR_RETRIES:-2}"
VR_OFFLINE="${VR_OFFLINE:-0}"
GW="${GATEWAY_URL:-http://localhost:4000}"
HIST="$ROOT_DIR/logs/training/reports/verifier_history.tsv"
mkdir -p "$(dirname "$HIST")"
[ -f "$HIST" ] || printf 'ts\tn\tsuccess\tsucc_pct\tfirst_try_pct\tclient_429s\tp50_ms\tp90_ms\ttop_alias\tn_providers\thealth\n' > "$HIST"

TS="$(date '+%Y-%m-%dT%H:%M:%S')"
regression=0

echo "==================== verify_router @ $TS ===================="

# ── 1) HEALTH ───────────────────────────────────────────────────────────────
RELOAD="$(curl -s --max-time 6 -X POST "$GW/router/reload-policy" 2>/dev/null || echo '{}')"
MODE="$(printf '%s' "$RELOAD" | python3 -c 'import json,sys;
try: print(json.load(sys.stdin).get("mode","?"))
except: print("DOWN")' 2>/dev/null)"
if [ "$MODE" = "DOWN" ] || [ -z "$MODE" ]; then
  echo "HEALTH: ❌ router not responding on $GW — restart it (kill PID on :4000, then nohup uv run python smart_router.py; never pkill -f from inline shell)."
  printf '%s\t-\t-\t-\t-\t-\t-\t-\t-\t-\tDOWN\n' "$TS" >> "$HIST"
  exit 3
fi
echo "HEALTH: ✅ mode=$MODE  $(printf '%s' "$RELOAD" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("rows="+str(d.get("n_train_rows"))+" buckets="+str(d.get("buckets")))' 2>/dev/null)"

# ── 2) REALIZED PERFORMANCE (free — reads v2 logs) ──────────────────────────
echo; echo "---- realized performance (router_feedback.py) ----"
python3 scripts/router_feedback.py 2>/dev/null | sed -n '1,18p'

# ── 3) LIVE PROBE (quota-aware) ─────────────────────────────────────────────
if [ "$VR_OFFLINE" = "1" ]; then
  echo; echo "PROBE: skipped (VR_OFFLINE=1) — appending health-only trend line"
  printf '%s\t0\t0\t-\t-\t-\t-\t-\toffline\t-\t%s\n' "$TS" "$MODE" >> "$HIST"
  echo "history -> $HIST"
  exit 0
fi

echo; echo "---- live probe: $VR_N requests (delay=$VR_DELAY timeout=$VR_TIMEOUT retries=$VR_RETRIES) ----"
GATE=$(( VR_N * 80 / 100 ))
PROBE_OUT="$(python3 -u scripts/test_auto_router.py --n "$VR_N" --delay "$VR_DELAY" \
    --timeout "$VR_TIMEOUT" --retries "$VR_RETRIES" --require "$GATE" 2>&1)"
printf '%s\n' "$PROBE_OUT" | sed -n '/^=====/,$p'
REPORT="$(printf '%s\n' "$PROBE_OUT" | sed -n 's/^wrote //p' | tail -1)"

# ── 4) TREND LINE + 5) REGRESSION FLAGS ─────────────────────────────────────
if [ -n "$REPORT" ] && [ -f "$REPORT" ]; then
  PREV_P90="$(tail -1 "$HIST" | awk -F'\t' '{print $8}')"
  # One python call: writes the TSV line to the history file and prints a
  # human summary + "FLAGS: ..." to stdout (last line = "REGRESSION" or "OK").
  STATUS="$(python3 - "$REPORT" "$TS" "$PREV_P90" "$HIST" <<'PY'
import json, sys
rep, ts, prev_p90, hist = sys.argv[1:5]
d = json.load(open(rep))
n, ok = d.get("n", 0), d.get("success", 0)
succ = round(100*ok/max(n, 1), 1)
ftp = round(100*d.get("first_try_ok", 0)/max(n, 1), 1)
c429 = d.get("client_429s", 0)
p50 = d.get("latency_p50_ms") or 0
p90 = d.get("latency_p90_ms") or 0
nprov = len([k for k in d.get("providers", {}) if k and k != "?"])
ra = d.get("routed_alias", {})
top = max(ra, key=ra.get) if ra else "?"
with open(hist, "a") as f:
    f.write("\t".join(str(x) for x in [ts, n, ok, succ, ftp, c429, p50, p90, top, nprov, "OK"]) + "\n")
flags = []
if succ < 90: flags.append(f"success {succ}%<90%")
if c429 > 0: flags.append(f"client_429s={c429}")
try:
    pv = float(prev_p90)
    if pv > 0 and p90 > 2*pv: flags.append(f"p90 {p90}ms >2x prev {int(pv)}ms")
except (TypeError, ValueError):
    pass
print(f"TREND: success={succ}% first_try={ftp}% client_429s={c429} "
      f"p50={p50}ms p90={p90}ms top_alias={top} providers={nprov}")
print("FLAGS: " + ("; ".join(flags) if flags else "none"))
print("REGRESSION" if flags else "OK")
PY
)"
  echo; printf '%s\n' "$STATUS" | sed '$d'   # everything except the last marker line
  [ "$(printf '%s\n' "$STATUS" | tail -1)" = "REGRESSION" ] && { echo "⚠️  REGRESSION — diagnose providers (litellm :4002) and Ollama rescue health"; regression=1; } || echo "✅ within thresholds"
  echo "history -> $HIST"
else
  echo "PROBE: ⚠️ no JSON report produced; inspect probe output above"
fi

exit "$([ "$regression" = 1 ] && echo 2 || echo 0)"
