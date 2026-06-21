#!/usr/bin/env python3
"""
router_feedback.py — close the loop: measure how the LEARNED router performed.

Reads the v2 telemetry that smart_router now logs (schema_version="v2", with
learned_alias / learned_reason / final_alias / final_status) and reports how the
data-driven decisions actually fared in production:

  - learned-mode request volume and decision-reason mix
    (learned_override / heuristic_kept / agree / +explore)
  - realized client-visible success rate per decision reason
  - realized success + latency per chosen alias
  - override scorecard: did learned_override decisions succeed more than the
    heuristic alias they replaced (compared against that alias's overall rate)?
  - a recommendation for the next iteration

Run it routinely (e.g. daily) before rebuilding the policy table — it is the
"routinely check the data, clean it, iterate" step of the loop.

Usage:
    python3 scripts/router_feedback.py            # current log
    python3 scripts/router_feedback.py --all      # all rotated logs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router_data as rd  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))])


def reason_bucket(reason: str) -> str:
    if not reason:
        return "none"
    r = reason
    if "learned:" in r:
        r = r.split("learned:", 1)[1]
    for tag in ("learned_override", "heuristic_kept", "agree", "learned_low_support", "learned"):
        if tag in r:
            return tag + ("+explore" if "explore" in r else "")
    return r[:24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    paths = (sorted(glob.glob(rd.DEFAULT_LOG_GLOB)) if args.all
             else [os.path.join(REPO, "logs/training/routing.jsonl")])

    v2 = 0
    by_reason = defaultdict(lambda: {"n": 0, "ok": 0})
    by_alias = defaultdict(lambda: {"n": 0, "ok": 0, "lat": []})
    alias_overall = defaultdict(lambda: {"n": 0, "ok": 0})   # for override scorecard
    override_rows = []
    explore_n = 0
    cost_bias = Counter()

    for o in rd.iter_raw_rows(paths):
        # overall per-final-alias success (all rows, for the override baseline)
        fa = o.get("final_alias") or o.get("routed_alias")
        st = o.get("final_status", o.get("status"))
        if fa and isinstance(st, int) and not o.get("cache_hit"):
            alias_overall[fa]["n"] += 1
            alias_overall[fa]["ok"] += int(st == 200)

        if o.get("schema_version") != "v2":
            continue
        if o.get("policy_mode") not in ("learned", "shadow"):
            continue
        if o.get("cache_hit"):
            continue
        v2 += 1
        reason = o.get("route_reason") or o.get("learned_reason") or ""
        rb = reason_bucket(reason)
        if "explore" in rb:
            explore_n += 1
        ok = int(st == 200) if isinstance(st, int) else 0
        by_reason[rb]["n"] += 1
        by_reason[rb]["ok"] += ok
        alias = o.get("learned_alias") or o.get("routed_alias")
        if alias:
            by_alias[alias]["n"] += 1
            by_alias[alias]["ok"] += ok
            lat = o.get("total_latency_ms") or o.get("latency_ms")
            if ok and isinstance(lat, (int, float)) and lat > 0:
                by_alias[alias]["lat"].append(lat)
        if o.get("cost_bias") is not None:
            cost_bias[o.get("cost_bias")] += 1
        # override scorecard
        if "learned_override" in rb:
            sugg = o.get("suggested_alias")
            override_rows.append((o.get("learned_alias"), sugg, ok))

    print(f"=== LEARNED-ROUTER FEEDBACK ({len(paths)} file(s)) ===")
    print(f"v2 learned/shadow routed requests: {v2}")
    if not v2:
        print("\nNo v2 learned-mode rows yet. Once the gateway serves auto/default "
              "traffic in learned mode, rerun to see realized performance.")
        return
    print(f"exploration (+explore) decisions : {explore_n} ({100*explore_n/v2:.1f}%)")
    print(f"cost_bias seen                   : {dict(cost_bias)}")

    print(f"\n{'decision reason':26s} {'n':>6s} {'succ%':>6s}")
    for rb, d in sorted(by_reason.items(), key=lambda x: -x[1]["n"]):
        print(f"{rb:26s} {d['n']:6d} {100*d['ok']/max(d['n'],1):6.1f}")

    print(f"\n{'chosen alias':16s} {'n':>6s} {'succ%':>6s} {'p50ms':>7s} {'p95ms':>7s}")
    for a, d in sorted(by_alias.items(), key=lambda x: -x[1]["n"]):
        print(f"{a:16s} {d['n']:6d} {100*d['ok']/max(d['n'],1):6.1f} "
              f"{str(pct(d['lat'],50)):>7s} {str(pct(d['lat'],95)):>7s}")

    # override scorecard
    if override_rows:
        ov_n = len(override_rows)
        ov_ok = sum(r[2] for r in override_rows)
        # baseline: success the replaced heuristic alias gets overall
        base_hits = base_tot = 0
        for _, sugg, _ in override_rows:
            b = alias_overall.get(sugg)
            if b and b["n"]:
                base_hits += b["ok"]
                base_tot += b["n"]
        print(f"\n=== OVERRIDE SCORECARD ===")
        print(f"learned_override decisions: {ov_n}, realized success {100*ov_ok/ov_n:.1f}%")
        if base_tot:
            print(f"replaced-heuristic aliases' overall success (baseline): "
                  f"{100*base_hits/base_tot:.1f}%")

    # recommendation
    print(f"\n=== NEXT-ITERATION RECOMMENDATION ===")
    worst = sorted(((d["ok"]/max(d["n"],1), a, d["n"]) for a, d in by_alias.items()
                    if d["n"] >= 20), key=lambda x: x[0])[:3]
    if worst:
        print("Lowest-success chosen aliases (>=20 reqs):")
        for sr, a, n in worst:
            print(f"  {a}: {100*sr:.1f}% over {n} reqs"
                  + ("  <- consider demoting / capability-gating" if sr < 0.85 else ""))
    print("If volume is sufficient and success is stable, rebuild the table:")
    print("  python3 scripts/build_router_table.py && curl -XPOST localhost:4000/router/reload-policy")


if __name__ == "__main__":
    main()
