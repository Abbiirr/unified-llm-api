#!/usr/bin/env python3
"""
build_router_table.py — turn cleaned routing logs into the v1 auto-router policy.

The policy is an empirical, hierarchically-smoothed utility table:

    table[bucket_key][alias] = [n, ok, rate_limited, lat_n, loglat_sum]

At serve time (router_policy.py) we score each capability-eligible candidate
alias by Empirical-Bayes shrinking its success rate / latency from the most
specific request-class cell toward more general cells, then combine into a
utility. This is the multi-alias generalization of RouteLLM's
`calculate_strong_win_rate` (https://www.lmsys.org/blog/2024-07-01-routellm/),
using observed success/latency labels instead of pairwise preference, and a
cost/quality dial in the spirit of OpenRouter's Auto Router
(https://openrouter.ai/docs/guides/routing/routers/auto-router).

No native ML deps — pure stdlib + the shared router_features module — so it is
safe to load inside the live gateway (<0.1 ms dict lookups, no GPU).

Usage:
    python3 scripts/build_router_table.py            # build from recent logs
    python3 scripts/build_router_table.py --all      # use every rotated log
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import router_features as rf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router_data as rd  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(REPO, "models")
OUT_MODEL = os.path.join(MODELS_DIR, "router_v1.json")
OUT_METRICS = os.path.join(MODELS_DIR, "router_v1.metrics.json")

# Default utility weights — also embedded in the model so the runtime reads them.
WEIGHTS = {
    "w_success": 1.00,       # reward predicted client-visible success
    "w_loglat": 0.22,        # penalize latency (scaled by the cost dial at serve time)
    "w_rate_limit": 0.30,    # penalize 429-prone routes
    "prior_strength": 25.0,  # Empirical-Bayes shrinkage strength
    "cost_bias_default": 5,  # 0=most capable .. 10=cheapest/fastest (OpenRouter-style)
    "min_confidence": 100,   # below this much specific support -> defer to heuristic
    "override_margin": 0.06, # learned winner must beat heuristic by this to override
}
LOGLAT_LO = math.log1p(200)     # ~5.30  (fast floor)
LOGLAT_HI = math.log1p(90000)   # ~11.4  (slow ceiling)


def empty_cell():
    return [0, 0, 0, 0, 0.0]  # n, ok, rate_limited, lat_n, loglat_sum


def build_table(rows):
    table = defaultdict(lambda: defaultdict(empty_cell))
    for r in rows:
        alias = r["alias"]
        for key in r["bucket_keys"]:
            c = table[key][alias]
            c[0] += 1
            if r["success"]:
                c[1] += 1
            if r["rate_limited"]:
                c[2] += 1
            lat = r["latency_ms"]
            if r["success"] and lat:
                c[3] += 1
                c[4] += math.log1p(lat)
    # plain dict-of-dicts for JSON
    return {k: {a: v for a, v in al.items()} for k, al in table.items()}


def shrink_success(table, keys, alias, prior_strength, global_p):
    """Empirical-Bayes: shrink specific cell toward general cells -> global_p."""
    p = global_p
    support = 0
    # general -> specific so each level refines the previous
    for key in reversed(keys):
        cell = table.get(key, {}).get(alias)
        if not cell:
            continue
        n, ok = cell[0], cell[1]
        p = (ok + prior_strength * p) / (n + prior_strength)
        support = n  # support from the most specific non-empty level
    return p, support


def shrink_metric(table, keys, alias, prior_strength, global_v, idx_num, idx_den):
    v = global_v
    for key in reversed(keys):
        cell = table.get(key, {}).get(alias)
        if not cell:
            continue
        den = cell[idx_den]
        num = cell[idx_num]
        if den <= 0:
            continue
        rate = num / den
        v = (num + prior_strength * global_v) / (den + prior_strength)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="use all rotated logs (default: current + .1)")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    args = ap.parse_args()

    if args.all:
        paths = None  # router_data default glob = everything
    else:
        paths = [
            os.path.join(REPO, "logs/training/routing.jsonl.1"),
            os.path.join(REPO, "logs/training/routing.jsonl"),
        ]
        paths = [p for p in paths if os.path.exists(p)]

    print(f"loading clean rows from {paths or 'ALL rotated logs'} ...")
    rows = rd.load_clean(paths)
    rows = [r for r in rows if r["timestamp"]]
    rows.sort(key=lambda r: r["timestamp"])
    print(f"  {len(rows)} clean labeled rows  ({rows[0]['timestamp']} .. {rows[-1]['timestamp']})")

    # Chronological split for honest evaluation (provider behavior drifts).
    cut = int(len(rows) * (1 - args.holdout_frac))
    train, test = rows[:cut], rows[cut:]
    print(f"  train={len(train)}  test={len(test)} (chronological)")

    table = build_table(train)
    global_p = sum(r["success"] for r in train) / max(len(train), 1)
    # global geometric latency
    lat_rows = [r for r in train if r["success"] and r["latency_ms"]]
    global_loglat = (sum(math.log1p(r["latency_ms"]) for r in lat_rows) / len(lat_rows)) if lat_rows else LOGLAT_LO

    model = {
        "schema_version": rf.FEATURE_SCHEMA_VERSION,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_files": [os.path.relpath(p, REPO) for p in (paths or [])] or ["ALL"],
        "n_train_rows": len(train),
        "train_window": [train[0]["timestamp"], train[-1]["timestamp"]],
        "auto_pool": {
            "text": rd.AUTO_POOL_TEXT,
            "tools": rd.AUTO_POOL_TOOLS,
            "image": rd.AUTO_POOL_IMAGE,
        },
        "weights": WEIGHTS,
        "loglat_lo": LOGLAT_LO,
        "loglat_hi": LOGLAT_HI,
        "global_success": round(global_p, 4),
        "global_loglat": round(global_loglat, 4),
        "table": table,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(OUT_MODEL, "w") as f:
        json.dump(model, f)
    print(f"wrote {OUT_MODEL}  ({os.path.getsize(OUT_MODEL)//1024} KB, {len(table)} buckets)")

    # ── Sanity report: best alias per common class (within capability pool) ──
    from collections import Counter
    class_counts = Counter(r["class_key"] for r in train)
    report_lines = []
    print("\nBest data-driven alias per common request class (cost_bias=5):")
    for ck, cnt in class_counts.most_common(14):
        mod = ck.split("|", 1)[0]
        pool = rd.candidate_pool(mod)
        keys_for_class = [ck, "|".join(ck.split("|")[:2]), mod, "global"]
        scored = []
        for a in pool:
            p, support = shrink_success(table, keys_for_class, a, WEIGHTS["prior_strength"], global_p)
            loglat = shrink_metric(table, keys_for_class, a, WEIGHTS["prior_strength"], global_loglat, 4, 3)
            rl = shrink_metric(table, keys_for_class, a, WEIGHTS["prior_strength"], 1 - global_p, 2, 0)
            norm_lat = max(0.0, min(1.0, (loglat - LOGLAT_LO) / (LOGLAT_HI - LOGLAT_LO)))
            util = (WEIGHTS["w_success"] * p
                    - WEIGHTS["w_loglat"] * norm_lat
                    - WEIGHTS["w_rate_limit"] * rl)
            scored.append((util, a, p, support, loglat))
        scored.sort(reverse=True)
        best = scored[0]
        line = (f"  {ck:20s} n={cnt:6d}  -> {best[1]:12s} "
                f"util={best[0]:.3f} p_succ={best[2]:.2f} support={best[3]}")
        print(line)
        report_lines.append({"class": ck, "n": cnt, "best_alias": best[1],
                             "util": round(best[0], 4), "p_success": round(best[2], 3)})

    metrics = {
        "built_at": model["built_at"],
        "n_train": len(train),
        "n_test": len(test),
        "global_success": round(global_p, 4),
        "per_class_best": report_lines,
    }
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {OUT_METRICS}")


if __name__ == "__main__":
    main()
