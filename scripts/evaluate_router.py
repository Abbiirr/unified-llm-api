#!/usr/bin/env python3
"""
evaluate_router.py — offline, counterfactual-honest evaluation of the policy.

We can only ever observe the outcome of the alias that was actually chosen
(bandit feedback, https://arxiv.org/abs/2510.07429), so a per-request "what if"
is impossible. Instead we evaluate on a CHRONOLOGICAL holdout using real
held-out outcomes matched by request-class:

  1. Build the policy table on the train split only.
  2. From the TEST split, tabulate real outcomes per (class, alias).
  3. For each class, compare the realized success rate of the alias the LEARNED
     policy picks vs the alias the HEURISTIC picks, using only test rows where
     both have observed outcomes (a direct/replay estimator).
  4. Aggregate weighted by class frequency -> estimated success lift.
  5. Report agreement, coverage, and router inference p99 latency (<=2 ms gate).

This is directional, not a guarantee — it is bounded to classes/aliases that
appear in held-out traffic. Online A/B via the feedback log is the final word.

Usage:
    python3 scripts/evaluate_router.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import router_features as rf  # noqa: E402
import router_policy as rp  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router_data as rd  # noqa: E402
import build_router_table as brt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIN_CELL = 15  # need this many held-out outcomes for a (class, alias) estimate


def feat_for_class(ck: str) -> dict:
    mod, size, content = (ck.split("|") + ["text", "s", "plain"])[:3]
    tok = {"xs": 50, "s": 400, "m": 1500, "l": 6000, "xl": 30000}.get(size, 400)
    feat = {"estimated_total_tokens": tok}
    if mod == "image":
        feat["has_images"] = True
    elif mod == "tools":
        feat["has_tools"] = True
        feat["tool_count"] = 2
    if content == "code":
        feat["has_code_blocks"] = True
        feat["primary_language"] = "python"
    elif content == "error":
        feat["has_error_trace"] = True
    elif content == "json":
        feat["has_json"] = True
    return feat


def main():
    paths = [p for p in [os.path.join(REPO, "logs/training/routing.jsonl.1"),
                         os.path.join(REPO, "logs/training/routing.jsonl")]
             if os.path.exists(p)]
    rows = [r for r in rd.load_clean(paths) if r["timestamp"]]
    rows.sort(key=lambda r: r["timestamp"])
    cut = int(len(rows) * 0.8)
    train, test = rows[:cut], rows[cut:]
    print(f"train={len(train)} test={len(test)} (chronological holdout)")

    # build policy from train only
    table = brt.build_table(train)
    global_p = sum(r["success"] for r in train) / max(len(train), 1)
    import math
    lat_rows = [r for r in train if r["success"] and r["latency_ms"]]
    global_loglat = (sum(math.log1p(r["latency_ms"]) for r in lat_rows) / len(lat_rows)) if lat_rows else brt.LOGLAT_LO
    model = {
        "schema_version": "v2", "table": table, "weights": brt.WEIGHTS,
        "auto_pool": {"text": rd.AUTO_POOL_TEXT, "tools": rd.AUTO_POOL_TOOLS, "image": rd.AUTO_POOL_IMAGE},
        "loglat_lo": brt.LOGLAT_LO, "loglat_hi": brt.LOGLAT_HI,
        "global_success": global_p, "global_loglat": global_loglat,
    }
    policy = rp.RouterPolicy(model)

    # real held-out outcomes per (class, alias)
    test_cell = defaultdict(lambda: [0, 0])  # (class, alias) -> [n, ok]
    class_freq = Counter()
    heur_for_class = defaultdict(Counter)
    for r in test:
        test_cell[(r["class_key"], r["alias"])][0] += 1
        test_cell[(r["class_key"], r["alias"])][1] += int(r["success"])
        class_freq[r["class_key"]] += 1
        if r["suggested_alias"]:
            heur_for_class[r["class_key"]][r["suggested_alias"]] += 1

    # compare learned vs heuristic per class (deterministic: explore_temp=0)
    agree = differ = 0
    weighted_h = weighted_l = weight_cov = 0.0
    rows_report = []
    for ck, freq in class_freq.most_common():
        feat = feat_for_class(ck)
        L = policy.decide(feat, explore_temp=0.0).alias
        H = heur_for_class[ck].most_common(1)[0][0] if heur_for_class[ck] else None
        if L == H:
            agree += freq
        else:
            differ += freq
        lc = test_cell.get((ck, L))
        hc = test_cell.get((ck, H))
        if lc and hc and lc[0] >= MIN_CELL and hc[0] >= MIN_CELL:
            l_sr, h_sr = lc[1] / lc[0], hc[1] / hc[0]
            weighted_l += freq * l_sr
            weighted_h += freq * h_sr
            weight_cov += freq
            rows_report.append((ck, freq, H, round(h_sr, 3), L, round(l_sr, 3),
                                round(l_sr - h_sr, 3), lc[0], hc[0]))

    print(f"\nlearned vs heuristic agreement: agree={agree} differ={differ} "
          f"({100*agree/max(agree+differ,1):.1f}% agree)")
    print(f"coverage for outcome comparison: {weight_cov:.0f}/{len(test)} test rows "
          f"({100*weight_cov/max(len(test),1):.1f}%)")
    if weight_cov:
        print(f"\nEstimated success rate on covered classes:")
        print(f"  heuristic : {weighted_h/weight_cov:.4f}")
        print(f"  learned   : {weighted_l/weight_cov:.4f}")
        print(f"  lift      : {(weighted_l-weighted_h)/weight_cov:+.4f}")

    print(f"\n{'class':18s} {'freq':>6s} {'heur':>10s} {'h_sr':>5s} {'learn':>10s} {'l_sr':>5s} {'dlt':>6s}")
    for ck, freq, H, hsr, L, lsr, dlt, ln, hn in rows_report[:20]:
        print(f"{ck:18s} {freq:6d} {str(H):>10s} {hsr:5.2f} {str(L):>10s} {lsr:5.2f} {dlt:+6.2f}")

    # router inference latency gate
    feats = [feat_for_class(ck) for ck, _ in class_freq.most_common(50)] or [{}]
    t0 = time.perf_counter()
    iters = 5000
    for i in range(iters):
        policy.decide(feats[i % len(feats)], explore_temp=0.08)
    per_call_ms = (time.perf_counter() - t0) / iters * 1000
    print(f"\nrouter inference: {per_call_ms*1000:.1f} µs/call "
          f"({'PASS' if per_call_ms <= 2 else 'FAIL'} <=2ms gate)")

    out = os.path.join(REPO, "models", "router_v1.eval.json")
    with open(out, "w") as f:
        json.dump({
            "train": len(train), "test": len(test),
            "agree": agree, "differ": differ,
            "coverage_rows": weight_cov,
            "heuristic_success": (weighted_h/weight_cov) if weight_cov else None,
            "learned_success": (weighted_l/weight_cov) if weight_cov else None,
            "lift": ((weighted_l-weighted_h)/weight_cov) if weight_cov else None,
            "inference_us_per_call": round(per_call_ms*1000, 1),
            "per_class": [
                {"class": ck, "freq": fr, "heuristic": H, "h_sr": hsr,
                 "learned": L, "l_sr": lsr, "delta": dlt}
                for ck, fr, H, hsr, L, lsr, dlt, ln, hn in rows_report
            ],
        }, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
