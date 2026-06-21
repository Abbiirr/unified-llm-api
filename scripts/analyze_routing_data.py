#!/usr/bin/env python3
"""
analyze_routing_data.py — routine data-quality + routing-performance report.

Run this whenever you want to inspect the training corpus before rebuilding the
router (the "routinely check the data, clean it, iterate" step). It prints, and
writes to logs/training/reports/, a snapshot of:

  - raw vs clean row counts and why rows were dropped
  - telemetry-quality blockers (schema_version, has_tools, provider=unknown)
  - status / error-category distribution
  - per-alias client-visible success rate + latency
  - per request-class best data-driven alias (via the live policy)

Usage:
    python3 scripts/analyze_routing_data.py            # current + .1
    python3 scripts/analyze_routing_data.py --all      # every rotated log
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import router_features as rf  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router_data as rd  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(REPO, "logs", "training", "reports")


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        paths = sorted(glob.glob(rd.DEFAULT_LOG_GLOB))
    else:
        paths = [p for p in [os.path.join(REPO, "logs/training/routing.jsonl.1"),
                             os.path.join(REPO, "logs/training/routing.jsonl")]
                 if os.path.exists(p)]

    raw = 0
    drop = Counter()
    schema = Counter()
    has_tools_true = 0
    prov_unknown = 0
    status = Counter()
    errcat = Counter()
    per_alias = defaultdict(lambda: {"n": 0, "ok": 0, "lat": []})
    per_class = Counter()
    clean_rows = []

    for o in rd.iter_raw_rows(paths):
        raw += 1
        schema[o.get("schema_version", "<none>")] += 1
        if rf._as_bool(o.get("has_tools")) or (o.get("tool_count") or 0) > 0:
            has_tools_true += 1
        if o.get("provider", "unknown") in ("unknown", "", None):
            prov_unknown += 1
        status[o.get("status")] += 1
        errcat[o.get("error_category")] += 1
        c = rd.clean_row(o)
        if c is None:
            if o.get("cache_hit"):
                drop["cache_hit"] += 1
            elif o.get("error_category") == "auth":
                drop["auth"] += 1
            elif not o.get("routed_alias"):
                drop["no_alias"] += 1
            else:
                drop["other"] += 1
            continue
        clean_rows.append(c)
        d = per_alias[c["alias"]]
        d["n"] += 1
        if c["success"]:
            d["ok"] += 1
            if c["latency_ms"]:
                d["lat"].append(c["latency_ms"])
        per_class[c["class_key"]] += 1

    n_clean = len(clean_rows)
    print(f"\n=== DATA QUALITY ({len(paths)} files) ===")
    print(f"raw rows           : {raw}")
    print(f"clean rows         : {n_clean} ({100*n_clean/max(raw,1):.1f}%)")
    print(f"dropped            : {dict(drop)}")
    print(f"schema_version     : {dict(schema)}")
    print(f"has_tools/tool_count>0: {has_tools_true} ({100*has_tools_true/max(raw,1):.1f}%)")
    print(f"provider unknown   : {prov_unknown} ({100*prov_unknown/max(raw,1):.1f}%)")
    print(f"status (top)       : {status.most_common(8)}")
    print(f"error_category     : {errcat.most_common(8)}")

    print(f"\n=== PER-ALIAS CLIENT-VISIBLE OUTCOME (clean) ===")
    print(f"{'alias':20s} {'n':>7s} {'succ%':>6s} {'p50ms':>7s} {'p95ms':>8s}")
    for a, d in sorted(per_alias.items(), key=lambda x: -x[1]["n"]):
        if d["n"] < 30:
            continue
        sr = 100 * d["ok"] / d["n"]
        print(f"{a:20s} {d['n']:7d} {sr:6.1f} {str(pct(d['lat'],50)):>7s} {str(pct(d['lat'],95)):>8s}")

    # per-class best alias via live policy (if a model exists)
    best_by_class = []
    try:
        import router_policy
        pol = router_policy.RouterPolicy.load()
        if pol:
            print(f"\n=== DATA-DRIVEN BEST ALIAS PER CLASS (cost_bias=5) ===")
            for ck, cnt in per_class.most_common(15):
                mod = ck.split("|", 1)[0]
                feat = {"_ck": ck}
                # synthesize a feat that reproduces this class key
                parts = ck.split("|")
                feat = _feat_for_class(parts)
                d = pol.decide(feat)
                best_by_class.append({"class": ck, "n": cnt, "alias": d.alias,
                                      "util": d.utility, "support": d.confidence})
                print(f"  {ck:20s} n={cnt:6d} -> {str(d.alias):12s} util={(d.utility or 0):.3f} support={d.confidence}")
    except Exception as e:
        print("policy report skipped:", e)

    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated_at": stamp,
        "files": [os.path.relpath(p, REPO) for p in paths],
        "raw_rows": raw, "clean_rows": n_clean, "dropped": dict(drop),
        "schema_version": dict(schema),
        "has_tools_or_toolcount": has_tools_true,
        "provider_unknown": prov_unknown,
        "status": {str(k): v for k, v in status.items()},
        "per_alias": {a: {"n": d["n"], "succ_rate": round(d["ok"]/d["n"], 4),
                          "p50_ms": pct(d["lat"], 50), "p95_ms": pct(d["lat"], 95)}
                      for a, d in per_alias.items() if d["n"] >= 30},
        "best_by_class": best_by_class,
    }
    out = os.path.join(REPORT_DIR, f"routing_report_{stamp}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


def _feat_for_class(parts):
    """Build a minimal feature dict that maps back to a given modality|size|content key."""
    mod, size, content = (parts + ["text", "s", "plain"])[:3]
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


if __name__ == "__main__":
    main()
