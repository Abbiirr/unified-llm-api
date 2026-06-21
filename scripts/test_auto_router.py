#!/usr/bin/env python3
"""
test_auto_router.py — live verification of the learned auto-router.

Fires N diverse requests at the gateway with model="auto" (paced + unique
nonces so nothing is served from cache), records the routed/final alias, served
provider, status and latency, then reports:

  - success count (gate: >= --require successes, default 100)
  - status-code distribution
  - alias distribution (routed + final) and served-provider spread
  - latency percentiles
  - North-Star check: how many 429s reached the client

Usage:
    python3 scripts/test_auto_router.py                 # 110 reqs, gate 100
    python3 scripts/test_auto_router.py --n 120 --delay 1.0 --require 100
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:4000")
REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "logs", "training", "reports"))

# Diverse prompts spanning the router's request classes. Kept tiny so the test
# barely touches provider quotas; a per-request nonce defeats response caching.
PROMPTS = [
    ("short", "hi"),
    ("short", "say OK"),
    ("factual", "what is the capital of France"),
    ("factual", "what is 12 times 11"),
    ("define", "define entropy in one sentence"),
    ("define", "what is a hash map in one line"),
    ("code", "write a python function to reverse a string"),
    ("code", "write a JS function to debounce a callback"),
    ("code", "fix this bug: def add(a,b): return a-b"),
    ("code", "write a SQL query to select top 5 rows by score"),
    ("debug", "why does this raise IndexError: a=[]; a[0]"),
    ("reason", "analyze the pros and cons of monorepos in two sentences"),
    ("reason", "compare REST and GraphQL briefly"),
    ("reason", "explain why the sky is blue in one sentence"),
    ("translate", "translate 'good morning, friend' to Spanish"),
    ("summarize", "summarize photosynthesis in one sentence"),
    ("math", "what is the derivative of x^2"),
    ("list", "name three sorting algorithms"),
    ("medium", "describe the lifecycle of a TCP connection in three short steps"),
    ("medium", "outline a plan to learn Rust in a week, briefly"),
    ("essay", "write two sentences about the history of computing"),
    ("howto", "how do I center a div in CSS, one line"),
    ("rewrite", "rewrite 'the cat sat' more vividly, one line"),
    ("classify", "is 'I love this' positive or negative sentiment"),
]


def _do_request(prompt: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps({
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(GATEWAY + "/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        code, hdr, raw = r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        code, hdr, raw = e.code, dict(e.headers), e.read()
    except Exception as e:
        return {"code": f"ERR:{type(e).__name__}", "dt": time.time() - t0,
                "alias": "?", "final": "?", "provider": "?", "reason": "", "content_len": 0}
    dt = time.time() - t0
    content_len = 0
    try:
        d = json.loads(raw)
        msg = d.get("choices", [{}])[0].get("message", {}) if d.get("choices") else {}
        content_len = len((msg.get("content") or "")) + len(json.dumps(msg.get("tool_calls") or ""))
    except Exception:
        pass
    prov = hdr.get("x-litellm-model-api-base", "") or hdr.get("x-router-final-alias", "")
    return {
        "code": code, "dt": dt,
        "alias": hdr.get("x-router-alias", "?"),
        "final": hdr.get("x-router-final-alias", hdr.get("x-router-alias", "?")),
        "reason": hdr.get("x-router-reason", ""),
        "provider": prov,
        "content_len": content_len,
    }


def call(prompt: str, max_tokens: int, timeout: float):
    """Run a request with a HARD wall-clock cap. urllib's socket timeout does not
    reliably interrupt a half-hung gateway connection, so we run it in a daemon
    thread and abandon it if it overruns — no single hang can stall the suite."""
    box = {}
    t0 = time.time()

    def worker():
        box["res"] = _do_request(prompt, max_tokens, timeout)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout + 4)  # hard cap a bit above the socket timeout
    if th.is_alive() or "res" not in box:
        return {"code": "ERR:HardTimeout", "dt": time.time() - t0,
                "alias": "?", "final": "?", "provider": "?", "reason": "", "content_len": 0}
    return box["res"]


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=110)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--require", type=int, default=100)
    ap.add_argument("--retries", type=int, default=0,
                    help="client-side retries on non-200 (normal resilience; "
                         "downstream provider stalls, never 429s)")
    args = ap.parse_args()

    print(f"firing {args.n} auto requests at {GATEWAY} (delay={args.delay}s, "
          f"max_tokens={args.max_tokens})\n")
    results = []
    codes = Counter()           # final (post-retry) status per request
    alias_dist = Counter()
    final_dist = Counter()
    prov_dist = Counter()
    ok_lat = []
    ok = 0                      # final success (with retries)
    first_try_ok = 0            # success on the very first attempt
    client_429s = 0            # any 429 seen across attempts (North-Star metric)
    for i in range(args.n):
        cat, base = PROMPTS[i % len(PROMPTS)]
        res = None
        for attempt in range(args.retries + 1):
            prompt = f"{base} (req #{i+1}.{attempt})"  # nonce defeats cache
            res = call(prompt, args.max_tokens, args.timeout)
            if res["code"] == 429:
                client_429s += 1
            if attempt == 0 and res["code"] == 200:
                first_try_ok += 1
            if res["code"] == 200:
                break
            if attempt < args.retries:
                time.sleep(0.5)
        results.append({**res, "cat": cat})
        codes[res["code"]] += 1
        alias_dist[res["alias"]] += 1
        final_dist[res["final"]] += 1
        if res["provider"]:
            prov_dist[res["provider"].replace("https://", "").split("/")[0]] += 1
        if res["code"] == 200:
            ok += 1
            ok_lat.append(res["dt"] * 1000)
        flag = "OK " if res["code"] == 200 else "XX "
        print(f"{flag}{i+1:3d}/{args.n} [{str(res['code']):>4}] {res['dt']:5.1f}s "
              f"{res['alias']:>11s}->{res['final']:<11s} {cat:9s} :: {base[:30]}")
        if i < args.n - 1:
            time.sleep(args.delay)

    print("\n" + "=" * 70)
    print(f"SUCCESS: {ok}/{args.n}   (gate: >= {args.require})")
    print(f"first-attempt success: {first_try_ok}/{args.n} "
          f"({100*first_try_ok/max(args.n,1):.1f}%)  | with up to {args.retries} retries: "
          f"{ok}/{args.n} ({100*ok/max(args.n,1):.1f}%)")
    print(f"status codes : {dict(codes)}")
    print(f"client 429s  : {client_429s}  (North Star: should be 0)")
    print(f"routed alias : {dict(alias_dist.most_common())}")
    print(f"final alias  : {dict(final_dist.most_common())}")
    print(f"providers    : {dict(prov_dist.most_common())}")
    print(f"latency ms   : p50={pct(ok_lat,50)} p90={pct(ok_lat,90)} p99={pct(ok_lat,99)}")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, f"auto_router_test_{stamp}.json")
    with open(out, "w") as f:
        json.dump({
            "generated_at": stamp, "gateway": GATEWAY,
            "n": args.n, "success": ok, "require": args.require,
            "passed": ok >= args.require,
            "first_try_ok": first_try_ok, "retries": args.retries,
            "codes": {str(k): v for k, v in codes.items()},
            "client_429s": client_429s,
            "routed_alias": dict(alias_dist), "final_alias": dict(final_dist),
            "providers": dict(prov_dist),
            "latency_p50_ms": pct(ok_lat, 50), "latency_p90_ms": pct(ok_lat, 90),
            "latency_p99_ms": pct(ok_lat, 99),
            "results": results,
        }, f, indent=2)
    print(f"\nwrote {out}")
    passed = ok >= args.require
    print("RESULT:", "PASS ✅" if passed else "FAIL ❌")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
