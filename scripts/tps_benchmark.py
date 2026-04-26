#!/usr/bin/env python3
"""Benchmark TPS (tokens/second) for every alias via the gateway."""
import json, os, time, sys, concurrent.futures as cf
import urllib.request, urllib.error

GATEWAY = "http://localhost:4000/v1/chat/completions"
KEY = os.environ.get("LITELLM_MASTER_KEY", "")
TIMEOUT = 90

ALIASES = [
    # (alias, max_tokens) — thinking models need bigger budgets
    ("tools", 256),
    ("tools_stable", 256),
    ("tools_stable_cloud", 256),
    ("tools_large", 512),
    ("tools_local", 256),
    ("coding", 512),
    ("coding_cloud", 512),
    ("thinking", 1024),
    ("thinking_cloud", 1024),
    ("big", 512),
    ("fast", 256),
    ("default", 256),
    ("default_cloud", 256),
    ("swebench", 1024),
    ("terminal_bench", 1024),
    ("vision", 256),
    ("local", 1024),
    ("gemma4-26b-local", 1024),
    ("qwopus-4b", 1024),
    ("glm-4.7-flash-local", 512),
    ("groq_free", 256),
    ("gemini_free", 256),
    ("cerebras_free", 256),
    ("nvidia_free", 256),
    ("mistral_free", 256),
    ("openrouter_free", 256),
    ("cloudflare_free", 256),
    ("cohere_free", 256),
    ("github_free", 256),
    ("llama_local", 512),
]

PROMPT = "Write exactly three short sentences about cats."

def bench(alias, max_tokens):
    body = json.dumps({
        "model": alias,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(GATEWAY, data=body, headers={
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
    })
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read())
        dt = time.time() - t0
        u = data.get("usage") or {}
        ct = u.get("completion_tokens", 0) or 0
        pt = u.get("prompt_tokens", 0) or 0
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        model = data.get("model", "?")
        tps = ct / dt if dt > 0 else 0
        return dict(alias=alias, status=200, model=model, prompt_tokens=pt,
                    completion_tokens=ct, latency_s=round(dt, 2), tps=round(tps, 1),
                    content_chars=len(content))
    except urllib.error.HTTPError as e:
        return dict(alias=alias, status=e.code, err=e.read()[:200].decode("utf-8", "replace"),
                    latency_s=round(time.time()-t0, 2), tps=0)
    except Exception as e:
        return dict(alias=alias, status=0, err=str(e)[:200],
                    latency_s=round(time.time()-t0, 2), tps=0)

def main():
    results = []
    # Run up to 4 in parallel; providers overlap so too much concurrency starves
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(bench, a, mt): a for a, mt in ALIASES}
        for f in cf.as_completed(futs):
            r = f.result()
            print(f"{r['alias']:24} status={r.get('status','?'):<3} "
                  f"tps={r.get('tps',0):>6} ct={r.get('completion_tokens','?'):>5} "
                  f"lat={r.get('latency_s','?'):>6}s  model={r.get('model', r.get('err',''))[:60]}")
            results.append(r)

    results.sort(key=lambda r: -r.get("tps", 0))
    print("\n=== Ranked by TPS ===")
    print(f"{'alias':24} {'tps':>6} {'tokens':>7} {'latency':>8} {'served_model'}")
    for r in results:
        print(f"{r['alias']:24} {r.get('tps',0):>6} {r.get('completion_tokens','?'):>7} "
              f"{r.get('latency_s','?'):>7}s  {r.get('model','-')[:60]}")

    outfile = "/home/bs01763/projects/ai/unified-llm-api/logs/tps_benchmark.json"
    json.dump(results, open(outfile, "w"), indent=2)
    print(f"\nSaved: {outfile}")

if __name__ == "__main__":
    main()
