# Maintenance Guide

How to keep the gateway running and fix common problems.

## Restart procedure

```bash
# Full restart (use this after any config change, crash, or system reboot)
bash scripts/start.sh

# Start watchdog (auto-restarts crashed services every 60s)
nohup bash scripts/watchdog.sh >> logs/watchdog.log 2>&1 &
```

`scripts/start.sh` stops any existing processes first, then starts Redis → LiteLLM → Smart Router in order. Safe to run at any time.

## Common failure modes

### LiteLLM crashed (CONNECT ERROR in router log)

**Symptom:** Router log shows `CONNECT ERROR — cannot reach LiteLLM at http://localhost:4002` for every request.

**Fix:**
```bash
bash scripts/start.sh
```

**Prevention:** Run the watchdog. LiteLLM occasionally crashes under sustained load (OOM). The watchdog detects this and restarts it automatically.

---

### 429s returned to clients

**Symptom:** Router log shows `RATE LIMITED status=429 model=... (Ollama rescue attempted but failed)`

**What's happening:** All cloud providers for a given alias are rate-limited, and the Ollama fallback also failed (Ollama busy or in cooldown).

**Fix:**
```bash
# 1. Flush Redis provider cooldowns immediately
curl -s -X POST localhost:4000/router/flush-cooldowns

# 2. If Ollama is busy with large model requests, wait 60-120s and retry
tail -f logs/router/router.log
```

**Root cause clues to check:**
- `Cloudflare... you have used up your daily free allocation` → Cloudflare quota exhausted (resets midnight UTC). Already demoted to `order: 90`.
- `OpenrouterException - rate-limited upstream` → OpenRouter free models rate-limited. Transient.
- `Ollama_chatException - Connection timed out` → Ollama H1 busy with gemma4 (17GB model, 90-150s/req). Wait for current requests to finish.

---

### spec-rag all-providers-cooled cascade (high-volume indexing)

**Symptom:** `spec-rag` returning many 429s or fast 500s (<5s) with `ALL_COOLED` entries in router log. Response includes `Retry-After: 65` header.

**What's happening:** LightRAG or similar pipeline is sending requests faster than the 10-provider chain can handle. All providers exhaust their per-minute quotas and enter 60s cooldown simultaneously. LiteLLM returns immediately (no providers to try).

**For callers (e.g. SpecRAG):**
```python
# Correct handling of ALL_COOLED responses
import time, requests

def call_spec_rag(messages):
    resp = requests.post("http://localhost:4000/v1/chat/completions",
                         headers={"Authorization": "Bearer sk-my-secret-gateway-key"},
                         json={"model": "spec-rag-nofallback", "messages": messages})
    if resp.status_code in (429, 500):
        retry_after = int(resp.headers.get("Retry-After", 65))
        time.sleep(retry_after)
        return call_spec_rag(messages)  # retry once after cooldown clears
    return resp.json()
```

**Set `SPECRAG_V1_ALIAS=spec-rag-nofallback`** — this alias gives a fast 429/500 + Retry-After header instead of falling through to an 18-minute Ollama wait.

**Gateway-side check:**
```bash
grep "ALL_COOLED" logs/router/router.log | tail -20
# Check how many providers are currently cooled:
curl -s -H "Authorization: Bearer $LITELLM_MASTER_KEY" localhost:4002/health \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('unhealthy:', len(d.get('unhealthy_endpoints',[])))"
```

---

### 504 timeouts (requests taking 300s then failing)

**Symptom:** Router log shows `TIMEOUT status=504 model=tools 300004ms`

**What's happening:** LiteLLM is waiting >300s on a provider that isn't responding.

**Most likely causes:**

1. **llama.cpp H2 down** (most common) — `192.168.0.73:8080` is unreachable but LiteLLM is trying to connect (hangs silently). All llama.cpp entries have `timeout: 10` to fast-fail, but if cooldowns expired they get retried.

   ```bash
   # Test if llama.cpp is reachable
   curl --max-time 5 http://192.168.0.73:8080/health
   # If unreachable: it will fail fast (10s) and go into 60s cooldown — self-resolving
   ```

2. **Ollama saturated** — gemma4 (17GB) occupies GPU, new requests queue. With `num_retries: 1` and `timeout: 60` on `tools_local`, each failed attempt burns 60s then gives up.

3. **Cloud provider hanging** (not returning 429, just stalling) — usually Cloudflare when quota exhausted but returning slow errors instead of immediate 429s.

   ```bash
   # Check LiteLLM log for which provider is stalling
   tail -100 logs/litellm/gateway.log | grep -E "Exception|Error|timeout"
   ```

**Fix for persistent 504s:**
```bash
# Restart clears all in-flight hanging requests
bash scripts/start.sh
```

---

### H2 llama.cpp goes down

**Symptom:** `tools_large`, `tools_local`, `coding` requests timing out at 300s. LiteLLM log shows silent TCP hangs on `192.168.0.73:8080`.

**Fix (H2 down):**
```bash
# Reduce LLAMA_CPP_HOST timeouts to 10s so they fast-fail
python3 -c "
content = open('litellm_config.yaml').read()
old = '    api_base: os.environ/LLAMA_CPP_HOST\n    api_key: fake-key\n    timeout: 300\n    stream_timeout: 600'
new = '    api_base: os.environ/LLAMA_CPP_HOST\n    api_key: fake-key\n    timeout: 10\n    stream_timeout: 20'
print('Replacing', content.count(old), 'entries')
open('litellm_config.yaml', 'w').write(content.replace(old, new))
"
bash scripts/start.sh
```

**Fix (H2 back up):**
```bash
# Restore timeouts and verify model name matches what llama.cpp is serving
curl -s http://192.168.0.73:8080/v1/models | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
# Update litellm_config.yaml model name if needed, then:
python3 -c "
content = open('litellm_config.yaml').read()
old = '    api_base: os.environ/LLAMA_CPP_HOST\n    api_key: fake-key\n    timeout: 10\n    stream_timeout: 20'
new = '    api_base: os.environ/LLAMA_CPP_HOST\n    api_key: fake-key\n    timeout: 300\n    stream_timeout: 600'
print('Replacing', content.count(old), 'entries')
open('litellm_config.yaml', 'w').write(content.replace(old, new))
"
bash scripts/start.sh
```

**Note:** The llama.cpp model (`Qwopus3.5-27B-v3-TQ3_4S.gguf`) is a thinking model. Clients need `max_tokens ≥ 512` for visible output — reasoning tokens consume budget first.

---

### Ollama H1 health flapping (UP/DOWN repeatedly)

**Symptom:** Router log alternates `OLLAMA_HOST_1 is UP` / `OLLAMA_HOST_1 is DOWN`

**What's happening:** H1 is under heavy load (gemma4 requests) and occasionally fails the health probe. Not a real outage — requests continue to work.

**No action needed.** The bypass logic handles this: if all Ollama hosts appear down, the router routes to cloud aliases automatically.

---

### Gateway down after system reboot

**Symptom:** Nothing responds on port 4000.

**Fix:**
```bash
cd /home/bs01763/projects/ai/unified-llm-api
bash scripts/start.sh
nohup bash scripts/watchdog.sh >> logs/watchdog.log 2>&1 &
```

---

## Checking gateway health

```bash
# Quick health check
bash scripts/status.sh

# Is the router up?
curl -s localhost:4000/router/health | python3 -m json.tool

# Is LiteLLM up?
curl -s localhost:4002/health/readiness

# What's Ollama H1 doing right now?
curl -s http://10.112.30.10:11434/api/ps   # running models

# Recent errors only
grep -E "ERROR|WARNING|TIMEOUT|RATE LIMIT" logs/router/router.log | tail -20

# 429s in last hour
grep "RATE LIMITED" logs/router/router.log | tail -20
```

---

## Config changes

After any change to `litellm_config.yaml` or `smart_router.py`:

```bash
bash scripts/start.sh   # restarts both LiteLLM and Router
```

**Important:** LiteLLM reads the config file only on startup. Editing the YAML has no effect until restart.

**⚠️ Do NOT use `pkill -HUP litellm`** — uvicorn treats SIGHUP as a shutdown signal, not a reload. It silently kills LiteLLM. Use `bash scripts/start.sh` for all restarts.

---

## Known limits and design decisions

| Thing | Value | Why |
|-------|-------|-----|
| `num_retries` | 1 | Higher values cause retry storms: 4×60s=240s per failed alias |
| `cooldown_time` | 60s | Prevents 24hr lockouts on transient errors |
| Cloudflare order | 90 | Quota exhausts in ~4h at high traffic; preserved for overflow only |
| llama.cpp timeout | 300s (10s when H2 down) | Set to 10s when H2 unreachable to prevent 300s silent TCP hangs; restore to 300s when H2 comes back |
| `tools_local` qwen3.5:9b timeout | 60s | Balance: enough for Ollama to respond, not so long it blocks rescue chain |
| Router read timeout | 300s | Generous for large model responses (gemma4, moophlo) |
| gemma4:26b aliases | thinking, big, swebench, terminal_bench, default, local, gemma4-26b-local | Thinking model: burns 100+ tokens reasoning before content. Never in tool-call aliases with short `max_tokens` |
| spec-rag Ollama | spec-rag only (not spec-rag-nofallback) | Ollama fallback at order 50 is a last resort. For production indexing workloads, always use `spec-rag-nofallback` — Ollama causes 18-min stalls on large payloads |
| spec-rag Retry-After | 65s | 60s cooldown + 5s buffer. On any 429 or fast 500 from spec-rag/spec-rag-nofallback, the `Retry-After` header tells the client exactly when to retry |

---

## Monthly checks

### Verify providers are still responding

```bash
for alias in tools tools_stable thinking coding fast; do
  echo -n "$alias: "
  curl -s --max-time 15 localhost:4000/v1/chat/completions \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); c=d.get('choices'); print('OK' if c and c[0]['message']['content'] else 'FAIL: '+str(d))" 2>/dev/null || echo "FAIL"
done
```

### Check for deprecated models

| Provider | Where to check |
|---|---|
| Groq | https://console.groq.com/docs/models |
| Google | https://ai.google.dev/gemini-api/docs/models |
| NVIDIA NIM | https://build.nvidia.com/models |
| OpenRouter | https://openrouter.ai/collections/free-models |
| Mistral | https://docs.mistral.ai/getting-started/models/models_overview |
| Cerebras | https://inference-docs.cerebras.ai/models/overview |
| Cloudflare | https://developers.cloudflare.com/workers-ai/models/ |
| Cohere | https://docs.cohere.com/docs/models |
| GitHub Models | https://github.com/marketplace/models |

---

## Provider status — last audited 2026-06-14

Method: probed each provider's live `/models` endpoint with our own keys (ground truth of what this gateway can actually call) **and** cross-checked official docs via web search for context windows, free-tier limits, and deprecations. IDs are in LiteLLM format. Exact RPM/RPD figures shift often and several providers no longer publish them statically — treat rate-limit numbers as guidance and verify on each provider's console.

### ⚠️ Dead models still wired in `litellm_config.yaml` (remove/replace, then `bash scripts/start.sh`)

Referenced in the config but **absent from the provider's live catalog as of 2026-06-14** — every call fails (500/timeout) and wastes a fallback slot:

| Wired model (config) | Lines | Problem | Replace with |
|---|---|---|---|
| `cloudflare/@cf/deepseek/deepseek-r1-distill-qwen-32b` | 75, 444, 1257 | **Wrong prefix** — must be `@cf/deepseek-ai/...`; the old "fix" note was backwards | `cloudflare/@cf/deepseek-ai/deepseek-r1-distill-qwen-32b` (ctx 80K) |
| `nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct` | 2088, 2096, 2528 | Removed from NIM | `nvidia_nim/qwen/qwen3.5-122b-a10b` (stable) or `…/qwen3.5-397b-a17b` (flaky) |
| `nvidia_nim/mistralai/devstral-2-123b-instruct-2512` | 2080, 2537 | Removed from NIM | `nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512` (wired — `mistral-small-4-119b-2603` was unresponsive at audit) |
| `nvidia_nim/google/gemma-3-27b-it` | 192, 403 | Removed from NIM | `nvidia_nim/google/gemma-4-31b-it` |
| `gemini/gemma-3-27b-it` | 289 | Removed from Gemini API serving | `gemini/gemma-4-31b-it` or `gemini/gemma-4-26b-a4b-it` |
| `gemini/gemma-3n-e4b-it` | 298 | Removed from Gemini API serving | `gemini/gemma-4-26b-a4b-it` (multimodal) |

### Mistral (free Experiment tier — ~1 req/s, ~500K tok/min, ~1B tok/month; verify at admin.mistral.ai)

| Model ID | Context | Notes |
|---|---|---|
| `mistral/mistral-small-latest` → `mistral-small-2603` | 256K | Mistral Small 4, hybrid instruct/reasoning/coding. Primary fast model |
| `mistral/mistral-medium-latest` → `mistral-medium-2604` | 256K | Mistral Medium 3.5, dense 128B, multimodal |
| `mistral/mistral-large-latest` → `mistral-large-2512` | 256K | Mistral Large 3, MoE 675B/41B active. Flagship |
| `mistral/codestral-latest` → `codestral-2508` | 256K | Code completion / FIM |
| `mistral/devstral-latest` → `devstral-2512` | 256K | Agentic coding, 123B (72% SWE-bench Verified) |
| `mistral/devstral-medium-latest` | 256K | **New** — 24B coding companion to devstral |
| `mistral/magistral-medium-latest` → `magistral-medium-2509` | 128K | **New — reasoning/thinking model.** Needs generous max_tokens |
| `mistral/magistral-small-latest` → `magistral-small-2509` | 128K | **New — reasoning** (24B open-weight); degrades past ~40K |
| `mistral/ministral-14b-latest` → `ministral-14b-2512` | 256K | **New** — Ministral 3, 14B dense |
| `mistral/ministral-8b-latest` | 128K | **New** — Ministral 3, 8B |
| `mistral/ministral-3b-latest` | 128K | **New** — Ministral 3, 3B edge |
| `mistral/open-mistral-nemo` | 128K | Nemo 12B, Apache-2.0 |
| `mistral/mistral-ocr-latest` | — | OCR pools only (document/PDF), not chat |

**Changes:** Added Magistral reasoning models + Ministral 3 family + devstral-medium. `mistral-medium-latest` now resolves to `mistral-medium-2604` (= "mistral-medium-3.5"). All still on free tier. Magistral are thinking models — route to thinking aliases with large max_tokens.

### Cerebras (free tier: 5 RPM / 30K TPM / 1M TPD per model — official; some trackers cite 30 RPM)

| Model ID | Context | Notes |
|---|---|---|
| `cerebras/gpt-oss-120b` | 131K | Primary, GA. ~3000 tok/s |
| `cerebras/zai-glm-4.7` | 131K | Still **Preview** (eval only, not GA). 355B, agentic/coding |

**Deprecated (final, May 27 2026):** `llama3.1-8b`, `qwen-3-235b` — removed. Live catalog is exactly these 2 models (confirmed). Catalog unchanged since last audit.
**Verify:** some sources report a temporary **8K-token free-tier context cap** despite the 131K window — check your key's effective `max_tokens`.

### Google Gemini (free tier: exact RPM/RPD no longer published — verify in AI Studio; flash ~10–15 RPM / 250–1500 RPD, pro ~5 RPM / 50–100 RPD)

| Model ID | Context | Notes |
|---|---|---|
| `gemini/gemini-2.5-flash` | 1M | Workhorse. ~10–15 RPM, 250–1500 RPD |
| `gemini/gemini-2.5-flash-lite` | 1M | Highest RPD. ~15–30 RPM, 1000–1500 RPD |
| `gemini/gemini-3-flash-preview` | 1M | Preview; low RPD |
| `gemini/gemini-3.1-flash-lite` | 1M | **GA May 2026.** Prefer over the `-preview` alias |
| `gemini/gemini-3.5-flash` | 1M (65K out) | **New — GA May 19 2026.** "Most intelligent Flash" / agentic |
| `gemini/gemini-2.5-pro` | 1M | Free but heavily capped (~5 RPM / 50–100 RPD) |
| `gemini/gemini-3.1-pro-preview` | 1M | Pro preview; minimal/uncertain free access — verify |
| `gemini/gemma-4-31b-it` | 256K | Dense, multimodal |
| `gemini/gemma-4-26b-a4b-it` | 256K | MoE, multimodal |

**Removed from Gemini API serving (confirmed):** `gemma-3-27b-it`, `gemma-3n-e4b-it` — the API's supported-Gemma list now shows only the two gemma-4 IDs (see dead-wired table). **Retired:** `gemini-2.0-flash` / `-flash-lite` (shut down ~Mar 2026, still appear in catalog); `gemini-3-pro-preview` now aliases `gemini-3.1-pro-preview`.
**Note:** Google no longer publishes static per-model free-tier numbers; trackers disagree (older 250/1000 RPD vs newer ~1500 RPD). Verify in AI Studio.

### Groq (free tier: ~30 RPM; per-model TPM 6K–12K and RPD ~1K are the binding limits — verify on console)

| Model ID | Context | Notes |
|---|---|---|
| `groq/llama-3.1-8b-instant` | 131K | Fastest/cheapest. ~14.4K RPD |
| `groq/llama-3.3-70b-versatile` | 131K | 12K TPM, **only ~1K RPD** — watch token budget |
| `groq/meta-llama/llama-4-scout-17b-16e-instruct` | 131K | MoE, multimodal |
| `groq/openai/gpt-oss-20b` | 131K | OpenAI open-weight, Apache 2.0 |
| `groq/openai/gpt-oss-120b` | 131K | Recommended; replaced kimi-k2 |
| `groq/qwen/qwen3-32b` | 131K | **Now live** (was "not yet wired") |

**Removed:** `kimi-k2-0905` — deprecated, **shut down 2026-04-15**, replaced by `gpt-oss-120b` (gone from catalog; not wired). Also retired upstream: `llama-4-maverick`, `gemma2-9b-it`.
**Available but not chat:** `groq/openai/gpt-oss-safeguard-20b` (moderation classifier — don't wire to chat pools); `groq/allam-2-7b` (Arabic, only 4K context); `groq/compound` / `compound-mini` (agentic systems with built-in web+code tools, ~250 RPD).

### OpenRouter (free `:free` tier — 20 RPM; 50 req/day, raised to 1000/day permanently after a one-time ≥$10 credit purchase)

| Model ID | Context | Status |
|---|---|---|
| `openrouter/qwen/qwen3-coder:free` | **1.0M** | Top coding pick (was 262K) |
| `openrouter/nvidia/nemotron-3-ultra-550b-a55b:free` | **1.0M** | Top reasoning, 65K out (was 131K) |
| `openrouter/nvidia/nemotron-3-super-120b-a12b:free` | **1.0M** | Strong general/reasoning (was 262K) |
| `openrouter/qwen/qwen3-next-80b-a3b-instruct:free` | 262K | Top general-purpose |
| `openrouter/poolside/laguna-m.1:free` | **262K** | Agentic coding (was 131K) |
| `openrouter/nvidia/nemotron-3-nano-30b-a3b:free` | 256K | Valid (OpenRouter ID; NIM ID differs) |
| `openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 256K | **New** — reasoning + multimodal |
| `openrouter/google/gemma-4-26b-a4b-it:free` | 262K | Valid |
| `openrouter/google/gemma-4-31b-it:free` | 262K | Valid |
| `openrouter/meta-llama/llama-3.3-70b-instruct:free` | 131K | Reliable workhorse |
| `openrouter/openai/gpt-oss-120b:free` | 131K | Valid |
| `openrouter/openai/gpt-oss-20b:free` | 131K | Valid |
| `openrouter/nousresearch/hermes-3-llama-3.1-405b:free` | 131K | Valid |
| `openrouter/nvidia/nemotron-nano-12b-v2-vl:free` | 128K | Valid, vision |
| `openrouter/nex-agi/nex-n2-pro:free` | 262K | **New** — coding/agentic, worth wiring |

**New (smaller/niche, optional):** `poolside/laguna-xs.2:free` (262K), `nvidia/nemotron-nano-9b-v2:free` (128K), `meta-llama/llama-3.2-3b-instruct:free`, `liquid/lfm-2.5-1.2b-{instruct,thinking}:free` (33K), `cognitivecomputations/dolphin-mistral-24b-venice-edition:free` (33K). Exclude `nvidia/nemotron-3.5-content-safety:free` (moderation classifier, not chat).
**⚠️ Still removed (no longer free):** `z-ai/glm-4.5-air:free`, `moonshotai/kimi-k2.6:free`.
**Context corrections:** super-120b / ultra-550b / qwen3-coder are 1.0M; laguna-m.1 is 262K (doc values were stale).

### GitHub Models (free tier: low = 150 RPD, high = 50 RPD, custom = 8–12 RPD)

LiteLLM ID format = **`github/<bare-id>`** (strip the publisher prefix, e.g. `github/gpt-4.1-mini`, not `github/openai/...`) — per docs.litellm.ai/docs/providers/github.

| Model ID | Tier | Context | Notes |
|---|---|---|---|
| `github/gpt-4.1-mini` | low | 1M | Default workhorse, 150 RPD |
| `github/gpt-4o-mini` | low | 128K | Spread low-tier RPD budget |
| `github/Llama-4-Scout-17B-16E-Instruct` | high | 10M | Long-context pool, 50 RPD |
| `github/Llama-4-Maverick-17B-128E-Instruct-FP8` | high | ~1M | **New** — stronger than Scout |
| `github/llama-3.3-70b-instruct` | high | 128K | Reliable 70B |
| `github/gpt-4.1` | high | 1M | Flagship-quality, 1M ctx |
| `github/DeepSeek-V3-0324` | high | 128K | **New** — strong V3 (was "not yet wired") |
| `github/phi-4-reasoning` | high | 32K | **New** — reasoning at 50 RPD (vs R1's 8) |
| `github/deepseek-r1` | custom | 128K | 1 RPM / **8 RPD** — last-resort reasoning only |

**New but restricted (custom tier, often Copilot-Pro-gated → may 4xx on a plain token):** `github/gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat`, `github/o3`, `o4-mini`, `deepseek-r1-0528`. Verify token entitlement before wiring.
**⚠️ Confirmed absent:** `github/grok-3` — not in our token's catalog (GitHub's rate-limit page still lists Grok tiers, so it's account/region-gated). Keep removed; re-check periodically.

### NVIDIA NIM (free tier: ~40 RPM **global per key**, shared across all models; 1,000 signup credits, +4,000 with the free AI Enterprise license)

| Model ID | Context | Notes |
|---|---|---|
| `nvidia_nim/meta/llama-3.3-70b-instruct` | 128K | Stable workhorse |
| `nvidia_nim/nvidia/nemotron-3-super-120b-a12b` | 1M | Mamba-Transformer, 120B/12B |
| `nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b` | 1M | **New** (Jun 4 2026) — long-agentic reasoning |
| `nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | 256K | Reasoning nano |
| `nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5` | 128K | Mid-tier |
| `nvidia_nim/qwen/qwen3.5-122b-a10b` | 256K | **New** — stable coding (qwen3-coder replacement) |
| `nvidia_nim/qwen/qwen3.5-397b-a17b` | 256K | Top coding scores but **FLAKY** (timeouts/500s) — keep off `order:0` |
| `nvidia_nim/qwen/qwen3-next-80b-a3b-instruct` | 256K | **New** — efficient MoE |
| `nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512` | 256K | **New** — flagship MoE; wired as devstral replacement (responds fast) |
| `nvidia_nim/mistralai/mistral-small-4-119b-2603` | 256K | **New** — Instruct+reasoning+coding, but **was hanging/down at 2026-06-14 audit** — verify before wiring |
| `nvidia_nim/deepseek-ai/deepseek-v4-flash` | 1M | **New** — fast long-context |
| `nvidia_nim/minimaxai/minimax-m3` | 1M | **New** — frontier coding, multimodal |
| `nvidia_nim/moonshotai/kimi-k2.6` | 256K | Avoid thinking mode (latency). Ctx is 256K, not 128K |
| `nvidia_nim/z-ai/glm-5.1` | 128K | **New** — coding/agentic |
| `nvidia_nim/google/gemma-4-31b-it` | 128K | Replaces gemma-3-27b |
| `nvidia_nim/nvidia/llama-3.1-nemotron-ultra-253b-v1` | 128K | **Superseded** by nemotron-3-ultra-550b — demote to fallback |

**Removed from NIM (still wired in config — fix):** `qwen3-coder-480b-a35b-instruct`, `devstral-2-123b-instruct-2512`, `gemma-3-27b-it` (see dead-wired table).
**Note:** 40 RPM is one global budget shared across all NIM models, not per-model — relevant for failover fan-out.

### Cloudflare Workers AI (10,000 Neurons/day free, resets 00:00 UTC)

| Model ID | Context | Notes |
|---|---|---|
| `cloudflare/@cf/openai/gpt-oss-120b` | 128K | Best general pick |
| `cloudflare/@cf/openai/gpt-oss-20b` | 128K | Lower-latency sibling (**128K, not 32K**) |
| `cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast` | 24K | Workhorse; small context |
| `cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct` | 131K | MoE, multimodal |
| `cloudflare/@cf/zai-org/glm-4.7-flash` | 131K | Multilingual, tool-calling |
| `cloudflare/@cf/moonshotai/kimi-k2.7-code` | 262K | Best coding pick — agentic |
| `cloudflare/@cf/moonshotai/kimi-k2.6` | 262K | **New** — frontier general/agentic |
| `cloudflare/@cf/deepseek-ai/deepseek-r1-distill-qwen-32b` | **80K** | **Prefix is `deepseek-ai`, ctx 80K** — doc had BOTH wrong (see dead-wired table) |
| `cloudflare/@cf/qwen/qwen2.5-coder-32b-instruct` | 32K | Coding |
| `cloudflare/@cf/qwen/qwen3-30b-a3b-fp8` | 32K | **New** — MoE general |
| `cloudflare/@cf/qwen/qwq-32b` | 24K | **New** — reasoning (small ctx) |
| `cloudflare/@cf/mistralai/mistral-small-3.1-24b-instruct` | 128K | **New** — text+vision |
| `cloudflare/@cf/nvidia/nemotron-3-120b-a12b` | 256K | **New** — large MoE |
| `cloudflare/@cf/google/gemma-4-26b-a4b-it` | 256K | Largest-context Gemma |
| `cloudflare/@cf/ibm-granite/granite-4.0-h-micro` | 131K | **New** — small, function-calling |

**Critical fix:** the deepseek entry must use prefix `@cf/deepseek-ai/` (not `@cf/deepseek/`) and its context is 80K — the previous "fixed prefix to deepseek" note was backwards and the wiring is currently broken.

### Cohere (free trial: 20 RPM, 1,000 API calls/month — shared cap)

| Model ID | Context | Notes |
|---|---|---|
| `cohere/command-a-03-2025` | 256K | **New** — flagship general pick |
| `cohere/command-a-plus-05-2026` | 128K | **New** — first Cohere MoE; vision + agentic |
| `cohere/command-a-reasoning-08-2025` | 256K | Reasoning. **Trial-callable** (NOT enterprise-only) |
| `cohere/command-a-vision-07-2025` | 128K | Multimodal (**128K, not 256K**) |
| `cohere/command-r-08-2024` | 128K | Stable workhorse |
| `cohere/command-r-plus-08-2024` | 128K | **New** — larger R-series |
| `cohere/command-r7b-12-2024` | 128K | **New** — small/fast, RAG/tools |
| `cohere/c4ai-aya-expanse-32b` | 128K | Multilingual research |
| `cohere/north-mini-code-1-0` | n/a | **New** — agentic coding MoE; verify trial chat access first |

**Corrections:** command-a-vision is 128K (not 256K); command-a-reasoning/vision are NOT enterprise-only — trial keys can call them (shared 1,000/mo cap). **Retired upstream (not wired):** `command-r-03-2024`, `command-r-plus-04-2024`, bare `command` / `command-light`.

### Add or remove models

**Add a model:**
1. Find LiteLLM format at https://docs.litellm.ai/docs/providers
2. Add entry to `litellm_config.yaml` under the target alias, with `cooldown_time: 60`
3. `bash scripts/start.sh`

**Remove a dead model:**
1. Delete its entry from `litellm_config.yaml`
2. `bash scripts/start.sh`

**Probe all providers automatically:**
```bash
bash scripts/probe_providers.sh --fix   # detects and removes broken models
```

---

## Upgrading LiteLLM

```bash
cd /home/bs01763/projects/ai/unified-llm-api
uv pip install --upgrade litellm
bash scripts/start.sh
```
