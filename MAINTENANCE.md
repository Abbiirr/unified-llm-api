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
