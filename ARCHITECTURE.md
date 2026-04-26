# Architecture

Technical deep-dive into how the Unified Free LLM Gateway works.

## System overview

```
                        ┌────────────────────────────────────────────────┐
                        │             Gateway Stack (uv run)              │
                        │                                                 │
  Client Apps           │  ┌─────────────────────────────────────────┐   │
  (OpenAI SDK,          │  │         Smart Router  :4000  (public)   │   │
   Open WebUI,   ──────────▶                                         │   │
   cURL, etc.)   ◀──────────│  • classifies requests by type         │   │
                        │  │  • repairs malformed messages           │   │
                        │  │  • two-stage 429 rescue                 │   │
                        │  │  • timeout rescue (max 3 concurrent)    │   │
                        │  │  • context-window rescue                │   │
                        │  │  • logs to training JSONL               │   │
                        │  └──────────────────┬──────────────────────┘   │
                        │                     │                          │
                        │  ┌──────────────────▼──────────────────────┐   │
                        │  │       LiteLLM Proxy  :4002  (internal)  │   │
                        │  │                                         │   │
                        │  │  • latency-based routing                │   │
                        │  │  • provider failover + cooldowns        │   │
                        │  │  • response caching (via Redis)         │   │
                        │  │  • context-window pre-call checks       │   │
                        │  └──────┬──────┬──────┬──────┬────────────┘   │
                        │         │      │      │      │                 │
                        └─────────┼──────┼──────┼──────┼─────────────────┘
                                  │      │      │      │
                        ┌─────────▼─┐ ┌──▼──┐ ┌▼────┐ ┌▼──────────────┐
                        │  Groq     │ │NVID │ │Gemni│ │ Ollama H1     │
                        │  Cerebras │ │ NIM │ │ etc.│ │ 10.112.30.10  │
                        │  Mistral  │ └─────┘ └─────┘ │ :11434        │
                        │  OpenRtr  │                  └───────────────┘
                        │  Cloudflr │
                        └───────────┘

         Redis :6379  (response cache + provider cooldown state)
```

## Process topology

The gateway runs as three processes launched by `scripts/start.sh`:

| Process | Port | Started by | Config |
|---------|------|-----------|--------|
| `redis:7-alpine` (Docker) | 6379 | `docker run` (container: `llm-cache`) | `maxmemory 2gb, allkeys-lru` |
| `uv run litellm --config litellm_config.yaml` | 4002 | `uv run` | `litellm_config.yaml` |
| `uv run python smart_router.py` | 4000 | `uv run` | env vars from `.env` |

The watchdog (`scripts/watchdog.sh`) polls every 60s and restarts any crashed process.

## Request lifecycle

### Step 1: Smart Router receives request (:4000)

Client sends `POST /v1/chat/completions` with `Authorization: Bearer <LITELLM_MASTER_KEY>`.

The router:
1. **Classifies** the request — inspects `model`, message content, payload size
2. **Repairs** messages — fixes orphan tool results, null content, missing schema fields
3. **Rewrites model** — e.g., large payload on `tools` → rewrite to `tools_large`
4. **Ollama bypass** — if all Ollama hosts are down, rewrites `local` → `local_cloud` etc.
5. **Forwards** to LiteLLM at `localhost:4002`

### Step 2: LiteLLM routes to provider (:4002)

1. **Cache check** — hash of (alias + messages + params) looked up in Redis
2. **Filter** — remove providers in cooldown (failed within last 60s)
3. **Pre-call check** — filter providers whose context window can't fit the input
4. **Select** — pick lowest-latency healthy provider
5. **Call** — send to provider API

### Step 3: Failover

If provider fails (429, 500, timeout):
1. Provider enters 60s cooldown
2. LiteLLM retries with next provider (up to `num_retries: 1` retry per alias)
3. If all providers exhausted → return error to router

### Step 4: Router rescue logic

The router intercepts errors from LiteLLM and applies rescue layers:

```
LiteLLM returns 429
  └─▶ 429_RESCUE stage 1: retry with model=tools_local (Ollama qwen3.5:9b)
        ├─ 200 OK → return to client ✓
        └─ 500 (Ollama busy) → 429_RESCUE stage 2: retry with model=tools_stable (Groq)
              ├─ 200 OK → return to client ✓
              └─ fail → return original 429 (last resort)
  NOTE: spec-rag / spec-rag-nofallback skip the Ollama rescue entirely.
        They return 429 + Retry-After: 65 header instead.

LiteLLM returns 408 (provider timeout)
  └─▶ 408_RESCUE: retry with model=big → then model=default_cloud
        ├─ 200 OK → return to client ✓
        └─ fail → return original 408
  NOTE: spec-rag / spec-rag-nofallback skip big (reasoning models) → try default_cloud only.

LiteLLM times out at 300s
  └─▶ TIMEOUT_RESCUE: retry with model=tools_local (Ollama, max 3 concurrent)
        ├─ 200 OK → return to client ✓
        └─ fail/exception → return 504

LiteLLM returns 413 (context too large)
  └─▶ CONTEXT_RESCUE: retry with model=tools_large (128K+ models)
        ├─ 200 OK → return to client ✓
        └─ fail → return original 413

spec-rag / spec-rag-nofallback all-providers-cooled (500 in <5s)
  └─▶ Return 500 + Retry-After: 65 header
        (LiteLLM exhausted all providers in <5s = all in 60s cooldown)
        Callers should wait 65s then retry.
```

### Step 5: Response caching

Successful responses stored in Redis:
- Key: hash of (alias + messages + params)
- TTL: 18,000 seconds (5 hours)
- Eviction: LRU when Redis hits 2GB

## Alias system

Each alias is a named group of deployments. LiteLLM picks the fastest healthy one.

```
Alias                Purpose                         Fallback chain (litellm_config.yaml)
───────────────────  ──────────────────────────────  ─────────────────────────────────────
tools                Tool/function calling            tools → tools_local
tools_large          Large-context tool calling       tools_large → tools_local → default_cloud
tools_stable         Stable tool calling (Groq)       tools_stable → tools_local
coding               Code generation                  coding → coding_cloud
thinking             Deep reasoning                   thinking → thinking_cloud
big                  Large models (70B-235B)          big → tools_local → default_cloud
default              General purpose                  (no fallback — many providers in alias)
fast                 Speed-critical                   fast → default_cloud
local                Ollama only (privacy)            (no cloud fallback by design)
swebench             Long-context agent tasks         swebench → tools_local
spec-rag             SpecRAG V1 (with Ollama sink)    spec-rag → default_cloud
spec-rag-nofallback  SpecRAG V1 production alias      spec-rag-nofallback → default_cloud
                     No Ollama. Fast-fail with
                     Retry-After: 65 on exhaustion.
```

**spec-rag / spec-rag-nofallback provider chain (10 providers, priority order):**

| Order | Provider | Model | Notes |
|-------|----------|-------|-------|
| 1 | Groq | llama-3.3-70b-versatile | Primary — ~1s, separate RPM bucket |
| 1 | Groq | llama-4-scout-17b-16e | Separate Groq quota from llama-3.3-70b |
| 1 | NVIDIA NIM | llama-3.3-70b-instruct | 40 RPM |
| 1 | Gemini | gemini-2.5-flash | 1M ctx |
| 2 | Cloudflare | llama-4-scout-17b-16e | Separate provider quota |
| 5 | Gemini | gemini-2.5-flash-lite | Higher RPM than flash |
| 5 | Mistral | mistral-small-latest | Independent rate limits |
| 8 | GitHub | Llama-4-Scout-17B | 150 req/day, separate quota |
| 10 | OpenRouter | llama-3.3-70b-instruct:free | Free tier backup |
| 20 | Cloudflare | llama-3.3-70b-instruct-fp8-fast | Last cloud resort |
| 50 | Ollama H1 | qwen3.5:9b | **spec-rag only** (not nofallback) |

**Why reasoning models are excluded from spec-rag:** gpt-oss, qwen3-235b, nemotron-ultra, and gemma4:26b all place answers in the `reasoning` field or prefix content with `<think>` blocks. SpecRAG parses `choices[0].message.content` directly and would receive empty or garbled output.

## Ollama host handling

The router maintains a health map of Ollama hosts, probed every 30s:

```python
OLLAMA_HOSTS = {
    "OLLAMA_HOST_1": "http://10.112.30.10:11434",   # H1 — UP
    "OLLAMA_HOST_2": "http://192.168.0.73:11434",    # H2 — DOWN
}
```

When routing to an alias that uses Ollama, the router checks which hosts are healthy and builds `OLLAMA_MAP` (alias → healthy hosts). If all Ollama hosts are down, requests to Ollama-dependent aliases are rewritten to their `_cloud` counterpart.

**llama.cpp (H2 port 8080):** Separate from Ollama. All entries use `timeout: 10` for fast-fail when H2 is down.

## Key config values

### `litellm_config.yaml` router_settings

| Setting | Value | Reason |
|---------|-------|--------|
| `routing_strategy` | `latency-based-routing` | Fastest healthy provider wins |
| `num_retries` | 1 | Higher → retry storms (4×60s=240s per failed alias) |
| `retry_after` | 2s | Backoff between retries |
| `allowed_fails` | 1 | Failures before cooldown |
| `cooldown_time` | 60s | Prevents 24hr lockouts on transient errors |
| `timeout` | 600s | LiteLLM-level timeout per request |
| `enable_pre_call_checks` | true | Skip providers that can't fit context |

### Per-model timeouts

| Model / group | timeout | stream_timeout | Why |
|---------------|---------|---------------|-----|
| `tools_local` qwen3.5:9b | 60s | 120s | Balance: enough time for Ollama, not too long to block rescue |
| `moophlo` / gemma4 (Ollama) | 300s | 600s | Large models — 90-150s inference time |
| `LLAMA_CPP_HOST` entries | 10s | 20s | H2 is down; fast-fail prevents 300s silent hangs |
| Cloudflare models | (default) | (default) | order: 90 — only used when all others exhausted |

### Smart Router timeouts

| Setting | Value | Location |
|---------|-------|---------|
| `httpx read timeout` | 300s | `smart_router.py` line ~421 |
| Rescue concurrent limit | 3 | `_MAX_CONCURRENT_RESCUES` |
| Cooldown flush | Redis `deployment:*:cooldown` keys | Pre-rescue in `_auto_flush_if_needed` |

## Cloudflare quota management

Cloudflare Workers AI has a **10,000 neuron/day** limit that exhausts in ~4 hours at high traffic. All Cloudflare models are set to `order: 90` (16 entries across all aliases) — they're the last resort, not the primary. This preserves the quota for genuine overflow situations.

Quota resets at **midnight UTC**. After reset, Cloudflare responds normally and the order:90 placement means it's only reached when all higher-priority providers (Groq, Gemini, NVIDIA) are exhausted or rate-limited.

## Security

```
Client ──bearer token──▶ Smart Router ──forward──▶ LiteLLM ──provider keys──▶ Providers
         (LITELLM_MASTER_KEY)                                  (from .env)
```

- Master key: all clients use the same key (`LITELLM_MASTER_KEY` in `.env`)
- Provider keys: in `.env` (gitignored), referenced as `os.environ/KEY_NAME` in YAML
- LiteLLM runs on `:4002` — bind to localhost only, not exposed externally
- No database required — stateless operation

## Training data

Every request is logged to two datasets:

| Dataset | Path | Format |
|---------|------|--------|
| Routing features (68 fields) | `logs/training/routing.jsonl` | JSONL, one record per request |
| Full conversations | `logs/training/conversations/convos.jsonl` | JSONL, request + response pairs |

These feed future ML-based routing improvements.
