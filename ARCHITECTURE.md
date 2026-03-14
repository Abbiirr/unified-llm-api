# Architecture

Technical deep-dive into how the Unified Free LLM Gateway works.

## System overview

```
                          ┌─────────────────────────────────────────────┐
                          │            Docker Compose Stack             │
                          │                                             │
  Client Apps             │  ┌───────────────────────┐                  │
  (OpenAI SDK,            │  │    LiteLLM Proxy       │                  │
   Open WebUI,    ──────────▶│    (port 4000)         │                  │
   cURL, etc.)    ◀──────────│                        │                  │
       │                  │  │  ┌─────────────────┐   │    Provider Pool │
       │                  │  │  │  Router Engine   │───────▶ OpenRouter  │
       │                  │  │  │  (latency-based) │───────▶ Google      │
       │                  │  │  └────────┬────────┘   │    ▶ Cerebras   │
       │                  │  │           │            │    ▶ Groq       │
       │                  │  │  ┌────────▼────────┐   │    ▶ GitHub     │
       │                  │  │  │  Cache Layer     │   │    ▶ Mistral   │
       │                  │  │  │  (check/store)   │   │    ▶ NVIDIA    │
       │                  │  │  └────────┬────────┘   │    ▶ Cloudflare │
       │                  │  │           │            │    ▶ Cohere     │
       │                  │  └───────────┼───────────┘    ▶ Ollama     │
       │                  │              │                              │
       │                  │  ┌───────────▼───────────┐                  │
       │                  │  │    Redis 7 Alpine      │                  │
       │                  │  │    (2GB LRU cache)     │                  │
       │                  │  │    port 6379            │                  │
       │                  │  └───────────────────────┘                  │
       │                  └─────────────────────────────────────────────┘
       │
       ▼
  OpenAI-compatible API
  POST /v1/chat/completions
  GET  /v1/models
  GET  /health/readiness
```

## Request lifecycle

### Step 1: Authentication

Client sends a request with `Authorization: Bearer <LITELLM_MASTER_KEY>`. LiteLLM validates the key. Invalid or missing keys are rejected with 401.

### Step 2: Alias resolution

The `model` field in the request (e.g., `"default"`, `"coding"`, `"vision"`) is resolved to a list of **deployments** — each deployment is a specific provider + model combination.

```
"coding" resolves to:
  1. github/gpt-4.1              (GitHub Models)
  2. github/DeepSeek-R1          (GitHub Models)
  3. mistral/devstral-small      (Mistral)
  4. mistral/codestral           (Mistral)
  5. groq/qwen3-32b              (Groq)
  6. cerebras/qwen-3-235b        (Cerebras)
  7. cloudflare/qwen2.5-coder    (Cloudflare)
  8. gemini/gemma-3-27b          (Google)
```

### Step 3: Cache check

LiteLLM hashes the request into a cache key based on:
- Model alias
- Messages array
- Parameters (temperature, max_tokens, etc.)

If a matching entry exists in Redis and is within the 5-hour TTL, the cached response is returned immediately (~8-10ms) without contacting any provider.

### Step 4: Provider selection (latency-based routing)

If not cached, the router selects a deployment:

1. **Filter**: Remove deployments in cooldown (failed within last 120s)
2. **Pre-call check**: If `enable_pre_call_checks` is on, filter out deployments whose context window can't fit the input
3. **Select**: Pick the deployment with the **lowest rolling-average latency**
4. **Call**: Send the request to that provider's API

### Step 5: Failover

If the selected provider returns an error (429 rate limit, 500 server error, timeout):

1. The deployment enters a **120-second cooldown**
2. The router **retries with the next-fastest deployment**
3. This repeats up to **6 times** across different providers
4. After retry 1, there's a **1-second backoff** between retries

```
Request → OpenRouter (429) → Google (timeout) → Cerebras (200 OK) ✓
           retry 1             retry 2            success
```

### Step 6: Response caching

Successful responses are stored in Redis with:
- Key: hash of (alias + messages + params)
- Value: full response JSON
- TTL: 18,000 seconds (5 hours)
- Eviction: LRU when Redis hits the 2GB memory cap

### Step 7: Response

The response is returned to the client in OpenAI-compatible format:

```json
{
  "id": "...",
  "model": "default",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "..."
    }
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50
  }
}
```

## Alias system

Aliases are groups of deployments curated by capability. Models within each alias are ordered by benchmark quality (best first).

```
Alias          Purpose                    Selection criteria
─────────────  ─────────────────────────  ──────────────────────────────────
default        General purpose            All providers, strongest models first
fast           Speed-critical             Small models on hardware-accelerated infra
thinking       Deep reasoning / CoT       Models with native thinking (R1, QwQ, Gemini)
coding         Code generation            Code-specialist models (Codestral, Devstral, GPT-4.1)
vision         Image understanding        Models that accept image input
tools          Function/tool calling      Models with reliable tool_use support
big            Maximum parameters         253B, 405B, 235B, 120B models
local          Privacy-sensitive          Ollama only, never leaves the machine
```

### Provider-level aliases (for debugging)

Each provider also has a dedicated alias for isolated testing:

```
openrouter_free, google_free, cerebras_free, groq_free,
github_free, mistral_free, nvidia_free, cloudflare_free, cohere_free
```

## Caching architecture

```
Request arrives
    │
    ▼
┌─────────────────┐     hit      ┌─────────────┐
│  Compute cache   │────────────▶│ Return cached │
│  key (hash of    │             │ response      │
│  alias+messages  │             │ (~8ms)        │
│  +params)        │             └─────────────┘
└────────┬────────┘
         │ miss
         ▼
┌─────────────────┐             ┌─────────────┐
│ Route to provider│────────────▶│ Store in     │
│ get response     │             │ Redis + return│
│ (200-3000ms)     │             │ (TTL: 5hrs)  │
└─────────────────┘             └─────────────┘
```

**Cache key components:**
- Model alias (e.g., `"coding"`)
- Full messages array (exact content)
- temperature, max_tokens, and other generation params

**What invalidates the cache:**
- Different messages (even a single character change)
- Different temperature or max_tokens
- Different model alias
- TTL expiry (5 hours)
- LRU eviction (when Redis hits 2GB)

**What does NOT invalidate the cache:**
- Provider going down (cached responses still serve)
- Restarting LiteLLM (Redis persists across LiteLLM restarts)
- Restarting Redis (cache is ephemeral — cold start, repopulates naturally)

## Failover architecture

```
                    ┌─────────────────────────────────┐
                    │        Deployment Pool           │
                    │   (sorted by latency, lowest     │
                    │    first, cooled-down excluded)   │
                    │                                   │
Incoming request ──▶│  1. openrouter/llama-3.3  [45ms]  │──▶ try
                    │  2. gemini/2.5-flash      [120ms] │
                    │  3. groq/llama-3.3        [165ms] │
                    │  4. cerebras/qwen-235b    [350ms] │
                    │  5. nvidia/llama-3.3      [460ms] │
                    │  6. ...                           │
                    │                                   │
                    │  ✗ cloudflare/llama-3.3  [COOLED] │ ← failed 90s ago
                    └─────────────────────────────────┘

On failure:
  Provider returns 429/500/timeout
       │
       ▼
  Mark deployment as failed
  Enter 120s cooldown
       │
       ▼
  Wait 1s (retry_after)
       │
       ▼
  Try next deployment in latency order
  (up to 6 retries total)
```

**Router settings:**

| Setting | Value | Purpose |
|---|---|---|
| `routing_strategy` | `latency-based-routing` | Pick fastest healthy deployment |
| `num_retries` | 6 | Max retries across providers |
| `retry_after` | 1s | Backoff between retries |
| `allowed_fails` | 1 | Consecutive failures before cooldown |
| `cooldown_time` | 120s | How long a failed deployment is excluded |
| `timeout` | 35s | Max wait for provider response |
| `stream_timeout` | 90s | Max wait for streaming responses |
| `enable_pre_call_checks` | true | Filter by context window before calling |
| `drop_params` | true | Silently drop unsupported params per provider |

## Security model

```
Client ──bearer token──▶ LiteLLM ──provider API keys──▶ Providers
         (master key)               (from .env file)
```

- **Master key**: Single shared secret. All clients authenticate with this.
- **Provider keys**: Stored in `.env` (gitignored). Loaded as environment variables. Referenced in YAML as `os.environ/KEY_NAME` — never hardcoded.
- **No database**: `disable_database` not needed — LiteLLM runs stateless with no DB by default when no `database_url` is set.
- **Debug mode off**: `--detailed_debug` is commented out to prevent API keys from appearing in logs.

## Docker topology

```yaml
services:
  litellm:
    image: ghcr.io/berriai/litellm:main-stable
    ports: ["4000:4000"]
    depends_on: redis (healthy)
    env_file: .env
    volumes: ./litellm_config.yaml → /app/config.yaml

  redis:
    image: redis:7-alpine
    maxmemory: 2gb
    eviction: allkeys-lru
    persistence: none (ephemeral cache)
```

Both containers restart automatically (`unless-stopped`). Redis must be healthy before LiteLLM starts (`depends_on: condition: service_healthy`).

## Limitations

1. **No automatic capability detection**: The gateway doesn't inspect requests to auto-route vision/tools/thinking requests. Clients must pick the right alias. (Planned: smart routing middleware.)

2. **No per-user rate limiting**: All clients share the same master key and the same provider quotas. Adding per-user keys requires enabling the LiteLLM database.

3. **No usage tracking**: Without a database, there's no spend tracking, usage analytics, or per-key budgets. Enable PostgreSQL via `database_url` if needed.

4. **Free tier volatility**: Provider free tiers can change without notice. Models can be deprecated. Monthly checks are recommended (see [MAINTENANCE.md](MAINTENANCE.md)).

5. **Thinking model responses**: Some models (DeepSeek R1, QwQ, Cloudflare thinking models) return content in `reasoning_content` instead of `content`. Clients must handle both fields.
