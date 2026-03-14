# Unified Free LLM Gateway

A self-hosted API gateway that unifies **9 permanently free LLM providers** + local Ollama behind a single OpenAI-compatible endpoint. Zero cost, automatic failover, latency-based routing.

## What you get

- **One endpoint** (`localhost:4000/v1`) for all your LLM needs
- **9 cloud providers** with automatic failover — if one is down or rate-limited, the next one handles it
- **160,000+ free requests/day** when all providers are configured
- **Task-specific aliases** — request `coding`, `thinking`, `vision`, `tools`, or `fast` and get the best model for that task
- **Redis cache** (2GB LRU, 5-hour TTL) — identical requests served from cache, saving provider quota
- **Runs anywhere** Docker runs — laptop, server, cloud VM

## Prerequisites

- Docker and Docker Compose
- (Optional) Ollama running locally for the `local` alias

## Quick start

```bash
# 1. Clone and configure
git clone <this-repo>
cd unified-llm-api
cp .env.template .env
# Edit .env — fill in at least GEMINI_API_KEY + GROQ_API_KEY
# (full step-by-step signup instructions are in the file)

# 2. Launch
docker compose up -d

# 3. Verify
curl http://localhost:4000/health/readiness

# 4. Send a request
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_MASTER_KEY" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'
```

Minimum viable setup: just `GEMINI_API_KEY` + `GROQ_API_KEY` = ~15,000 free requests/day with 2 layers of failover. Add the rest later for more capacity.

## How routing works

The gateway does **not** auto-detect what kind of request you're sending. The client picks a model alias, and the gateway handles provider selection and failover within that alias.

### 1. Client picks an alias

```
model: "default"   → all providers, general purpose
model: "vision"    → only vision-capable models
model: "coding"    → only coding-optimized models
model: "thinking"  → only deep reasoning models
```

### 2. Gateway picks the fastest provider

Within the chosen alias, the router:
1. Selects the deployment with the **lowest recent latency**
2. If it fails (rate limit, timeout, error) → **retries with the next fastest**
3. Up to **6 retries** across different providers
4. Failed providers enter a **120-second cooldown**

### 3. Cache short-circuits repeated requests

Identical requests (same alias, messages, parameters) are served from Redis cache for 1 hour without hitting any provider.

### Important: pick the right alias

| If your request has... | Use this alias |
|---|---|
| Plain text | `default` or `fast` |
| Images attached | `vision` |
| Tool/function definitions | `tools` |
| Needs deep reasoning | `thinking` |
| Code generation | `coding` |

Using `default` for everything works but may waste retries on providers that don't support your request type (e.g., sending images to a text-only model).

## Model aliases

| Alias | Best for | Top models in this alias |
|---|---|---|
| `default` | General purpose | Llama 3.3 70B, Gemini 2.5 Pro/Flash, GPT-4.1, Qwen3 235B |
| `fast` | Speed-critical | Gemini 2.5 Flash-Lite, Groq Llama 8B, Cerebras Llama 8B |
| `thinking` | Deep reasoning | Gemini 2.5 Pro, Qwen3 235B, DeepSeek R1, QwQ-32B, GPT-OSS-120B |
| `coding` | Code gen/review | GPT-4.1, Devstral, Codestral, Qwen3-32B, Qwen2.5-Coder-32B |
| `vision` | Image understanding | Gemini 2.5 Pro/Flash, GPT-4o, GPT-4.1, Pixtral Large |
| `tools` | Function calling | GPT-4.1, GPT-4o, Gemini 2.5, Mistral Small, Llama 3.3 70B |
| `big` | Largest models | Nemotron Ultra 253B, Qwen3 235B, GPT-OSS-120B |
| `local` | Privacy (no cloud) | Ollama only |

Models within each alias are ordered by benchmark quality (best first). See [MAINTENANCE.md](MAINTENANCE.md) for the benchmarks used.

## Providers

All providers are permanently free. No trial credits, no expiration.

| # | Provider | Free limits | Signup needs |
|---|---|---|---|
| 1 | **OpenRouter** | 50 req/day (1K with $10 topup) | Email only |
| 2 | **Google AI Studio** | 5-15 RPM per model | Google account |
| 3 | **Cerebras** | 30 RPM, 1M tok/day | Email only |
| 4 | **Groq** | 1K-14K req/day | Email only |
| 5 | **GitHub Models** | Varies by Copilot tier | GitHub account |
| 6 | **Mistral** | 2 RPM, 1B tok/month | Phone required |
| 7 | **NVIDIA NIM** | 40 RPM | Phone required |
| 8 | **Cloudflare** | 10K neurons/day | Credit card ($0 plan) |
| 9 | **Cohere** | 1K req/month | Email only |

Step-by-step signup instructions for each provider are in [`.env.template`](.env.template).

## Provider-level testing

Each provider has a dedicated test alias (`google_free`, `cerebras_free`, etc.) for smoke testing and debugging:

```bash
# Test a specific provider
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_MASTER_KEY" \
  -d '{"model":"groq_free","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

## Client integration

Any app that supports OpenAI-compatible APIs works:

```
API Base:  http://localhost:4000/v1
API Key:   your LITELLM_MASTER_KEY value
Model:     default  (or any alias: fast, thinking, coding, vision, tools, big, local)
```

### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key="your-master-key")
response = client.chat.completions.create(
    model="coding",
    messages=[{"role": "user", "content": "Write a hello world in Rust"}]
)
print(response.choices[0].message.content)
```

### Open WebUI
Settings > Connections > OpenAI API:
- URL: `http://localhost:4000/v1`
- API Key: your master key
- Model: `default`

### cURL
```bash
# List available aliases
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer YOUR_MASTER_KEY"
```

## Architecture

```
Client App  →  localhost:4000 (LiteLLM Proxy)  →  Provider Pool
                        ↕                         ├─ OpenRouter
                   Redis Cache                     ├─ Google AI Studio
                   (2GB LRU)                       ├─ Cerebras
                                                   ├─ Groq
                  Latency-based                    ├─ GitHub Models
                    routing                        ├─ Mistral
                        +                          ├─ NVIDIA NIM
                  Auto-failover                    ├─ Cloudflare Workers AI
                  (6 retries)                      ├─ Cohere
                                                   └─ Ollama (local)
```

### Router settings

| Setting | Value | What it does |
|---|---|---|
| `routing_strategy` | `latency-based-routing` | Picks the fastest healthy provider |
| `num_retries` | 6 | Retries across different providers on failure |
| `cooldown_time` | 120s | How long a failed provider is excluded |
| `timeout` | 35s | Max wait for a provider response |
| `cache TTL` | 3600s | How long cached responses are kept |
| `drop_params` | true | Silently drops unsupported params per provider |

## Files

| File | Purpose | Commit? |
|---|---|---|
| `.env.template` | Setup guide + API key template | Yes |
| `.env` | Your actual API keys | **Never** (gitignored) |
| `litellm_config.yaml` | Model routing, aliases, settings | Yes |
| `docker-compose.yaml` | LiteLLM + Redis containers | Yes |
| `ARCHITECTURE.md` | Routing, caching, failover deep-dive | Yes |
| `MAINTENANCE.md` | Updating models, health checks | Yes |
| `scripts/smoke_test_gateway.sh` | Automated provider health check | Yes |
| `docs/` | Original design spec | Yes |

## Maintenance

See [MAINTENANCE.md](MAINTENANCE.md) for:
- Monthly provider health check script
- How to add/remove models and providers
- Upgrading LiteLLM and Redis
- Benchmark resources for model ordering
- Google AI Studio free tier rate limits
