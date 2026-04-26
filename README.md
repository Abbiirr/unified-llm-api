# Unified Free LLM Gateway

One OpenAI-compatible endpoint, 10 free cloud providers + local Ollama behind it. Clients never see rate limits — the gateway handles all failover.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Docker** (for Redis cache only)
- **API keys** from at least 2 providers (see `.env.template` for signup links)

## Quick Start

```bash
# 1. First-time setup
cp .env.template .env     # fill in API keys (minimum: GEMINI_API_KEY + GROQ_API_KEY)
uv sync                   # install dependencies

# 2. Start everything
bash scripts/start.sh

# 3. Keep it alive (auto-restarts crashed services every 60s)
nohup bash scripts/watchdog.sh >> logs/watchdog.log 2>&1 &

# 4. Use it
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{"model":"tools","messages":[{"role":"user","content":"hello"}]}'
```

## Architecture

```
Client → :4000 Smart Router → :4002 LiteLLM Proxy → cloud providers / Ollama
                  ↓
         • classifies requests (tools/coding/thinking/etc.)
         • repairs malformed messages
         • two-stage 429 rescue (Ollama → cloud fallback)
         • timeout rescue (retries via Ollama)
         • context-window rescue (upsizes to large-context alias)
```

- **Port 4000** — Smart Router (public endpoint for clients)
- **Port 4002** — LiteLLM proxy (internal, do not expose)
- **Port 6379** — Redis (response cache, provider cooldowns)

## Scripts

| Script | What it does |
|--------|-------------|
| `scripts/start.sh` | Start full stack (Redis + LiteLLM + Router) |
| `scripts/stop.sh` | Stop everything |
| `scripts/watchdog.sh` | Auto-restart crashed services — run with `nohup` |
| `scripts/status.sh` | Show gateway health, traffic, and Ollama host status |
| `scripts/test_gateway.sh` | Full smoke test suite |
| `scripts/probe_providers.sh` | Probe all providers, detect broken models, `--fix` to repair config |

## Day-to-day operations

```bash
# Check if gateway is up
bash scripts/status.sh

# Restart everything (after config changes, crashes, reboots)
bash scripts/start.sh

# Tail live request log
tail -f logs/router/router.log

# Check what LiteLLM is doing internally
tail -f logs/litellm/gateway.log

# Check watchdog
tail -f logs/watchdog.log
```

## Model Aliases

| Alias | Use for | Notes |
|-------|---------|-------|
| `tools` | Tool/function calling | Auto-promotes large payloads to `tools_large` |
| `tools_large` | Tool calling, large context (>5K chars) | 128K+ context window models |
| `tools_stable` | Tool calling, stable/predictable | Groq-primary, fast and reliable |
| `coding` | Code generation | Code-specialist models |
| `thinking` | Deep reasoning / CoT | Gemini 2.5 Pro, DeepSeek-R1, thinking-capable models |
| `big` | Maximum parameters | 70B–400B models, includes Grok-3 |
| `default` | General purpose | All providers, strongest first |
| `fast` | Speed-critical, simple tasks | Small models on fast hardware |
| `vision` | Image understanding | Multimodal models, Llama 4 Scout, GPT-4.1 |
| `local` | Privacy-sensitive | Ollama only, never leaves the machine |
| `swebench` | Long-context agent tasks | 262K+ context |
| `spec-rag` | SpecRAG V1 pipeline (with Ollama last resort) | 32K+ context, no reasoning leakage, JSON-safe |
| `spec-rag-nofallback` | SpecRAG V1 — production alias | Same chain, no Ollama. Returns 429/500 + `Retry-After: 65` on exhaustion. Use this for LightRAG indexing. |
| `gemma4-26b-local` | Gemma 4 26B thinking model | Direct Ollama, ~90-150s/req |

## Providers

| Provider | Free tier | Notes |
|----------|-----------|-------|
| Groq | 1K–14K req/day | Very fast; Llama 4 Scout, QwQ-32B, GPT-OSS-120B |
| NVIDIA NIM | 40 RPM | Nemotron Super 120B, Ultra 253B, Kimi K2 |
| Cerebras | 1M tokens/day | Hardware-accelerated; Qwen3 235B, GPT-OSS-120B |
| Gemini | 5–20 RPM | 1M context; 2.5 Pro/Flash, Gemma 4 31B |
| Mistral | 2 RPM, 1B tokens/month | Devstral, Magistral, Codestral, Pixtral |
| GitHub Models | 50–150 req/day | GPT-4.1, Grok-3, DeepSeek-R1, Llama-4-Maverick, o3 |
| OpenRouter | 50–1K req/day | Free `:free` models — Nemotron 120B, Trinity 400B, MiniMax M2.5 |
| Cloudflare Workers AI | 10K neurons/day | **order:90** — Llama 4 Scout, Kimi K2.5, GPT-OSS-120B |
| Cohere | 1K req/month | Command A Reasoning/Vision, emergency fallback |
| Ollama H1 | Unlimited | `10.112.30.10:11434` — qwen3.5:9b (~8.5 TPS), gemma4:26b, moophlo 27B |
| Ollama H2 | Unlimited | `192.168.0.73:11434` — **currently DOWN** |
| llama.cpp H2 | Unlimited | `192.168.0.73:8080` — Qwopus3.5-27B (thinking, needs `max_tokens ≥ 512`) |

## Key Config Files

- `litellm_config.yaml` — all model aliases, providers, fallback chains, retry settings
- `smart_router.py` — request classification, repair, rescue logic
- `.env` — API keys (never commit this file)

## Safety Features

- **Two-stage 429 rescue** — cloud 429 → retry via `tools_local` (Ollama) → if Ollama busy, retry via `tools_stable` (Groq)
- **Timeout rescue** — 300s timeout → retry via Ollama (max 3 concurrent slots)
- **408 rescue** — provider timeout → retry via `big` then `default_cloud` (spec-rag variants skip `big` — reasoning models leak)
- **Context-window rescue** — payload too large → auto-promote to `tools_large`
- **`Retry-After: 65` header** — `spec-rag` and `spec-rag-nofallback` add this header on any 429 or fast 500, telling clients exactly when all 60s provider cooldowns will have cleared
- **Auto-flush** — pre-rescue: clear Redis provider cooldowns so Ollama isn't blocked
- **Message repair** — orphan tool results, null content, missing schema properties
- **Cloudflare demotion** — all Cloudflare models at `order: 90` (last resort only, preserves daily quota)
- **Fast-fail on down hosts** — llama.cpp H2 entries use `timeout: 10` so a dead host fails in 10s not 300s
- **Cooldown cap** — `cooldown_time: 60` on all providers (prevents 24hr lockouts)
- **North star** — never return HTTP 429 to the client; exhaust every fallback path first
