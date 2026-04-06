# Unified Free LLM Gateway

One OpenAI-compatible endpoint, 9 free cloud providers + local Ollama behind it. Clients never see rate limits.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Docker** (for Redis cache)
- **API keys** from at least 2 providers (see `.env.template` for signup links)

## Quick Start

```bash
# 1. Setup
cp .env.template .env     # edit and add your API keys (minimum: GEMINI + GROQ)
uv sync                   # install dependencies

# 2. Start everything
bash scripts/start.sh

# 3. Keep it alive (auto-restarts crashed services)
nohup bash scripts/watchdog.sh &

# 4. Use it
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{"model":"tools","messages":[{"role":"user","content":"hello"}]}'
```

## Architecture

```
Client → :4000 Smart Router → :4002 LiteLLM → providers
              ↓                    ↓
         classifies            routes to best
         repairs msgs          provider via
         rescues 429s          latency-based
         rescues timeouts      routing
```

## Model Aliases

| Alias | Use for |
|-------|---------|
| `tools` | Tool/function calling |
| `swebench` | SWE-bench / large-context agent tasks |
| `coding` | Code generation |
| `thinking` | Deep reasoning |
| `default` | General purpose |
| `fast` | Simple, speed-critical |
| `vision` | Image understanding |
| `local` | Privacy-sensitive (Ollama only) |
| `terminal_bench` | Terminal/CLI coding agents, strongest models |

Provider-specific aliases (`groq_free`, `nvidia_free`, etc.) also exist for direct testing.

## Scripts

| Script | What it does |
|--------|-------------|
| `scripts/start.sh` | Start full stack (Redis + LiteLLM + Router) |
| `scripts/stop.sh` | Stop everything |
| `scripts/watchdog.sh` | Auto-restart crashed services (run with nohup) |
| `scripts/status.sh` | Show gateway health, traffic, training data stats |
| `scripts/test_gateway.sh` | Full smoke test suite |
| `scripts/test_models.sh` | Test each model against its alias requirements |
| `scripts/test_ollama_models.sh` | Test and benchmark Ollama models |
| `scripts/probe_providers.sh` | Auto-probe provider APIs, detect broken models, `--fix` to repair |

## Running Without Claude

The watchdog handles everything:

```bash
# Start gateway + watchdog (survives terminal close)
bash scripts/start.sh
nohup bash scripts/watchdog.sh >> logs/watchdog.log 2>&1 &

# Check status anytime
bash scripts/status.sh

# Stop everything
bash scripts/stop.sh
```

The watchdog checks every 60s and restarts any crashed service (Redis, LiteLLM, Router).

## Training Data

The gateway collects two datasets for ML:

| Dataset | Location | Content | Purpose |
|---------|----------|---------|---------|
| Router features | `logs/training/routing.jsonl` | 68 features per request | Train ML router |
| Conversations | `logs/training/conversations/convos.jsonl` | Full request+response pairs | Fine-tune local LLM |

## Providers

9 cloud providers (all free tier) + 2 Ollama hosts:

| Provider | Context | Speed | Notes |
|----------|---------|-------|-------|
| NVIDIA NIM | 128K | Fast | Highest throughput |
| Cerebras | 65K | Fast | Daily quota limit |
| Groq | 128-262K | Very fast | 10K TPM limit |
| Mistral | 256K | Fast | codestral for large context |
| Gemini | 1M | Medium | 5-20 RPM free tier |
| GitHub Models | 8K | Medium | Small input cap |
| OpenRouter | varies | Medium | Free :free models |
| Cloudflare | 32K | Medium | Workers AI |
| Cohere | 256K | Medium | Command models |
| Ollama H1 | 262K | Slow | Unlimited, 27B models |
| Ollama H2 | 262K | Slow | Unlimited, 9B models |

## Key Config

- `litellm_config.yaml` — all model aliases, providers, fallback chains
- `smart_router.py` — request classification, repair, rescue logic
- `.env` — API keys (never commit)

## Safety Features

- **429 rescue** — rate limit from cloud → retry via Ollama
- **Timeout rescue** — 120s timeout → retry via Ollama (max 3 concurrent)
- **ContextWindow rescue** — too large for provider → retry with large-context models
- **Auto-flush** — 3 consecutive 429s → flush all provider cooldowns
- **Message repair** — orphan tool results, null content, missing schema properties
- **Cooldown cap** — all providers capped at 60s (prevents 24hr lockouts)
