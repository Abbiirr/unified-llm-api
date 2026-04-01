# CLAUDE.md

## What this is

A self-hosted LLM API gateway. One OpenAI-compatible endpoint, many free providers behind it. Clients never see rate limits — the gateway handles all failover.

## North Star

**Never return a 429 to the client.** Exhaust every fallback path first.

## Architecture

- **Port 4000** — Smart Router (public). Classifies requests, repairs messages, rescues failures.
- **Port 4002** — LiteLLM proxy (internal). Routes to 9 cloud providers + 2 Ollama hosts.
- **Port 6379** — Redis cache.

## Key files

- `smart_router.py` — the router logic
- `litellm_config.yaml` — model aliases, providers, fallback chains
- `.env` — API keys (never commit)

## Rules

- Never remove Ollama as a fallback — it's unlimited
- Never commit `.env` or API keys
- Keep cooldown_time: 60 on every deployment in the config (prevents 24hr provider lockouts)
- Log every request to `logs/training/routing.jsonl` — this trains the future ML router
- Use free/low-cost models only — max $1/day spend

## Allowed commands (no permission needed)

`docker compose *`, `curl localhost:4000|4002`, `pkill litellm|smart_router`, `uv run`, `python3`, `bash scripts/*.sh`, `git` read commands, `redis-cli` via docker

## Testing

```bash
bash scripts/test_gateway.sh      # full smoke test
bash scripts/test_ollama_models.sh # test Ollama models
```
