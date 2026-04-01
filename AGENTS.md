# AGENTS.md — For AI agents using this gateway

## Endpoint

```
Base URL: http://localhost:4000/v1
Auth:     Bearer <LITELLM_MASTER_KEY from .env>
```

## Pick a model alias

| Alias | Use for |
|-------|---------|
| `tools` | Tool/function calling (default for agents) |
| `swebench` | SWE-bench and large-context tasks |
| `coding` | Code generation |
| `thinking` | Deep reasoning |
| `default` | General purpose |
| `fast` | Simple, speed-critical tasks |
| `vision` | Image understanding |
| `local` | Privacy-sensitive (Ollama only) |

## What the gateway handles for you

- Failover across providers (cloud first, Ollama backup)
- Large prompts auto-routed to large-context models
- Tool call ID normalization across providers
- Orphan tool results and null content auto-repaired
- Streaming enabled by default

## If you get errors

1. **429/503** — Call `POST /router/flush-cooldowns`, then retry
2. **Still failing** — Wait 60s and retry
3. Don't retry yourself — the gateway already retried internally

## Diagnostics

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/router/health` | GET | Health check (backend, Ollama status) |
| `/router/provider-status` | GET | Per-provider error rates, latency, circuit breaker state |
| `/router/classify` | POST | Debug: classify a request without forwarding |
| `/router/flush-cooldowns` | POST | Emergency: clear all provider cooldowns |
| `/router/rebuild-model-map` | POST | Rebuild model identity lookup table |
