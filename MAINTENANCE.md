# Maintenance Guide

How to keep the gateway config up to date as providers change their free tiers.

## Quick checks (monthly)

### 1. Verify providers are still responding

```bash
# Test each provider alias
for alias in openrouter_free google_free cerebras_free groq_free github_free mistral_free nvidia_free cloudflare_free cohere_free; do
  echo -n "$alias: "
  curl -s --max-time 15 http://localhost:4000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -d "{\"model\":\"$alias\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); c=d.get('choices'); print(c[0]['message']['content'] if c else 'FAIL')" 2>/dev/null || echo "FAIL"
done
```

### 2. Check for model deprecations

Each provider periodically retires models. Check these pages:

| Provider | Where to check |
|---|---|
| Google AI Studio | https://ai.google.dev/gemini-api/docs/models |
| Cerebras | https://inference-docs.cerebras.ai/models/overview |
| Groq | https://console.groq.com/docs/models |
| GitHub Models | https://github.com/marketplace?type=models |
| Mistral | https://docs.mistral.ai/getting-started/models/models_overview |
| NVIDIA NIM | https://build.nvidia.com/models |
| Cloudflare | https://developers.cloudflare.com/workers-ai/models/ |
| OpenRouter | https://openrouter.ai/collections/free-models |
| Cohere | https://docs.cohere.com/docs/models |

### 3. Check rate limit changes

Free tier limits can change without notice. Monitor:
- https://github.com/cheahjs/free-llm-api-resources (community-maintained tracker)
- Provider pricing/rate-limit pages linked above

## Adding a new model

1. Find the LiteLLM model ID format in https://docs.litellm.ai/docs/providers
2. Add it to `litellm_config.yaml` under the appropriate alias
3. Place it in benchmark-priority order (best first within each alias)
4. Restart: `docker compose restart litellm`
5. Test: `curl ... -d '{"model":"<provider>/<model-id>",...}'`

## Adding a new provider

1. Sign up and get an API key (see `.env.template` for instructions)
2. Add the key variable to `.env.template` (with setup instructions) and `.env`
3. Add model entries to `litellm_config.yaml`
4. Restart: `docker compose restart litellm`

## Removing a dead model

1. Comment it out or delete from `litellm_config.yaml`
2. Restart: `docker compose restart litellm`
3. The gateway routes around missing models automatically via failover

## Upgrading LiteLLM

```bash
docker compose pull litellm
docker compose up -d
```

The `main-stable` tag always points to the latest stable release.

## Upgrading Redis

```bash
docker compose pull redis
docker compose up -d
```

Redis data is ephemeral (cache only, no persistence), so upgrades are safe.

## Benchmark resources for ordering models

When adding models, order them best-first using these benchmarks:

| Use case | Benchmark to check |
|---|---|
| Overall quality | Chatbot Arena ELO (https://lmarena.ai) |
| Reasoning | GPQA Diamond, MATH-500, AIME |
| Coding | SWE-bench Verified, LiveCodeBench, HumanEval |
| Tool calling | Berkeley Function Calling (BFCL-v3) |
| Vision | MMMU, Video-MME |

## Google AI Studio free tier reference

| Model | RPM | RPD | Best for |
|---|---|---|---|
| Gemini 2.5 Pro | 5 | ~100 | Complex reasoning, coding |
| Gemini 2.5 Flash | 10 | ~250 | General purpose, fast |
| Gemini 2.5 Flash-Lite | 15 | ~1,000 | High volume, simple tasks |
| Gemma 3 27B | 30 | ~14,400 | Open-weight workhorse |
