# Gateway Improvement Report — 2026-03-29 (Rev 2)

Analysis of 64,317 requests across 149,648 router log lines and 1.3M LiteLLM log lines.
Second pass added: provider-level analysis, cooldown investigation, auto-flush, ContextWindow rescue.

## Overall Health

| Metric | Value | Assessment |
|--------|-------|------------|
| Success rate | 97.3% (62,591/64,317) | Good, but 1,726 failures |
| 429s reaching client | 1.9% (1,202) | **Violates north star** |
| 400 errors | 0.4% (247) | Context window + bad requests |
| 500 errors | 0.2% (139) | Provider internal errors |
| 413 body too large | 0.1% (84) | Missing large-payload rewrite |
| Cache hit rate | 3.3% (2,127) | Low — room for improvement |
| Streaming ratio | 93.3% | Expected (default stream=true) |

## Critical Issues Found & Fixed

### 1. 429 Rescue Cascade (61% failure rate) — FIXED

**Problem:** When the router intercepted a 429 and retried via `tools_local`, the fallback chain `tools_local → tools_cloud → default_cloud` sent the rescue request right back to the same cloud providers that were already rate-limited. Result: 60 out of 89 rescue failures were status 429 from the rescue itself.

**Fix applied:** Removed `tools_local → tools_cloud` fallback in `litellm_config.yaml`. `tools_local` is now Ollama-only — if Ollama fails, the rescue just returns the original 429 (honest failure rather than cascade).

**Expected impact:** Rescue success rate should improve from 39% to 70%+ (remaining failures only from Ollama being genuinely down).

### 2. Log Noise: 109,946 Orphan Tool Repairs — FIXED

**Problem:** Each orphan tool result removal logged a separate WARNING line. A single request with 7 orphan tool results generated 7 log lines. Total: 109,946 individual repair warnings flooding the router log.

**Root cause:** SWE-bench agent sends tool messages with empty `tool_call_id=""` — these don't match any assistant tool_call and get removed.

**Fix applied:** Batched repair logging — now one line per request: `REPAIR 7 orphan tool result(s), 2 null content fix(es)`.

### 3. ContextWindowExceeded on swebench (84 × 413 + 10 errors) — FIXED

**Problem:** Large payloads to `swebench` hit Cerebras (65K context, order:1 in the alias) before larger models. `enable_pre_call_checks` rejected the request entirely with "No models have context window large enough" instead of trying Gemini (1M) or Ollama (262K).

**Fix applied:** Added `swebench` to the large payload rewrite in `smart_router.py`. Payloads >6K chars on `swebench` now rewrite to `swebench_large` (which only has large-context models). Previously only `tools` and `bench` were rewritten.

**Note:** `swebench_large` alias needs to be created (doesn't exist yet). Until then, the existing `context_window_fallbacks: swebench → tools_large` handles some cases.

## Issues Requiring Further Work

### P1: Training Log Quality

**Problem:** `served_model` field contains opaque SHA-256 deployment hashes (e.g., `f90b38e27e39...`), not readable model names. Out of 64K entries, we can identify provider via `provider_base` but not which specific model served the request.

**Impact:** ML router training data is missing a key feature (actual model identity). We can only train on provider-level patterns, not model-level patterns.

**Recommendation:** At router startup, query `GET /v1/models` from LiteLLM and build a `model_id_hash → model_name` lookup table. Inject readable model name into training samples. Alternatively, parse `x-litellm-model-group` (already captured as `routed_alias`) plus `provider_base` as a composite key — this is 90% as useful since most providers only have 1-2 models per alias.

### P2: Streaming Token Counts Missing

**Problem:** 93.3% of requests are streaming, but streaming responses don't capture `prompt_tokens`/`completion_tokens` in the training log. Only 6.7% of requests have token usage data.

**Impact:** Can't train the ML router on token-level features for most of the data.

**Recommendation:** Parse SSE `data:` events in the streaming response to extract the final `usage` chunk (LiteLLM includes it in the last SSE event). This requires buffering the last event while still streaming through.

### P3: Provider-Level Health Tracking

**Problem:** We track Ollama host health (up/down probe every 30s), but not cloud provider health. When a cloud provider starts returning 429s consistently, we keep sending requests until LiteLLM's cooldown kicks in (60s).

**Recommendation:** Track per-provider error rates in the router. If a provider's 429 rate exceeds 50% over a 5-minute window, proactively skip it for new requests (soft circuit breaker). Log the bypass for training data.

### P4: Auto-Flush Cooldowns

**Problem:** After a burst of 429s (like the Mar 25 19:12 cascade — 10 rescue failures in 12 seconds), all providers may be simultaneously cooled down for up to 60s. The gateway becomes effectively dead for a minute.

**Recommendation:** If the router sees 3+ consecutive 429s within 10 seconds, automatically POST to `/router/flush-cooldowns` to clear LiteLLM's internal cooldown state, then retry. This is the automated version of the manual emergency recovery step in AGENTS.md.

### P5: 401 Errors on Default (40 occurrences)

**Problem:** 40 authentication errors on `default` alias. These are likely expired API keys or misconfigured providers.

**Recommendation:** Add a `/router/provider-status` endpoint that tests each provider with a minimal request and reports which keys are valid. Run this on startup and periodically (every hour). Alert when a provider starts returning 401s.

### P6: Cache Hit Rate (3.3%)

**Problem:** Only 2,127 cache hits out of 64,317 requests. Redis cache is configured with 5-hour TTL and 2GB LRU, but most SWE-bench/benchmark requests are unique (different code contexts).

**Assessment:** Low cache hit rate is expected for benchmark workloads (unique prompts). For production workloads with repeated queries, the cache would be more effective. No action needed unless workload changes.

## Model Placement Update

Two new Ollama H1 models tested and deployed:

| Model | Speed | Tools | Reasoning | Context | Aliases |
|-------|-------|-------|-----------|---------|---------|
| gurubot/jackrong-qwen3.5-claude-reasoning | 15.5 tok/s | Yes | Built-in | 262K | tools_local, thinking, coding, swebench, tools_large |
| qwen3.5-claude-262k | 15.5 tok/s* | Yes | No | 262K | swebench, tools_large |

*qwen3.5-claude-262k has `num_ctx 262144` baked in Modelfile — must use `openai/` prefix via Ollama's `/v1` endpoint, not `ollama/` prefix, to avoid 262K KV cache allocation on every request.

**LiteLLM `ollama/` tool calling bug:** LiteLLM's `ollama/` provider does not properly pass through tool_calls from the Ollama native API response. Both new models use `openai/` prefix with `api_base` pointed at Ollama's OpenAI-compatible endpoint (`/v1`), which works correctly.

## Traffic Patterns

| Alias | Requests | % | p50 Latency | p95 Latency |
|-------|----------|---|-------------|-------------|
| swebench | 37,638 | 58.5% | 743ms | 6,507ms |
| tools | 23,409 | 36.4% | 830ms | 12,671ms |
| bench_stable | 1,691 | 2.6% | 772ms | 25,899ms |
| tools_large | 455 | 0.7% | - | - |
| tools_stable | 297 | 0.5% | - | - |
| coding | 286 | 0.4% | 795ms | 35,404ms |
| default | 181 | 0.3% | 6ms* | 6,284ms |
| bench | 81 | 0.1% | 18ms* | 2,355ms |

*p50 for default/bench is low likely due to cache hits.

## 429 Rescue Performance

| Metric | Before Fix | Expected After |
|--------|-----------|----------------|
| Rescue attempts | 145 | Same |
| Success | 56 (39%) | ~100 (70%+) |
| Failed — 429 cascade | 60 | 0 (cascade removed) |
| Failed — Ollama 500 | 24 | ~24 (Ollama internal) |
| Failed — 400 bad request | 4 | ~4 (format issues) |
| Rescued models | swebench: 118, bench_stable: 27 | Same |

## Critical Issues Found & Fixed (Rev 2)

### 4. Provider Cooldown Times Not Capped (86,400s lockout) — PARTIALLY FIXED

**Problem:** Despite `retry_after_header: false` and `cooldown_time: 60` in router_settings, LiteLLM honored provider Retry-After headers:
- **Cerebras**: 86,400s (24 HOURS!)
- **GitHub**: 18,833s (5.2 hours)
- **Groq**: 2,507s (42 minutes)

When cooled, the provider is completely excluded from routing. A 24-hour lockout on Cerebras means all requests that would go to Cerebras fail for a full day.

**Root cause:** Likely a LiteLLM v1.82.2 bug — `retry_after_header: false` may not prevent the Retry-After header from being used. The config IS set correctly.

**Mitigation applied:** Auto-flush cooldowns at the router level. When the router sees 3+ 429s within 30 seconds, it automatically POSTs to `/router/flush-cooldowns` to clear all LiteLLM cooldown state. This limits dead periods to ~30 seconds maximum instead of hours.

**Further action:** File a bug with LiteLLM about `retry_after_header: false` being ignored.

### 5. ContextWindowExceeded Not Falling Through — FIXED

**Problem:** LiteLLM's `enable_pre_call_checks` rejects requests at the Cerebras model (65K context, order:1) and returns 400 ContextWindowExceeded without trying larger-context models in the same alias. The `context_window_fallbacks` config doesn't trigger when `pre_call_checks` rejects.

**Fix applied:** Added ContextWindow rescue at the router level. When the router receives a 400 with "ContextWindow" in the error message for aliases like tools/bench/swebench, it retries with `tools_large` (which only has 128K+ models).

### 6. Ollama Map Missing openai/ Prefix Models — FIXED

**Problem:** The alias→Ollama host map only detected `ollama/` prefixed models. The new gurubot and qwen3.5-claude-262k entries use `openai/` prefix with Ollama api_base URLs. This meant the dynamic Ollama bypass wouldn't trigger correctly for these aliases.

**Fix applied:** Detection now checks both the `ollama/` prefix AND whether the `api_base` URL matches a known Ollama host.

## Provider-Level Analysis

| Provider | Requests | Success% | 429s | 500s | Notes |
|----------|----------|----------|------|------|-------|
| NVIDIA NIM | 55,044 | 100.0% | 2 | 0 | **Workhorse** — serves 85.5% of all traffic |
| Cerebras | 1,699 | 99.9% | 0 | 1 | Reliable but 65K context limit |
| Groq | 1,239 | 97.6% | 0 | 30 | Occasional 500s (internal errors) |
| Gemini | 1,067 | 62.4% | 358 | 5 | **Main 429 source** — 20 req/min free limit |
| Ollama H1 | 591 | 100.0% | 0 | 0 | Never fails, just slower |
| Ollama H2 | 495 | 100.0% | 0 | 0 | Down since Mar 29 |
| GitHub | 271 | 100.0% | 0 | 0 | Low traffic, 8K input cap |
| Mistral | 251 | 100.0% | 0 | 0 | Reliable |
| unknown* | 2,990 | 56.8% | 842 | 103 | Requests where provider wasn't captured |

*"unknown" entries are likely requests that failed during LiteLLM's internal routing (all providers exhausted), so no `x-litellm-model-api-base` header was set.

## Daily Success Rate Trend

| Date | Requests | Success | Worst Error |
|------|----------|---------|-------------|
| Mar 15 | 5,458 | 96.8% | 413: 74 (body too large) |
| Mar 16 | 8,732 | 98.6% | 400: 75 |
| Mar 17 | 11,374 | 96.2% | **429: 330** (burst day) |
| Mar 18 | 4,758 | 93.7% | **429: 237** (worst day) |
| Mar 19 | 9,763 | 98.6% | 429: 126 |
| Mar 20 | 4,667 | 93.2% | **429: 268** |
| Mar 21 | 8,579 | **99.9%** | 500: 7 (best day — after fixes) |
| Mar 25 | 9,478 | 98.1% | 429: 157 |
| Mar 26 | 1,497 | 96.5% | 429: 42 |
| Mar 29 | 17 | **100%** | None (post-fix) |

## Timeout Analysis

5 total 600s timeouts across the log:

| When | Alias | Duration | Likely Cause |
|------|-------|----------|-------------|
| Mar 25 21:21 | swebench | 600s | LiteLLM retrying across all providers, all slow/failing |
| Mar 26 13:28 | swebench | 600s | Same pattern |
| Mar 26 13:28 | swebench | 600s | Same (concurrent request) |
| Mar 29 13:43 | swebench | 600s | Post-restart, providers still warming up |
| Mar 29 15:19 | thinking | 600s | New gurubot model loading + cloud providers slow |

**Recommendation:** Add timeout abort at 300s with automatic retry on a different model (e.g., tools_large or tools_local). A client waiting 10 minutes is almost as bad as a 429.

## Recommended Priority Order (Updated)

1. ~~Create `swebench_large` alias~~ — Fixed: swebench now rewrites to tools_large + CTX_RESCUE catches remaining cases
2. ~~Auto-flush cooldowns on 429 burst~~ — **DONE** (3 429s in 30s triggers auto-flush)
3. ~~ContextWindow rescue~~ — **DONE** (router-level retry with tools_large)
4. **Investigate LiteLLM `retry_after_header: false` bug** — File issue, find workaround
5. ~~Capture streaming token counts~~ — **DONE** (stream_options.include_usage=true + SSE parsing, 100% of streaming requests now have token counts)
6. ~~Add 300s soft timeout with retry~~ — **DONE** (asyncio.wait_for wraps streaming send; on timeout, retries via tools_local with concurrency limit)
7. ~~Provider health endpoint~~ — **DONE** (GET /router/provider-status — rolling 5-min stats, circuit breaker state, model identity count)
8. ~~Resolve training log model identity~~ — **DONE** (queries /model/info at startup, builds 296 model_id_hash→name mappings, injects served_model_name into training logs)
9. ~~Soft circuit breaker per provider~~ — **DONE** (providers exceeding 50% error rate over 5-min window are flagged; logged as CIRCUIT_BREAK warnings)
10. **Reduce Gemini allocation** — only 62.4% success, consider removing from high-traffic aliases
