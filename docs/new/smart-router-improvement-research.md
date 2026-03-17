# Smart Router Improvement Research

Research date: 2026-03-16

## Executive summary

The biggest problem is not the keyword classifier in [`docs/new/smart_router.py`](./smart_router.py). The biggest problem is the combination of:

1. Mixed-quality provider pools inside the same alias
2. Very high retry depth in LiteLLM
3. Capability mismatches that should be filtered before a provider is ever tried
4. Conversation-history corruption after interrupted tool streams
5. Runtime/docs drift, which makes the system harder to reason about

The live router in [`../../smart_router.py`](../../smart_router.py) is already much more advanced than the `docs/new` example. It has request repair, tool-call ID normalization, training logs, and large-payload alias rewrites. But the current LiteLLM config still causes avoidable failures:

- `num_retries: 12`, `retry_after: 0`, `allowed_fails: 0`, `cooldown_time: 3600` in [`../../litellm_config.yaml`](../../litellm_config.yaml)
- `enable_pre_call_checks: false`, even though the config contains several very small-context models
- `tools`, `bench`, and `default` aliases still contain providers with known structural incompatibilities for large tool-heavy traffic
- fallbacks like `tools -> default` widen the blast radius instead of preserving capability

My recommendation is to treat this as a provider-policy problem first and a classification problem second.

## What the repo is doing today

### 1. The live router is not the same as the docs/new router

The `docs/new` router is a thin classifier/proxy only. The live router in [`../../smart_router.py`](../../smart_router.py) already does all of this:

- request classification
- tool-call ID normalization
- conversation repair for orphaned tool messages and empty assistant content
- automatic `tools_large` / `bench_large` rewrite for big payloads
- training-data logging to `logs/training/routing.jsonl`

That means improvements should target the live router and the root LiteLLM config first, then sync `docs/new` afterward.

### 2. Local evidence from your logs

From `logs/training/routing.jsonl` in this workspace:

- 6,959 total logged requests
- 6,849 were `tools` requests (98.4 percent)
- 6,757 returned `200`
- 74 returned `413`
- 88 returned `400`
- 16 returned `500`

From `logs/router/router.log`:

- 175 router-level timeouts
- 51 logged `413`s
- 69 logged `400`s
- 19 connect errors
- 6 logged `500`s

Two important implications:

1. Your real hot path is tool calling, not general chat.
2. The router should optimize for long-lived, multi-turn, tool-heavy conversations first.

### 3. The current configuration is amplifying failures

Key runtime choices in [`../../litellm_config.yaml`](../../litellm_config.yaml):

- `tools` includes Gemini, Groq, Cerebras, NVIDIA NIM, GitHub Models, and Ollama
- `bench` includes Cerebras, Groq, NVIDIA NIM, Gemini, OpenRouter free, Cloudflare, GitHub Models, and Ollama
- `tools -> default` and `bench -> default` fallbacks are enabled
- `enable_pre_call_checks` is disabled

That is exactly the kind of mixed pool that turns one bad provider into many wasted retries.

## Root causes

## A. Alias design is too broad for the workload

`tools`, `bench`, and `default` mix providers with very different:

- context limits
- tool-call validators
- rate-limit shapes
- free-tier quotas
- tolerance for repaired conversation history

For tool-heavy workloads, "supports tools" is not enough. You need at least these sub-capabilities:

- reliable multi-turn tool state
- large prompt tolerance
- tolerant OpenAI-compatible message parsing
- predictable rate limits

## B. Retry policy is treating structural errors like transient errors

The current config retries a lot, very quickly, with no backoff:

- `num_retries: 12`
- `retry_after: 0`
- `allowed_fails: 0`
- `cooldown_time: 3600`

That is the wrong shape for three of your most common failures:

- `413` request-too-large errors are structural, not transient
- `400` tool/message validation errors are usually structural unless the request is transformed
- daily quota exhaustion should cool down until quota reset, not for a generic one-hour window

## C. Context filtering is off

`enable_pre_call_checks: false` is likely contributing directly to the `413`s. The config still includes GitHub entries marked with `max_input_tokens: 8000` inside `tools`, `bench`, and `default`, so large tool-heavy requests can reach those deployments even though they are obviously poor fits.

LiteLLM's own docs recommend pre-call checks and also support context-window fallbacks and tag-based routing:

- Reliability / retries / timeouts: https://docs.litellm.ai/docs/proxy/reliability
- Routing by request metadata / tags: https://docs.litellm.ai/docs/proxy/load_balancing

## D. The router repairs messages, but it does not manage interrupted tool-call state

The live router already repairs some history problems in [`../../smart_router.py`](../../smart_router.py), but it does not keep a durable notion of "pending tool loop state". That leaves a gap for the exact failure you described:

- provider streams part of a tool-calling assistant turn
- stream is cut off
- client retries with a tool response that no longer matches the last complete assistant tool call
- LiteLLM or the upstream provider rejects the conversation

For tool-heavy workloads, this needs a local state-machine check, not just best-effort cleanup.

## E. Failure attribution is incomplete

Many non-200 rows in `logs/training/routing.jsonl` have empty `provider_base`. That makes it hard to answer "which provider is really causing this status code?" from training logs alone. You need better provider attribution on failure paths, ideally before LiteLLM retries away from the first bad deployment.

## F. Docs and runtime have drifted

Examples:

- [`../../README.md`](../../README.md) still describes 6 retries and 120-second cooldowns, while runtime config uses 12 retries and 3600-second cooldowns.
- [`../../README.md`](../../README.md) says the gateway does not auto-detect request type, but the live router does.
- [`../../README.md`](../../README.md) points to `logs/routing_training.jsonl`, while the live router writes `logs/training/routing.jsonl`.
- [`../../scripts/run_router.sh`](../../scripts/run_router.sh) defaults `LITELLM_BASE` to `http://localhost:4002`, which matches the `CONNECT ERROR` pattern in the router log. Port `4002` is your docs service, not LiteLLM.

## Provider-by-provider recommendations

| Provider | Recommendation | Why |
|---|---|---|
| Cerebras | Make primary for `tools_stable`, `bench_stable`, and long tool chains | It is your best reliability candidate, and the local sample shows it succeeds without the same class of structural failures |
| NVIDIA NIM | Keep as secondary/tertiary in stable aliases | Recent local logs show it is actually serving a large share of successful tool traffic, so it should stay in the stable pool |
| Groq | Keep, but only in controlled aliases | Groq's tool-use docs require strict OpenAI-style tool loops, so use it where request history is short and fully validated, not as a generic long-run fallback |
| Gemini / Google AI Studio | Remove from `tools` and `bench` hot paths | Official free-tier rate limits are low enough that multi-turn benchmarks can burn quota quickly; good for selected long-context text tasks, bad as a default fallback for tool-heavy traffic |
| Cohere | Remove from general/tool aliases | Cohere tool calls use a stricter message shape than the OpenAI format many other providers tolerate; this is a poor fit for repaired cross-provider histories |
| Mistral | Keep only where your 9-char tool-call normalization is guaranteed | Mistral's own function-calling examples use 9-character alphanumeric tool call IDs, which matches the current normalizer |
| GitHub Models | Remove from `tools`, `bench`, and large-prompt aliases | GitHub explicitly frames the service as a prototyping API with per-request token ceilings and rate limits that are too small for SWE-bench-style prompts |
| OpenRouter free | Keep only in dedicated low-priority aliases | OpenRouter documents shared free-model rate limits and daily caps, so one free-model failure can poison the rest of the free pool |
| Cloudflare Workers AI | Keep out of primary stable pools | Useful as extra capacity, but not the first place to send your highest-cost tool loops |
| Ollama | Keep as final capability-preserving fallback | Valuable for privacy and offline continuity, but it should not silently mask cloud-provider instability in your metrics |

## What current docs say, and how that maps to your issues

### Google AI Studio / Gemini

Official Gemini rate limits are low on the free tier for the models you are using. Google's rate-limit page currently lists examples such as:

- Gemini 2.5 Pro: 5 RPM / 100 RPD
- Gemini 2.5 Flash: 10 RPM / 250 RPD
- Gemini 2.5 Flash-Lite: 15 RPM / 1000 RPD

Source: https://ai.google.dev/gemini-api/docs/rate-limits

Implication: your observed "quota exhausted early in the day" is consistent with the official free-tier shape. Gemini should not sit near the front of a hot `tools` or `bench` alias.

### Groq

Groq's docs show explicit tool-use sequencing and model-specific rate/throughput limits. Source pages:

- Models / rates: https://console.groq.com/docs/models
- Tool use: https://console.groq.com/docs/tool-use

Inference: the `role:tool` and "Failed to call a function" failures are consistent with a provider that is stricter about the exact tool-call loop than some others. Keep Groq only where the router can guarantee a valid loop.

### Cohere

Cohere's docs model tool use as a structured sequence with explicit tool plans and tool results, and its message schema is stricter than the relaxed OpenAI-ish traffic some providers accept.

- Tool use overview: https://docs.cohere.com/docs/tool-use-overview
- Chat API docs: https://docs.cohere.com/v2/reference/chat

Inference: your `content: null` and message-shape failures fit this behavior. Cohere should be isolated to dedicated aliases or removed from the tool path entirely.

### Mistral

Mistral's own function-calling example uses tool call IDs like `D681PevKs`, which supports keeping your current 9-character alphanumeric normalization.

Source: https://docs.mistral.ai/capabilities/function_calling/

Implication: keep the normalizer, but do not assume Mistral will tolerate unnormalized cross-provider IDs.

### GitHub Models

GitHub's docs describe the service as a place to prototype, not a production-grade long-context router target. The current prototyping docs show per-request token limits and rate limits by tier, for example:

- Free: 50 requests/day, 1500 requests/day with a paid Copilot plan
- Example per-request token ceilings from the docs are far below your large benchmark prompts

Sources:

- https://docs.github.com/en/github-models/prototyping-with-ai-models
- https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models

Implication: GitHub Models should be routed only when the prompt is known to be short enough.

### OpenRouter free

OpenRouter documents shared free-model limits such as:

- 20 requests/minute
- 50 requests/day for free models, unless you top up credits

Source: https://openrouter.ai/docs/api-reference/limits

Implication: free-model failures are not independent. Treat the entire free pool as a shared-quota class.

## Recommended target architecture

## Layer 1: classify by capability, not just intent

Replace the current coarse aliases with narrower ones for the hot path:

- `tools_stable`: Cerebras, NVIDIA NIM, Ollama
- `tools_experimental`: Groq, Gemini, Mistral
- `tools_large`: only long-context providers that also survive multi-turn tools
- `bench_stable`: Cerebras, NVIDIA NIM, OpenRouter free (last), Ollama
- `github_short_ctx`: GitHub only, gated by strict prompt-size check
- `thinking_long`: Cerebras, Gemini, NVIDIA NIM

The current `tools -> default` fallback should be removed. Fall back to another tool-capable alias, not to a generic alias with incompatible providers.

## Layer 2: add provider policy before LiteLLM

The smart router should own a small provider policy registry with rules like:

- if request has tools and more than `N` prior turns, never send to GitHub / Gemini / Cohere
- if payload estimate exceeds GitHub allowance, never send to GitHub
- if provider class is `shared_free_quota` and one returns `402` or shared `429`, quarantine the class
- if Gemini returns daily quota exhaustion, quarantine until next reset window, not 300 seconds or 3600 seconds
- if error is structural (`400`, `413`), do not cool down the whole provider as if it were transient; instead mark it incompatible for that request shape

LiteLLM can still do load-balancing and retries, but only after the smart router has reduced the pool to compatible candidates.

## Layer 3: local conversation-state validator

Before forwarding any tool request:

1. walk the conversation in order
2. ensure every `tool` message has a matching prior assistant tool call
3. ensure there is no partial assistant tool call left over from an interrupted stream
4. normalize IDs once, then validate again
5. if invalid, repair conservatively or fail fast with a clear router error

The existing `repair_messages()` function is a good base, but it should become a deterministic validator/state machine rather than a best-effort scrubber.

## Layer 4: better retry semantics

Recommended LiteLLM/router behavior:

- reduce `num_retries` from 12 to something like 3-5
- add non-zero backoff
- turn `enable_pre_call_checks` back on
- use capability-preserving fallbacks
- distinguish transient vs structural failures

Suggested retry classes:

- Retryable: `429`, `500`, `502`, `503`, transport timeouts
- Retryable only after transform: provider-specific `400` message/tool-shape errors
- Non-retryable: `413`, unsupported context, known incompatible request shape

## Concrete changes I would make first

## P0: highest ROI

1. Remove Cohere from tool-capable aliases.
2. Remove Gemini from `tools`, `tools_large`, and `bench` hot paths.
3. Remove GitHub Models from `tools`, `bench`, and any large-prompt alias.
4. Turn `enable_pre_call_checks` back on.
5. Replace `tools -> default` with `tools -> tools_stable`.
6. Replace `bench -> default` with `bench -> bench_stable`.
7. Reduce retries and add backoff.
8. Fix [`../../scripts/run_router.sh`](../../scripts/run_router.sh) to default `LITELLM_BASE` to LiteLLM, not the docs server.

## P1: make tool runs robust

1. Promote `repair_messages()` into a full tool-loop validator.
2. Persist pending tool-call state across interrupted streams.
3. Add per-provider request transforms instead of one generic repair path.
4. Quarantine whole provider classes for shared-quota failures.
5. Capture provider identity before LiteLLM retries away from it.

## P2: smarter routing

1. Train a score-based router from `logs/training/routing.jsonl`.
2. Add provider health probes that populate a live policy cache.
3. Refresh model context/rate metadata automatically so `model_info.max_input_tokens` stays accurate.
4. Sync `docs/new` from runtime code instead of maintaining parallel behavior by hand.

## Specific file-level recommendations

### `smart_router.py`

- Keep the current ID normalizer and conversation repair logic.
- Add a provider-policy layer before alias selection is finalized.
- Add request-size estimation that uses token estimates, not only `len(json.dumps(messages))`.
- Add a validator for interrupted tool streams.
- Emit provider-attempt telemetry for both success and failure.

### `litellm_config.yaml`

- split broad aliases into stable vs experimental pools
- re-enable pre-call checks
- lower retry count
- add backoff
- remove structurally bad providers from the hot path
- stop falling back from capability-specific aliases to `default`

### `docs/new/smart_router.py`

Treat it as documentation or sample code, not the implementation to optimize first. It is missing the important resilience work already present in the live router:

- shared HTTP client
- normalization
- repair logic
- training logs
- `tools_large` / `bench_large`

### `README.md` and docs

Update the docs so they match runtime reality:

- retries
- cooldowns
- cache TTL
- auto-routing behavior
- log paths
- public entrypoint (`4001` router vs `4000` LiteLLM)

## Success criteria

You will know this is working when:

- `413` rates drop sharply after re-enabling pre-call checks and removing GitHub from large-prompt paths
- `400` tool-history errors drop after adding a real tool-loop validator
- average provider attempts per successful tool request fall materially
- full-task success rate becomes more stable run-to-run
- recent log samples show fewer long retry chains before a successful provider is found

## Bottom line

The fastest win is to stop sending the hottest workload to the wrong provider pool.

If you do only four things now, do these:

1. shrink `tools` and `bench` to stable providers
2. re-enable pre-call checks
3. remove capability-destroying fallbacks to `default`
4. add a deterministic tool-loop validator for multi-turn conversations

That should buy more stability than spending time on better keyword classification alone.

## Sources

- Google Gemini API rate limits: https://ai.google.dev/gemini-api/docs/rate-limits
- Groq model and rate docs: https://console.groq.com/docs/models
- Groq tool-use docs: https://console.groq.com/docs/tool-use
- Cohere tool-use overview: https://docs.cohere.com/docs/tool-use-overview
- Cohere Chat API reference: https://docs.cohere.com/v2/reference/chat
- Mistral function calling docs: https://docs.mistral.ai/capabilities/function_calling/
- GitHub Models prototyping docs: https://docs.github.com/en/github-models/prototyping-with-ai-models
- GitHub Models usage docs: https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models
- OpenRouter limits: https://openrouter.ai/docs/api-reference/limits
- LiteLLM reliability docs: https://docs.litellm.ai/docs/proxy/reliability
- LiteLLM load-balancing / tag routing docs: https://docs.litellm.ai/docs/proxy/load_balancing
