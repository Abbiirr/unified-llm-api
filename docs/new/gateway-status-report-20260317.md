# Gateway Status Report — 2026-03-17

## Current Health

| Service | Status |
|---|---|
| LiteLLM (4002) | Healthy |
| Smart Router (4000) | Healthy |
| Redis (6379) | PONG |

## Cumulative Stats (all training data)

| Metric | Value |
|---|---|
| Total requests | 13,144 |
| Success rate | **97.8%** |
| Training samples | 13,144 |
| 200 OK | 12,852 |
| 400 Bad Request | 143 |
| 413 Too Large | 74 |
| 429 Rate Limited | 37 |
| 500 Internal | 26 |
| 401 Auth Error | 12 |

## Since Last Restart (P0 fixes applied)

| Metric | Value |
|---|---|
| 200 OK | 2,530 |
| 400s | **0** (was 143 before P0) |
| 413s | **0** (was 74 before P0) |
| 429s | 15 |
| 500s | 6 |

**The P0 fixes eliminated 400s and 413s entirely.** Remaining issues are 429 rate limits and 500s from cooldown exhaustion.

## Traffic Pattern

98.4% of all requests go to the `tools` alias. This is a tool-heavy, multi-turn agentic workload (SWE-bench benchmarks). The router should be optimized primarily for this use case.

## Remaining Issues

### Issue 1: 80,000 repairs — orphan tool results dominating

The `repair_messages()` function has fired 80,501 times:
- 79,786 orphan tool result removals
- 712 conversation history fixes
- 3 null content fixes

**Analysis**: The massive orphan count (79K) suggests the benchmark agent is consistently sending malformed conversation histories with tool results that don't match any tool_call. This is agent-side, but the repair function is doing its job — without it, these would all be 500s.

**Recommendation**: The agent should fix its conversation history management. The gateway repair is a valid safety net but shouldn't be the primary path.

### Issue 2: 500s from cooldown exhaustion (6 incidents)

All 6 500s occurred when every provider in the alias was simultaneously cooled down from rate limits. The cooldown state shows providers being cooled for up to 86,400s (24 hours) — this is LiteLLM's own dynamic cooldown, not our configured 300s.

**Root cause**: LiteLLM calculates cooldown from the provider's `Retry-After` header. Groq and Cerebras both return long retry-after values when daily quotas are exhausted. When all 3 providers in `tools` (Cerebras + 2x Groq) hit daily limits simultaneously, there's no provider left.

**Recommendation**:
- Add NVIDIA NIM back to `tools_stable` as a fallback (it works for single tool calls)
- Or add OpenRouter free models to `tools_stable` as last resort
- Long-term: conversation pinning would reduce the number of provider switches and therefore rate limit hits

### Issue 3: 429s reaching clients (15 + 37 cumulative)

Most 429s are from:
- Gemini 2.5 Pro daily quota (in `default` alias)
- OpenRouter free models upstream rate limits
- Groq per-minute limits during sustained tool conversations

The `default` alias still has Gemini 2.5 Pro which exhausts daily. It's not in `tools` anymore but clients hitting `default` still see it.

**Recommendation**: Move Gemini 2.5 Pro lower in the `default` alias priority, or remove it entirely and keep only Flash/Flash-Lite.

### Issue 4: Router timeouts (600s)

9 router-level timeouts at exactly 600s. These are requests where LiteLLM tried all providers, all were either cooled down or slow, and the router's 600s httpx timeout expired.

**Recommendation**: This is a consequence of Issue 2. Fixing the provider pool exhaustion fixes this.

### Issue 5: 401 auth errors on `fast` (12 incidents)

12 requests to the `fast` alias got 401 "No api key passed in." This means the smart router forwarded a request without the Authorization header, or a client sent without auth and the router passed it through.

**Recommendation**: Add auth validation in the smart router before forwarding to LiteLLM.

## What's Working Well

1. **P0 fixes eliminated 400s and 413s** — zero since restart
2. **`tools` alias is clean** — only Cerebras + Groq, no incompatible providers
3. **`repair_messages()` prevented 80K potential 500s** — massive impact
4. **Pre-call checks** — auto-skipping small-context models
5. **Capability-preserving fallbacks** — tools → tools_stable → default
6. **31/31 smoke tests passing**

## Improvement Priorities

### P0 (do now)
- [x] All done — 400s and 413s eliminated

### P1 (next session)
1. Add NVIDIA NIM `gpt-oss-120b` back to `tools_stable` for when Cerebras+Groq are both cooled down
2. Add auth validation in smart router
3. Remove/demote Gemini 2.5 Pro from `default`

### P2 (future)
1. Conversation pinning — keep multi-turn tool conversations on same provider
2. Train ML classifier on 13K+ training samples
3. Provider health dashboard endpoint
4. Smart cooldown — distinguish daily quota exhaustion from per-minute limits
5. Agent-side conversation history fix to eliminate orphan tool results at source
