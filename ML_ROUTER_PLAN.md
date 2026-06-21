# ML Router Plan

**Updated:** 2026-06-01

---

## ✅ Implementation status (2026-06-21) — v1 shipped

A first learned auto-router is **built and deployed in `learned` mode**. It is a
pure-stdlib empirical/Empirical-Bayes utility table (no LightGBM/sklearn — those
aren't installed and native deps on the live gateway were deemed too risky),
which is the §6.1 "fallback" tier done first for safety. Heuristic-as-prior +
softmax exploration + capability/health gating wrap it.

| Plan item | Status |
|---|---|
| `router_features.py` (shared train/serve features, §4/§6.1) | ✅ done |
| Schema v2 telemetry (`schema_version`,`request_id`,`final_alias`,`final_status`,`learned_*`,`cost_bias`, §4.1) | ✅ done (logged now) |
| `final_status`/`final_alias` after rescue (§4.2 #4) | ✅ done |
| `scripts/analyze_routing_data.py`, `scripts/build_router_table.py` (§5) | ✅ done |
| `router_policy.py` runtime scorer + safe fallback (§6.1) | ✅ done (<25µs/call, well under 2ms gate) |
| `ROUTER_MODE=heuristic\|shadow\|learned` (§9 Phase 2) | ✅ done (deployed=learned) |
| Cost/quality dial (`ROUTER_COST_BIAS`, OpenRouter-style) | ✅ done |
| Exploration for counterfactuals (§7.2) | ✅ done (`ROUTER_EXPLORATION` softmax) |
| `scripts/evaluate_router.py` chronological holdout (§8) | ✅ done — learned **+18pt** success vs heuristic on covered held-out classes |
| `scripts/router_feedback.py` online scorecard | ✅ done |
| Provider attribution <10% unknown (§4.2 #1) | ⛔ still ~65% unknown — provider-aware learning deferred to v2 |
| LightGBM/ranker, LLM-judge preference labels (§6.1.1, §6.2, Phase 4) | ⛔ v2 |
| Semantic/embedding fallback (§6.2) | ⛔ v2 |

**Headline data finding:** conditioned on cleaned client-visible outcomes,
`coding`/`tools_large`/`big`/`thinking` deliver 88–99% success at low latency,
while the heuristic's `fast` route is only ~66% and slow — so the learned router
chiefly fixes "short/simple → fast" and "tools → tools_large". See
`models/router_v1.*.json` and `logs/training/reports/`.

---

**Goal:** add a learned runtime router that chooses the best gateway alias per request while adding **<=2 ms p99** on the normal hot path. The current heuristic remains the fallback whenever the model is missing, low-confidence, or unhealthy.

This plan is based on the current local logs in `logs/training/`, the current `smart_router.py` telemetry path, and current routing research. The main conclusion changed from the older April snapshot: **we now have enough raw volume for a v1 router, but only after fixing telemetry quality.**

---

## 1. Current Data Inventory

Parsed files on 2026-06-01:

| File | Rows | Date range |
|---|---:|---|
| `logs/training/routing.jsonl.4` | 88,349 | 2026-03-15 to 2026-04-14 |
| `logs/training/routing.jsonl.3` | 58,544 | 2026-04-14 to 2026-04-20 |
| `logs/training/routing.jsonl.2` | 53,411 | 2026-04-20 to 2026-05-04 |
| `logs/training/routing.jsonl.1` | 56,723 | 2026-05-04 to 2026-05-24 |
| `logs/training/routing.jsonl` | 3,383 | 2026-05-24 to 2026-06-01 |
| **Total** | **260,409** | **2026-03-15 to 2026-06-01** |

Additional local data:

| Artifact | Size / Count | Use |
|---|---:|---|
| `logs/training/conversations/convos.jsonl*` | ~2.0 GB | Offline judge/replay candidates only, not hot-path features |
| `logs/litellm/gateway.log` | ~8.8 GB | Provider diagnostics and rescue-chain reconstruction |

### 1.1 Dataset Quality

| Metric | Current value | Training impact |
|---|---:|---|
| Raw rows | 260,409 | Enough for v1 |
| Status 200 | 154,279 / 59.2% | Good success/failure diversity, but many failures are provider/outage behavior |
| Cache hits | 19,515 / 7.5% | Exclude from training labels |
| `has_tools=true` | 104,354 / 40.1% overall | Historically useful |
| Active log `has_tools=true` | 0 / 3,383 | Blocker: active telemetry is broken |
| `has_images=true` | 15,174 / 5.8% | Enough for OCR/vision routing signals |
| `provider=unknown` or missing | 183,479 / 70.5% | Biggest blocker for provider-aware training |
| Estimated trainable rows now | 68,212 / 26.2% | Enough for v1 alias model after cleanup |
| `schema_version` | 0% | Must add before new collection |
| `is_training_sample` | 0% | Must add before clean loaders |
| `rescue_path` | 0% | Cannot learn rescue effectiveness yet |
| `session_id` | 0% | Cannot infer implicit retries yet |
| `response_format` | 0% | Cannot learn JSON-schema routing yet |

### 1.2 Current Alias Distribution

Top routed aliases in raw logs:

| Alias | Rows |
|---|---:|
| `swebench` | 43,778 |
| `spec-rag` | 41,618 |
| `tools` | 26,337 |
| `tools_stable_cloud` | 20,338 |
| `<null>` | 18,300 |
| `coding` | 16,297 |
| `tools_cloud` | 10,713 |
| `lighton-ocr` | 10,299 |
| `default_cloud` | 7,156 |
| `qwopus-4b` | 6,593 |
| `fast` | 6,476 |
| `terminal_bench` | 6,471 |

Train v1 on **aliases**, not individual provider deployments. Provider-level choice is still mostly hidden behind LiteLLM, and the provider field is too incomplete today.

### 1.3 Latency Signals From Clean Successful Rows

Non-cache, provider-known, status-200 latency examples:

| Alias | n | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| `coding` | 14,263 | 739 ms | 4,527 ms | 37,600 ms |
| `terminal_bench` | 5,202 | 1,885 ms | 11,875 ms | 25,435 ms |
| `tools_stable_cloud` | 5,752 | 2,341 ms | 29,628 ms | 135,525 ms |
| `fast` | 2,825 | 2,664 ms | 34,326 ms | 124,322 ms |
| `tools` | 14,891 | 6,161 ms | 73,132 ms | 249,641 ms |
| `swebench` | 3,767 | 22,161 ms | 154,888 ms | 303,932 ms |
| `qwopus-4b` | 1,330 | 223,055 ms | 292,889 ms | 299,003 ms |

These numbers support a cost/latency-aware policy rather than pure success classification.

---

## 2. Research Summary That Changes The Plan

Use these research points as design constraints:

- RouteLLM trains efficient routers from preference data to balance response quality and cost, and reports large cost reductions without quality loss. For this gateway, status/latency labels are useful for v1, but preference or judge labels are needed before quality-sensitive cutover. Source: https://arxiv.org/abs/2406.18665
- vLLM Semantic Router uses a lightweight semantic classifier, currently ModernBERT, plus routing rules around fast/reasoning paths, and emphasizes latency, token, and tool-catalog control. That validates a hybrid design: deterministic gates first, learned scoring second, semantic model only when needed. Source: https://vllm.ai/blog/2025-09-11-semantic-router
- Not Diamond RoRF shows a production-practical random-forest router over embeddings, with pairwise model scores and tunable thresholds. Its training format requires prompt input plus per-model score columns. That is a good template for our offline judge/replay dataset, but online embeddings should wait until v2 because of hot-path latency. Source: https://github.com/Not-Diamond/RoRF
- Bandit-feedback routing work points out the deployment reality: you only observe the model you chose. Therefore this gateway must add epsilon-greedy exploration or offline shadow replay, otherwise a learned router just imitates today's heuristic and its outages. Source: https://arxiv.org/abs/2510.07429
- RouterBench-style evaluation treats routing as a cost/quality frontier problem, not only classification accuracy. Our evaluation must report success, latency, and quota/cost regret together. Source: https://arxiv.org/abs/2403.12031

---

## 3. Target Runtime Architecture

### 3.1 Decision Flow

```text
incoming /v1/chat/completions
  |
  v
Stage 0: deterministic safety gates (<0.1 ms)
  - explicit non-auto model: pass through, but shadow-score it
  - image payload: only vision/OCR-capable aliases
  - local/privacy alias: only local-capable aliases
  - huge context: only large-context aliases
  - no healthy candidates: current heuristic + fail-fast rules
  |
  v
Stage 1: candidate set builder (<0.2 ms)
  - load alias capabilities from litellm_config.yaml at startup
  - filter by tools, vision, context, JSON mode, local-only, cooldown, circuit breaker
  - keep top candidate aliases, not provider deployments
  |
  v
Stage 2: learned tabular scorer (target <=1.5 ms p99)
  - LightGBM multiclass/ranker, or sklearn HistGradientBoosting fallback
  - input: schema-v2 surface features + rolling provider health + alias capability features
  - output: utility score per candidate alias
  |
  v
policy layer (<0.1 ms)
  - pick max utility unless confidence below threshold
  - low confidence: use current heuristic
  - optional exploration when ROUTER_EXPLORATION_RATE > 0
```

### 3.2 Utility, Not Just Classification

For each candidate alias `a`, predict:

```text
success_p(a), log_latency_ms(a), error_p429(a), error_p5xx(a), quality_proxy(a)

utility(a) =
    1.00 * success_p(a)
  + 0.30 * quality_proxy(a)
  - 0.15 * normalized_log_latency(a)
  - 0.25 * error_p429(a)
  - 0.20 * error_p5xx(a)
  - quota_penalty(a)
```

For v1, `quality_proxy` is weak:

```text
quality_proxy =
  + response_tool_calls_valid when tools are required
  + response_is_valid_json when JSON is required
  - response_truncated
  - implicit_retry
```

For v2, replace or augment this with offline LLM-judge scores.

### 3.3 Why Not A Neural Router First

A neural/embedding router is valuable later, but not first:

- Current hot-path target is <=2 ms p99. Any request-time embedding model risks exceeding that budget.
- The current feature schema already captures strong routing signals: tools, images, prompt size, code/error/math flags, system-agent markers, alias, status, latency, and rolling provider health.
- Current telemetry has provider and tool-detection defects. A neural model would learn those defects faster than it learns good routing.

So v1 is a tabular model. v2 can add async/offline embeddings and a semantic stage only for low-confidence traffic.

---

## 4. Schema v2: Add Before More Collection

Implement this in `smart_router.py`, but move feature extraction into `router_features.py` so training and serving cannot drift.

### 4.1 Required New Fields

| Field | Type | Reason |
|---|---|---|
| `schema_version` | string, `"v2"` | Cleanly separates old broken rows from new rows |
| `request_id` | string | Join routing, conversation, and LiteLLM logs |
| `session_id` | hash string | Enables implicit retry detection |
| `minute_of_day` | int | Better temporal rate-limit feature than minute-only |
| `max_message_length` | int | Long single messages behave differently from many short messages |
| `image_count` | int | Vision/OCR routing |
| `tool_schema_chars` | int | Tool-catalog bloat impacts accuracy and provider failures |
| `top_p` | float/null | Sampling setting |
| `response_format` | `text/json_object/json_schema` | JSON routing and validation |
| `has_math` | bool | Reasoning/routing signal |
| `has_non_english` | bool | Provider/model capability signal |
| `is_training_sample` | bool | Loader can filter without re-deriving |
| `final_status` | int | Outcome after rescue |
| `final_alias` | string | Alias that actually returned |
| `total_latency_ms` | int | Full user-visible latency |
| `rescue_path` | list | Learn which rescues work |
| `provider_cooldown_at_route` | bool | Important confounder |
| `alias_candidate_count` | int | Candidate availability feature |
| `learned_alias` | string/null | Shadow-mode prediction |
| `learned_confidence` | float/null | Shadow-mode confidence |
| `randomized_for_exploration` | bool | Exclude/weight exploration separately |

### 4.2 Fix Existing Fields First

Blockers:

1. **Provider attribution:** `provider=unknown/missing` is 70.5%. Fix provider inference from both `provider_base` and LiteLLM model identity. Rows with no provider are still usable for alias imitation, but not for provider health or provider-aware cost learning.
2. **Tool detection:** active log has `has_tools=false` for every row since 2026-05-24. Verify against a live tool-call request and the current request body shape. This field is too important to train around.
3. **Null routed aliases:** 18,300 rows have `routed_alias=null`, mostly older explicit-pass rows. v2 must always log both `requested_alias` and `final_alias`.
4. **Outcome after rescue:** current rows log first-attempt-ish status, but not full rescue sequence. That mislabels cases where a bad first route was rescued successfully.

---

## 5. Training Dataset Construction

Create `scripts/analyze_routing_data.py` and `scripts/build_router_dataset.py`.

### 5.1 Clean Filters

Use rows when:

```python
row["schema_version"] == "v2"
and row["is_training_sample"] is True
and row["cache_hit"] is False
and row["final_status"] in (200, 400, 408, 413, 422, 429, 500, 502, 503, 504)
and row["error_category"] != "auth"
and row["final_alias"] in PUBLIC_ALIASES
```

For old rows, build a legacy dataset only for baseline experiments:

```python
not row.get("cache_hit")
and row.get("status") in (200, 429, 500, 503, 504)
and row.get("provider") not in ("", "unknown", None)
and (row.get("routed_alias") or "") in PUBLIC_ALIASES
```

Do not mix old and v2 rows in final training without a `schema_version` feature.

### 5.2 Labels

For v1:

- `success_label`: `final_status == 200`
- `latency_label`: `log1p(total_latency_ms)`
- `rate_limit_label`: `final_status == 429`
- `server_error_label`: `final_status in (500, 502, 503, 504)`
- `selected_alias`: `final_alias`
- `heuristic_alias`: current `suggested_alias`

For v2:

- `llm_judge_score`: 1 to 5 from offline replay
- `pairwise_winner`: alias A, alias B, or tie
- `implicit_retry`: same session repeats similar prompt within 2 minutes

---

## 6. Model Plan

### 6.1 V1: Tabular Alias Scorer

Train one of these, in this order:

1. **LightGBM LambdaRank/ranker** if LightGBM is accepted as a dependency.
2. **LightGBM multiclass classifier + latency/error regressors** if ranking setup is too slow to build.
3. **sklearn HistGradientBoosting** fallback if avoiding native dependencies matters more than accuracy.

Features:

- Request structure: message counts, role counts, user/system lengths, max message length, total chars, estimated tokens.
- Tool features: `has_tools`, `tool_count`, `tool_schema_chars`, nested schemas, prior tool rounds, tool choice.
- Content flags: code block count, primary language, URLs, file paths, JSON, error trace, math, non-English.
- Mode flags: stream, response format, temperature, max tokens, top_p.
- Capability flags per candidate alias: supports tools, supports vision, supports large context, local/cloud, current candidate count.
- Health features: provider recent error rate, avg latency, cooldown state.
- Temporal features: hour sin/cos, day-of-week sin/cos, minute-of-day sin/cos.

Artifacts:

| Artifact | Purpose |
|---|---|
| `models/router_v1.model` | trained model |
| `models/router_v1.schema.json` | feature order, categorical maps, alias classes |
| `models/router_v1.metrics.json` | evaluation report and gates |
| `router_features.py` | shared feature extraction for logging, training, and serving |
| `router_model.py` | load model, score candidates, fallback safely |

Runtime contract:

```python
decision = learned_router.route(features, candidates)
if not decision.ok or decision.confidence < ROUTER_MIN_CONFIDENCE:
    alias = heuristic_alias
else:
    alias = decision.alias
```

### 6.2 V2: Semantic Fallback For Ambiguous Requests

Only after v1 is stable:

- Precompute prompt embeddings offline for training.
- Do **not** compute embeddings for every request on day one.
- Add an optional low-confidence path using MiniLM or ModernBERT only when Stage 2 confidence is below threshold and `ROUTER_SEMANTIC_FALLBACK=1`.
- Measure router p99 separately. If semantic fallback pushes router p99 over 15 ms, keep it shadow-only.

---

## 7. Exploration And Counterfactual Data

We need counterfactuals because current logs only show the chosen alias.

### 7.1 Safer First Step: Offline Shadow Replay

Nightly job:

1. Sample 500 to 2,000 requests from clean non-sensitive v2 rows.
2. Replay each prompt through 2 or 3 eligible aliases.
3. Save model outputs, latency, status, tokens, and tool/JSON validity.
4. Judge a subset with a cheap reliable judge model.
5. Store as `logs/training/replay/replay-YYYY-MM-DD.jsonl.gz`.

This gives pairwise labels without user-visible regressions.

### 7.2 Production Exploration Later

Add:

```text
ROUTER_EXPLORATION_RATE=0.00
ROUTER_EXPLORATION_MAX_LATENCY_CLASS=normal
ROUTER_EXPLORATION_EXCLUDE_ALIASES=local,ocr,vision
```

When enabled, explore only non-critical requests:

- no images
- no explicit local/privacy alias
- no huge context
- no JSON-schema hard requirement
- no currently degraded provider set

Log `randomized_for_exploration=true` and exclude these rows from normal serving-quality dashboards.

---

## 8. Evaluation Gates

Run `scripts/evaluate_router_policy.py` on a chronological holdout split. Do not use random split as the primary gate because provider/model behavior drifts over time.

| Metric | Gate for shadow | Gate for serving 10% |
|---|---:|---:|
| Router inference p99 | <=2 ms | <=2 ms |
| Status-200 rate vs heuristic | >=98% of heuristic | >=100% of heuristic |
| 429 rate vs heuristic | no worse | lower or no worse |
| p50 latency | no worse | >=10% better |
| p95 latency | no worse | >=10% better |
| Macro-F1 vs known good aliases | report only | improves minority aliases |
| Calibration ECE | <=10% | <=5% |
| Tool-call valid rate | no worse | no worse |
| JSON valid rate | no worse | no worse |
| Fallback rate to heuristic | <=30% | <=15% |

Also report:

- Alias distribution before/after policy.
- Provider distribution before/after policy.
- Quota/cost estimate per provider.
- Worst 20 regressions with request ids.
- SHAP or feature-importance summary for v1.

---

## 9. Rollout Plan

### Phase 0: Telemetry Repair

Target: 1 to 2 days.

- [ ] Add `router_features.py` and make `smart_router.py` use it.
- [ ] Fix provider attribution to get unknown/missing below 10% for new rows.
- [ ] Fix active `has_tools` detection and validate with a live tool-call request.
- [ ] Add `schema_version`, `is_training_sample`, `request_id`, `session_id`, `response_format`.
- [ ] Log `final_status`, `final_alias`, `total_latency_ms`, and `rescue_path`.
- [ ] Add daily gzip rotation for routing logs.

Exit gate: 24 hours of v2 logs with provider unknown below 10%, realistic tool rate, and no JSONL parse failures.

### Phase 1: Baseline Data And Replay

Target: 1 to 2 days after Phase 0.

- [ ] Write `scripts/analyze_routing_data.py`.
- [ ] Write `scripts/build_router_dataset.py`.
- [ ] Write heuristic replay test: feature extraction from logs should reproduce `suggested_alias` with >=95% agreement.
- [ ] Build legacy baseline from the current 68,212 estimated trainable rows, but mark it as exploratory.

Exit gate: reproducible dataset artifact and a metrics JSON checked into `models/` or stored under `logs/training/reports/`.

### Phase 2: V1 Model In Shadow Mode

Target: 2 to 4 days after enough v2 rows exist.

- [ ] Train tabular router.
- [ ] Add `router_model.py`.
- [ ] Add `ROUTER_MODE=heuristic|shadow|learned`.
- [ ] In shadow mode, log `learned_alias`, `learned_confidence`, and `learned_utility`.
- [ ] Add tests proving model-load failure falls back to heuristic.

Exit gate: shadow metrics meet evaluation gates for 3 consecutive days.

### Phase 3: 10% Learned Routing

Target: after shadow gate.

- [ ] Enable learned routing for 10% of eligible auto/default requests.
- [ ] Keep explicit aliases pass-through.
- [ ] Auto-disable learned routing if status-200 rate drops, 429 rate rises, or p95 latency regresses.
- [ ] Continue logging heuristic and learned decisions side-by-side.

Exit gate: 7 days with no regression and at least 10% latency improvement on eligible traffic.

### Phase 4: Preference Labels And V2

Target: after v1 is stable.

- [ ] Build offline replay set from v2 rows.
- [ ] Judge 5,000 to 10,000 pairwise outputs.
- [ ] Train pairwise/ranking model with judged quality labels.
- [ ] Add semantic fallback only for low-confidence cases.

Exit gate: quality-sensitive replay beats v1 and heuristic on heldout judged pairs.

---

## 10. Immediate TODO

Do these before training anything serious:

1. Fix active `has_tools` logging.
2. Fix provider attribution for new rows.
3. Add schema v2 and `is_training_sample`.
4. Add final outcome and rescue-path logging.
5. Add analysis/build scripts and freeze a legacy baseline report.

The raw data volume is no longer the bottleneck. The bottleneck is label correctness.

---

## References

- RouteLLM: https://arxiv.org/abs/2406.18665
- vLLM Semantic Router blog: https://vllm.ai/blog/2025-09-11-semantic-router
- vLLM Semantic Router repository: https://github.com/vllm-project/semantic-router
- Not Diamond RoRF: https://github.com/Not-Diamond/RoRF
- Bandit-feedback routing: https://arxiv.org/abs/2510.07429
- RouterBench: https://arxiv.org/abs/2403.12031
