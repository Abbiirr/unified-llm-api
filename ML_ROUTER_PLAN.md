# ML Router Training Plan

**Goal:** replace `smart_router.py`'s rule-based classifier with a learned model that picks the best alias per request in **≤2 ms on CPU**. Fast enough to run on every incoming request without becoming a bottleneck.

**Operating constraints:**
- Router adds <2 ms to request latency (tail budget: <5 ms p99)
- Runs on the same host as `smart_router.py` — no dedicated GPU
- Multi-objective: optimize success rate, latency, and free-tier-quota consumption simultaneously
- Must degrade gracefully — if model loads fail, fall back to current heuristic

---

## 1. Current state (2026-04-17 snapshot)

| Metric | Value | Verdict |
|---|---|---|
| Records in `routing.jsonl` | 3,123 | Phase-2 viable |
| Daily volume | ~780/day | Phase-3 in ~3 weeks |
| Success rate (status=200) | 81.9% | Good label diversity |
| Cache hits | 32.2% | **Poisons training — must exclude** |
| `has_tools=true` | 0.2% (7 rows) | **Broken detection — must fix** |
| `provider="unknown"` | 47% | **Broken attribution — must fix** |
| Top aliases | tools 31%, gemma4-26b-local 25%, tools_stable 10% | Severe imbalance, minority classes near-zero |
| Time span | 4 days | Too short for weekly cycles |

**Current logged fields (68):** `original_model`, `routed_alias`, `suggested_alias`, `route_reason`, 7× message-count fields, 8× content flags (`has_code_blocks`, `has_urls`, etc.), `detected_languages`, `primary_task`, `estimated_input_tokens`, `payload_chars`, `temperature`, `max_tokens`, `provider`, `served_model_name`, `status`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `finish_reason`, temporal (`hour_utc`, `day_of_week`), `provider_recent_error_rate`, `provider_recent_avg_latency_ms`.

---

## 2. Research synthesis

### 2.1 Architecture choices and their latency budgets

Tested inference latencies (single request, CPU, ~1K token prompt):

| Approach | Latency | Params | Accuracy ceiling | Notes |
|---|---|---|---|---|
| Pure rule-based (current) | **<0.1 ms** | 0 | low | deterministic, brittle |
| LightGBM on hand-features | **~1 ms** | ~10MB | medium | best cost/quality for us |
| MiniLM-L6-v2 + MLP head | **5–15 ms** | 22M | high | 384-dim embeddings |
| DistilBERT + classification head | 20–30 ms | 66M | high | overkill for us |
| ModernBERT-base + head | 30–50 ms (PyTorch) / **2–5 ms (Rust+Candle+ONNX)** | 150M | highest | vLLM Semantic Router approach |
| DeBERTa multi-head (NVIDIA) | ~50 ms | 184M | highest | produces rich task+complexity features |

**Our target:** two-stage hybrid.
- **Stage 1 (always):** LightGBM on hand-crafted features → ~1 ms. Handles 80%+ of obvious cases.
- **Stage 2 (only when stage 1 is low-confidence):** MiniLM embedding + small MLP → adds ~10 ms, used on <20% of traffic.

### 2.2 Canonical feature taxonomy from prior art

Synthesized from [NVIDIA prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier), [RouteLLM](https://arxiv.org/abs/2406.18665), [vLLM Semantic Router](https://vllm-semantic-router.com/), [LLMRank (2510.01234)](https://arxiv.org/html/2510.01234v1), [LLMRouterBench](https://www.emergentmind.com/topics/llmrouterbench).

**Three feature families:**

1. **Surface / structural** (cheap, deterministic, what we mostly have now):
   - length stats, tool presence, image presence, system prompt length, etc.

2. **Semantic / task-level** (from a small classifier, <10 ms):
   - `task_type` in 11 categories: Open QA, Closed QA, Summarization, Text Generation, Code Generation, Chatbot, Classification, Rewrite, Brainstorming, Extraction, Other (NVIDIA taxonomy)
   - 6 complexity dimensions: creativity, reasoning, constraints, domain_knowledge, contextual_knowledge, num_few_shots
   - `complexity_score` = weighted ensemble (0–1)

3. **Embeddings** (dense semantic, <5 ms with MiniLM):
   - 384-dim sentence-transformer embedding of user prompt
   - 384-dim embedding of system prompt
   - optional: 384-dim embedding of concatenated tool schemas

### 2.3 Label strategies (in order of signal strength)

From RouteLLM and Anyscale tutorials, binary success/fail labels are *insufficient* — a 1–5 quality score per (prompt, model) pair is required for serious training.

| Strategy | Collection | Cost | Quality |
|---|---|---|---|
| Status code (200/non-200) | free, automatic | 0 | low — "fast" ≠ "good" |
| Latency | free, automatic | 0 | low |
| LLM-judge 1–5 | offline batch via `gemma-4-26b-a4b-it:free` | <$5/10k samples | medium-high |
| Pairwise preference (2 models, same prompt) | runs each prompt through 2 aliases + judge | ~$20/5k pairs | high (RouteLLM-grade) |
| Human preference | manual annotation | expensive | gold |
| Implicit retry signal | free (infer from session) | 0 | low but unbiased |

**Our path:** log status+latency automatically now; add LLM-judge offline pipeline at Phase 3 entry.

### 2.4 Counterfactual exploration is not optional

From [Learning to Route LLMs from Bandit Feedback (2510.07429)](https://arxiv.org/html/2510.07429v1): without exploration, the learned router can only imitate the heuristic. To improve on it, we must see what *other* aliases would have done on the same prompt.

Two options:
- **Epsilon-greedy:** on 5–10% of non-critical traffic, pick a random eligible alias. Log the random assignment.
- **A/B shadow replay:** offline, replay a sample of prompts through 2–3 aliases and LLM-judge the outputs.

A/B shadow replay is safer in production. Epsilon-greedy produces cleaner data at the cost of occasional user-visible latency regressions.

---

## 3. Exact field schema (v2)

This is the schema every `routing.jsonl` row must have after the upgrades in section 7.

### 3.1 Request identity

| Field | Type | Why |
|---|---|---|
| `schema_version` | `"v2"` | Invalidates old rows on breaking change |
| `request_id` | string | Unique per request |
| `session_id` | string (hash of bearer-prefix + ip) | Link multi-turn conversations |
| `timestamp` | ISO8601 UTC | |
| `hour_utc` | int 0–23 | Cyclic feature |
| `day_of_week` | int 0–6 | Cyclic feature |
| `minute_of_day` | int 0–1439 | Finer temporal |

### 3.2 Request structure (surface features — all cheap)

| Field | Type | Current | Notes |
|---|---|---|---|
| `original_model` | string | ✓ | Alias the client asked for |
| `message_count` | int | ✓ | |
| `user_messages` / `assistant_messages` / `tool_messages` / `system_messages` | int | ✓ | |
| `user_text_length` | int | ✓ | chars |
| `total_conversation_chars` | int | ✓ | |
| `payload_chars` | int | ✓ | |
| `estimated_input_tokens` | int | ✓ | |
| `estimated_total_tokens` | int | ✓ | |
| `avg_message_length` | int | ✓ | |
| `max_message_length` | int | **new** | outlier messages matter |
| `has_images` | bool | ✓ | |
| `image_count` | int | **new** | |
| `has_tools` | bool | ✓ (broken) | **fix detection** |
| `tool_count` | int | ✓ | |
| `tool_names` | list[str] | ✓ | |
| `tool_has_nested_objects` | bool | ✓ | |
| `tool_rounds` | int | ✓ | prior tool-result messages |
| `tool_schema_chars` | int | **new** | size of tool definitions |
| `has_tool_choice` | bool | ✓ | |
| `has_system_prompt` | bool | ✓ | |
| `system_prompt_length` | int | ✓ | |
| `system_is_agent` | bool | ✓ | |
| `temperature` | float | ✓ | |
| `max_tokens` | int \| null | ✓ | |
| `top_p` | float \| null | **new** | |
| `stream` | bool | ✓ | streaming changes failure modes |
| `response_format` | enum `text`/`json_object`/`json_schema` | **new** | JSON requests route differently |

### 3.3 Content flags (surface, from regex/heuristics)

| Field | Type | Current |
|---|---|---|
| `has_code_blocks` | bool | ✓ |
| `code_block_count` | int | ✓ |
| `detected_languages` | list[str] | ✓ |
| `primary_language` | string \| null | ✓ |
| `has_urls` | bool | ✓ |
| `has_file_paths` | bool | ✓ |
| `has_json` | bool | ✓ |
| `has_error_trace` | bool | ✓ |
| `has_math` | bool | **new** | LaTeX/equations |
| `has_non_english` | bool | **new** | unicode block heuristic |

### 3.4 Semantic features (from task classifier, Phase 3+)

| Field | Type | Source |
|---|---|---|
| `task_type_1` | enum (11 values) | Task classifier head |
| `task_type_2` | enum (11 values) | |
| `task_type_prob` | float 0–1 | |
| `complexity_score` | float 0–1 | Weighted sum of 6 dims |
| `creativity` | float 0–1 | |
| `reasoning` | float 0–1 | |
| `constraints` | float 0–1 | |
| `domain_knowledge` | float 0–1 | |
| `contextual_knowledge` | float 0–1 | |
| `number_of_few_shots` | int | |

### 3.5 Embeddings (Phase 3+, logged as base64)

| Field | Type | Notes |
|---|---|---|
| `prompt_embedding_b64` | base64 of float16[384] | MiniLM-L6 of last user message |
| `system_embedding_b64` | base64 of float16[384] \| null | MiniLM-L6 of system prompt |
| `embedding_model_id` | string | e.g., `"sentence-transformers/all-MiniLM-L6-v2@1.0"` |

### 3.6 Routing decision

| Field | Type | Current | Notes |
|---|---|---|---|
| `suggested_alias` | string | ✓ | What heuristic proposed |
| `suggested_reason` | string | ✓ | Explanation |
| `routed_alias` | string | ✓ | What was actually dispatched |
| `route_reason` | string | ✓ | `"explicit"` / `"classify"` / `"bypass"` / `"random_exploration"` |
| `randomized_for_exploration` | bool | **new** | True if epsilon-greedy picked this |
| `learned_alias` | string \| null | **new** | What the learned router *would* have picked (shadow mode) |
| `learned_confidence` | float 0–1 \| null | **new** | Softmax max from learned model |

### 3.7 Outcome — first-attempt

| Field | Type | Current |
|---|---|---|
| `provider_base` | string | ✓ |
| `provider` | enum | ✓ (broken) — fix regex |
| `served_model` | string (hash) | ✓ |
| `served_model_name` | string | ✓ |
| `status` | int | ✓ |
| `latency_ms` | int | ✓ |
| `cache_hit` | bool | ✓ |
| `prompt_tokens` | int | ✓ |
| `completion_tokens` | int | ✓ |
| `total_tokens` | int | ✓ |
| `response_content_length` | int | ✓ |
| `response_has_tool_calls` | bool | ✓ |
| `response_tool_count` | int | ✓ |
| `finish_reason` | enum | ✓ |
| `error_category` | enum | ✓ |
| `error_msg` | string | ✓ |

### 3.8 Outcome — full rescue chain (NEW)

| Field | Type | Notes |
|---|---|---|
| `rescue_path` | list[RescueStep] | Empty if first attempt succeeded |
| `final_status` | int | Status after all rescues |
| `final_alias` | string | Alias that finally succeeded (or last attempted) |
| `total_latency_ms` | int | Includes rescue overhead |
| `rescue_count` | int | Number of rescue stages hit |

Where `RescueStep = { stage: string, alias: string, provider: string, status: int, latency_ms: int, error_msg: string }`.

### 3.9 Quality labels (NEW, mostly Phase 3+)

| Field | Type | Source | Populated |
|---|---|---|---|
| `is_training_sample` | bool | filter: `cache_hit=false AND status in (200, 429, 500, 504) AND provider != deprecated_provider` | log-time |
| `llm_judge_score` | int 1–5 \| null | offline batch via `gemma-4-26b-a4b-it` | Phase 3 batch |
| `llm_judge_model` | string \| null | | |
| `response_is_valid_json` | bool \| null | JSON.parse attempt when `response_format=json_object` | log-time |
| `response_tool_calls_valid` | bool \| null | JSON schema validation against tool defs | log-time |
| `response_truncated` | bool | `finish_reason == "length"` | log-time |
| `implicit_retry` | bool | True if next request from same session repeats this prompt within 2 min | offline job |

### 3.10 Provider context (NEW)

| Field | Type | Notes |
|---|---|---|
| `provider_recent_error_rate` | float 0–1 | ✓ existing |
| `provider_recent_avg_latency_ms` | int | ✓ existing |
| `provider_recent_request_count` | int | ✓ existing |
| `provider_cooldown_at_route` | bool | **new** | was provider in 60s cooldown when selected |
| `alias_candidate_count` | int | **new** | how many deployments were eligible for this alias |
| `provider_deprecated_at` | ISO8601 \| null | **new** | set retroactively when a model is retired |

---

## 4. Model architecture: the fast NN engine

### 4.1 Two-stage router

```
┌─────────────────────────────────────────────────────────┐
│ Incoming request                                         │
└───────────┬─────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 0: fast path (rule-based, <0.1 ms)                 │
│   if explicit model in request → passthrough            │
│   if has_images → vision                                │
│   if payload_chars > 200K → tools_large                 │
│   if private=true → local                               │
└───────────┬─────────────────────────────────────────────┘
            ▼ (80% of traffic exits here — was easy)
┌─────────────────────────────────────────────────────────┐
│ Stage 1: LightGBM classifier (~1 ms)                     │
│   Input: 60 hand-crafted features from §3.2–§3.3        │
│   Output: softmax over 20 aliases                       │
│   Confidence threshold: 0.7                             │
└───────────┬─────────────────────────────────────────────┘
            ▼ (if max-prob ≥ 0.7 → exit with prediction)
┌─────────────────────────────────────────────────────────┐
│ Stage 2: MiniLM + MLP (ambiguous cases, ~10 ms)          │
│   Input: MiniLM-L6 embedding (384d) + handcrafted       │
│          features → 3-layer MLP                         │
│   Output: softmax over 20 aliases + quality/latency     │
│           regression heads                              │
└─────────────────────────────────────────────────────────┘
```

**Latency budget per stage:**
- Stage 0: 0.1 ms (simple booleans)
- Stage 1: 1 ms (LightGBM predict on 60 features)
- Stage 2: 8–12 ms (MiniLM tokenize 10 tokens/ms, forward-pass ~5 ms on CPU, MLP ~1 ms)
- Total p50: ~1 ms (most traffic exits at stages 0 or 1)
- Total p99: ~15 ms (worst case goes through stage 2)

### 4.2 Why this shape and not a single big model

- A single ModernBERT over every request = 30–50 ms per request → with 20 RPS sustained that's 600–1000 ms of CPU occupied per second on routing alone. Unacceptable.
- Stage 0 handles most obvious routing (client sent `model=tools` explicitly) with zero ML.
- Stage 1 is a gradient-boosted model — state of the art for tabular features, <1 ms on CPU, trains on 5k samples, interpretable (SHAP values explain decisions).
- Stage 2 is invoked only when stage 1 is uncertain, typically <20% of traffic. Amortizes embedding cost.

### 4.3 Serving runtime

Two options:
1. **Python `lightgbm` + `sentence-transformers`** — simplest. ~1 ms LightGBM, ~8 ms MiniLM on CPU. In-process in `smart_router.py`.
2. **ONNX Runtime + Rust service** — `cargo` sidecar serving both models via ONNX. ~0.3 ms LightGBM, ~3 ms MiniLM on CPU. IPC adds ~0.5 ms. Total <5 ms p99. Overkill for v1, target for v2.

**Start with option 1.** Migrate to option 2 only if profiling shows router latency >5% of request latency.

### 4.4 Loss and training objective

Multi-objective loss (trained only in Phase 4+):

```
L = L_alias + λ_q · L_quality + λ_l · L_latency

where:
  L_alias    = cross-entropy(predicted_alias, oracle_alias)   # classification
  L_quality  = MSE(predicted_quality, observed_quality_label) # regression head
  L_latency  = MSE(predicted_latency, observed_latency)       # regression head
  λ_q = 0.3,  λ_l = 0.1   (tune via ablation)
```

Why regress latency and quality separately: at inference, policy can apply user-defined tradeoffs (`--cheap` mode picks low-latency-low-quality; `--best` picks highest-quality).

---

## 5. Training recipe (concrete, by phase)

### 5.1 Phase 1 — Replay the heuristic (validates data pipeline)

Needs ~1k rows. Already feasible.

```python
# tests/replay.py
import json, pandas as pd
from smart_router import _classify_request, _extract_features
rows = [json.loads(l) for l in open("logs/training/routing.jsonl")]
correct = 0
for r in rows:
    pred = _classify_request(r["original_model"], _extract_features(r))
    if pred == r["routed_alias"]:
        correct += 1
print(f"Heuristic replay accuracy: {correct/len(rows):.3f}")
```

If replay accuracy <95%, feature extraction is non-deterministic between log-time and replay-time. Fix before continuing.

### 5.2 Phase 2 — LightGBM on surface features (target: 5k rows, 2 weeks out)

```python
# train_v1.py
import json, lightgbm as lgb, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Load clean samples only
rows = [json.loads(l) for l in open("logs/training/clean/*.jsonl.gz")]
rows = [r for r in rows if r["is_training_sample"]]

FEATURES = [
    "user_text_length", "tool_count", "tool_rounds", "system_prompt_length",
    "estimated_total_tokens", "payload_chars", "max_tokens", "temperature",
    "top_p", "image_count", "code_block_count", "tool_schema_chars",
    "has_images", "has_tools", "has_code_blocks", "has_urls", "has_file_paths",
    "has_json", "has_error_trace", "has_math", "has_non_english",
    "has_tool_choice", "stream", "has_system_prompt", "system_is_agent",
    "hour_utc", "day_of_week",
]
CATEGORICAL = ["primary_task", "primary_language", "response_format"]
# one-hot CATEGORICAL → extra ~30 features

X = build_matrix(rows, FEATURES, CATEGORICAL)
le = LabelEncoder()
y = le.fit_transform([r["routed_alias"] for r in rows])

params = dict(
    objective="multiclass",
    num_class=len(le.classes_),
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    class_weight="balanced",   # critical: imbalanced aliases
)

# Stratified 5-fold CV for honest accuracy estimate
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for fold, (tr, va) in enumerate(kf.split(X, y)):
    model = lgb.train(params, lgb.Dataset(X[tr], y[tr]),
                      num_boost_round=500, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(30)])
    scores.append(model.best_score["valid_0"]["multi_logloss"])
print("CV logloss:", np.mean(scores))

# Final model on all data
final = lgb.train(params, lgb.Dataset(X, y), num_boost_round=int(model.best_iteration*1.1))
final.save_model("models/router_v1.lgb")

# Export label encoder + feature schema for inference
json.dump({"classes": le.classes_.tolist(), "features": FEATURES, "categorical": CATEGORICAL},
          open("models/router_v1.schema.json", "w"))
```

Inference contract used at serve-time:
```python
# smart_router.py (new code path)
import lightgbm as lgb, json
_model = lgb.Booster(model_file="models/router_v1.lgb")
_schema = json.load(open("models/router_v1.schema.json"))

def learned_route(features):
    x = build_row(features, _schema)
    probs = _model.predict(x.reshape(1, -1))[0]
    top = probs.argmax()
    return _schema["classes"][top], float(probs[top])
```

Success criteria:
- Replay accuracy ≥ heuristic
- p99 latency ≤ 3 ms
- Cross-validated logloss ≤ 1.0

### 5.3 Phase 3 — Embedding model + stage-2 fallback (target: 15k rows, 3 weeks out)

Prerequisite: task classifier running, embeddings being logged at request-time.

```python
# train_v2.py
import numpy as np, torch, torch.nn as nn
from datasets import Dataset
from sentence_transformers import SentenceTransformer

emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
rows = load_clean_rows()

# Build features: hand-crafted + 384d embedding + 384d system embedding
X_hand = build_hand_features(rows)       # [N, ~60]
X_emb  = np.stack([decode_b64(r["prompt_embedding_b64"]) for r in rows])  # [N, 384]
X_sys  = np.stack([decode_b64(r["system_embedding_b64"] or ZERO384) for r in rows])
X = np.concatenate([X_hand, X_emb, X_sys], axis=1)  # [N, ~828]

class Router(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
        )
        self.alias_head   = nn.Linear(128, n_classes)
        self.quality_head = nn.Linear(128, 1)   # 1–5 from LLM judge
        self.latency_head = nn.Linear(128, 1)   # log-ms
    def forward(self, x):
        h = self.mlp(x)
        return self.alias_head(h), self.quality_head(h), self.latency_head(h)

# Multi-objective training loop (standard PyTorch, not shown in full)
# Save as ONNX for 2–5× speedup at serving:
torch.onnx.export(model, dummy_input, "models/router_v2.onnx",
                  input_names=["features"], output_names=["alias","quality","latency"],
                  opset_version=17, dynamic_axes={"features":{0:"batch"}})
```

Serve via `onnxruntime`:
```python
import onnxruntime as ort
_sess = ort.InferenceSession("models/router_v2.onnx",
                             providers=["CPUExecutionProvider"])
alias_logits, q, lat = _sess.run(None, {"features": x.astype(np.float32)})
```

Success criteria:
- Stage-2 p99 latency ≤ 15 ms
- Stage-1 + stage-2 combined beats LightGBM-only on minority aliases (target: +5 pp macro-F1)
- Calibration ECE ≤ 5%

### 5.4 Phase 4 — Preference labels via LLM judge (target: 30k rows + 5k judged pairs)

Offline batch: pick N=5,000 prompts from `routing.jsonl`, replay each through 2–3 aliases, score each response with `gemma-4-26b-a4b-it:free` using:

```
You are judging which response better answers the user's request.
Score each response 1–5 where:
  5 = fully correct, complete, well-structured
  4 = correct, minor issues
  3 = partially correct
  2 = off-topic or wrong
  1 = unusable

Rubric: correctness, completeness, structure, tool-call validity (if applicable).
Return JSON: {"response_a_score": int, "response_b_score": int, "winner": "a"|"b"|"tie", "reason": str}
```

Build preference pairs, retrain stage-2 with pairwise loss (Bradley-Terry or directly as triplet):

```
L_pref = -log σ(score_winner - score_loser)
```

### 5.5 Phase 5 — Shadow deploy, then A/B, then cutover

```
Week 1: shadow mode — log learned_alias alongside routed_alias on 100% of traffic
Week 2: 10% A/B — if learned_alias ≠ routed_alias and learned_confidence > 0.8, serve learned
Week 3: evaluate — require ≥95% of heuristic's success rate AND ≥10% latency improvement
Week 4: 100% cutover — keep heuristic as fallback if model inference errors
```

---

## 6. Data quality fixes — do these FIRST

These are blockers. No training can start until they're done.

- [ ] **F1: Fix `provider=unknown` (47% → <10%).** In `smart_router.py`, `_infer_provider` uses a regex that misses new model IDs like `openai/gpt-oss-120b`. Audit the regex against all 322 entries in `MODEL_MAP`.
- [ ] **F2: Fix `has_tools` detection (0.2% → realistic).** Almost certainly `_extract_features` is checking the wrong field (e.g. looking for `tools` in the top-level request body when it's nested under a different key). Verify against a live tool-call request.
- [ ] **F3: Add `is_training_sample` filter.** One-liner in the logger: `row["is_training_sample"] = not row["cache_hit"] and row["status"] in (200, 429, 500, 503, 504) and row["error_category"] != "auth"`.
- [ ] **F4: Log full `rescue_path`.** Currently only first-attempt outcome is captured. Extend the rescue functions in `smart_router.py` to append to a list, dump on final response.
- [ ] **F5: `schema_version: "v2"`.** Add to every row. Training loaders filter by version.
- [ ] **F6: Daily log rotation + gzip.** `routing.jsonl` → `routing-YYYY-MM-DD.jsonl.gz`, kept 90 days.
- [ ] **F7: `session_id`.** Derive from `hash(bearer_prefix + client_ip)` or request header. Enables retry detection.
- [ ] **F8: Epsilon-greedy hook.** Env `ROUTER_EXPLORATION_RATE=0.0` by default. When non-zero, on that fraction of non-critical requests pick a random eligible alias; log `randomized_for_exploration=true`.

Order of work: F1 → F2 → F3 → F5 → F6 → F4 → F7 → F8. F1/F2 unblock everything else.

---

## 7. Data volume thresholds

| Phase | Threshold | ETA (at ~780 clean samples/day after F3 filter) |
|---|---|---|
| 1 — heuristic replay | 1k rows | ✅ today |
| 2 — LightGBM baseline | 5k rows, ≥200 per top-10 aliases | ~2 weeks |
| 3 — MiniLM + MLP | 15k rows, exploration data present | ~4 weeks |
| 4 — Preference router | 30k rows + 5k LLM-judge labels | ~2 months + $5–20 budget |
| 5 — Production cutover | A/B data passes gates (see §5.5) | ~3 months |

Minority-class remediation: from week 2 onward, route 10% of `default` traffic to underrepresented aliases (`swebench`, `tools_large`, `terminal_bench`) specifically to gather training examples. Tag those rows with `route_reason="minority_shaping"` and exclude from serving-quality metrics.

---

## 8. Evaluation harness

Required before any model ships. Runs on a held-out 20% slice chronologically (later half of available data).

Report per model candidate:

| Metric | Computation | Gate |
|---|---|---|
| Accuracy vs heuristic | agreement on `routed_alias` | ≥ baseline or justified drop |
| Macro-F1 | over all aliases | ≥ 0.6 |
| Expected latency | avg(`latency_ms`) under simulated policy | ≤ heuristic latency |
| Oracle regret | vs. post-hoc best alias | ≤ 15% |
| Tail p99 | `latency_ms` 99th pct | ≤ 2× heuristic's p99 |
| 429 rate preservation | frac(`status=429`) at client | ≤ heuristic's rate |
| Free-tier quota consumption | est. Cloudflare neurons / day | no regression |
| Calibration ECE | for stage-2 | ≤ 5% |
| Router inference p99 | synthetic load | ≤ 5 ms stage-1, ≤ 15 ms stage-2 |

---

## 9. Risks

- **Concept drift from model churn.** Today's Groq kimi deprecation just invalidated ~6 aliases' ground truth. Mitigation: `provider_deprecated_at` field, retrain monthly, weight recent data higher.
- **Reward hacking.** A single-objective router (latency-only) picks Ollama forever. Multi-objective loss in §4.4 is mandatory.
- **Feedback loop.** If learned router always picks Groq, we stop learning about Gemini. Mitigation: keep `ROUTER_EXPLORATION_RATE=0.05` in production indefinitely.
- **Training/serving skew.** Feature extraction at log-time must be bit-identical to inference-time. Mitigation: one `features.py` shared by both paths, unit-tested.
- **Privacy.** User text is logged. Before training-data export: truncate `user_text_preview` to 200 chars (current), do not log embeddings of sensitive data, gate `embedding_model_id` export by env var.
- **LLM-judge bias.** `gemma-4-26b-a4b-it` may systematically favor verbose answers. Mitigation: audit 100 judge decisions manually before trusting.
- **Model-load failures at serve-time.** If `router_v1.lgb` fails to load, silently falling back to heuristic is fine — but must be logged loudly.

---

## 10. First-day TODO when it's time to start

Pre-training (do now, before data collection continues another day):

1. Fix F1 (provider attribution regex)
2. Fix F2 (has_tools detection)
3. Add F3 (is_training_sample) and F5 (schema_version)
4. Add F6 (log rotation)

Training start (when 5k clean rows accumulate):

5. Write `tests/replay.py` (Phase 1 validation)
6. Write `train_v1.py` (Phase 2 LightGBM)
7. Export to `models/router_v1.lgb` and load in `smart_router.py` behind `ROUTER_MODE=shadow` env var
8. Shadow for 1 week, collect `learned_alias` vs `routed_alias` agreement
9. Flip to A/B when metrics in §5.5 gate is cleared

Everything after Phase 2 is best-effort — the LightGBM baseline alone should capture most of the available signal given the feature richness.

---

## References

- [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665) — LMSYS foundational paper
- [RouteLLM GitHub](https://github.com/lm-sys/RouteLLM)
- [NVIDIA LLM Router Blueprint](https://github.com/NVIDIA-AI-Blueprints/llm-router)
- [NVIDIA prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) — 11 task categories, 6 complexity dimensions
- [vLLM Semantic Router blog](https://blog.vllm.ai/2025/09/11/semantic-router.html) — Rust + Candle + ModernBERT production architecture
- [98× Faster LLM Routing Without a Dedicated GPU](https://arxiv.org/html/2603.12646v1) — compression + MiniLM tricks
- [Learning to Route LLMs from Bandit Feedback](https://arxiv.org/html/2510.07429v1) — counterfactual exploration strategies
- [LLMRank: Understanding LLM Strengths for Model Routing](https://arxiv.org/html/2510.01234v1) — feature taxonomy
- [Anyscale: Building an LLM Router](https://www.anyscale.com/blog/building-an-llm-router-for-high-quality-and-cost-effective-responses) — 1–5 scoring rubric
- [ModernBERT fine-tuning guide (Philipp Schmid)](https://www.philschmid.de/fine-tune-modern-bert-in-2025)
- [Sentence-Transformers all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 384-dim embeddings, 22M params
