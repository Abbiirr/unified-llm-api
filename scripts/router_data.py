"""
router_data.py — clean + label loader for routing.jsonl training logs.

Single source of truth for turning raw, noisy gateway telemetry into clean,
labeled routing examples. Used by analyze_routing_data.py, build_router_table.py,
and evaluate_router.py so every consumer applies identical cleaning rules.

Cleaning / labeling rules (see ML_ROUTER_PLAN.md §5.1):
  - drop cache hits        (not a real routing outcome)
  - drop auth errors       (infra/credential issue, not routing quality)
  - drop rows with no routed_alias or no status
  - label success = (final status == 200)

Labeling subtlety we deliberately accept: the gateway rescues failed routes
(429/408/5xx) via Ollama and cloud fallbacks, and the logged `status` already
reflects the *final, client-visible* outcome for non-image text traffic. We
therefore credit `routed_alias` (the alias the router CHOSE) with that final
status — that is exactly the quantity an auto-router wants to optimize:
"if I pick alias X, what does the client ultimately get?" Image requests skip
the rescue cascade, so their failures are genuine failures of that route.
"""

from __future__ import annotations

import json
import os
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import router_features as rf  # noqa: E402

SUCCESS_STATUS = 200
RATE_LIMIT_STATUS = 429
SERVER_ERROR_STATUSES = (500, 502, 503, 504)
TIMEOUT_STATUSES = (408, 504)
# Statuses worth training on (exclude pure client/auth noise).
TRAINABLE_STATUSES = {200, 408, 413, 422, 429, 500, 502, 503, 504}

DEFAULT_LOG_GLOB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "training", "routing.jsonl*")
)


def iter_raw_rows(paths=None):
    """Yield parsed JSON objects from routing.jsonl files (newest-first file order)."""
    if paths is None:
        paths = sorted(glob.glob(DEFAULT_LOG_GLOB))
    elif isinstance(paths, str):
        paths = [paths]
    for path in paths:
        try:
            f = open(path, encoding="utf-8")
        except OSError:
            continue
        with f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue


def clean_row(o: dict) -> dict | None:
    """Return a cleaned, labeled example, or None if the row should be dropped."""
    if o.get("cache_hit"):
        return None
    if o.get("error_category") == "auth":
        return None
    alias = o.get("routed_alias")
    if not alias:
        return None
    status = o.get("status")
    if not isinstance(status, int):
        return None
    if status not in TRAINABLE_STATUSES:
        return None

    lat = o.get("latency_ms")
    lat = int(lat) if isinstance(lat, (int, float)) and lat > 0 else None

    return {
        "alias": alias,
        "status": status,
        "success": status == SUCCESS_STATUS,
        "rate_limited": status == RATE_LIMIT_STATUS,
        "server_error": status in SERVER_ERROR_STATUSES,
        "latency_ms": lat,
        "timestamp": o.get("timestamp", ""),
        # request-class keys (specific -> general) via the shared module
        "bucket_keys": rf.bucket_keys(o),
        "class_key": rf.class_key(o),
        "modality": rf.modality(o),
        "size_bucket": rf.size_bucket(o),
        "content_bucket": rf.content_bucket(o),
        "suggested_alias": o.get("suggested_alias"),
        "provider": o.get("provider", "unknown"),
        "has_images": rf._as_bool(o.get("has_images")),
    }


def load_clean(paths=None, limit=None):
    """Load all cleaned rows. Optionally cap at `limit` (most recent files first)."""
    out = []
    for o in iter_raw_rows(paths):
        c = clean_row(o)
        if c is not None:
            out.append(c)
            if limit and len(out) >= limit:
                break
    return out


# ── Curated candidate pool the auto-router may pick among ──
# General-purpose aliases only. The quota-hammered batch-OCR aliases
# (tools_cloud / tools_stable_cloud / default_cloud) and single-provider _free
# pools are intentionally excluded — the rescue cascade already covers
# cloud-only fallback when Ollama is down.
AUTO_POOL_TEXT = ["fast", "default", "coding", "thinking", "big",
                  "bench", "swebench", "tools_large"]
AUTO_POOL_TOOLS = ["coding", "tools", "tools_large", "tools_stable",
                   "swebench", "bench", "default"]
AUTO_POOL_IMAGE = ["vision", "ocr"]


def candidate_pool(modality: str) -> list[str]:
    if modality == "image":
        return list(AUTO_POOL_IMAGE)
    if modality == "tools":
        return list(AUTO_POOL_TOOLS)
    return list(AUTO_POOL_TEXT)


if __name__ == "__main__":
    rows = load_clean()
    print(f"loaded {len(rows)} clean rows")
    if rows:
        print("first:", json.dumps(rows[0], indent=2)[:600])
