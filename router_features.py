"""
router_features.py — canonical request featurization + bucketing for the auto-router.

Shared by BOTH the offline table builder (scripts/build_router_table.py) and the
online scorer (router_policy.py) so training and serving can never drift.

The two callers feed different-shaped dicts:
  - offline: a row parsed from logs/training/routing.jsonl
  - online:  the `training_features` dict smart_router builds per live request

Both dicts share the same field names (has_images, has_tools, tool_count,
estimated_total_tokens, payload_chars, has_code_blocks, has_error_trace,
has_json, primary_language, user_text_length, ...), so a single set of pure
functions over that dict works for both. No I/O, no config, no network — keep
this module dependency-free and <0.05 ms so it is safe on the hot path.
"""

from __future__ import annotations

# ── Size buckets (by estimated total conversation tokens) ──
# Chosen from the observed distribution: most chat is <800 tok; OCR/agent
# payloads run large. Boundaries are deliberately coarse so cells stay dense.
_SIZE_BOUNDS = (
    (150, "xs"),
    (800, "s"),
    (3000, "m"),
    (12000, "l"),
)  # anything >= 12000 -> "xl"


def _as_bool(v) -> bool:
    return bool(v) and v not in (0, "0", "false", "False", None)


def estimated_tokens(feat: dict) -> int:
    """Best-effort total token estimate that works on both log rows and live features."""
    for k in ("estimated_total_tokens", "estimated_input_tokens"):
        v = feat.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    pc = feat.get("payload_chars") or feat.get("total_conversation_chars")
    if isinstance(pc, (int, float)) and pc > 0:
        return int(pc) // 4
    utl = feat.get("user_text_length")
    if isinstance(utl, (int, float)) and utl > 0:
        return int(utl) // 4
    return 0


def modality(feat: dict) -> str:
    """image | tools | text — the hard capability axis."""
    if _as_bool(feat.get("has_images")):
        return "image"
    if _as_bool(feat.get("has_tools")) or (feat.get("tool_count") or 0) > 0:
        return "tools"
    return "text"


def size_bucket(feat: dict) -> str:
    tok = estimated_tokens(feat)
    for bound, name in _SIZE_BOUNDS:
        if tok < bound:
            return name
    return "xl"


def content_bucket(feat: dict) -> str:
    """Coarse content type that correlates with which model class does well."""
    if _as_bool(feat.get("has_code_blocks")) or feat.get("primary_language"):
        return "code"
    if _as_bool(feat.get("has_error_trace")):
        return "error"
    if _as_bool(feat.get("has_json")):
        return "json"
    return "plain"


def class_key(feat: dict) -> str:
    """The most specific request-class key: modality|size|content."""
    return f"{modality(feat)}|{size_bucket(feat)}|{content_bucket(feat)}"


def bucket_keys(feat: dict) -> list[str]:
    """
    Hierarchical keys from most-specific to most-general, for back-off smoothing
    when a specific (class, alias) cell is sparse:

        modality|size|content  ->  modality|size  ->  modality  ->  global
    """
    mod = modality(feat)
    return [
        class_key(feat),
        f"{mod}|{size_bucket(feat)}",
        mod,
        "global",
    ]


# ── Capability requirements derived from features ──
# Used by the runtime to filter the candidate alias set before scoring.

def requires_vision(feat: dict) -> bool:
    return modality(feat) == "image"


def requires_tools(feat: dict) -> bool:
    return modality(feat) == "tools"


def requires_large_context(feat: dict) -> bool:
    return estimated_tokens(feat) >= 12000


FEATURE_SCHEMA_VERSION = "v2"
