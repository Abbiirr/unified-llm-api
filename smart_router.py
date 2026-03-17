"""
Smart LLM Router — thin classification layer in front of LiteLLM.

Inspects incoming requests and rewrites the model alias based on:
  - Image content in messages     → "vision"
  - Tool/function definitions     → "tools"
  - Code-related keywords         → "coding"
  - Reasoning/analysis keywords   → "thinking"
  - Privacy keywords              → "local"
  - Short/simple messages         → "fast"
  - Long/complex prompts          → "big"
  - Everything else               → "default"

Only rewrites when model is "auto" or "default".
Explicit alias picks (e.g. "coding") pass through untouched.

Ports:
  - This router listens on :4001 (public-facing)
  - LiteLLM runs on :4000 (internal, proxied by this router)
  - Docs endpoint moved to :4002
"""

import hashlib
import httpx
import re
import os
import time
import json
import logging
import logging.handlers
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — timestamped, separated by concern, rotated
# ─────────────────────────────────────────────────────────────────────────────
#
# logs/
#   router/       — smart router decisions, errors, latencies
#   training/     — ML training data (JSONL)
#   litellm/      — LiteLLM proxy logs (managed by run_gateway.sh)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.environ.get("LOG_DIR", "logs")

os.makedirs(f"{LOG_DIR}/router", exist_ok=True)
os.makedirs(f"{LOG_DIR}/training", exist_ok=True)

LOG_FORMAT = "%(asctime)s %(levelname)-5s [router] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Console handler
console = logging.StreamHandler()
console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

# Rotating file handler — 50MB per file, keep 10 files (500MB total)
file_handler = logging.handlers.RotatingFileHandler(
    f"{LOG_DIR}/router/router.log",
    maxBytes=50 * 1024 * 1024,  # 50MB
    backupCount=10,
    encoding="utf-8",
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

log = logging.getLogger("smart_router")
log.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
log.addHandler(console)
log.addHandler(file_handler)
log.propagate = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://litellm:4000")
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "4001"))

# Model names that trigger smart routing. All others pass through as-is.
AUTO_ROUTE_MODELS = {"auto", "default", ""}

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED LOGGING (JSONL for ML training data)
# ─────────────────────────────────────────────────────────────────────────────

import datetime
import uuid

# Training data — rotating JSONL, 100MB per file, keep 20 files (2GB total)
_training_handler = logging.handlers.RotatingFileHandler(
    f"{LOG_DIR}/training/routing.jsonl",
    maxBytes=100 * 1024 * 1024,  # 100MB
    backupCount=20,
    encoding="utf-8",
)
_training_handler.setFormatter(logging.Formatter("%(message)s"))
_training_logger = logging.getLogger("training_data")
_training_logger.setLevel(logging.INFO)
_training_logger.addHandler(_training_handler)
_training_logger.propagate = False


def log_training_sample(sample: dict):
    """Append a structured JSONL record for future ML router training."""
    sample["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    sample["id"] = uuid.uuid4().hex[:12]
    try:
        _training_logger.info(json.dumps(sample, default=str))
    except Exception as e:
        log.warning("Failed to write training log: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HTTP CLIENT (connection pooling)
# ─────────────────────────────────────────────────────────────────────────────

http_client: httpx.AsyncClient | None = None


import asyncio

# Ollama host health state — updated by background probe
ollama_health: dict[str, bool] = {}
OLLAMA_HOSTS = {
    "OLLAMA_HOST_1": os.environ.get("OLLAMA_HOST_1", ""),
    "OLLAMA_HOST_2": os.environ.get("OLLAMA_HOST_2", ""),
}
# Reverse map: URL → env var name (for matching config entries)
OLLAMA_URL_TO_NAME = {url: name for name, url in OLLAMA_HOSTS.items() if url}


def is_ollama_host_healthy(api_base_env: str) -> bool:
    """Check if an Ollama host is healthy based on the env var reference.
    api_base_env is like 'os.environ/OLLAMA_HOST_1' or the resolved URL.
    """
    # Extract env var name from 'os.environ/OLLAMA_HOST_1'
    if api_base_env.startswith("os.environ/"):
        env_name = api_base_env.split("/", 1)[1]
        return ollama_health.get(env_name, True)  # default True if unknown
    # Check by URL
    name = OLLAMA_URL_TO_NAME.get(api_base_env)
    if name:
        return ollama_health.get(name, True)
    return True  # not an Ollama host, assume healthy


async def probe_ollama_hosts():
    """Background task: check Ollama hosts every 30s, update health state."""
    probe_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=3, read=3, write=3, pool=3))
    while True:
        for name, url in OLLAMA_HOSTS.items():
            if not url:
                continue
            try:
                resp = await probe_client.get(f"{url}/api/tags")
                was_up = ollama_health.get(name)
                ollama_health[name] = resp.status_code == 200
                if ollama_health[name] and not was_up:
                    log.info("OLLAMA  %s (%s) is UP", name, url)
                elif not ollama_health[name] and was_up:
                    log.warning("OLLAMA  %s (%s) is DOWN", name, url)
            except Exception:
                if ollama_health.get(name, True):  # was up or unknown
                    log.warning("OLLAMA  %s (%s) is DOWN (unreachable)", name, url)
                ollama_health[name] = False
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        base_url=LITELLM_BASE,
        timeout=httpx.Timeout(connect=10, read=600, write=30, pool=10),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    # Start background Ollama health probe
    probe_task = asyncio.create_task(probe_ollama_hosts())
    log.info("Router started — forwarding to %s", LITELLM_BASE)
    yield
    probe_task.cancel()
    await http_client.aclose()
    log.info("Router shut down")


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

KEYWORD_RULES: list[tuple[str, int, re.Pattern]] = [
    # Priority 100: explicit privacy override
    ("local", 100, re.compile(
        r"\b(private|confidential|sensitive|secret|internal.?only|"
        r"do.?not.?share|keep.?local|offline|no.?cloud|air.?gap)\b",
        re.IGNORECASE
    )),

    # Priority 90: coding signals — require word boundary to avoid false matches
    ("coding", 90, re.compile(
        r"\b(write\s+(?:a\s+)?(?:function|class|method|script|program|code|test|api|endpoint)|"
        r"fix\s+(?:the\s+)?(?:bug|error|code|issue)|"
        r"debug|refactor|implement|code\s+review|pull.?request|"
        r"traceback|syntax.?error|compile|regex|"
        r"sql\b|query\b|schema|migration|"
        r"html\b|css\b|javascript\b|typescript\b|python\b|java\b|rust\b|go\b|kotlin\b|swift\b|"
        r"dockerfile|kubernetes|terraform|ansible|"
        r"pytest|jest|unittest|"
        r"algorithm|data.?structure|leetcode|hackerrank|"
        r"fastapi|django|flask|react|vue|angular|next\.?js|spring.?boot)\b",
        re.IGNORECASE
    )),

    # Priority 80: reasoning / analysis
    ("thinking", 80, re.compile(
        r"\b(analyze|compare\s+and|evaluate|critique|"
        r"pros.?and.?cons|trade.?off|"
        r"reason\s+(?:about|through|step)|think\s+(?:step|through|carefully)|"
        r"math\b|equation|calcul|proof|theorem|derive|"
        r"research|hypothesis|"
        r"strategy|approach|recommend\s+(?:the\s+)?best|"
        r"root.?cause|why\s+does|how\s+does)\b",
        re.IGNORECASE
    )),

    # Priority 60: large model signals
    ("big", 60, re.compile(
        r"\b(write\s+(?:a\s+)?(?:essay|report|article|document|paper|proposal)|"
        r"comprehensive|thorough|detailed.?analysis|in.?depth|"
        r"multi.?step|large.?scale|"
        r"entire|complete\s+(?:implementation|solution|guide))\b",
        re.IGNORECASE
    )),
]

# Short conversational patterns for "fast"
FAST_PATTERN = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|yes|no|sure|got it|"
    r"what is|who is|when was|where is|define|"
    r"translate|summarize|tldr|tl;?dr|"
    r"convert|format)\b",
    re.IGNORECASE
)


def has_images(messages: list[dict]) -> bool:
    """Check if any message contains image content."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    return True
                if part.get("type") == "image":
                    return True
    return False


def has_tools(data: dict) -> bool:
    """Check if the request defines tools or functions."""
    if data.get("tools"):
        return True
    if data.get("functions"):
        return True
    if data.get("tool_choice") and data["tool_choice"] != "none":
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-PROVIDER NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
# Problem: providers generate tool_call_ids in different formats.
#   Cerebras: call_5167de7ddfe14dc7b2e1bf19  (underscores)
#   Mistral:  requires [a-zA-Z0-9] only
#   Groq:     requires strict tool_call_id ↔ assistant pairing
#
# Solution: strip all non-alphanumeric chars from tool_call_ids in both
# requests (conversation history) and responses (new tool calls).
# This is idempotent — applying it twice gives the same result.

def _clean_id(tool_call_id: str) -> str:
    """
    Normalize a tool_call_id to exactly 9 alphanumeric characters.
    Uses a deterministic hash so the same input always maps to the same output.
    This satisfies Mistral's strict 9-char [a-zA-Z0-9] requirement while
    remaining compatible with all other providers.
    """
    # Already clean and 9 chars? Pass through.
    if len(tool_call_id) == 9 and tool_call_id.isalnum():
        return tool_call_id
    # Deterministic hash → 9 hex chars (0-9, a-f)
    return hashlib.sha256(tool_call_id.encode()).hexdigest()[:9]


def normalize_request(data: dict) -> bool:
    """
    Normalize tool_call_ids in the request's conversation history.
    Modifies data in-place. Returns True if any IDs were changed.
    """
    changed = False
    for msg in data.get("messages", []):
        # Normalize tool_call_id on role:tool messages
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            clean = _clean_id(msg["tool_call_id"])
            if clean != msg["tool_call_id"]:
                msg["tool_call_id"] = clean
                changed = True

        # Normalize tool_calls[].id on role:assistant messages
        for tc in msg.get("tool_calls", []):
            if tc.get("id"):
                clean = _clean_id(tc["id"])
                if clean != tc["id"]:
                    tc["id"] = clean
                    changed = True

    return changed


def normalize_response(body: bytes) -> bytes:
    """
    Normalize tool_call_ids in the provider's response.
    Returns modified body bytes.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body

    changed = False
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        for tc in msg.get("tool_calls", []):
            if tc.get("id"):
                clean = _clean_id(tc["id"])
                if clean != tc["id"]:
                    tc["id"] = clean
                    changed = True

    if changed:
        return json.dumps(data).encode()
    return body


def repair_messages(data: dict) -> bool:
    """
    Fix conversation history issues that cause 400/500 errors.
    Handles both loose (most providers) and strict (Groq) validation:

    1. Remove orphan tool results (no matching tool_call in ANY preceding assistant)
    2. Enforce strict sequential pairing (tool messages must follow their assistant)
    3. Fix assistant messages with content: null but no tool_calls
    4. Remove empty messages
    5. Ensure tool_call_ids are present on tool messages

    Returns True if any repairs were made.
    """
    messages = data.get("messages", [])
    if not messages:
        return False

    repaired = False

    # Pass 1: collect valid tool_call IDs and fix null content
    valid_tc_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            # Fix null content on assistant with tool_calls
            if msg.get("content") is None and msg.get("tool_calls"):
                msg["content"] = ""
                repaired = True
            # Fix null content on assistant without tool_calls
            elif msg.get("content") is None and not msg.get("tool_calls"):
                msg["content"] = ""
                repaired = True
            # Collect tool_call IDs
            for tc in msg.get("tool_calls", []):
                if tc.get("id"):
                    valid_tc_ids.add(tc["id"])
                    valid_tc_ids.add(_clean_id(tc["id"]))

    # Pass 2: enforce strict sequential pairing + remove orphans
    cleaned = []
    last_assistant_tc_ids = set()

    for msg in messages:
        role = msg.get("role", "")

        # Skip empty messages
        if not role:
            repaired = True
            continue

        if role == "assistant":
            # Track this assistant's tool_call IDs for strict pairing
            last_assistant_tc_ids = set()
            for tc in msg.get("tool_calls", []):
                if tc.get("id"):
                    last_assistant_tc_ids.add(tc["id"])
                    last_assistant_tc_ids.add(_clean_id(tc["id"]))

        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            normalized_id = _clean_id(tc_id) if tc_id else ""

            # Check: does this tool result match ANY assistant tool_call?
            if tc_id not in valid_tc_ids and normalized_id not in valid_tc_ids:
                log.warning("REPAIR  removed orphan tool result (tool_call_id=%s)", tc_id)
                repaired = True
                continue

            # Check: does this tool result have a tool_call_id at all?
            if not tc_id:
                log.warning("REPAIR  removed tool message with no tool_call_id")
                repaired = True
                continue

        elif role in ("user", "system"):
            # Reset strict pairing tracker on user/system messages
            last_assistant_tc_ids = set()

        cleaned.append(msg)

    if repaired:
        data["messages"] = cleaned

    return repaired


def extract_user_text(messages: list[dict]) -> str:
    """Extract text from the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
    return ""


def classify_request(data: dict) -> tuple[str, str]:
    """
    Classify a chat completion request.
    Returns: (alias, reason)
    """
    messages = data.get("messages", [])

    # ── Structural detection (highest priority after "local") ──

    # Images → vision
    if has_images(messages):
        return "vision", "image content detected in messages"

    # Tools/functions → tools
    if has_tools(data):
        return "tools", "tool/function definitions in request"

    # ── Text-based classification ──
    user_text = extract_user_text(messages)

    if not user_text:
        return "default", "no user text found"

    # Very short messages → fast (unless they contain coding/reasoning keywords)
    stripped = user_text.strip()
    if len(stripped) < 20:
        # Check if short but has important keywords
        lower = stripped.lower()
        if any(kw in lower for kw in ("code", "fix", "debug", "explain", "analyze", "write")):
            pass  # fall through to keyword matching
        elif FAST_PATTERN.search(stripped):
            return "fast", f"short message matching fast pattern"
        else:
            return "fast", f"very short message ({len(stripped)} chars)"

    # Keyword pattern matching
    matches: list[tuple[str, int, str]] = []
    for alias, priority, pattern in KEYWORD_RULES:
        m = pattern.search(user_text)
        if m:
            matches.append((alias, priority, m.group(0)))

    if matches:
        matches.sort(key=lambda x: x[1], reverse=True)
        winner = matches[0]
        return winner[0], f"keyword match: '{winner[2]}'"

    # Code block detection
    if "```" in user_text or user_text.count("    ") > 3:
        return "coding", "code block detected in message"

    # Length-based fallback
    if len(user_text) > 1500:
        return "big", f"long message ({len(user_text)} chars)"
    if len(user_text) > 500:
        return "thinking", f"medium-length message ({len(user_text)} chars)"

    return "default", "no specific signal detected"


# ─────────────────────────────────────────────────────────────────────────────
# DOCS (served at /docs and /docs/json, inline — no separate service needed)
# ─────────────────────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from gateway_docs import build_docs, render_html, DocsHandler
    _docs_available = True
except ImportError:
    _docs_available = False


@app.get("/docs")
@app.get("/docs/")
async def docs_html():
    if not _docs_available:
        return Response(content="Docs module not found", status_code=503)
    docs = build_docs()
    html = render_html(docs)
    return Response(content=html, media_type="text/html")


@app.get("/docs/json")
async def docs_json():
    if not _docs_available:
        return Response(content=json.dumps({"error": "Docs module not found"}), status_code=503)
    docs = build_docs()
    return Response(
        content=json.dumps(docs, indent=2),
        media_type="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTROSPECTION (must be defined before the catch-all proxy route)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/router/classify")
async def classify_only(request: Request):
    """
    Debug endpoint: classify a request without forwarding.
    POST {"messages": [...], "tools": [...]}
    Returns: {"alias": "coding", "reason": "keyword match: 'python'"}
    """
    data = await request.json()
    alias, reason = classify_request(data)
    return {"alias": alias, "reason": reason}


@app.get("/router/health")
async def router_health():
    """Health check for the router itself."""
    return {
        "status": "healthy",
        "backend": LITELLM_BASE,
        "ollama_hosts": {
            name: {"url": url, "healthy": ollama_health.get(name, False)}
            for name, url in OLLAMA_HOSTS.items() if url
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROXY (catch-all — must be last)
# ─────────────────────────────────────────────────────────────────────────────

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """Forward all requests to LiteLLM, with smart model rewriting on chat completions."""

    url = f"/{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "transfer-encoding")}
    body = await request.body()

    original_model = None
    chosen_alias = None
    reason = None
    training_features = None
    start = time.monotonic()

    # ── Smart routing: only for chat completions ──
    is_chat = path.rstrip("/") in ("v1/chat/completions", "chat/completions") and request.method == "POST"
    if is_chat:
        try:
            data = json.loads(body)
            original_model = data.get("model", "default")

            # Extract features for ML training log
            messages = data.get("messages", [])
            user_text = extract_user_text(messages)
            # Count message types for conversation structure
            role_counts = {}
            for m in messages:
                r = m.get("role", "unknown")
                role_counts[r] = role_counts.get(r, 0) + 1

            # Estimate tokens (~4 chars per token)
            payload_chars = len(json.dumps(messages))
            estimated_tokens = payload_chars // 4

            training_features = {
                "original_model": original_model,
                "message_count": len(messages),
                "user_text_length": len(user_text),
                "has_images": has_images(messages),
                "has_tools": has_tools(data),
                "has_system_prompt": any(m.get("role") == "system" for m in messages),
                "tool_count": len(data.get("tools", [])),
                "has_tool_choice": bool(data.get("tool_choice")),
                "temperature": data.get("temperature"),
                "max_tokens": data.get("max_tokens") or data.get("max_completion_tokens"),
                "stream": data.get("stream"),
                "user_text_preview": user_text[:200],
                # Conversation structure
                "user_messages": role_counts.get("user", 0),
                "assistant_messages": role_counts.get("assistant", 0),
                "tool_messages": role_counts.get("tool", 0),
                "system_messages": role_counts.get("system", 0),
                "estimated_input_tokens": estimated_tokens,
                "payload_chars": payload_chars,
            }

            # Always classify for training data, even if we don't reroute
            suggested_alias, suggested_reason = classify_request(data)
            training_features["suggested_alias"] = suggested_alias
            training_features["suggested_reason"] = suggested_reason

            if original_model in AUTO_ROUTE_MODELS:
                chosen_alias = suggested_alias
                reason = suggested_reason
                data["model"] = chosen_alias
                log.info(
                    "ROUTE  %s → %s  reason=%s",
                    original_model, chosen_alias, reason
                )
            else:
                chosen_alias = original_model
                reason = "explicit"
                log.info("PASS   model=%s (explicit)", original_model)

            # Large payload? Use _large alias variant (no small-context models)
            payload_chars = len(json.dumps(data.get("messages", [])))
            effective_model = data.get("model", "")
            if payload_chars > 6000 and effective_model in ("tools", "bench"):
                data["model"] = f"{effective_model}_large"
                log.info(
                    "LARGE  %s → %s  payload=%d chars",
                    effective_model, data["model"], payload_chars
                )

            # Default to streaming — avoids timeouts on thinking models,
            # faster time-to-first-token, better UX
            if "stream" not in data:
                data["stream"] = True
                log.info("STREAM  defaulting to stream=true")

            # Repair broken conversation history (orphan tool results, null content)
            if repair_messages(data):
                log.info("REPAIR  fixed conversation history issues")

            # Normalize tool_call_ids for cross-provider compatibility
            if normalize_request(data):
                log.info("NORMALIZE  sanitized tool_call_ids in request")

            body = json.dumps(data).encode()
        except (json.JSONDecodeError, KeyError):
            log.warning("Failed to parse request body, forwarding as-is")

    # ── Check if streaming ──
    is_stream = False
    try:
        if body:
            is_stream = json.loads(body).get("stream", False)
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Forward to LiteLLM ──
    try:
        if is_stream:
            req = http_client.build_request(request.method, url, headers=headers, content=body)
            resp = await http_client.send(req, stream=True)

            async def stream_gen():
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()

            response_headers = dict(resp.headers)
            if chosen_alias:
                response_headers["X-Router-Alias"] = chosen_alias
                response_headers["X-Router-Original-Model"] = original_model or "default"
                response_headers["X-Router-Reason"] = reason or ""

            elapsed = time.monotonic() - start
            log.info(
                "STREAM status=%d model=%s %.0fms",
                resp.status_code, chosen_alias or original_model or "?", elapsed * 1000
            )

            # ── Training log (streaming) ──
            if training_features:
                provider = resp.headers.get("x-litellm-model-api-base", "")
                served_model = resp.headers.get("x-litellm-model-id", "")
                log_training_sample({
                    **training_features,
                    "routed_alias": chosen_alias,
                    "route_reason": reason,
                    "provider_base": provider,
                    "served_model": served_model,
                    "status": resp.status_code,
                    "latency_ms": round(elapsed * 1000),
                    "stream": True,
                    "cache_hit": resp.headers.get("x-litellm-cache-key", "") != "",
                })

            return StreamingResponse(
                stream_gen(),
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type"),
            )
        else:
            resp = await http_client.request(request.method, url, headers=headers, content=body)

            elapsed = time.monotonic() - start

            # ── Log response details ──
            status = resp.status_code
            log_model = chosen_alias or original_model or path
            resp_body_override = None

            if status == 429:
                log.warning(
                    "RATE LIMITED  status=429 model=%s %.0fms — provider returned 429",
                    log_model, elapsed * 1000
                )
            elif status >= 400:
                error_msg = ""
                provider_name = resp.headers.get("x-litellm-model-api-base", "unknown")
                failed_model = resp.headers.get("x-litellm-model-id", "")
                try:
                    err_data = json.loads(resp.content)
                    error_msg = err_data.get("error", {}).get("message", "")[:200]
                    # Inject provider info into error response for client visibility
                    if isinstance(err_data.get("error"), dict):
                        err_data["error"]["provider"] = provider_name
                        err_data["error"]["failed_model"] = failed_model
                        resp_body_override = json.dumps(err_data).encode()
                    else:
                        resp_body_override = None
                except Exception:
                    resp_body_override = None
                log.error(
                    "ERROR  status=%d model=%s provider=%s %.0fms %s",
                    status, log_model, provider_name, elapsed * 1000, error_msg
                )
            else:
                log.info(
                    "OK     status=%d model=%s %.0fms",
                    status, log_model, elapsed * 1000
                )

            # ── Normalize response tool_call_ids ──
            resp_body = resp.content
            if status == 200 and path.rstrip("/") in ("v1/chat/completions", "chat/completions"):
                normalized = normalize_response(resp_body)
                if normalized is not resp_body:
                    log.info("NORMALIZE  sanitized tool_call_ids in response")
                    resp_body = normalized

            # ── Training log (non-streaming) ──
            if training_features and is_chat:
                provider = resp.headers.get("x-litellm-model-api-base", "")
                served_model = resp.headers.get("x-litellm-model-id", "")
                usage = {}
                try:
                    usage = json.loads(resp_body).get("usage", {})
                except Exception:
                    pass
                log_training_sample({
                    **training_features,
                    "routed_alias": chosen_alias,
                    "route_reason": reason,
                    "provider_base": provider,
                    "served_model": served_model,
                    "status": status,
                    "latency_ms": round(elapsed * 1000),
                    "stream": False,
                    "cache_hit": resp.headers.get("x-litellm-cache-key", "") != "",
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                })

            # ── Inject routing metadata, fix Content-Length ──
            response_headers = {k: v for k, v in resp.headers.items()
                                if k.lower() not in ("content-length", "transfer-encoding")}
            if chosen_alias:
                response_headers["X-Router-Alias"] = chosen_alias
                response_headers["X-Router-Original-Model"] = original_model or "default"
                response_headers["X-Router-Reason"] = reason or ""

            # Use enriched error body if available (includes provider name)
            final_body = resp_body
            if status >= 400 and resp_body_override:
                final_body = resp_body_override

            return Response(
                content=final_body,
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type"),
            )

    except httpx.TimeoutException:
        elapsed = time.monotonic() - start
        log.error(
            "TIMEOUT  model=%s %.0fms — LiteLLM did not respond",
            chosen_alias or original_model or "?", elapsed * 1000
        )
        return Response(
            content=json.dumps({"error": {"message": "Gateway timeout", "code": 504}}),
            status_code=504,
            media_type="application/json",
        )
    except httpx.ConnectError:
        log.error("CONNECT ERROR — cannot reach LiteLLM at %s", LITELLM_BASE)
        return Response(
            content=json.dumps({"error": {"message": "LiteLLM backend unavailable", "code": 502}}),
            status_code=502,
            media_type="application/json",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ROUTER_PORT)
