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
LOG_DIR = os.path.abspath(os.environ.get("LOG_DIR", os.path.join(os.path.dirname(__file__), "logs")))

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

LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://localhost:4002")
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "4000"))

# Model names that trigger smart routing. All others pass through as-is.
AUTO_ROUTE_MODELS = {"auto", "default", ""}
LOCAL_ONLY_ALIASES = {"local", "tools_local", "llama_local"}

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

# Full conversation logs — for LLM fine-tuning / distillation
# Stores complete request + response pairs for successful tool-calling conversations
os.makedirs(f"{LOG_DIR}/training/conversations", exist_ok=True)
_convo_handler = logging.handlers.RotatingFileHandler(
    f"{LOG_DIR}/training/conversations/convos.jsonl",
    maxBytes=200 * 1024 * 1024,  # 200MB per file
    backupCount=10,              # 2GB total
    encoding="utf-8",
)
_convo_handler.setFormatter(logging.Formatter("%(message)s"))
_convo_logger = logging.getLogger("conversation_data")
_convo_logger.setLevel(logging.INFO)
_convo_logger.addHandler(_convo_handler)
_convo_logger.propagate = False


def log_conversation(sample: dict):
    """Log full request+response conversation for LLM fine-tuning."""
    try:
        _convo_logger.info(json.dumps(sample, default=str))
    except Exception as e:
        log.warning("Failed to write conversation log: %s", e)


def _categorize_error(status: int, error_msg: str) -> str:
    """Categorize errors for ML training — provider failure mode learning."""
    if status == 200:
        return "success"
    if status == 429:
        return "rate_limit"
    if "ContextWindow" in error_msg:
        return "context_window"
    if "schema" in error_msg.lower() or "properties" in error_msg.lower():
        return "schema_validation"
    if status == 401 or status == 403:
        return "auth"
    if status == 413:
        return "body_too_large"
    if status == 500:
        return "provider_internal"
    if status == 504:
        return "timeout"
    return f"error_{status}"


def log_training_sample(sample: dict):
    """Append a structured JSONL record for future ML router training."""
    now = datetime.datetime.utcnow()
    sample["timestamp"] = now.isoformat() + "Z"
    sample["id"] = uuid.uuid4().hex[:12]
    # Error categorization
    sample["error_category"] = _categorize_error(
        sample.get("status", 200), sample.get("error_msg", "")
    )
    # Time features — critical for learning rate-limit patterns
    sample["hour_utc"] = now.hour
    sample["minute_utc"] = now.minute
    sample["day_of_week"] = now.weekday()  # 0=Monday
    # Clean provider name from URL
    pb = sample.get("provider_base", "")
    if "cerebras" in pb: sample["provider"] = "cerebras"
    elif "groq" in pb: sample["provider"] = "groq"
    elif "nvidia" in pb: sample["provider"] = "nvidia"
    elif "gemini" in pb or "generativelanguage" in pb: sample["provider"] = "gemini"
    elif "github" in pb or "azure" in pb: sample["provider"] = "github"
    elif "mistral" in pb: sample["provider"] = "mistral"
    elif "openrouter" in pb: sample["provider"] = "openrouter"
    elif "cohere" in pb: sample["provider"] = "cohere"
    elif "cloudflare" in pb: sample["provider"] = "cloudflare"
    elif any(url in pb for url in OLLAMA_HOSTS.values() if url): sample["provider"] = "ollama"
    elif os.environ.get("LLAMA_CPP_HOST", "") and os.environ["LLAMA_CPP_HOST"].rstrip("/v1") in pb: sample["provider"] = "llama_cpp"
    else: sample["provider"] = "unknown"
    # Rolling provider health stats
    provider_name = sample["provider"]
    if provider_name != "unknown":
        stats = _get_provider_stats(provider_name)
        sample["provider_recent_error_rate"] = stats["error_rate"]
        sample["provider_recent_avg_latency_ms"] = stats["avg_latency"]
        sample["provider_recent_request_count"] = stats["count"]
    try:
        _training_logger.info(json.dumps(sample, default=str))
    except Exception as e:
        log.warning("Failed to write training log: %s", e)
    # Update provider stats after logging
    _record_provider_outcome(sample.get("provider", ""), sample.get("status", 0), sample.get("latency_ms", 0))


# ── Rolling provider health stats (last 5 minutes) ──
_provider_history: dict[str, list[tuple[float, int, int]]] = {}  # provider → [(timestamp, status, latency_ms)]
_PROVIDER_STATS_WINDOW = 300  # 5 minutes


def _record_provider_outcome(provider: str, status: int, latency_ms: int):
    """Record a provider outcome for rolling stats."""
    if not provider or provider == "unknown":
        return
    now = time.monotonic()
    _provider_history.setdefault(provider, []).append((now, status, latency_ms))
    # Prune old entries
    cutoff = now - _PROVIDER_STATS_WINDOW
    _provider_history[provider] = [(t, s, l) for t, s, l in _provider_history[provider] if t > cutoff]


def _get_provider_stats(provider: str) -> dict:
    """Get rolling stats for a provider over the last 5 minutes."""
    now = time.monotonic()
    cutoff = now - _PROVIDER_STATS_WINDOW
    entries = [(t, s, l) for t, s, l in _provider_history.get(provider, []) if t > cutoff]
    if not entries:
        return {"error_rate": 0.0, "avg_latency": 0, "count": 0}
    errors = sum(1 for _, s, _ in entries if s >= 400)
    avg_lat = sum(l for _, _, l in entries) // len(entries)
    return {"error_rate": round(errors / len(entries), 3), "avg_latency": avg_lat, "count": len(entries)}


def get_circuit_broken_providers() -> list[str]:
    """Return list of providers whose error rate exceeds the circuit breaker threshold."""
    broken = []
    now = time.monotonic()
    cutoff = now - _PROVIDER_STATS_WINDOW
    for provider, history in _provider_history.items():
        entries = [(t, s, l) for t, s, l in history if t > cutoff]
        if len(entries) < CIRCUIT_BREAKER_MIN_REQUESTS:
            continue
        errors = sum(1 for _, s, _ in entries if s >= 400)
        error_rate = errors / len(entries)
        if error_rate >= CIRCUIT_BREAKER_THRESHOLD:
            broken.append(provider)
    return broken


# ── Model identity resolver ──
# Maps opaque model ID hashes from LiteLLM to readable model names.
# Built at startup by querying /v1/models.
_model_id_to_name: dict[str, str] = {}


async def _build_model_identity_map():
    """Query LiteLLM /model/info and build model_id_hash → readable model name map."""
    global _model_id_to_name
    master_key = os.environ.get("LITELLM_MASTER_KEY", "")
    auth_headers = {"Authorization": f"Bearer {master_key}"} if master_key else {}
    try:
        resp = await http_client.get("/model/info", headers=auth_headers)
        if resp.status_code == 200:
            data = resp.json()
            for entry in data.get("data", []):
                id_hash = entry.get("model_info", {}).get("id", "")
                litellm_model = entry.get("litellm_params", {}).get("model", "")
                if id_hash and litellm_model:
                    _model_id_to_name[id_hash] = litellm_model
            log.info("MODEL_MAP  built %d model identity mappings", len(_model_id_to_name))
        else:
            log.warning("MODEL_MAP  failed to query /model/info (status=%d)", resp.status_code)
    except Exception as e:
        log.warning("MODEL_MAP  failed: %s", e)


def resolve_model_name(model_id_hash: str) -> str:
    """Resolve an opaque model ID hash to a readable model name."""
    return _model_id_to_name.get(model_id_hash, model_id_hash)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HTTP CLIENT (connection pooling)
# ─────────────────────────────────────────────────────────────────────────────

http_client: httpx.AsyncClient | None = None


import asyncio

# ── Auto-flush cooldowns on 429 burst ──
# Track recent 429s. If we see 3+ in 30 seconds, flush LiteLLM cooldowns.
_recent_429_timestamps: list[float] = []

# ── Concurrent rescue limiter ──
# Prevent Ollama overwhelm when many requests timeout simultaneously.
_active_rescues = 0
_MAX_CONCURRENT_RESCUES = 3  # Ollama can handle ~3 concurrent 27B requests
_last_auto_flush: float = 0
AUTO_FLUSH_THRESHOLD = 3      # consecutive 429s to trigger flush
AUTO_FLUSH_WINDOW = 30        # seconds window for burst detection
AUTO_FLUSH_COOLDOWN = 60      # don't auto-flush more than once per 60s

# ── Soft circuit breaker per provider ──
# If a provider's error rate exceeds this threshold over the last 5 minutes,
# the router will proactively skip it by adding X-LiteLLM-Exclude header.
CIRCUIT_BREAKER_THRESHOLD = 0.50  # 50% error rate triggers circuit break
CIRCUIT_BREAKER_MIN_REQUESTS = 5  # need at least 5 requests to evaluate

# ── Soft timeout with retry ──
# If a streaming response doesn't start within this many seconds, abort and
# retry via Ollama. Prevents clients waiting 600s for a hung provider.
SOFT_TIMEOUT_SECONDS = 300    # 5 minutes (vs 600s hard limit)


def _pick_stage2_alias(original_alias: str) -> str:
    """Pick the right alias for 429 rescue stage 2.

    Stage-1 rescue always tries Ollama (`tools_local`). When that fails, stage-2
    falls back to a different cloud alias. Crucially, this must NOT be the same
    alias as the original — otherwise we just retry the same providers that
    already 429'd. `default_cloud` has the broadest provider diversity (Gemini,
    Mistral, NVIDIA, OpenRouter, GitHub, Cohere) so it's the best last-resort
    when the primary alias is itself stable/large.
    """
    if original_alias in {"tools_stable", "tools_stable_cloud",
                          "tools_large", "swebench"}:
        return "default_cloud"
    return "tools_stable"


async def _auto_flush_if_needed(force: bool = False):
    """Flush LiteLLM cooldowns if we're in a 429 burst, or immediately if force=True."""
    global _last_auto_flush
    now = time.monotonic()

    # Prune old timestamps
    _recent_429_timestamps[:] = [t for t in _recent_429_timestamps if now - t < AUTO_FLUSH_WINDOW]
    _recent_429_timestamps.append(now)

    should_flush = force or (
        len(_recent_429_timestamps) >= AUTO_FLUSH_THRESHOLD
        and (now - _last_auto_flush) > AUTO_FLUSH_COOLDOWN
    )
    if should_flush:
        _last_auto_flush = now
        try:
            import redis as _redis
            r = _redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
            )
            keys = r.keys("deployment:*:cooldown")
            count = sum(1 for k in keys if r.delete(k))
            log.warning(
                "AUTO_FLUSH  %s → cleared %d Redis cooldowns",
                "forced pre-rescue" if force else f"{len(_recent_429_timestamps)} 429s in {AUTO_FLUSH_WINDOW}s",
                count,
            )
            _recent_429_timestamps.clear()
        except Exception as e:
            log.warning("AUTO_FLUSH  failed to flush cooldowns: %s", e)


# Ollama host health state — updated by background probe
ollama_health: dict[str, bool] = {}
OLLAMA_HOSTS = {
    "OLLAMA_HOST_1": os.environ.get("OLLAMA_HOST_1", ""),
    "OLLAMA_HOST_2": os.environ.get("OLLAMA_HOST_2", ""),
    "OLLAMA_HOST_3": os.environ.get("OLLAMA_HOST_3", ""),
}
# Reverse map: URL → env var name (for matching config entries)
OLLAMA_URL_TO_NAME = {url: name for name, url in OLLAMA_HOSTS.items() if url}

llama_cpp_health: dict[str, bool] = {}
LLAMA_CPP_HOSTS = {
    "LLAMA_CPP_HOST": os.environ.get("LLAMA_CPP_HOST", ""),
}
LLAMA_CPP_URL_TO_NAME = {url: name for name, url in LLAMA_CPP_HOSTS.items() if url}


# ── Alias → Ollama host mapping (built from config at startup) ──
# Parsed lazily on first request to avoid import-time file reads
_alias_ollama_map: dict[str, set[str]] | None = None
_alias_llama_cpp_map: dict[str, set[str]] | None = None


def _build_alias_ollama_map():
    """Parse litellm_config.yaml and build alias → set of Ollama host env vars."""
    global _alias_ollama_map, _alias_llama_cpp_map
    _alias_ollama_map = {}
    _alias_llama_cpp_map = {}
    config_path = os.environ.get("CONFIG_PATH", "litellm_config.yaml")
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for entry in cfg.get("model_list", []):
            model = entry.get("litellm_params", {}).get("model", "")
            alias = entry.get("model_name", "")
            api_base = entry.get("litellm_params", {}).get("api_base", "")
            # Detect Ollama entries: ollama/ or ollama_chat/ prefix, or api_base pointing to Ollama host
            is_ollama = (
                model.startswith("ollama/")
                or model.startswith("ollama_chat/")
                or any(url in api_base for url in OLLAMA_HOSTS.values() if url)
            )
            if is_ollama and api_base:
                _alias_ollama_map.setdefault(alias, set()).add(api_base)
            is_llama_cpp = (
                api_base == "os.environ/LLAMA_CPP_HOST"
                or any(url in api_base for url in LLAMA_CPP_HOSTS.values() if url)
            )
            if is_llama_cpp and api_base:
                _alias_llama_cpp_map.setdefault(alias, set()).add(api_base)
        log.info("OLLAMA_MAP  built alias→host map: %s",
                 {k: list(v) for k, v in _alias_ollama_map.items()})
        log.info("LLAMA_CPP_MAP  built alias→host map: %s",
                 {k: list(v) for k, v in _alias_llama_cpp_map.items()})
    except Exception as e:
        log.warning("OLLAMA_MAP  failed to parse config: %s", e)
        _alias_ollama_map = {}
        _alias_llama_cpp_map = {}


def alias_has_ollama(alias: str) -> bool:
    """Check if an alias contains Ollama deployments."""
    if _alias_ollama_map is None:
        _build_alias_ollama_map()
    return alias in _alias_ollama_map


def alias_ollama_all_down(alias: str) -> bool:
    """Check if ALL Ollama hosts for an alias are down."""
    if _alias_ollama_map is None:
        _build_alias_ollama_map()
    hosts = _alias_ollama_map.get(alias, set())
    if not hosts:
        return False  # no Ollama in this alias
    return all(not is_ollama_host_healthy(h) for h in hosts)


def alias_has_healthy_llama_cpp(alias: str) -> bool:
    """Check if an alias contains at least one healthy llama.cpp deployment."""
    if _alias_llama_cpp_map is None:
        _build_alias_ollama_map()
    hosts = _alias_llama_cpp_map.get(alias, set())
    if not hosts:
        return False
    return any(is_llama_cpp_host_healthy(h) for h in hosts)


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


def is_llama_cpp_host_healthy(api_base_env: str) -> bool:
    """Check if a llama.cpp host is healthy based on the env var reference."""
    if api_base_env.startswith("os.environ/"):
        env_name = api_base_env.split("/", 1)[1]
        return llama_cpp_health.get(env_name, True)
    name = LLAMA_CPP_URL_TO_NAME.get(api_base_env)
    if name:
        return llama_cpp_health.get(name, True)
    return True


def is_local_only_alias(alias: str) -> bool:
    """Aliases that must never escape to cloud providers."""
    return (
        alias in LOCAL_ONLY_ALIASES
        or alias.startswith("ollama/")
        or alias.startswith("ollama_chat/")
    )


def pick_408_rescue_aliases(current_model: str) -> tuple[str, ...]:
    """Timeout rescue chain for provider 408s."""
    if current_model == "spec-rag":
        return ("default_cloud",)
    if current_model == "llama_local":
        return ("local",)
    if is_local_only_alias(current_model):
        return ()
    return ("big", "default_cloud")


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

        for name, url in LLAMA_CPP_HOSTS.items():
            if not url:
                continue
            base = url.rstrip("/")
            models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
            try:
                resp = await probe_client.get(models_url)
                was_up = llama_cpp_health.get(name)
                llama_cpp_health[name] = resp.status_code == 200
                if llama_cpp_health[name] and not was_up:
                    log.info("LLAMA_CPP  %s (%s) is UP", name, url)
                elif not llama_cpp_health[name] and was_up:
                    log.warning("LLAMA_CPP  %s (%s) is DOWN", name, url)
            except Exception:
                if llama_cpp_health.get(name, True):
                    log.warning("LLAMA_CPP  %s (%s) is DOWN (unreachable)", name, url)
                llama_cpp_health[name] = False
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        base_url=LITELLM_BASE,
        timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    # Start background Ollama health probe
    probe_task = asyncio.create_task(probe_ollama_hosts())
    # Build model identity map (resolve opaque hashes to readable names)
    await _build_model_identity_map()
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


def repair_tool_schemas(data: dict) -> bool:
    """
    Fix tool/function schemas that strict providers (GitHub) reject.
    GitHub requires 'properties' key in object-type parameters, even if empty.
    Returns True if any repairs were made.
    """
    tools = data.get("tools", [])
    if not tools:
        return False
    repaired = False
    for tool in tools:
        fn = tool.get("function", {})
        params = fn.get("parameters", {})
        if params.get("type") == "object" and "properties" not in params:
            params["properties"] = {}
            repaired = True
        # Also fix nested parameter schemas
        for prop in params.get("properties", {}).values():
            if isinstance(prop, dict) and prop.get("type") == "object" and "properties" not in prop:
                prop["properties"] = {}
                repaired = True
    return repaired


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
    orphan_count = 0
    null_content_count = 0

    # Pass 0: strip non-standard fields that strict providers (Mistral) reject
    ALLOWED_ASSISTANT_KEYS = {"role", "content", "tool_calls", "name", "refusal"}
    extra_field_count = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            extra_keys = set(msg.keys()) - ALLOWED_ASSISTANT_KEYS
            for key in extra_keys:
                del msg[key]
                extra_field_count += 1
                repaired = True

    # Pass 0b: fix malformed tool_call arguments (Groq requires valid JSON)
    bad_args_count = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if args is None:
                    fn["arguments"] = "{}"
                    bad_args_count += 1
                    repaired = True
                elif isinstance(args, str):
                    args_stripped = args.strip()
                    if not args_stripped:
                        fn["arguments"] = "{}"
                        bad_args_count += 1
                        repaired = True
                    else:
                        try:
                            json.loads(args_stripped)
                        except (json.JSONDecodeError, ValueError):
                            # Try to salvage: wrap in object if it looks like bare key-values
                            fn["arguments"] = "{}"
                            bad_args_count += 1
                            repaired = True

    # Pass 1: collect valid tool_call IDs and fix null content
    valid_tc_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            # Fix null content on assistant with tool_calls
            if msg.get("content") is None and msg.get("tool_calls"):
                msg["content"] = ""
                null_content_count += 1
                repaired = True
            # Fix null content on assistant without tool_calls
            elif msg.get("content") is None and not msg.get("tool_calls"):
                msg["content"] = ""
                null_content_count += 1
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
                orphan_count += 1
                repaired = True
                continue

            # Check: does this tool result have a tool_call_id at all?
            if not tc_id:
                orphan_count += 1
                repaired = True
                continue

        elif role in ("user", "system"):
            # Reset strict pairing tracker on user/system messages
            last_assistant_tc_ids = set()

        cleaned.append(msg)

    if repaired:
        data["messages"] = cleaned
        # Batch log repairs — one line per request instead of per orphan
        parts = []
        if extra_field_count:
            parts.append(f"{extra_field_count} extra field(s) stripped")
        if bad_args_count:
            parts.append(f"{bad_args_count} malformed tool arg(s) fixed")
        if orphan_count:
            parts.append(f"{orphan_count} orphan tool result(s)")
        if null_content_count:
            parts.append(f"{null_content_count} null content fix(es)")
        if parts:
            log.warning("REPAIR  %s", ", ".join(parts))

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


# ── Fast content feature extraction for ML router training ──
# All regex-based, <0.1ms total — cheaper than a single string comparison

_CODE_BLOCK_RE = re.compile(r"```(\w*)")
_LANG_PATTERNS = {
    "python": re.compile(r"\bdef \w+|import \w+|class \w+.*:|\.py\b|python", re.I),
    "javascript": re.compile(r"\bfunction\s|const\s|let\s|=>\s|\.js\b|\.tsx?\b|javascript|typescript", re.I),
    "java": re.compile(r"\bpublic class|System\.out|\.java\b", re.I),
    "rust": re.compile(r"\bfn \w+|impl \w+|\.rs\b|rust\b", re.I),
    "go": re.compile(r"\bfunc \w+|package \w+|\.go\b", re.I),
    "bash": re.compile(r"\bbash\b|#!/bin|\.sh\b|\$\(|apt-get|pip install", re.I),
    "sql": re.compile(r"\bSELECT\s|INSERT\s|CREATE TABLE|\.sql\b", re.I),
    "html": re.compile(r"<div|<html|<body|\.html\b|\.css\b", re.I),
}
_TASK_PATTERNS = {
    "generate": re.compile(r"\b(write|create|generate|implement|build|make)\b", re.I),
    "debug": re.compile(r"\b(fix|debug|error|bug|traceback|exception|failing)\b", re.I),
    "explain": re.compile(r"\b(explain|what does|how does|why does|describe)\b", re.I),
    "refactor": re.compile(r"\b(refactor|improve|optimize|clean up|simplify)\b", re.I),
    "test": re.compile(r"\b(test|unittest|pytest|jest|spec)\b", re.I),
    "review": re.compile(r"\b(review|check|audit|validate)\b", re.I),
}


def extract_content_features(text: str, messages: list[dict]) -> dict:
    """Extract cheap content features for ML router training. <0.1ms."""
    features = {}

    # Code detection
    code_blocks = _CODE_BLOCK_RE.findall(text)
    features["has_code_blocks"] = len(code_blocks) > 0
    features["code_block_count"] = len(code_blocks)

    # Language detection (from code blocks + text)
    all_text = text
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            all_text += " " + c[:2000]  # sample first 2K chars per message

    detected_langs = []
    for lang, pattern in _LANG_PATTERNS.items():
        if pattern.search(all_text):
            detected_langs.append(lang)
    features["detected_languages"] = detected_langs[:5]
    features["primary_language"] = detected_langs[0] if detected_langs else None

    # Task type detection
    detected_tasks = []
    for task, pattern in _TASK_PATTERNS.items():
        if pattern.search(text):
            detected_tasks.append(task)
    features["task_types"] = detected_tasks
    features["primary_task"] = detected_tasks[0] if detected_tasks else None

    # Content complexity signals
    features["has_urls"] = bool(re.search(r"https?://", text))
    features["has_file_paths"] = bool(re.search(r"[/\\]\w+\.\w{1,5}\b", text))
    features["has_json"] = bool(re.search(r'[{"\[]\s*"?\w+"?\s*:', text))
    features["has_error_trace"] = bool(re.search(r"Traceback|Error:|Exception:|FAILED", text))
    features["avg_message_length"] = sum(
        len(str(m.get("content", ""))) for m in messages
    ) // max(len(messages), 1)

    return features


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


@app.get("/docs.json")
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
        "llama_cpp_hosts": {
            name: {"url": url, "healthy": llama_cpp_health.get(name, False)}
            for name, url in LLAMA_CPP_HOSTS.items() if url
        },
    }


@app.post("/router/flush-cooldowns")
async def flush_cooldowns():
    """Emergency endpoint: clear all provider cooldowns in Redis.
    Use when all providers are locked out after a restart/outage.
    POST /router/flush-cooldowns
    """
    try:
        import redis
        r = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
        )
        keys = r.keys("deployment:*:cooldown")
        count = 0
        for key in keys:
            r.delete(key)
            count += 1
        log.warning("FLUSH_COOLDOWNS  cleared %d provider cooldowns", count)
        return {"status": "flushed", "cooldowns_cleared": count}
    except Exception as e:
        log.error("FLUSH_COOLDOWNS  failed: %s", e)
        return Response(
            content=json.dumps({"error": {"message": f"Failed to flush: {e}", "code": 500}}),
            status_code=500,
            media_type="application/json",
        )


@app.get("/router/provider-status")
async def provider_status():
    """Report per-provider health stats (rolling 5-min window) and circuit breaker state."""
    now = time.monotonic()
    cutoff = now - _PROVIDER_STATS_WINDOW
    providers = {}
    for provider, history in _provider_history.items():
        entries = [(t, s, l) for t, s, l in history if t > cutoff]
        if not entries:
            continue
        errors = sum(1 for _, s, _ in entries if s >= 400)
        error_429s = sum(1 for _, s, _ in entries if s == 429)
        avg_lat = sum(l for _, _, l in entries) // len(entries)
        error_rate = round(errors / len(entries), 3)
        providers[provider] = {
            "requests_5m": len(entries),
            "errors_5m": errors,
            "rate_limits_5m": error_429s,
            "error_rate": error_rate,
            "avg_latency_ms": avg_lat,
            "circuit_broken": error_rate >= CIRCUIT_BREAKER_THRESHOLD and len(entries) >= CIRCUIT_BREAKER_MIN_REQUESTS,
        }
    broken = get_circuit_broken_providers()
    return {
        "providers": providers,
        "circuit_broken": broken,
        "ollama_hosts": {
            name: {"url": url, "healthy": ollama_health.get(name, False)}
            for name, url in OLLAMA_HOSTS.items() if url
        },
        "llama_cpp_hosts": {
            name: {"url": url, "healthy": llama_cpp_health.get(name, False)}
            for name, url in LLAMA_CPP_HOSTS.items() if url
        },
        "model_identity_count": len(_model_id_to_name),
    }


@app.post("/router/rebuild-model-map")
async def rebuild_model_map():
    """Force rebuild the model ID → name lookup table."""
    await _build_model_identity_map()
    return {"status": "rebuilt", "mappings": len(_model_id_to_name)}


# ─────────────────────────────────────────────────────────────────────────────
# PROXY (catch-all — must be last)
# ─────────────────────────────────────────────────────────────────────────────

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """Forward all requests to LiteLLM, with smart model rewriting on chat completions."""
    global _active_rescues

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

            # Tool names — important: some tools fail on specific providers
            tool_names = []
            tool_has_nested_objects = False
            for t in data.get("tools", []):
                fn = t.get("function", {})
                if fn.get("name"):
                    tool_names.append(fn["name"])
                # Detect complex schemas that some providers reject
                params = fn.get("parameters", {})
                for prop in params.get("properties", {}).values():
                    if isinstance(prop, dict) and prop.get("type") == "object":
                        tool_has_nested_objects = True

            # System prompt features
            system_text = ""
            for m in messages:
                if m.get("role") == "system":
                    c = m.get("content", "")
                    if isinstance(c, str):
                        system_text = c
                    break
            system_prompt_length = len(system_text)
            system_is_agent = any(w in system_text.lower() for w in
                ["agent", "tool", "function", "execute", "run command", "bash"]) if system_text else False

            # Conversation depth — count completed tool-call rounds
            tool_rounds = 0
            for m in messages:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    tool_rounds += 1

            # Total conversation tokens estimate (all messages, not just user text)
            total_conversation_chars = sum(
                len(json.dumps(m.get("content", "") or "")) for m in messages
            )

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
                # Tool details for provider compatibility learning
                "tool_names": tool_names[:20],  # cap at 20 to avoid bloat
                "tool_has_nested_objects": tool_has_nested_objects,
                # Conversation depth features
                "tool_rounds": tool_rounds,
                "total_conversation_chars": total_conversation_chars,
                "estimated_total_tokens": total_conversation_chars // 4,
                # System prompt features
                "system_prompt_length": system_prompt_length,
                "system_is_agent": system_is_agent,
                # Content features for ML routing (cheap regex, <0.1ms)
                **extract_content_features(user_text, messages),
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
            LARGE_REWRITE = {"tools": "tools_large", "bench": "bench_large", "swebench": "tools_large"}
            if payload_chars > 6000 and effective_model in LARGE_REWRITE:
                data["model"] = LARGE_REWRITE[effective_model]
                log.info(
                    "LARGE  %s → %s  payload=%d chars",
                    effective_model, data["model"], payload_chars
                )

            # ── Ollama health-gated bypass ──
            # If the chosen alias has Ollama and all Ollama hosts are down,
            # rewrite to _cloud variant to avoid 60-85s dead-host timeouts.
            # For explicit local/ollama requests, fail fast with 503.
            effective_model = data.get("model", "")
            is_local_request = is_local_only_alias(effective_model)

            if (is_local_request
                    and alias_ollama_all_down(effective_model)
                    and not alias_has_healthy_llama_cpp(effective_model)):
                # Explicit local request but Ollama is down — fail fast
                log.warning(
                    "OLLAMA_503  model=%s — all Ollama hosts down, returning 503",
                    effective_model
                )
                return Response(
                    content=json.dumps({
                        "error": {
                            "message": f"Local Ollama backend unavailable (all hosts down). "
                                       f"Use a cloud alias instead of '{effective_model}'.",
                            "code": 503,
                        }
                    }),
                    status_code=503,
                    media_type="application/json",
                )

            # Only aliases that have a dedicated _cloud variant get bypassed.
            # Mixed aliases like tools_large already contain cloud providers — no bypass needed.
            CLOUD_BYPASS_ALIASES = {"default", "tools", "tools_stable", "bench", "bench_stable",
                                    "swebench", "coding", "thinking"}
            if (effective_model in CLOUD_BYPASS_ALIASES
                    and alias_has_ollama(effective_model) and alias_ollama_all_down(effective_model)):
                if alias_has_healthy_llama_cpp(effective_model):
                    log.info(
                        "OLLAMA_BYPASS_SKIP  %s  reason=llama.cpp healthy",
                        effective_model
                    )
                else:
                    cloud_alias = f"{effective_model}_cloud"
                    data["model"] = cloud_alias
                    log.info(
                        "OLLAMA_BYPASS  %s → %s  reason=all Ollama hosts down",
                        effective_model, cloud_alias
                    )

            # Respect client's stream preference (OpenAI spec: default false).
            # Only default to streaming if client didn't specify.
            # Note: benchmarks and SDK clients expect JSON when stream is unset.
            if "stream" not in data:
                data["stream"] = False

            # Request streaming usage for ML training data collection
            if data.get("stream"):
                data.setdefault("stream_options", {})["include_usage"] = True

            # Repair broken conversation history (orphan tool results, null content)
            if repair_messages(data):
                log.info("REPAIR  fixed conversation history issues")

            # Repair tool schemas (add missing 'properties' for strict providers like GitHub)
            if repair_tool_schemas(data):
                log.info("REPAIR  fixed tool schemas (added missing properties)")


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

    # ── Forward to LiteLLM (with 429 → Ollama retry) ──
    # If LiteLLM returns 429 (all cloud providers exhausted), retry once with
    # tools_local (pure Ollama) before returning the error to the client.
    # This is the last-resort catch — the north star is: never 429 the client.
    OLLAMA_RETRY_ALIASES = {"tools", "tools_large", "tools_cloud", "tools_stable",
                            "bench", "bench_large", "bench_cloud", "bench_stable",
                            "swebench", "swebench_cloud", "default", "default_cloud",
                            "coding", "coding_cloud", "thinking", "thinking_cloud",
                            "terminal_bench", "big"}
    ollama_retried = False

    # ── Circuit breaker: log proactive provider skipping ──
    if is_chat:
        broken = get_circuit_broken_providers()
        if broken:
            log.warning("CIRCUIT_BREAK  providers failing >50%%: %s", broken)

    try:
        if is_stream:
            req = http_client.build_request(request.method, url, headers=headers, content=body)
            try:
                resp = await asyncio.wait_for(
                    http_client.send(req, stream=True),
                    timeout=SOFT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                # ── Soft timeout: abort slow provider, retry via Ollama ──
                elapsed_so_far = time.monotonic() - start
                current_model = ""
                try:
                    current_model = json.loads(body).get("model", "") if body else ""
                except Exception:
                    pass
                can_rescue = (is_chat and current_model in OLLAMA_RETRY_ALIASES
                              and not alias_ollama_all_down("local")
                              and _active_rescues < _MAX_CONCURRENT_RESCUES)
                if can_rescue:
                    _active_rescues += 1
                    log.warning(
                        "SOFT_TIMEOUT  %s timed out after %.0fs → retrying via tools_local",
                        chosen_alias or current_model or "?", elapsed_so_far
                    )
                    try:
                        retry_data = json.loads(body)
                        retry_data["model"] = "tools_local"
                        retry_body = json.dumps(retry_data).encode()
                        req2 = http_client.build_request("POST", url, headers=headers, content=retry_body)
                        resp = await http_client.send(req2, stream=True)
                        _active_rescues = max(0, _active_rescues - 1)
                        ollama_retried = True
                        log.info("SOFT_TIMEOUT  rescue started via Ollama")
                    except Exception as e:
                        _active_rescues = max(0, _active_rescues - 1)
                        log.warning("SOFT_TIMEOUT  rescue failed: %s", e)
                        raise httpx.TimeoutException(f"Soft timeout after {elapsed_so_far:.0f}s")
                else:
                    raise httpx.TimeoutException(f"Soft timeout after {elapsed_so_far:.0f}s")

            # ── Streaming wrapper: captures TTFT, token counts, tool_calls from SSE ──
            _stream_meta = {
                "ttft_ms": None,       # time to first token
                "has_tool_calls": False,
                "finish_reason": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "stream_error": None,   # mid-stream error if any
                "chunks": 0,
            }
            _stream_start = time.monotonic()

            async def stream_gen():
                try:
                    async for chunk in resp.aiter_bytes():
                        _stream_meta["chunks"] += 1
                        # TTFT: time from request start to first data chunk
                        if _stream_meta["ttft_ms"] is None:
                            _stream_meta["ttft_ms"] = round((time.monotonic() - _stream_start) * 1000)
                        # Parse SSE events for usage/tool_calls (cheap — only parse data: lines)
                        try:
                            for line in chunk.decode("utf-8", errors="ignore").split("\n"):
                                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                                    continue
                                evt = json.loads(line[6:])
                                # Extract usage from final chunk
                                usage = evt.get("usage")
                                if usage:
                                    _stream_meta["prompt_tokens"] = usage.get("prompt_tokens")
                                    _stream_meta["completion_tokens"] = usage.get("completion_tokens")
                                    _stream_meta["total_tokens"] = usage.get("total_tokens")
                                # Detect tool_calls in delta
                                choices = evt.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    if delta.get("tool_calls"):
                                        _stream_meta["has_tool_calls"] = True
                                    fr = choices[0].get("finish_reason")
                                    if fr:
                                        _stream_meta["finish_reason"] = fr
                        except Exception:
                            pass  # Never break the stream for logging
                        yield chunk
                finally:
                    await resp.aclose()

            # ── 429 interception for streaming ──
            if resp.status_code == 429 and not ollama_retried and is_chat:
                original_429_resp = resp
                current_model = json.loads(body).get("model", "") if body else ""
                if current_model in OLLAMA_RETRY_ALIASES and not alias_ollama_all_down("local"):
                    ollama_retried = True
                    await resp.aclose()
                    # Flush LiteLLM cooldowns before rescue so Ollama models aren't blocked
                    await _auto_flush_if_needed(force=True)
                    retry_data = json.loads(body)
                    retry_data["model"] = "tools_local"
                    retry_body = json.dumps(retry_data).encode()
                    log.warning(
                        "429_RESCUE  %s → tools_local (Ollama retry, stream)",
                        current_model
                    )
                    try:
                        req2 = http_client.build_request(request.method, url, headers=headers, content=retry_body)
                        rescue_resp = await http_client.send(req2, stream=True)
                        if rescue_resp.status_code == 200:
                            resp = rescue_resp
                            log.info("429_RESCUE  succeeded via Ollama (stream)")
                        else:
                            await rescue_resp.aclose()
                            stage2_alias = _pick_stage2_alias(current_model)
                            log.warning("429_RESCUE  tools_local failed (status=%d), trying %s (stream)", rescue_resp.status_code, stage2_alias)
                            retry_data2 = json.loads(body)
                            retry_data2["model"] = stage2_alias
                            retry_body2 = json.dumps(retry_data2).encode()
                            try:
                                req3 = http_client.build_request(request.method, url, headers=headers, content=retry_body2)
                                rescue_resp2 = await http_client.send(req3, stream=True)
                                if rescue_resp2.status_code == 200:
                                    resp = rescue_resp2
                                    log.info("429_RESCUE  succeeded via %s (stream)", stage2_alias)
                                else:
                                    await rescue_resp2.aclose()
                                    log.warning("429_RESCUE  failed (status=%d), returning original 429 (stream)", rescue_resp2.status_code)
                            except Exception as e2:
                                log.warning("429_RESCUE  %s exception: %s (stream)", stage2_alias, e2)
                    except Exception as e:
                        log.warning("429_RESCUE  exception: %s, returning original 429 (stream)", e)

            # Auto-flush if streaming 429 reached client
            if resp.status_code == 429:
                await _auto_flush_if_needed()

            response_headers = dict(resp.headers)
            if chosen_alias:
                response_headers["X-Router-Alias"] = chosen_alias
                response_headers["X-Router-Original-Model"] = original_model or "default"
                response_headers["X-Router-Reason"] = reason or ""
                if data.get("model", "").endswith("_cloud") and not original_model.endswith("_cloud"):
                    response_headers["X-Router-Ollama-Bypass"] = "true"
            if ollama_retried:
                response_headers["X-Router-429-Rescue"] = "ollama"
            # spec-rag all-providers-cooled: tell caller when providers will exit cooldown
            _spec_rag_exhausted = chosen_alias == "spec-rag"
            if _spec_rag_exhausted and resp.status_code == 429:
                response_headers["Retry-After"] = "65"
                log.warning("ALL_COOLED  %s: rate-limited, adding Retry-After: 65", chosen_alias)
            if (_spec_rag_exhausted and resp.status_code == 500
                    and (time.monotonic() - start) < 5.0):
                response_headers["Retry-After"] = "65"
                log.warning("ALL_COOLED  %s: all providers cooled, adding Retry-After: 65", chosen_alias)

            elapsed = time.monotonic() - start
            log.info(
                "STREAM status=%d model=%s %.0fms%s",
                resp.status_code, chosen_alias or original_model or "?", elapsed * 1000,
                " (429-rescued by Ollama)" if ollama_retried else ""
            )

            # ── Training log (streaming) — deferred to after stream completes ──
            _training_log_data = None
            if training_features:
                provider = resp.headers.get("x-litellm-model-api-base", "")
                served_model = resp.headers.get("x-litellm-model-id", "")
                resp_duration = resp.headers.get("x-litellm-response-duration-ms", "")
                resp_cost = resp.headers.get("x-litellm-response-cost-original", "")
                retries = resp.headers.get("x-litellm-attempted-retries", "")
                fallbacks = resp.headers.get("x-litellm-attempted-fallbacks", "")
                _training_log_data = {
                    **training_features,
                    "routed_alias": chosen_alias,
                    "route_reason": reason,
                    "provider_base": provider,
                    "served_model": served_model,
                    "served_model_name": resolve_model_name(served_model),
                    "status": resp.status_code,
                    "stream": True,
                    "cache_hit": resp.headers.get("x-litellm-cache-key", "") != "",
                    "litellm_duration_ms": float(resp_duration) if resp_duration else None,
                    "litellm_cost": float(resp_cost) if resp_cost else None,
                    "litellm_retries": int(retries) if retries else 0,
                    "litellm_fallbacks": int(fallbacks) if fallbacks else 0,
                    "ollama_rescued": ollama_retried,
                }

            async def logging_stream_gen():
                """Wraps stream_gen to log training data AFTER stream completes."""
                async for chunk in stream_gen():
                    yield chunk
                # Stream finished — now we have all the SSE-extracted metadata
                if _training_log_data is not None:
                    stream_elapsed = time.monotonic() - start
                    _training_log_data["latency_ms"] = round(stream_elapsed * 1000)
                    # Inject SSE-extracted response data (the big win for ML)
                    _training_log_data["ttft_ms"] = _stream_meta.get("ttft_ms")
                    _training_log_data["response_has_tool_calls"] = _stream_meta.get("has_tool_calls", False)
                    _training_log_data["finish_reason"] = _stream_meta.get("finish_reason")
                    _training_log_data["prompt_tokens"] = _stream_meta.get("prompt_tokens")
                    _training_log_data["completion_tokens"] = _stream_meta.get("completion_tokens")
                    _training_log_data["total_tokens"] = _stream_meta.get("total_tokens")
                    _training_log_data["stream_chunks"] = _stream_meta.get("chunks", 0)
                    # Compute throughput if we have completion tokens and timing
                    ct = _stream_meta.get("completion_tokens")
                    ttft = _stream_meta.get("ttft_ms")
                    if ct and ttft and stream_elapsed * 1000 > ttft:
                        gen_time_s = (stream_elapsed * 1000 - ttft) / 1000
                        if gen_time_s > 0:
                            _training_log_data["tokens_per_second"] = round(ct / gen_time_s, 1)
                    log_training_sample(_training_log_data)

                    # ── Full conversation log (streaming) ──
                    # Log request + extracted response metadata for successful tool-calling streams
                    if (_training_log_data.get("status") == 200
                            and _stream_meta.get("has_tool_calls")
                            and is_chat):
                        try:
                            req_data = json.loads(body)
                            log_conversation({
                                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                                "id": uuid.uuid4().hex[:12],
                                "provider": _training_log_data.get("provider", "unknown"),
                                "model_alias": chosen_alias,
                                "latency_ms": _training_log_data["latency_ms"],
                                "prompt_tokens": _stream_meta.get("prompt_tokens"),
                                "completion_tokens": _stream_meta.get("completion_tokens"),
                                "ttft_ms": _stream_meta.get("ttft_ms"),
                                "tokens_per_second": _training_log_data.get("tokens_per_second"),
                                # Full request
                                "messages": req_data.get("messages"),
                                "tools": req_data.get("tools"),
                                "tool_choice": req_data.get("tool_choice"),
                                "temperature": req_data.get("temperature"),
                                "max_tokens": req_data.get("max_tokens") or req_data.get("max_completion_tokens"),
                                # Streaming — response metadata (can't capture full content without buffering)
                                "response_has_tool_calls": True,
                                "finish_reason": _stream_meta.get("finish_reason"),
                                "stream": True,
                            })
                        except Exception:
                            pass

            return StreamingResponse(
                logging_stream_gen(),
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
                # ── 429 interception for non-streaming ──
                # Save original 429 response in case rescue fails
                original_429_resp = resp
                current_model = json.loads(body).get("model", "") if body else ""
                if not ollama_retried and is_chat and current_model in OLLAMA_RETRY_ALIASES and not alias_ollama_all_down("local"):
                    ollama_retried = True
                    # Flush LiteLLM cooldowns before rescue so Ollama models aren't blocked
                    await _auto_flush_if_needed(force=True)
                    retry_data = json.loads(body)
                    retry_data["model"] = "tools_local"
                    retry_body = json.dumps(retry_data).encode()
                    log.warning(
                        "429_RESCUE  %s → tools_local (Ollama retry, non-stream)",
                        current_model
                    )
                    try:
                        rescue_resp = await http_client.request(request.method, url, headers=headers, content=retry_body)
                        if rescue_resp.status_code == 200:
                            resp = rescue_resp
                            status = 200
                            log_model = "tools_local"
                            log.info("429_RESCUE  succeeded via Ollama")
                        else:
                            stage2_alias = _pick_stage2_alias(current_model)
                            log.warning("429_RESCUE  tools_local failed (status=%d), trying %s", rescue_resp.status_code, stage2_alias)
                            retry_data2 = json.loads(body)
                            retry_data2["model"] = stage2_alias
                            retry_body2 = json.dumps(retry_data2).encode()
                            try:
                                rescue_resp2 = await http_client.request(request.method, url, headers=headers, content=retry_body2)
                                if rescue_resp2.status_code == 200:
                                    resp = rescue_resp2
                                    status = 200
                                    log_model = stage2_alias
                                    log.info("429_RESCUE  succeeded via %s", stage2_alias)
                                else:
                                    log.warning("429_RESCUE  failed (status=%d), returning original 429", rescue_resp2.status_code)
                            except Exception as e2:
                                log.warning("429_RESCUE  %s exception: %s, returning original 429", stage2_alias, e2)
                    except Exception as e:
                        log.warning("429_RESCUE  exception: %s, returning original 429", e)
                    elapsed = time.monotonic() - start

                if status == 429:
                    log.warning(
                        "RATE LIMITED  status=429 model=%s %.0fms — provider returned 429%s",
                        log_model, elapsed * 1000,
                        " (Ollama rescue attempted but failed)" if ollama_retried else ""
                    )
                    await _auto_flush_if_needed()
            elif status == 408 and is_chat:
                # ── 408 rescue: LiteLLM timeout, fallbacks not triggered automatically ──
                current_model = json.loads(body).get("model", "") if body else ""
                _408_aliases = pick_408_rescue_aliases(current_model)
                for rescue_alias in _408_aliases:
                    log.warning("408_RESCUE  %s → %s (provider timeout)", current_model, rescue_alias)
                    try:
                        retry_408 = json.loads(body)
                        retry_408["model"] = rescue_alias
                        rescue_resp = await http_client.request(request.method, url, headers=headers,
                                                                content=json.dumps(retry_408).encode())
                        if rescue_resp.status_code == 200:
                            resp = rescue_resp
                            status = 200
                            log_model = rescue_alias
                            resp_body_override = None
                            elapsed = time.monotonic() - start
                            log.info("408_RESCUE  succeeded via %s (%.0fms)", rescue_alias, elapsed * 1000)
                            break
                        else:
                            log.warning("408_RESCUE  %s failed (status=%d)", rescue_alias, rescue_resp.status_code)
                    except Exception as e:
                        log.warning("408_RESCUE  %s exception: %s", rescue_alias, e)
                elapsed = time.monotonic() - start

            if status >= 400 and status != 429:
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

                # ── ContextWindowExceeded rescue ──
                # LiteLLM's pre_call_checks may reject before trying large-context models.
                # Retry with tools_large which only has 128K+ models.
                CTX_RESCUE_ALIASES = {"tools", "bench", "swebench", "bench_stable", "tools_stable"}
                current_model = json.loads(body).get("model", "") if body else ""
                if ("ContextWindow" in error_msg and current_model in CTX_RESCUE_ALIASES
                        and is_chat):
                    log.warning(
                        "CTX_RESCUE  %s → tools_large (ContextWindowExceeded, retrying with large-context models)",
                        current_model
                    )
                    try:
                        retry_data = json.loads(body)
                        retry_data["model"] = "tools_large"
                        retry_body = json.dumps(retry_data).encode()
                        ctx_resp = await http_client.request(request.method, url, headers=headers, content=retry_body)
                        if ctx_resp.status_code == 200:
                            resp = ctx_resp
                            status = 200
                            log_model = "tools_large"
                            resp_body_override = None
                            error_msg = ""
                            elapsed = time.monotonic() - start
                            log.info("CTX_RESCUE  succeeded via tools_large (%.0fms)", elapsed * 1000)
                        else:
                            log.warning("CTX_RESCUE  failed (status=%d)", ctx_resp.status_code)
                    except Exception as e:
                        log.warning("CTX_RESCUE  exception: %s", e)
                    elapsed = time.monotonic() - start

                if status >= 400:
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
                # Response quality features
                resp_data = {}
                try:
                    resp_data = json.loads(resp_body)
                except Exception:
                    pass
                resp_msg = resp_data.get("choices", [{}])[0].get("message", {}) if resp_data.get("choices") else {}
                has_tool_response = bool(resp_msg.get("tool_calls"))
                finish = resp_data.get("choices", [{}])[0].get("finish_reason", "") if resp_data.get("choices") else ""
                resp_content_len = len(resp_msg.get("content", "") or "")

                log_training_sample({
                    **training_features,
                    "routed_alias": chosen_alias,
                    "route_reason": reason,
                    "provider_base": provider,
                    "served_model": served_model,
                    "served_model_name": resolve_model_name(served_model),
                    "status": status,
                    "latency_ms": round(elapsed * 1000),
                    "stream": False,
                    "cache_hit": resp.headers.get("x-litellm-cache-key", "") != "",
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    # NEW: response quality signals
                    "response_has_tool_calls": has_tool_response,
                    "response_content_length": resp_content_len,
                    "finish_reason": finish,
                    "response_tool_count": len(resp_msg.get("tool_calls", [])),
                })

                # ── Full conversation log for LLM fine-tuning ──
                # Only log successful conversations (quality training data)
                if status == 200 and is_chat:
                    try:
                        req_data = json.loads(body)
                        log_conversation({
                            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                            "id": uuid.uuid4().hex[:12],
                            "provider": training_features.get("provider", "unknown") if training_features else "unknown",
                            "model_alias": chosen_alias,
                            "latency_ms": round(elapsed * 1000),
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            # Full request
                            "messages": req_data.get("messages"),
                            "tools": req_data.get("tools"),
                            "tool_choice": req_data.get("tool_choice"),
                            "temperature": req_data.get("temperature"),
                            "max_tokens": req_data.get("max_tokens") or req_data.get("max_completion_tokens"),
                            # Full response
                            "response": resp_msg,
                            "finish_reason": finish,
                        })
                    except Exception:
                        pass  # never break request flow for logging

            # ── Inject routing metadata, fix Content-Length ──
            response_headers = {k: v for k, v in resp.headers.items()
                                if k.lower() not in ("content-length", "transfer-encoding")}
            if chosen_alias:
                response_headers["X-Router-Alias"] = chosen_alias
                response_headers["X-Router-Original-Model"] = original_model or "default"
                response_headers["X-Router-Reason"] = reason or ""
                if data.get("model", "").endswith("_cloud") and not original_model.endswith("_cloud"):
                    response_headers["X-Router-Ollama-Bypass"] = "true"
            # spec-rag all-providers-cooled: tell caller when providers will exit cooldown
            _spec_rag_exhausted = chosen_alias == "spec-rag"
            if _spec_rag_exhausted and status == 429:
                response_headers["Retry-After"] = "65"
                log.warning("ALL_COOLED  %s: rate-limited, adding Retry-After: 65", chosen_alias)
            if _spec_rag_exhausted and status == 500 and elapsed < 5.0:
                response_headers["Retry-After"] = "65"
                log.warning("ALL_COOLED  %s: all providers cooled, adding Retry-After: 65", chosen_alias)

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
        current_model = ""
        try:
            current_model = json.loads(body).get("model", "") if body else ""
        except Exception:
            pass

        # ── Timeout rescue: retry via Ollama (with concurrency limit) ──
        can_rescue = (is_chat and current_model in OLLAMA_RETRY_ALIASES
                      and not alias_ollama_all_down("local")
                      and _active_rescues < _MAX_CONCURRENT_RESCUES)
        if can_rescue:
            _active_rescues += 1
            log.warning(
                "TIMEOUT_RESCUE  %s timed out after %.0fms → retrying via tools_local (Ollama, %d/%d slots)",
                chosen_alias or original_model or "?", elapsed * 1000,
                _active_rescues, _MAX_CONCURRENT_RESCUES
            )
            try:
                retry_data = json.loads(body)
                retry_data["model"] = "tools_local"
                retry_body = json.dumps(retry_data).encode()
                rescue_resp = await http_client.request(
                    "POST", f"/{path}", headers=headers, content=retry_body
                )
                if rescue_resp.status_code == 200:
                    rescue_elapsed = time.monotonic() - start
                    log.info(
                        "TIMEOUT_RESCUE  succeeded via Ollama (%.0fms total)",
                        rescue_elapsed * 1000
                    )
                    response_headers = {k: v for k, v in rescue_resp.headers.items()
                                        if k.lower() not in ("content-length", "transfer-encoding")}
                    response_headers["X-Router-Timeout-Rescue"] = "ollama"
                    _active_rescues -= 1
                    return Response(
                        content=rescue_resp.content,
                        status_code=200,
                        headers=response_headers,
                        media_type=rescue_resp.headers.get("content-type"),
                    )
                else:
                    log.warning("TIMEOUT_RESCUE  failed (status=%d)", rescue_resp.status_code)
            except Exception as e:
                log.warning("TIMEOUT_RESCUE  exception: %s (%s)", e or repr(e), type(e).__name__)
            finally:
                _active_rescues = max(0, _active_rescues - 1)
        elif is_chat and current_model in OLLAMA_RETRY_ALIASES and _active_rescues >= _MAX_CONCURRENT_RESCUES:
            log.warning(
                "TIMEOUT_RESCUE  skipped — Ollama at capacity (%d/%d rescues active)",
                _active_rescues, _MAX_CONCURRENT_RESCUES
            )

        log.error(
            "TIMEOUT  status=504 model=%s %.0fms — LiteLLM did not respond",
            chosen_alias or original_model or "?", elapsed * 1000
        )
        # Log to training data so 504s are visible in ML metrics
        if training_features:
            log_training_sample({
                **training_features,
                "routed_alias": chosen_alias,
                "route_reason": reason,
                "provider_base": "",
                "served_model": "",
                "status": 504,
                "latency_ms": round(elapsed * 1000),
                "stream": is_stream,
                "error_category": "timeout",
                "error_msg": "Gateway timeout — all providers exhausted",
            })
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
