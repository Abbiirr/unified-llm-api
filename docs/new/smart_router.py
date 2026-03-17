"""
Smart LLM Router — thin classification layer in front of LiteLLM.

Sits at :4001, forwards to LiteLLM at :4000.
Analyzes the last user message and rewrites model → best alias.

Clients hit this instead of LiteLLM directly.
"""

import httpx
import re
import os
import time
import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI()

LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://localhost:4000")

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION RULES
# ─────────────────────────────────────────────────────────────────────────────
# Each rule: (alias, priority, pattern_or_callable)
# Higher priority wins when multiple rules match.
# If nothing matches → "default"

KEYWORD_RULES: list[tuple[str, int, re.Pattern]] = [
    # ── Coding signals ──
    ("coding", 90, re.compile(
        r"(write|fix|debug|refactor|implement|code|function|class|module|script|api|endpoint|"
        r"bug|error|exception|traceback|syntax|compile|regex|sql|query|schema|migration|"
        r"html|css|javascript|typescript|python|java|rust|go|kotlin|swift|"
        r"docker|kubernetes|k8s|yaml|config|nginx|terraform|ansible|"
        r"git|merge|rebase|commit|branch|pull.?request|"
        r"test|unittest|pytest|jest|spec|mock|stub|"
        r"algorithm|data.?structure|leetcode|hackerrank|"
        r"spring.?boot|fastapi|django|flask|react|vue|angular|next\.?js)",
        re.IGNORECASE
    )),

    # ── Reasoning / analysis signals ──
    ("smart", 80, re.compile(
        r"(explain.{0,20}(how|why|what)|analyze|compare|evaluate|critique|"
        r"pros.?and.?cons|trade.?off|design.?(decision|pattern|system)|"
        r"architecture|reason|think.?(step|through|carefully)|"
        r"math|equation|calcul|proof|theorem|derive|"
        r"research|paper|study|findings|evidence|hypothesis|"
        r"strategy|plan|approach|recommend|suggest.{0,10}(best|optimal)|"
        r"implication|consequence|root.?cause|why.?does|how.?does)",
        re.IGNORECASE
    )),

    # ── Speed-critical signals (short, simple, conversational) ──
    ("fast", 70, re.compile(
        r"^(hi|hello|hey|thanks|ok|yes|no|sure|got it|"
        r"what.?is|who.?is|when.?was|where.?is|define|"
        r"translate|summarize.{0,5}$|tldr|tl;?dr|"
        r"list|name|count|how.?many|"
        r"convert|format|rewrite.?this)",
        re.IGNORECASE
    )),

    # ── Large model signals (complex, multi-step, long-form) ──
    ("big", 60, re.compile(
        r"(write.{0,20}(essay|report|article|document|paper|proposal)|"
        r"comprehensive|thorough|detailed.?analysis|in.?depth|"
        r"multi.?step|complex.?system|large.?scale|"
        r"entire|complete|full.?(implementation|solution|guide))",
        re.IGNORECASE
    )),

    # ── Privacy signals → force local ──
    ("local", 100, re.compile(  # highest priority — explicit override
        r"(private|confidential|sensitive|secret|internal.?only|"
        r"do.?not.?share|keep.?local|offline|no.?cloud|air.?gap)",
        re.IGNORECASE
    )),
]


def classify_request(messages: list[dict]) -> str:
    """
    Analyze the last user message and return the best model alias.
    Returns: "default", "fast", "smart", "coding", "big", or "local"
    """
    # Extract the last user message text
    user_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                # multimodal: extract text parts
                user_text = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            break

    if not user_text:
        return "default"

    # ── Length-based heuristic ──
    # Very short messages (<20 chars) are almost always simple → fast
    if len(user_text.strip()) < 20 and not any(
        kw in user_text.lower() for kw in ("code", "fix", "debug", "explain", "analyze")
    ):
        return "fast"

    # ── Pattern matching ──
    matches: list[tuple[str, int]] = []
    for alias, priority, pattern in KEYWORD_RULES:
        if pattern.search(user_text):
            matches.append((alias, priority))

    if matches:
        # Highest priority wins
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0][0]

    # ── Code block detection ──
    if "```" in user_text or user_text.count("    ") > 2:
        return "coding"

    # ── Fallback: message length heuristic ──
    if len(user_text) > 1500:
        return "big"  # long prompts benefit from larger models
    if len(user_text) > 500:
        return "smart"  # medium prompts → reasoning model

    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# PROXY LOGIC
# ─────────────────────────────────────────────────────────────────────────────

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """Forward all requests to LiteLLM, with smart model rewriting on chat completions."""

    url = f"{LITELLM_BASE}/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    body = await request.body()

    # ── Smart routing: only for chat completions ──
    original_model = None
    chosen_alias = None

    if path.rstrip("/") in ("v1/chat/completions", "chat/completions") and request.method == "POST":
        try:
            data = json.loads(body)
            original_model = data.get("model", "default")

            # Only reroute if client sent "auto" or "default" or no specific alias
            if original_model in ("auto", "default", ""):
                chosen_alias = classify_request(data.get("messages", []))
                data["model"] = chosen_alias
                body = json.dumps(data).encode()
        except (json.JSONDecodeError, KeyError):
            pass  # forward as-is

    # ── Check if streaming ──
    is_stream = False
    try:
        is_stream = json.loads(body).get("stream", False) if body else False
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Forward to LiteLLM ──
    client = httpx.AsyncClient(timeout=120)

    if is_stream:
        # Stream the response back
        req = client.build_request(request.method, url, headers=headers, content=body)
        resp = await client.send(req, stream=True)

        async def stream_gen():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_gen(),
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type"),
        )
    else:
        resp = await client.request(request.method, url, headers=headers, content=body)
        await client.aclose()

        # ── Inject routing metadata in non-stream response ──
        response_headers = dict(resp.headers)
        if chosen_alias:
            response_headers["X-Router-Alias"] = chosen_alias
            response_headers["X-Router-Original-Model"] = original_model or "default"

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTROSPECTION ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/router/classify")
async def classify_only(request: Request):
    """
    Debug endpoint: classify a request without forwarding.
    POST {"messages": [{"role": "user", "content": "..."}]}
    Returns: {"alias": "coding", "reason": "matched pattern: code|function|..."}
    """
    data = await request.json()
    messages = data.get("messages", [])
    alias = classify_request(messages)
    return {"alias": alias}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4001)
