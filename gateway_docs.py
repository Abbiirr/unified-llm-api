"""
Lightweight docs endpoint for the Unified Free LLM Gateway.
Reads litellm_config.yaml and serves live API documentation.
Runs as a sidecar on port 4001.
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

import yaml

CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "litellm_config.yaml"))
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:4000")


def load_aliases():
    """Parse litellm_config.yaml and extract alias → model mappings with context windows."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        return {}

    aliases = {}
    for entry in cfg.get("model_list", []):
        name = entry.get("model_name", "")
        model = entry.get("litellm_params", {}).get("model", "")
        ctx = entry.get("model_info", {}).get("max_input_tokens", 0)
        if name and model:
            aliases.setdefault(name, {"models": [], "min_ctx": float("inf"), "max_ctx": 0})
            aliases[name]["models"].append(model)
            if ctx:
                aliases[name]["min_ctx"] = min(aliases[name]["min_ctx"], ctx)
                aliases[name]["max_ctx"] = max(aliases[name]["max_ctx"], ctx)
    # Fix inf
    for a in aliases.values():
        if a["min_ctx"] == float("inf"):
            a["min_ctx"] = 0
    return aliases


ALIAS_DETAILS = {
    "default": {
        "description": "General purpose. Tries all providers, best model first.",
        "best_for": "Simple chat, Q&A, translations, summaries",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": False,
    },
    "fast": {
        "description": "Lowest latency. Small models on hardware-accelerated infrastructure.",
        "best_for": "Speed-critical tasks, simple lookups, short responses",
        "streaming": True, "tool_calling": False, "vision": False, "thinking": False,
    },
    "thinking": {
        "description": "Deep reasoning and chain-of-thought. DeepSeek R1, QwQ, Gemini 2.5 Pro.",
        "best_for": "Math, analysis, complex reasoning, step-by-step problem solving",
        "streaming": True, "tool_calling": False, "vision": False, "thinking": True,
    },
    "coding": {
        "description": "Code generation, review, and agentic coding. Codestral, Devstral, Qwen3.",
        "best_for": "Writing code, debugging, refactoring, code review",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": False,
    },
    "vision": {
        "description": "Image and multimodal understanding. Gemini, GPT-4o, Pixtral.",
        "best_for": "Image analysis, OCR, visual Q&A, multimodal tasks",
        "streaming": True, "tool_calling": True, "vision": True, "thinking": False,
    },
    "tools": {
        "description": "Function/tool calling. Cloud-only for reliable streaming. Cerebras, Groq, NVIDIA.",
        "best_for": "Agentic workloads, multi-turn tool conversations, function calling",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": False,
        "notes": "Ollama excluded (streaming issues). 429 rescue falls back to tools_local (Ollama).",
    },
    "tools_large": {
        "description": "Large-context tool calling. 256K-1M providers for huge prompts.",
        "best_for": "Tool calling with large codebases (>65K tokens)",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": False,
        "notes": "Auto-routed when payload >6K chars. Includes Gemini (1M), Kimi K2 (262K).",
    },
    "tools_stable": {
        "description": "Reliable tool calling fallback. Cerebras + Groq.",
        "best_for": "Fallback for tools alias when primary providers fail",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": False,
    },
    "tools_local": {
        "description": "Pure Ollama. Used as 429 rescue when all cloud providers are rate-limited.",
        "best_for": "Last resort when cloud is exhausted. Unlimited capacity, slower.",
        "streaming": False, "tool_calling": True, "vision": False, "thinking": False,
        "notes": "Returns 503 if all Ollama hosts are down.",
    },
    "swebench": {
        "description": "SWE-bench tasks. Massive context (up to 1M). 12 providers across 7 services.",
        "best_for": "Large repo contexts, multi-turn coding agents, SWE-bench benchmarks",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": False,
        "notes": "Includes Gemini 3 (1M), Groq Kimi K2 (262K), OpenRouter 256K models, Ollama 262K.",
    },
    "bench": {
        "description": "General benchmarking. Cerebras, Groq, NVIDIA, OpenRouter + Ollama fallback.",
        "best_for": "Running benchmark suites across diverse providers",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": False,
    },
    "bench_stable": {
        "description": "Reliable benchmarking fallback.",
        "best_for": "Fallback for bench alias",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": False,
    },
    "big": {
        "description": "Largest parameter models (120B-253B). Maximum quality.",
        "best_for": "Tasks requiring maximum model quality, complex generation",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": False,
    },
    "local": {
        "description": "Ollama only. Data never leaves the machine.",
        "best_for": "Privacy-sensitive work, confidential data, air-gapped environments",
        "streaming": False, "tool_calling": True, "vision": False, "thinking": True,
        "notes": "Returns 503 if all Ollama hosts are down. Slower than cloud.",
    },
    "terminal_bench": {
        "description": "Terminal/CLI coding benchmark. Strongest reasoning + code models, 10 providers.",
        "best_for": "Terminal-based coding agents, CLI tool use, shell scripting, SWE benchmarks",
        "streaming": True, "tool_calling": True, "vision": False, "thinking": True,
        "notes": "Led by Cerebras gpt-oss-120b, Gemini 3.1 Pro, Mistral Small 4, Nemotron-3-nano. Ollama fallback.",
    },
}

PROVIDER_ALIASES = {
    "openrouter_free", "google_free", "cerebras_free", "groq_free",
    "github_free", "mistral_free", "nvidia_free", "cloudflare_free", "cohere_free",
}


def build_docs():
    aliases = load_aliases()

    task_aliases = {}
    for name, data in aliases.items():
        if name in PROVIDER_ALIASES:
            continue
        details = ALIAS_DETAILS.get(name, {})
        providers = sorted(set(m.split("/")[0] for m in data["models"] if "/" in m))
        task_aliases[name] = {
            "description": details.get("description", ""),
            "best_for": details.get("best_for", ""),
            "models": data["models"],
            "model_count": len(data["models"]),
            "context_window": {
                "min_tokens": data["min_ctx"],
                "max_tokens": data["max_ctx"],
            },
            "capabilities": {
                "streaming": details.get("streaming", True),
                "tool_calling": details.get("tool_calling", False),
                "vision": details.get("vision", False),
                "thinking": details.get("thinking", False),
            },
            "providers": providers,
            "notes": details.get("notes", ""),
        }

    provider_aliases = {}
    for name, data in aliases.items():
        if name in PROVIDER_ALIASES:
            provider_aliases[name] = {
                "description": f"Direct access to {name.replace('_free', '').title()} provider models for testing.",
                "models": data["models"],
                "model_count": len(data["models"]),
            }

    return {
        "name": "Unified Free LLM Gateway",
        "version": "4.0",
        "description": "OpenAI-compatible API gateway aggregating 9 free LLM providers with automatic failover and caching.",
        "base_url": f"{GATEWAY_URL}/v1",
        "endpoints": {
            "chat": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "description": "Send a chat completion request. Works exactly like the OpenAI API.",
                "authentication": "Bearer token in Authorization header",
                "example": {
                    "curl": (
                        f'curl {GATEWAY_URL}/v1/chat/completions \\\n'
                        f'  -H "Content-Type: application/json" \\\n'
                        f'  -H "Authorization: Bearer YOUR_KEY" \\\n'
                        f'  -d \'{{"model": "default", "messages": [{{"role": "user", "content": "Hello!"}}]}}\''
                    ),
                    "python": (
                        'from openai import OpenAI\n'
                        f'client = OpenAI(base_url="{GATEWAY_URL}/v1", api_key="YOUR_KEY")\n'
                        'response = client.chat.completions.create(\n'
                        '    model="default",\n'
                        '    messages=[{"role": "user", "content": "Hello!"}]\n'
                        ')'
                    ),
                },
            },
            "models": {
                "method": "GET",
                "path": "/v1/models",
                "description": "List all available model aliases.",
            },
            "health": {
                "method": "GET",
                "path": "/health/readiness",
                "description": "Check if the gateway is ready to accept requests.",
            },
        },
        "model_aliases": task_aliases,
        "provider_aliases": provider_aliases,
        "usage_guide": {
            "choosing_an_alias": {
                "plain_text": "Use 'default' or 'fast'",
                "images_attached": "Use 'vision'",
                "tool_or_function_calls": "Use 'tools'",
                "deep_reasoning_needed": "Use 'thinking'",
                "code_generation": "Use 'coding'",
                "maximum_quality": "Use 'big'",
                "privacy_sensitive": "Use 'local' (Ollama, no cloud)",
            },
            "how_routing_works": [
                "1. You pick a model alias (e.g. 'default', 'coding').",
                "2. The gateway resolves it to a list of provider+model deployments.",
                "3. It picks the deployment with the lowest recent latency.",
                "4. If that provider fails (rate limit, timeout, error), it retries with the next fastest.",
                "5. Up to 6 retries across different providers.",
                "6. Identical requests are cached in Redis for 5 hours.",
            ],
            "notes": [
                "All responses follow the OpenAI chat completion format.",
                "Streaming is supported (set stream: true).",
                "Some reasoning models return thinking tokens in <think> tags within the content field.",
                "The gateway does NOT auto-detect request type. Pick the right alias for your use case.",
            ],
        },
        "providers": [
            {"name": "OpenRouter", "free_limit": "50-1K req/day"},
            {"name": "Google AI Studio", "free_limit": "5-15 RPM per model"},
            {"name": "Cerebras", "free_limit": "30 RPM, 1M tok/day"},
            {"name": "Groq", "free_limit": "1K-14K req/day"},
            {"name": "GitHub Models", "free_limit": "Varies by Copilot tier"},
            {"name": "Mistral", "free_limit": "2 RPM, 1B tok/month"},
            {"name": "NVIDIA NIM", "free_limit": "40 RPM"},
            {"name": "Cloudflare Workers AI", "free_limit": "10K neurons/day"},
            {"name": "Cohere", "free_limit": "1K req/month"},
        ],
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Gateway — API Docs</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
         background: #0d1117; color: #c9d1d9; line-height: 1.6; padding: 2rem; max-width: 900px; margin: 0 auto; }
  h1 { color: #58a6ff; margin-bottom: 0.5rem; }
  h2 { color: #58a6ff; margin: 2rem 0 0.5rem; border-bottom: 1px solid #21262d; padding-bottom: 0.3rem; }
  h3 { color: #79c0ff; margin: 1.2rem 0 0.3rem; }
  p, li { margin-bottom: 0.4rem; }
  code { background: #161b22; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
  pre { background: #161b22; padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 0.5rem 0 1rem; }
  table { width: 100%%; border-collapse: collapse; margin: 0.5rem 0 1rem; }
  th, td { text-align: left; padding: 0.5rem 0.8rem; border-bottom: 1px solid #21262d; }
  th { color: #58a6ff; }
  .tag { display: inline-block; background: #1f6feb33; color: #58a6ff; padding: 2px 8px;
         border-radius: 4px; font-size: 0.8em; margin: 2px; }
  .subtitle { color: #8b949e; font-size: 1.1em; margin-bottom: 1.5rem; }
  a { color: #58a6ff; text-decoration: none; }
  .method { display: inline-block; background: #238636; color: white; padding: 2px 8px;
            border-radius: 4px; font-weight: bold; font-size: 0.8em; margin-right: 0.5rem; }
  .method-post { background: #1f6feb; }
</style>
</head>
<body>
<h1>Unified Free LLM Gateway</h1>
<p class="subtitle">OpenAI-compatible API &middot; 9 providers &middot; automatic failover &middot; zero cost</p>

<h2>Quick start</h2>
<pre>curl %(base_url)s/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_KEY" \\
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'</pre>

<h2>Endpoints</h2>
<table>
<tr><th>Method</th><th>Path</th><th>Description</th></tr>
<tr><td><span class="method method-post">POST</span></td><td><code>/v1/chat/completions</code></td><td>Chat completion (OpenAI-compatible)</td></tr>
<tr><td><span class="method">GET</span></td><td><code>/v1/models</code></td><td>List available model aliases</td></tr>
<tr><td><span class="method">GET</span></td><td><code>/health/readiness</code></td><td>Health check</td></tr>
<tr><td><span class="method">GET</span></td><td><code>/docs</code></td><td>This page (HTML)</td></tr>
<tr><td><span class="method">GET</span></td><td><code>/docs/json</code></td><td>Machine-readable API docs (JSON)</td></tr>
</table>

<h2>Model aliases</h2>
<p>Set <code>"model"</code> in your request to one of these:</p>
<table>
<tr><th>Alias</th><th>Best for</th><th>Context</th><th>Capabilities</th><th>Models</th></tr>
%(alias_rows)s
</table>

<h2>Which alias should I use?</h2>
<table>
<tr><th>Your request has...</th><th>Use this</th></tr>
<tr><td>Plain text</td><td><code>default</code> or <code>fast</code></td></tr>
<tr><td>Images attached</td><td><code>vision</code></td></tr>
<tr><td>Tool/function definitions</td><td><code>tools</code></td></tr>
<tr><td>Needs deep reasoning</td><td><code>thinking</code></td></tr>
<tr><td>Code generation</td><td><code>coding</code></td></tr>
<tr><td>Privacy-sensitive</td><td><code>local</code></td></tr>
</table>

<h2>How routing works</h2>
<ol>
<li>You pick a model alias (e.g. <code>default</code>, <code>coding</code>).</li>
<li>The gateway resolves it to a list of provider+model deployments.</li>
<li>It picks the one with the <strong>lowest recent latency</strong>.</li>
<li>If that provider fails (rate limit, timeout), it <strong>retries</strong> with the next fastest.</li>
<li>Up to <strong>6 retries</strong> across different providers.</li>
<li>Identical requests are <strong>cached for 5 hours</strong>.</li>
</ol>

<h2>Python example</h2>
<pre>from openai import OpenAI

client = OpenAI(base_url="%(base_url)s/v1", api_key="YOUR_KEY")
response = client.chat.completions.create(
    model="coding",
    messages=[{"role": "user", "content": "Write fizzbuzz in Python"}]
)
print(response.choices[0].message.content)</pre>

<h2>Providers</h2>
<table>
<tr><th>Provider</th><th>Free limit</th></tr>
%(provider_rows)s
</table>

<h2>Notes</h2>
<ul>
<li>All responses follow the OpenAI chat completion format.</li>
<li>Streaming is supported (<code>"stream": true</code>).</li>
<li>Some reasoning models return thinking tokens in <code>&lt;think&gt;</code> tags.</li>
<li>The gateway does NOT auto-detect request type — pick the right alias.</li>
</ul>

<p style="margin-top:2rem;color:#8b949e">
  <a href="/docs/json">JSON version</a> of these docs for programmatic access.
</p>
</body>
</html>"""


def render_html(docs):
    alias_rows = ""
    order = ["default", "tools", "swebench", "tools_large", "coding", "thinking",
             "vision", "fast", "big", "bench", "terminal_bench", "local", "tools_local"]
    for name in order:
        info = docs["model_aliases"].get(name, {})
        if not info:
            continue
        ctx = info.get("context_window", {})
        min_ctx = ctx.get("min_tokens", 0)
        max_ctx = ctx.get("max_tokens", 0)
        ctx_str = f"{max_ctx // 1024}K" if max_ctx else "?"
        if min_ctx and min_ctx != max_ctx:
            ctx_str = f"{min_ctx // 1024}K–{max_ctx // 1024}K"

        caps = info.get("capabilities", {})
        cap_tags = []
        if caps.get("streaming"): cap_tags.append("stream")
        if caps.get("tool_calling"): cap_tags.append("tools")
        if caps.get("vision"): cap_tags.append("vision")
        if caps.get("thinking"): cap_tags.append("thinking")
        cap_str = " ".join(f'<span class="tag">{t}</span>' for t in cap_tags)

        alias_rows += (
            f'<tr><td><code>{name}</code></td>'
            f'<td>{info.get("best_for", info.get("description", ""))}</td>'
            f'<td>{ctx_str}</td>'
            f'<td>{cap_str}</td>'
            f'<td>{info["model_count"]} models</td></tr>\n'
        )

    provider_rows = ""
    for p in docs["providers"]:
        provider_rows += f'<tr><td>{p["name"]}</td><td>{p["free_limit"]}</td></tr>\n'

    return HTML_TEMPLATE % {
        "base_url": GATEWAY_URL,
        "alias_rows": alias_rows,
        "provider_rows": provider_rows,
    }


class DocsHandler(BaseHTTPRequestHandler):
    docs_cache = None
    html_cache = None

    def do_GET(self):
        if DocsHandler.docs_cache is None:
            DocsHandler.docs_cache = build_docs()
            DocsHandler.html_cache = render_html(DocsHandler.docs_cache)

        if self.path == "/docs/json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(DocsHandler.docs_cache, indent=2).encode())
        elif self.path in ("/", "/docs", "/docs/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(DocsHandler.html_cache.encode())
        else:
            self.send_response(301)
            self.send_header("Location", "/docs")
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silence access logs


if __name__ == "__main__":
    port = int(os.environ.get("DOCS_PORT", "4001"))
    server = HTTPServer(("0.0.0.0", port), DocsHandler)
    print(f"Gateway docs serving on http://0.0.0.0:{port}/docs")
    server.serve_forever()
