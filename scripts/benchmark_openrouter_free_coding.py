#!/usr/bin/env python3
"""Benchmark OpenRouter free models on a compact Python coding task."""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODELS_URL = "https://openrouter.ai/api/v1/models"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OUTPUT = Path("logs/openrouter_free_coding_benchmark.json")
DEFAULT_TITLE = "unified-llm-api free model coding benchmark"

PROMPT = """Return only Python code, no Markdown and no explanation.

Define this function exactly:
def longest_balanced_abc(s: str) -> tuple[int, int, int]:

Input: s contains only the characters A, B, and C.
Return (start, end, length) for the longest non-empty contiguous substring where
the counts of A, B, and C are all equal. end is exclusive. If multiple substrings
have the same maximum length, return the one with the smallest start index. If no
non-empty balanced substring exists, return (-1, -1, 0).

Examples:
longest_balanced_abc("ABC") == (0, 3, 3)
longest_balanced_abc("AABBC") == (1, 4, 3)
longest_balanced_abc("AAAA") == (-1, -1, 0)

The implementation should be O(n) time and O(n) space or better.
"""

FORBIDDEN_NAMES = {
    "__import__",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "os",
    "pathlib",
    "setattr",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "vars",
}


def request_json(url: str, api_key: str | None = None, body: dict[str, Any] | None = None, timeout: int = 90) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "HTTP-Referer": "http://localhost:4000",
        "X-Title": DEFAULT_TITLE,
    }
    data = None
    if body is not None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for chat requests")
        data = json.dumps(body).encode("utf-8")
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    # The caller only passes fixed OpenRouter HTTPS endpoints.
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def load_free_models() -> list[dict[str, Any]]:
    data = request_json(MODELS_URL)
    models: list[dict[str, Any]] = []
    for model in data.get("data", []):
        model_id = model.get("id", "")
        if not model_id.endswith(":free"):
            continue
        architecture = model.get("architecture") or {}
        models.append(
            {
                "id": model_id,
                "name": model.get("name"),
                "description": model.get("description"),
                "context_length": model.get("context_length"),
                "input_modalities": architecture.get("input_modalities") or [],
                "output_modalities": architecture.get("output_modalities") or [],
                "supported_parameters": model.get("supported_parameters") or [],
            }
        )
    return sorted(models, key=lambda item: item["id"])


def expected_result(s: str) -> tuple[int, int, int]:
    best = (-1, -1, 0)
    for start in range(len(s)):
        counts = {"A": 0, "B": 0, "C": 0}
        for end, char in enumerate(s[start:], start + 1):
            counts[char] += 1
            length = end - start
            if counts["A"] == counts["B"] == counts["C"] and length > best[2]:
                best = (start, end, length)
    return best


def build_tests() -> list[tuple[str, tuple[int, int, int]]]:
    fixed = [
        "",
        "A",
        "ABC",
        "CBA",
        "AABBC",
        "AAAA",
        "AABBCC",
        "AAABBBCCC",
        "ABCCBAAA",
        "BACACABBCC",
        "CCCAAABBB",
        "ABABCC",
        "ACBBACCAABBC",
        "BBBAAACCCABC",
        "ACACACBBB",
        "CBACBACBA",
        "AABCABCBBCCAA",
    ]
    rng = random.Random(20260503)
    generated = [
        "".join(rng.choice("ABC") for _ in range(size))
        for size in [*range(2, 15), 20, 25, 30]
        for _ in range(2)
    ]
    return [(case, expected_result(case)) for case in [*fixed, *generated]]


TESTS = build_tests()


def extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0].strip()
    function_start = text.find("def longest_balanced_abc")
    if function_start >= 0:
        return text[function_start:].strip()
    return text.strip()


def validate_ast(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            return False, "forbidden:import"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"forbidden:{node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"forbidden_attr:{node.attr}"
    return True, "ok"


def run_candidate_worker(code: str, queue: mp.Queue) -> None:
    os.environ.clear()
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    try:
        # Candidate code is AST-filtered, run with restricted builtins, and isolated in a timed child process.
        exec(code, namespace, namespace)  # nosec B102
        function = namespace.get("longest_balanced_abc")
        if not callable(function):
            queue.put({"ok": False, "passed": 0, "failed": "missing_function"})
            return
        for index, (case, want) in enumerate(TESTS):
            got = function(case)
            if got != want:
                queue.put({"ok": False, "passed": index, "failed": {"input": case, "want": want, "got": got}})
                return
        queue.put({"ok": True, "passed": len(TESTS), "failed": None})
    except Exception as exc:  # noqa: BLE001 - candidate code exceptions are benchmark output
        queue.put({"ok": False, "passed": 0, "failed": f"exception:{type(exc).__name__}:{exc}"})


def grade_code(code: str) -> dict[str, Any]:
    if not code.strip():
        return {"ok": False, "passed": 0, "total": len(TESTS), "reason": "empty_code"}
    valid, reason = validate_ast(code)
    if not valid:
        return {"ok": False, "passed": 0, "total": len(TESTS), "reason": reason}
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=run_candidate_worker, args=(code, queue))
    process.start()
    process.join(3.0)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        return {"ok": False, "passed": 0, "total": len(TESTS), "reason": "timeout"}
    if queue.empty():
        return {"ok": False, "passed": 0, "total": len(TESTS), "reason": "no_result"}
    result = queue.get()
    return {
        "ok": bool(result["ok"]),
        "passed": int(result["passed"]),
        "total": len(TESTS),
        "reason": "ok" if result["ok"] else result["failed"],
    }


def chat_completion(model_id: str, api_key: str, timeout: int, max_tokens: int) -> dict[str, Any]:
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a careful senior Python engineer."},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    start = time.time()
    try:
        data = request_json(CHAT_URL, api_key=api_key, body=body, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return {
            "status": "http_error",
            "http_status": exc.code,
            "latency_s": round(time.time() - start, 2),
            "error": exc.read().decode("utf-8", "replace")[:1200],
        }
    except Exception as exc:  # noqa: BLE001 - network/API errors are benchmark output
        return {"status": "error", "http_status": 0, "latency_s": round(time.time() - start, 2), "error": str(exc)}

    latency_s = round(time.time() - start, 2)
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    code = extract_code(content)
    return {
        "status": "ok",
        "http_status": 200,
        "served_model": data.get("model"),
        "latency_s": latency_s,
        "usage": data.get("usage") or {},
        "content_chars": len(content),
        "code_chars": len(code),
        "finish_reason": choice.get("finish_reason"),
        "grade": grade_code(code),
        "code_preview": code[:500],
    }


def metadata_score(model: dict[str, Any]) -> int:
    text = f"{model['id']} {model.get('name') or ''} {model.get('description') or ''}".lower()
    params = set(model.get("supported_parameters") or [])
    score = 0
    if any(term in text for term in ("coder", "coding", "code generation", "software engineering", "poolside")):
        score += 6
    if any(term in text for term in ("agentic", "tool", "function calling", "real-world productivity")):
        score += 2
    if "tools" in params:
        score += 2
    if "structured_outputs" in params or "response_format" in params:
        score += 1
    if int(model.get("context_length") or 0) >= 128_000:
        score += 1
    return score


def ranking_key(row: dict[str, Any]) -> tuple[float, int, float, int]:
    grade = row.get("grade") or {}
    total = grade.get("total") or len(TESTS)
    pass_rate = (grade.get("passed") or 0) / total
    return (pass_rate, metadata_score(row), -(row.get("latency_s") or 999.0), int(row.get("context_length") or 0))


def daily_limit_exhausted(row: dict[str, Any]) -> bool:
    try:
        error = json.loads(row.get("error") or "{}")
    except json.JSONDecodeError:
        return False
    headers = ((error.get("error") or {}).get("metadata") or {}).get("headers") or {}
    return headers.get("X-RateLimit-Remaining") == "0"


def reset_time(row: dict[str, Any]) -> str | None:
    try:
        error = json.loads(row.get("error") or "{}")
        reset_ms = ((error.get("error") or {}).get("metadata") or {}).get("headers", {}).get("X-RateLimit-Reset")
        if reset_ms:
            return datetime.fromtimestamp(int(reset_ms) / 1000, UTC).isoformat()
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def selected_models(models: Iterable[dict[str, Any]], requested: list[str]) -> list[dict[str, Any]]:
    if not requested:
        return list(models)
    wanted = set(requested)
    return [model for model in models if model["id"] in wanted]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-only", action="store_true", help="fetch and rank metadata without chat requests")
    parser.add_argument("--model", action="append", default=[], help="specific model ID to test; repeatable")
    parser.add_argument("--max-models", type=int, default=0, help="limit number of models tested")
    parser.add_argument("--delay", type=float, default=3.4, help="seconds between chat requests")
    parser.add_argument("--timeout", type=int, default=95)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--keep-going-on-daily-limit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not args.metadata_only and not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required unless --metadata-only is used")
    models = selected_models(load_free_models(), args.model)
    if args.max_models > 0:
        models = models[: args.max_models]

    print(f"Found {len(models)} OpenRouter :free models")
    rows: list[dict[str, Any]] = []
    for index, model in enumerate(models, 1):
        if args.metadata_only:
            row = {**model, "status": "metadata_only", "metadata_score": metadata_score(model)}
            rows.append(row)
            continue

        print(f"[{index:02d}/{len(models)}] testing {model['id']}", flush=True)
        row = {**model, **chat_completion(model["id"], api_key, args.timeout, args.max_tokens)}
        rows.append(row)
        grade = row.get("grade") or {}
        if row["status"] == "ok":
            print(f"    -> {grade.get('passed', 0)}/{grade.get('total', len(TESTS))} tests, lat={row['latency_s']}s")
        else:
            print(f"    -> {row['status']} {row.get('http_status')} {str(row.get('error', ''))[:180]}")
        if daily_limit_exhausted(row) and not args.keep_going_on_daily_limit:
            print(f"Stopping: OpenRouter free-model quota is exhausted. Reset: {reset_time(row)}")
            break
        if index != len(models):
            time.sleep(args.delay)

    ranked = sorted(rows, key=ranking_key, reverse=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": MODELS_URL,
        "test_count": len(TESTS),
        "prompt": PROMPT,
        "results": rows,
        "ranking": [row["id"] for row in ranked],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== Ranking ===")
    for index, row in enumerate(ranked, 1):
        grade = row.get("grade") or {}
        params = set(row.get("supported_parameters") or [])
        score = f"{grade.get('passed', 0)}/{grade.get('total', len(TESTS))}" if grade else f"metadata={metadata_score(row)}"
        print(
            f"{index:02d}. {row['id']} | {score} | tools={'yes' if 'tools' in params else 'no'} "
            f"| ctx={row.get('context_length')} | status={row['status']}"
        )
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
