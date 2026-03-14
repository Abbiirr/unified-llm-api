#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_BASE="${API_BASE:-http://localhost:4000}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

MASTER_KEY="${LITELLM_MASTER_KEY:-}"
if [[ -z "$MASTER_KEY" ]]; then
  echo "LITELLM_MASTER_KEY is not set."
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for this smoke test."
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
RUN_ID="$(date +%s%N)"

request_model() {
  local model="$1"
  local prompt="$2"
  local headers_file="$TMP_DIR/${model//[^a-zA-Z0-9_-]/_}.headers"
  local body_file="$TMP_DIR/${model//[^a-zA-Z0-9_-]/_}.body"
  local status

  status="$(
    curl -sS \
      -o "$body_file" \
      -D "$headers_file" \
      -w "%{http_code}" \
      "$API_BASE/v1/chat/completions" \
      -H "Authorization: Bearer $MASTER_KEY" \
      -H "Content-Type: application/json" \
      -d "$(jq -nc --arg model "$model" --arg prompt "$prompt" \
        '{model:$model,messages:[{role:"user",content:$prompt}],temperature:0,max_tokens:32}')"
  )"

  printf "%s\n%s\n%s\n" "$status" "$headers_file" "$body_file"
}

print_pass() {
  printf "PASS  %s\n" "$1"
}

print_fail() {
  printf "FAIL  %s\n" "$1"
}

health_status="$(curl -sS "$API_BASE/health/readiness" | jq -r '.status')"
if [[ "$health_status" == "healthy" ]]; then
  print_pass "readiness endpoint is healthy"
else
  print_fail "readiness endpoint returned status=$health_status"
  exit 1
fi

declare -a aliases=(
  "default"
  "fast"
  "thinking"
  "coding"
  "vision"
  "tools"
  "big"
  "openrouter_free"
  "google_free"
  "cerebras_free"
  "groq_free"
  "github_free"
  "mistral_free"
  "nvidia_free"
  "cloudflare_free"
  "cohere_free"
)

for alias in "${aliases[@]}"; do
  mapfile -t result < <(request_model "$alias" "Reply with a single short word for alias $alias in run $RUN_ID.")
  status="${result[0]}"
  headers_file="${result[1]}"
  body_file="${result[2]}"
  if [[ "$status" != "200" ]]; then
    print_fail "$alias returned HTTP $status: $(jq -r '.error.message // "unknown error"' "$body_file")"
    exit 1
  fi

  provider_base="$(awk -F': ' 'tolower($1)=="x-litellm-model-api-base" {gsub("\r","",$2); print $2}' "$headers_file")"
  if [[ -z "$provider_base" ]]; then
    print_pass "$alias succeeded (LiteLLM omitted x-litellm-model-api-base)"
    continue
  fi

  print_pass "$alias routed via $provider_base"
done

mapfile -t cache_first < <(request_model "default" "Cache verification probe 2026-03-14.")
mapfile -t cache_second < <(request_model "default" "Cache verification probe 2026-03-14.")
cache_status="${cache_second[0]}"
cache_headers="${cache_second[1]}"
if [[ "$cache_status" != "200" ]]; then
  print_fail "cache verification second request failed with HTTP $cache_status"
  exit 1
fi

if grep -qi '^x-litellm-cache-key:' "$cache_headers"; then
  print_pass "Redis response cache is active"
else
  print_fail "second identical request did not expose x-litellm-cache-key"
  exit 1
fi

if [[ "${INCLUDE_LOCAL:-0}" == "1" ]]; then
  mapfile -t local_result < <(request_model "local" "Reply with the word local.")
  local_status="${local_result[0]}"
  local_body="${local_result[2]}"
  if [[ "$local_status" == "200" ]]; then
    print_pass "local alias responded"
  else
    print_fail "local alias returned HTTP $local_status: $(jq -r '.error.message // "unknown error"' "$local_body")"
    exit 1
  fi
else
  printf "SKIP  local alias smoke test (set INCLUDE_LOCAL=1 to enable)\n"
fi
