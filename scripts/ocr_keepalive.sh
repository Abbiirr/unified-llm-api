#!/usr/bin/env bash
set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://10.112.30.10:11434}"
INTERVAL_SECONDS=${INTERVAL_SECONDS:-1500}
LOG_FILE="${LOG_FILE:-./ocr_keepalive.log}"
KEEP_ALIVE='1h'

# VRAM budget (24GB GPU):
#   qwen3.5:9b   = 14.5GB  (gateway's permanent model — not a squatter)
#   LightOnOCR-2 =  3.0GB  (fits alongside qwen3.5:9b: 17.5GB)
#   glm-ocr      =  0GB    (CPU-only, always fits)
#   deepseek-ocr =  9.5GB  (would need 27GB with both above — not kept warm)
OCR_MODELS=(
  "maternion/LightOnOCR-2:latest"
  "glm-ocr:latest"
)

CPU_MODELS=("glm-ocr:latest")

# These models are owned by the gateway and are not considered squatters.
GATEWAY_MODELS=("qwen3.5:9b")

PARENT_PID=$(awk '/^PPid:/{print $2}' /proc/$$/status | tr -d ' ')

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG_FILE"
}

require_cmd() {
  command -v "$1" >/dev/null || { echo "Missing required command: $1"; exit 1; }
}

require_cmd curl
require_cmd jq

parent_still_there() {
  local current_parent
  current_parent=$(awk '/^PPid:/{print $2}' /proc/$$/status | tr -d ' ')
  [ "$current_parent" = "$PARENT_PID" ]
}

is_cpu_model() {
  local model="$1"
  for m in "${CPU_MODELS[@]}"; do [[ "$m" == "$model" ]] && return 0; done
  return 1
}

loaded_models() {
  curl --noproxy '*' --connect-timeout 5 --max-time 20 -sS "$OLLAMA_BASE_URL/api/ps" \
    | jq -r '.models[].name' 2>/dev/null || true
}

# Unload a model by sending a real generation with keep_alive=0.
# Plain keep_alive=0 with no prompt is unreliable when the model is idle.
unload_model() {
  local model="$1"
  curl --noproxy '*' --connect-timeout 5 --max-time 30 -sS -X POST "$OLLAMA_BASE_URL/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"prompt\":\"x\",\"stream\":false,\"keep_alive\":0,\"options\":{\"num_predict\":1}}" \
    > /dev/null 2>&1 || true
}

# Unload every model currently in VRAM (including non-OCR squatters).
flush_vram() {
  local names
  names=$(loaded_models)
  if [[ -z "$names" ]]; then
    log "FLUSH: VRAM already empty"
    return
  fi
  log "FLUSH: unloading $(echo "$names" | tr '\n' ' ')"
  while IFS= read -r model; do
    [[ -z "$model" ]] && continue
    unload_model "$model"
  done <<< "$names"
  sleep 5
}

probe_model() {
  local model="$1"
  local num_gpu_opt
  if is_cpu_model "$model"; then
    num_gpu_opt=0
  else
    num_gpu_opt=99
  fi
  local resp
  resp=$(curl --noproxy '*' --connect-timeout 5 --max-time 300 -sS -X POST "$OLLAMA_BASE_URL/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"prompt\":\"OCR_OK\",\"stream\":false,\"keep_alive\":\"$KEEP_ALIVE\",\"options\":{\"num_predict\":4,\"temperature\":0.0,\"num_gpu\":$num_gpu_opt}}") || return 1

  if echo "$resp" | jq -e '.done == true and (((.error // "") | length) == 0)' >/dev/null; then
    return 0
  fi

  log "probe failed for $model: $resp"
  return 1
}

load_ocr_models() {
  for model in "${OCR_MODELS[@]}"; do
    log "LOADING: $model"
    if probe_model "$model"; then
      log "LOADED: $model"
    else
      log "ERROR: failed to load $model"
    fi
  done
}

while true; do
  if ! parent_still_there; then
    log "STOP: parent process changed; exiting keepalive loop"
    exit 0
  fi

  # Check if any true squatter (non-OCR, non-gateway) is in VRAM, or any OCR model is missing.
  current=$(loaded_models)
  ocr_set=$(printf '%s\n' "${OCR_MODELS[@]}")
  gateway_set=$(printf '%s\n' "${GATEWAY_MODELS[@]}")
  # Squatters = loaded models that are neither OCR models nor gateway-owned models.
  squatters=$(comm -23 <(echo "$current" | sort) \
    <({ echo "$ocr_set"; echo "$gateway_set"; } | sort) | grep -v '^$' || true)
  missing_ocr=""
  for m in "${OCR_MODELS[@]}"; do
    echo "$current" | grep -qxF "$m" || missing_ocr="$missing_ocr $m"
  done

  if [[ -n "$squatters" ]]; then
    log "SQUATTER: true squatter(s) in VRAM: $(echo "$squatters" | tr '\n' ' ') — flushing squatters only"
    # Flush only the squatters, leave OCR + gateway models untouched.
    while IFS= read -r model; do
      [[ -z "$model" ]] && continue
      unload_model "$model"
    done <<< "$squatters"
  fi

  if [[ -n "$missing_ocr" ]]; then
    log "MISSING:$(echo "$missing_ocr") — loading missing OCR models"
    for model in "${OCR_MODELS[@]}"; do
      echo "$current" | grep -qxF "$model" && continue   # already loaded
      log "LOADING: $model"
      if probe_model "$model"; then
        log "LOADED: $model"
      else
        log "ERROR: failed to load $model"
      fi
    done
  elif [[ -z "$squatters" ]]; then
    log "OK: all OCR models loaded, no squatters"
  else
    log "OK: squatters flushed, OCR models already warm"
  fi

  sleep "$INTERVAL_SECONDS"
done
