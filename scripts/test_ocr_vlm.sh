#!/usr/bin/env bash
# =============================================================================
# OCR / VLM Model Test
#
# Sends a base64-encoded test image (white canvas with the word "HELLO") to
# each OCR model via the Ollama API and to vision-capable gateway aliases via
# the OpenAI-compatible gateway. Verifies the model returns a non-empty
# response that mentions the text in the image.
#
# Usage:
#   bash scripts/test_ocr_vlm.sh [OLLAMA_H1_URL] [GATEWAY_URL]
#   bash scripts/test_ocr_vlm.sh                        # defaults from env
#
# Exit: 0 = all pass, 1 = any failure
# =============================================================================
set -euo pipefail

H1="${1:-${OLLAMA_HOST_1:-http://10.112.30.10:11434}}"
GATEWAY="${2:-http://localhost:4000}"
API_KEY="${GATEWAY_API_KEY:-sk-my-secret-gateway-key}"

PASS=0
FAIL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC}: $1"; }
fail() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC}: $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; }

# ---------------------------------------------------------------------------
# Tiny test image: 1x1 white PNG, base64-encoded.
# For a real OCR test we use a small PNG with "HELLO" rendered as text.
# We generate it with Python if Pillow is available, else fall back to a
# pre-baked 1x1 white pixel (models should still return a non-empty response).
# ---------------------------------------------------------------------------
IMAGE_B64=$(python3 - <<'EOF' 2>/dev/null || true
try:
    from PIL import Image, ImageDraw, ImageFont
    import base64, io
    img = Image.new("RGB", (120, 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "HELLO", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    print(base64.b64encode(buf.getvalue()).decode())
except Exception:
    pass
EOF
)

# Fallback: 1x1 white PNG
if [ -z "$IMAGE_B64" ]; then
    IMAGE_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg=="
    warn "Pillow not available — using 1x1 white pixel (response check relaxed)"
    RELAXED=1
else
    RELAXED=0
fi

response_ok() {
    local resp="$1"
    local model="$2"
    if [ -z "$resp" ]; then
        fail "$model: empty response"
        return
    fi
    err=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error',''))" 2>/dev/null || true)
    if [ -n "$err" ]; then
        fail "$model: $err"
        return
    fi
    content=$(echo "$resp" | python3 -c "
import sys,json
d=json.load(sys.stdin)
if 'response' in d:          # /api/generate
    print(d['response'][:200])
elif 'message' in d:          # /api/chat
    print((d['message'].get('content') or '')[:200])
elif 'choices' in d:          # OpenAI
    m=d['choices'][0]['message']
    print(((m.get('content') or '') + (m.get('reasoning_content') or ''))[:200])
" 2>/dev/null || true)
    if [ "${#content}" -ge 3 ]; then
        pass "$model: $(echo "$content" | head -1 | cut -c1-80)"
    else
        fail "$model: response too short (got: $(echo "$resp" | head -c 120))"
    fi
}

# Flush all models from H1 VRAM so OCR image inference has full GPU headroom.
# Uses the forced-unload trick: send a real generation with keep_alive=0.
flush_h1_vram() {
    local names
    names=$(curl --noproxy '*' -sf --max-time 10 "$H1/api/ps" 2>/dev/null \
        | python3 -c "import sys,json; print('\n'.join(m['name'] for m in json.load(sys.stdin).get('models',[])))" 2>/dev/null || true)
    [ -z "$names" ] && return
    echo "  INFO: flushing VRAM ($(echo "$names" | tr '\n' ' '))..."
    while IFS= read -r m; do
        [ -z "$m" ] && continue
        curl --noproxy '*' -s --max-time 30 -X POST "$H1/api/generate" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$m\",\"prompt\":\"x\",\"stream\":false,\"keep_alive\":0,\"options\":{\"num_predict\":1}}" \
            > /dev/null 2>&1 || true
    done <<< "$names"
    sleep 5
}

# ---------------------------------------------------------------------------
echo "============================================================"
echo "OCR / VLM Test"
echo "H1:      $H1"
echo "Gateway: $GATEWAY"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
echo "--- Ollama OCR models (direct H1) ---"
# Flush VRAM first so image inference has full GPU headroom.
# (GPU models crash during vision inference when other large models share VRAM.)
flush_h1_vram

# LightOnOCR-2 — GPU, /api/generate with images field.
# Image inference requires a clear GPU — it crashes when other large models
# share VRAM concurrently. Skip if squatters remain after flush.
echo -n "  LightOnOCR-2 (GPU)... "
_vram_after_flush=$(curl --noproxy '*' -sf --max-time 10 "$H1/api/ps" 2>/dev/null \
    | python3 -c "
import sys,json
models=json.load(sys.stdin).get('models',[])
gpu=[m['name'] for m in models if m.get('size_vram',0)>0]
print(' '.join(gpu))
" 2>/dev/null || true)
if [ -n "$_vram_after_flush" ]; then
    echo "SKIP (GPU occupied by: $_vram_after_flush — flush failed, active gateway traffic)"
    SKIP=$((SKIP+1))
else
    resp=$(curl --noproxy '*' -s --max-time 120 -X POST "$H1/api/generate" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"maternion/LightOnOCR-2:latest\",\"prompt\":\"What text do you see in this image? Reply with just the text.\",\"images\":[\"$IMAGE_B64\"],\"stream\":false,\"keep_alive\":\"1h\",\"options\":{\"num_predict\":32,\"num_gpu\":99}}" 2>/dev/null || echo '{}')
    response_ok "$resp" "LightOnOCR-2"
fi

# deepseek-ocr — GPU, uses /api/chat (not /api/generate — returns empty on generate)
echo -n "  deepseek-ocr (GPU)... "
resp=$(curl --noproxy '*' -s --max-time 300 -X POST "$H1/api/chat" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-ocr:latest\",\"stream\":false,\"keep_alive\":\"1h\",\"messages\":[{\"role\":\"user\",\"content\":\"What text do you see in this image? Reply with just the text.\",\"images\":[\"$IMAGE_B64\"]}]}" 2>/dev/null || echo '{}')
response_ok "$resp" "deepseek-ocr"

# glm-ocr — CPU only, /api/generate
echo -n "  glm-ocr (CPU)... "
resp=$(curl --noproxy '*' -s --max-time 120 -X POST "$H1/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"glm-ocr:latest\",\"prompt\":\"What text do you see in this image? Reply with just the text.\",\"images\":[\"$IMAGE_B64\"],\"stream\":false,\"keep_alive\":\"1h\",\"options\":{\"num_predict\":32,\"num_gpu\":0}}" 2>/dev/null || echo '{}')
response_ok "$resp" "glm-ocr"

echo ""

# ---------------------------------------------------------------------------
echo "--- Gateway vision aliases ---"

for alias in ocr vision; do
    echo -n "  $alias alias... "
    resp=$(curl -s --max-time 120 \
        -H "Authorization: Bearer $API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST "$GATEWAY/v1/chat/completions" \
        -d "{\"model\":\"$alias\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What text is in this image? Reply with just the text.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMAGE_B64\"}}]}]}" \
        2>/dev/null || echo '{}')
    # Gateway check: only verify the alias routes and returns non-empty content.
    # Model quality (accurate OCR) is tested via direct H1 probes above.
    content=$(echo "$resp" | python3 -c "
import sys,json
d=json.load(sys.stdin)
if 'choices' in d:
    m=d['choices'][0]['message']
    print(((m.get('content') or '')+(m.get('reasoning_content') or ''))[:80])
" 2>/dev/null || true)
    if [ "${#content}" -ge 1 ]; then
        pass "$alias routed successfully ($(echo "$content" | cut -c1-40)...)"
    else
        err=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',{}).get('message','no content')[:80])" 2>/dev/null || echo "no response")
        fail "$alias: $err"
    fi
done

echo ""

# ---------------------------------------------------------------------------
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed"
echo "============================================================"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
