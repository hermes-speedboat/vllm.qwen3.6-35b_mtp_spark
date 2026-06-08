#!/usr/bin/env bash
# test_vllm.sh — vLLM Server testen (Chat-Completion + Model-Liste)
set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"

echo "=== vLLM Test ==="
echo "Server: ${BASE_URL}"
echo ""

# 1) Model-Liste
echo "--- GET /v1/models ---"
curl -s "${BASE_URL}/v1/models" | jq . || echo "(FAIL)"

echo ""

# 2) Chat-Completion
echo "--- POST /v1/chat/completions ---"
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-27B",
    "messages": [
      {"role": "system", "content": "Antworte kurz und prazise."},
      {"role": "user", "content": "Was ist 2+2? Antworte in einem Wort."}
    ],
    "max_tokens": 100,
    "temperature": 0.0
  }' | jq '.choices[0].message.content' 2>/dev/null || echo "(FAIL)"

echo ""
echo "=== Fertig ==="