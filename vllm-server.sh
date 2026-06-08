#!/usr/bin/env bash
# vllm-server.sh — vLLM Server für Qwen3.6-35B-A3B (MoE, AWQ 4-bit)
# DGX Spark (GB10, 128GB Unified Memory, Grace Blackwell ARM64)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"

MODEL_DIR="${SCRIPT_DIR}/models"
MODEL_REPO="${MODEL_REPO:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
MODEL_NAME="${MODEL_REPO##*/}"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

# ----------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
LANGUAGE_ONLY="${LANGUAGE_ONLY:-true}"
SP_TOK_DEFAULT=2                         # MTP default
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-${SP_TOK_DEFAULT}}

# ----------------------------------------------------------------------
# Prüfen ob Modell existiert
# ----------------------------------------------------------------------
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Modell nicht gefunden: ${MODEL_PATH}"
    echo "Zuerst download_model.sh ausführen."
    echo "  bash ${SCRIPT_DIR}/download_model.sh"
    exit 1
fi

# ----------------------------------------------------------------------
# GPU-Info
# ----------------------------------------------------------------------
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(kein nvidia-smi)"
echo ""

# ----------------------------------------------------------------------
# Server starten
# ----------------------------------------------------------------------
echo "=== vLLM Server starten ==="
echo "  Modell: ${MODEL_PATH}"
echo "  Port:   ${HOST}:${PORT}"
echo "  Kontext: ${MAX_MODEL_LEN}"
echo "  GPU Mem: ${GPU_MEM_UTIL}"
echo "  Quant:  AWQ 4-bit"
echo "  MTP:    ${NUM_SPEC_TOKENS} Draft-Tokens"
echo ""

ARGS=(
    "${MODEL_PATH}"
    "--port" "${PORT}"
    "--host" "${HOST}"
    "--max-model-len" "${MAX_MODEL_LEN}"
    "--gpu-memory-utilization" "${GPU_MEM_UTIL}"
    "--api-key" ""
    "--trust-remote-code"
    "--quantization" "awq"
    "--dtype" "float16"
    "--default-chat-template-kwargs" '{"enable_thinking": false}'
)

# Vision-Encoder deaktivieren (Text-only)
if [ "${LANGUAGE_ONLY}" = "true" ]; then
    ARGS+=("--language-model-only")
fi

# MTP (Multi Token Prediction)
if [ "${NUM_SPEC_TOKENS}" -gt 0 ]; then
    ARGS+=("--speculative-config" '{"method":"mtp","num_speculative_tokens":'"${NUM_SPEC_TOKENS}"'}')
fi

echo "> vllm serve ${ARGS[*]}"
vllm serve "${ARGS[@]}"

echo ""
echo "Server beendet."
