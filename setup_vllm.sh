#!/usr/bin/env bash
# setup_vllm.sh — vLLM + HuggingFace Setup auf der DGX Spark (GB10, Grace Blackwell ARM64)
# Ausführen als normaler User (z.B. hermes). Kein Root nötig.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
MODEL_DIR="${SCRIPT_DIR}/models"

echo "=== vLLM Setup für DGX Spark ==="
echo "Ziel: ${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# 1) uv installieren (falls nicht vorhanden)
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[1/5] uv installieren ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    # shellcheck disable=SC1091
    source "$HOME/.bashrc" 2>/dev/null || true
else
    echo "[1/5] uv bereits installiert: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 2) Python-Umgebung mit uv anlegen
# ---------------------------------------------------------------------------
echo "[2/5] Python-Venv mit uv anlegen ..."
uv venv "${VENV_DIR}" --python 3.12

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# 3) huggingface-hub installieren
# ---------------------------------------------------------------------------
echo "[3/5] huggingface-hub installieren ..."
uv pip install huggingface-hub

# ---------------------------------------------------------------------------
# 4) vLLM installieren (ARM64/CUDA via --torch-backend=auto)
#    vLLM >= 0.19.0 empfohlen für Qwen3.6 MTP
# ---------------------------------------------------------------------------
echo "[4/5] vLLM installieren (dauert etwas) ..."
uv pip install vllm --torch-backend=auto

# ---------------------------------------------------------------------------
# 5) Modell-Verzeichnis anlegen
# ---------------------------------------------------------------------------
echo "[5/5] Modell-Verzeichnis anlegen: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# ---------------------------------------------------------------------------
# Fertig
# ---------------------------------------------------------------------------
echo ""
echo "=== Setup abgeschlossen ==="
echo ""
echo "  Venv:   source ${VENV_DIR}/bin/activate"
echo "  Model:  ${MODEL_DIR}/"
echo ""
echo "Jetzt Modell herunterladen:"
echo "  bash ${SCRIPT_DIR}/download_model.sh"
echo ""
echo "Danach Server starten:"
echo "  bash ${SCRIPT_DIR}/vllm-server.sh"