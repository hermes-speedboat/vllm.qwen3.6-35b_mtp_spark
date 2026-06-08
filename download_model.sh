#!/usr/bin/env bash
# download_model.sh — Qwen3.6-35B-A3B MoE AWQ 4-bit via huggingface_hub herunterladen
# Läuft im gleichen Verzeichnis wie setup_vllm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"

MODEL_DIR="${SCRIPT_DIR}/models"
MODEL_REPO="${MODEL_REPO:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"

mkdir -p "${MODEL_DIR}"

echo "=== Modell-Download ===
  Repo:   ${MODEL_REPO}
  Format: MoE AWQ 4-bit (3B aktiv / Token)
  Ziel:   ${MODEL_DIR}/
  Grösse: ~24 GB
"

MODEL_NAME="${MODEL_REPO##*/}"
if [ -d "${MODEL_DIR}/${MODEL_NAME}" ]; then
    echo "Modell existiert bereits: ${MODEL_DIR}/${MODEL_NAME}"
    du -sh "${MODEL_DIR}/${MODEL_NAME}"
    echo "Überspringe Download. Löschen und neu starten mit:
  rm -rf ${MODEL_DIR:?}/${MODEL_NAME}
  bash $0"
else
    echo "Lade ${MODEL_REPO} herunter (ca. 24 GB) ..."
    python3 -c "
from huggingface_hub import snapshot_download
import os

model_path = snapshot_download(
    repo_id='${MODEL_REPO}',
    local_dir=os.path.join('${MODEL_DIR}', '${MODEL_NAME}'),
    ignore_patterns=['*.md', '*.h5', '*.ot', '*.msgpack'],
)
print(f'Downloaded to: {model_path}')
"
    echo ""
    echo "Download abgeschlossen:"
    du -sh "${MODEL_DIR}/${MODEL_NAME}"
fi

echo ""
echo "=== Verzeichnis ${MODEL_DIR}/${MODEL_NAME} ==="
ls -lh "${MODEL_DIR}/${MODEL_NAME}" | head -20

echo ""
echo "Fertig. Server starten mit:"
echo "  bash ${SCRIPT_DIR}/vllm-server.sh"