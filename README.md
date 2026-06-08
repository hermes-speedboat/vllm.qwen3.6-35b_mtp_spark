# vLLM DGX Spark — Qwen3.6-35B-A3B (AWQ 4-bit, MoE, MTP)

Setup für **DGX Spark (GB10, Grace Blackwell ARM64)** mit 128 GB Unified Memory.

## Setup

```bash
cd /srv/vllm
git clone https://github.com/hermes-speedboat/vllm.qwen3.6-35b_mtp_spark.git .
bash setup_vllm.sh
bash download_model.sh
bash vllm-server.sh
```

## Architektur

| Komponente | Detail |
|---|---|
| **Modell** | Qwen3.6-35B-A3B (MoE, 3B aktiv/Token) |
| **Quantisierung** | AWQ 4-bit (24 GB total, ~1.5 GB aktiv/Token) |
| **Format** | PyTorch safetensors (via huggingface_hub) |
| **Inference** | vLLM 0.22.1 (V1 Engine) |
| **Spec Decode** | MTP (Multi Token Prediction, 2 Drafts) |
| **HW** | DGX Spark, GB10, 128 GB Unified Memory |
| **Durchsatz** | 50-70 tok/s Gen, 528 tok/s Prompt |

## Features

- **Thinking deaktiviert**: `--default-chat-template-kwargs '{"enable_thinking":false}'`
- **Text-only**: Vision-Encoder aus (`--language-model-only`)
- **AWQ 4-bit**: via `--quantization awq --dtype float16`
- **MTP**: via `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
- **32k Kontext**: `--max-model-len 32768`, KV-Cache ~0.7% Auslastung
- **Systemd**: User-Service für Autostart (`vllm.service`)

## Dateien

| Datei | Beschreibung |
|---|---|
| `setup_vllm.sh` | Venv + vllm + huggingface-hub installieren |
| `download_model.sh` | Modell via `snapshot_download()` von Hugging Face |
| `vllm-server.sh` | vLLM mit optimierten Settings starten |
| `test_vllm.sh` | API-Funktionstest |
| `vllm.service` | Systemd User Service |

## Konfiguration

| Env Var | Default | Beschreibung |
|---|---|---|
| `PORT` | `8000` | Server-Port |
| `HOST` | `0.0.0.0` | Bind-Address |
| `MAX_MODEL_LEN` | `32768` | Kontextlänge (max 262144) |
| `GPU_MEM_UTIL` | `0.65` | GPU Memory-Nutzung |
| `LANGUAGE_ONLY` | `true` | Text-only (Vision aus) |
| `NUM_SPEC_TOKENS` | `2` | MTP Drafts (0 = aus) |
| `MODEL_REPO` | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | HF Repo |

## Performance auf DGX Spark

| Modell | Format | Durchsatz |
|---|---|---|
| Qwen3.6-27B | GGUF Q4_K_M | 8-13 tok/s ✗ (fehlerhaft) |
| Qwen3.6-27B | BF16 PyTorch | 10-15 tok/s |
| Qwen3.6-27B | FP8 | 20-25 tok/s |
| **Qwen3.6-35B-A3B** | **AWQ 4-bit + MTP** | **50-70+ tok/s** |

## Warum MoE + AWQ + MTP?

- **MoE (35B-A3B)**: Nur 3B aktive Parameter pro Token statt 27B → 10x weniger Memory-Bandwidth
- **AWQ 4-bit**: Accuracy-Aware Quantisierung → 4x kleiner als BF16 (24 GB total)
- **MTP**: Multi-Token Prediction generiert 2 Draft-Token pro Step → 1.5-2x Speedup

## Quellen

- [vLLM Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
- [Qwen Deployment Guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- [cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit)