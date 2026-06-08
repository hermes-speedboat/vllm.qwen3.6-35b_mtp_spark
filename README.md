# vLLM DGX Spark — Qwen3.6-35B-A3B (AWQ 4-bit, MoE, MTP)

Optimized vLLM inference setup for **DGX Spark (GB10, Grace Blackwell ARM64)** with 128 GB unified memory.

## Quickstart

```bash
cd /srv/vllm
git clone https://github.com/hermes-speedboat/vllm.qwen3.6-35b_mtp_spark.git .
bash setup_vllm.sh        # uv + venv + vllm + huggingface-hub
bash download_model.sh     # ~24 GB model via snapshot_download()
bash vllm-server.sh        # start server on port 8000
```

## Performance

| Metric | Value |
|--------|-------|
| Generation throughput | 44-53 tok/s |
| Prompt throughput | 26-528 tok/s |
| MTP acceptance (1st draft) | 77-85% |
| MTP acceptance (2nd draft) | 56-71% |
| Mean acceptance length | 2.3-2.6 |
| GPU KV cache usage | <1% at 256k context |

## Architecture

- **Model:** Qwen3.6-35B-A3B (MoE, 256 experts, 8 active/token, 3B active params/token)
- **Quantization:** AWQ 4-bit (group_size=32), ~24 GB disk, ~1.5 GB active/forward
- **Format:** PyTorch safetensors (via `snapshot_download` from `huggingface_hub`)
- **Inference:** vLLM 0.22.1
- **Spec Decode:** MTP (Multi Token Prediction), 2 draft tokens
- **Hardware:** DGX Spark (GB10), 128 GB unified memory, ~1.5 TB/s bandwidth
- **Why MoE + AWQ:** 3B active params/token × 0.5 bytes = 1.5 GB moved per forward pass vs 27 GB for a comparable dense FP8 model — 10x less memory bandwidth required

## Features

- **Reasoning:** Extracted to separate `reasoning` field via `--reasoning-parser qwen3` — clean content
- **Tool calling:** Native OpenAI `tool_calls` via `--tool-call-parser qwen3_coder`
- **Vision:** On by default (Qwen3.6 has native vision encoder). Set `LANGUAGE_ONLY=true` to disable and save memory
- **GPU memory:** `GPU_MEM_UTIL=0.5` (~60 GB reserved, ~30 GB actual usage) — leaves headroom for other workloads on the DGX Spark
- **MTP:** Speculative decoding with 2 draft tokens (`--speculative-config`)
- **No CUDA graphs:** `--enforce-eager` for instant first-request response
- **Model name:** Served as `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (not filesystem path)
- **Context:** 262144 tokens (256k)

## Files

| File | Purpose |
|------|---------|
| `setup_vllm.sh` | Install uv, venv, vllm, huggingface-hub |
| `download_model.sh` | Download model via snapshot_download() |
| `vllm-server.sh` | Start vLLM with all optimized flags |
| `vllm.service` | Systemd user service file |
| `test_vllm.sh` | API functional test |

## Systemd Autostart

**Prerequisites (headless/SSH-only):**

```bash
sudo apt install dbus dbus-user-session
sudo loginctl enable-linger $USER
echo 'export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"' >> ~/.bashrc
source ~/.bashrc
```

**Install:**

```bash
mkdir -p ~/.config/systemd/user/
cp vllm.service ~/.config/systemd/user/
chmod +x /srv/vllm/vllm-server.sh
# Fix ownership if .venv was created with sudo
[ ! -d /srv/vllm/.venv ] || sudo chown -R $USER:$USER /srv/vllm/.venv
systemctl --user daemon-reload
systemctl --user enable --now vllm
```

**Troubleshooting:** If the service fails with `status=203/EXEC`, the script isn't executable (`chmod +x`) or `.venv` is owned by root. Fix both, then `systemctl --user restart vllm`.

## Hermes Configuration

```bash
hermes config set model.base_url http://192.168.66.113:8000/v1
hermes config set model.provider custom
hermes config set model.default cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
```

## Known Issues

- **GGUF not supported:** The `qwen35` architecture is not supported by the transformers GGUF parser. Use PyTorch/safetensors format.
- **AWQ needs float16:** `--dtype float16` is required. `bfloat16` (the default with `--dtype auto`) produces an error.
- **OpenWebUI 404s:** `/api/tags`, `/api/v1/models` 404s — configure OpenWebUI to use OpenAI-compatible endpoint (`http://192.168.66.113:8000/v1`) instead of Ollama.
