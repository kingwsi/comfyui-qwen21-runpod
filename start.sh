#!/usr/bin/env bash
# Start only ComfyUI. Leave RunPod's /start.sh, SSH and Jupyter startup unchanged.
set -Eeuo pipefail
umask 077

PACK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
MODEL_DIR="${MODEL_DIR:-/workspace/comfyui-models}"
RUNTIME_DIR="${RUNTIME_DIR:-/workspace/comfyui-data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
# OutfitSwap is opt-in; ordinary workflows never need this adapter.
DOWNLOAD_LORA="${DOWNLOAD_LORA:-0}"
export COMFY_DIR MODEL_DIR RUNTIME_DIR

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# == 0 ]] || die 'This wrapper takes no arguments. See README for supported environment variables.'
[[ -f "$COMFY_DIR/main.py" ]] || die 'ComfyUI is missing. Run init.sh first with the same COMFY_DIR and PYTHON_BIN.'
[[ "$MODEL_DIR" == /* && "$RUNTIME_DIR" == /* ]] || die 'MODEL_DIR and RUNTIME_DIR must be absolute paths.'
[[ "$DOWNLOAD_MODELS" == 0 || "$DOWNLOAD_MODELS" == 1 ]] || die 'DOWNLOAD_MODELS must be 0 or 1.'
[[ "$DOWNLOAD_LORA" == 0 || "$DOWNLOAD_LORA" == 1 ]] || die 'DOWNLOAD_LORA must be 0 or 1.'
command -v flock >/dev/null || die 'flock (util-linux) is required.'
mkdir -p -- "$MODEL_DIR" "$RUNTIME_DIR"/{input,output,temp,user}
exec 9>"$RUNTIME_DIR/comfyui.lock"
flock -n 9 || die 'Another start.sh instance already owns this RUNTIME_DIR.'

"$PYTHON_BIN" - <<'PY'
import socket, torch
assert torch.__version__ == '2.8.0+cu128', 'Use the documented CUDA/PyTorch base image.'
assert torch.cuda.is_available(), 'CUDA GPU is unavailable. Check the Pod/GPU runtime before downloading models.'
s=socket.socket()
try:
    s.bind(('127.0.0.1',8188))
finally:
    s.close()
print('GPU:', torch.cuda.get_device_name(0))
print('GPU VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
PY

download_args=(--manifest "$PACK_DIR/model-manifest.json" --model-dir "$MODEL_DIR")
[[ "$DOWNLOAD_MODELS" == 1 ]] || download_args+=(--verify-only)
"$PYTHON_BIN" "$PACK_DIR/scripts/download_models.py" "${download_args[@]}"
if [[ "$DOWNLOAD_LORA" == 1 ]]; then
  lora_args=(--manifest "$PACK_DIR/lora-manifest.json" --model-dir "$MODEL_DIR")
  [[ "$DOWNLOAD_MODELS" == 1 ]] || lora_args+=(--verify-only)
  "$PYTHON_BIN" "$PACK_DIR/scripts/download_models.py" "${lora_args[@]}"
  "$PYTHON_BIN" "$PACK_DIR/scripts/convert_outfit_lora.py" \
    --source "$MODEL_DIR/loras/qwen-image-2.1-outfit-swap.safetensors" \
    --base-header "$MODEL_DIR/unet/qwen-image-2.1-UC-Q8_0.gguf" \
    --output "$MODEL_DIR/loras/OutfitSwap-LoRA-GGUF-compatible.safetensors"
else
  printf 'Optional OutfitSwap LoRA disabled (DOWNLOAD_LORA=0). Base/general workflows are available.\n'
fi
printf '\nStarting ComfyUI on 127.0.0.1:8188. Use the existing authenticated Jupyter /proxy/8188/ path.\n'
cd -- "$COMFY_DIR"
exec "$PYTHON_BIN" main.py \
  --listen 127.0.0.1 --port 8188 --preview-method none \
  --models-directory "$MODEL_DIR" \
  --input-directory "$RUNTIME_DIR/input" \
  --output-directory "$RUNTIME_DIR/output" \
  --temp-directory "$RUNTIME_DIR/temp" \
  --user-directory "$RUNTIME_DIR/user"
