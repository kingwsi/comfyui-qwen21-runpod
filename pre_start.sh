#!/usr/bin/env bash
# Called by the unchanged official RunPod startup, before authenticated Jupyter.
set -Eeuo pipefail
[[ -n "${JUPYTER_PASSWORD:-}" ]] || { echo 'JUPYTER_PASSWORD must be configured by RunPod before ComfyUI starts.' >&2; exit 1; }
mkdir -p "${RUNTIME_DIR:-/workspace/comfyui-data}"
umask 077
nohup bash /opt/comfyui-reusable/start.sh </dev/null >"${RUNTIME_DIR:-/workspace/comfyui-data}/comfyui.log" 2>&1 &
