# Qwen Image 2.1 / ComfyUI for RunPod

Source-only image recipe for `ghcr.io/kingwsi/comfyui-qwen21-runpod`.
An image is usable only after a successful Actions run and public GHCR package
visibility. This repository existing does not mean an image has been published.

## Build contract

- Linux amd64; immutable official RunPod PyTorch 2.8.0 CUDA 12.8 base digest.
- ComfyUI v0.38.0 and GGUF source commits are pinned in `sources.json`.
- Dependencies are version-locked in `payload-requirements.txt`, installed into a
  Python 3.12 venv during image construction. Base CUDA torch is reused.
- No host site-packages, weights, personal images, credentials or private paths.
- Official NVIDIA entrypoint and RunPod `/start.sh` remain inherited.
- Source tests run first. Build begins only with at least 40 GiB free in the
  Docker filesystem. This is a conservative admission floor, not a guarantee;
  the ~9.84 GiB compressed base expands substantially. There is no runner
  cleanup, paid runner, paid GPU or paid fallback.
- The image is built once, CPU-smoked locally without network, then pushed with
  the workflow's short-lived GITHUB_TOKEN. Contents read + packages write only.
- CPU tests are not GPU cold-start or inference validation.

## RunPod use (once publication is verified)

Select the published immutable image tag or digest, with a compatible NVIDIA
GPU. Preserve the image's default startup command. Provide JUPYTER_PASSWORD
through RunPod's existing environment settings; never put its value in Git.
Expose HTTP port 8888 only. Open Jupyter, authenticate, then open
`/proxy/8188/` on the same origin. Do not expose port 8188 publicly.

Application code lives in `/opt`; persistent models and outputs live under
`/workspace`. A mounted `/workspace` therefore does not hide the application.
On startup, after confirming CUDA, the launcher downloads four pinned public
files from Hugging Face and verifies each size and SHA-256. They total
25,960,838,712 bytes (about 24.18 GiB); reserve additional space for temporary
parts, input/output and operational headroom. Use at least 40 GB of persistent
workspace for initial testing, more for regular image generation.
`DOWNLOAD_MODELS=0` verifies already-present files without downloading.

ComfyUI logs: `/workspace/comfyui-data/comfyui.log`.
Models: `/workspace/comfyui-models`; input/output: `/workspace/comfyui-data`.
Workflows are shipped in `/opt/comfyui-reusable/workflows` and this repository.
Import a workflow in ComfyUI and supply your own input images. The two 2K
workflows have been UI-import tested; their 2K inference is not claimed verified.
Upstream model/code licenses apply; review them before using or redistributing.

## Proxy patch

Jupyter server proxy 4.6.0 decodes captured URL paths before forwarding them.
The narrowly scoped patch preserves encoded slashes for the standard localhost
port proxy, enabling ComfyUI workflow save/load. Authentication, host checks,
XSRF enforcement and WebSocket handling remain enabled. The CPU image smoke
covers authenticated HTTP, userdata save/read, WebSocket and anonymous denial.
