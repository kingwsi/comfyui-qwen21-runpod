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
By default, after confirming CUDA, the launcher downloads three pinned base
model files from Hugging Face and verifies each size and SHA-256. They total
25,801,402,224 bytes (about 24.03 GiB). No model weights are baked into the image.
Reserve additional space for temporary parts, input/output and operational
headroom. Use at least 40 GB of persistent workspace for initial testing, more
for regular image generation. `DOWNLOAD_MODELS=0` verifies already-present base
files without downloading.

### Optional Turbo model

Use a newly built image containing this change. In the Pod environment, set
`MODEL_VARIANT=turbo` and leave `DOWNLOAD_LORA=0`, then restart the Pod.
For the first Turbo start, keep `DOWNLOAD_MODELS=1`; a previous offline setting
of `0` will correctly refuse to download the newly required files.
`MODEL_VARIANT=base` (the default) selects the original models and workflows.
No existing model or workflow is overwritten. An older published image does
not acquire this feature just by setting the environment variable.

The Turbo profile downloads only the four files in `turbo-model-manifest.json`:

- AtomicChat `Qwen-Image-2.1-Turbo-Q8_0.gguf`: the Turbo denoiser.
- AtomicChat `Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf`: the modified Qwen3-VL
  text encoder from the requested repository.
- Its matching `mmproj-Qwen-Image-2.1-Turbo-Uncensored-BF16.gguf`: the vision
  component needed for image editing. It must stay beside the text encoder.
- The existing Qwen 2.1 BF16 VAE, reused when already present and verified.

All four have immutable repository revisions, exact sizes and LFS SHA-256
values. They total 18,139,153,496 bytes (about 16.89 GiB). Existing base weights
remain on disk when switching profiles, so allow additional persistent storage.
Keeping both complete profiles needs about 40.29 GiB for unique weights alone;
use at least 60 GB of persistent workspace when retaining both plus inputs/outputs.
`DOWNLOAD_MODELS=0` verifies the selected profile without downloading.

Import the matching workflow from `/opt/comfyui-reusable/workflows`:

- `Qwen21-Turbo-General-Image-Edit-1024.json`: one image; longest edge 1024.
- `Qwen21-Turbo-Multi-Reference-2Images-1024-UI.json`: main image plus one reference.
- `Qwen21-Turbo-Multi-Reference-3Images-1024-UI.json`: main image plus two references.

The Turbo workflows retain the existing prompts, image inputs, sizing and
reference cache. The main image in multi-reference workflows is still
center-cropped to 1024×1024. They use `CLIPLoaderGGUF`, `SamplerCustom`, Euler,
CFG 1 and `ManualSigmas` with the checkpoint's exact eight-step schedule:

```text
1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0
```

Do not replace this sequence with the base workflow's `simple` scheduler.
The explicit sequence bypasses the base model's schedule calculation. Turbo
does not require a LoRA. `MODEL_VARIANT=turbo` with `DOWNLOAD_LORA=1` is refused
before downloads; the dedicated OutfitSwap adapter remains a base-only option.

Compatibility was checked against the pinned ComfyUI and GGUF source paths:
the denoiser loader strips `model.diffusion_model.` and detects Qwen Image 2.1
without architecture metadata; ComfyUI supports its fused `gate_up` layout.
The first MiB of the actual AtomicChat Q8 denoiser was inspected (265 tensor
descriptors). The pinned GGUF loader supports Qwen3-VL and finds the matching
mmproj by filename. The actual encoder (399 tensors) and vision component
(352 tensors, `qwen3vl_merger`) headers were also inspected, and all four files'
metadata was rechecked at their immutable revisions. This establishes
source/header compatibility, **not** a
successful model load or GPU inference. Turbo cold start, single/multi-image
editing quality, peak VRAM and speed remain unverified. No paid GPU was started.
The image's CPU smoke now checks all seven workflow schemas and the Turbo
nodes' runtime input/output contracts before publication. Source-level download
tests also cover retrying truncated HTTP responses from the saved byte offset.

Sources: [AtomicChat Turbo denoiser](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF),
[requested Uncensored encoder](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF),
and [official Turbo sampling configuration](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/model_index.json).
The encoder author has not tested ComfyUI and reports no measurable improvement
in sensitive-image outcomes; the name is not an image-quality guarantee.

### Optional OutfitSwap LoRA

`DOWNLOAD_LORA=0` is the default. General editing and multi-image reference
workflows do not load, download, verify, or require this LoRA. To use the dedicated
outfit workflow, set `DOWNLOAD_LORA=1` in the Pod environment before startup.
This downloads the separately pinned 159,436,488-byte original and preserves its
SHA-256 from `lora-manifest.json`. With `DOWNLOAD_MODELS=0`, the original must
already exist; it is still verified and converted locally.

A guarded runtime converter creates a separate
`loras/OutfitSwap-LoRA-GGUF-compatible.safetensors` and a provenance/hash sidecar.
It accepts only the exact pinned OutfitSwap source and checks the split GGUF
MLP tensor layout. It duplicates A and splits B (gate first, projection second),
without changing rank, strength, or the other weights. Existing output is checked
against its manifest before reuse. Original files are never overwritten;
conversion is atomic and fails closed. A failed opt-in conversion stops that
startup rather than silently loading a partially matched adapter. Set
`DOWNLOAD_LORA=0` to run the base workflows independently.

The converted copy needs approximately 160 MiB additional persistent space;
the original remains approximately 152 MiB. Neither file is uploaded or included
in Git or the image. The outfit workflow references the converted filename at
strength 1.0. This is a narrow compatibility repair, not a general LoRA converter.

### Bundled 1024 workflows

ComfyUI logs: `/workspace/comfyui-data/comfyui.log`.
Models: `/workspace/comfyui-models`; input/output: `/workspace/comfyui-data`.
Import JSON from `/opt/comfyui-reusable/workflows` or this repository, then supply
your own images:

- `Qwen21-General-Image-Edit-1024.json`: one image, no LoRA; longest edge 1024,
  aspect ratio preserved. The output is not necessarily a 1024×1024 square.
- `Qwen21-Outfit-Swap-1024.json`: person and clothing reference; optional LoRA
  required only for this workflow. Main and reference longest edges are 1024,
  with aspect ratio preserved.
- `Qwen21-Multi-Reference-2Images-1024-UI.json`: main image plus one reference,
  no LoRA. Main image is center-cropped to 1024×1024; reference keeps its aspect
  ratio with longest edge 1024. Edge content may be cropped.
- `Qwen21-Multi-Reference-3Images-1024-UI.json`: main image plus two references,
  same sizing and no LoRA. All three slots must be supplied; use the two-image
  workflow when only one reference is needed.

The previous 2K defaults have been replaced, not silently rescaled at runtime.
Workflow graph checks and CPU service tests cannot establish image quality.
The prior image passed a 1024 basic GPU generation test, but a general-edit
background instruction was not followed in one test. That quality issue remains
unresolved and is not claimed fixed by the optional LoRA repair.

### Compatibility validation limits

Offline verification of the exact OutfitSwap source and pinned GGUF layout
found 448 matched tensor keys, zero missing keys (previously 64), with all 320
unaffected tensors byte-identical. All 3,221,225,472 FP32 delta elements matched
exactly after the split. Adapter scale remains unchanged. Source unit tests and
network-isolated image tests run before publication, including the authenticated
proxy HTTP/WebSocket and save/load tests. The new conversion has **not** been
GPU-inference or image-quality tested. No paid GPU is started by this workflow.

Upstream model/code licenses apply; review them before using or redistributing.

## Proxy patch

Jupyter server proxy 4.6.0 decodes captured URL paths before forwarding them.
The narrowly scoped patch preserves encoded slashes for the standard localhost
port proxy, enabling ComfyUI workflow save/load. Authentication, host checks,
XSRF enforcement and WebSocket handling remain enabled. The CPU image smoke
covers authenticated HTTP, userdata save/read, WebSocket and anonymous denial.
