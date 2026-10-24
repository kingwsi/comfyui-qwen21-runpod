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
`MODEL_VARIANT=turbo`, then restart the Pod.
For the first Turbo start, keep `DOWNLOAD_MODELS=1`; a previous offline setting
of `0` will correctly refuse to download the newly required files.
`MODEL_VARIANT=base` (the default) selects the original models and workflows.
No existing model or workflow is overwritten. An older published image does
not acquire this feature just by setting the environment variable.

The Turbo profile downloads the five files in `turbo-model-manifest.json`:

- AtomicChat `Qwen-Image-2.1-Turbo-Q8_0.gguf`: the Turbo denoiser.
- AtomicChat `Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf`: the modified Qwen3-VL
  text encoder from the requested repository.
- Its matching `mmproj-Qwen-Image-2.1-Turbo-Uncensored-BF16.gguf`: the vision
  component needed for image editing. It must stay beside the text encoder.
- The existing Qwen 2.1 BF16 VAE, reused when already present and verified.
- BFS Head V1.1, loaded only by the studio face/head-swap template.

All five have immutable repository revisions, exact sizes and LFS SHA-256
values. They total 18,399,249,640 bytes (about 17.14 GiB). Existing base weights
remain on disk when switching profiles, so allow additional persistent storage.
Keeping both complete profiles needs about 40.54 GiB for unique weights alone;
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
does not require a LoRA.

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
The image's CPU smoke now checks all six workflow schemas and the Turbo
nodes' runtime input/output contracts before publication. Source-level download
tests also cover retrying truncated HTTP responses from the saved byte offset.

Sources: [AtomicChat Turbo denoiser](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF),
[requested Uncensored encoder](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF),
and [official Turbo sampling configuration](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/model_index.json).
The encoder author has not tested ComfyUI and reports no measurable improvement
in sensitive-image outcomes; the name is not an image-quality guarantee.

### Bundled 1024 workflows

ComfyUI logs: `/workspace/comfyui-data/comfyui.log`.
Models: `/workspace/comfyui-models`; input/output: `/workspace/comfyui-data`.
Import JSON from `/opt/comfyui-reusable/workflows` or this repository, then supply
your own images:

- `Qwen21-General-Image-Edit-1024.json`: one image, no LoRA; longest edge 1024,
  aspect ratio preserved. The output is not necessarily a 1024×1024 square.
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
unresolved.

### Compatibility validation limits

Source unit tests and network-isolated image tests run before publication,
including the authenticated proxy HTTP/WebSocket and save/load tests.
These checks do not establish GPU inference or image quality.
No paid GPU is started by this workflow.

Upstream model/code licenses apply; review them before using or redistributing.

## Proxy patch

Jupyter server proxy 4.6.0 decodes captured URL paths before forwarding them.
The narrowly scoped patch preserves encoded slashes for the standard localhost
port proxy, enabling ComfyUI workflow save/load. Authentication, host checks,
XSRF enforcement and WebSocket handling remain enabled. The CPU image smoke
covers authenticated HTTP, userdata save/read, WebSocket and anonymous denial.

## 图片工作台（模板入口）

新版镜像包含一个简单的图片生成页面。沿用原来的 Jupyter 登录和端口，
登录后在同一域名打开 `/proxy/8188/qwen-studio/`，无需导入或切换工作流。
原来的 `/proxy/8188/` ComfyUI 界面仍然可用。不增加公开端口。

基础模式提供三个模板，Turbo 模式另提供换脸 / 换头：

- **文生图**：只填写画面描述，生成 1024×1024 图片。
- **通用编辑**：上传一张主图，填写需要修改的地方。
- **多图参考**：上传 2–3 张图片；图 1 是主图，图 2–3 是参考，填写各图的用途。
- **换脸 / 换头（Turbo）**：上传目标图片和参考头像，替换目标图中的头部与身份，可能同时改变发型。

选择模板，上传图片，填写要求，点击“生成图片”。可以拖入或粘贴图片，
查看排队/生成状态，放大结果和下载原图。每次都是独立任务，
不会拼接上一次提示词，也不会自动使用上一次生成结果。
编辑时输入图片保持比例，最长边缩放到 1024，输出尺寸跟随主图（模型会按 32 像素对齐）。
PNG、JPEG、WebP 每张最多 20 MB / 4000 万像素；上传后转为 PNG 并校正 EXIF 方向。

工作台自动跟随 `MODEL_VARIANT=base|turbo`，使用与已安装模型匹配的采样设置；
只有换脸 / 换头模板加载 BFS LoRA，其余模板不加载 LoRA。
不需要额外语言模型或新的 Python 依赖。
任务记录在 `$RUNTIME_DIR/studio/studio.sqlite3`，图片存放在原有 input/output 目录。
刷新页面会恢复当前浏览器最近一次任务，运行中的任务不会因关闭页面而取消。
网络错误后重复提交同一请求不会创建重复任务；生成失败后可以调整输入重新生成。
服务在提交时中断的任务不会自动重发，请先检查 ComfyUI 队列。

工作台面向当前 Pod 的单用户环境，沿用 Jupyter 登录保护。浏览器通过随机会话
Cookie 隔离图片与任务；清除 Cookie 后不会再显示之前的记录，但磁盘文件仍保留。
不要直接公开 8188 端口。旧镜像需要更新镜像后才能使用此页面。

本地验证：`python3 -m unittest discover -s tests -v`。HTTP 集成测试需要已有
`aiohttp` 和 `Pillow`，镜像内会强制运行；本地缺少依赖时这部分测试会跳过。
接口集成测试使用模拟 ComfyUI，不加载模型，不代表已验证 GPU 出图质量。


### BFS 换脸 / 换头（Turbo）

设置 `MODEL_VARIANT=turbo`。首次使用新镜像时，设置 `DOWNLOAD_MODELS=1`，
启动过程会下载约 260 MB 的 `bfs_head_v1.1_qwen_2.1.safetensors`，
并核对 `turbo-model-manifest.json` 内固定版本的文件大小及 SHA-256。
权重存放在 `$MODEL_DIR/loras/`，不烘焙进镜像；后续可用 `DOWNLOAD_MODELS=0`
离线校验。没有其他 LoRA 开关，也不会下载已移除的换装适配器。

选择“换脸 / 换头”，上传：

1. **目标图片**：保留身体、姿势、场景、光线和背景。
2. **参考头像**：提供人物身份、五官和发型。

默认要求会自动填好，可补充修改后点击生成。后台自动添加作者的 `head_swap:`
指令，使用 `<image1>` / `<image2>` 顺序，以 `LoraLoaderModelOnly` 强度 1.0
加载 BFS，再进入 Qwen 缓存。维持当前 Turbo 专用 8 步、Euler、CFG 1 和
ManualSigmas，不叠加作者工作流中可选的 Pruna 加速 LoRA。
输出沿用目标图比例，最长边约 1024；两张输入会按模型要求对齐尺寸。

当前仅开放 Turbo 版本；基础模式不会显示或接受此模板。
源文件头检查显示 BFS 的 176 个模块名称与尺寸均匹配固定的 AtomicChat Turbo
Q8 模型。实际 GPU 加载、人物相似度、表情保留和峰值显存仍待 RunPod 验证，
结构匹配不代表生成质量已验证。

作者指南：<https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md>。
BFS 权重固定版本为 `0ca3913ade4b4ada458d60c232354e8586c4c181`。
