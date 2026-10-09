FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851
LABEL org.opencontainers.image.source="https://github.com/kingwsi/comfyui-qwen21-runpod"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV COMFY_DIR=/opt/ComfyUI MODEL_DIR=/workspace/comfyui-models RUNTIME_DIR=/workspace/comfyui-data PYTHONUNBUFFERED=1
# Isolated, reproducibly installed dependencies; CUDA torch remains inherited.
RUN python -c 'import sys,torch; assert sys.version_info[:2] == (3,12); assert torch.__version__ == "2.8.0+cu128"' \
 && python -m venv --system-site-packages /opt/comfyui-venv
ENV PATH=/opt/comfyui-venv/bin:$PATH
COPY payload-requirements.txt /opt/comfyui-reusable/payload-requirements.txt
RUN python -m pip install --no-cache-dir --no-deps --ignore-installed -r /opt/comfyui-reusable/payload-requirements.txt \
 && python -m pip check
# Source-only git context. Both upstream checkouts are immutable commits.
RUN git init /opt/ComfyUI \
 && git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git \
 && git -C /opt/ComfyUI fetch --depth 1 origin 6b747c0428c343e1417219641db93a4fb7cb69ae \
 && git -C /opt/ComfyUI checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/ComfyUI rev-parse HEAD)" = 6b747c0428c343e1417219641db93a4fb7cb69ae \
 && git init /opt/ComfyUI/custom_nodes/ComfyUI-GGUF \
 && git -C /opt/ComfyUI/custom_nodes/ComfyUI-GGUF remote add origin https://github.com/leejet/ComfyUI-GGUF.git \
 && git -C /opt/ComfyUI/custom_nodes/ComfyUI-GGUF fetch --depth 1 origin 373048b8403a7820620065210a691263d4da0a61 \
 && git -C /opt/ComfyUI/custom_nodes/ComfyUI-GGUF checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/ComfyUI/custom_nodes/ComfyUI-GGUF rev-parse HEAD)" = 373048b8403a7820620065210a691263d4da0a61 \
 && rm -rf /opt/ComfyUI/.git /opt/ComfyUI/custom_nodes/ComfyUI-GGUF/.git
COPY proxy-encoded-path.patch /opt/comfyui-reusable/proxy-encoded-path.patch
COPY scripts/ /opt/comfyui-reusable/scripts/
RUN python /opt/comfyui-reusable/scripts/patch_proxy.py \
 && python -m jupyter server extension enable --py jupyter_server_proxy --sys-prefix \
 && python -c 'import torch,gguf,jupyter_server_proxy; assert torch.__version__ == "2.8.0+cu128"'
COPY start.sh sources.json model-manifest.json lora-manifest.json /opt/comfyui-reusable/
COPY workflows/ /opt/comfyui-reusable/workflows/
COPY tests/ /opt/comfyui-reusable/tests/
COPY Dockerfile README.md pre_start.sh /opt/comfyui-reusable/
COPY .github/ /opt/comfyui-reusable/.github/
RUN test ! -e /pre_start.sh
COPY pre_start.sh /pre_start.sh
RUN chmod 755 /pre_start.sh /opt/comfyui-reusable/start.sh
# Inherit the official NVIDIA ENTRYPOINT and /start.sh CMD. No model downloads here.
