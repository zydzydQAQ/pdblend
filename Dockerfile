# 重建 pdblend:l20-cu128-vllm-v1（8×L20 + CUDA 12.8 + vLLM 0.10.1.1 + P2pNccl 补丁）
# 构建：docker build -t pdblend:l20-cu128-vllm-v1 .
# 非阿里云网络：--build-arg PIP_INDEX=https://pypi.org/simple --build-arg APT_MIRROR=archive.ubuntu.com
ARG CUDA_BASE=nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97
FROM ${CUDA_BASE}
ARG PIP_INDEX=https://mirrors.aliyun.com/pypi/simple
ARG VLLM_VERSION=0.10.1.1
ARG APT_MIRROR=mirrors.aliyun.com
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PDBLEND_MODELS_DIR=/models
RUN sed -i "s|archive.ubuntu.com|${APT_MIRROR}|g; s|security.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv python3-pip build-essential git curl \
    ca-certificates tzdata libnuma1 libibverbs1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python -m pip install --no-cache-dir -i ${PIP_INDEX} 'pip<26'
RUN pip install --no-cache-dir -i ${PIP_INDEX} "vllm==${VLLM_VERSION}" "torch==2.7.1" \
    "nvidia-ml-py==12.535.133" "aiohttp==3.12.15" "numpy==2.2.6" "PyYAML==6.0.2" "scipy==1.15.3" \
    "pytest==8.4.2" "pytest-asyncio==1.2.0" matplotlib \
    && (pip install --no-cache-dir -i ${PIP_INDEX} nixl || echo "nixl unavailable: use P2pNcclConnector")
# 不钉会解析到 transformers 5.x，删掉 vLLM 0.10.1.1 依赖的 all_special_tokens_extended
RUN pip install --no-cache-dir -i ${PIP_INDEX} "transformers==4.55.2" "tokenizers==0.21.4" "huggingface_hub==0.34.4"
WORKDIR /workspace
COPY pyproject.toml setup.py ./
COPY src ./src
RUN pip install --no-cache-dir -i ${PIP_INDEX} .
# vLLM 补丁（关键，不可省）：按 manifest 覆盖 2 个文件并做 sha256 校验
COPY engine_patches/vllm-${VLLM_VERSION} /tmp/engine_patches
RUN python - <<'EOF'
import hashlib, json, shutil, pathlib, vllm
site = pathlib.Path(vllm.__file__).parent.parent
for f in json.load(open("/tmp/engine_patches/manifest.json"))["files"]:
    dst, src = site / f["path"], pathlib.Path("/tmp/engine_patches") / f["path"]
    assert hashlib.sha256(dst.read_bytes()).hexdigest() == f["upstream_sha256"], f["path"]
    shutil.copyfile(src, dst)
    assert hashlib.sha256(dst.read_bytes()).hexdigest() == f["patched_sha256"], f["path"]
EOF
RUN rm -rf /tmp/engine_patches
CMD ["bash"]
