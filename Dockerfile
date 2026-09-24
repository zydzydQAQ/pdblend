# 重建已锁定的 v3 运行环境（CUDA base 12.8.1，Torch 实际使用 cu126）
# 与当前镜像逐层相同请按 RESTART.md 导入快照；源码重建具有新的镜像 ID。
# 构建：docker build -t pdblend:pdblend4-v3-rebuilt .
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
    && python -m pip install --no-cache-dir -i ${PIP_INDEX} 'pip==22.0.2'
# 全部传递依赖来自当前镜像；不得重新解析未来版本或静默跳过 nixl。
COPY requirements/pdblend4-v3-container.lock /tmp/pdblend-container.lock
RUN pip install --no-cache-dir --no-deps -i ${PIP_INDEX} -r /tmp/pdblend-container.lock \
    && pip check
WORKDIR /workspace
COPY pyproject.toml setup.py ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -i ${PIP_INDEX} . && pip check
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
