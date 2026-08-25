FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/root/.local/bin:/opt/venv/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    gcc-11 \
    g++-11 \
    cmake \
    ninja-build \
    ccache \
    pkg-config \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Python 3.10
RUN uv python install 3.10 && \
    uv venv /opt/venv --python 3.10 --seed

# 统一使用镜像内 venv(/opt/venv);开发模式挂载宿主代码也不会被宿主 .venv 干扰
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# ---- 依赖安装:默认最小 PD 分离依赖集,自包含、可重建 ----
# 国内网络构建建议换镜像:
#   --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#   --build-arg UV_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128
ARG REQUIREMENTS_FILE=requirements-pd.txt
ARG UV_INDEX_URL=https://pypi.org/simple
ARG UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
COPY ${REQUIREMENTS_FILE} /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt \
    --index-url ${UV_INDEX_URL} \
    --extra-index-url ${UV_EXTRA_INDEX_URL} \
    && /opt/venv/bin/python -c "import torch, vllm, aiohttp; print('torch', torch.__version__, '| vllm', vllm.__version__)"

# ---- 实验代码内置:镜像自包含,运行不依赖宿主任何目录 ----
# 需要改代码立即生效时,用宿主目录挂载覆盖:
#   WORKSPACE=/home/zyd/code/pdblend ./docker/run_dual_gpu.sh
COPY experiments /workspace/experiments
COPY scripts /workspace/scripts

WORKDIR /workspace

CMD ["/bin/bash"]
