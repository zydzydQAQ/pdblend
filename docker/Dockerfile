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

# /workspace is bind-mounted from the host and may contain a host .venv whose
# python symlink does not exist in the container. Point uv at the image venv.
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# ---- 依赖安装(可重建路径,替代 docker commit)----
# 锁定清单由 pdblend:v2 导出:docker run --rm pdblend:v2 /opt/venv/bin/pip freeze > requirements.txt
# torch 系列 +cu128 轮子来自 PyTorch 官方索引,其余包走 PyPI。
# 国内构建可换镜像,例如:
#   --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#   --build-arg UV_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128
COPY requirements.txt /tmp/requirements.txt
ARG UV_INDEX_URL=https://pypi.org/simple
ARG UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN uv pip install -r /tmp/requirements.txt \
    --index-url ${UV_INDEX_URL} \
    --extra-index-url ${UV_EXTRA_INDEX_URL} \
    && /opt/venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)"

WORKDIR /workspace

CMD ["/bin/bash"]