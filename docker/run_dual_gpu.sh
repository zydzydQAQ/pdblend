#!/usr/bin/env bash
# 启动 PD 分离双卡容器(GPU0=A100, GPU1=V100)。
# 代码已内置在镜像 /workspace;以下环境变量可选(均为宿主路径,未设置则不挂载):
#   IMAGE        镜像(默认 pdblend:latest)
#   NAME         容器名(默认 pdblend-selective-pd)
#   WORKSPACE    开发模式:挂载宿主代码目录到 /workspace(覆盖镜像内置代码)
#   MODELS_DIR   模型目录 → /models(不设置则用容器内 /models,可自行下载)
#   HF_CACHE     宿主 HF 缓存 → /root/.cache/huggingface
#   BIND_ADDR    端口绑定地址(默认 127.0.0.1;需局域网访问改 0.0.0.0)
set -euo pipefail

IMAGE="${IMAGE:-pdblend:latest}"
NAME="${NAME:-pdblend-selective-pd}"

docker_bin() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  elif sudo -n docker info >/dev/null 2>&1; then
    sudo -n docker "$@"
  else
    echo "Docker access is required." >&2
    exit 1
  fi
}

mounts=()
[[ -n "${WORKSPACE:-}" ]] && mounts+=(-v "${WORKSPACE}:/workspace")
[[ -n "${MODELS_DIR:-}" ]] && mounts+=(-v "${MODELS_DIR}:/models")
[[ -n "${HF_CACHE:-}" ]] && mounts+=(-v "${HF_CACHE}:/root/.cache/huggingface")

if docker_bin ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  docker_bin start "${NAME}" >/dev/null
  exec docker_bin exec -it "${NAME}" bash
fi

exec docker_bin run \
  --gpus '"device=0,1"' \
  --cap-add SYS_ADMIN \
  --name "${NAME}" \
  --ipc=host \
  --shm-size=16g \
  -p "${BIND_ADDR:-127.0.0.1}:8000:8000" \
  -p "${BIND_ADDR:-127.0.0.1}:8100:8100" \
  -p "${BIND_ADDR:-127.0.0.1}:8200:8200" \
  "${mounts[@]}" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VIRTUAL_ENV=/opt/venv \
  -e UV_PROJECT_ENVIRONMENT=/opt/venv \
  -it "${IMAGE}" bash
