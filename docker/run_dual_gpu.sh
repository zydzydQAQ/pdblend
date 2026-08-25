#!/usr/bin/env bash
# Start the Selective-PD image with A100 GPU0 and V100 GPU1.
set -euo pipefail

IMAGE="${IMAGE:-pdblend:vllm-cu128}"
NAME="${NAME:-pdblend-selective-pd}"
WORKSPACE="${WORKSPACE:-/home/zyd/code/pdblend}"
MODELS_DIR="${MODELS_DIR:-${HOME}/data/models}"

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

if docker_bin ps -a --format '{{.Names}}' |
  awk -v name="${NAME}" '$0 == name { found=1 } END { exit !found }'; then
  docker_bin start "${NAME}" >/dev/null
  exec docker_bin exec -it "${NAME}" bash
fi

exec docker_bin run \
  --gpus '"device=0,1"' \
  --cap-add SYS_ADMIN \
  --name "${NAME}" \
  --ipc=host \
  --shm-size=16g \
  -p 8000:8000 \
  -p 8100:8100 \
  -p 8200:8200 \
  -v "${WORKSPACE}:/workspace" \
  -v "${MODELS_DIR}:/models" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VIRTUAL_ENV=/opt/venv \
  -e UV_PROJECT_ENVIRONMENT=/opt/venv \
  -it "${IMAGE}" bash
