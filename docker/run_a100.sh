#!/usr/bin/env bash
# Start the pdblend CUDA 12.8 image with only GPU0 (A100).
# Interactive (default):
#   ./docker/run_a100.sh
# Detached (then docker exec):
#   ./docker/run_a100.sh --detach
set -euo pipefail

IMAGE="${IMAGE:-pdblend:vllm-cu128}"
NAME="${NAME:-pdblend-vllm}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
WORKSPACE="${WORKSPACE:-/home/zyd/code/pdblend}"
DATA_DIR="${DATA_DIR:-/mnt/data}"
MODELS_DIR="${MODELS_DIR:-${HOME}/data/models}"

docker_bin() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  elif sudo -n docker info >/dev/null 2>&1; then
    sudo -n docker "$@"
  else
    echo "Need docker. If the image is already running, inside it run:" >&2
    echo "  bash /workspace/scripts/in_container.sh" >&2
    echo "To start a new container from a host terminal:" >&2
    echo "  sudo docker run --gpus '\"device=0\"' --name ${NAME} --ipc=host --shm-size=16g -it \\" >&2
    echo "    -p 8000:8000 -v ${WORKSPACE}:/workspace -v ${DATA_DIR}:${DATA_DIR} \\" >&2
    echo "    -v ${MODELS_DIR}:/models -v ${HF_CACHE}:/root/.cache/huggingface \\" >&2
    echo "    -e CUDA_VISIBLE_DEVICES=0 -e VIRTUAL_ENV=/opt/venv -e UV_PROJECT_ENVIRONMENT=/opt/venv -e HF_TOKEN ${IMAGE} bash" >&2
    exit 1
  fi
}

mkdir -p "${HF_CACHE}" "${MODELS_DIR}"

detach=0
extra=()
for arg in "$@"; do
  if [[ "${arg}" == "--detach" || "${arg}" == "-d" ]]; then
    detach=1
  else
    extra+=("${arg}")
  fi
done

if docker_bin ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "container ${NAME} already exists; starting / attaching"
  docker_bin start "${NAME}" >/dev/null
  if [[ ${detach} -eq 1 ]]; then
    exit 0
  fi
  exec docker_bin exec -it \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e VIRTUAL_ENV=/opt/venv \
    -e UV_PROJECT_ENVIRONMENT=/opt/venv \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
    "${NAME}" bash
fi

run=(
  run
  --gpus '"device=0"'
  --name "${NAME}"
  --ipc=host
  --shm-size=16g
  -p 8000:8000
  -v "${WORKSPACE}:/workspace"
  -v "${DATA_DIR}:${DATA_DIR}"
  -v "${MODELS_DIR}:/models"
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -e CUDA_VISIBLE_DEVICES=0
  -e VIRTUAL_ENV=/opt/venv
  -e UV_PROJECT_ENVIRONMENT=/opt/venv
  -e HF_TOKEN="${HF_TOKEN:-}"
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
)
if [[ ${detach} -eq 1 ]]; then
  run+=(-d --restart unless-stopped)
  exec docker_bin "${run[@]}" "${extra[@]}" "${IMAGE}" sleep infinity
fi
exec docker_bin "${run[@]}" -it "${extra[@]}" "${IMAGE}" bash
