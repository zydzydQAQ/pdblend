#!/usr/bin/env bash
# 单卡(A100=GPU0)容器,约定与 run_dual_gpu.sh 相同(宿主无关)。
# 可选环境变量: IMAGE / NAME / WORKSPACE / MODELS_DIR / DATA_DIR / HF_CACHE / BIND_ADDR
# 用法: ./docker/run_a100.sh [--detach] [其他 docker run 参数]
set -euo pipefail

IMAGE="${IMAGE:-pdblend:latest}"
NAME="${NAME:-pdblend-vllm}"

docker_bin() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  elif sudo -n docker info >/dev/null 2>&1; then
    sudo -n docker "$@"
  else
    echo "Need docker access (docker group or passwordless sudo)." >&2
    exit 1
  fi
}

detach=0
extra=()
for arg in "$@"; do
  if [[ "${arg}" == "--detach" || "${arg}" == "-d" ]]; then
    detach=1
  else
    extra+=("${arg}")
  fi
done

mounts=()
[[ -n "${WORKSPACE:-}" ]] && mounts+=(-v "${WORKSPACE}:/workspace")
[[ -n "${MODELS_DIR:-}" ]] && mounts+=(-v "${MODELS_DIR}:/models")
[[ -n "${DATA_DIR:-}" ]] && mounts+=(-v "${DATA_DIR}:${DATA_DIR}")
[[ -n "${HF_CACHE:-}" ]] && mounts+=(-v "${HF_CACHE}:/root/.cache/huggingface")

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
  -p "${BIND_ADDR:-127.0.0.1}:8000:8000"
  "${mounts[@]}"
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
