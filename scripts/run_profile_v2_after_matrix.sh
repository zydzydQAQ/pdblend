#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
IMAGE=pdblend:l20-cu128-vllm-v1
OUT="$ROOT/results/2026-09-21/pdblend-optimization/profile-v2c"
LOG="$ROOT/results/2026-09-21/pdblend-optimization/profile-v2-queue.log"
exec >>"$LOG" 2>&1

echo "[$(date -Is)] waiting for current matrix"
while docker ps --format '{{.Names}}' | grep -qx pdb2-matrix-v2; do sleep 60; done
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do sleep 15; done
if [ -e "$OUT" ]; then
  echo "[$(date -Is)] output already exists; refusing to overwrite"
  exit 0
fi

IMAGE_DIGEST=$(docker image inspect "$IMAGE" --format '{{.Id}}')
SOURCE_HASH=$(find "$ROOT/src" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
echo "[$(date -Is)] starting profile-v2 image=$IMAGE_DIGEST source=$SOURCE_HASH"
docker run --rm --name pdb4-profile-v2c --ulimit nofile=65536:65536 \
  --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g --network host \
  -v "$ROOT:$ROOT" -v /home/models:/models \
  -e PYTHONPATH="$ROOT/src" -e PDBLEND_MODELS_DIR=/models \
  -e PDBLEND_IMAGE_DIGEST="$IMAGE_DIGEST" -e PDBLEND_SOURCE_HASH="$SOURCE_HASH" \
  -w "$ROOT" "$IMAGE" python -B -m pdblend.cli profile \
  --model Qwen2.5-7B-Instruct --gpus 0,1 \
  --freqs 900,1200,1500,1800,2100,2520 \
  --sections prefill,decode,mixed,static,transfer \
  --decode-repeats 3 --decode-settle 2 --decode-measure 5 \
  --mixed-freqs 1500,2100,2520 --out "$OUT"

echo "[$(date -Is)] running profile-v2 audit"
docker run --rm --cpus 2 --network none -v "$ROOT:$ROOT:ro" \
  -e PYTHONPATH="$ROOT/src" -w "$ROOT" "$IMAGE" \
  python -B scripts/audit_pdblend_profile.py "$OUT/profile.json" "$OUT/audit.json"
touch "$OUT/AUDIT-PASSED"
echo "[$(date -Is)] profile-v2 audit passed"
