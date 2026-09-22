#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
PROFILE_CONTAINER=pdb4-opt-profile-6f
LOG="$ROOT/results/2026-09-21/pdblend-optimization/matrix-after-profile.log"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for $PROFILE_CONTAINER"
while docker ps --format '{{.Names}}' | grep -qx "$PROFILE_CONTAINER"; do sleep 60; done
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do sleep 15; done
if [ ! -s "$ROOT/results/v2/profile-7b-pdblend/profile.json" ]; then
  echo "[$(date -Is)] PDblend profile was not produced; refusing to resume matrix"
  exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx pdb2-matrix-v2; then
  echo "[$(date -Is)] matrix already running"
  exit 0
fi
echo "[$(date -Is)] resuming eval-7b-v2 matrix with regenerated PDblend profile"
setsid nohup docker run --rm --name pdb2-matrix-v2 \
  --ulimit nofile=65536:65536 --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g \
  --network host -v "$ROOT:$ROOT" -v /home/models:/models \
  -e PYTHONPATH="$ROOT/src" -e PDBLEND_MODELS_DIR=/models -w "$ROOT" \
  pdblend:l20-cu128-vllm-v1 python -m pdblend.cli matrix "$ROOT/results/v2/eval-7b-v2/spec.json" \
  >>"$ROOT/results/v2/logs/eval-7b-v2-matrix.log" 2>&1 &
