#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
OUT="$ROOT/results/2026-09-21/pdblend-optimization/adaptive-calibrated"
PROFILE="$ROOT/results/2026-09-21/pdblend-optimization/calibration-2100-2520/measurement/profile.json"
LOG="$ROOT/results/2026-09-21/pdblend-optimization/adaptive-calibrated-queue.log"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for eval-7b-v2 matrix to finish"
while docker ps --format '{{.Names}}' | grep -qx pdb2-matrix-v2; do sleep 60; done
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do sleep 15; done
if [ -e "$OUT" ]; then
  echo "[$(date -Is)] output already exists; refusing to overwrite"
  exit 0
fi
echo "[$(date -Is)] starting adaptive PDblend with isolated candidate profile"
python3 "$ROOT/scripts/2026-09-21_optimization_experiment.py" bench \
  --out "$OUT" --policy pdblend --profile "$PROFILE" \
  --clocks P=2520,D=2520,M=2100 --seed 701
echo "[$(date -Is)] adaptive candidate complete"

# Resume the matrix after the isolated adaptive run. The matrix command skips
# completed summaries and therefore retries only the interrupted point onward.
if ! docker ps --format '{{.Names}}' | grep -qx pdb2-matrix-v2; then
  echo "[$(date -Is)] resuming eval-7b-v2 matrix after adaptive candidate"
  setsid nohup docker run --rm --name pdb2-matrix-v2 \
    --ulimit nofile=65536:65536 --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g \
    --network host -v "$ROOT:$ROOT" -v /home/models:/models \
    -e PYTHONPATH="$ROOT/src" -e PDBLEND_MODELS_DIR=/models -w "$ROOT" \
    pdblend:l20-cu128-vllm-v1 python -m pdblend.cli matrix "$ROOT/results/v2/eval-7b-v2/spec.json" \
    >>"$ROOT/results/v2/logs/eval-7b-v2-matrix.log" 2>&1 &
fi
