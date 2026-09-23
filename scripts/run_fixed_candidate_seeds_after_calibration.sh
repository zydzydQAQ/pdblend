#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
OUT="$ROOT/results/2026-09-21/pdblend-optimization"
CAL=pdb4-opt-calibration-2100-2520
LOG="$OUT/fixed-m4-2100-seeds.log"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for calibration container $CAL"
while docker ps --format '{{.Names}}' | grep -qx "$CAL"; do sleep 30; done
echo "[$(date -Is)] calibration ended; waiting for all GPU processes to exit"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do sleep 10; done
for seed in 701; do
  name="fixed-m4-2100-seed-${seed}"
  echo "[$(date -Is)] starting $name"
  python3 "$ROOT/scripts/2026-09-21_optimization_experiment.py" bench \
    --out "$OUT/$name" --policy manual --layout M=4,L1=4 --clocks P=2520,D=2520,M=2100 --seed "$seed"
done
echo "[$(date -Is)] fixed candidate seed validation complete"

# Resume the original matrix from its first unfinished point with the updated
# controller. The matrix command is only started after the dedicated 8-GPU
# candidate validation has released the lease.
if ! docker ps --format '{{.Names}}' | grep -qx pdb2-matrix-v2; then
  echo "[$(date -Is)] resuming eval-7b-v2 matrix"
  setsid nohup docker run --rm --name pdb2-matrix-v2 \
    --ulimit nofile=65536:65536 --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g \
    --network host -v "$ROOT:$ROOT" -v /home/models:/models \
    -e PYTHONPATH="$ROOT/src" -e PDBLEND_MODELS_DIR=/models -w "$ROOT" \
    pdblend:l20-cu128-vllm-v1 python -m pdblend.cli matrix "$ROOT/results/v2/eval-7b-v2/spec.json" \
    >>"$ROOT/results/v2/logs/eval-7b-v2-matrix.log" 2>&1 &
fi
