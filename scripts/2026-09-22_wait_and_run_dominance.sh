#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
LOG="$ROOT/results/2026-09-22/pdblend-dominance-round1/matrix.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for frozen baseline matrix"
while docker ps --format '{{.Names}}' | grep -qx pdb2-matrix-v2; do
  sleep 30
done
echo "[$(date -Is)] baseline container ended; waiting for GPU cleanup"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do
  sleep 15
done
"$ROOT/scripts/2026-09-22_run_dominance_round1.sh"
python3 "$ROOT/scripts/2026-09-22_finalize_matrix_evidence.py" \
  "$ROOT/results/2026-09-22/pdblend-dominance-round1" \
  "$ROOT/results/2026-09-22/decode900/round6/profile.json" \
  "$ROOT/results/2026-09-22/pdblend-dominance-round1/spec.json" \
  --source "$ROOT/src"
python3 "$ROOT/scripts/2026-09-22_compare_dominance.py" \
  "$ROOT/results/2026-09-22/pdblend-dominance-round1" \
  "$ROOT/results/v2/eval-7b-v2" \
  "$ROOT/results/2026-09-22/pdblend-dominance-round1/compare-pdblend-dominance-round1.csv"
