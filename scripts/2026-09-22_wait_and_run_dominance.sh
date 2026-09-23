#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
RUN="$ROOT/results/2026-09-22/pdblend-dominance-round1"
LOG="$RUN/matrix.log"
LEASE=/tmp/pdblend4-dominance-watcher.lock
exec 9>"$LEASE"
flock -n 9 || { echo "dominance watcher already running"; exit 2; }
mkdir -p "$RUN"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for frozen baseline matrix"
while docker ps --format '{{.Names}}' | grep -qx pdb2-matrix-v2; do
  sleep 30
done
echo "[$(date -Is)] baseline container ended; waiting for GPU cleanup"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do
  sleep 15
done
echo "[$(date -Is)] starting dominance matrix"
"$ROOT/scripts/2026-09-22_run_dominance_round1.sh"

python3 - "$RUN/execution.json" "$ROOT/scripts/2026-09-22_audit_matrix_evidence.py" <<'PY'
import json, pathlib, subprocess, sys
run = pathlib.Path(sys.argv[1]); d = json.loads((run / 'execution.json').read_text())
cmd = [sys.executable, sys.argv[2], str(run), '--profile-sha', d['profile_sha256'],
       '--source-sha', d['source_sha256'], '--corpus-sha', d['corpus_sha256'], '--expected', '27']
raise SystemExit(subprocess.call(cmd))
PY
