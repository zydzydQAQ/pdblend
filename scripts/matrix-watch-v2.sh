#!/bin/bash
# One-shot matrix progress snapshot; meant to be invoked after a sleep by the monitoring loop.
L=/home/pdblend4/results/v2/logs/eval-7b-v2-matrix.log
D=/home/pdblend4/results/v2/eval-7b-v2
echo "=== $(date '+%H:%M:%S') matrix v2 watch ==="
if docker ps --format '{{.Names}}' | grep -q '^pdb2-matrix-v2$'; then
  echo "container: up"
else
  echo "container: DOWN"
fi
done=$(find "$D" -mindepth 2 -maxdepth 2 -name summary.json 2>/dev/null | wc -l)
echo "summaries: $done/156"
grep -aE '^\[[0-9:]+\] .*: \{' "$L" 2>/dev/null | tail -2
grep -aiE 'error|failed|traceback|exception' "$L" 2>/dev/null | grep -avE '"error":|error=None' | tail -3
tail -1 "$L" 2>/dev/null
