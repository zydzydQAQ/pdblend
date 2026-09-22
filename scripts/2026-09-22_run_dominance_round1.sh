#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
RUN=results/2026-09-22/pdblend-dominance-round1
SPEC="$ROOT/$RUN/spec.json"
PROFILE="$ROOT/results/2026-09-22/decode900/round6/profile.json"
if docker ps --format '{{.Names}}' | grep -Eq '^(pdb2-matrix-v2|pdb4-dominance-round1)$'; then
  echo "an exclusive matrix already owns the GPUs" >&2
  exit 2
fi
if [ ! -s "$PROFILE" ] || [ ! -s "$ROOT/results/2026-09-22/decode900/round6/AUDIT-PASSED" ]; then
  echo "accepted round6 profile evidence is missing" >&2
  exit 3
fi
if [ ! -s "$SPEC" ]; then
  echo "dominance spec is missing" >&2
  exit 4
fi
exec docker run --rm --name pdb4-dominance-round1 \
  --ulimit nofile=65536:65536 --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g \
  --network host -v "$ROOT:$ROOT" -v /home/models:/models \
  -e PYTHONPATH="$ROOT/src" -e PDBLEND_MODELS_DIR=/models -w "$ROOT" \
  pdblend:l20-cu128-vllm-v1 python -m pdblend.cli matrix "$SPEC"
