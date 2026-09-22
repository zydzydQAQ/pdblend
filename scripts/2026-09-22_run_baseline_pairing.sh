#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
SPEC=${1:?baseline pairing spec path required}
if docker ps --format '{{.Names}}' | grep -Eq '^(pdb2-matrix-v2|pdb4-dominance-round1|pdb4-baseline-pairing)$'; then
  echo "an exclusive matrix already owns the GPUs" >&2
  exit 2
fi
if [ ! -s "$ROOT/$SPEC" ] && [ ! -s "$SPEC" ]; then
  echo "pairing spec missing: $SPEC" >&2
  exit 3
fi
case "$SPEC" in
  /*) SPEC_ABS="$SPEC";;
  *) SPEC_ABS="$ROOT/$SPEC";;
esac
exec docker run --rm --name pdb4-baseline-pairing \
  --ulimit nofile=65536:65536 --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g \
  --network host -v /home/pdblend4-frozen:/home/pdblend4-frozen:ro \
  -v "$ROOT:$ROOT" -v /home/models:/models \
  -e PYTHONPATH=/home/pdblend4-frozen/src -e PDBLEND_MODELS_DIR=/models -w "$ROOT" \
  pdblend:l20-cu128-vllm-v1 python -m pdblend2.cli matrix "$SPEC_ABS"
