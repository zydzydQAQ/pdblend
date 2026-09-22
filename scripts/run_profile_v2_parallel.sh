#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
IMAGE=pdblend:l20-cu128-vllm-v1
BASE="$ROOT/results/2026-09-21/pdblend-optimization/profile-v2-parallel"
LOGDIR="$BASE/logs"
mkdir -p "$BASE" "$LOGDIR"
exec >>"$BASE/parallel.log" 2>&1

echo "[$(date -Is)] waiting for matrix"
while docker ps --format '{{.Names}}' | grep -qx pdb2-matrix-v2; do sleep 60; done
if [ -e "$BASE/AUDIT-PASSED" ]; then
  echo "[$(date -Is)] profile already audited; refusing to overwrite"
  exit 0
fi

IMAGE_DIGEST=$(docker image inspect "$IMAGE" --format '{{.Id}}')
SOURCE_HASH=$(find "$ROOT/src" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
echo "[$(date -Is)] launching four frequency shards image=$IMAGE_DIGEST source=$SOURCE_HASH"

run_shard() {
  local name=$1 devices=$2 freqs=$3 mixed=$4 port=$5 sections=$6 out=$7
  mkdir -p "$BASE/$out"
  # Other experiments may occupy an unrelated pair. Wait only for this shard's
  # physical devices so idle pairs start immediately and all eight GPUs are used.
  while true; do
    busy=0
    IFS=',' read -r -a devs <<< "$devices"
    for d in "${devs[@]}"; do
      mem=$(nvidia-smi --id="$d" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
      if [ "${mem:-0}" -gt 100 ]; then busy=1; fi
    done
    [ "$busy" -eq 0 ] && break
    echo "[$(date -Is)] shard $name waiting for GPUs $devices"
    sleep 30
  done
  docker run --rm --name "pdb4-profile-$name" --ulimit nofile=65536:65536 \
    --gpus '"device='"$devices"'"' --cap-add SYS_ADMIN --ipc=host --shm-size=16g --network host \
    -v "$ROOT:$ROOT" -v /home/models:/models \
    -e PYTHONPATH="$ROOT/src" -e PDBLEND_MODELS_DIR=/models \
    -e PDBLEND_IMAGE_DIGEST="$IMAGE_DIGEST" -e PDBLEND_SOURCE_HASH="$SOURCE_HASH" \
    -w "$ROOT" "$IMAGE" python -B -m pdblend.cli profile \
    --model Qwen2.5-7B-Instruct --gpus 0,1 --base-port "$port" \
    --freqs "$freqs" --sections "$sections" --decode-repeats 3 \
    --decode-settle 2 --decode-measure 5 --mixed-freqs "$mixed" \
    --out "$BASE/$out" >"$LOGDIR/$name.log" 2>&1
}

run_shard 01 0,1 900,1200 9999 8100 prefill,decode,static,transfer pair-01 & p01=$!
run_shard 23 2,3 1500,1800 1500 8200 prefill,decode,mixed,static pair-23 & p23=$!
run_shard 45 4,5 2100 2100 8300 prefill,decode,mixed,static pair-45 & p45=$!
run_shard 67 6,7 2520 2520 8400 prefill,decode,mixed,static pair-67 & p67=$!

status=0
for p in "$p01" "$p23" "$p45" "$p67"; do wait "$p" || status=1; done
if [ "$status" -ne 0 ]; then
  echo "[$(date -Is)] one or more shards failed; raw diagnostics retained"
  exit "$status"
fi

# Finalization can also be rerun independently without restarting any shard.
bash "$ROOT/scripts/2026-09-22_finalize_profile_v2.sh" "$BASE"
