#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pdblend4
IMAGE=pdblend:l20-cu128-vllm-v1
MATRIX=pdb2-matrix-v2
OUT="$ROOT/results/2026-09-21/pdblend-optimization"
PROFILE="$ROOT/results/v2/profile-7b/profile.json"
CORPUS="$ROOT/datasets/prepared/2026-09-21-7b-v2-half"
LOG="$OUT/stage1-run.log"

mkdir -p "$OUT"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for $MATRIX to release the GPU lease"
while docker ps --format '{{.Names}}' | grep -qx "$MATRIX"; do
  sleep 30
done
echo "[$(date -Is)] matrix released the GPU lease"

run_bench() {
  local name=$1
  shift
  echo "[$(date -Is)] start $name"
  docker run --rm --name "pdb2-opt-${name}" \
    --ulimit nofile=65536:65536 --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g \
    --network host -v "$ROOT:$ROOT" -v /home/models:/models \
    -e PYTHONPATH="$ROOT/src" -e PDBLEND_MODELS_DIR=/models -w "$ROOT" "$IMAGE" \
    python -m pdblend.cli bench --profile "$PROFILE" --corpus "$CORPUS" \
      --dataset sharegpt --rate 8.109 --duration 300 --seed 701 --policy "$@" \
      --out "$OUT/$name"
  echo "[$(date -Is)] done $name"
}

run_bench adaptive-stage1 pdblend

winner=$(python3 - "$OUT/adaptive-stage1/summary.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print("yes" if p.get("j_per_token", 1e9) <= 0.5077995974174578 and p["slo"]["joint_slo_rate"] >= 0.9 else "no")
PY
)

if [ "$winner" = no ]; then
  run_bench fixed-m5 manual --layout M=5,L1=3 --clocks P=2520,D=2520,M=2100
  run_bench fixed-m4-2100 manual --layout M=4,L1=4 --clocks P=2520,D=2520,M=2100
  run_bench fixed-m4-2520 manual --layout M=4,L1=4 --clocks P=2520,D=2520,M=2520
fi

python3 - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob("*/summary.json")):
    s = json.loads(path.read_text())
    print(json.dumps({"name": path.parent.name, "j_per_token": s.get("j_per_token"),
                      "mean_power_w": s.get("mean_power_w"),
                      "joint_slo_rate": s["slo"].get("joint_slo_rate"),
                      "ttft_p90": s["slo"].get("ttft_p90"),
                      "ttft_p99": s["slo"].get("ttft_p99"),
                      "controller": s.get("controller", {}).get("events", {})}, ensure_ascii=False))
PY
