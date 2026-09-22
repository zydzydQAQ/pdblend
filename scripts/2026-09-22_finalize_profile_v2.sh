#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/results/2026-09-21/pdblend-optimization/profile-v2-parallel"}
OUT=$(realpath "$OUT")
IMAGE=pdblend:l20-cu128-vllm-v1
PYTHONPATH="$ROOT/src" python3 "$ROOT/scripts/merge_profile_raw.py" "$OUT" \
  "$OUT/pair-01/raw.json" "$OUT/pair-23/raw.json" "$OUT/pair-45/raw.json" "$OUT/pair-67/raw.json"
docker run --rm --network none --cpus 2 -v "$ROOT:$ROOT:ro" -v "$OUT:$OUT" \
  -e PYTHONPATH="$ROOT/src" -w "$ROOT" "$IMAGE" python -B -c \
  'import json,sys; from pathlib import Path; from pdblend.profile.profiler import load_raw; p=Path(sys.argv[1]); r=json.loads((p/"raw.json").read_text()); load_raw(p/"raw.json",r["model"],r["tp"],r["kv_bytes_per_token"]).save(p/"profile.json")' "$OUT"
docker run --rm --network none --cpus 2 -v "$ROOT:$ROOT:ro" -v "$OUT:$OUT" \
  -e PYTHONPATH="$ROOT/src" -w "$ROOT" "$IMAGE" python -B scripts/audit_pdblend_profile.py "$OUT/profile.json" "$OUT/audit.json"
