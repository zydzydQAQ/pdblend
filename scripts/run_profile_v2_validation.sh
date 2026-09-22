#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pdblend4
OUT="$ROOT/results/2026-09-21/pdblend-optimization"
BASE="${PROFILE_V2_BASE:-$ROOT/results/2026-09-22/decode900/round6}"
PROFILE="${PROFILE_V2_PROFILE:-$BASE/profile.json}"
LOG="$BASE/acceptance/validation.log"
mkdir -p "$BASE/acceptance"
exec >>"$LOG" 2>&1
if [ ! -s "$PROFILE" ] || [ ! -s "$BASE/audit.json" ] || [ ! -s "$BASE/independent-report.json" ]; then echo "[$(date -Is)] profile-v2 is not finalized"; exit 1; fi
if ! python3 - "$BASE/audit.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('basic_passed') else 1)
PY
then echo "[$(date -Is)] profile-v2 audit failed; GPU validation withheld"; exit 2; fi
if ! python3 - "$BASE/independent-report.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); g=d.get('gate',{})
raise SystemExit(0 if g.get('status') == 'PASS' and g.get('max_error_le_10') and g.get('complete') else 1)
PY
then echo "[$(date -Is)] independent decode audit failed; GPU validation withheld"; exit 3; fi
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do sleep 15; done
run_fixed() {
  local name=$1 layout=$2 clocks=$3 seed=$4 target="$OUT/profile-v2-round6-$1-seed-$4"
  if [ -s "$target/measurement/summary.json" ]; then echo "skip complete $target"; return; fi
  if [ -e "$target" ]; then echo "refusing partial target $target"; exit 1; fi
  python3 "$ROOT/scripts/2026-09-21_optimization_experiment.py" bench --out "$target" --profile "$PROFILE" --policy manual --layout "$layout" --clocks "$clocks" --seed "$seed"
}
for seed in 701 1701 2701; do run_fixed m2-1800 M=2,L1=6 P=2520,D=2520,M=1800 "$seed"; done
run_fixed m4-2100 M=4,L1=4 P=2520,D=2520,M=2100 701
run_fixed m5-2100 M=5,L1=3 P=2520,D=2520,M=2100 701
run_fixed m6-2100 M=6,L1=2 P=2520,D=2520,M=2100 701
run_fixed m4-2520 M=4,L1=4 P=2520,D=2520,M=2520 701
docker run --rm --network none --cpus 2 -v "$ROOT:$ROOT" -e PYTHONPATH="$ROOT/src" -w "$ROOT" pdblend:l20-cu128-vllm-v1 \
  python -B "$ROOT/scripts/collect_profile_v2_acceptance.py" "$OUT" "$BASE/acceptance/fixed-summary.json" "$BASE/profile.json"
if python3 - "$BASE/acceptance/fixed-summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['m2_gate']['passed'] else 1)
PY
then echo "[$(date -Is)] M2 gate passed; M2 safety floor may be lifted"; else echo "[$(date -Is)] M2 gate failed; M>=4 safety floor remains active"; fi
python3 "$ROOT/scripts/2026-09-21_optimization_experiment.py" bench --out "$OUT/adaptive-profile-v2-round6-seed-701" --profile "$PROFILE" --policy pdblend --clocks P=2520,D=2520,M=2100 --seed 701
python3 "$ROOT/scripts/build_profile_v2_compare.py" "$OUT" "$BASE/compare-pdblend-profile-v2.csv"
