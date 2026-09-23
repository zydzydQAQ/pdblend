#!/usr/bin/env bash
set -euo pipefail

# Run the dominance candidate only after the independent baseline matrix has
# released all GPUs. The container sees read-only source/profile/corpus input;
# only the experiment directory is writable.
ROOT=/home/pdblend4
RUN="$ROOT/results/2026-09-22/pdblend-dominance-round1"
SPEC="$RUN/spec.json"
PROFILE="$ROOT/results/2026-09-22/decode900/round6/profile.json"
CORPUS="$ROOT/datasets/prepared/2026-09-21-7b-v2-half"
IMAGE=pdblend:l20-cu128-vllm-v1
NAME=pdb4-dominance-round1
LEASE=/tmp/pdblend4-gpu-experiment.lock

exec 9>"$LEASE"
flock -n 9 || { echo "another exclusive GPU experiment owns $LEASE" >&2; exit 2; }

if docker ps --format '{{.Names}}' | grep -Eq '^(pdb2-matrix-v2|pdb4-dominance-round1)$'; then
  echo "an exclusive matrix already owns the GPUs" >&2
  exit 2
fi
if [ ! -s "$PROFILE" ] || [ ! -s "$ROOT/results/2026-09-22/decode900/round6/AUDIT-PASSED" ]; then
  echo "accepted round6 profile evidence is missing" >&2
  exit 3
fi
if [ ! -s "$SPEC" ] || [ ! -d "$CORPUS" ]; then
  echo "dominance spec or corpus is missing" >&2
  exit 4
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU compute processes are still active" >&2
  exit 5
fi

SNAP="$RUN/source-snapshot"
if [ -e "$SNAP" ]; then
  echo "refusing to replace existing source snapshot: $SNAP" >&2
  exit 6
fi
mkdir -p "$RUN"
cp -a "$ROOT/src" "$SNAP"
find "$SNAP" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$SNAP" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

sha_tree() {
  python3 - "$1" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
h = hashlib.sha256()
for p in sorted(x for x in root.rglob('*') if x.is_file()):
    h.update(str(p.relative_to(root)).encode()); h.update(b'\0'); h.update(hashlib.sha256(p.read_bytes()).digest())
print(h.hexdigest())
PY
}
sha_file() { sha256sum "$1" | awk '{print $1}'; }

SOURCE_SHA=$(python3 - "$SNAP" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1]); h = hashlib.sha256()
for p in sorted(x for x in root.rglob('*.py') if x.is_file()):
    h.update(str(p.relative_to(root)).encode()); h.update(b'\0'); h.update(hashlib.sha256(p.read_bytes()).digest())
print(h.hexdigest())
PY
)
PROFILE_SHA=$(sha_file "$PROFILE")
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
GPU_UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)
HARDWARE=$(nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader | paste -sd';' -)
CORPUS_SHA=$(sha_tree "$CORPUS")
SPEC_SHA=$(sha_file "$SPEC")
cat > "$RUN/execution.json" <<EOF
{
  "status": "running",
  "started_s": $(date +%s),
  "image": "$IMAGE_ID",
  "image_tag": "$IMAGE",
  "source_sha256": "$SOURCE_SHA",
  "profile_sha256": "$PROFILE_SHA",
  "corpus_sha256": "$CORPUS_SHA",
  "spec_sha256": "$SPEC_SHA",
  "hardware_uuids": "$GPU_UUIDS",
  "hardware": "$HARDWARE",
  "clock_protocol": "nvidia-smi-lock-clock-v1",
  "energy_protocol": "nvml-0.1s-trapezoid-v1",
  "command": ["python", "-B", "-m", "pdblend.cli", "matrix", "$SPEC"]
}
EOF

set +e
docker run --rm --name "$NAME" \
  --ulimit nofile=65536:65536 --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g \
  --network host \
  -v "$ROOT:$ROOT:rw" \
  -v "$SNAP:/opt/pdblend-src:ro" \
  -v "$PROFILE:$PROFILE:ro" \
  -v "$CORPUS:$CORPUS:ro" \
  -v /home/models:/models:ro \
  -e PYTHONPATH=/opt/pdblend-src -e PDBLEND_MODELS_DIR=/models \
  -e PDBLEND_SOURCE_SHA256="$SOURCE_SHA" -e PDBLEND_IMAGE_ID="$IMAGE_ID" \
  -e PDBLEND_HARDWARE_UUIDS="$GPU_UUIDS" \
  -e PDBLEND_CLOCK_PROTOCOL=nvidia-smi-lock-clock-v1 \
  -e PDBLEND_ENERGY_PROTOCOL=nvml-0.1s-trapezoid-v1 \
  -w "$ROOT" "$IMAGE" python -B -m pdblend.cli matrix "$SPEC" \
  2>&1 | tee -a "$RUN/matrix.log"
RC=${PIPESTATUS[0]}
set -e

python3 - "$RUN/execution.json" "$RC" "$PROFILE_SHA" "$SOURCE_SHA" "$CORPUS_SHA" <<'PY'
import json, pathlib, sys, time
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
d.update(status='complete' if int(sys.argv[2]) == 0 else 'failed', returncode=int(sys.argv[2]), finished_s=time.time())
d['inputs_unchanged'] = d['profile_sha256'] == sys.argv[3] and d['source_sha256'] == sys.argv[4] and d['corpus_sha256'] == sys.argv[5]
p.write_text(json.dumps(d, indent=2, sort_keys=True))
if d['returncode'] or not d['inputs_unchanged']:
    raise SystemExit(d['returncode'] or 1)
PY
exit "$RC"
