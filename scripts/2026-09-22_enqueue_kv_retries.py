#!/usr/bin/env python3
"""Queue deterministic repeated KV retries for the failed TP smoke points."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from pdblend.experimentation.lease import GPULeaseQueue

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2026-09-22/three-model"
SNAPSHOT = OUT / "smoke-kv-retry-source"
QUEUE = OUT / "queue.json"
IMAGE = subprocess.check_output(
    ["docker", "image", "inspect", "pdblend:l20-cu128-vllm-v1", "--format", "{{.Id}}"], text=True
).strip()


def main() -> None:
    if SNAPSHOT.exists():
        raise SystemExit(f"snapshot already exists: {SNAPSHOT}")
    shutil.copytree(ROOT / "src", SNAPSHOT / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {
        str(p.relative_to(SNAPSHOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(SNAPSHOT.rglob("*")) if p.is_file()
    }
    source = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (SNAPSHOT / "manifest.json").write_text(json.dumps({"files": files, "source_sha256": source,
                                                           "image_digest": IMAGE}, indent=2, sort_keys=True) + "\n")
    queue = GPULeaseQueue(QUEUE)
    failed = [("7b", 2), ("7b", 4), ("14b", 4), ("32b", 2), ("32b", 4)]
    jobs = []
    for model, tp in failed:
        name = f"kv-retry-{model}-tp{tp}-pp1"
        argv = ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--cap-add", "SYS_ADMIN",
                "--ipc=host", "--network", "host", "--shm-size", "16g", "--ulimit", "nofile=65536:65536",
                "--entrypoint", "/opt/venv/bin/python", "-v", f"{SNAPSHOT}/src:/opt/pdblend-src:ro",
                "-v", "/home/models:/models:ro", "-v", "{attempt_dir}:/output:rw",
                "-e", "PYTHONPATH=/opt/pdblend-src", "-e", "PDBLEND_MODELS_DIR=/models",
                "-e", f"PDBLEND_SOURCE_SHA256={source}", "-e", f"PDBLEND_IMAGE_ID={IMAGE}",
                IMAGE, "-B", "-m", "pdblend.bench.topology_smoke", "--model", model,
                "--tp", str(tp), "--pp", "1", "--out", "/output"]
        queue.enqueue(name, {"argv": argv, "gpu_count": 8, "exclusive": True, "container_name": name,
                             "required_receipts": ["completion.json"], "timeout_s": 2400,
                             "evidence_class": "kv-retry"},
                      priority=10, depends_on=["profile-pilot-7b-tp1-pdblend-mixed"], max_attempts=1)
        jobs.append(name)
    print(json.dumps({"jobs": jobs, "source_sha256": source, "priority": 10}, indent=2))


if __name__ == "__main__":
    main()
