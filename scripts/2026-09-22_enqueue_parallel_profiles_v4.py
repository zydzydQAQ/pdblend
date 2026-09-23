#!/usr/bin/env python3
"""Retry missing TP-only profiles with the long-prefill progress timeout fix."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from pdblend.experimentation.lease import GPULeaseQueue

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2026-09-22/three-model"
SNAPSHOT = OUT / "profile-parallel-source-v4"
QUEUE = OUT / "queue.json"
IMAGE = subprocess.check_output(
    ["docker", "image", "inspect", "pdblend:l20-cu128-vllm-v1", "--format", "{{.Id}}"], text=True
).strip()


def freeze_source() -> str:
    shutil.copytree(ROOT / "src", SNAPSHOT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {str(p.relative_to(SNAPSHOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(SNAPSHOT.rglob("*")) if p.is_file()}
    source = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (SNAPSHOT / "manifest.json").write_text(json.dumps(
        {"files": files, "source_sha256": source, "image_digest": IMAGE}, indent=2, sort_keys=True) + "\n")
    return source


def main() -> None:
    source = freeze_source()
    queue = GPULeaseQueue(QUEUE)
    # Stop queued v3 jobs from competing with the corrected snapshot. Running
    # jobs are left alone and their completed artifacts remain reusable.
    for job in queue.list_jobs(status="queued"):
        if job.job_id.startswith("profile-parallel-v3-"):
            queue.block(job.job_id, reason="superseded by v4 progress-timeout profile snapshot")
    models = {"7b": "/models/Qwen2.5-7B-Instruct", "14b": "/models/Qwen2.5-14B-Instruct",
              "32b": "/models/Qwen2.5-32B-Instruct"}
    topology = [("7b", 2), ("7b", 4), ("14b", 2), ("14b", 4), ("32b", 2), ("32b", 4)]
    existing = set()
    for job in queue.list_jobs():
        if not job.job_id.startswith(("profile-parallel-v2-", "profile-parallel-v3-")) or job.status not in {"running", "succeeded"}:
            continue
        parts = job.job_id.split("-")
        if len(parts) >= 6:
            existing.add((parts[3], parts[4]))
    jobs = []
    for model, tp in topology:
        if (model, f"tp{tp}") in existing:
            continue
        name = f"profile-parallel-v4-{model}-tp{tp}-mixed"
        argv = [
            "docker", "run", "--rm", "--name", name, "--gpus", "all",
            "--cap-add", "SYS_ADMIN", "--ipc=host", "--network", "host",
            "--shm-size", "16g", "--ulimit", "nofile=65536:65536",
            "--entrypoint", "/opt/venv/bin/python", "-v", f"{SNAPSHOT}:/opt/pdblend-src:ro",
            "-v", "/home/models:/models:ro", "-v", "/home/pdblend4/results/2026-09-22/three-model/model-verification-container.json:/verification/model-verification.json:ro",
            "-v", "{attempt_dir}:/output:rw", "-v", f"{OUT}/profile-wave:/coord:rw",
            "-e", "PYTHONPATH=/opt/pdblend-src", "-e", "PDBLEND_MODELS_DIR=/models",
            "-e", "PDBLEND_MODEL_VERIFICATION_RECEIPT=/verification/model-verification.json",
            "-e", f"PDBLEND_SOURCE_SHA256={source}", "-e", f"PDBLEND_IMAGE_ID={IMAGE}",
            "-e", "PDBLEND_HARDWARE_ID=8xL20-lease", "-e", "PDBLEND_VLLM_VERSION=0.10.1.1",
            "-e", "PDBLEND_TORCH_VERSION=2.7.0", "-e", "CUDA_VERSION=12.8.1",
            "-e", "PDBLEND_GPU_UUIDS={lease_gpu_uuids}", "-e", "PDBLEND_COORD_DIR=/coord",
            IMAGE, "-B", "-m", "pdblend.cli", "profile", "--model", models[model],
            "--gpus", "{lease_local_indices}", "--tp", str(tp), "--pp", "1",
            "--system", "pdblend", "--role", "mixed", "--hardware-id", "8xL20-lease",
            "--engine-revision", "vllm-0.10.1.1", "--out", "/output", "--base-port", "{lease_port}",
            "--parallel-instances", "--decode-repeats", "3", "--decode-settle", "2", "--decode-measure", "5",
        ]
        queue.enqueue(name, {"argv": argv, "gpu_count": 2 * tp, "exclusive": False,
                             "global_lock": False, "container_name": name,
                             "required_receipts": ["completion.json"], "timeout_s": 7200,
                             "evidence_class": "profile", "profile_mode": "parallel_group"},
                      depends_on=["three-model-smoke-7b-tp1-pp1"], max_attempts=1)
        jobs.append(name)
    print(json.dumps({"jobs": jobs, "source_sha256": source, "image": IMAGE}, indent=2))


if __name__ == "__main__":
    main()
