#!/usr/bin/env python3
"""Freeze a verified source/receipt pair and queue independent PDBlend profiles.

Every job has its own model/topology/profile identity and output directory.
Each job requests the GPU group used by its P/D topology, allowing disjoint
profiles to run concurrently. Jobs that need the whole host may still opt into
an exclusive lease through the queue payload.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from pdblend.experimentation.lease import GPULeaseQueue


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/2026-09-22/three-model"
SNAPSHOT = OUT / "profile-formal-source"
RECEIPT_HOST = OUT / "model-verification-container.json"
QUEUE = OUT / "queue.json"
DEPENDENCY = "profile-pilot-7b-tp1-pdblend-mixed"
IMAGE = subprocess.check_output(
    ["docker", "image", "inspect", "pdblend:l20-cu128-vllm-v1", "--format", "{{.Id}}"],
    text=True,
).strip()


def freeze_source() -> str:
    if SNAPSHOT.exists():
        raise SystemExit(f"source snapshot already exists: {SNAPSHOT}")
    shutil.copytree(ROOT / "src", SNAPSHOT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {
        str(path.relative_to(SNAPSHOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(SNAPSHOT.rglob("*"))
        if path.is_file()
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (SNAPSHOT / "manifest.json").write_text(
        json.dumps({"files": files, "source_sha256": digest, "image_digest": IMAGE}, indent=2, sort_keys=True)
        + "\n"
    )
    return digest


def freeze_container_receipt() -> None:
    source = OUT / "model-verification.json"
    receipt = json.loads(source.read_text())
    for item in receipt.get("models", {}).values():
        item["model_path"] = str(item["model_path"]).replace("/home/models", "/models")
        for file in item.get("files", []):
            if isinstance(file.get("path"), str):
                file["path"] = file["path"].replace("/home/models", "/models")
    RECEIPT_HOST.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main() -> None:
    source = freeze_source()
    freeze_container_receipt()
    queue = GPULeaseQueue(QUEUE)
    # P/D profile requires two instances, so TP8 cannot fit on this 8-GPU host.
    topologies = [("7b", 1), ("7b", 2), ("7b", 4),
                  ("14b", 1), ("14b", 2), ("14b", 4),
                  ("32b", 2), ("32b", 4)]
    jobs = []
    for model, tp in topologies:
        devices = list(range(2 * tp))
        name = f"profile-{model}-tp{tp}-pdblend-mixed"
        model_path = f"/models/Qwen2.5-{model.upper().replace('B', 'B')}-Instruct"
        # The directory names are normalized by the model registry, but retain
        # the exact on-disk Qwen name for the engine.
        model_path = {"7b": "/models/Qwen2.5-7B-Instruct",
                      "14b": "/models/Qwen2.5-14B-Instruct",
                      "32b": "/models/Qwen2.5-32B-Instruct"}[model]
        argv = [
            "docker", "run", "--rm", "--name", name, "--gpus", "all",
            "--cap-add", "SYS_ADMIN", "--ipc=host", "--network", "host",
            "--shm-size", "16g", "--ulimit", "nofile=65536:65536",
            "--entrypoint", "/opt/venv/bin/python",
            "-v", f"{SNAPSHOT}:/opt/pdblend-src:ro",
            "-v", "/home/models:/models:ro",
            "-v", f"{RECEIPT_HOST}:/verification/model-verification.json:ro",
            "-v", "{attempt_dir}:/output:rw",
            "-e", "PYTHONPATH=/opt/pdblend-src", "-e", "PDBLEND_MODELS_DIR=/models",
            "-e", "PDBLEND_MODEL_VERIFICATION_RECEIPT=/verification/model-verification.json",
            "-e", f"PDBLEND_SOURCE_SHA256={source}", "-e", f"PDBLEND_IMAGE_ID={IMAGE}",
            "-e", "PDBLEND_HARDWARE_ID=8xL20-lease", "-e", "PDBLEND_VLLM_VERSION=0.10.1.1",
            "-e", "PDBLEND_TORCH_VERSION=2.7.0", "-e", "CUDA_VERSION=12.8.1", IMAGE,
            "-B", "-m", "pdblend.cli", "profile", "--model", model_path,
            "--gpus", ",".join(map(str, devices)), "--tp", str(tp), "--pp", "1",
            "--system", "pdblend", "--role", "mixed", "--hardware-id", "8xL20-lease",
            "--engine-revision", "vllm-0.10.1.1", "--out", "/output",
            "--decode-repeats", "3", "--decode-settle", "2", "--decode-measure", "5",
        ]
        queue.enqueue(name, {
            "argv": argv, "gpu_count": len(devices), "container_name": name,
            "required_receipts": ["completion.json"], "timeout_s": 7200,
            "evidence_class": "profile",
        }, depends_on=[DEPENDENCY], max_attempts=1)
        jobs.append(name)
    print(json.dumps({"jobs": jobs, "source_sha256": source, "image": IMAGE}, indent=2))


if __name__ == "__main__":
    main()
