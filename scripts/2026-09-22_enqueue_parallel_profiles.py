#!/usr/bin/env python3
"""Enqueue immutable, resumable PP1 profile jobs without duplicating a topology."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdblend.experimentation.lease import GPULeaseQueue

OUT = ROOT / "results/2026-09-22/three-model"
MODEL_IDS = {key: f"Qwen2.5-{key.upper()}-Instruct" for key in ("7b", "14b", "32b")}
TOPOLOGIES = (("7b", 1), ("7b", 2), ("7b", 4), ("14b", 1), ("14b", 2),
              ("14b", 4), ("32b", 2), ("32b", 4))


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_files(directory: Path, *, include_caches: bool = False) -> dict[str, str]:
    files = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if relative == Path("manifest.json"):
            continue
        if not include_caches and ("__pycache__" in relative.parts or path.suffix == ".pyc"):
            continue
        if path.is_symlink():
            raise ValueError(f"source snapshot cannot contain a symlink: {path}")
        if path.is_file():
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError(f"empty source tree: {directory}")
    return files


def verify_snapshot(snapshot: Path, expected: dict) -> None:
    if snapshot.is_symlink() or (snapshot / "manifest.json").is_symlink():
        raise ValueError(f"source snapshot cannot use symlinks: {snapshot}")
    try:
        manifest = json.loads((snapshot / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"snapshot has no valid manifest: {snapshot}") from exc
    actual = source_files(snapshot, include_caches=True)
    if (manifest.get("files") != actual or actual != expected
            or manifest.get("source_sha256") != _digest(actual)):
        raise ValueError(f"source snapshot checksum mismatch: {snapshot}")


def freeze_source(source_dir: Path, snapshot_root: Path, *, dry_run: bool = False) -> tuple[Path, str]:
    """Reuse only a byte-verified content address; publish new snapshots atomically."""
    files = source_files(source_dir)
    digest = _digest(files)
    snapshot = snapshot_root / digest
    if snapshot.exists():
        verify_snapshot(snapshot, files)
    elif not dry_run:
        snapshot_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".source-", dir=snapshot_root))
        try:
            for name in files:
                destination = temporary / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_dir / name, destination)
            manifest = {"schema": 1, "source_sha256": digest, "files": files}
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            verify_snapshot(temporary, files)
            try:
                os.rename(temporary, snapshot)
            except OSError:
                if not snapshot.exists():
                    raise
                verify_snapshot(snapshot, files)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return snapshot, digest


def freeze_receipt(source: Path, receipt_root: Path, *, dry_run: bool = False) -> tuple[Path, str]:
    receipt = json.loads(source.read_text())
    for item in receipt.get("models", {}).values():
        for record in [item, *item.get("files", [])]:
            for key in ("model_path", "path"):
                value = record.get(key)
                if isinstance(value, str) and (value == "/home/models" or value.startswith("/home/models/")):
                    record[key] = "/models" + value[len("/home/models"):]
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    path = receipt_root / f"model-verification-{digest}.json"
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"model receipt checksum mismatch: {path}")
    elif not dry_run:
        receipt_root.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".receipt-", dir=receipt_root)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f"model receipt checksum mismatch: {path}")
        finally:
            temporary.unlink()
    return path, digest


def parse_only(value: str | None) -> tuple[tuple[str, int], ...]:
    if value is None:
        return TOPOLOGIES
    result = []
    for item in value.split(","):
        try:
            model, tp = item.strip().lower().split(":")
            topology = (model, int(tp))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected model:tp, got {item!r}") from exc
        if topology not in TOPOLOGIES:
            raise argparse.ArgumentTypeError(f"unsupported PP1 topology: {item!r}")
        if topology in result:
            raise argparse.ArgumentTypeError(f"duplicate topology: {item!r}")
        result.append(topology)
    return tuple(result)


def build_specs(topologies, *, snapshot: Path, source_sha256: str, image_digest: str,
                receipt_path: Path, receipt_sha256: str, out: Path, full_host: bool = False) -> list[dict]:
    specs = []
    for model, tp in topologies:
        if (model, tp) not in TOPOLOGIES:
            raise ValueError(f"unsupported topology: {model}:TP{tp}")
        gpu_count = 8 if full_host else 2 * tp
        identity = dict(schema=2, system="pdblend", model_id=MODEL_IDS[model], tp=tp, pp=1, role="mixed",
                        gpu_count=gpu_count, source_sha256=source_sha256, image_digest=image_digest,
                        model_receipt_sha256=receipt_sha256)
        name = f"profile-parallel-{model}-tp{tp}-pp1-{_digest(identity)[:20]}"
        argv = [
            "docker", "run", "--rm", "--name", name, "--gpus", "all",
            "--cap-add", "SYS_ADMIN", "--ipc=host", "--network", "host", "--shm-size", "16g",
            "--ulimit", "nofile=65536:65536", "--entrypoint", "/opt/venv/bin/python",
            "-v", f"{snapshot}:/opt/pdblend-src:ro", "-v", "/home/models:/models:ro",
            "-v", f"{receipt_path}:/verification/model-verification.json:ro",
            "-v", "{attempt_dir}:/output:rw", "-v", f"{out}/profile-wave:/coord:rw",
            "-e", "PYTHONPATH=/opt/pdblend-src", "-e", "PDBLEND_MODELS_DIR=/models",
            "-e", "PDBLEND_MODEL_VERIFICATION_RECEIPT=/verification/model-verification.json",
            "-e", "PDBLEND_SOURCE_SHA256=" + source_sha256, "-e", "PDBLEND_IMAGE_ID=" + image_digest,
            "-e", "PDBLEND_HARDWARE_ID=8xL20-lease", "-e", "PDBLEND_VLLM_VERSION=0.10.1.1",
            "-e", "CUDA_VERSION=12.8.1", "-e", "PDBLEND_GPU_UUIDS={lease_gpu_uuids}",
            "-e", "PDBLEND_COORD_DIR=/coord", "-e", "PDBLEND_CONCURRENCY_ENVIRONMENT",
            "-e", "PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256",
            image_digest, "-B", "-m", "pdblend.cli", "profile", "--model", "/models/" + MODEL_IDS[model],
            "--gpus", "{lease_local_indices}", "--tp", str(tp), "--pp", "1", "--system", "pdblend",
            "--role", "mixed", "--hardware-id", "8xL20-lease", "--engine-revision", "vllm-0.10.1.1",
            "--out", "/output", "--base-port", "{lease_port}", "--parallel-instances", "--resume",
            "--decode-repeats", "3", "--decode-settle", "2", "--decode-measure", "5",
        ]
        payload = dict(identity, argv=argv, exclusive=False, global_lock=False, container_name=name,
                       required_receipts=["completion.json"], timeout_s=7200, resume_profile=True,
                       evidence_class="profile", profile_mode="parallel_full_host" if full_host else "parallel_group",
                       topology={"model_id": MODEL_IDS[model], "tp": tp, "pp": 1, "gpu_count": gpu_count},
                       formal_eligible=False, completion_is_formal_qualification=False,
                       missing_gates=["independent_holdout", "topology_correctness_and_kv", "campaign_acceptance"],
                       depends_on=[])
        specs.append(dict(job_id=name, payload=payload, priority=0, max_attempts=2))
    return specs


def job_topology(job: dict) -> tuple[str, int] | None:
    """Read new metadata and older argv-only profile jobs, excluding other systems."""
    payload = job.get("payload", {})
    argv = payload.get("argv", [])
    if not (job.get("job_id", "").startswith("profile-") or payload.get("evidence_class") == "profile"):
        return None

    def option(name, default=None):
        try:
            return argv[argv.index(name) + 1]
        except (ValueError, IndexError):
            return default

    if payload.get("system", option("--system", "pdblend")) != "pdblend":
        return None
    if payload.get("role", option("--role", "mixed")) != "mixed":
        return None
    model = str(payload.get("model_id", option("--model", "")))
    key = next((key for key, model_id in MODEL_IDS.items() if model in (key, model_id, "/models/" + model_id)), None)
    if key is None:
        match = re.search(r"(?:^|[-/])(7b|14b|32b)(?:-|$)", job.get("job_id", ""), re.I)
        key = match[1].lower() if match else None
    try:
        tp = int(payload.get("tp", option("--tp", 1)))
        pp = int(payload.get("pp", option("--pp", 1)))
    except (TypeError, ValueError):
        return None
    return (key, tp) if pp == 1 and (key, tp) in TOPOLOGIES else None


def plan_jobs(jobs: dict, specs: list[dict], *, replace_queued: bool = False) -> list[dict]:
    plan = []
    for spec in specs:
        name = spec["job_id"]
        topology = job_topology(spec)
        matches = [job for job in jobs.values() if job_topology(job) == topology]
        covered = [job for job in matches if job.get("status") in ("running", "succeeded")]
        same = jobs.get(name)
        if covered:
            plan.append(dict(job_id=name, action="skip", reason="running_or_succeeded_topology",
                             existing=[job["job_id"] for job in covered]))
        elif same is not None:
            if same.get("payload") != spec["payload"] or same.get("max_attempts") != spec["max_attempts"]:
                raise ValueError(f"canonical job has a different immutable spec: {name}")
            plan.append(dict(job_id=name, action="keep", reason="existing_canonical_job", status=same["status"]))
        else:
            queued = [job["job_id"] for job in matches if job.get("status") == "queued"]
            if queued and not replace_queued:
                plan.append(dict(job_id=name, action="skip", reason="preserved_queued_topology", existing=queued))
            else:
                plan.append(dict(job_id=name, action="enqueue", replaces=queued, spec=spec))
    return plan


def enqueue_specs(queue: GPULeaseQueue, specs: list[dict], *, replace_queued: bool = False) -> list[dict]:
    """Commit replacements under the same queue lock used by workers.

    The queue currently has no batch public API. Using its transaction primitives
    here prevents a worker claiming an old queued job between enqueue and block.
    """
    with queue._lock():
        state = queue._read()
        plan = plan_jobs(state["jobs"], specs, replace_queued=replace_queued)
        now = queue.clock()
        changed = False
        for item in plan:
            if item["action"] != "enqueue":
                continue
            spec = item["spec"]
            state["jobs"][item["job_id"]] = dict(spec, attempts=0, status="queued", created_at=now,
                                                  updated_at=now, lease_id=None, last_error=None)
            for old in item["replaces"]:
                state["jobs"][old].update(status="blocked", lease_id=None, updated_at=now,
                                          last_error="replaced by " + item["job_id"])
            changed = True
        if changed:
            queue._write(state)
    return plan


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "src")
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--image", default="pdblend:l20-cu128-vllm-v1")
    parser.add_argument("--image-digest", help="Use an already resolved immutable image ID")
    parser.add_argument("--only", type=parse_only, default=TOPOLOGIES, help="e.g. 7b:4,14b:4,32b:2,32b:4")
    parser.add_argument("--full-host", action="store_true", help="lease 8 GPUs for resident within-job sampling")
    parser.add_argument("--replace-queued", action="store_true", help="atomically replace matching old queued jobs")
    parser.add_argument("--dry-run", action="store_true", help="read/verify only; create no snapshot, receipt, or queue")
    parser.add_argument("--spec-out", type=Path, help="save all selected immutable jobs for the queue CLI")
    parser.add_argument("--prepare-only", action="store_true", help="freeze/export specs without changing the queue")
    args = parser.parse_args(argv)
    if args.prepare_only and args.spec_out is None:
        parser.error("--prepare-only requires --spec-out")
    if args.dry_run and (args.spec_out is not None or args.prepare_only):
        parser.error("--dry-run cannot write --spec-out or prepare snapshots")
    image = args.image_digest or subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError(f"immutable sha256 Docker image ID required, got {image!r}")
    out = args.out.resolve()
    queue_path = (args.queue or out / "queue.json").resolve()
    snapshot, source_sha = freeze_source(args.source_dir.resolve(), out / "profile-sources", dry_run=args.dry_run)
    receipt, receipt_sha = freeze_receipt(args.model_receipt or out / "model-verification.json",
                                          out / "profile-receipts", dry_run=args.dry_run)
    specs = build_specs(args.only, snapshot=snapshot, source_sha256=source_sha, image_digest=image,
                        receipt_path=receipt, receipt_sha256=receipt_sha, out=out, full_host=args.full_host)
    if args.spec_out is not None:
        data = json.dumps(specs, indent=2, sort_keys=True) + "\n"
        if args.spec_out.exists() and args.spec_out.read_text() != data:
            raise ValueError(f"refusing to replace immutable job spec: {args.spec_out}")
    if args.dry_run or args.prepare_only:
        state = json.loads(queue_path.read_text()) if queue_path.exists() else {"jobs": {}}
        plan = plan_jobs(state["jobs"], specs, replace_queued=args.replace_queued)
    else:
        plan = enqueue_specs(GPULeaseQueue(queue_path), specs, replace_queued=args.replace_queued)
    if args.spec_out is not None:
        args.spec_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.spec_out.open("x") as target:
                target.write(data)
        except FileExistsError:
            if args.spec_out.read_text() != data:
                raise ValueError(f"refusing to replace immutable job spec: {args.spec_out}")
    result = dict(dry_run=args.dry_run, prepare_only=args.prepare_only,
                  spec_out=str(args.spec_out) if args.spec_out else None,
                  source_sha256=source_sha, source_snapshot=str(snapshot),
                  image_digest=image, model_receipt_sha256=receipt_sha, plan=plan, formal_eligible=False)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
