"""GPU worker for disjoint queue leases.

Workers do not hold a host-wide lock while a process runs.  Queue claims are
atomic and disjoint, so independent GPU groups can execute concurrently;
exclusive jobs remain admitted only when the queue can allocate all GPUs.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import time
from pathlib import Path

from .lease import GPULeaseQueue, LeaseConflict


def descendants(pid: int) -> set[int]:
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                parents[int(entry.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                pass
    result = {pid}
    while True:
        updated = result | {p for p, parent in parents.items() if parent in result}
        if updated == result:
            return result
        result = updated


def receipt(attempt: Path, required: list[str]) -> dict:
    if not required:
        raise ValueError('job requires at least one completion receipt')
    hashes = {}
    for name in required:
        path = (attempt / name).resolve()
        if not path.is_relative_to(attempt.resolve()) or not path.is_file():
            raise ValueError(f'missing or unsafe receipt: {name}')
        data = json.loads(path.read_text())
        if data.get('status') != 'passed' or data.get('complete') is not True:
            raise ValueError(f'unsuccessful receipt: {name}')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _source_identity(value) -> tuple[str, ...]:
    """Extract explicit source/image identities without inventing provenance."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source_hash", "source_sha256", "image_digest", "image_id"} and isinstance(item, str) and item:
                found.add(f"{key}={item}")
            found.update(_source_identity(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_source_identity(item))
    elif isinstance(value, str):
        for key, item in re.findall(r"(PDBLEND_SOURCE_SHA256|PDBLEND_SOURCE_HASH|PDBLEND_IMAGE_ID|PDBLEND_IMAGE_DIGEST)=([^ '\"]+)", value):
            found.add(f"{key.lower()}={item}")
    return tuple(sorted(found))


def _strict_source_identity(value) -> tuple[str, ...] | None:
    identity = _source_identity(value)
    source = tuple(x for x in identity if x.split("=", 1)[0] in {"source_hash", "source_sha256", "pdblend_source_hash", "pdblend_source_sha256"})
    image = tuple(x for x in identity if x.split("=", 1)[0] in {"image_digest", "image_id", "pdblend_image_digest", "pdblend_image_id"})
    if not source or not image or any(x.split("=", 1)[1].lower() in {"unknown", "none", "null"} for x in source + image):
        return None
    return tuple(sorted(source + image))


def restore_resume_artifacts(queue: GPULeaseQueue, lease, payload: dict) -> dict | None:
    """Copy raw/samples from the prior attempt only for an identical lease/source.

    Raw measurements are copied byte-for-byte, preserving their provenance.
    Missing or ambiguous source identity, changed image/source, or a changed
    UUID allocation disables resume rather than guessing.
    """
    if payload.get("resume_profile") is not True:
        return None
    current_identity = _strict_source_identity(payload)
    if not current_identity:
        return None
    current = Path(lease.attempt_dir).resolve()
    candidates = []
    for manifest_path in current.parent.glob("attempt-*/manifest.json"):
        if manifest_path.parent.resolve() == current:
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if int(manifest.get("attempt", 0)) >= int(lease.attempt):
            continue
        if tuple(manifest.get("gpu_uuids", ())) != tuple(lease.gpu_uuids):
            continue
        if _strict_source_identity(manifest.get("payload", {})) != current_identity:
            continue
        candidates.append((int(manifest.get("attempt", 0)), manifest_path.parent))
    if not candidates:
        return None
    _, previous = max(candidates)
    copied = []
    sources = [previous / "raw.json"]
    samples = previous / "samples"
    if samples.is_dir():
        sources.extend(p for p in samples.rglob("*") if p.is_file())
    for source in sources:
        if not source.is_file():
            continue
        target = current / source.relative_to(previous)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(target.relative_to(current)))
    if not copied:
        return None
    metadata = {"previous_attempt": str(previous), "source_identity": list(current_identity),
                "gpu_uuids": list(lease.gpu_uuids), "files": copied}
    (current / "resume.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def write_concurrency_environment(queue: GPULeaseQueue, lease, attempt: Path,
                                  started_s: float, *, topology=None) -> str:
    """Record lease layout and changing peer leases without exposing tokens."""
    inventory = [{"uuid": g.uuid, "index": g.index, "pids": list(g.pids)} for g in queue._probe()]
    physical_uuids = [item["uuid"] for item in inventory]
    peers = [{"lease_id": p.lease_id, "job_id": p.job_id,
              "gpu_uuids": list(p.gpu_uuids), "attempt": p.attempt}
             for p in queue.active_leases() if p.lease_id != lease.lease_id]
    path = attempt / "concurrency-environment.json"
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        document = {"schema": 1, "lease_id": lease.lease_id,
                    "allocated_gpu_uuids": list(lease.gpu_uuids),
                    "started_s": started_s, "topology": topology,
                    "inventory": inventory, "physical_gpu_uuids": physical_uuids,
                    "all_gpu_uuids": physical_uuids, "peer_jobs": [],
                    "lease_manifest_file": "manifest.json",
                    "lease_manifest_sha256": hashlib.sha256((attempt / "manifest.json").read_bytes()).hexdigest(),
                    "peer_snapshots": []}
    previous = document.get("peer_snapshots", [])
    if not previous or previous[-1].get("peers") != peers:
        previous.append({"at_s": time.time(), "peers": peers})
    document.update({"inventory": inventory, "last_updated_s": time.time(),
                     "physical_gpu_uuids": physical_uuids, "all_gpu_uuids": physical_uuids,
                     "peer_jobs": peers,
                     "peer_snapshots": previous})
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    os.replace(temp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finish(queue: GPULeaseQueue, lease, payload: dict, *, failed: str | None,
            metadata: dict | None = None) -> None:
    """Release a completed lease, or safely requeue an opted-in retry."""
    if failed and payload.get("resume_profile") is True:
        job = queue.snapshot()["jobs"].get(lease.job_id, {})
        if int(job.get("attempts", 0)) < int(job.get("max_attempts", 1)):
            queue.retry(lease.lease_id, lease.token, error=failed)
            return
    queue.complete(lease.lease_id, lease.token,
                   status="failed" if failed else "succeeded", metadata=metadata)


def run_one(queue: GPULeaseQueue, *, gpu_count: int | None = None,
            lock_path: str = '/tmp/pdblend4-gpu-experiment.lock') -> bool:
    """Run one payload selected by the queue.

    ``lock_path`` is shared by legacy workers: group jobs take a shared flock,
    while formal jobs take it exclusively, so groups overlap but exclusives
    still get a quiescent host.
    """
    # Shared locks allow disjoint group jobs to overlap.  A formal exclusive
    # worker takes the same lock exclusively, preserving compatibility with
    # older workers that still coordinate through this host lock.
    with open(lock_path, 'a') as guard:
        shared = False
        try:
            fcntl.flock(guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
            shared = True
            lease = queue.claim(gpu_count=gpu_count, lock_mode=False)
        except BlockingIOError:
            lease = None
        except LeaseConflict:
            return False
        if lease is None and shared:
            fcntl.flock(guard, fcntl.LOCK_UN)
            shared = False
        if lease is None:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lease = queue.claim(gpu_count=gpu_count, lock_mode=True)
            except (BlockingIOError, LeaseConflict):
                return False
        if lease is None:
            return False
        spec = queue.snapshot()['jobs'][lease.job_id]['payload']
        attempt = Path(lease.attempt_dir).resolve()
        try:
            restore_resume_artifacts(queue, lease, spec)
        except BaseException as exc:
            queue.retry(lease.lease_id, lease.token, error=f"resume restore failed: {type(exc).__name__}: {exc}")
            return True
        started = time.time()
        try:
            environment_hash = write_concurrency_environment(queue, lease, attempt, started,
                                                              topology=spec.get("topology"))
        except BaseException as exc:
            queue.retry(lease.lease_id, lease.token,
                        error=f"concurrency environment failed: {type(exc).__name__}: {exc}")
            return True
        gpu_uuids = ','.join(lease.gpu_uuids)
        gpu_indices = ','.join(lease.gpu_indices)
        local_indices = ','.join(str(i) for i in range(len(lease.gpu_uuids)))
        # Keep ports deterministic for the attempt while avoiding collisions
        # between simultaneously active leases in normal operation.
        # Disjoint active groups have distinct first physical indices. Reserve
        # 100 ports per group, keeping HTTP+20000 KV ports below 65535.
        lease_port = 10000 + 100 * int(lease.gpu_indices[0])
        replacements = {
            '{attempt_dir}': str(attempt), '{lease_id}': lease.lease_id,
            '{lease_gpu_uuids}': gpu_uuids, '{lease_gpu_indices}': gpu_indices,
            '{lease_local_indices}': local_indices, '{lease_port}': str(lease_port),
        }
        argv = spec.get('argv')
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            _finish(queue, lease, spec, failed="invalid argv", metadata={'error': 'invalid argv'})
            return True
        for old, new in replacements.items():
            argv = [arg.replace(old, new) for arg in argv]
        # Never let a container claim every device when the lease only covers
        # a group.  Docker's subprocess form receives the quoted device JSON
        # as one argument, e.g. '"device=GPU-a,GPU-b"'.
        for index, arg in enumerate(argv[:-1]):
            if arg == '--gpus' and argv[index + 1] in {'all', '"device=all"'}:
                # Docker's --gpus parser requires the complete device list to
                # remain one quoted CSV value (the shell normally adds these
                # quote characters for an interactive command).
                argv[index + 1] = f'"device={gpu_uuids}"'
        container = spec.get('container_name')
        child = None
        failure = None
        hashes = {}
        with (attempt / 'worker.log').open('ab') as output:
            try:
                if container and subprocess.run(['docker', 'inspect', container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    raise RuntimeError(f'container name already exists: {container}')
                child_env = dict(os.environ)
                child_env.update({"PDBLEND_CONCURRENCY_ENVIRONMENT": "/output/concurrency-environment.json",
                                  "PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256": environment_hash})
                child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=spec.get('cwd'), env=child_env)
                deadline = started + float(spec.get('timeout_s', 7200))
                while child.poll() is None:
                    if time.time() >= deadline:
                        raise TimeoutError('job deadline exceeded')
                    allowed = descendants(child.pid)
                    if container:
                        rows = subprocess.run(['docker', 'top', container, '-eo', 'pid'], capture_output=True, text=True)
                        if rows.returncode == 0:
                            allowed.update(int(x.strip()) for x in rows.stdout.splitlines() if x.strip().isdigit())
                    queue.heartbeat(lease.lease_id, lease.token, process_pids=sorted(allowed))
                    write_concurrency_environment(queue, lease, attempt, started,
                                                  topology=spec.get("topology"))
                    try:
                        child.wait(timeout=min(queue.heartbeat_s, 5))
                    except subprocess.TimeoutExpired:
                        pass
                if child.returncode != 0:
                    raise RuntimeError(f'process exited {child.returncode}')
                hashes = receipt(attempt, list(spec.get('required_receipts', [])))
            except BaseException as exc:
                failure = f'{type(exc).__name__}: {exc}'
                if spec.get('cohort_dir') and spec.get('cohort_member'):
                    directory = Path(spec['cohort_dir'])
                    directory.mkdir(parents=True, exist_ok=True)
                    name = directory / (spec['cohort_member'] + '.error.json')
                    temporary = name.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'error': failure}))
                    os.replace(temporary, name)
            finally:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=15)
                # This worker may only stop the container created by its child.
                if container and child is not None:
                    subprocess.run(['docker', 'stop', '-t', '10', container], stdout=output, stderr=output)
                result = {'status': 'passed' if failure is None else 'failed', 'complete': failure is None,
                          'started_s': started, 'finished_s': time.time(), 'argv': argv,
                          'returncode': child.returncode if child else None, 'error': failure,
                          'receipt_sha256': hashes, 'formal_eligible': False}
                (attempt / 'execution.json').write_text(json.dumps(result, indent=2) + '\n')
                # Completion refuses to release GPUs with compute processes.
                release_deadline = time.time() + 45
                while time.time() < release_deadline:
                    try:
                        _finish(queue, lease, spec, failed=failure, metadata=result)
                        break
                    except LeaseConflict:
                        time.sleep(1)
                else:
                    # Keep the lease active and diagnosable; a later reclaim
                    # must see the still-live process rather than pretending
                    # that the energy window ended cleanly.
                    raise LeaseConflict('GPU processes did not release after worker cleanup')
        return True
