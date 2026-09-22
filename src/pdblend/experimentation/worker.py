"""GPU worker for disjoint queue leases.

Workers do not hold a host-wide lock while a process runs.  Queue claims are
atomic and disjoint, so independent GPU groups can execute concurrently;
exclusive jobs remain admitted only when the queue can allocate all GPUs.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
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


def run_one(queue: GPULeaseQueue, *, gpu_count: int | None = None,
            lock_path: str = '/tmp/pdblend4-gpu-experiment.lock') -> bool:
    """Run one payload selected by the queue.

    ``lock_path`` remains accepted for callers of the old API, but allocation
    no longer uses a host-wide flock and therefore does not serialize workers.
    """
    del lock_path
    with contextlib.nullcontext():
        try:
            lease = queue.claim(gpu_count=gpu_count)
        except LeaseConflict:
            return False
        if lease is None:
            return False
        spec = queue.snapshot()['jobs'][lease.job_id]['payload']
        attempt = Path(lease.attempt_dir).resolve()
        replacements = {'{attempt_dir}': str(attempt), '{lease_id}': lease.lease_id,
                        '{lease_gpu_indices}': ','.join(lease.gpu_indices)}
        argv = spec.get('argv')
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            queue.complete(lease.lease_id, lease.token, status='failed', metadata={'error': 'invalid argv'})
            return True
        for old, new in replacements.items():
            argv = [arg.replace(old, new) for arg in argv]
        container = spec.get('container_name')
        child = None
        failure = None
        started = time.time()
        hashes = {}
        with (attempt / 'worker.log').open('ab') as output:
            try:
                if container and subprocess.run(['docker', 'inspect', container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    raise RuntimeError(f'container name already exists: {container}')
                child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=spec.get('cwd'))
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
                    try:
                        child.wait(timeout=min(queue.heartbeat_s, 5))
                    except subprocess.TimeoutExpired:
                        pass
                if child.returncode != 0:
                    raise RuntimeError(f'process exited {child.returncode}')
                hashes = receipt(attempt, list(spec.get('required_receipts', [])))
            except BaseException as exc:
                failure = f'{type(exc).__name__}: {exc}'
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
                        queue.complete(lease.lease_id, lease.token, status='succeeded' if failure is None else 'failed', metadata=result)
                        break
                    except LeaseConflict:
                        time.sleep(1)
                else:
                    # Keep the lease active and diagnosable; a later reclaim
                    # must see the still-live process rather than pretending
                    # that the energy window ended cleanly.
                    raise LeaseConflict('GPU processes did not release after worker cleanup')
