from __future__ import annotations

import json
import os
import threading

import pytest

from pdblend.experimentation.lease import GPULeaseQueue, LeaseConflict, LeaseExpired


class FakeClock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def probe():
    return [{"uuid": "GPU-a", "index": "0", "pids": []}, {"uuid": "GPU-b", "index": "1", "pids": []}]


def make_queue(tmp_path, clock=None, max_attempts=3):
    q = GPULeaseQueue(tmp_path / "queue.json", ttl_s=180, heartbeat_interval_s=30, gpu_probe=probe, clock=clock or FakeClock())
    q.enqueue("first", {"model_id": "7b"}, max_attempts=max_attempts)
    return q


def test_atomic_gpu_conflict_and_distinct_attempt_dirs(tmp_path):
    q = make_queue(tmp_path)
    q.enqueue("second", {"model_id": "14b"})
    first = q.claim(gpu_uuids=["GPU-a"], owner_pid=os.getpid())
    assert first is not None
    with pytest.raises(LeaseConflict):
        q.claim(gpu_uuids=["GPU-a"], owner_pid=os.getpid())
    second = q.claim(gpu_uuids=["GPU-b"], owner_pid=os.getpid())
    assert second is not None and second.attempt_dir != first.attempt_dir
    manifest = json.loads((__import__("pathlib").Path(first.attempt_dir) / "manifest.json").read_text())
    assert manifest["immutable"] is True


def test_expired_lease_reclaimed_and_next_attempt_is_new(tmp_path):
    clock = FakeClock()
    q = make_queue(tmp_path, clock)
    first = q.claim(gpu_uuids=["GPU-a"], owner_pid=os.getpid())
    assert first is not None
    clock.advance(181)
    q.pid_probe = lambda pid: False
    assert q.reclaim() == [first.lease_id]
    with pytest.raises(LeaseExpired):
        q.heartbeat(first.lease_id, first.token)
    second = q.claim(gpu_uuids=["GPU-a"], owner_pid=os.getpid())
    assert second is not None
    assert second.attempt == 2
    assert second.attempt_dir != first.attempt_dir
    events = (__import__("pathlib").Path(first.attempt_dir) / "events.jsonl").read_text()
    assert '"event": "expired"' in events


def test_retry_budget_blocks_job(tmp_path):
    q = make_queue(tmp_path, max_attempts=1)
    lease = q.claim(gpu_uuids=["GPU-a"], owner_pid=os.getpid())
    assert lease is not None
    job = q.retry(lease.lease_id, lease.token, error="worker failed")
    assert job.status == "blocked"
    assert job.last_error == "worker failed"
    assert q.claim(gpu_uuids=["GPU-a"], owner_pid=os.getpid()) is None


def test_heartbeat_rejects_foreign_gpu_process(tmp_path):
    clock = FakeClock()
    gpu_state = [{"uuid": "GPU-a", "index": "0", "pids": []}]
    q = GPULeaseQueue(tmp_path / "queue.json", gpu_probe=lambda: gpu_state, clock=clock)
    q.enqueue("first")
    lease = q.claim(gpu_uuids=["GPU-a"], owner_pid=os.getpid())
    assert lease is not None
    gpu_state[0]["pids"] = [999999]
    with pytest.raises(LeaseConflict):
        q.heartbeat(lease.lease_id, lease.token)


def test_two_threads_cannot_claim_same_gpu(tmp_path):
    q = make_queue(tmp_path)
    q.enqueue("second")
    results = []

    def claim():
        try:
            results.append(q.claim(gpu_uuids=["GPU-a"], owner_pid=os.getpid()))
        except LeaseConflict:
            results.append("conflict")

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(result is not None and result != "conflict" for result in results) == 1
    assert "conflict" in results


def test_claim_infers_payload_gpu_count_and_allows_disjoint_groups(tmp_path):
    q = GPULeaseQueue(tmp_path / "queue.json", gpu_probe=probe)
    q.enqueue("pair", {"gpu_count": 1})
    q.enqueue("other", {"gpu_count": 1})
    first = q.claim(owner_pid=os.getpid())
    second = q.claim(owner_pid=os.getpid())
    assert first is not None and second is not None
    assert len(first.gpu_uuids) == len(second.gpu_uuids) == 1
    assert set(first.gpu_uuids).isdisjoint(second.gpu_uuids)


def test_exclusive_job_waits_for_all_gpus_and_then_claims_them(tmp_path):
    q = GPULeaseQueue(tmp_path / "queue.json", gpu_probe=probe)
    q.enqueue("group", {"gpu_count": 1})
    q.enqueue("whole-host", {"gpu_count": 2, "exclusive": True})
    group = q.claim(owner_pid=os.getpid())
    assert group is not None
    assert q.claim(owner_pid=os.getpid()) is None
    q.complete(group.lease_id, group.token)
    whole = q.claim(owner_pid=os.getpid())
    assert whole is not None and set(whole.gpu_uuids) == {"GPU-a", "GPU-b"}
