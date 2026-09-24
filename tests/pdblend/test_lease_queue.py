from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from pdblend.experimentation.lease import GPULeaseQueue, LeaseConflict, LeaseExpired
from pdblend.experimentation.worker import run_one


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


def test_workers_execute_disjoint_subprocesses_concurrently(tmp_path):
    q = GPULeaseQueue(tmp_path / "queue.json", gpu_probe=probe)
    code = (
        "import json, pathlib, sys, time; "
        "start=time.time(); time.sleep(0.35); end=time.time(); "
        "p=pathlib.Path(sys.argv[1]); "
        "(p/'completion.json').write_text(json.dumps({'status':'passed','complete':True,'gpu_uuids':sys.argv[2],'local':sys.argv[3],'start':start,'end':end}))"
    )
    for name in ("a", "b"):
        q.enqueue(name, {
            "gpu_count": 1,
            "argv": [sys.executable, "-c", code, "{attempt_dir}", "{lease_gpu_uuids}", "{lease_local_indices}"],
            "required_receipts": ["completion.json"],
    })
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_one(GPULeaseQueue(q.path, gpu_probe=probe),
            lock_path=str(tmp_path/'worker.lock')), range(2)))
    assert results == [True, True]
    receipts = [json.loads(path.read_text()) for path in tmp_path.glob("queue-attempts/*/*/completion.json")]
    assert len(receipts) == 2
    assert min(item["end"] for item in receipts) > max(item["start"] for item in receipts)
    assert {job.status for job in q.list_jobs()} == {"succeeded"}


def test_ready_formal_reservation_drains_groups_without_preemption(tmp_path):
    q = GPULeaseQueue(tmp_path / 'queue.json', gpu_probe=probe)
    q.enqueue('existing', {'gpu_count': 1})
    running = q.claim(owner_pid=os.getpid(), lock_mode=False)
    q.enqueue('formal', {'gpu_count': 2, 'exclusive': True, 'reserve_host': True})
    q.enqueue('new-group', {'gpu_count': 1}, priority=999)
    assert q.claim(owner_pid=os.getpid(), lock_mode=False) is None
    assert q.snapshot()['jobs']['existing']['status'] == 'running'
    assert q.claim(owner_pid=os.getpid(), lock_mode=True) is None
    q.complete(running.lease_id, running.token)
    formal = q.claim(owner_pid=os.getpid(), lock_mode=True)
    assert formal.job_id == 'formal'
    q.complete(formal.lease_id, formal.token)
    assert q.claim(owner_pid=os.getpid(), lock_mode=False).job_id == 'new-group'


def test_unready_formal_reservation_does_not_idle_independent_work(tmp_path):
    q = GPULeaseQueue(tmp_path / 'queue.json', gpu_probe=probe)
    q.enqueue('formal', {'gpu_count': 2, 'exclusive': True, 'reserve_host': True},
              depends_on=['missing-profile'])
    q.enqueue('independent', {'gpu_count': 1})
    assert q.claim(owner_pid=os.getpid(), lock_mode=False).job_id == 'independent'


def test_explicit_ready_prerequisite_precedes_reservation_across_worker_modes(tmp_path):
    q = GPULeaseQueue(tmp_path/'queue.json', gpu_probe=probe)
    q.enqueue('formal', {'gpu_count':2,'exclusive':True,'reserve_host':True}, priority=798)
    q.enqueue('prerequisite', {'gpu_count':1,'precedes_host_reservations':True}, priority=799)
    q.enqueue('ordinary', {'gpu_count':1}, priority=999)
    assert q.claim(owner_pid=os.getpid(), lock_mode=True) is None
    first=q.claim(owner_pid=os.getpid(), lock_mode=False)
    assert first.job_id=='prerequisite'
    # Existing host reservation again blocks ordinary backfill; no lease is
    # interrupted to start the eight-GPU measurement prematurely.
    assert q.claim(owner_pid=os.getpid(), lock_mode=False) is None
    assert q.claim(owner_pid=os.getpid(), lock_mode=True) is None
    q.complete(first.lease_id,first.token)
    assert q.claim(owner_pid=os.getpid(), lock_mode=True).job_id=='formal'


@pytest.mark.parametrize('kind',['equal_priority','lower_priority','unready','non_boolean_flag'])
def test_prerequisite_override_requires_explicit_ready_higher_priority(tmp_path,kind):
    q=GPULeaseQueue(tmp_path/'queue.json',gpu_probe=probe)
    q.enqueue('formal',{'gpu_count':2,'exclusive':True,'reserve_host':True},priority=10)
    priority={'equal_priority':10,'lower_priority':9}.get(kind,11)
    q.enqueue('prerequisite',{'gpu_count':1,'precedes_host_reservations':'true' if kind=='non_boolean_flag' else True},
        priority=priority,depends_on=['missing'] if kind=='unready' else [])
    assert q.claim(owner_pid=os.getpid(),lock_mode=False) is None
    assert q.claim(owner_pid=os.getpid(),lock_mode=True).job_id=='formal'


def test_prerequisite_override_never_splits_an_active_sampling_cohort(tmp_path):
    q=GPULeaseQueue(tmp_path/'queue.json',gpu_probe=probe)
    q.enqueue('peer-a',{'gpu_count':1,'sampling_cohort':'wave'})
    first=q.claim(owner_pid=os.getpid(),lock_mode=False)
    q.enqueue('formal',{'gpu_count':2,'exclusive':True,'reserve_host':True},priority=10)
    q.enqueue('prerequisite',{'gpu_count':1,'precedes_host_reservations':True},priority=11)
    q.enqueue('peer-b',{'gpu_count':1,'sampling_cohort':'wave'})
    assert q.claim(owner_pid=os.getpid(),lock_mode=True) is None
    second=q.claim(owner_pid=os.getpid(),lock_mode=False)
    assert second.job_id=='peer-b'
    q.complete(first.lease_id,first.token)
    assert q.claim(owner_pid=os.getpid(),lock_mode=False) is None
    q.complete(second.lease_id,second.token)
    assert q.claim(owner_pid=os.getpid(),lock_mode=False).job_id=='prerequisite'


def test_sampling_cohort_rejects_unknown_peer_but_admits_barrier_members(tmp_path):
    q = GPULeaseQueue(tmp_path/'queue.json', gpu_probe=probe)
    q.enqueue('profile-a', {'gpu_count':1, 'sampling_cohort':'wave'})
    a = q.claim(owner_pid=os.getpid(), lock_mode=False)
    q.enqueue('formal', {'gpu_count':2, 'exclusive':True, 'reserve_host':True}, priority=999)
    q.enqueue('unknown-peer', {'gpu_count':1}, priority=1000)
    q.enqueue('profile-b', {'gpu_count':1, 'sampling_cohort':'wave'})
    b = q.claim(owner_pid=os.getpid(), lock_mode=False)
    assert b.job_id == 'profile-b'
    q.complete(a.lease_id, a.token)
    assert q.claim(owner_pid=os.getpid(), lock_mode=False) is None
    q.complete(b.lease_id, b.token)
    assert q.claim(owner_pid=os.getpid(), lock_mode=True).job_id == 'formal'


def test_profile_cohort_cannot_start_while_unqualified_functional_job_runs(tmp_path):
    q = GPULeaseQueue(tmp_path/'queue.json', gpu_probe=probe)
    q.enqueue('functional', {'gpu_count':1})
    run = q.claim(owner_pid=os.getpid())
    q.enqueue('profile', {'gpu_count':1, 'sampling_cohort':'wave'})
    assert q.claim(owner_pid=os.getpid()) is None
    q.complete(run.lease_id, run.token)
    assert q.claim(owner_pid=os.getpid()).job_id == 'profile'


@pytest.mark.parametrize('terminal', ['succeeded', 'failed', 'cancelled'])
def test_after_terminal_waits_for_release_but_not_success(tmp_path, terminal):
    q = GPULeaseQueue(tmp_path / 'queue.json', gpu_probe=probe)
    q.enqueue('before', {'gpu_count': 1})
    q.enqueue('after', {'gpu_count': 1}, after_terminal=['before'])
    first = q.claim(owner_pid=os.getpid())
    assert first.job_id == 'before'
    assert q.claim(owner_pid=os.getpid()) is None
    q.complete(first.lease_id, first.token, status=terminal)
    assert q.claim(owner_pid=os.getpid()).job_id == 'after'


def test_after_terminal_does_not_override_required_success(tmp_path):
    q = GPULeaseQueue(tmp_path / 'queue.json', gpu_probe=probe)
    q.enqueue('before')
    lease = q.claim(owner_pid=os.getpid())
    q.complete(lease.lease_id, lease.token, status='failed')
    q.enqueue('after', depends_on=['before'], after_terminal=['before'])
    q.enqueue('unknown', after_terminal=['missing'])
    assert q.claim(owner_pid=os.getpid()) is None


def test_terminal_status_with_live_lease_is_not_ready(tmp_path):
    q = GPULeaseQueue(tmp_path / 'queue.json', gpu_probe=probe)
    state = {'jobs': {'before': {'status': 'failed', 'lease_id': None}},
             'leases': {'lease': {'job_id': 'before', 'status': 'active'}}}
    assert not q._deps_ready(state, {'payload': {'after_terminal': ['before']}})


def test_supersede_preserves_payload_and_rejects_executed_job(tmp_path):
    q = GPULeaseQueue(tmp_path / 'queue.json', gpu_probe=probe)
    q.enqueue('old', {'gpu_count': 1, 'depends_on': ['missing']})
    q.enqueue('new')
    old_payload = q.snapshot()['jobs']['old']['payload']
    assert q.supersede_queued('old', 'new').status == 'cancelled'
    assert q.snapshot()['jobs']['old']['payload'] == old_payload
    assert q.supersede_queued('old', 'new').status == 'cancelled'
    q.claim(owner_pid=os.getpid())
    with pytest.raises(LeaseConflict):
        q.supersede_queued('new', 'old')


def test_atomic_reschedule_keeps_running_job_and_rolls_back_failed_batch(tmp_path):
    q = GPULeaseQueue(tmp_path / 'queue.json', gpu_probe=probe)
    q.enqueue('running')
    lease = q.claim(owner_pid=os.getpid())
    q.enqueue('pending', after_terminal=['running'])
    before = q.snapshot()
    with pytest.raises(LeaseConflict):
        q.enqueue_replacements([
            {'job_id': 'new-pending', 'payload': {'supersedes_job_id': 'pending'}},
            {'job_id': 'bad', 'payload': {'supersedes_job_id': 'running'}}])
    assert q.snapshot() == before
    specs = [{'job_id': 'retry', 'payload': {'after_terminal': ['running']}},
             {'job_id': 'new-pending', 'payload': {
                 'after_terminal': ['retry'], 'supersedes_job_id': 'pending'}}]
    q.enqueue_replacements(specs)
    q.enqueue_replacements(specs)
    assert q.snapshot()['jobs']['pending']['status'] == 'cancelled'
    assert q.active_leases()[0].lease_id == lease.lease_id
    assert q.claim(owner_pid=os.getpid()) is None
