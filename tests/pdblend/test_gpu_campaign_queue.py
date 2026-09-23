from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path


def _load_queue_cli():
    path = Path(__file__).parents[2] / "scripts/2026-09-22_gpu_campaign_queue.py"
    spec = importlib.util.spec_from_file_location("gpu_campaign_queue", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_worker_slots_are_replenished_when_fast_job_finishes(monkeypatch, tmp_path):
    cli = _load_queue_cli()
    lock = threading.Lock()
    starts: dict[int, float] = {}
    ends: dict[int, float] = {}
    next_job = 0
    completed = 0
    durations = (0.03, 0.25, 0.03)

    class FakeQueue:
        def __init__(self, _path):
            pass

        def list_jobs(self):
            return [type("Job", (), {"status": "queued"})()] if next_job < 3 else []

    def fake_run_one(_queue):
        nonlocal next_job, completed
        with lock:
            job = next_job
            next_job += 1
            starts[job] = time.monotonic()
        if job >= 3:
            return False
        time.sleep(durations[job])
        with lock:
            ends[job] = time.monotonic()
            completed += 1
        return True

    monkeypatch.setattr(cli, "GPULeaseQueue", FakeQueue)
    monkeypatch.setattr(cli, "run_one", fake_run_one)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(cli._worker_loop, tmp_path / "queue.json",
                               once=False, idle_exit=True) for _ in range(2)]
        for future in futures:
            future.result()

    assert starts[2] < ends[1]


def test_resume_copies_raw_and_samples_only_for_same_source_and_uuids(tmp_path):
    from pdblend.experimentation.lease import GPULeaseQueue
    from pdblend.experimentation.worker import restore_resume_artifacts

    probe = lambda: [{"uuid": "GPU-a", "index": "0", "pids": []}]
    queue = GPULeaseQueue(tmp_path / "queue.json", gpu_probe=probe)
    payload = {"gpu_count": 1, "resume_profile": True,
               "argv": ["env", "PDBLEND_SOURCE_SHA256=abc", "PDBLEND_IMAGE_ID=sha256:image"]}
    queue.enqueue("resume", payload, max_attempts=2)
    first = queue.claim(owner_pid=__import__("os").getpid())
    previous = Path(first.attempt_dir)
    (previous / "raw.json").write_text('{"source_hash":"abc"}\n')
    (previous / "samples").mkdir()
    (previous / "samples" / "decode.json").write_text('{"sample":1}\n')
    queue.retry(first.lease_id, first.token, error="interrupted")
    second = queue.claim(owner_pid=__import__("os").getpid())
    metadata = restore_resume_artifacts(queue, second, payload)
    assert metadata and (Path(second.attempt_dir) / "raw.json").read_text() == '{"source_hash":"abc"}\n'
    assert (Path(second.attempt_dir) / "samples/decode.json").is_file()

    assert restore_resume_artifacts(
        queue, second, {"gpu_count": 1, "resume_profile": True,
                        "argv": ["env", "PDBLEND_SOURCE_SHA256=def", "PDBLEND_IMAGE_ID=sha256:image"]}) is None


def test_opted_in_failed_attempt_is_requeued(tmp_path):
    import sys
    from pdblend.experimentation.lease import GPULeaseQueue
    from pdblend.experimentation.worker import run_one

    queue = GPULeaseQueue(tmp_path / "queue.json",
                          gpu_probe=lambda: [{"uuid": "GPU-a", "index": "0", "pids": []}])
    queue.enqueue("retry", {
        "gpu_count": 1, "resume_profile": True, "source_sha256": "src",
        "image_digest": "sha256:image", "argv": [sys.executable, "-c", "raise SystemExit(3)"],
    }, max_attempts=2)
    assert run_one(queue, lock_path=str(tmp_path / "worker.lock"))
    job = queue.list_jobs()[0]
    assert job.status == "queued" and job.attempts == 1


def test_stop_file_drains_without_claiming_new_work(monkeypatch, tmp_path):
    cli = _load_queue_cli()
    stop = tmp_path / "stop"
    stop.write_text("drain\n")
    called = []
    monkeypatch.setattr(cli, "run_one", lambda _queue: called.append(True))
    cli._worker_loop(tmp_path / "queue.json", once=False, idle_exit=False, stop_file=stop)
    assert called == []


def test_reschedule_changes_only_staging_and_preserves_immutable_inputs():
    import copy
    path = Path(__file__).parents[2] / 'scripts/2026-09-23_resume_independent_acceptance.py'
    spec = importlib.util.spec_from_file_location('resume_acceptance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = [[{'job_id': 'dist', 'payload': {
        'depends_on': ['dynamo'], 'container_name': 'dist',
        'argv': ['docker', '--name', 'dist', '-v', '/frozen:/src:ro'],
        'source_sha256': 'frozen', 'exact_inputs_sha256': {'dependencies': ['dynamo']}}}],
        [{'job_id': 'profile', 'payload': {'depends_on': ['dist'],
          'container_name': 'profile', 'argv': ['docker', '--name', 'profile'],
          'immutable_input_sha256': 'samples', 'cohort_dir': '/original/wave'}}]]
    before = copy.deepcopy(groups)
    jobs, mapping = module.replacements(groups)
    assert groups == before
    assert jobs[0]['payload']['after_terminal'] == ['dynamo']
    assert jobs[1]['payload']['after_terminal'] == [mapping['dist']]
    assert all(job['payload']['depends_on'] == [] for job in jobs)
    assert jobs[0]['payload']['source_sha256'] == 'frozen'
    assert jobs[1]['payload']['immutable_input_sha256'] == 'samples'
    assert jobs[1]['payload']['cohort_dir'] == '/original/wave'
    assert module.replacements(groups) == (jobs, mapping)
