import importlib.util
import json
from pathlib import Path


def worker():
    path = Path(__file__).resolve().parents[2] / 'scripts/2026-09-24_run_matrix_worker.py'
    spec = importlib.util.spec_from_file_location('matrix_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    queue = module.MatrixQueue.__new__(module.MatrixQueue)
    queue.run_id = 'new-matrix'
    return queue


def test_worker_cannot_claim_an_unrelated_or_missing_scope():
    queue = worker()
    state = dict(jobs={}, leases={})
    assert not queue._deps_ready(state, dict(payload={'run_id': 'low-m-extra'}))
    assert not queue._deps_ready(state, dict(payload={}))
    assert queue._deps_ready(state, dict(payload={'run_id': 'new-matrix'}))


def test_scope_filter_keeps_native_dependencies_and_active_lease_exclusion():
    queue = worker()
    job = dict(payload={'run_id': 'new-matrix', 'after_terminal': ['prior']})
    state = dict(jobs={'prior': dict(status='succeeded', lease_id=None)},
                 leases={'lease': dict(job_id='prior', status='active')})
    assert not queue._deps_ready(state, job)
    state['leases']['lease']['status'] = 'succeeded'
    assert queue._deps_ready(state, job)


def test_shared_cli_worker_obeys_handoff_and_retains_dependencies(tmp_path):
    path = Path(__file__).resolve().parents[2] / 'scripts/2026-09-22_gpu_campaign_queue.py'
    spec = importlib.util.spec_from_file_location('shared_queue_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    queue = module.QueueWithinScope(tmp_path / 'queue.json')
    queue.scope_path.write_text(json.dumps(dict(status='active', allowed_run_ids=['matrix'])))
    state = dict(jobs={}, leases={})
    assert not queue._deps_ready(state, dict(payload={'run_id': 'low-m-extra'}))
    assert queue._deps_ready(state, dict(payload={'run_id': 'matrix'}))
    assert not queue._deps_ready(state, dict(payload={'run_id': 'matrix', 'after_terminal': ['missing']}))
    queue.scope_path.write_text(json.dumps(dict(status='released', allowed_run_ids=['matrix'])))
    assert queue._deps_ready(state, dict(payload={'run_id': 'low-m-extra'}))
