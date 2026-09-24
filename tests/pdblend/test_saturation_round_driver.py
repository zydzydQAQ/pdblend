"""The operational driver must resume without changing frozen GPU jobs."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[2] / 'scripts/2026-09-24_run_saturation_round.py'
    spec = importlib.util.spec_from_file_location('saturation_round_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_user_priority_handoff_stops_before_loading_jobs_or_touching_queue(driver, tmp_path, monkeypatch):
    package = tmp_path / 'old-round'
    write(package / 'orchestration/priority-coordination.json',
          {'status': 'user_confirmed_repairs_first'})
    monkeypatch.setattr(driver, 'GPULeaseQueue',
                        lambda *a: pytest.fail('held round reached queue constructor'))
    with pytest.raises(driver.PriorityHandoff):
        driver.Runner(package, tmp_path / 'not-needed.json')
    assert not (package / 'orchestration/owner.lock').exists()


def test_late_priority_handoff_prevents_new_submission(driver, tmp_path):
    runner = driver.Runner.__new__(driver.Runner)
    runner.package = tmp_path
    write(tmp_path / 'orchestration/priority-coordination.json',
          {'status': 'awaiting_priority_clarification'})
    with pytest.raises(driver.PriorityHandoff):
        runner.enqueue({'job_id': 'must-not-be-submitted'})


def test_restart_restores_campaign_and_completed_models_before_publish(driver, tmp_path, monkeypatch):
    package = tmp_path / 'round'
    campaign = package / 'baselines-7b/campaign.json'
    write(campaign, {'points': ['old', 'dynamic']})
    write(package / 'jobs.json', [])
    write(package / 'orchestration/status.json', dict(run_id='round', started_s=10,
        phase='baseline_supplements_and_endpoints', model='Qwen2.5-14B-Instruct', jobs=['done'],
        current_campaign=str(campaign), completed_models=['Qwen2.5-7B-Instruct'],
        extension_manifests=[{'path': 'immutable-manifest', 'sha256': 'identity'}]))
    launch = tmp_path / 'old.json'
    write(launch, {'argv': []})
    monkeypatch.setattr(driver, 'QUEUE', tmp_path / 'queue.json')
    runner = driver.Runner(package, launch)
    try:
        assert runner.campaign == campaign
        assert runner.state['started_s'] == 10
        assert runner.state['completed_models'] == ['Qwen2.5-7B-Instruct']
        assert runner.manifests == [{'path': 'immutable-manifest', 'sha256': 'identity'}]
    finally:
        runner.lock.close()


def test_baselines_submitted_sequentially_and_completed_models_skipped(driver, tmp_path, monkeypatch):
    runner = driver.Runner.__new__(driver.Runner)
    runner.package, runner.out = tmp_path, tmp_path / 'orchestration'
    runner.out.mkdir()
    monkeypatch.setattr(driver, 'QUEUE', tmp_path / 'queue.json')
    runner.jobs = [dict(payload={'model_id': m}) for m in ('done', 'current')]
    runner.state = dict(completed_models=['done'], baseline_campaigns={}, jobs=[])
    runner.publish = lambda **kw: None
    runner.emit = lambda *args, **kw: None
    calls = []
    runner.pd = lambda job: calls.append(('pd', job['payload']['model_id']))
    runner.selection = lambda *args: tmp_path / 'selection.json'
    runner.prepare_baselines = lambda *args: (tmp_path, {'groups': [{'points': [1, 2]}]},
        [{'job_id': 'mixed'}, {'job_id': 'distserve'}])
    runner.enqueue = lambda job: calls.append(('enqueue', job['job_id']))
    # blocked is terminal too: no dependency on a succeeded-only queue edge.
    runner.wait = lambda ids: calls.append(('wait', ids[0])) or {'status': 'blocked'}
    # Recovery has its own queue/group fixtures; this test checks model and
    # baseline ordering through the public runner boundary.
    def run_baseline(job):
        runner.enqueue(job)
        return runner.wait([job['job_id']])
    runner.run_baseline = run_baseline
    runner.audit_results = lambda: dict(has_obstructions=False, all_scope_accounted=False)
    runner.run()
    assert calls == [('pd', 'current'), ('enqueue', 'mixed'), ('wait', 'mixed'),
                     ('enqueue', 'distserve'), ('wait', 'distserve')]
    assert runner.state['completed_models'] == ['done', 'current']
    assert runner.state['phase'] == 'needs_attention'
    assert (runner.out / 'worker.stop').is_file()
    # A restart must retain the same stop evidence and still require an audit.
    stop = (runner.out / 'worker.stop').read_bytes()
    runner.run()
    assert (runner.out / 'worker.stop').read_bytes() == stop


def test_partial_preparation_is_preserved_and_rebuilt_at_new_bound_path(driver, tmp_path):
    runner = driver.Runner.__new__(driver.Runner)
    runner.package, runner.campaign = tmp_path, tmp_path / 'campaign.json'
    runner.state = {'baseline_campaigns': {}}
    runner.emit = lambda *args, **kw: None
    partial = tmp_path / 'baselines-7b'
    write(partial / 'campaign.json', {'incomplete': True})
    calls = []

    class Builder:
        def prepare(self, campaign, selection, out):
            calls.append(out)
            out.mkdir()
            return {'groups': []}, []

    runner.baselines = Builder()
    out, _, jobs = runner.prepare_baselines('Qwen2.5-7B-Instruct', tmp_path / 'selection.json')
    assert calls == [tmp_path / 'baselines-7b-recovery-0001']
    assert out == calls[0] and jobs == []
    assert json.loads((partial / 'campaign.json').read_text()) == {'incomplete': True}


def test_prepared_marker_rejects_changed_selection(driver, tmp_path):
    runner = driver.Runner.__new__(driver.Runner)
    runner.package = tmp_path
    runner.state = {'baseline_campaigns': {}}
    out = tmp_path / 'baselines-7b'
    old, new = tmp_path / 'old-selection.json', tmp_path / 'new-selection.json'
    write(old, {'lower': 1.25})
    write(new, {'lower': 1.5625})
    write(out / 'prepared.json', {'selection': driver.cc.binding(old)})
    with pytest.raises(ValueError, match='selection changed'):
        runner.prepare_baselines('Qwen2.5-7B-Instruct', new)


def test_historical_registry_is_passed_to_new_preparation_only(driver, tmp_path):
    runner = driver.Runner.__new__(driver.Runner)
    runner.package, runner.campaign = tmp_path, tmp_path / 'campaign.json'
    runner.state = {'baseline_campaigns': {}}
    runner.baseline_registry_ref = {'path': 'frozen-history', 'sha256': 'h' * 64}
    runner.emit = lambda *a, **kw: None
    calls = []

    class Builder:
        def prepare(self, campaign, selection, out, **kwargs):
            calls.append(kwargs)
            return {'groups': []}, []

    runner.baselines = Builder()
    runner.prepare_baselines('Qwen2.5-7B-Instruct', tmp_path / 'selection.json')
    assert calls == [{'historical_baseline_registry': runner.baseline_registry_ref}]


def test_cold_export_does_not_rehash_during_an_active_lease(driver, tmp_path, monkeypatch):
    runner = driver.Runner.__new__(driver.Runner)
    runner.package = tmp_path / 'round'
    runner.cold_export_pending = runner.defer_cold_export_until_idle = True
    runner.state = {}
    events = []
    runner.emit = lambda event, **fields: events.append((event, fields))
    monkeypatch.setattr(driver, 'read', lambda _: dict(
        leases={'l': {'status': 'active', 'job_id': 'measuring'}},
        jobs={'measuring': {'payload': {'run_id': 'round'}}}))
    monkeypatch.setattr(driver.cc, 'comparison_watch_inputs',
                        lambda *a, **kw: pytest.fail('cold evidence scan reached an active lease'))
    runner.publish(force=True)
    runner.publish(force=True)
    assert events == [('cold_export_deferred_until_lease_release', {'jobs': ['measuring']})]
    assert runner.cold_export_pending


def test_publication_preserves_old_receipts_and_all_metric_values(driver):
    prior = [{'receipt_sha256': 'old', 'energy_service_j': '10', 'goodput_request_s': '1'}]
    driver.preserve_published_metrics(prior, prior + [{'receipt_sha256': 'new', 'energy_service_j': '5'}])
    with pytest.raises(ValueError, match='drop'):
        driver.preserve_published_metrics(prior, [])
    with pytest.raises(ValueError, match='change'):
        driver.preserve_published_metrics(prior, [{'receipt_sha256': 'old', 'energy_service_j': '9'}])


def test_publication_rejects_inconsistent_duplicate_receipt(driver):
    with pytest.raises(ValueError, match='conflicting'):
        driver.preserve_published_metrics([], [
            {'receipt_sha256': 'same', 'energy_service_j': '1'},
            {'receipt_sha256': 'same', 'energy_service_j': '2'}])


def test_cold_export_defers_for_other_authorized_gpu_owner(driver, tmp_path, monkeypatch):
    runner = driver.Runner.__new__(driver.Runner)
    runner.package = tmp_path / 'matrix'
    runner.cold_export_pending = runner.defer_cold_export_until_idle = True
    runner.state = {}
    runner.emit = lambda *a, **kw: None
    monkeypatch.setattr(driver, 'read', lambda _: dict(
        leases={'l': {'status': 'active', 'job_id': 'repairs'}},
        jobs={'repairs': {'payload': {'run_id': 'other-authorized-round'}}}))
    monkeypatch.setattr(driver.cc, 'comparison_watch_inputs',
                        lambda *a, **kw: pytest.fail('cold scan interrupted another GPU owner'))
    runner.publish(force=True)
    assert runner.cold_export_pending


def test_historical_manifests_are_kept_out_of_new_continuation_state(driver, tmp_path, monkeypatch):
    package = tmp_path / 'round'
    history = {'path': 'old-boundary', 'sha256': 'old'}
    write(package / 'campaign.json', {'historical_extension_manifests': [history]})
    write(package / 'jobs.json', [])
    launch = tmp_path / 'old.json'
    write(launch, {'argv': []})
    monkeypatch.setattr(driver, 'QUEUE', tmp_path / 'queue.json')
    runner = driver.Runner(package, launch)
    try:
        assert runner.historical_manifests == [history]
        assert runner.manifests == []
        assert runner.state['extension_manifests'] == []
    finally:
        runner.lock.close()
