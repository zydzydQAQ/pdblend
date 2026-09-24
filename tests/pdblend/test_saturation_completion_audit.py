"""Completion accounting preserves observations and refuses identity splicing."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[2] / 'scripts/2026-09-24_audit_saturation_completion.py'
    spec = importlib.util.spec_from_file_location('saturation_completion_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bound(audit, tmp_path, name, obj):
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return audit.binding(path)


def test_slo_failure_and_missing_energy_remain_observations(audit, tmp_path):
    point = dict(name='pd-new', revision='new')
    receipt = bound(audit, tmp_path, 'receipt.json',
                    dict(point_sha256=audit.digest(point), recorded_window_complete=True))
    row = dict(point_id=point['name'], point_sha256=audit.digest(point),
               receipt_path=receipt['path'], receipt_sha256=receipt['sha256'],
               analysis_slo_pass='False', energy_service_j='', energy_tail_j='42')
    result = audit.observed(point, [row])
    assert result['status'] == 'observed'
    assert result['observations'][0]['slo_pass'] is False
    assert result['observations'][0]['service_energy_present'] is False


def test_old_revision_cannot_complete_new_point(audit, tmp_path):
    old = dict(name='pd', revision='old')
    new = dict(name='pd', revision='new')
    row = dict(point_id='pd', point_sha256=audit.digest(old), receipt_sha256='old')
    assert audit.observed(new, [row])['status'] == 'pending'


def test_terminal_queue_without_receipt_or_bound_failure_is_pending(audit):
    point = dict(name='pd', revision='new')
    row = dict(point_id='pd', point_sha256=audit.digest(point), status='succeeded')
    assert audit.observed(point, [row])['status'] == 'pending'


def test_bound_execution_failure_accounts_for_unmeasured_point(audit, tmp_path):
    point = dict(name='pd', revision='new')
    report = bound(audit, tmp_path, 'completion.json',
                   dict(planned_points={'pd': audit.digest(point)}, complete=False, error='engine startup failed'))
    row = dict(point_id='pd', point_sha256=audit.digest(point),
               session_completion_path=report['path'], session_completion_sha256=report['sha256'])
    result = audit.observed(point, [row])
    assert result['status'] == 'obstructed'
    assert not result['observations']


def test_failure_for_another_revision_does_not_count(audit, tmp_path):
    point = dict(name='pd', revision='new')
    report = bound(audit, tmp_path, 'completion.json',
                   dict(planned_points={'pd': 'a' * 64}, complete=False, error='failed'))
    row = dict(point_id='pd', point_sha256=audit.digest(point),
               session_completion_path=report['path'], session_completion_sha256=report['sha256'])
    with pytest.raises(ValueError, match='not bound'):
        audit.observed(point, [row])


def test_changed_receipt_bytes_rejected(audit, tmp_path):
    point = dict(name='pd', revision='new')
    ref = bound(audit, tmp_path, 'receipt.json', dict(point_sha256=audit.digest(point)))
    Path(ref['path']).write_text('{}')
    row = dict(point_id='pd', point_sha256=audit.digest(point), receipt_path=ref['path'],
               receipt_sha256=ref['sha256'])
    with pytest.raises(ValueError):
        audit.observed(point, [row])


def test_multiple_attempts_are_preserved_without_energy_selection(audit, tmp_path):
    point = dict(name='pd', revision='new')
    rows = []
    for index in range(2):
        ref = bound(audit, tmp_path, f'receipt{index}.json',
                    dict(point_sha256=audit.digest(point), recorded_window_complete=True, index=index))
        rows.append(dict(point_id='pd', point_sha256=audit.digest(point), receipt_path=ref['path'],
                         receipt_sha256=ref['sha256'], energy_service_j=str(100 + index)))
    result = audit.observed(point, rows)
    assert len(result['observations']) == 2


def test_reused_frozen_receipt_requires_publication(audit, tmp_path):
    ref = bound(audit, tmp_path, 'receipt.json', dict(point_sha256='a' * 64))
    assert audit.reused_observation(ref, [])['status'] == 'pending'


def test_reset_failure_receipt_is_obstruction_not_completed_window(audit, tmp_path):
    point = dict(name='pd', revision='new')
    ref = bound(audit, tmp_path, 'receipt.json',
                dict(point_sha256=audit.digest(point), recorded_window_complete=False,
                     error='reset failed before measurement'))
    row = dict(point_id='pd', point_sha256=audit.digest(point),
               receipt_path=ref['path'], receipt_sha256=ref['sha256'])
    outcome = audit.observed(point, [row])
    assert outcome['status'] == 'obstructed'
    assert len(outcome['observations']) == 1
    assert outcome['observations'][0]['recorded_window_complete'] is False


def test_new_group_failure_requires_exact_point_and_finished_scoped_lease(audit, tmp_path):
    point = dict(name='baseline', revision='frozen')
    session = tmp_path / 'session'
    session.mkdir()
    bound(audit, session, 'completion.json', dict(schema='resident-group-session/v1',
        planned_points={'baseline': audit.digest(point)}, complete=False, error='load failed'))
    queue = dict(jobs={'job': dict(payload={'run_id': 'new'})}, leases={
        'lease': dict(job_id='job', attempt_dir=str(tmp_path), status='active')})
    assert audit.group_failure(point, queue, 'new') == []
    queue['leases']['lease']['status'] = 'failed'
    assert len(audit.group_failure(point, queue, 'new')) == 1
    assert audit.group_failure(point, queue, 'old') == []
    assert audit.group_failure(dict(point, revision='different'), queue, 'new') == []
