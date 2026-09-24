import json
import os
from pathlib import Path

import pytest

from pdblend.bench.comparison_campaign import binding
from pdblend.bench.recovery_campaign import arm_options, report
from pdblend.bench.resident_session import digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return binding(path)


def campaign(tmp_path, *, energy=100., status='succeeded'):
    shared = put(tmp_path/'shared.json', {})
    points, groups, jobs = [], [], []
    for repeat in range(3):
        for arm in ('control', 'shield'):
            name = f'{arm}-{repeat}'
            meta = dict(case='case', arm=arm, repeat=repeat, qualification='diagnostic_only')
            point = dict(name=name, revision=shared['sha256'], trace=shared, source_manifest=shared,
                recovery_experiment=meta, inputs=dict(profiles=[shared], system_config=shared,
                    planning_trace=shared, offline_choice=shared))
            points.append(point)
            group = dict(session_id=name, engine_signature='engine', points=[point]); groups.append(group)
            payload = dict(session_id=name, source_sha256=point['revision'])
            jobs.append(dict(job_id=name, payload=payload))
            attempt = tmp_path/'queue-attempts'/name/'attempt-1'
            put(attempt/'manifest.json', dict(immutable=True, job_id=name, payload=payload))
            window = attempt/'session/windows'/name
            put(window/'point.json', point)
            result = dict(metrics=dict(energy_service_j=energy, energy_tail_j=10.,
                good_output_tokens=10, slo_pass=True), acceptance=dict(measurement_evidence_valid=True, checked_gates=[
                    'metering.raw_eight_gpu_window', 'pdblend.canonical_metrics']))
            refs = dict(result=put(window/'result.json', result), drain=put(window/'drain.json', {'passed': True}))
            put(window/'receipt.json', dict(point_sha256=digest(point), result=result,
                cleanup_passed=True, session_id=name, engine_signature='engine',
                artifacts={k+'.json': v['sha256'] for k,v in refs.items()}))
    path = tmp_path/'campaign.json'
    put(path, dict(points=points, groups=groups, repeats=3))
    put(tmp_path/'jobs.json', jobs)
    queue = tmp_path/'queue.json'
    put(queue, dict(jobs=[dict(job_id=j['job_id'], status=status) for j in jobs]))
    return path, queue


def test_arm_settings_isolate_mechanisms():
    assert arm_options('recovery')['shield_mode'] == 'legacy'
    assert arm_options('recovery')['slo_routing']
    assert not arm_options('shield')['slo_routing']
    assert not arm_options('shield')['safety_recovery']
    assert arm_options('all_m')['experiment_mode'] == 'freeze_initial_all_m'
    with pytest.raises(ValueError, match='historical source'):
        arm_options('control')


def test_complete_observations_do_not_manufacture_statistical_or_profile_acceptance(tmp_path):
    value = report(*campaign(tmp_path))
    assert value['all_jobs_terminal']
    row = value['comparisons'][0]
    assert row['complete'] and row['both_arms_slo_pass']
    assert row['observed_energy_saving'] == 0
    assert not row['improvement_proven']
    assert all(r['artifact_valid'] and r['measurement_valid'] for r in value['windows'])


@pytest.mark.parametrize('fault', ['point', 'metrics', 'raw_hash', 'missing_repeat'])
def test_changed_or_missing_evidence_cannot_enter_mean(tmp_path, fault):
    paths = campaign(tmp_path)
    window = tmp_path/'queue-attempts/shield-0/attempt-1/session/windows/shield-0'
    if fault == 'point':
        value = json.loads((window/'point.json').read_text()); value['revision'] = 'foreign'
        put(window/'point.json', value)
    elif fault == 'metrics':
        value = json.loads((window/'receipt.json').read_text())
        value['result']['metrics']['energy_service_j'] = 1
        put(window/'receipt.json', value)
    elif fault == 'raw_hash':
        (window/'drain.json').write_text('{"passed":true, "changed":true}')
    else:
        (window/'receipt.json').unlink()
    value = report(*paths)
    assert value['comparisons'][0]['observed_energy_saving'] is None
    assert not value['comparisons'][0]['improvement_proven']


def test_zero_total_energy_and_blocked_job_do_not_claim_completion(tmp_path):
    paths = campaign(tmp_path, energy=-10., status='blocked')
    value = report(*paths)
    assert value['comparisons'][0]['observed_energy_saving'] is None
    assert not value['all_jobs_terminal']


def test_slo_claim_requires_replayed_canonical_measurements(tmp_path):
    paths = campaign(tmp_path)
    window = tmp_path/'queue-attempts/shield-0/attempt-1/session/windows/shield-0'
    receipt = json.loads((window/'receipt.json').read_text())
    receipt['result']['acceptance']['checked_gates'].remove('pdblend.canonical_metrics')
    ref = put(window/'result.json', receipt['result'])
    receipt['artifacts']['result.json'] = ref['sha256']
    put(window/'receipt.json', receipt)
    value = report(*paths)
    assert value['comparisons'][0]['complete']
    assert not value['comparisons'][0]['both_arms_slo_pass']


def test_in_progress_partial_receipt_is_not_a_completed_trial(tmp_path):
    paths = campaign(tmp_path)
    window = tmp_path/'queue-attempts/shield-0/attempt-1/session/windows/shield-0'
    (window/'receipt.json').write_text('{"result":')
    value = report(*paths)
    assert len(value['unreadable_windows']) == 1
    assert not value['comparisons'][0]['complete']


def test_file_read_access_time_is_not_content_mutation(tmp_path):
    paths = campaign(tmp_path)
    drain = tmp_path/'queue-attempts/shield-0/attempt-1/session/windows/shield-0/drain.json'
    os.utime(drain, ns=(1, drain.stat().st_mtime_ns))
    value = report(*paths)
    assert all(r['artifact_valid'] for r in value['windows'])


def test_raw_energy_observation_does_not_promote_failed_clock_qualification(tmp_path):
    paths = campaign(tmp_path)
    window = tmp_path/'queue-attempts/shield-0/attempt-1/session/windows/shield-0'
    receipt = json.loads((window/'receipt.json').read_text())
    receipt['result']['acceptance']['measurement_evidence_valid'] = False
    receipt['result']['acceptance']['gate_failures'] = {'pdblend.physical_clocks': 'missing evidence'}
    ref = put(window/'result.json', receipt['result'])
    receipt['artifacts']['result.json'] = ref['sha256']
    put(window/'receipt.json', receipt)
    value = report(*paths)
    comparison = value['comparisons'][0]
    assert comparison['observed_energy_saving'] == 0
    assert not comparison['measurement_qualified'] and not comparison['improvement_proven']
