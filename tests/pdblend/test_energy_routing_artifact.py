"""Artifact contracts use synthetic observations, never hardware qualification."""
import hashlib
import json

import pytest

from pdblend.online.energy_routing import SCHEMA, load_energy_estimator, route_key
from pdblend.planner.pool import SLO
from test_resident_energy import pools


def artifact(tmp_path, router, corrupt=None):
    datasets = {}
    for partition, time_offset in [('training', 1000), ('holdout', 2000)]:
        rows = []
        for iid in router.pools:
            choice = ('M', iid, iid)
            context = router._route_context(choice, 100, 10)
            key = route_key(router, choice, context)
            energy = 100 if iid == 'fast' else 10
            for repeat in range(3):
                start = time_offset + 10 * repeat
                rows.append(dict(sample_id=f'{partition}-{iid}-{repeat}',
                    request_id=f'req-{partition}-{iid}-{repeat}', key=key, measurement='paired_common_window',
                    power_source='nvml_total_energy_counter', gpu_uuids=[f'GPU-{iid}-{n}' for n in range(key['tp'])],
                    completion_tokens=10, error=None, ttft_s=.1, tpot_s=.01,
                    baseline=dict(start_s=start, end_s=start+1, energy_j=100),
                    with_request=dict(start_s=start+2, end_s=start+3, energy_j=100+energy)))
        datasets[partition] = rows
    if corrupt:
        corrupt(datasets)
    manifest = dict(schema=SCHEMA, system='pdblend')
    for partition, rows in datasets.items():
        payload = json.dumps(rows).encode()
        name = partition + '.json'
        (tmp_path / name).write_bytes(payload)
        manifest[partition] = dict(path=name, sha256=hashlib.sha256(payload).hexdigest())
    path = tmp_path / 'energy.json'
    path.write_text(json.dumps(manifest))
    return path


def test_measured_energy_is_recomputed_and_independent_holdouts_bind_exact_route(tmp_path):
    router = pools()
    estimator = load_energy_estimator(artifact(tmp_path, router), router=router)
    router.configure_energy_routing(slo=SLO(1, .1), estimator=estimator)
    record = router.dispatch('one', 100, 10)
    assert record.decode_instance == 'efficient'
    energy = record.route_estimate['energy']
    assert energy['incremental_energy_j'] == 10
    assert energy['holdout']['windows'] == 3
    assert not energy['evidence']['hardware_qualified']
    # Changed queue/backlog has no measured incremental-energy point. Both
    # candidates are compared in latency units rather than mixing J and s.
    assert router.dispatch('two', 100, 10).decode_instance == 'fast'


@pytest.mark.parametrize('corrupt', [
    lambda d: d['holdout'][0].update(sample_id=d['training'][0]['sample_id']),
    lambda d: d['holdout'][0].update(request_id=d['training'][0]['request_id']),
    lambda d: d['holdout'][0]['with_request'].update(energy_j=300),
    lambda d: d['holdout'].pop(),
    lambda d: d['training'][0].update(power_source='affine_decode_power'),
    lambda d: d['training'][0].update(completion_tokens=9),
    lambda d: d['training'][0].update(gpu_uuids=['GPU-incomplete']),
    lambda d: d['holdout'][0]['with_request'].update(start_s=1002, end_s=1003),
    lambda d: d['holdout'][0]['baseline'].update(end_s=2005),
    lambda d: d['training'][0]['key'].pop('reservation_tokens'),
])
def test_unqualified_affine_incomplete_or_reused_samples_are_rejected(tmp_path, corrupt):
    router = pools()
    with pytest.raises(ValueError):
        load_energy_estimator(artifact(tmp_path, router, corrupt), router=router)


def test_manifest_digest_prevents_silent_replacement_of_component(tmp_path):
    router = pools()
    path = artifact(tmp_path, router)
    (tmp_path / 'holdout.json').write_text('[]')
    with pytest.raises(ValueError, match='digest'):
        load_energy_estimator(path, router=router)


def test_affine_profile_or_qualified_boolean_cannot_create_incremental_measurements(tmp_path):
    path = tmp_path / 'ordinary-power.json'
    path.write_text(json.dumps(dict(qualified=True, prefill_power={'1500': [100, 0]},
                                    decode_power={'1500': [100, 2]})))
    with pytest.raises(ValueError, match='versioned'):
        load_energy_estimator(path, router=pools())


def test_ranking_reversal_is_rejected_even_inside_numeric_error_tolerance(tmp_path):
    def reverse(datasets):
        for partition, rows in datasets.items():
            for row in rows:
                fast = row['key']['profile_keys'] == ['fast']
                value = 100 if (partition == 'training') == fast else 101
                row['with_request']['energy_j'] = row['baseline']['energy_j'] + value
    router = pools()
    with pytest.raises(ValueError, match='reverses'):
        load_energy_estimator(artifact(tmp_path, router, reverse), router=router)


def test_measured_holdout_slo_violation_cannot_be_hidden_by_optimistic_latency_model(tmp_path):
    def slow(datasets):
        for row in datasets['holdout']:
            if row['key']['profile_keys'] == ['efficient']:
                row['tpot_s'] = .5
    router = pools()
    estimator = load_energy_estimator(artifact(tmp_path, router, slow), router=router)
    router.configure_energy_routing(slo=SLO(1, .1), estimator=estimator)
    assert router.dispatch('r', 100, 10).decode_instance == 'fast'
