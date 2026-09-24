import hashlib
import json

import pytest

from pdblend.bench.optimization_acceptance import BASELINES, DATASETS, MODELS, SCHEMA, evaluate_stage


def campaign(tmp_path, mutate_run=None, mutate_manifest=None):
    def write(name, value):
        payload = json.dumps(value, sort_keys=True).encode()
        (tmp_path / name).write_bytes(payload)
        return dict(path=name, sha256=hashlib.sha256(payload).hexdigest())

    manifest = dict(schema=SCHEMA, stage='stage2-candidate-ranking', pairs=[], traces=[], independent_baselines=[])
    stamp = 1000
    for model in sorted(MODELS):
        tp = 2 if '32B' in model else 1
        profile = write(f'{model}-pdblend-profile.json', dict(system='pdblend', model_id=model, tp=tp, pp=1, accepted=True))
        qualification = write(f'{model}-qualification.json', dict(system='pdblend', model_id=model, tp=tp, pp=1,
                               hardware_qualified=True, functional_passed=True, native_cleanup_complete=True))
        for system in sorted(BASELINES):
            manifest['independent_baselines'].append(write(f'{model}-{system}-profile.json',
                dict(system=system, model_id=model, tp=tp, pp=1, accepted=True)))
        for dataset in sorted(DATASETS):
            trace = write(f'{model}-{dataset}-trace.json', dict(seed=701, model_id=model, dataset=dataset,
                          requests=[dict(idx=i, arrival_s=i/10, prompt=[1], max_tokens=20) for i in range(100)]))
            manifest['traces'].append(trace)
            for repeat in range(3):
                pair = dict(repeat=repeat)
                for arm in ['control', 'candidate']:
                    conditions = dict(model_id=model, dataset=dataset, tp=tp, pp=1, seed=701,
                                      trace_sha256=trace['sha256'], rate_rps=10,
                                      slo=dict(ttft_s=1, tpot_s=.1), engine_sha256='a'*64, weights_sha256='b'*64)
                    energy = 1000 if arm == 'control' else 800
                    summary = dict(run_id=f'{model}-{dataset}-{repeat}-{arm}',
                        energy_j=energy, window_s=10, goodput_request_s=9.5, goodput_token_s=100,
                        j_per_goodput_token=energy/1000,
                        slo=dict(offered=100, joint_slo_rate=.95, joint_output_tokens=1000, joint_slo_requests=95),
                        metering=dict(source=dict(mode='instant', source_id='nvml:field:186:scope:0:mW',
                            field_id=186, scope_id=0, unit='W'), error=None), quarantined_instances=[])
                    execution = dict(system='pdblend', stage=manifest['stage'], arm=arm, conditions=conditions,
                        exclusive=True, lease_verified=True, gpu_uuids=[f'GPU-{i}' for i in range(8)],
                        start_s=stamp, end_s=stamp+10)
                    stamp += 20
                    uncertainty = dict(absolute_energy_j=10, method='synthetic instrument bound for contract test only')
                    if mutate_run:
                        mutate_run(arm, model, dataset, repeat, conditions, summary, execution, uncertainty)
                    stem = summary['run_id']
                    summary_ref = write(stem + '-summary.json', summary)
                    execution.update(summary_sha256=summary_ref['sha256'], profile_sha256=profile['sha256'])
                    uncertainty['summary_sha256'] = summary_ref['sha256']
                    pair[arm] = dict(conditions=conditions, summary=summary_ref, profile=profile,
                                     qualification=qualification, execution=write(stem+'-execution.json', execution),
                                     uncertainty=write(stem+'-uncertainty.json', uncertainty))
                manifest['pairs'].append(pair)
    if mutate_manifest:
        mutate_manifest(manifest)
    path = tmp_path / 'stage.json'
    path.write_text(json.dumps(manifest))
    return path


def test_complete_controlled_stage_accepts_only_named_stage(tmp_path):
    result = evaluate_stage(campaign(tmp_path))
    assert result['accepted'], result
    assert result['claim_scope'] == 'stage2-candidate-ranking'
    assert len(result['cells']) == 9 and not result['formal_eligible']
    assert all(row['energy_improvement'] > row['combined_uncertainty'] for row in result['cells'])


@pytest.mark.parametrize('fault', ['seed', 'trace', 'rate', 'slo', 'tp', 'lease', 'gpu_count',
                                   'joint_slo', 'missing_metrics', 'uncertainty', 'metering', 'overlap'])
def test_incomparable_or_unqualified_runs_fail_closed(tmp_path, fault):
    def mutate(arm, model, dataset, repeat, conditions, summary, execution, uncertainty):
        if arm != 'candidate':
            return
        if fault == 'seed': conditions['seed'] = 702
        if fault == 'trace': conditions['trace_sha256'] = 'c'*64
        if fault == 'rate': conditions['rate_rps'] = 11
        if fault == 'slo': conditions['slo']['tpot_s'] = .2
        if fault == 'tp': conditions['tp'] = 4
        if fault == 'lease': execution['exclusive'] = False
        if fault == 'gpu_count': execution['gpu_uuids'].pop()
        if fault == 'joint_slo': summary['slo']['joint_slo_rate'] = .899
        if fault == 'missing_metrics': summary.pop('j_per_goodput_token')
        if fault == 'uncertainty': uncertainty.pop('absolute_energy_j')
        if fault == 'metering': summary['metering']['source']['mode'] = 'legacy'
        if fault == 'overlap': execution['start_s'], execution['end_s'] = 1000, 1010
    result = evaluate_stage(campaign(tmp_path, mutate))
    assert not result['accepted'] and result['errors']


@pytest.mark.parametrize('fault', ['incomplete_grid', 'two_repeats', 'duplicate_repeat', 'no_baseline', 'no_trace'])
def test_grid_repeats_independent_baselines_and_bound_trace_required(tmp_path, fault):
    def mutate(manifest):
        if fault == 'incomplete_grid': manifest['pairs'] = manifest['pairs'][:3]
        if fault == 'two_repeats': manifest['pairs'].pop()
        if fault == 'duplicate_repeat': manifest['pairs'][1]['repeat'] = 0
        if fault == 'no_baseline': manifest['independent_baselines'].pop()
        if fault == 'no_trace': manifest['traces'] = []
    result = evaluate_stage(campaign(tmp_path, mutate_manifest=mutate))
    assert not result['accepted'] and result['errors']


@pytest.mark.parametrize('fault', ['goodput_regression', 'inside_uncertainty'])
def test_energy_requires_goodput_preservation_and_resolved_measurement_improvement(tmp_path, fault):
    def mutate(arm, model, dataset, repeat, conditions, summary, execution, uncertainty):
        if arm != 'candidate':
            return
        if fault == 'goodput_regression':
            summary['slo']['joint_output_tokens'] = 999
            summary['goodput_token_s'] = 99.9
            summary['j_per_goodput_token'] = summary['energy_j']/999
        else:
            summary['energy_j'], summary['j_per_goodput_token'] = 990, .99
    result = evaluate_stage(campaign(tmp_path, mutate))
    assert not result['accepted'] and not result['errors']
    assert not any(c['accepted'] for c in result['cells'])


def test_cpu_functional_receipt_cannot_become_hardware_energy_claim(tmp_path):
    path = campaign(tmp_path)
    manifest = json.loads(path.read_text())
    ref = manifest['pairs'][0]['candidate']['qualification']
    qualification = json.loads((tmp_path/ref['path']).read_text())
    qualification['hardware_qualified'] = False
    payload = json.dumps(qualification).encode()
    (tmp_path/ref['path']).write_bytes(payload)
    # Keep the component binding valid, so the rejection tests eligibility.
    ref['sha256'] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(manifest))
    result = evaluate_stage(path)
    assert not result['accepted']
