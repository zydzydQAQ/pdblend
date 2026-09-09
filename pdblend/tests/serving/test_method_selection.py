import asyncio
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from ecopadg.serving import method_selection as search


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def experiment(tmp_path):
    profiles = save(tmp_path/'profiles.json', dict(certification_artifacts={}))
    template = save(tmp_path/'engine.json', {})
    instances = [dict(id=f'i{i}', tp=1, gpus=[i], port=18000+i, kv_port=19000+i, role=role)
                 for i, role in enumerate(('mixed', 'prefill', 'decode'))]
    base = dict(strategy='pdblend-joint', instances=instances, profiles=str(profiles),
                slo_ttft_s=5, slo_tpot_s=.1, role_costs=[{'measured': True}])
    base_path = save(tmp_path/'base.json', base)
    control = deepcopy(base)
    control['strategy'] = 'mixed_dvfs'
    for instance in control['instances']:
        instance['role'] = 'mixed'
    control_path = save(tmp_path/'control.json', control)
    results = []
    for dataset in ('alpaca', 'sharegpt', 'longbench'):
        for system in ('mixed', 'mixed_dvfs', 'distserve', 'ecoserve', 'dynamollm'):
            config = save(tmp_path/'calibrated'/(dataset+'-'+system+'.json'), dict(output_prior=4))
            result = dict(system=system, dataset=dataset, capacity_rps=2., infeasible_upper_rps=3., passed=True,
                          config=str(config),
                          confirmation=dict(rate=2., slo_attainment=1., validity='ok', split='calibration',
                                            completed=128, n_expected=128))
            results.append(result)
    calibrated = save(tmp_path/'calibration.json', dict(passed=True, source_unchanged=True, results=results))
    records = [dict(prompt=[i, 7, 9], input_tokens=3, output_tokens=4,
                    request_shape_sha256=f'dev-{i}') for i in range(4)]
    save(tmp_path/'corpus/alpaca.json', dict(dataset='alpaca', development=records,
        calibration=[dict(records[0], request_shape_sha256='calibration-forbidden')],
        formal={'101': [dict(records[0], request_shape_sha256='formal-forbidden')]}))
    save(tmp_path/'campaign/budget.json', dict(limit_s=86400, started_s=time.time()-60))
    manifest = dict(base_config=str(base_path), control_config=str(control_path), corpus=str(tmp_path/'corpus'),
        calibration=str(calibrated), campaign_root=str(tmp_path/'campaign'), requests=2,
        points=[dict(dataset='alpaca', load='middle', fraction=.6)], seeds=[11],
        method_budget_s=2000, cell_limit_s=100, slo_min=.99, max_slo_drop=.01,
        restore=dict(instances=instances, initial_instances=instances, image='sha256:'+'a'*64,
                     engine_template=str(template), ownership_root=str(tmp_path)))
    path = save(tmp_path/'manifest.json', manifest)
    return path, tmp_path/'search'


def complete(prepared, energies=None, attainment=None):
    energies = energies or {'pdblend-greedy': 95., 'pdblend-joint': 80., 'pdblend-dynamic': 90.}
    for entry in prepared['jobs']:
        job = search.document(entry['path'])
        summary = {k: job[k] for k in ('dataset', 'load', 'seed', 'trace_sha256', 'n_expected', 'expected_generated_tokens')}
        summary.update(split='development', formal_eligible=False, variant=job['variant'], validity='ok',
            completed=job['n_expected'], generated_tokens=job['expected_generated_tokens'], gpu_count=8,
            measurement_schema=2, energy_j=energies.get(job['method'], 100.),
            power_mode='instant',power_field_id=186,power_source_id='nvml:field:186:scope:0:mW',
            power_source_verified=True,
            slo_attainment=(attainment or {}).get(job['method'], 1.))
        path = save(Path(job['out'])/'summary.json', summary)
        save(path.parent/'selection.status.json', dict(complete=True, summary_sha256=search.sha256(path)))


def test_prepare_has_one_paired_development_trace_and_all_candidates_ablations(experiment):
    manifest, out = experiment
    prepared = search.prepare(manifest, out)
    assert len(prepared['jobs']) == 13
    assert set(prepared['configurations']) == set(search.VARIANTS) | {
        v+'.'+a for v in search.VARIANTS for a in search.ABLATIONS} | {'mixed_dvfs'}
    jobs = [search.document(j['path']) for j in prepared['jobs']]
    assert len({j['trace_sha256'] for j in jobs}) == 1
    trace = search.document(jobs[0]['trace'])
    assert trace['split'] == 'development' and trace['seed'] == 11
    assert trace['rate'] == 1.2
    assert all(shape.startswith('dev-') for shape in trace['source_shapes'])
    for job in jobs:
        config = search.document(job['config'])
        assert config['output_prior'] == 4
        assert config['output_priors'] == prepared['output_priors'] == {'alpaca':4}
        assert job['restore']['instances'] == config['instances']
        if job['ablation'] == 'full_frequency':
            assert config['dvfs'] is False and config['park_idle'] is False
        if job['ablation'] == 'mixed_only':
            assert config['allow_pd'] is False and config['dynamic_pools'] is False
            assert {i['role'] for i in config['instances']} == {'mixed'}
        if job['ablation'] == 'fixed_pools':
            assert config['dynamic_pools'] is False and config['slow_topology'] is False
    result = search.document(out/'selection.json')
    assert result['status'] == 'evidence_insufficient' and result['selected_variant'] is None


def test_each_dataset_shares_its_calibrated_prior_across_every_method_and_control(experiment):
    path,out = experiment
    manifest = search.document(path)
    corpus = Path(manifest['corpus'])
    records = search.document(corpus/'alpaca.json')
    records['dataset'] = 'sharegpt'
    records['calibration'][0]['output_tokens'] = 19
    save(corpus/'sharegpt.json',records)
    calibration = search.document(manifest['calibration'])
    for row in calibration['results']:
        if row['dataset'] == 'sharegpt':
            save(Path(row['config']),dict(output_prior=19))
    manifest['points'].append(dict(dataset='sharegpt',load='middle',fraction=.6))
    manifest['method_budget_s'] = 3000
    save(path,manifest)
    prepared = search.prepare(path,out)
    assert prepared['output_priors'] == {'alpaca':4,'sharegpt':19}
    for entry in prepared['jobs']:
        job = search.document(entry['path']); config = search.document(job['config'])
        assert config['output_prior'] == prepared['output_priors'][job['dataset']]
        assert config == search.config_for_point(
            search.document(prepared['configurations'][job['method']]['config']),job['dataset'])
    assert {source['split'] for source in prepared['output_prior_sources'].values()} == {'calibration'}
    assert not search.check_preparation(prepared)
    source = Path(calibration['results'][0]['config']); save(source,dict(output_prior=100))
    assert search.check_preparation(prepared)


def test_prior_cannot_be_reselected_after_independent_baseline_calibration(experiment):
    path,out = experiment; manifest = search.document(path)
    calibrated = search.document(manifest['calibration'])
    save(Path(calibrated['results'][0]['config']),dict(output_prior=3))
    with pytest.raises(ValueError,match='independently calibrated'):
        search.prepare(path,out)


@pytest.mark.parametrize('change', [
    {'seeds': [101]}, {'seeds': [11, 11]}, {'requests': 5}, {'method_budget_s': 100},
    {'method_budget_s': 90000}, {'common_capacity': {'alpaca': 99}},
])
def test_prepare_rejects_formal_leakage_incomplete_pair_budget_or_uncalibrated_rate(experiment, change):
    path, out = experiment
    save(path, dict(search.document(path), **change))
    with pytest.raises(ValueError):
        search.prepare(path, out)


def test_complete_pairs_select_energy_minimum_subject_to_each_slo_constraint(experiment):
    prepared = search.prepare(*experiment)
    complete(prepared, attainment={'pdblend-joint': .97})
    result = search.summarize(Path(prepared['out'])/'prepared.json')
    assert result['selected_variant'] == 'pdblend-dynamic'
    assert result['dataset_equal_energy_ratios']['pdblend-joint'] == .8
    assert result['candidate_feasible']['pdblend-joint'] is False
    assert result['formal_eligible'] is False and result['split'] == 'development'


@pytest.mark.parametrize('mutation', [
    {'generated_tokens': 7}, {'completed': 1}, {'gpu_count': 7}, {'split': 'formal'},
    {'formal_eligible': True}, {'energy_j': 0}, {'trace_sha256': 'wrong'},
    {'power_mode':'average'}, {'power_source_verified':False},
])
def test_invalid_or_unpaired_cell_blocks_selection_even_if_other_candidates_win(experiment, mutation):
    prepared = search.prepare(*experiment)
    complete(prepared)
    job = search.document(prepared['jobs'][0]['path'])
    summary_path = Path(job['out'])/'summary.json'
    save(summary_path, dict(search.document(summary_path), **mutation))
    save(summary_path.parent/'selection.status.json', dict(complete=True, summary_sha256=search.sha256(summary_path)))
    result = search.summarize(Path(prepared['out'])/'prepared.json')
    assert result['selected_variant'] is None
    assert any('invalid or unequal output work' in reason for reason in result['reasons'])


def test_per_point_limits_charge_only_actual_complete_groups_and_preserve_pairs(experiment):
    path,out=experiment
    manifest=search.document(path)
    manifest.update(seeds=[11,22],cell_limit_s=100,method_budget_s=8000,
        points=[dict(dataset='alpaca',load='short',fraction=.3),
                dict(dataset='alpaca',load='long',fraction=.6,cell_limit_s=200)])
    save(path,manifest)
    prepared=search.prepare(path,out)
    assert len(prepared['groups'])==4 and len(prepared['jobs'])==52
    assert prepared['stage_upper_bound_s']==7800
    stages=search.document(prepared['campaign'])['stages']
    assert sum(s['limit_s'] for s in stages)==7800
    for entry,stage in zip(prepared['jobs'],stages):
        job=search.document(entry['path'])
        assert job['cell_limit_s']==stage['limit_s']==(100 if job['load']=='short' else 200)
        assert job['n_expected']==2 and job['seed'] in (11,22)
    assert all(len(group['jobs'])==13 for group in prepared['groups'])


@pytest.mark.parametrize('limit',[0,-1,float('inf'),None])
def test_per_point_limit_must_be_finite_positive(experiment,limit):
    path,out=experiment
    manifest=search.document(path);manifest['points'][0]['cell_limit_s']=limit
    # Write directly to exercise the input validator even with nonfinite JSON.
    path.write_text(__import__('json').dumps(manifest))
    with pytest.raises(ValueError,match='finite positive'):
        search.prepare(path,out)


def test_over_budget_complete_group_is_rejected_without_dropping_requests_or_methods(experiment):
    path,out=experiment
    manifest=search.document(path);manifest['points'][0]['cell_limit_s']=200
    save(path,manifest)
    with pytest.raises(ValueError,match='complete paired method group'):
        search.prepare(path,out)
    assert not out.exists()


def test_missing_or_changed_evidence_is_not_replaced_with_a_good_subset(experiment):
    prepared = search.prepare(*experiment)
    complete(prepared)
    assert search.summarize(Path(prepared['out'])/'prepared.json')['selected_variant'] == 'pdblend-joint'
    job = search.document(prepared['jobs'][0]['path'])
    (Path(job['out'])/'selection.status.json').unlink()
    result = search.summarize(Path(prepared['out'])/'prepared.json')
    assert result['selected_variant'] is None
    complete(prepared)
    config = search.document(job['config'])
    save(Path(config['profiles']), dict(changed=True))
    result = search.summarize(Path(prepared['out'])/'prepared.json')
    assert result['selected_variant'] is None
    assert any('source, profile or input changed' in reason for reason in result['reasons'])


def test_cell_restores_layout_before_running_shared_measurement(experiment, monkeypatch):
    from ecopadg.serving import calibration_setup, cell
    prepared = search.prepare(*experiment)
    job_path = prepared['jobs'][0]['path']
    job = search.document(job_path)
    calls = []

    async def restore(layout, out):
        calls.append('restore')
        assert layout == job['restore']
        assert str(out).endswith('.preparation')
        return {'changed': False}

    async def run(options):
        calls.append('cell')
        assert calls == ['restore', 'cell']
        assert options.split == 'development' and options.freeze is None
        complete(prepared)
        return search.document(options.out/'summary.json')

    monkeypatch.setattr(calibration_setup, 'restore_layout', restore)
    monkeypatch.setattr(cell, 'run_cell', run)
    asyncio.run(search.run_job(job_path))
    assert calls == ['restore', 'cell']
    assert search.document(Path(job['out'])/'selection.status.json')['complete'] is True


def test_candidate_that_meets_absolute_slo_but_loses_to_control_is_excluded(experiment):
    path, out = experiment
    save(path, dict(search.document(path), slo_min=.9))
    prepared = search.prepare(path, out)
    complete(prepared, attainment={'pdblend-joint': .96})
    result = search.summarize(out/'prepared.json')
    assert result['selected_variant'] == 'pdblend-dynamic'
    assert result['candidate_feasible']['pdblend-joint'] is False


def test_restore_failure_cannot_start_measurement_or_leave_success_status(experiment, monkeypatch):
    from ecopadg.serving import calibration_setup, cell
    prepared = search.prepare(*experiment)
    job_path = prepared['jobs'][0]['path']
    job = search.document(job_path)
    measured = []

    async def restore(layout, out):
        raise RuntimeError('old decode request still running')

    async def measure(options):
        measured.append(True)

    monkeypatch.setattr(calibration_setup, 'restore_layout', restore)
    monkeypatch.setattr(cell, 'run_cell', measure)
    with pytest.raises(RuntimeError, match='still running'):
        asyncio.run(search.run_job(job_path))
    assert not measured
    assert search.document(Path(job['out'])/'selection.status.json')['complete'] is False
    assert search.document(Path(prepared['out'])/'selection.json')['selected_variant'] is None


def test_changed_summary_is_rejected_even_if_it_claims_better_energy(experiment):
    prepared = search.prepare(*experiment)
    complete(prepared)
    job = search.document(prepared['jobs'][0]['path'])
    path = Path(job['out'])/'summary.json'
    save(path, dict(search.document(path), energy_j=1.))
    result = search.summarize(Path(prepared['out'])/'prepared.json')
    assert result['selected_variant'] is None
    assert any('failed or changed cell' in reason for reason in result['reasons'])
