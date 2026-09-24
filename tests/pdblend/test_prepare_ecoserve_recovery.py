"""CPU-only recovery planning with actual source/artifact hash verification."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from pdblend.bench.comparison_campaign import binding
from pdblend.bench.resident_session import digest, file_sha, write_new


ROOT = Path(__file__).parents[2]


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    module = load_script(ROOT/'scripts/2026-09-24_prepare_ecoserve_recovery.py', 'eco_recovery_test')
    project = tmp_path/'project'
    (project/'scripts').mkdir(parents=True)
    freezer_name = '2026-09-22_enqueue_parallel_profiles.py'
    shutil.copyfile(ROOT/'scripts'/freezer_name, project/'scripts'/freezer_name)
    for name, text in {
        'pdblend_runtime/serve.py': 'public runtime = unchanged\n',
        'pdblend/measure/power.py': 'public sampler = unchanged\n',
        'pdblend/bench/comparison_metering.py': 'public integrator = unchanged\n',
        'pdblend/bench/comparison_runtime.py': 'original wrapper\n',
    }.items():
        target = project/'src'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    freezer = load_script(project/'scripts'/freezer_name, 'eco_freezer_test')
    source, source_sha = freezer.freeze_source(project/'src', project/'prior-sources')
    files = json.loads((source/'manifest.json').read_text())['files']
    runtime = {k: v for k, v in files.items() if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))}
    measurement = {k: v for k, v in files.items() if k.startswith('pdblend/measure/') or k in (
        'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py', 'pdblend/bench/client.py')}
    model = 'Qwen2.5-7B-Instruct'
    identity = dict(model_hash='weights', tokenizer_hash='tokenizer', image_digest='image',
        runtime_source_sha256=digest(runtime), measurement_source_sha256=digest(measurement),
        entrypoint='native', worker_extension=None, dtype='bfloat16', environment={},
        fleet_gpu_uuids=[f'GPU-{i}' for i in range(8)],
        instances=[dict(instance_id=f'mixed{i}', tp=1, pp=1, gpu_uuids=[f'GPU-{i}'], launch_options={}) for i in range(8)])
    write_new(project/'trace.json', dict(requests=[]))
    def point(name, dataset, scale, system='ecoserve'):
        return dict(name=name, model_id=model, dataset=dataset, scale=scale, system=system,
            trace=binding(project/'trace.json'), slo=dict(ttft_s=1., tpot_s=.1),
            engine_identity=deepcopy(identity), source_manifest=binding(source/'manifest.json'),
            revision=source_sha, status='prepared', blockers=[], original_field='preserve me')
    points = [point('mixed', 'alpaca', .5, 'mixed'),
        point('frozen-pass', 'alpaca', .5), point('frozen-slo-failure', 'sharegpt', .5),
        point('invalid-old', 'longbench', .5), point('never-executed', 'alpaca', .25),
        point('no-trace', 'sharegpt', .25)]
    points[-1].update(trace=None, blockers=['trace missing'], status='blocked')
    prior = project/'prior-session'
    for name, valid, slo in [('frozen-pass', True, True), ('frozen-slo-failure', True, False), ('invalid-old', False, False)]:
        item = next(p for p in points if p['name'] == name)
        window = prior/'windows'/name
        result = dict(evidence_valid=valid, formal_eligible=valid, slo_pass=slo, energy_service_j=123.)
        write_new(window/'point.json', item)
        write_new(window/'result.json', result)
        write_new(window/'receipt.json', dict(point_sha256=digest(item), result=result,
            evidence_valid=valid, baseline_frozen=valid, cleanup_passed=True,
            artifacts={key: file_sha(window/key) for key in ('point.json', 'result.json')}))
    base = project/'base'/'campaign.json'
    write_new(base, dict(campaign_id='prior', points=points, execution_campaigns=[]))
    write_new(base.parent/'execution-inputs.json', dict(image_digest='image', model_verification=dict(path=str(project/'verified.json'))))
    artifact = binding(project/'trace.json')
    evidence = {key: artifact for key in ('profile_csv', 'profile_manifest', 'mechanism_completion',
                                        'automatic_completion', 'independent_mechanism_review')}
    readiness = project/'readiness.json'
    write_new(readiness, dict(models={model: dict(ecoserve=evidence)},
        ecosystem_source_continuity={key: dict(source_manifest=binding(source/'manifest.json')) for key in ('profile', 'mechanism')}))
    checked = []
    def validate(point, engine, *, source_manifest):
        checked.append(deepcopy(point))
        assert source_manifest == point['source_manifest'] == point['inputs']['source_manifest']
        assert engine['metering_execution'] == point['metering_execution'] == 'isolated_process'
        return dict(preflight_ready=True, missing_gates=[])
    monkeypatch.setattr(module, 'ROOT', project)
    monkeypatch.setattr(module, 'validate_ecoserve_inputs', validate)
    out = project/'recovery'
    argv = ['prepare', '--base', str(base), '--out', str(out), '--source-base', str(source),
        '--previous', str(prior), '--readiness', str(readiness), '--isolated-meter',
        '--overlay', 'pdblend/bench/comparison_runtime.py']
    (project/'src/pdblend/bench/comparison_runtime.py').write_text('new isolated wrapper\n')
    monkeypatch.setattr(sys, 'argv', argv)
    return dict(module=module, root=project, source=source, out=out, prior=prior,
                points=points, checked=checked, argv=argv, base=base)


def test_frozen_pass_and_slo_failure_preserved_while_only_unfinished_points_are_prepared(recovery):
    x = recovery
    before = {str(p): file_sha(p) for p in x['prior'].rglob('*') if p.is_file()}
    old_source = {str(p): file_sha(p) for p in x['source'].rglob('*') if p.is_file()}
    x['module'].main()
    campaign = json.loads((x['out']/'campaign.json').read_text())
    preserved = campaign['preserved_baseline_receipts']
    assert set(preserved) == {'frozen-pass', 'frozen-slo-failure'}
    assert before == {str(p): file_sha(p) for p in x['prior'].rglob('*') if p.is_file()}
    assert old_source == {str(p): file_sha(p) for p in x['source'].rglob('*') if p.is_file()}
    by_name = {p['name']: p for p in campaign['points']}
    for name in ('mixed', 'frozen-pass', 'frozen-slo-failure', 'no-trace'):
        assert by_name[name] == next(p for p in x['points'] if p['name'] == name)
    assert {p['name'] for p in x['checked']} == {'invalid-old', 'never-executed'}
    jobs = json.loads((x['out']/'jobs.json').read_text())
    assert len(jobs) == 1 and campaign['summary']['new_ecoserve_points'] == 2
    groups = [json.loads(p.read_text()) for p in (x['out']/'groups').glob('*.json')]
    assert {p['name'] for group in groups for p in group['points']} == {'invalid-old', 'never-executed'}
    assert jobs[0]['payload']['prior_sessions'] == [str(x['prior'].resolve())]
    assert jobs[0]['payload']['argv'][-2:] == ['--previous', str(x['prior'].resolve())]
    # The source manifest variable must not overwrite the frozen receipt map.
    extension = json.loads((x['out']/'source-extension.json').read_text())
    assert extension['base_manifest'] == binding(x['source']/'manifest.json')
    assert preserved['frozen-pass'] == binding(x['prior']/'windows/frozen-pass/receipt.json')
    new_source = json.loads((x['out']/'execution-inputs.json').read_text())['source']
    assert Path(new_source, 'pdblend/bench/comparison_runtime.py').read_text() == 'new isolated wrapper\n'


@pytest.mark.parametrize('artifact', ['point.json', 'result.json'])
def test_changed_frozen_artifact_rejected_before_new_source_or_job_creation(recovery, artifact):
    x = recovery
    path = x['prior']/'windows/frozen-pass'/artifact
    path.write_text('{}\n')
    with pytest.raises(ValueError, match='bytes changed'):x['module'].main()
    assert not (x['out']/'sources').exists() and not (x['out']/'jobs.json').exists()


def test_bound_result_must_equal_receipt_embedded_result(recovery):
    x = recovery
    window = x['prior']/'windows/frozen-pass'
    path = window/'result.json'
    result = json.loads(path.read_text()); result['energy_service_j'] = 999.
    path.write_text(json.dumps(result))
    receipt = json.loads((window/'receipt.json').read_text())
    receipt['artifacts']['result.json'] = file_sha(path)
    (window/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='point/result identity'):x['module'].main()


def test_changed_frozen_source_rejected_even_with_valid_prior_window(recovery):
    x = recovery
    (x['source']/'pdblend/measure/power.py').write_text('tampered public sampler\n')
    with pytest.raises(ValueError, match='checksum mismatch'):x['module'].main()


def test_parent_cannot_change_a_frozen_point_or_duplicate_its_observation(recovery):
    x = recovery
    base = json.loads(x['base'].read_text())
    next(p for p in base['points'] if p['name'] == 'frozen-pass')['revision'] = 'different'
    x['base'].write_text(json.dumps(base))
    with pytest.raises(ValueError, match='unique frozen'):x['module'].main()


def test_public_meter_overlay_cannot_change_frozen_comparison_identity(recovery):
    x = recovery
    x['argv'] += ['--overlay', 'pdblend/measure/power.py']
    (x['root']/'src/pdblend/measure/power.py').write_text('changed public sampler\n')
    with pytest.raises(ValueError, match='public engine/measurement'):x['module'].main()
    assert not (x['out']/'jobs.json').exists()


def test_lifecycle_review_applies_only_to_new_points_and_binds_all_three_owners(recovery,monkeypatch):
    x=recovery;review=x['root']/'review.json'
    write_new(review,dict(mode=x['module'].LIFECYCLE_MODE,hardware_qualified=False))
    monkeypatch.setattr(x['module'],'LIFECYCLE_REVIEW_SHA256',file_sha(review))
    x['argv']+=['--lifecycle-review',str(review)]
    x['module'].main()
    campaign=json.loads((x['out']/'campaign.json').read_text())
    for point in campaign['points']:
        if point['name'] not in ('invalid-old','never-executed'):
            assert 'eco_comparison_lifecycle' not in point
            continue
        config=json.loads(Path(point['inputs']['system_config']['path']).read_text())
        assert point['eco_comparison_lifecycle']==point['engine_identity']['eco_comparison_lifecycle']==config['eco_comparison_lifecycle']==x['module'].LIFECYCLE_MODE
        assert point['inputs']['eco_lifecycle_review']==binding(review)


def test_lifecycle_rejects_unreviewed_bytes_before_creating_output(recovery):
    x=recovery;review=x['root']/'unreviewed.json';review.write_text('{}')
    x['argv']+=['--lifecycle-review',str(review)]
    with pytest.raises(ValueError,match='unreviewed'):x['module'].main()
    assert not x['out'].exists()


def terminal_inventory(x):
    from pdblend.bench.comparison_campaign import group_points
    points=[p for p in x['points'] if p['system']=='ecoserve' and p['trace']]
    group=group_points(points)[0];group_path=x['root']/'prior-group.json';write_new(group_path,group)
    job_id='comparison-7b-'+digest(group)[:16]
    payload=dict(argv=['python','--group',str(group_path)],container_name=job_id,session_id=group['session_id'])
    manifest=dict(immutable=True,job_id=job_id,lease_id='previous-lease',payload=payload)
    write_new(x['root']/'manifest.json',manifest)
    write_new(x['root']/'execution.json',dict(status='failed',finished_s=3.))
    windows=[dict(point=p.parent.name,**binding(p)) for p in sorted(x['prior'].glob('windows/*/receipt.json'))]
    report=dict(schema='resident-group-session/v1',status='failed',finished_s=2.,cleanup_errors=[],
        cleanup=dict(passed=True,process_cleanup_verified=True),session_id=group['session_id'],
        group_sha256=digest(group),planned_points={p['name']:digest(p) for p in group['points']},windows=windows,skipped=[])
    write_new(x['prior']/'completion.json',report)
    queue=dict(jobs={job_id:dict(job_id=job_id,status='failed',lease_id=None,payload=payload)},
        leases={'previous-lease':dict(job_id=job_id,status='failed',attempt_dir=str(x['root']))})
    path=x['root']/'queue.json';write_new(path,queue)
    return path,report


def test_unattempted_only_preserves_invalid_and_frozen_points_without_repeating_them(recovery):
    x=recovery;queue,_=terminal_inventory(x)
    x['argv']+=['--unattempted-from',str(x['prior']),'--queue',str(queue)]
    before={str(p):file_sha(p) for p in x['prior'].rglob('*') if p.is_file()}
    x['module'].main()
    campaign=json.loads((x['out']/'campaign.json').read_text())
    selected=json.loads((x['out']/'unattempted-selection.json').read_text())
    assert selected['points']=={'never-executed':digest(next(p for p in x['points'] if p['name']=='never-executed'))}
    assert not selected['invalid_observations_retried'] and not selected['frozen_baselines_retried']
    assert {p['name'] for g in campaign['groups'] for p in g['points']}=={'never-executed'}
    assert len(json.loads((x['out']/'jobs.json').read_text()))==1
    assert {p['name'] for p in x['checked']}=={'never-executed'}
    assert next(p for p in campaign['points'] if p['name']=='invalid-old')==next(p for p in x['points'] if p['name']=='invalid-old')
    assert before=={str(p):file_sha(p) for p in x['prior'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage',['running','held_lease','cleanup','omitted_window','raw_binding',
                                  'changed_parent','changed_group','skipped_fabrication','duplicate_session',
                                  'joint_group_rebind','wrong_container','wrong_session'])
def test_unattempted_selection_rejects_incomplete_or_unbound_history(recovery,damage):
    x=recovery;queue,report=terminal_inventory(x);base=json.loads(x['base'].read_text());sessions=[x['prior']]
    state=json.loads(queue.read_text())
    job=next(iter(state['jobs'].values()))
    if damage=='running':job['status']='running'
    elif damage=='held_lease':job['lease_id']='previous-lease'
    elif damage=='cleanup':report['cleanup']['process_cleanup_verified']=False
    elif damage=='omitted_window':report['windows'].pop()
    elif damage=='raw_binding':report['windows'][0]['sha256']='0'*64
    elif damage=='changed_parent':next(p for p in base['points'] if p['name']=='never-executed')['revision']='changed'
    elif damage=='changed_group':report['group_sha256']='0'*64
    elif damage=='skipped_fabrication':report['skipped']=[dict(point='never-executed',frozen_receipt=binding(x['prior']/'windows/invalid-old/receipt.json'))]
    elif damage=='duplicate_session':sessions*=2
    elif damage=='joint_group_rebind':
        path=x['root']/'prior-group.json';group=json.loads(path.read_text());group['forged_metadata']='different'
        path.write_text(json.dumps(group));report['group_sha256']=digest(group)
    elif damage in ('wrong_container','wrong_session'):
        manifest_path=x['root']/'manifest.json';manifest=json.loads(manifest_path.read_text())
        key='container_name' if damage=='wrong_container' else 'session_id'
        manifest['payload'][key]=job['payload'][key]='wrong'
        manifest_path.write_text(json.dumps(manifest))
    queue.write_text(json.dumps(state));(x['prior']/'completion.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):x['module'].unattempted_selection(base,sessions,queue)


@pytest.mark.parametrize('reverse',[False,True])
def test_an_observation_in_any_selected_session_excludes_the_point_globally(recovery,reverse):
    x=recovery;queue,_=terminal_inventory(x)
    point=deepcopy(next(p for p in x['points'] if p['name']=='never-executed'))
    second_root=x['root']/'other-attempt';second_root.mkdir()
    second=dict(root=second_root,prior=second_root/'session',points=[point])
    window=second['prior']/'windows'/point['name'];write_new(window/'point.json',point)
    write_new(window/'receipt.json',dict(point_sha256=digest(point),evidence_valid=False,
                                       baseline_frozen=False,cleanup_passed=True))
    second_queue,_=terminal_inventory(second)
    manifest_path=second_root/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['lease_id']='second-lease';manifest_path.write_text(json.dumps(manifest))
    state=json.loads(queue.read_text());other=json.loads(second_queue.read_text())
    state['jobs'].update(other['jobs']);state['leases']['second-lease']=other['leases']['previous-lease']
    queue.write_text(json.dumps(state))
    sessions=[x['prior'],second['prior']]
    if reverse:sessions.reverse()
    result=x['module'].unattempted_selection(json.loads(x['base'].read_text()),sessions,queue)
    assert result['points']=={} and not result['invalid_observations_retried']
