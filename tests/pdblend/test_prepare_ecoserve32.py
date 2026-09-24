"""Recovery plans preserve the first valid observation, including SLO failures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from pdblend.bench.comparison_campaign import binding, group_points
from pdblend.bench.resident_session import ResidentGroupSession, digest, engine_signature, file_sha, write_new


spec = importlib.util.spec_from_file_location('prepare_ecoserve32',
    Path(__file__).resolve().parents[2]/'scripts/2026-09-24_prepare_ecoserve32_comparison.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def campaign(*, longbench=False, prepared=False):
    identity = dict(model_hash='model', tokenizer_hash='tokenizer', image_digest='image',
        runtime_source_sha256='engine', entrypoint='serve', worker_extension='worker', dtype='bfloat16',
        environment={}, instances=[dict(instance_id=f'eco{i}', tp=2, pp=1,
            gpu_uuids=[f'GPU-{i*2}', f'GPU-{i*2+1}'], launch_options={}) for i in range(4)])
    points = []
    for dataset in ('alpaca', 'sharegpt', 'longbench'):
        for scale in (.5, .25, .75, 1.):
            has_trace = longbench or dataset != 'longbench'
            point = dict(name=f'32b-ecoserve-{dataset}-x{scale:g}-seed701', system='ecoserve',
                model_id='Qwen2.5-32B-Instruct', dataset=dataset, scale=scale,
                trace={'path':'trace', 'sha256':'trace'} if has_trace else None,
                status='prepared' if prepared and has_trace else 'blocked', blockers=[] if has_trace else ['anchor'])
            if prepared and has_trace:
                point['engine_identity'] = deepcopy(identity)
            points.append(point)
    return dict(points=points)


def write_session(root, points, *, slo_pass=False, evidence_valid=True):
    signature = engine_signature(points[0]['engine_identity'])
    windows = []
    for point in points:
        window = root/'windows'/point['name']
        result = dict(evidence_valid=evidence_valid, formal_eligible=evidence_valid, slo_pass=slo_pass,
                      metrics=dict(energy_service_j=1500.))
        write_new(window/'point.json', point)
        write_new(window/'result.json', result)
        receipt = dict(point=point['name'], point_sha256=digest(point), engine_signature=signature,
            evidence_valid=evidence_valid, baseline_frozen=evidence_valid, cleanup_passed=True, result=result,
            artifacts={name:file_sha(window/name) for name in ('point.json','result.json')})
        write_new(window/'receipt.json', receipt)
        ref = binding(window/'receipt.json')
        windows.append(dict(point=point['name'], **ref, evidence_valid=evidence_valid))
    write_new(root/'completion.json', dict(status='passed' if evidence_valid else 'failed',
        finished_s=42., complete=evidence_valid, engine_signature=signature, windows=windows, skipped=[]))
    return root


def overwrite(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('longbench,count', [(False,8),(True,12)])
def test_plan_eight_or_twelve_before_any_source_is_written(longbench, count):
    base = campaign(longbench=longbench)
    assert prepare.frozen_points(base, []) == {}
    assert len(prepare.planned_points(base['points'], ['alpaca','sharegpt','longbench'], {})) == count


def test_partial_or_duplicate_scale_dataset_is_rejected():
    base = campaign(longbench=True)
    base['points'][-1]['trace'] = None
    with pytest.raises(ValueError, match='four-scale'):
        prepare.planned_points(base['points'], ['alpaca','sharegpt','longbench'], {})
    base = campaign()
    base['points'][1]['scale'] = .5
    with pytest.raises(ValueError, match='four-scale'):
        prepare.planned_points(base['points'], ['alpaca','sharegpt'], {})


@pytest.mark.parametrize('marker', ['engine_identity','qualification_mode','metering_execution','status'])
def test_existing_preparation_requires_previous(marker):
    base = campaign()
    base['points'][0][marker] = 'prepared' if marker == 'status' else 'already-bound'
    with pytest.raises(ValueError, match='--previous is required'):
        prepare.frozen_points(base, [])


@pytest.mark.parametrize('slo_pass', [True,False])
def test_frozen_point_is_unchanged_and_consumed_by_runtime_skip(tmp_path, slo_pass):
    base = campaign(prepared=True); original = deepcopy(base)
    previous = write_session(tmp_path/'previous', base['points'][:1], slo_pass=slo_pass)
    preserved = prepare.frozen_points(base, [previous])
    assert base == original
    pending = prepare.planned_points(base['points'], ['alpaca','sharegpt','longbench'], preserved)
    assert len(pending) == 7 and base['points'][0]['name'] not in pending
    groups = prepare.pending_groups(group_points(base['points']), pending)
    assert len(groups) == 1 and len(groups[0]['points']) == 8
    session = ResidentGroupSession(groups[0], None, tmp_path/'next', previous=[previous])
    assert session._completed(base['points'][0]) == preserved[base['points'][0]['name']]
    job = dict(payload=dict(argv=['python','runner']))
    prepare.bind_previous(job, [previous])
    assert job['payload']['argv'][-2:] == ['--previous', str(previous.resolve())]
    assert job['payload']['prior_sessions'] == [str(previous.resolve())]


def test_all_frozen_creates_no_new_jobs(tmp_path):
    base = campaign(prepared=True)
    previous = write_session(tmp_path/'previous', base['points'][:8])
    preserved = prepare.frozen_points(base, [previous])
    pending = prepare.planned_points(base['points'], ['alpaca','sharegpt'], preserved)
    assert pending == []
    assert prepare.pending_groups(group_points(base['points']), pending) == []


def test_new_longbench_points_can_follow_eight_frozen_points(tmp_path):
    base = campaign(longbench=True, prepared=True)
    previous = write_session(tmp_path/'previous', base['points'][:8])
    preserved = prepare.frozen_points(base, [previous])
    pending = prepare.planned_points(base['points'], ['alpaca','sharegpt','longbench'], preserved)
    assert len(pending) == 4
    groups = prepare.pending_groups(group_points(base['points']), pending)
    assert len(groups) == 1 and len(groups[0]['points']) == 12
    assert all('longbench' in name for name in pending)


@pytest.mark.parametrize('tamper', ['receipt','artifact','point','result','engine','escape','duplicate'])
def test_frozen_evidence_tampering_rejected(tmp_path, tamper):
    base = campaign(prepared=True)
    previous = write_session(tmp_path/'previous', base['points'][:1])
    window = previous/'windows'/base['points'][0]['name']
    receipt_path = window/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if tamper == 'artifact':
        (window/'result.json').write_text('{}')
    elif tamper == 'point':
        base['points'][0]['revision'] = 'new-revision-cannot-rewrite-valid-point'
    elif tamper == 'duplicate':
        with pytest.raises(ValueError, match='unique frozen'):
            prepare.frozen_points(base, [previous,previous])
        return
    else:
        if tamper in ('receipt','result'):receipt['result']['slo_pass'] = True
        if tamper == 'engine':receipt['engine_signature'] = 'different-engine'
        if tamper == 'escape':receipt['artifacts']['../outside.json'] = 'anything'
        overwrite(receipt_path, receipt)
        if tamper != 'receipt':
            completion_path = previous/'completion.json'
            completion = json.loads(completion_path.read_text())
            completion['windows'][0]['sha256'] = file_sha(receipt_path)
            overwrite(completion_path, completion)
    with pytest.raises(ValueError):
        prepare.frozen_points(base, [previous])


def test_failed_measurement_can_be_retried_but_unrelated_previous_cannot(tmp_path):
    base = campaign(prepared=True)
    previous = write_session(tmp_path/'failed', base['points'][:1], evidence_valid=False)
    assert prepare.frozen_points(base, [previous]) == {}
    assert len(prepare.planned_points(base['points'], ['alpaca','sharegpt'], {})) == 8
    path = previous/'completion.json'; report = json.loads(path.read_text())
    report['engine_signature'] = 'unrelated-session'; overwrite(path, report)
    with pytest.raises(ValueError, match='no matching terminal'):
        prepare.frozen_points(base, [previous])


def test_inherited_receipt_requires_original_previous_session(tmp_path):
    base = campaign(prepared=True)
    first = write_session(tmp_path/'first', base['points'][:1])
    ref = binding(first/'windows'/base['points'][0]['name']/'receipt.json')
    base['preserved_baseline_receipts'] = {base['points'][0]['name']:ref}
    later = write_session(tmp_path/'later', base['points'][1:2], evidence_valid=False)
    with pytest.raises(ValueError, match='original --previous'):
        prepare.frozen_points(base, [later])
    assert prepare.frozen_points(base, [first,later]) == base['preserved_baseline_receipts']


def test_nonterminal_previous_is_rejected(tmp_path):
    previous = tmp_path/'running'; previous.mkdir()
    with pytest.raises(ValueError, match='terminal'):
        prepare.frozen_points(campaign(prepared=True), [previous])


def queued_fixture(tmp_path, base=None):
    base=campaign(longbench=True,prepared=True) if base is None else base
    for point in base['points']:
        if prepare.is_ecoserve32(point):point.setdefault('revision','old-source')
    groups=[g for g in group_points(base['points']) if all(prepare.is_ecoserve32(p) for p in g['points'])]
    assert len(groups)==1
    group=groups[0];path=tmp_path/'group.json';write_new(path,group)
    job_id='comparison-32b-'+digest(group)[:16]
    payload=dict(argv=['python','-m','pdblend.bench.comparison_runtime','--group',str(path)],
        session_id=group['session_id'],system='ecoserve',model_id='Qwen2.5-32B-Instruct',
        gpu_count=8,exclusive=True,reserve_host=True,source_sha256=group['points'][0]['revision'],prior_sessions=[])
    job=dict(job_id=job_id,status='queued',attempts=0,lease_id=None,payload=payload)
    queue=tmp_path/'queue.json';write_new(queue,dict(jobs={job_id:job},leases={}))
    return base,queue,job_id,path


def test_exact_unclaimed_queue_exception_is_read_only(tmp_path):
    base,queue,job_id,path=queued_fixture(tmp_path);before=queue.read_bytes();old=deepcopy(base)
    result=prepare.queued_replacement(base,queue,job_id)
    assert queue.read_bytes()==before and base==old
    assert result['supersedes_job_id']==job_id and result['old_group']==binding(path)
    assert result['atomic_scheduling_recheck_required'] and not result['queue_modified']
    assert len(result['point_sha256'])==12
    # The exception must not weaken the default terminal --previous rule.
    with pytest.raises(ValueError,match='--previous is required'):prepare.frozen_points(base,[])


@pytest.mark.parametrize('fault',['running','attempted','bool_attempts','lease','lease_history',
    'attempt_directory','frozen_marker','preserved','frozen_elsewhere','point_changed','group_changed',
    'missing_dataset','duplicate_group_flag','prior_session','wrong_source'])
def test_queued_replacement_rejects_executed_frozen_or_mismatched_inventory(tmp_path,fault):
    base,queue,job_id,path=queued_fixture(tmp_path);q=json.loads(queue.read_text());job=q['jobs'][job_id]
    if fault=='running':job['status']='running'
    elif fault=='attempted':job['attempts']=1
    elif fault=='bool_attempts':job['attempts']=False
    elif fault=='lease':job['lease_id']='claimed'
    elif fault=='lease_history':q['leases']['claimed']=dict(job_id=job_id,status='released')
    elif fault=='attempt_directory':(tmp_path/'queue-attempts'/job_id/'attempt-0001').mkdir(parents=True)
    elif fault=='frozen_marker':base['points'][0]['evidence_valid']=True
    elif fault=='preserved':base['preserved_baseline_receipts']={base['points'][0]['name']:dict(path='frozen',sha256='hash')}
    elif fault=='frozen_elsewhere':
        p=tmp_path/'queue-attempts/older-job/attempt-0001/session/windows'/base['points'][0]['name']/'receipt.json'
        write_new(p,dict(evidence_valid=True,baseline_frozen=True))
    elif fault=='point_changed':base['points'][0]['changed_trace']='different'
    elif fault=='group_changed':
        g=json.loads(path.read_text());g['points'][0]['changed_trace']='different';overwrite(path,g)
    elif fault=='missing_dataset':base['points']=base['points'][:8]
    elif fault=='duplicate_group_flag':job['payload']['argv']+=['--group',str(path)]
    elif fault=='prior_session':job['payload']['prior_sessions']=['earlier-session']
    elif fault=='wrong_source':job['payload']['source_sha256']='different-source'
    overwrite(queue,q)
    with pytest.raises(ValueError):prepare.queued_replacement(base,queue,job_id)


def test_queue_flags_are_explicit_and_one_overlay_replaces_defaults():
    args=['--base','base','--out','out','--source-base','source','--after-terminal','anchor']
    assert prepare.parse_args(args).overlay==list(prepare.DEFAULT_OVERLAYS)
    parsed=prepare.parse_args(args+['--overlay','pdblend/bench/resident_session.py',
        '--replace-queued-job','old','--queue','queue','--continue-frequency-rejections'])
    assert parsed.overlay==['pdblend/bench/resident_session.py'] and parsed.continue_frequency_rejections
    for flags in (['--replace-queued-job','old'],['--queue','queue']):
        with pytest.raises(ValueError,match='supplied together'):prepare.parse_args(args+flags)


def test_replacement_main_preserves_other_points_and_binds_current_source_without_default_overlay(tmp_path,monkeypatch):
    project=tmp_path/'project';(project/'scripts').mkdir(parents=True)
    script='2026-09-22_enqueue_parallel_profiles.py'
    shutil.copyfile(prepare.ROOT/'scripts'/script,project/'scripts'/script)
    content={'pdblend_runtime/serve.py':'unchanged engine\n','pdblend/measure/power.py':'unchanged meter\n',
        'pdblend/bench/resident_session.py':'old coordinator\n','pdblend/bench/comparison_runtime.py':'frozen wrapper\n'}
    for name,text in content.items():
        target=project/'src'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    spec=importlib.util.spec_from_file_location('test_eco32_freezer',project/'scripts'/script)
    freezer=importlib.util.module_from_spec(spec);spec.loader.exec_module(freezer)
    source,sha=freezer.freeze_source(project/'src',project/'old-sources');files=json.loads((source/'manifest.json').read_text())['files']
    base=campaign(longbench=True,prepared=True)
    identity=base['points'][0]['engine_identity'];identity.update(
        runtime_source_sha256=digest({k:v for k,v in files.items() if k.startswith('pdblend_runtime/')}),
        measurement_source_sha256=digest({k:v for k,v in files.items() if k.startswith('pdblend/measure/')}),
        fleet_gpu_uuids=[f'GPU-{i}' for i in range(8)])
    trace=project/'trace.json';write_new(trace,dict(requests=[]));artifact=binding(trace)
    for point in base['points']:
        point.update(engine_identity=deepcopy(identity),revision=sha,trace=artifact,slo=dict(ttft_s=5.,tpot_s=.2))
    mixed=dict(deepcopy(base['points'][0]),name='32b-mixed-preserved',system='mixed')
    base['points'].append(mixed);base['execution_source_manifest']=binding(source/'manifest.json')
    base,queue,job_id,_=queued_fixture(project,base);base_path=project/'base/campaign.json';write_new(base_path,base)
    write_new(base_path.parent/'execution-inputs.json',dict(image_digest='image',model_verification=dict(path='verified.json')))
    evidence={key:artifact for key in ('profile_csv','profile_manifest','mechanism_completion','automatic_completion','independent_mechanism_review')}
    ready=project/'ready.json';write_new(ready,dict(models={'Qwen2.5-32B-Instruct':dict(ecoserve=evidence)},
        ecosystem_source_continuity={key:dict(source_manifest=binding(source/'manifest.json')) for key in ('profile','mechanism')}))
    (project/'src/pdblend/bench/resident_session.py').write_text('new strict frequency coordinator\n')
    (project/'src/pdblend/bench/comparison_runtime.py').write_text('MUST NOT OVERLAY\n')
    monkeypatch.setattr(prepare,'ROOT',project)
    monkeypatch.setattr(prepare,'validate_ecoserve_inputs',lambda *a,**kw:dict(preflight_ready=True,missing_gates=[]))
    out=project/'new';before=queue.read_bytes()
    prepare.main(['--base',str(base_path),'--out',str(out),'--source-base',str(source),'--after-terminal','anchor',
        '--overlay','pdblend/bench/resident_session.py','--readiness',str(ready),'--replace-queued-job',job_id,
        '--queue',str(queue),'--continue-frequency-rejections'])
    current=json.loads((out/'campaign.json').read_text());jobs=json.loads((out/'jobs.json').read_text())
    assert queue.read_bytes()==before and len(jobs)==1 and jobs[0]['payload']['supersedes_job_id']==job_id
    assert current['parent_campaign']==binding(base_path) and current['parent_execution_source_manifest']==base['execution_source_manifest']
    execution=json.loads((out/'execution-inputs.json').read_text());new_source=Path(execution['source'])
    assert current['execution_source_manifest']==binding(new_source/'manifest.json')
    assert current['execution_inputs']==binding(out/'execution-inputs.json')
    assert (new_source/'pdblend/bench/comparison_runtime.py').read_text()=='frozen wrapper\n'
    assert next(p for p in current['points'] if p['system']=='mixed')==mixed
    assert current['summary']['new_ecoserve_points']==12 and all(
        p['observation_failure_policy']=='continue_after_verified_frequency_rejection'
        for p in current['points'] if prepare.is_ecoserve32(p))
