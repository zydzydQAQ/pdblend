from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pdblend.bench.comparison_qualification_evidence import (
    SCHEMA, SCOPE, QualificationEvidence, annotate, host_path, load_qualification_evidence,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bound(path, value=None):
    path = Path(path)
    if value is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(ref):
    return json.loads(Path(ref['path']).read_bytes())


def fixture(root, revision='r1'):
    base = root/revision
    source = base/'source'
    source.mkdir(parents=True)
    code = source/'pdblend/profile/collection/native_timing_collect.py'
    code.parent.mkdir(parents=True)
    code.write_text('# CPU fixture collector\n')
    files = {'pdblend/profile/collection/native_timing_collect.py': bound(code)['sha256']}
    source_ref = bound(source/'manifest.json', dict(files=files, source_sha256=digest(files)))
    verification = bound(base/'verification.json', {'model': 'Qwen2.5-7B-Instruct'})
    plan = bound(base/'points.json', dict(system='pdblend', model_id='Qwen2.5-7B-Instruct',
                                        tp=1, pp=1, evaluation_used_for_selection=False))
    inputs = bound(base/'inputs.json', dict(system='pdblend', model_id='Qwen2.5-7B-Instruct',
        source_manifest=source_ref, source_sha256=digest(files), point_plan=plan, model_verification=verification, image_digest='sha256:fixture'))
    jid, lid = 'timing-' + revision, 'lease-' + revision
    attempt = base/jid/('attempt-0001-' + lid)
    payload = dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', source_sha256=digest(files),
        input_manifest=inputs, image_digest='sha256:fixture', required_receipts=['native-timing/completion.json'], argv=[
            'docker', 'run', '--gpus', 'all', '-v', str(source)+':/opt/pdblend-src:ro',
            '-v', str(root)+':'+str(root)+':ro', '-v', '{attempt_dir}:{attempt_dir}:rw',
            '-e', 'PDBLEND_SOURCE_MANIFEST='+source_ref['path'], '-e', 'PDBLEND_SOURCE_SHA256='+digest(files),
            '-e', 'PDBLEND_MODEL_VERIFICATION_RECEIPT='+verification['path'],
            '-e', 'PDBLEND_IMAGE_ID=sha256:fixture', 'sha256:fixture', '-m', 'pdblend.profile.collection.native_timing_collect',
            '--model', '/models/Qwen2.5-7B-Instruct', '--point-plan', plan['path'],
            '--input-manifest', inputs['path'], '--out', '{attempt_dir}/native-timing'])
    attempt_manifest = dict(schema=1, immutable=True, job_id=jid, lease_id=lid, attempt=1,
        claimed_at=10., gpu_indices=['0'], gpu_uuids=['GPU-a'], owner='host:123', owner_pid=123, payload=payload)
    mref = bound(attempt/'manifest.json', attempt_manifest)
    argv = [v.replace('{attempt_dir}', str(attempt)) for v in payload['argv']]
    argv[3] = '"device=GPU-a"'
    execution = bound(attempt/'execution.json', dict(status='failed', complete=False, returncode=2,
        started_s=11., finished_s=12., argv=argv, formal_eligible=False, receipt_sha256={}))
    component = bound(attempt/'native-timing/runtime/completion.json', dict(complete=True,
        component_qualified=False, safe_restore=True, formal_eligible=False))
    completion = bound(attempt/'native-timing/completion.json', dict(status='failed', complete=False,
        formal_eligible=False, error='reported failure', resident_runtime=component))
    lease = {k:v for k,v in attempt_manifest.items() if k not in ('schema','immutable','payload')}
    lease.update(status='failed', attempt_dir=str(attempt))
    queue = bound(base/'capture.json', dict(schema='terminal-queue-metadata-capture/v1', capture_only=True,
        captured_at_s=13., jobs={jid: dict(status='failed', attempts=1, lease_id=None, payload=payload)}, leases={lid:lease}))
    manifest = bound(base/'history.json', dict(schema=SCHEMA, scope=SCOPE, formal_eligible=False,
        current_point_failure=False, terminal_queue_capture=queue, attempts=[dict(attempt_manifest=mref,
            execution=execution, completion=completion, input_manifest=inputs, source_manifest=source_ref,
            point_plan=plan, model_verification=verification)]))
    return manifest


def edit_entry(ref, name, edit):
    manifest = load(ref)
    child_ref = manifest['attempts'][0][name]
    child = load(child_ref)
    edit(child)
    manifest['attempts'][0][name] = bound(child_ref['path'], child)
    return bound(ref['path'], manifest)


def edit_capture(ref, edit):
    manifest = load(ref)
    queue_ref = manifest['terminal_queue_capture']
    queue = load(queue_ref)
    edit(queue)
    manifest['terminal_queue_capture'] = bound(queue_ref['path'], queue)
    return bound(ref['path'], manifest)


def test_real_binding_and_watch_inventory_no_raw_replay(tmp_path):
    ref = fixture(tmp_path)
    evidence = load_qualification_evidence(ref)
    row, = evidence.by_system_model['pdblend', 'Qwen2.5-7B-Instruct']
    assert row['reported']['error'] == 'reported failure'
    assert row['components'][0]['reported']['safe_restore'] is True
    assert row['applicability'] == 'historical_related_component_only'
    assert row['current_engine_meter_domain_equivalence'] == 'unproven'
    assert row['raw_qualification_replayed'] is False
    assert str(tmp_path/'r1/source/pdblend/profile/collection/native_timing_collect.py') in evidence.watch_paths
    assert len(evidence.refs) == len(set(evidence.watch_paths))
    assert any(p.endswith('/runtime/completion.json') for p in evidence.watch_paths)


@pytest.mark.parametrize('artifact', ['execution', 'completion', 'source_manifest', 'point_plan', 'input_manifest'])
def test_unrebound_artifact_tamper_rejected(tmp_path, artifact):
    ref = fixture(tmp_path)
    target = load(ref)['attempts'][0][artifact]
    Path(target['path']).write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        load_qualification_evidence(ref)


@pytest.mark.parametrize('name,edit,match', [
    ('execution', lambda x:x.update(status='running'), 'terminal'),
    ('execution', lambda x:x.update(finished_s=9.), 'timestamps'),
    ('execution', lambda x:x['argv'].append('--invented'), 'argv'),
    ('completion', lambda x:x.update(formal_eligible=True), 'completion'),
    ('completion', lambda x:x.update(system='distserve'), 'system'),
    ('point_plan', lambda x:x.update(model_id='Qwen2.5-14B-Instruct'), 'model'),
    ('point_plan', lambda x:x.update(evaluation_used_for_selection=True), 'selection'),
    ('attempt_manifest', lambda x:x.update(immutable=False), 'immutable'),
])
def test_rebinding_does_not_hide_identity_or_terminal_change(tmp_path, name, edit, match):
    ref = edit_entry(fixture(tmp_path), name, edit)
    with pytest.raises(ValueError, match=match):
        load_qualification_evidence(ref)


@pytest.mark.parametrize('edit,match', [
    (lambda x:next(iter(x['leases'].values())).update(status='running'), 'terminal'),
    (lambda x:next(iter(x['leases'].values())).update(owner_pid=456), 'lease identity'),
    (lambda x:next(iter(x['leases'].values())).update(token='secret'), 'token'),
    (lambda x:next(iter(x['jobs'].values())).update(attempts=0), 'terminal'),
    (lambda x:next(iter(x['jobs'].values())).update(lease_id='active'), 'active lease'),
])
def test_running_unexecuted_or_wrong_lease_capture_rejected(tmp_path, edit, match):
    ref = edit_capture(fixture(tmp_path), edit)
    with pytest.raises(ValueError, match=match):
        load_qualification_evidence(ref)


def test_actual_source_bytes_are_checked(tmp_path):
    ref = fixture(tmp_path)
    source = tmp_path/'r1/source/pdblend/profile/collection/native_timing_collect.py'
    source.write_text('# changed source\n')
    with pytest.raises(ValueError, match='source bytes'):
        load_qualification_evidence(ref)


def test_other_attempt_completion_cannot_be_spliced(tmp_path):
    one, two = fixture(tmp_path, 'r1'), fixture(tmp_path, 'r2')
    manifest = load(one)
    manifest['attempts'][0]['completion'] = load(two)['attempts'][0]['completion']
    ref = bound(one['path'], manifest)
    with pytest.raises(ValueError, match='another attempt'):
        load_qualification_evidence(ref)


def test_component_cannot_escape_attempt(tmp_path):
    one, two = fixture(tmp_path, 'r1'), fixture(tmp_path, 'r2')
    other = load(load(two)['attempts'][0]['completion'])['resident_runtime']
    ref = edit_entry(one, 'completion', lambda x:x.update(resident_runtime=other))
    with pytest.raises(ValueError, match='another output'):
        load_qualification_evidence(ref)


def test_duplicate_attempt_rejected_all_revisions_retained(tmp_path):
    one, two = fixture(tmp_path, 'r1'), fixture(tmp_path, 'r2')
    first, second = load(one), load(two)
    queue = load(first['terminal_queue_capture'])
    other_queue = load(second['terminal_queue_capture'])
    queue['jobs'].update(other_queue['jobs']); queue['leases'].update(other_queue['leases'])
    first['terminal_queue_capture'] = bound(first['terminal_queue_capture']['path'], queue)
    first['attempts'] += second['attempts']
    ref = bound(one['path'], first)
    assert len(load_qualification_evidence(ref).by_system_model['pdblend','Qwen2.5-7B-Instruct']) == 2
    first['attempts'].append(deepcopy(first['attempts'][0]))
    with pytest.raises(ValueError, match='duplicate qualification attempt'):
        load_qualification_evidence(bound(one['path'], first))


def test_annotation_only_receiptless_blocked_exact_system_model_and_never_changes_flags(tmp_path):
    evidence = load_qualification_evidence(fixture(tmp_path))
    base = dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', status='blocked',
        formal_eligible=False, evidence_valid=False, rank_eligible=False, energy_service_j='',
        failure_reason='missing_profile;offline_choice', receipt_path='', receipt_sha256='')
    rows = [base, dict(base, status='measured', baseline_frozen=True), dict(base, status='invalid'),
            dict(base, receipt_path='/a/receipt'), dict(base, receipt_sha256='a'),
            dict(base, system='ecoserve'), dict(base, model_id='Qwen2.5-14B-Instruct')]
    saved = deepcopy(rows)
    annotated = annotate(rows, evidence)
    assert rows == saved and annotated[1:] == rows[1:]
    assert annotated[0]['qualification_attempt_count'] == 1
    assert {k:annotated[0][k] for k in base} == base
    assert annotated[0]['qualification_evidence_scope'] == SCOPE


def test_loader_and_hash_cache_injection_used_once(tmp_path):
    ref = fixture(tmp_path)
    calls, hashes = [], []
    def checked_load(reference):
        calls.append(reference['path'])
        assert bound(reference['path']) == reference
        return load(reference)
    def cached_sha(path):
        hashes.append(str(path))
        return bound(path)['sha256']
    evidence = load_qualification_evidence(ref, load_bound=checked_load, file_sha=cached_sha)
    before = (calls[:], hashes[:])
    annotate([], evidence)
    assert (calls, hashes) == before and len(calls) == len(set(calls)) and len(hashes) == 1


def test_container_longest_mount_resolution_and_ambiguity(tmp_path):
    argv = ['docker', 'run', '-v', str(tmp_path)+':/spec:ro', '-v', str(tmp_path/'special')+':/spec/points.json:ro']
    assert host_path(argv, '/spec/points.json') == str(tmp_path/'special')
    with pytest.raises(ValueError, match='not bound'):
        host_path(argv, '/other')
    with pytest.raises(ValueError, match='ambiguous'):
        host_path(argv+['-v', str(tmp_path/'wrong')+':/spec/points.json:ro'], '/spec/points.json')


def dist_fixture(root):
    ref = fixture(root)
    manifest = load(ref)
    entry = manifest['attempts'][0]
    m = load(entry['attempt_manifest'])
    p = m['payload']
    source_root = Path(entry['source_manifest']['path']).parent
    module = 'pdblend_baselines.distserve.stage_cohort'
    code = source_root/'pdblend_baselines/distserve/stage_cohort.py'
    code.parent.mkdir(parents=True)
    code.write_text('# DistServe CPU fixture\n')
    files = {module.replace('.', '/')+'.py':bound(code)['sha256']}
    entry['source_manifest'] = bound(entry['source_manifest']['path'], dict(files=files,source_sha256=digest(files)))
    plan = load(entry['point_plan']); plan['system'] = 'distserve'
    entry['point_plan'] = bound(entry['point_plan']['path'],plan)
    base = Path(ref['path']).parent
    cohort = bound(base/'cohort.json',dict(source_sha256=digest(files), members={'member':dict(
        model_id=plan['model_id'],tp=1,pp=1,gpu_count=1,
        point_plan=dict(path='points.json',sha256=entry['point_plan']['sha256']))}))
    entry['cohort_inputs'] = cohort
    entry['input_manifest'] = bound(entry['input_manifest']['path'], dict(source_sha256=digest(files),
        image_digest='sha256:fixture', cohort_member='member',cohort_sha256=cohort['sha256'],
        exact_inputs_sha256={k:entry[k]['sha256'] for k in ('source_manifest','point_plan','model_verification')}))
    old_sha = p['source_sha256']
    p.update(system='distserve',tp=1,pp=1,source_sha256=digest(files),cohort_sha256=cohort['sha256'])
    p.pop('input_manifest')
    p['argv']=[v.replace('pdblend.profile.collection.native_timing_collect',module).replace(old_sha,digest(files)) for v in p['argv']]
    image_index=p['argv'].index('sha256:fixture')
    p['argv'][image_index:image_index]=['-e','PDBLEND_PROFILE_MEMBER=member']
    p['argv']+=['--tp','1','--cohort-inputs',cohort['path'],'--member','member']
    entry['attempt_manifest']=bound(entry['attempt_manifest']['path'],m)
    actual=[v.replace('{attempt_dir}',str(Path(entry['execution']['path']).parent)) for v in p['argv']]
    actual[3]='"device=GPU-a"'
    execution=load(entry['execution']);execution['argv']=actual
    entry['execution']=bound(entry['execution']['path'],execution)
    capture=load(manifest['terminal_queue_capture'])
    capture['jobs'][m['job_id']]['payload']=p
    manifest['terminal_queue_capture']=bound(manifest['terminal_queue_capture']['path'],capture)
    return bound(ref['path'],manifest)


def test_dist_cohort_model_owned_source_and_plan_binding(tmp_path):
    evidence=load_qualification_evidence(dist_fixture(tmp_path))
    assert ('distserve','Qwen2.5-7B-Instruct') in evidence.by_system_model
    assert any(path.endswith('/cohort.json') for path in evidence.watch_paths)


@pytest.mark.parametrize('name,edit,match',[
    ('cohort_inputs',lambda x:x['members']['member'].update(model_id='Qwen2.5-32B-Instruct'),'cohort source/input'),
    ('input_manifest',lambda x:x['exact_inputs_sha256'].update(point_plan='f'*64),'exact input'),
    ('point_plan',lambda x:x.update(tp=2),'topology'),
])
def test_dist_rebound_cohort_or_model_input_mismatch_rejected(tmp_path,name,edit,match):
    ref=edit_entry(dist_fixture(tmp_path),name,edit)
    with pytest.raises(ValueError,match=match):
        load_qualification_evidence(ref)


def test_two_manifests_cannot_claim_conflicting_sha_for_same_source_file(tmp_path):
    one, two = fixture(tmp_path, 'r1'), fixture(tmp_path, 'r2')
    first, second = load(one), load(two)
    first_entry, other = first['attempts'][0], second['attempts'][0]
    shared_root = Path(first_entry['source_manifest']['path']).parent
    source = load(other['source_manifest'])
    source['files'] = {name:'f'*64 for name in source['files']}
    source['source_sha256'] = digest(source['files'])
    # Both inventory manifests are in the SAME source directory, but the second
    # falsely declares different bytes for the already checked collector file.
    prior_root = Path(other['source_manifest']['path']).parent
    other['source_manifest'] = bound(shared_root/'alternative-manifest.json', source)
    inputs = load(other['input_manifest'])
    prior_sha = inputs['source_sha256']
    inputs.update(source_manifest=other['source_manifest'],source_sha256=source['source_sha256'])
    other['input_manifest'] = bound(other['input_manifest']['path'],inputs)
    m = load(other['attempt_manifest'])
    p = m['payload']
    p.update(input_manifest=other['input_manifest'],source_sha256=source['source_sha256'])
    p['argv']=[value.replace(str(prior_root),str(shared_root)).replace(prior_sha,source['source_sha256'])
               for value in p['argv']]
    p['argv']=[('PDBLEND_SOURCE_MANIFEST='+other['source_manifest']['path'])
               if value.startswith('PDBLEND_SOURCE_MANIFEST=') else value for value in p['argv']]
    other['attempt_manifest'] = bound(other['attempt_manifest']['path'],m)
    execution = load(other['execution'])
    actual=[value.replace('{attempt_dir}',str(Path(other['execution']['path']).parent)) for value in p['argv']]
    actual[3]='"device=GPU-a"'
    execution['argv']=actual
    other['execution']=bound(other['execution']['path'],execution)
    queue, extra = load(first['terminal_queue_capture']), load(second['terminal_queue_capture'])
    extra['jobs'][m['job_id']]['payload']=p
    queue['jobs'].update(extra['jobs']);queue['leases'].update(extra['leases'])
    first['terminal_queue_capture']=bound(first['terminal_queue_capture']['path'],queue)
    first['attempts'].append(other)
    with pytest.raises(ValueError,match='conflicting evidence hashes'):
        load_qualification_evidence(bound(one['path'],first))


def test_export_integrates_history_without_changing_measurement_and_refuses_tamper_atomically(tmp_path,monkeypatch):
    import csv
    from pdblend.bench import comparison_campaign as campaign
    ref=fixture(tmp_path/'evidence')
    blocked=dict(name='blocked-pd',model_id='Qwen2.5-7B-Instruct',dataset='alpaca',system='pdblend',
        scale=.5,rate_rps=1.,seed=701,duration_s=150.,revision='initial',status='blocked',
        blockers=['missing_profile','offline_choice'],trace={'sha256':'d'*64},
        measurement_protocol_version=campaign.PROTOCOL,slo={'ttft_s':1.,'tpot_s':.1})
    measured=dict(blocked,name='frozen-mixed',system='mixed',status='prepared',blockers=[])
    campaign_path=tmp_path/'campaign.json'
    bound(campaign_path,dict(campaign_id='test',points=[blocked,measured]))
    session=tmp_path/'session'
    window=session/'windows'/measured['name']
    result=dict(evidence_valid=True,formal_eligible=True,identity={},
                metrics=dict(slo_pass=False,energy_service_j=10.,energy_tail_j=2.))
    result_ref=bound(window/'result.json',result)
    point_ref=bound(window/'point.json',measured)
    receipt=bound(window/'receipt.json',dict(point=measured['name'],point_sha256=digest(measured),result=result,
        session_id='session',engine_signature='engine',evidence_valid=True,cleanup_passed=True,baseline_frozen=True,
        artifacts={'result.json':result_ref['sha256'],'point.json':point_ref['sha256']}))
    target=tmp_path/'compare.csv'
    original_summary=campaign.export(campaign_path,target,session_roots=[session])
    original=list(csv.DictReader(target.open()))
    summary=campaign.export(campaign_path,target,session_roots=[session],qualification_evidence_ref=ref)
    annotated=list(csv.DictReader(target.open()))
    assert summary==original_summary and summary['measured']==1
    for old,row in zip(original,annotated):
        assert all(row[key]==value for key,value in old.items())
        if row['point_id']=='blocked-pd':
            assert row['qualification_attempt_count']=='1'
        else:
            assert row['qualification_attempt_count']=='' and row['baseline_frozen']=='True'
            assert row['receipt_sha256']==receipt['sha256'] and row['slo_pass']=='False'
    before=target.read_bytes()
    entry=load(ref)['attempts'][0]
    Path(entry['completion']['path']).write_text('{}')
    replaced=[]
    monkeypatch.setattr(campaign.os,'replace',lambda *args:replaced.append(args))
    with pytest.raises(ValueError,match='checksum'):
        campaign.export(campaign_path,target,session_roots=[session],qualification_evidence_ref=ref)
    assert target.read_bytes()==before and replaced==[]
