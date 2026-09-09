import copy
import importlib.util
import json
from pathlib import Path
import time

import pytest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('batch16_capacity_test',ROOT/'capacity.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def runtime():
    return dict(timestamp=100.,id='nextv3b0',generation=8,acknowledged_generation=8,
        observed_control_generation=8,acknowledged_generations=[8],scheduler_budget_pending=None,
        scheduler_io=[dict(controls=dict(runtime=dict(generation=8,error=None)))],
        scheduler_budget_effective=dict(max_num_batched_tokens=8192,max_num_seqs=32),
        scheduler_budget=dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32),
        scheduler_budget_limits=dict(max_num_batched_tokens=8192,max_num_seqs=32,max_model_len=8192,max_num_partial_prefills=1),
        role='mixed',mode='continuous',accepting=True,admit_prefill=True,admit_decode=True,
        transport_healthy=True,active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={},
        transfer_buffered_tensors=0,transfer_inflight_receives=0,transfer_inflight_sends=0,
        free_kv_tokens=12288,total_kv_tokens=47104)


def test_minimum_exact_complete_work_capacity():
    proof=c.validate_runtime_capacity(runtime(),now=100.5)
    assert proof['required_kv_tokens']==16*(512+256)==12288
    assert proof['initial_prefill_token_sum']==8192 and proof['max_num_seqs']==32
    assert 'no first-step or KV correctness result' in proof['interpretation']


@pytest.mark.parametrize('key,value',[
    ('free_kv_tokens',12287),('free_kv_tokens',None),('free_kv_tokens',True),
    ('total_kv_tokens',10000),('timestamp',98.99),('timestamp',100.1),
    ('acknowledged_generation',7),('acknowledged_generations',[8,8]),
    ('observed_control_generation',7),('scheduler_budget_pending',{'max_num_seqs':16}),
    ('scheduler_budget_effective',{'max_num_batched_tokens':2048,'max_num_seqs':32}),
    ('transport_healthy',False),('running',1),('waiting',1),
    ('transfer_inflight_sends',None),('kv_allocations',{'live':16}),('mode','temporal'),
])
def test_capacity_unknown_or_unsafe_runtime_rejected(key,value):
    state=runtime();state[key]=value
    with pytest.raises(RuntimeError):c.validate_runtime_capacity(state,now=100.)


def test_max_context_and_cache_are_not_invented_or_changed():
    state=runtime();state['scheduler_io'][0]['controls']['runtime']['generation']=7
    with pytest.raises(RuntimeError,match='cache'):c.validate_runtime_capacity(state,now=100.)
    state=runtime();state['scheduler_budget_limits']['max_model_len']=16384
    with pytest.raises(RuntimeError,match='startup'):c.validate_runtime_capacity(state,now=100.)


def test_between_point_drained_admission_is_allowed_but_not_initial():
    state=runtime();state.update(accepting=False,admit_prefill=False)
    with pytest.raises(RuntimeError,match='admission'):c.validate_runtime_capacity(state,now=100.)
    c.validate_runtime_capacity(state,now=100.,require_accepting=False)


def point():
    rows=[dict(request_id=f'r{i}',success=True,prompt_token_ids=[1]*512,output_token_ids=[2]*256,
               token_received_s=list(range(256)),usage=dict(prompt_tokens=512,completion_tokens=256)) for i in range(16)]
    spec=dict(batch_size=16,input_lengths=[512]*16,output_lengths=[256]*16,budget_tokens=8192,max_num_seqs=32,tp=2)
    event=dict(generation=8,role='mixed',mode='continuous',tokens=16,prefill=0,decode=16,request_ids=[r['request_id'] for r in rows])
    prefill=dict(event,prefill=16,decode=0,tokens=8192)
    return dict(spec=spec,requests=rows,generation=8),[prefill]+[dict(event) for _ in range(64)]


def test_only_true_sustained16_full_token_observation_accepted():
    raw,events=point();result=c.validate_batch16_observation(raw,events)
    assert result['full_batch_decode_steps']==64 and result['exact_output_equal_within_batch']
    assert not result['profile_point_generated'] and not result['kv_correctness_certified']


@pytest.mark.parametrize('what',['eight_relabelled','missing_output','different_output','too_few_steps','wrong_generation','overbudget'])
def test_not_derived_from_batch8_or_partial_work(what):
    raw,events=point()
    if what=='eight_relabelled':
        for e in events[1:]:e['decode']=8;e['tokens']=8
    if what=='missing_output':raw['requests'][0]['output_token_ids'].pop()
    if what=='different_output':raw['requests'][0]['output_token_ids'][5]=3
    if what=='too_few_steps':events.pop()
    if what=='wrong_generation':events[3]['generation']=7
    if what=='overbudget':events[0]['tokens']=8193
    with pytest.raises(RuntimeError):c.validate_batch16_observation(raw,events)


def release_fixture(tmp_path,monkeypatch):
    root=tmp_path/'campaign/candidate';root.mkdir(parents=True)
    (root/'manifest.json').write_text('candidate')
    deadline=root.parent/'deadline-24h-v1';deadline.mkdir();(deadline/'protocol.json').write_text(json.dumps(dict(deadline_s=2000.)))
    main=root.parent/'B32B-main-scale-fixed-window-v1';(main/'invocations').mkdir(parents=True)
    (main/'package-manifest.json').write_text('main')
    monkeypatch.setattr(c,'MAIN_PACKAGE_SHA',c.sha(main/'package-manifest.json'))
    monkeypatch.setattr(c,'DEADLINE_SHA',c.sha(deadline/'protocol.json'))
    refs={}
    for n,phase in enumerate(('main','scale')):
        p=main/'invocations'/f'{n:06d}.json'
        p.write_text(json.dumps(dict(selected_phase=phase,complete=True,phase='finished',finished_s=900.,selected_phase_execution_complete=True)))
        refs[phase]=dict(path=str(p),sha256=c.sha(p))
    release=dict(authorized_by='root',execute_once=True,candidate_manifest_sha256=c.sha(root/'manifest.json'),expires_s=1800.,phase_terminal_evidence=refs)
    path=tmp_path/'release.json';path.write_text(json.dumps(release))
    return root,path,release


def test_release_never_schedules_current_main_and_scale(tmp_path,monkeypatch):
    root,path,release=release_fixture(tmp_path,monkeypatch)
    proof=c.validate_execution_release(path,root,now=1000.)
    assert not proof['scheduled_automatically']
    phasepath=Path(release['phase_terminal_evidence']['main']['path'])
    status=json.loads(phasepath.read_text());status.update(complete=False,phase='cell:live')
    phasepath.write_text(json.dumps(status));release['phase_terminal_evidence']['main']['sha256']=c.sha(phasepath);path.write_text(json.dumps(release))
    with pytest.raises(RuntimeError,match='terminal'):c.validate_execution_release(path,root,now=1000.)


def test_incomplete_phase_cannot_be_silently_called_complete(tmp_path,monkeypatch):
    root,path,release=release_fixture(tmp_path,monkeypatch)
    phasepath=Path(release['phase_terminal_evidence']['scale']['path'])
    status=json.loads(phasepath.read_text());status.update(selected_phase_execution_complete=False,phase='stopped_by_deadline')
    phasepath.write_text(json.dumps(status));release['phase_terminal_evidence']['scale']['sha256']=c.sha(phasepath);path.write_text(json.dumps(release))
    with pytest.raises(RuntimeError,match='incomplete'):c.validate_execution_release(path,root,now=1000.)
    release.update(allow_incomplete_phase_termination=True,incomplete_phase_reason='phase deadline stopped, missing cells explicitly retained')
    path.write_text(json.dumps(release));result=c.validate_execution_release(path,root,now=1000.)
    assert not result['phase_terminal_evidence']['scale']['execution_complete']


def test_deadline_reserves_all_work_and_cleanup(tmp_path,monkeypatch):
    root,path,release=release_fixture(tmp_path,monkeypatch)
    with pytest.raises(RuntimeError,match='deadline'):c.validate_execution_release(path,root,now=1111.)


def test_parent_observed_free_capacity_is_historical_only():
    original=ROOT.parent/'B32B-batch-coverage-long4096-v1/identity.before.json'
    for item in json.loads(original.read_text())['instances']:
        state=item['runtime']
        proof=c.validate_runtime_capacity(state,now=state['timestamp'])
        assert proof['observed_free_kv_tokens']==47104
        with pytest.raises(RuntimeError,match='stale'):
            c.validate_runtime_capacity(state,now=state['timestamp']+10)
