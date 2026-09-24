"""Independent DistServe protocol replay rejects plausible but invalid evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from pdblend.bench.comparison_distserve_acceptance import _kv, audit_protocol
from pdblend.bench.comparison_distserve_inputs import validate_distserve_inputs
from test_comparison_acceptance import state


def transfer_receipts(generation=5,tp=1):
    identity=dict(target_request_id='target',transaction_id='tx',generation=generation)
    ranks=[dict(rank=r,generation=generation) for r in range(tp)]
    tx=[dict(row,**identity,acknowledged=True) for row in ranks]
    return [dict(step='prefill',receipt=dict(acknowledged=True,retained_handle='held',
                ranks=[dict(row,held_requests=['held']) for row in ranks],
                outputs=[dict(token_ids=[50],token_index=1,finished=True)])),
            dict(step='expect_load',receipt=dict(acknowledged=True,ranks=deepcopy(tx))),
            dict(step='transfer',receipt=dict(acknowledged=True,ranks=[dict(row,layers=28) for row in tx])),
            dict(step='load_ack',receipt=dict(acknowledged=True,generation=generation,
                transaction_id='tx',request_id='target',ranks=[dict(row,expected_layers=28,loaded_layers=28) for row in tx])),
            dict(step='release',receipt=dict(acknowledged=True,released=True,ranks=[dict(row,released=True) for row in ranks]))]


def test_complete_rank_transfer_and_explicit_failure_prefix():
    assert _kv(transfer_receipts(tp=2),tp=2,generation=5,successful=True)==[50]
    assert _kv(transfer_receipts()[:1],tp=1,generation=5,successful=False)==[50]


@pytest.mark.parametrize('bad', ['epoch','rank','tx','layer','release','missing','duplicate'])
def test_false_success_kv_is_rejected(bad):
    rows=transfer_receipts(tp=2)
    if bad=='epoch':rows[2]['receipt']['ranks'][0]['generation']=4
    if bad=='rank':rows[2]['receipt']['ranks'].pop()
    if bad=='tx':rows[3]['receipt']['ranks'][0]['transaction_id']='other'
    if bad=='layer':rows[3]['receipt']['ranks'][0]['loaded_layers']=27
    if bad=='release':rows[-1]['receipt']['ranks'][0]['released']=False
    if bad=='missing':rows.pop(1)
    if bad=='duplicate':rows.append(deepcopy(rows[-1]))
    with pytest.raises(ValueError):_kv(rows,tp=2,generation=5,successful=True)


def protocol_fixture():
    requests=[dict(idx=i,prompt=[1,2,3],max_tokens=2,arrival_s=0.) for i in range(2)]
    trace=dict(requests=requests);origin=100.;instances={};events=[];outcomes=[]
    for pair in range(2):
        caps={};states={}
        for role in ('P','D'):
            iid=f'dist-{pair}-{role}';instances[iid]=dict(instance_id=iid,tp=1,pp=1,gpu_uuids=[f'GPU-{pair}-{role}'])
            caps[role]=dict(gpu_uuids=instances[iid]['gpu_uuids'])
            states[role]=dict(state(1,99.,5),max_num_batched_tokens=8192)
        events.append(dict(event='distserve_runtime_started',replica=pair,at_s=99.1,capabilities=caps,states=states))
    for i,req in enumerate(requests):
        events.append(dict(event='distserve_request_queued',replica=i,at_s=100.001,
            request_id=f'distserve-701-{i}',payload=dict(req,seed=701,ignore_eos=True,temperature=0)))
    for i,req in enumerate(requests):
        rid=f'distserve-701-{i}';receipts=transfer_receipts()
        def emit(kind,**kwargs):events.append(dict(event=kind,replica=i,at_s=100.1,**kwargs))
        emit('distserve_prefill_admission',request_ids=[rid],free_kv_tokens=1024)
        emit('distserve_native_receipt',request_id=rid,**receipts[0])
        emit('distserve_bridge_ready',request_id=rid)
        emit('distserve_client_sse',request_id=rid,payload=dict(token_ids=[50],token_index=1,finished=False,received_s=100.1))
        emit('distserve_native_sse',request_id=rid,payload=dict(token_ids=[51],token_index=1,finished=True,received_s=100.14))
        for receipt in receipts[1:]:emit('distserve_native_receipt',request_id=rid,**receipt)
        emit('distserve_client_sse',request_id=rid,payload=dict(token_ids=[51],token_index=2,finished=True,received_s=100.15))
        emit('distserve_request_finished',request_id=rid,status='completed',generated=2,cleanup_errors=[])
        outcomes.append(dict(request_id=rid,replica=i,arrival_s=0.,input_tokens=3,output_tokens=2,ok=True,
            finished_s=100.2,completion_tokens=2,event_count=2,token_ids_sha256=hashlib.sha256(b'[50,51]').hexdigest(),
            result=dict(status='completed',tokens=2,receipts=receipts)))
    for i in range(2):events.append(dict(event='distserve_runtime_closed',replica=i,at_s=250.1,
        quarantined=[],states={role:state(1,250.1,5) for role in ('P','D')}))
    return dict(trace=trace,native=dict(outcomes=outcomes),events=events,choice=dict(selected=dict(replicas=2,tp=1)),
                instances=instances,reset=dict(generation={i:5 for i in instances}),origin=origin)


def test_independent_prefill_queue_routing_and_carry():
    assert audit_protocol(**protocol_fixture())


def single_token_protocol_fixture():
    args=protocol_fixture();rid='distserve-701-0'
    args['trace']['requests'][0]['max_tokens']=1
    events=[]
    for event in args['events']:
        if event.get('request_id')==rid:
            kind=event['event']
            if kind=='distserve_request_queued':event['payload']['max_tokens']=1
            if kind=='distserve_native_sse':continue
            if kind=='distserve_native_receipt' and event['step'] not in ('prefill','release'):continue
            if kind=='distserve_client_sse':
                if event['payload']['token_index']!=1:continue
                event['payload']['finished']=True
            if kind=='distserve_request_finished':event['generated']=1
        events.append(event)
    args['events']=events
    row=args['native']['outcomes'][0]
    row.update(output_tokens=1,completion_tokens=1,event_count=1,
               token_ids_sha256=hashlib.sha256(b'[50]').hexdigest())
    row['result'].update(tokens=1,receipts=[r for r in row['result']['receipts'] if r['step'] in ('prefill','release')])
    return args


def test_single_token_prefill_release_needs_no_decode_stream():
    assert audit_protocol(**single_token_protocol_fixture())


@pytest.mark.parametrize('bad',['decode','missing_release','unclosed_client'])
def test_single_token_still_requires_own_complete_execution(bad):
    args=single_token_protocol_fixture();rid='distserve-701-0'
    if bad=='decode':
        args['events'].insert(-2,dict(event='distserve_native_sse',replica=0,at_s=100.2,request_id=rid,
            payload=dict(token_ids=[51],token_index=1,finished=True,received_s=100.2)))
    elif bad=='missing_release':
        args['events']=[e for e in args['events'] if not(e.get('request_id')==rid and e.get('step')=='release')]
    else:
        next(e for e in args['events'] if e.get('request_id')==rid and e['event']=='distserve_client_sse')['payload']['finished']=False
    with pytest.raises(ValueError):audit_protocol(**args)


@pytest.mark.parametrize('bad',['route','fcfs','prompt','carry','kv','tail','token_hash'])
def test_protocol_rejects_route_request_kv_or_clock_substitution(bad):
    args=protocol_fixture();events=args['events']
    if bad=='route':events[3]['replica']=0
    if bad=='fcfs':next(e for e in events if e['event']=='distserve_prefill_admission')['request_ids']=['distserve-701-1']
    if bad=='prompt':events[2]['payload']['prompt']=[4,5,6]
    if bad=='carry':next(e for e in events if e['event']=='distserve_client_sse')['payload']['token_ids']=[70]
    if bad=='kv':next(e for e in events if e['event']=='distserve_native_receipt' and e['step']=='load_ack')['receipt']['ranks'][0]['loaded_layers']=0
    if bad=='tail':events[-1]['at_s']=249.
    if bad=='token_hash':args['native']['outcomes'][0]['token_ids_sha256']='wrong'
    with pytest.raises((ValueError,KeyError)):audit_protocol(**args)


def test_missing_profile_and_choice_never_open_formal_gate():
    result=validate_distserve_inputs(dict(system='distserve',inputs={}),{},source_manifest=None)
    assert not result['preflight_ready'] and not result['formal_eligible']
    assert 'profile.raw_holdout_audit' in result['gate_failures']


@pytest.mark.parametrize('kind', ['distserve_native_receipt', 'distserve_native_sse',
                                 'distserve_client_sse', 'distserve_request_finished'])
def test_request_owned_events_cannot_move_to_another_pair(kind):
    args=protocol_fixture()
    event=next(e for e in args['events'] if e['event']==kind)
    event['replica']=1-event['replica']
    with pytest.raises(ValueError,match='request pair'):
        audit_protocol(**args)


@pytest.mark.parametrize('indices', [(1,), (0,2), (0,2,1)])
def test_failed_request_still_requires_an_actual_kv_prefix(indices):
    original=transfer_receipts()
    with pytest.raises(ValueError,match='prefix'):
        _kv([original[i] for i in indices],tp=1,generation=5,successful=False)


def pair_probe_fixture(tmp_path):
    from pdblend_baselines.distserve.gpu_probe import trace_rows
    from pdblend.bench.comparison_distserve_acceptance import binding
    point=dict(model_id='Qwen2.5-7B-Instruct')
    identity=dict(model_hash='weights',tokenizer_hash='tokenizer',image_digest='image')
    caps={role:dict(identity,model_id=point['model_id'],tp=1,pp=1,gpu_uuids=['GPU-'+role]) for role in ('P','D')}
    instances={f'dist-0-{role}':dict(tp=1,gpu_uuids=cap['gpu_uuids']) for role,cap in caps.items()}
    trace=dict(seed=701,model_id=point['model_id'],tokenizer_hash=identity['tokenizer_hash'],requests=trace_rows())
    events=[dict(event='distserve_runtime_started',at_s=80.,capabilities=caps,
                 states={role:state(1,79.99,5) for role in ('P','D')})]
    result=dict(system='distserve',status='passed',complete=True,seed=701,tp=1,pp=1,
        cleanup_errors=[],quarantined_roles=[],capabilities=caps,outcomes={},references={},
        concurrent_requests_observed=True)
    def output(rid,count,stamp):
        return [dict(request_id=rid,token_ids=[50+i],token_index=i+1,finished=i==count-1,
                     at_s=stamp+i*.01,received_s=stamp+i*.01) for i in range(count)]
    for index,row in enumerate(trace['requests'][:3]):
        rid=row['request_id'];stream=output('distserve-reference-'+rid,16,90.+index)
        result['references'][rid]=dict(events=stream,token_ids=list(range(50,66)))
    result['reference_repeat']=dict(events=output('distserve-reference-repeat',16,94.),token_ids=list(range(50,66)))
    for row,start in zip(trace['requests'],(100.,100.05,102.,104.,106.,108.)):
        rid=row['request_id'];cancel=row['phase']=='cancel';n=2 if cancel else row['max_tokens']
        stream=output(rid,n,start+.1)
        if cancel:stream[-1]['finished']=False
        native_receipts=transfer_receipts()
        if n==1:native_receipts=[native_receipts[0],native_receipts[-1]]
        if cancel:native_receipts=native_receipts[:3]+[
            dict(step='cancel_'+role,receipt=dict(acknowledged=True,generation=5)) for role in ('D','P')]
        events.append(dict(event='distserve_request_queued',request_id=rid,at_s=start,payload=row,
                           input_tokens=len(row['prompt']),output_tokens=row['max_tokens']))
        for index,receipt in enumerate(native_receipts):
            events.append(dict(event='distserve_native_receipt',request_id=rid,at_s=start+.01*(index+1),**receipt))
        for value in stream:events.append(dict(event='distserve_client_sse',request_id=rid,at_s=value['received_s'],payload=value))
        status='cancelled' if cancel else 'completed'
        events.append(dict(event='distserve_request_finished',request_id=rid,at_s=start+1.,
                           status=status,generated=n,cleanup_errors=[]))
        native=dict(request_id=rid,status=status,tokens=n,token_ids=list(range(50,50+n)),
                    receipts=native_receipts,events=stream)
        if cancel:
            before=state(1,106.5,5);before.update(all_queue=['target'],kv_allocations={'target':[[1]]})
            result['cancel']=dict(before=before,target_request_id='target',events=stream,result=native,
                acknowledgement=dict(acknowledged=True,request_id=rid))
        else:result['outcomes'][rid]=dict(submitted_s=start-.001,finished_s=start+1.001,
                result=native,events=stream,ok=True,golden_match=True)
    states={role:state(1,107.49,5) for role in ('P','D')}
    result['recovery']=dict(acknowledged=True,states=states)
    events.append(dict(event='distserve_recovered',at_s=107.5,states=states))
    events.sort(key=lambda row:row['at_s'])
    result['drain']=[dict(state(1,110.,5),acknowledged=True,drained=True) for _ in ('P','D')]
    def save():
        paths={name:tmp_path/name for name in ('runtime.json','trace.json','events.jsonl')}
        paths['trace.json'].write_text(json.dumps(trace))
        paths['events.jsonl'].write_text(''.join(json.dumps(row)+'\n' for row in events))
        result.update(trace_sha256=binding(paths['trace.json'])['sha256'],events_sha256=binding(paths['events.jsonl'])['sha256'])
        paths['runtime.json'].write_text(json.dumps(result))
        startup=dict(distserve_pair_probes=[dict(replica=0,result=binding(paths['runtime.json']),
            events=binding(paths['events.jsonl']),trace=binding(paths['trace.json']))])
        return point,identity,startup,instances
    return trace,result,events,save


def test_all_six_probe_cases_bind_raw_repeated_goldens_and_concurrency(tmp_path):
    from pdblend.bench.comparison_distserve_acceptance import audit_pair_probes
    *_,save=pair_probe_fixture(tmp_path)
    audit_pair_probes(*save())


@pytest.mark.parametrize('bad', ['missing_long','missing_single','missing_recovery','changed_prompt',
    'missing_reference','missing_repeat','different_repeat','copied_repeat','missing_repeat_terminal',
    'false_overlap','missing_concurrent_flag','unexecuted_case','missing_case_outcome',
    'single_token_no_release','missing_recovery_event','recovery_before_cancel','missing_P_capability',
    'cancel_stream_mismatch'])
def test_probe_subset_or_summary_claims_cannot_qualify(tmp_path,bad):
    from pdblend.bench.comparison_distserve_acceptance import audit_pair_probes
    trace,result,events,save=pair_probe_fixture(tmp_path)
    if bad.startswith('missing_') and bad.split('_',1)[1] in ('long','single','recovery'):
        phase={'long':'long','single':'single_token','recovery':'recovery'}[bad.split('_',1)[1]]
        trace['requests']=[row for row in trace['requests'] if row['phase']!=phase]
    elif bad=='changed_prompt':trace['requests'][0]['prompt'][0]+=1
    elif bad=='missing_reference':result['references'].pop('distserve-701-7168')
    elif bad=='missing_repeat':result.pop('reference_repeat')
    elif bad=='different_repeat':result['reference_repeat']['token_ids'][0]=999
    elif bad=='copied_repeat':result['reference_repeat']=deepcopy(result['references']['distserve-701-512'])
    elif bad=='missing_repeat_terminal':result['reference_repeat']['events'][-1]['finished']=False
    elif bad=='false_overlap':
        # Both result timestamps and its flag still claim overlap; raw queue
        # time exposes sequential execution of the purported concurrent pair.
        for event in events:
            if event.get('request_id')=='distserve-701-2048':
                event['at_s']+=2.
                if event['event']=='distserve_client_sse':
                    event['payload']['at_s']+=2.;event['payload']['received_s']+=2.
        result['outcomes']['distserve-701-2048']['finished_s']+=2.
        events.sort(key=lambda e:e['at_s'])
    elif bad=='missing_concurrent_flag':result.pop('concurrent_requests_observed')
    elif bad=='unexecuted_case':events[:]=[e for e in events if e.get('request_id')!='distserve-701-one']
    elif bad=='missing_case_outcome':result['outcomes'].pop('distserve-701-7168')
    elif bad=='single_token_no_release':
        result['outcomes']['distserve-701-one']['result']['receipts'].pop()
        events[:]=[e for e in events if not(e.get('request_id')=='distserve-701-one' and e.get('step')=='release')]
    elif bad=='missing_recovery_event':events[:]=[e for e in events if e['event']!='distserve_recovered']
    elif bad=='recovery_before_cancel':
        next(e for e in events if e['event']=='distserve_recovered')['at_s']=106.9
        events.sort(key=lambda e:e['at_s'])
    elif bad=='missing_P_capability':result['capabilities'].pop('P')
    elif bad=='cancel_stream_mismatch':result['cancel']['result']['token_ids'][1]=999
    with pytest.raises(ValueError,match='did not actually overlap' if bad=='false_overlap' else None):
        audit_pair_probes(*save())


@pytest.mark.asyncio
async def test_actual_run_resident_probe_schema_accepts_full_native_http_evidence(monkeypatch,tmp_path):
    """Exercise the real probe/runtime/HTTP routes; only GPU execution is fake."""
    import asyncio
    import importlib.util
    from pathlib import Path
    pytest.importorskip('fastapi')
    spec=importlib.util.spec_from_file_location('comparison_distserve_http_fixture',
        Path(__file__).parents[1]/'independent_baselines/test_distserve_request_runtime_http.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from pdblend_baselines.distserve.gpu_probe import run_resident
    from pdblend.bench.comparison_distserve_acceptance import audit_pair_probes, binding
    async with fixture.services(monkeypatch) as (engines,transport):
        original=engines['D'].generate
        async def model(prompt,params,rid):
            async for value in original(prompt,params,rid):
                if rid.startswith('distserve-reference-'):
                    for choice in value.outputs:choice.token_ids=[token-1 for token in choice.token_ids]
                yield value
        engines['D'].generate=model
        out=tmp_path/'probe'
        result=await asyncio.wait_for(run_resident(transport.prefill_url,transport.decode_url,
            transport.prefill_address,transport.decode_address,1,out),20)
        assert result['complete'],result
    cap=result['capabilities']['P']
    point={'model_id':cap['model_id']}
    identity={k:cap[k] for k in ('model_hash','tokenizer_hash','image_digest')}
    instances={f'dist-0-{role}':dict(tp=1,gpu_uuids=row['gpu_uuids'])
               for role,row in result['capabilities'].items()}
    startup=dict(distserve_pair_probes=[dict(replica=0,result=binding(out/'runtime.json'),
        trace=binding(out/'trace.json'),events=binding(out/'events.jsonl'))])
    audit_pair_probes(point,identity,startup,instances)
