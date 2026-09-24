"""Acceptance must derive claims from bound evidence, not successful flags."""
from copy import deepcopy
import hashlib
import json

import pytest

from pdblend.bench.comparison_acceptance import audit_window
from pdblend.bench.comparison_metering import summarize_comparison
from pdblend.bench.comparison_metrics import canonical_outcomes, reduce_comparison
from pdblend.bench.resident_session import digest, engine_signature
from pdblend.measure.backends import INSTANT_POWER_SOURCE_ID


def state(tp, at, generation=4):
    return dict(tp=tp, pp=1, generation=generation, acknowledged_generation=generation,
        max_num_seqs=32, max_model_len=8192, healthy=True, transport_healthy=True, native_evidence_complete=True, block_size=16,
        native_at_s=at-.01, response_at_s=at, rank_observation_started_s=at-.02,
        rank_observation_finished_s=at-.01, all_queue=[], running=[], waiting=[],
        kv_allocations={}, retained_kv_requests=[], transfer_allocations={}, pending_transfers=0,
        free_kv_tokens=1024, total_kv_tokens=1024, accepting=True, admit_prefill=True,
        admit_decode=True, role='mixed', mode='temporal',
        ranks=[dict(rank=i, generation=generation, healthy=True, native_evidence_complete=True,
            at_s=at-.015, pending_transfers=0, transfer_allocations={}, retained=dict(
                held_count=0, held_requests=[], held_bytes=0, receiving_transactions=0)) for i in range(tp)])


def drain_rows(instances, at, generation=4):
    return [dict(instance_id=i['instance_id'], received_s=at, state=state(i['tp'],at,generation),
        drain=dict(state(i['tp'],at,generation), acknowledged=True, drained=True)) for i in instances]


def put(root, name, value, *, journal=False):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in value) if journal else json.dumps(value))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(tmp_path, *, model='Qwen2.5-7B-Instruct', slow=False):
    uuids = [f'GPU-test-{i}' for i in range(8)]; tp=2 if '32B' in model else 1
    runtime={'pdblend_runtime/serve.py':hashlib.sha256(b'fixture runtime').hexdigest()}
    measurement={'pdblend/bench/comparison_metrics.py':hashlib.sha256(b'fixture metrics').hexdigest()}
    for name,contents in [('pdblend_runtime/serve.py','fixture runtime'),
                          ('pdblend/bench/comparison_metrics.py','fixture metrics')]:
        path=tmp_path/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(contents)
    files=runtime|measurement
    source=put(tmp_path,'source/manifest.json',dict(files=files,source_sha256=digest(files)))
    env=dict(VLLM_USE_V1='1')
    instances=[dict(instance_id=f'mixed{i}',tp=tp,pp=1,gpu_uuids=uuids[i*tp:(i+1)*tp],
        launch_options=dict(max_model_len=8192,max_num_seqs=32,kv_connector=None)) for i in range(8//tp)]
    identity=dict(model_hash='model-hash',tokenizer_hash='tokenizer-hash',image_digest='sha256:image',
        runtime_source_sha256=digest(runtime),measurement_source_sha256=digest(measurement),
        dtype='bfloat16',entrypoint='pdblend_runtime.serve',worker_extension='native_v1',
        fleet_gpu_uuids=uuids,instances=instances,environment=env)
    point=dict(model_id=model,system='mixed',dataset='alpaca',rate_rps=.01,
               slo=dict(ttft_s=1.,tpot_s=.1),engine_identity=identity)
    trace=dict(seed=701,duration_s=150,model_id=model,dataset='alpaca',rate_rps=.01,slo=point['slo'],
               requests=[dict(idx=0,arrival_s=0.,prompt=[1,2,3],max_tokens=2)])
    trace_ref=put(tmp_path,'trace.json',trace);point['trace']=trace_ref
    lease=put(tmp_path,'lease.json',dict(gpu_uuids=uuids,payload=dict(exclusive=True,reserve_host=True,gpu_count=8)))
    concurrency=put(tmp_path,'concurrency.json',dict(allocated_gpu_uuids=uuids,physical_gpu_uuids=uuids,
        lease_manifest_sha256=lease['sha256'],peer_jobs=[],peer_snapshots=[dict(at_s=90.,peers=[])]))
    launches=[];caps={};refs=[]
    for index,i in enumerate(instances):
        launches.append(dict(instance_id=i['instance_id'],gpu_uuids=i['gpu_uuids'],
            environment=env|dict(CUDA_VISIBLE_DEVICES=','.join(str(index*tp+j) for j in range(tp))),
            argv=['python','-B','-m','pdblend_runtime.serve','/models/'+model,'--tensor-parallel-size',str(tp),
                  '--pipeline-parallel-size','1','--max-num-seqs','32','--max-model-len','8192',
                  '--dtype','bfloat16','--no-enable-prefix-caching']))
        caps[i['instance_id']]=dict(supported=True,tp=tp,pp=1,model_id=model,
            engine_revision='vllm-0.10.1.1',gpu_uuids=i['gpu_uuids'],source_revision=digest(files),
            **{k:identity[k] for k in ('model_hash','tokenizer_hash','image_digest')},state=state(tp,91.))
        response=dict(token_ids=list(range(16)),events=[dict(token_ids=list(range(16)),finished=True)])
        refs.append(dict(instance_id=i['instance_id'],responses=[response,deepcopy(response)]))
    startup=dict(engine_signature=engine_signature(identity),exclusive_gpu_uuids=uuids,
        model_hash=identity['model_hash'],tokenizer_hash=identity['tokenizer_hash'],source_manifest=source,
        source_fingerprints=dict(runtime_files=runtime,measurement_files=measurement),
        actual_launch_identity=dict(instances=launches,source_revision=digest(files),
            **{k:identity[k] for k in ('image_digest','runtime_source_sha256','measurement_source_sha256')}),
        capabilities=caps,ordinary_reference=refs,drain=drain_rows(instances,92.),
        concurrency_environment=concurrency,lease_manifest=lease)
    reset=dict(initial_clock_mhz=2520,reset_s=2.,drain=drain_rows(instances,98.),
        generation={i['instance_id']:5 for i in instances},
        reopen_ack={i['instance_id']:dict(acknowledged=True,generation=5) for i in instances},
        reopen_state={i['instance_id']:state(tp,99.,5) for i in instances})
    first=102. if slow else 100.1;last=first+.05
    outcomes=[dict(idx=0,request_id='mixed-701-0',arrival_s=0.,scheduled_s=100.,submitted_s=100.001,
        input_tokens=3,max_tokens=2,completion_tokens=2,sampling_seed=701,correct=True,
        terminal=True,finished_s=last,instance_id='mixed0')]
    events=[dict(event='route',at_s=100.001,request_id='mixed-701-0',instance_id='mixed0',
                policy='fixed_tp_least_load',tp=tp,load=0.),
        dict(kind='mixed_client_sse',at_s=first,request_id='mixed-701-0',
             payload=dict(received_s=first,token_ids=[5],finished=False)),
        dict(kind='mixed_client_sse',at_s=last,request_id='mixed-701-0',
             payload=dict(received_s=last,token_ids=[6],finished=True)),
        dict(event='release',at_s=last+.001,request_id='mixed-701-0',active={i['instance_id']:0 for i in instances})]
    raw=dict(system='mixed',native_runner=True,seed=701,duration_s=150,started_s=100.,finished_s=250.,
        routing_policy='independent_least_load_fixed_tp',counts_reclaimed=True)
    rows=canonical_outcomes('mixed',trace,outcomes,service_started_s=100.,journal=events)
    metrics=reduce_comparison(trace,rows,service_started_s=100.,slo=(1.,.1))
    times=[99.5+i*.5 for i in range(303)]
    source=dict(mode='instant',source_id=INSTANT_POWER_SOURCE_ID,field_id=186,scope_id=0)
    power=dict(gpus=list(range(8)),gpu_uuids=uuids,gpu_uuid_binding_verified=True,
        samples=[(t,[100.]*8) for t in times],power_source=source,
        power_metadata=[dict(gpus=list(range(8)),read_finished_s=[t]*8,
            **{k:[v]*8 for k,v in dict(source,return_code=0,value_type=1).items()}) for t in times],
        utilization_readings=[dict(gpu=g,gpu_util_pct=20.,read_finished_s=t,error=None) for t in times for g in range(8)])
    meter=summarize_comparison(power,gpu_uuids=uuids,origin_s=100.,tail_end_s=250.1)
    drain=dict(states=drain_rows(instances,250.05,5),tail_end_s=250.1)
    refs=dict(trace=trace_ref,lease_manifest=lease,concurrency_environment=concurrency)
    for key,value in [('events',events),('outcomes',outcomes)]:
        refs[key]=put(tmp_path,key+'.jsonl',value,journal=True);raw[key+'_sha256']=refs[key]['sha256']
    values=dict(requests=dict(seed=701,duration_s=150,requests=[dict(trace['requests'][0],source='alpaca')]),
        power=power,native_result=raw,canonical_requests=metrics['request_metrics'],metering=meter,
        startup_qualification=startup,reset=reset,drain=drain)
    for key,value in values.items():refs[key]=put(tmp_path,key+'.json',value)
    return dict(point=point,engine_identity=identity,startup_qualification=startup,reset=reset,
                native_result=raw,canonical_metrics=metrics,metering=meter,drain=drain,raw_refs=refs)


def save_change(args, key, value):
    ref=args['raw_refs'][key];path=__import__('pathlib').Path(ref['path'])
    path.write_text(json.dumps(value));ref['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('model',['Qwen2.5-7B-Instruct','Qwen2.5-14B-Instruct','Qwen2.5-32B-Instruct'])
def test_all_raw_gates_qualify_one_observation_without_profile(tmp_path,model):
    result=audit_window(**fixture(tmp_path,model=model))
    assert result['missing_gates']==[],result['gate_failures']
    assert result['evidence_valid'] and result['formal_eligible'] and result['slo_pass']
    assert result['optimality_established'] is False


def test_slo_failure_is_valid_frozen_observation(tmp_path):
    result=audit_window(**fixture(tmp_path,slow=True))
    assert result['formal_eligible'],result['gate_failures']
    assert result['slo_pass'] is False


@pytest.mark.parametrize('key',['events','outcomes','power','canonical_requests','startup_qualification','drain'])
def test_missing_raw_evidence_never_inherits_passed_flags(tmp_path,key):
    args=fixture(tmp_path);del args['raw_refs'][key]
    args['canonical_metrics']['formal_eligible']=True
    result=audit_window(**args)
    assert not result['evidence_valid'] and not result['formal_eligible']
    assert 'raw.'+key in result['missing_gates']


def test_raw_sha_mismatch_invalidates_even_when_metrics_match(tmp_path):
    args=fixture(tmp_path);args['raw_refs']['events']['sha256']='0'*64
    result=audit_window(**args)
    assert not result['formal_eligible'] and 'raw.events' in result['missing_gates']


def test_metric_claim_cannot_replace_raw_token_timing(tmp_path):
    args=fixture(tmp_path);args['canonical_metrics']['goodput_token_s']=9999.
    result=audit_window(**args)
    assert 'metrics.client_canonical' in result['missing_gates']


def test_flags_do_not_replace_native_rank_cleanup(tmp_path):
    args=fixture(tmp_path);args['drain']['states'][0]['state']['ranks'][0]['retained']['held_count']=1
    save_change(args,'drain',args['drain'])
    result=audit_window(**args)
    assert 'mixed.full_native_drain' in result['missing_gates']


def test_wrong_actual_cuda_mapping_fails_startup(tmp_path):
    args=fixture(tmp_path)
    args['startup_qualification']['actual_launch_identity']['instances'][0]['environment']['CUDA_VISIBLE_DEVICES']='7'
    save_change(args,'startup_qualification',args['startup_qualification'])
    result=audit_window(**args)
    assert 'mixed.startup_identity' in result['missing_gates']


def test_launch_flag_override_or_kv_transfer_invalidates(tmp_path):
    args=fixture(tmp_path)
    args['startup_qualification']['actual_launch_identity']['instances'][0]['argv']+=['--kv-transfer-config','{}']
    save_change(args,'startup_qualification',args['startup_qualification'])
    assert 'mixed.startup_identity' in audit_window(**args)['missing_gates']


def test_stale_reopen_state_cannot_be_formal(tmp_path):
    args=fixture(tmp_path);args['reset']['reopen_state']['mixed0']['native_at_s']=91.
    save_change(args,'reset',args['reset'])
    assert 'mixed.reset' in audit_window(**args)['missing_gates']


def test_seven_gpu_fleet_fails_even_with_eight_power_rows(tmp_path):
    args=fixture(tmp_path);args['engine_identity']['fleet_gpu_uuids'].pop()
    assert 'mixed.fixed_tp_fleet' in audit_window(**args)['missing_gates']


def test_tampered_energy_summary_rejected(tmp_path):
    args=fixture(tmp_path);args['metering']['energy_service_j']=1.
    save_change(args,'metering',args['metering'])
    assert 'metering.raw_eight_gpu_window' in audit_window(**args)['missing_gates']


def test_missing_reference_not_replaced_by_functional_passed(tmp_path):
    args=fixture(tmp_path);args['startup_qualification']['ordinary_reference']=[]
    args['startup_qualification']['passed']=True
    save_change(args,'startup_qualification',args['startup_qualification'])
    assert 'mixed.startup_identity' in audit_window(**args)['missing_gates']


def test_actual_no_connector_native_rank_schema_is_accepted(tmp_path):
    args=fixture(tmp_path)
    # A no-KV native worker has explicit support=False and no retained object.
    def walk(value):
        if isinstance(value,dict):
            if 'retained' in value:
                value.pop('retained');value['retained_kv_supported']=False
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    for key in ('startup_qualification','reset','drain'):
        walk(args[key]);save_change(args,key,args[key])
    result=audit_window(**args)
    assert result['formal_eligible'],result['gate_failures']


def test_later_equals_flag_override_is_rejected(tmp_path):
    args=fixture(tmp_path)
    args['startup_qualification']['actual_launch_identity']['instances'][0]['argv']+=['--max-num-seqs=64']
    save_change(args,'startup_qualification',args['startup_qualification'])
    assert 'mixed.startup_identity' in audit_window(**args)['missing_gates']


def test_native_state_actual_capacity_is_checked(tmp_path):
    args=fixture(tmp_path);args['startup_qualification']['capabilities']['mixed0']['state']['max_num_seqs']=256
    save_change(args,'startup_qualification',args['startup_qualification'])
    assert 'mixed.startup_identity' in audit_window(**args)['missing_gates']


def replace_journal(args,key,rows):
    path=__import__('pathlib').Path(args['raw_refs'][key]['path'])
    args['raw_refs'][key]=put(path.parent,path.name,rows,journal=True)
    args['native_result'][key+'_sha256']=args['raw_refs'][key]['sha256']
    save_change(args,'native_result',args['native_result'])


def test_success_flag_cannot_replace_complete_client_token_journal(tmp_path):
    args=fixture(tmp_path)
    path=__import__('pathlib').Path(args['raw_refs']['events']['path'])
    events=[json.loads(line) for line in path.read_text().splitlines()]
    events.pop(1)
    replace_journal(args,'events',events)
    result=audit_window(**args)
    assert 'metrics.client_canonical' in result['missing_gates']


def test_independent_least_load_replay_detects_wrong_replica(tmp_path):
    args=fixture(tmp_path)
    path=__import__('pathlib').Path(args['raw_refs']['events']['path'])
    events=[json.loads(line) for line in path.read_text().splitlines()]
    events[0]['instance_id']='mixed1'
    replace_journal(args,'events',events)
    assert 'mixed.least_load_release' in audit_window(**args)['missing_gates']


def test_counts_reclaimed_flag_cannot_replace_release(tmp_path):
    args=fixture(tmp_path)
    path=__import__('pathlib').Path(args['raw_refs']['events']['path'])
    events=[json.loads(line) for line in path.read_text().splitlines()]
    replace_journal(args,'events',events[:-1])
    assert 'mixed.least_load_release' in audit_window(**args)['missing_gates']


def test_missing_terminal_outcome_is_not_removed_from_denominator(tmp_path):
    args=fixture(tmp_path);replace_journal(args,'outcomes',[])
    assert 'metrics.client_canonical' in audit_window(**args)['missing_gates']


def test_raw_power_gap_cannot_be_overridden_by_energy_comparable_true(tmp_path):
    args=fixture(tmp_path)
    path=__import__('pathlib').Path(args['raw_refs']['power']['path'])
    power=json.loads(path.read_text())
    del power['samples'][10:14];del power['power_metadata'][10:14]
    save_change(args,'power',power)
    assert args['metering']['energy_comparable']
    assert 'metering.raw_eight_gpu_window' in audit_window(**args)['missing_gates']


def test_actual_prompt_tamper_is_rejected(tmp_path):
    args=fixture(tmp_path)
    path=__import__('pathlib').Path(args['raw_refs']['requests']['path'])
    requests=json.loads(path.read_text());requests['requests'][0]['prompt']=[10,20,30]
    save_change(args,'requests',requests)
    assert 'trace.executed_request_identity' in audit_window(**args)['missing_gates']


def test_stale_observation_audits_receipt_freshness_not_current_clock(tmp_path,monkeypatch):
    args=fixture(tmp_path)
    monkeypatch.setattr('time.time',lambda: 1e12)
    assert audit_window(**args)['formal_eligible']
