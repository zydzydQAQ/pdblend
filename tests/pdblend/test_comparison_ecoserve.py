"""Small raw-receipt fixtures; no engine launches, GPU reads or archived journals."""
import hashlib
import json
from pathlib import Path
from copy import deepcopy

import pytest

from pdblend.bench import comparison_ecoserve_inputs as inputs_module
from pdblend.bench.comparison_ecoserve_inputs import validate_ecoserve_inputs
from pdblend.bench.comparison_ecoserve_acceptance import audit_ecoserve_window
from pdblend.bench.comparison_ecoserve_acceptance import _isolated_meter_method
from pdblend.bench.comparison_metrics import canonical_outcomes, reduce_comparison
from pdblend.bench.resident_session import digest, engine_signature
from pdblend_baselines.native_profile import ECO_LENGTHS, _sha
# Reuse only pure fixture construction; no other test suite is collected.
from test_comparison_acceptance import fixture as mixed_fixture, put, save_change, state


def eco_fixture(tmp_path,monkeypatch,*,slow=False):
    args=mixed_fixture(tmp_path,slow=slow);point=args['point'];identity=args['engine_identity']
    point.update(system='ecoserve',seed=701,duration_s=150)
    for item in identity['instances']:
        item['launch_options'].update(kv_connector='P2pNcclConnector',max_num_batched_tokens=8192)
    startup=args['startup_qualification']
    startup['engine_signature']=engine_signature(identity)
    for launch in startup['actual_launch_identity']['instances']:
        launch['argv']+=['--max-num-batched-tokens','8192','--kv-transfer-config',json.dumps(dict(kv_connector='P2pNcclConnector',kv_role='kv_both'))]
    for cap in startup['capabilities'].values():
        for rank in cap['state']['ranks']:rank['retained_kv_supported']=True
    source=Path(startup['source_manifest']['path']);manifest=json.loads(source.read_text())
    # Fixture source continuity is independently exercised below by tampering.
    protected='pdblend_runtime/serve.py'
    monkeypatch.setattr(inputs_module,'PROTECTED',(protected,))
    monkeypatch.setattr(inputs_module,'REVIEWED_WRAPPERS',{})
    monkeypatch.setattr(inputs_module,'PROFILE_SOURCE',manifest['source_sha256'])
    monkeypatch.setattr(inputs_module,'MECHANISM_SOURCE',manifest['source_sha256'])
    profile_csv=tmp_path/'eco.csv'
    profile_csv.write_text('Length,Prefill Time\n'+''.join(f'{n},{n/8}\n' for n in ECO_LENGTHS))
    profile_csv_ref=dict(path=str(profile_csv),sha256=hashlib.sha256(profile_csv.read_bytes()).hexdigest())
    rows=[]
    for n in ECO_LENGTHS:
        sample=dict(ranks=[dict(rank=0,samples=[dict(role='prefill',input_tokens=n,context_tokens=n,batch=1,
                    gpu_elapsed_ms=n/8,measurement_scope='forward',request_ids=['p'])])])
        rows.append(dict(input_tokens=n,minimum_ms=n/8,repetitions=[sample]*5,samples_sha256=[_sha(sample)]*5))
    meta=dict(tp=1,pp=1,frequency_mhz=2520,engine_version='vllm-0.10.1.1',source_revision=manifest['source_sha256'],
        gpu_uuids=[identity['fleet_gpu_uuids'][0]],**{k:identity[k] for k in ('model_hash','tokenizer_hash','image_digest')})
    profile=dict(schema='pdblend-baseline-profile-v1',system='ecoserve',model=point['model_id'],complete=True,
                 formal_eligible=False,metadata=meta,rows=rows)
    sidecar=put(tmp_path,'eco.csv.manifest.json',profile)
    checks={key:True for key in ('automatic_split','automatic_merge','split_live_kv_ack','merge_live_kv_ack',
        'automatic_park','policy_rotation','actual_hold','held_output_flush','continuous_complete_output',
        'no_manual_membership','all_native_drains_acknowledged','controller_healthy','no_execution_or_cleanup_error')}
    auto=put(tmp_path,'auto.json',dict(status='passed',checks=checks,formal_eligible=False))
    outer=put(tmp_path,'mechanism.json',dict(status='passed',checks=checks,formal_eligible=False,
        automatic_receipt_sha256=auto['sha256'],capabilities=startup['capabilities']))
    review=put(tmp_path,'review.json',dict(formal_eligible=False,reports=[dict(sha256=auto['sha256'],
        status='mechanism_passed',checks=dict(raw=True,original_periods=True))]))
    config=dict(system='ecoserve',model_id=point['model_id'],instances=[dict(id=i['instance_id'],gpus=[n],tp=1,pp=1)
        for n,i in enumerate(identity['instances'])],eco_prefill_csv=str(profile_csv),eco_profile_sha256=profile_csv_ref['sha256'],
        eco_macro_lower=2,eco_macro_upper=3,eco_initial_instances=2,eco_scale_period_s=5.,eco_history_window_s=60.,
        eco_state_poll_s=.05,eco_active_frequency_mhz=2520,slo_ttft_s=1.,slo_tpot_s=.1,
        request_timeout_s=180.,eco_drain_timeout_s=120.)
    trace=json.loads(Path(point['trace']['path']).read_text());trace['selection_split']='evaluation'
    trace_ref=put(tmp_path,'trace.json',trace);point['trace']=trace_ref;args['raw_refs']['trace']=trace_ref
    inputs=dict(trace=trace_ref,system_config=put(tmp_path,'config.json',config),eco_profile_csv=profile_csv_ref,
        eco_profile_manifest=sidecar,eco_mechanism_completion=outer,eco_automatic_completion=auto,
        eco_mechanism_review=review,eco_profile_source_manifest=startup['source_manifest'],
        eco_mechanism_source_manifest=startup['source_manifest'],source_manifest=startup['source_manifest'])
    catalog={point['model_id']:{key:inputs[key]['sha256'] for key in ('eco_profile_csv','eco_profile_manifest',
             'eco_mechanism_completion','eco_automatic_completion','eco_mechanism_review')}}
    monkeypatch.setattr(inputs_module,'CERTIFIED',catalog);point['inputs']=inputs
    origin=100.;first=102. if slow else 100.1;last=first+.05
    events=[]
    for index,i in enumerate(identity['instances']):
        active=index<2;path='/baseline/clock' if active else '/baseline/park'
        reply=dict(acknowledged=True,success=True,gpus=[dict(gpu_uuid=i['gpu_uuids'][0],frequency_mhz=2520 if active else 900)])
        if active:reply['requested_frequency_mhz']=2520
        events.append(dict(kind='eco_http_receipt',instance_id=i['instance_id'],method='POST',path=path,
            at_s=99.01+index*.01,body=dict(frequency_mhz=2520) if active else {},response=reply))
    events += [dict(kind='eco_startup',groups=[['mixed0','mixed1']],profile_sha256=profile_csv_ref['sha256'],
        prediction_formula='author_csv_ms_integer_truncation',origin='official_core_with_paper_supplement',frozen=False,at_s=99.8),
        dict(kind='eco_service_window_start',duration_s=150,at_s=100.),
        dict(kind='eco_admission',request_id='ecoserve-701-0',instance_id='mixed0',origin='official_core',
             prompt_blocks=1,predicted_prefill_ms=0,at_s=100.001)]
    for token,count,t,finished in [(5,1,first,False),(6,2,last,True)]:
        events += [dict(kind='eco_native_sse',request_id='ecoserve-701-0',instance_id='mixed0',at_s=t-.001,
                        payload=dict(token_ids=[token],token_index=count,finished=finished)),
                   dict(kind='eco_client_sse',request_id='ecoserve-701-0',at_s=t,
                        payload=dict(token_ids=[token],token_index=count,finished=finished))]
    events.append(dict(kind='eco_scale_observation',at_s=105.,origin='paper_supplement',
        history_window_s=60.,ttft_threshold_s=1.,history_count=1,mean_ttft_s=.1,ttft_exceeds_threshold=False,
        available_instances=[f'mixed{i}' for i in range(2,8)],groups=[['mixed0','mixed1']]))
    events.append(dict(kind='eco_closed',at_s=250.,failure=None,quarantined=[]))
    outcomes=[dict(request_id='ecoserve-701-0',input_tokens=3,output_tokens=2,arrival_s=0.,submitted_s=100.001,
        completion_tokens=2,event_count=2,token_ids_sha256=hashlib.sha256(b'[5,6]').hexdigest(),
        terminal=True,finished_s=last,ok=True)]
    events.insert(-2,dict(kind='eco_request_outcome',at_s=last+.0001,**outcomes[0]))
    events_ref=put(tmp_path,'eco-events.jsonl',events,journal=True);args['raw_refs']['events']=events_ref
    native=dict(system='ecoserve',model_id=point['model_id'],seed=701,duration_s=150,service_started_s=100.,
        service_finished_s=250.,trace_sha256=trace_ref['sha256'],events_sha256=events_ref['sha256'],
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),outcomes=outcomes,journal_rows=len(events),
        initial_native_states={i['instance_id']:state(1,99.9,5) for i in identity['instances']},
        drain_receipts={row['instance_id']:row['drain'] for row in args['drain']['states']},cleanup_errors=[])
    for s in native['initial_native_states'].values():
        for rank in s['ranks']:rank['retained_kv_supported']=True
    rows=canonical_outcomes('ecoserve',trace,outcomes,service_started_s=100.,journal=events)
    metrics=reduce_comparison(trace,rows,service_started_s=100.,slo=(1.,.1))
    args['native_result']=native;args['canonical_metrics']=metrics
    power=json.loads(Path(args['raw_refs']['power']['path']).read_text())
    power['frequency_samples']=[[99.5+i*.5,[2520]*2+[900]*6] for i in range(303)]
    save_change(args,'power',power)
    for key in ('startup_qualification','native_result'):
        save_change(args,key,args[key])
    save_change(args,'canonical_requests',metrics['request_metrics'])
    return args


def test_scoped_inputs_inherit_exact_false_artifacts_without_changing_them(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch)
    result=validate_ecoserve_inputs(args['point'],args['engine_identity'])
    assert result['preflight_ready'],result['gate_failures']
    assert result['formal_eligible'] is result['full_profile_qualified'] is False
    assert json.loads(Path(args['point']['inputs']['eco_mechanism_completion']['path']).read_text())['formal_eligible'] is False


def test_complete_eco_window_qualifies_only_single_observation(tmp_path,monkeypatch):
    result=audit_ecoserve_window(**eco_fixture(tmp_path,monkeypatch))
    assert result['missing_gates']==[],result['gate_failures']
    assert result['formal_eligible'] and result['slo_pass']
    assert not result['full_profile_qualified'] and not result['optimality_established']


def test_eco_slo_failure_is_still_valid_observation(tmp_path,monkeypatch):
    result=audit_ecoserve_window(**eco_fixture(tmp_path,monkeypatch,slow=True))
    assert result['formal_eligible'],result['gate_failures']
    assert not result['slo_pass']


def isolated_method_fixture(tmp_path, *, stopped=False):
    uuids=[f'GPU-test-{i}' for i in range(8)]
    implementations={
        'wrapper':('pdblend.bench.isolated_comparison_meter','IsolatedComparisonMeter'),
        'public_sampler':('pdblend.bench.comparison_metering','ComparisonMeteringSession'),
        'power_sampler':('pdblend.measure.power','PowerSampler'),
        'backend':('pdblend.measure.backends','PynvmlBackend'),
        'factory':('pdblend.bench.comparison_metering','ComparisonMeteringSession'),
        'sampler':('pdblend.bench.comparison_metering','ComparisonMeteringSession')}
    files={module.replace('.','/')+'.py':hashlib.sha256(module.encode()).hexdigest()
           for module,name in implementations.values()}
    source=put(tmp_path,'isolated-source.json',dict(files=files,source_sha256=digest(files)))
    operation='stop' if stopped else 'snapshot'
    method=dict(schema='isolated-comparison-meter/v1',process_start_method='spawn',test_factory=False,
        parent_pid=100,child_pid=101,gpu_ids=list(range(8)),gpu_uuids=uuids,polling_interval_s=.1,
        maximum_interpolation_gap_s=1.,sample_clocks=True,additional_frequency_observation=True,
        public_snapshot_fields_modified=False,additional_snapshot_fields=['frequency_samples'],
        rpc_scope='outside_service_and_drain_tail',start_requested_s=90.,started_s=91.,receipt_observed_s=253.,
        window_guard_active=False,status='stopped' if stopped else 'running',child_alive=not stopped,
        local_window_guards=[dict(begin_s=99.,end_s=251.)],
        commands=[dict(operation=operation,sequence=1,requested_s=251.1,finished_s=252.,passed=True)],
        liveness_observations=[dict(operation='start',sequence=0,observed_s=90.9,child_pid=101,
            sampler_thread_alive=True,sampler_error=None),dict(operation=operation,sequence=1,
            observed_s=251.9,child_pid=101,sampler_thread_alive=not stopped,sampler_error=None)])
    if stopped:method.update(finished_s=252.5,child_exitcode=0)
    for key,(module,name) in implementations.items():
        relative=module.replace('.','/')+'.py'
        method[key]=dict(module=module,name=name,path='/frozen/'+relative,sha256=files[relative])
    args=dict(point=dict(metering_execution='isolated_process'),identity=dict(
        metering_execution='isolated_process',fleet_gpu_uuids=uuids),startup=dict(source_manifest=source),
        native=dict(service_started_s=100.),metering=dict(tail_end_s=250.1),raw_refs={})
    # Actual source-format startup observations, replayed by the production
    # read-only preflight reducer. No GPU call or fake factory is used here.
    from pdblend.measure.backends import INSTANT_POWER_SOURCE_ID
    from pdblend.bench.comparison_meter_preflight import qualify_startup_snapshot
    times=[91.1+i*.1 for i in range(26)]
    sensor=dict(mode='instant',source_id=INSTANT_POWER_SOURCE_ID,field_id=186,scope_id=0)
    power=dict(gpus=list(range(8)),gpu_uuids=uuids,gpu_uuid_binding_verified=True,
        samples=[[t,[100.]*8] for t in times],power_source=sensor,
        frequency_samples=[[t,[2520]*8] for t in times],
        power_metadata=[dict(gpus=list(range(8)),read_finished_s=[t]*8,
            **{k:[v]*8 for k,v in dict(sensor,return_code=0,value_type=1).items()}) for t in times],
        utilization_readings=[dict(gpu=g,gpu_util_pct=0.,read_finished_s=t,error=None)
                              for t in times for g in range(8)])
    startup_command=dict(operation='snapshot',sequence=1,requested_s=93.7,finished_s=94.,passed=True)
    startup_live=dict(operation='snapshot',sequence=1,observed_s=93.9,child_pid=101,
                      sampler_thread_alive=True,sampler_error=None)
    method['commands'][0]['sequence']=2;method['liveness_observations'][-1]['sequence']=2
    method['commands'].insert(0,startup_command);method['liveness_observations'].insert(1,startup_live)
    startup_method=deepcopy(method)
    startup_method.update(status='running',child_alive=True,receipt_observed_s=94.1,
        commands=[startup_command],liveness_observations=deepcopy(method['liveness_observations'][:2]),local_window_guards=[])
    for key in ('finished_s','child_exitcode'):startup_method.pop(key,None)
    startup_ref=put(tmp_path,'metering-method-startup.json',startup_method)
    preflight=qualify_startup_snapshot(power,startup_method,uuids)
    preflight.update(method=startup_ref,raw_power=put(tmp_path,'metering-startup-power.json',power))
    args['startup'].update(metering_method_startup=startup_ref,
        metering_startup_preflight=put(tmp_path,'metering-startup-preflight.json',preflight))
    args['raw_refs']['metering_method']=put(tmp_path,'metering-method.json',method)
    return args,method


@pytest.mark.parametrize('stopped',[False,True])
def test_isolated_process_receipt_binds_source_and_full_tail(tmp_path,stopped):
    args,method=isolated_method_fixture(tmp_path,stopped=stopped)
    assert _isolated_meter_method(**args)==method


@pytest.mark.parametrize('change',['same_pid','test_factory','source','alternate_sampler','clock',
    'coverage','rpc_tail','rpc_previous_guard','missing_liveness','sampler_error','exited','mode'])
def test_isolated_receipt_cannot_hide_changed_source_or_tail_rpc(tmp_path,change):
    args,method=isolated_method_fixture(tmp_path)
    if change=='same_pid':method['child_pid']=method['parent_pid']
    elif change=='test_factory':method['test_factory']=True
    elif change=='source':method['power_sampler']['sha256']='bad'
    elif change=='alternate_sampler':method['sampler']['name']='AlternateSampler'
    elif change=='clock':method['additional_frequency_observation']=False
    elif change=='coverage':method['local_window_guards'][0]['end_s']=250.
    elif change=='rpc_tail':method['commands'][-1]['requested_s']=249.
    elif change=='rpc_previous_guard':
        method['local_window_guards'].append(dict(begin_s=251.2,end_s=252.1))
    elif change=='missing_liveness':method['liveness_observations'].pop()
    elif change=='sampler_error':method['liveness_observations'][-1]['sampler_error']='NVML read failed'
    elif change=='exited':method['child_alive']=False
    elif change=='mode':args['identity']['metering_execution']='thread'
    args['raw_refs']['metering_method']=put(tmp_path,'metering-method.json',method)
    with pytest.raises(ValueError):_isolated_meter_method(**args)


@pytest.mark.parametrize('change',['child','raw_power','receipt','method_ref'])
def test_isolated_startup_preflight_is_replayed_and_same_child(tmp_path,change):
    args,method=isolated_method_fixture(tmp_path)
    startup=args['startup'];ref=startup['metering_startup_preflight'];preflight=json.loads(Path(ref['path']).read_text())
    if change=='child':
        prior=json.loads(Path(startup['metering_method_startup']['path']).read_text());prior['child_pid']=103
        startup['metering_method_startup']=put(tmp_path,'metering-method-startup.json',prior)
        preflight['method']=startup['metering_method_startup']
    elif change=='raw_power':
        raw=json.loads(Path(preflight['raw_power']['path']).read_text());raw['samples'][1][1][0]=999.
        preflight['raw_power']=put(tmp_path,'metering-startup-power.json',raw)
    elif change=='receipt':preflight['observed_span_s']=20.
    elif change=='method_ref':preflight['method']['sha256']='bad'
    startup['metering_startup_preflight']=put(tmp_path,'metering-startup-preflight.json',preflight)
    with pytest.raises(ValueError):_isolated_meter_method(**args)


def test_evidence_cannot_be_substituted_by_arbitrary_passed_report(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch)
    args['point']['inputs']['eco_mechanism_review']=put(tmp_path,'fake.json',dict(formal_eligible=True))
    result=validate_ecoserve_inputs(args['point'],args['engine_identity'])
    assert not result['preflight_ready'] and 'inheritance.bound_catalog' in result['missing_gates']


def test_changed_protected_source_requires_new_qualification(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch)
    (tmp_path/'source/pdblend_runtime/serve.py').write_text('changed engine')
    result=validate_ecoserve_inputs(args['point'],args['engine_identity'])
    assert 'source.reviewed_compatibility' in result['missing_gates']


def test_plain_source_path_cannot_replace_binding(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch)
    result=validate_ecoserve_inputs(args['point'],args['engine_identity'],source_manifest=str(tmp_path/'source/manifest.json'))
    assert 'source.reviewed_compatibility' in result['missing_gates']


def mutate_events(args,fn):
    ref=args['raw_refs']['events'];path=Path(ref['path']);events=[json.loads(s) for s in path.read_text().splitlines()]
    fn(events);new=put(path.parent,path.name,events,journal=True);args['raw_refs']['events']=new
    args['native_result']['events_sha256']=new['sha256'];args['native_result']['journal_rows']=len(events)
    save_change(args,'native_result',args['native_result'])


def test_initial_macro_inventory_must_match_explicit_config(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch)
    mutate_events(args,lambda rows:next(r for r in rows if r['kind']=='eco_startup').update(groups=[['mixed0','mixed2']]))
    assert 'eco.raw_protocol_and_canonical_metrics' in audit_ecoserve_window(**args)['missing_gates']


def test_buffer_token_substitution_is_rejected(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch)
    mutate_events(args,lambda rows:next(r for r in rows if r['kind']=='eco_native_sse')['payload'].update(token_ids=[999]))
    assert 'eco.raw_protocol_and_canonical_metrics' in audit_ecoserve_window(**args)['missing_gates']


def test_requested_clock_does_not_replace_actual_frequency_samples(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch);power=json.loads(Path(args['raw_refs']['power']['path']).read_text())
    power['frequency_samples']=[];save_change(args,'power',power)
    result=audit_ecoserve_window(**args)
    assert result['missing_gates']==['eco.observed_active_frequency']
    assert 'eco.raw_protocol_and_canonical_metrics' in result['checked_gates']


def test_parked_gpus_remain_in_power_scope(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch);result=audit_ecoserve_window(**args)
    assert result['formal_eligible'],result['gate_failures']
    assert args['metering']['energy_service_j']==120000.


def test_incomplete_final_rank_drain_blocks_formal(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch);args['drain']['states'][7]['state']['ranks']=[]
    save_change(args,'drain',args['drain'])
    assert 'eco.all_rank_drain' in audit_ecoserve_window(**args)['missing_gates']


def test_automatic_add_requires_complete_trigger_prepare_and_clock_receipts(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch)
    def add(rows):
        before=[['mixed0','mixed1']];after=[['mixed0','mixed1','mixed2']]
        rows[-1:-1]=[
          dict(kind='eco_scale_observation',at_s=110.,origin='paper_supplement',history_window_s=60.,
               ttft_threshold_s=1.,history_count=1,mean_ttft_s=2.,ttft_exceeds_threshold=True,
               available_instances=[f'mixed{i}' for i in range(2,8)],groups=before),
          dict(kind='eco_membership_prepare',at_s=110.01,operation='add',instance_id='mixed2',
               trigger='mean_ttft',before=before,observed_engine_states={f'mixed{i}':state(1,110.,5) for i in range(8)}),
          dict(kind='eco_http_receipt',at_s=110.02,instance_id='mixed2',path='/baseline/clock',
               body=dict(frequency_mhz=2520),response=dict(acknowledged=True,success=True,requested_frequency_mhz=2520,
               gpus=[dict(gpu_uuid=args['engine_identity']['fleet_gpu_uuids'][2],frequency_mhz=2520)])),
          dict(kind='eco_membership_commit',at_s=110.03,origin='paper_supplement',operation='add',
               instance_id='mixed2',trigger='mean_ttft',version=1,before=before,after=after,worker_kv_preserved=True)]
    mutate_events(args,add)
    power=json.loads(Path(args['raw_refs']['power']['path']).read_text())
    for t,values in power['frequency_samples']:
        if t>=110.02:values[2]=2520
    save_change(args,'power',power)
    result=audit_ecoserve_window(**args)
    assert result['formal_eligible'],result['gate_failures']
    mutate_events(args,lambda rows:next(r for r in rows if r['kind']=='eco_membership_commit').update(trigger='explicit_functional'))
    assert 'eco.raw_protocol_and_canonical_metrics' in audit_ecoserve_window(**args)['missing_gates']


def test_missing_periodic_controller_observations_is_not_a_quiet_window(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch)
    mutate_events(args,lambda rows:rows.__setitem__(slice(None),[r for r in rows if r['kind']!='eco_scale_observation']))
    assert 'eco.raw_protocol_and_canonical_metrics' in audit_ecoserve_window(**args)['missing_gates']


def test_accelerated_controller_period_is_rejected(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch);ref=args['point']['inputs']['system_config'];path=Path(ref['path'])
    config=json.loads(path.read_text());config['eco_scale_period_s']=1.
    args['point']['inputs']['system_config']=put(path.parent,path.name,config)
    result=validate_ecoserve_inputs(args['point'],args['engine_identity'])
    assert 'config.fleet_policy' in result['missing_gates']


def failure_case(args,kind):
    """Use the exact fields emitted by payload_receipt and finally in Eco."""
    path=Path(args['raw_refs']['events']['path']);events=[json.loads(s) for s in path.read_text().splitlines()]
    row=args['native_result']['outcomes'][0]
    row.update(ok=False,terminal=False,error='TimeoutError()' if kind!='rejection' else 'RuntimeError: no capacity',
               finished_s=110. if kind!='global_timeout' else 430.)
    if kind=='global_timeout':
        # gather cancellation raises CancelledError (BaseException), bypassing
        # the request's except Exception while its finally still emits a row.
        row.pop('error')
    if kind in ('rejection','zero_timeout','global_timeout'):
        events=[e for e in events if e['kind'] not in ('eco_client_sse','eco_native_sse')]
        if kind=='rejection':events=[e for e in events if e['kind']!='eco_admission']
        ids=[];count=0
    else:
        # Native generated both IDs, but the client received only the first
        # before cancellation/timeout. That is a measured failed observation.
        events=[e for e in events if not (e['kind']=='eco_client_sse' and e['payload']['token_index']==2)]
        for event in events:
            if event['kind']=='eco_native_sse':event['payload']['finished']=False
        ids=[5];count=1
    row.update(completion_tokens=len(ids),event_count=count,
               token_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
    events=[e for e in events if e['kind']!='eco_request_outcome']
    events.insert(-1,dict(kind='eco_request_outcome',at_s=row['finished_s']+.001,**row))
    if kind=='global_timeout':
        args['native_result'].pop('service_finished_s')
        args['native_result']['error']='TimeoutError()'
        events[-1]['at_s']=430.01
        for receipt in args['drain']['states']:
            receipt['received_s']=430.05
            receipt['state']=state(1,430.05,5)
            receipt['drain']=dict(state(1,430.05,5),acknowledged=True,drained=True)
        args['drain']['tail_end_s']=430.1
        args['native_result']['drain_receipts']={r['instance_id']:r['drain'] for r in args['drain']['states']}
        save_change(args,'drain',args['drain'])
        power=json.loads(Path(args['raw_refs']['power']['path']).read_text())
        template=deepcopy(power['power_metadata'][0]);times=[99.5+i*.5 for i in range(663)]
        power['samples']=[[t,[100.]*8] for t in times]
        power['power_metadata']=[dict(template,read_finished_s=[t]*8) for t in times]
        power['utilization_readings']=[dict(gpu=g,gpu_util_pct=20.,read_finished_s=t,error=None) for t in times for g in range(8)]
        power['frequency_samples']=[[t,[2520]*2+[900]*6] for t in times]
        save_change(args,'power',power)
        from pdblend.bench.comparison_metering import summarize_comparison
        args['metering']=summarize_comparison(power,gpu_uuids=args['engine_identity']['fleet_gpu_uuids'],origin_s=100.,tail_end_s=430.1)
        save_change(args,'metering',args['metering'])
    # Native events are append ordered in the real journal.
    events.sort(key=lambda e:e['at_s'])
    args['raw_refs']['events']=put(path.parent,path.name,events,journal=True)
    args['native_result']['events_sha256']=args['raw_refs']['events']['sha256']
    args['native_result']['journal_rows']=len(events)
    save_change(args,'native_result',args['native_result'])
    trace=json.loads(Path(args['point']['trace']['path']).read_text())
    rows=canonical_outcomes('ecoserve',trace,args['native_result']['outcomes'],service_started_s=100.,journal=events)
    args['canonical_metrics']=reduce_comparison(trace,rows,service_started_s=100.,slo=(1.,.1))
    save_change(args,'canonical_requests',args['canonical_metrics']['request_metrics'])
    return args


@pytest.mark.parametrize('kind',['rejection','zero_timeout','partial_timeout','global_timeout'])
def test_explicit_failures_with_complete_journal_remain_valid_observations(tmp_path,monkeypatch,kind):
    args=failure_case(eco_fixture(tmp_path,monkeypatch),kind)
    result=audit_ecoserve_window(**args)
    assert result['formal_eligible'],result['gate_failures']
    assert not result['slo_pass'] and args['canonical_metrics']['failed_requests']==1
    assert args['canonical_metrics']['token_timing_complete'] is True
    assert args['canonical_metrics']['window_delivered_tokens']==(1 if kind=='partial_timeout' else 0)
    if kind=='global_timeout':
        assert 'service_finished_s' not in args['native_result']
        assert 'error' not in args['native_result']['outcomes'][0]
        assert result['observation_boundary']['source']=='complete_cohort_and_eco_closed'
        assert result['observation_boundary']['observation_end_s']==430.01


def test_partial_failure_cannot_reorder_delivered_tokens(tmp_path,monkeypatch):
    args=failure_case(eco_fixture(tmp_path,monkeypatch),'partial_timeout')
    row=args['native_result']['outcomes'][0]
    row['token_ids_sha256']=hashlib.sha256(b'[6]').hexdigest()
    def reorder(rows):
        next(e for e in rows if e['kind']=='eco_client_sse')['payload'].update(token_ids=[6])
        next(e for e in rows if e['kind']=='eco_request_outcome').update(row)
    mutate_events(args,reorder)
    result=audit_ecoserve_window(**args)
    assert not result['formal_eligible']
    assert 'lost/reordered' in result['gate_failures']['eco.raw_protocol_and_canonical_metrics']


def test_missing_failure_outcome_journal_is_not_inferred_from_count_zero(tmp_path,monkeypatch):
    args=failure_case(eco_fixture(tmp_path,monkeypatch),'rejection')
    mutate_events(args,lambda rows:rows.__setitem__(slice(None),[e for e in rows if e['kind']!='eco_request_outcome']))
    assert not audit_ecoserve_window(**args)['formal_eligible']


def test_missing_outcome_remains_invalid_even_with_complete_drain(tmp_path,monkeypatch):
    args=failure_case(eco_fixture(tmp_path,monkeypatch),'zero_timeout')
    args['native_result']['outcomes']=[];save_change(args,'native_result',args['native_result'])
    assert not audit_ecoserve_window(**args)['formal_eligible']


def test_missing_service_finish_is_not_fabricated_without_raw_failure_and_close(tmp_path,monkeypatch):
    args=eco_fixture(tmp_path,monkeypatch);args['native_result'].pop('service_finished_s')
    save_change(args,'native_result',args['native_result'])
    assert 'eco.observation_boundary' in audit_ecoserve_window(**args)['missing_gates']


def test_global_timeout_without_full_service_window_is_invalid(tmp_path,monkeypatch):
    args=failure_case(eco_fixture(tmp_path,monkeypatch),'global_timeout')
    mutate_events(args,lambda rows:next(e for e in rows if e['kind']=='eco_closed').update(at_s=249.))
    result=audit_ecoserve_window(**args)
    assert not result['formal_eligible']
    assert 'eco.observation_boundary' in result['missing_gates']


def shutdown_cancel_case(args):
    def add(rows):
        for index in range(8):
            rows.insert(-1,dict(kind='eco_http_receipt',instance_id='mixed'+str(index),method='GET',
                path='/baseline/state',body=None,error='CancelledError()',started_s=429.999,
                at_s=430.005+index*.0001))
    mutate_events(args,add)
    return args


def test_complete_global_timeout_allows_cancelled_shutdown_poll_with_native_drains(tmp_path,monkeypatch):
    args=shutdown_cancel_case(failure_case(eco_fixture(tmp_path,monkeypatch),'global_timeout'))
    result=audit_ecoserve_window(**args)
    assert result['formal_eligible'] and not result['slo_pass'],result['gate_failures']


@pytest.mark.parametrize('change',[
    dict(method='POST'),dict(path='/baseline/clock'),dict(error='RuntimeError: state failed'),
    dict(at_s=249.),dict(at_s=430.02),dict(started_s=430.02),
])
def test_shutdown_cancel_exception_never_hides_service_or_control_failures(tmp_path,monkeypatch,change):
    args=shutdown_cancel_case(failure_case(eco_fixture(tmp_path,monkeypatch),'global_timeout'))
    mutate_events(args,lambda rows:next(e for e in rows if e.get('error')=='CancelledError()').update(change))
    result=audit_ecoserve_window(**args)
    assert not result['formal_eligible']
    assert 'eco.raw_protocol_and_canonical_metrics' in result['missing_gates']
