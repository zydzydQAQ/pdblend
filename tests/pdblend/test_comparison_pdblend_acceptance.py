"""PD controller/raw-window contracts; no real profile promotion or GPU work."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pdblend.bench import comparison_pdblend_acceptance as audit
from pdblend.bench.comparison_metrics import canonical_outcomes, reduce_comparison
from test_comparison_native_acceptance import native_fixture
from test_comparison_acceptance import put, save_change, state, drain_rows


def fixture(tmp_path, monkeypatch, *, parked=False, failed=False, slow=False,
            mixed_instances=4, profile_key='profile-key'):
    args = native_fixture(tmp_path); point = args['point']; point['system'] = 'pdblend'
    identity = args['engine_identity']; specs = identity['instances']
    instances = {r['instance_id']:r for r in specs}
    trace = json.loads(Path(point['trace']['path']).read_text()); trace['selection_split'] = 'evaluation'
    point['trace'] = args['raw_refs']['trace'] = put(tmp_path,'trace.json',trace)
    roles = {iid:('M' if not parked or index < mixed_instances else 'off' if index == 7 else 'L1')
             for index,iid in enumerate(instances)}
    counts = dict(Counter(roles.values())); selected = dict(profile_key=profile_key, frequencies=[1500,2520],
        choice=dict(plan=dict(counts=counts,f_P=1500,f_D=1500,f_M=1500,tau=1024)))
    monkeypatch.setattr(audit,'_inputs',lambda *_:selected)
    reset = args['reset']; caps = deepcopy(args['startup_qualification']['capabilities'])
    for cap in caps.values(): cap['state'] = state(1,97.4,4)
    reset['pdblend_inventory_reset'] = dict(passed=True,allocated_to_previous_service=False,
        inventory_instance_ids=list(instances),restored_instances=[],engine_loads=0,
        started_s=96.,finished_s=97.5,before=dict(off_instances=[],tail_end_s=97.),capabilities=caps)
    first = 102. if slow else 100.1; last = first+.05
    outcome = dict(idx=0,arrival_s=0.,scheduled_s=100.,submitted_s=100.001,input_tokens=3,max_tokens=2,
        completion_tokens=0 if failed else 2,sampling_seed=701,terminal=not failed,finished_s=last,
        first_token_s=None if failed else first,last_token_s=None if failed else last,
        terminal_s=None if failed else last,path='' if failed else 'M',prefill='' if failed else 'mixed0',
        decode='' if failed else 'mixed0',error='503: no instance accepting requests' if failed else None,
        token_events=[] if failed else [dict(received_s=first,count=1,exact=True),dict(received_s=last,count=1,exact=True)],
        token_events_complete=True)
    routes = [] if failed else [dict(request_id='r0',input_tokens=3,path='M',submitted_s=100.002,
        prefill_instance='mixed0',decode_instance='mixed0',tp=1,pp=1,generation=5,profile_key='',
        terminal_state='completed',last_token_s=last,route_estimate={})]
    phases = []
    for index,(iid,role) in enumerate(roles.items()):
        operations = ['clock_set','route_publish'] if role == 'M' else [
            'route_publish','proxy_drain','native_drain',*(['stop','clock_reset'] if role == 'off' else ['clock_reset','park'])]
        for n,operation in enumerate(operations):
            started = 99.1+n*.06+index*.001; finished = started+.04
            row = dict(instance=iid,operation=operation,gpus=[index],started_s=started,finished_s=finished,
                duration_s=.04,generation=5,source_role='M',source_frequency_mhz=None,
                transition_id='initial',status='passed',energy_status='awaiting_common_sampler_integration')
            if operation == 'native_drain':row['native_receipt']=dict(state(1,finished,5),acknowledged=True,drained=True)
            phases.append(row)
    phases.sort(key=lambda r:r['finished_s'])
    plan = dict(kind='plan',t=99.7,roles=roles,**selected['choice']['plan'],
        plan_identity=dict(tp=1,pp=1,generation=5,profile_key=profile_key))
    complete = dict(kind='transition_complete',t=99.8,started_s=99.1,finished_s=99.8,
        transition_id='initial',affected=list(instances),validation_scope='native_drain_resume_and_proxy_publication',formal_eligible=False)
    events = [dict(row,kind='transition_phase',t=row['finished_s']) for row in phases]+[plan,complete,
        dict(kind='forecast',t=110.,decision_reason='periodic'),dict(kind='forecast',t=120.,decision_reason='periodic'),
        dict(kind='stop',t=250.005,roles=roles)]
    native = dict(service_started_s=100.,service_ended_s=250.,window_s=150.,request_finished_s=250.,finished_s=250.04,
        quarantined_instances=[],native_cleanup_complete=True,final_roles=roles,
        controller=dict(events=dict(Counter(r['kind'] for r in events)),transition_phases=phases,final_roles=roles))
    live = [s for s in specs if roles[s['instance_id']] != 'off']
    cleanup = {s['instance_id']:dict(state(1,250.025,5),acknowledged=True,drained=True) for s in live}
    absent = []
    for s in specs:
        iid = s['instance_id']
        if roles[iid] != 'off':continue
        phase = next(r for r in phases if r['instance']==iid and r['operation']=='stop')
        absent.append(dict(instance_id=iid,role='off',process_alive=False,process_state='off',
            compute_processes_gone=True,checked_s=250.075,owned_stop=dict(kind='stop',t_s=phase['finished_s']-.001),
            physical_gpus=[dict(local_index=7,uuid=identity['fleet_gpu_uuids'][7],compute_pids=[])]))
    drain = dict(states=drain_rows(live,250.05,5),tail_end_s=250.1,off_instances=absent,
        final_roles=roles,live_instance_ids=[s['instance_id'] for s in live],inventory_instance_ids=list(instances),
        policy_off_preserved=True,window_engine_loads=0,restart_epoch_receipts=[],
        engine_load_accounting=dict(engine_loads=8,starts=[dict(instance=i,kind='start',t_s=80.,pid=10+n)
            for n,i in enumerate(instances)]))
    rows = canonical_outcomes('pdblend',trace,[outcome],service_started_s=100.)
    metrics = reduce_comparison(trace,rows,service_started_s=100.,duration_s=150.,slo=(1.,.1))
    args.update(native_result=native,canonical_metrics=metrics,drain=drain)
    for key in ('startup_qualification','reset','native_result','drain'):save_change(args,key,args[key])
    save_change(args,'canonical_requests',metrics['request_metrics'])
    for key, values in (('outcomes',[outcome]),('routes',routes),('controller',events),
        ('frequencies',[[99.5+i*.5,[1500 if roles[s['instance_id']]=='M' else 210 for s in specs]] for i in range(303)])):
        args['raw_refs'][key] = put(tmp_path,key+'.jsonl',values,journal=True)
    for key, values in (('native_cleanup',cleanup),('transition_measurements',dict(
        phases=[dict(r,energy_status='measured',energy_j=1.) for r in phases],incremental_energy_j=None,formal_eligible=False))):
        args['raw_refs'][key] = put(tmp_path,key+'.json',values)
    return args


def raw(args,key):
    return audit._read_raw(args['raw_refs'][key],key)


def write(args,key,value):
    ref=args['raw_refs'][key];path=Path(ref['path'])
    args['raw_refs'][key]=put(path.parent,path.name,value,journal=key in audit.JOURNALS)


@pytest.mark.parametrize('parked,failed,slow',[(False,False,False),(True,False,False),(False,True,False),(False,False,True)])
def test_complete_raw_window_and_observable_failures(tmp_path,monkeypatch,parked,failed,slow):
    result=audit.audit_pdblend_window(**fixture(tmp_path,monkeypatch,parked=parked,failed=failed,slow=slow))
    assert result['evidence_valid'],result['gate_failures']
    assert result['slo_pass'] is (not failed and not slow)
    assert result['optimality_established'] is result['per_request_kv_transaction_audited'] is False


@pytest.mark.parametrize('change',['missing_route','route_epoch','outcome_missing','token_missing','freq_gap','freq_wrong',
    'park_native_ack','off_pid','off_uuid','off_stop','tail','reopened_off','reloaded','reset_stale','transition_raw','route_parked'])
def test_raw_window_rejects_missing_or_contradictory_evidence(tmp_path,monkeypatch,change):
    args=fixture(tmp_path,monkeypatch,parked=True)
    if change in ('missing_route','route_epoch','route_parked'):
        values=raw(args,'routes')
        if change=='missing_route':values=[]
        elif change=='route_epoch':values[0]['generation']=4
        else:values[0].update(prefill_instance='mixed7',decode_instance='mixed7')
        write(args,'routes',values)
    elif change in ('outcome_missing','token_missing'):
        values=raw(args,'outcomes')
        if change=='outcome_missing':values=[]
        else:values[0]['token_events'].pop()
        write(args,'outcomes',values)
    elif change.startswith('freq_'):
        values=raw(args,'frequencies')
        if change=='freq_gap':values=values[:20]+values[25:]
        else:values[20][1][0]=900
        write(args,'frequencies',values)
    elif change=='park_native_ack':
        values=raw(args,'controller');next(r for r in values if r.get('operation')=='native_drain')['native_receipt']['acknowledged']=False
        write(args,'controller',values)
    elif change=='reset_stale':
        args['reset']['pdblend_inventory_reset']['finished_s']=101.;save_change(args,'reset',args['reset'])
    elif change=='transition_raw':
        values=raw(args,'transition_measurements');values['phases'].pop();write(args,'transition_measurements',values)
    else:
        d=args['drain'];off=d['off_instances'][0]
        if change=='off_pid':off['physical_gpus'][0]['compute_pids']=[123]
        elif change=='off_uuid':off['physical_gpus'][0]['uuid']='GPU-foreign'
        elif change=='off_stop':off['owned_stop']['t_s']=98.
        elif change=='tail':d['states'][0]['received_s']=251.
        elif change=='reopened_off':d['policy_off_preserved']=False
        elif change=='reloaded':d['window_engine_loads']=1
        save_change(args,'drain',d)
    result=audit.audit_pdblend_window(**args)
    assert not result['evidence_valid'] and not result['formal_eligible']


def test_no_formal_profile_is_not_promoted_by_valid_window(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(audit,'_inputs',lambda *_:(_ for _ in ()).throw(ValueError('formal qualification is absent')))
    result=audit.audit_pdblend_window(**args)
    assert not result['formal_eligible'] and 'pdblend.formal_inputs' in result['missing_gates']


def test_frequency_array_journal_cannot_skip_its_hash(tmp_path):
    ref=put(tmp_path,'freq.jsonl',[[100.,[1500]*8]],journal=True)
    assert audit._read_raw(ref,'frequencies')==[[100.,[1500]*8]]
    Path(ref['path']).write_text('[]\n')
    with pytest.raises(ValueError,match='checksum'):audit._read_raw(ref,'frequencies')


def test_clock_settle_begins_after_completed_physical_transition(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    events=raw(args,'controller');plans=[r for r in events if r.get('kind')=='plan'];complete=[r for r in events if r.get('kind')=='transition_complete']
    complete[0]['finished_s']=101.
    values=raw(args,'frequencies')
    for row in values:
        if row[0]<102.:row[1]=[900]*8
    audit._frequencies(values,plans,complete,{r['instance_id']:r for r in args['engine_identity']['instances']},args['engine_identity'],100.)


@pytest.mark.parametrize('mutation',[None,'foreign_request','retained_rank','missing_ack'])
def test_failed_stream_requires_real_per_request_native_cancel(tmp_path,monkeypatch,mutation):
    args=fixture(tmp_path,monkeypatch)
    outcomes=raw(args,'outcomes');outcomes[0].update(error='TimeoutError()',terminal=False,completion_tokens=1,
        token_events=outcomes[0]['token_events'][:1],finished_s=100.3)
    routes=raw(args,'routes');routes[0]['terminal_state']='cancelled_acknowledged'
    receipt=dict(instance_id='mixed0',request_id='r0',generation=5,acknowledged=True,cancelled=True,native_state=state(1,100.4,5))
    if mutation=='foreign_request':receipt['request_id']='r9'
    elif mutation=='retained_rank':receipt['native_state']['ranks'][0]['retained']['held_requests']=['r0']
    elif mutation=='missing_ack':receipt['acknowledged']=False
    routes[0]['route_estimate']['native_cancel_receipts']={'mixed0':receipt}
    instances={r['instance_id']:r for r in args['engine_identity']['instances']}
    call=lambda:audit._routes(routes,outcomes,raw(args,'trace'),instances,args['reset'],args['native_result'])
    if mutation:
        with pytest.raises(ValueError):call()
    else:call()


@pytest.mark.parametrize('mutation',[None,'wrong_epoch','not_closed','wrong_model','missing_restart'])
def test_actual_off_restart_requires_restored_same_native_epoch(tmp_path,monkeypatch,mutation):
    args=fixture(tmp_path,monkeypatch);native=args['native_result'];drain=args['drain'];reset=args['reset'];identity=args['engine_identity']
    phases=raw(args,'controller');phases.append(dict(instance='mixed0',operation='start',started_s=140.,finished_s=141.))
    drain['engine_load_accounting']['starts'].append(dict(instance='mixed0',kind='start',t_s=140.5,pid=50))
    drain['engine_load_accounting']['engine_loads']=9;drain['window_engine_loads']=1
    closed=state(1,145.,5);closed.update(accepting=False,admit_prefill=False,admit_decode=False)
    receipt=dict(instance_id='mixed0',pid=50,started_s=144.,finished_s=145.01,
        capability=dict(tp=1,pp=1,source_revision='source',gpu_uuids=[identity['fleet_gpu_uuids'][0]],
                        **{k:identity[k] for k in ('model_hash','tokenizer_hash','image_digest')}),
        before=state(1,144.2,0),state=closed,control=dict(acknowledged=True,generation=5))
    if mutation=='wrong_epoch':receipt['control']['generation']=0
    elif mutation=='not_closed':receipt['state']['accepting']=True
    elif mutation=='wrong_model':receipt['capability']['model_hash']='foreign'
    drain['restart_epoch_receipts']=[] if mutation=='missing_restart' else [receipt]
    instances={r['instance_id']:r for r in identity['instances']}
    call=lambda:audit._drain(native,drain,raw(args,'native_cleanup'),instances,identity,reset,args['metering'],phases,'source')
    if mutation:
        with pytest.raises(ValueError):call()
    else:call()


@pytest.mark.parametrize('receipt_flags',[False,True])
def test_real_input_loader_rejects_incomplete_profile_despite_receipt_flags(tmp_path,receipt_flags):
    args=native_fixture(tmp_path);point=args['point'];point['system']='pdblend'
    trace=raw(args,'trace');trace['selection_split']='evaluation'
    point['trace']=put(tmp_path,'trace.json',trace)
    profile=dict(system='pdblend',model=point['model_id'],tp=1,pp=1,freqs=[1500],
        prefill_time={'1500':[0.,.001,0.]},prefill_power={'1500':[100.,0.]},
        decode_time={'1500':[.001,0.,0.,0.]},decode_power={'1500':[100.,0.]},
        static={},transfer=[0.,1e10],quality={'formal_eligible':False})
    profile_ref=put(tmp_path,'profile.json',profile)
    config=dict(system='pdblend',model_id=point['model_id'],profile=profile_ref)
    choice=dict(system='pdblend',model_id=point['model_id'],selection_split='tuning',evaluation_used_for_selection=False,
        profile_sha256=profile_ref['sha256'],plan=dict(counts={'M':8},tp=1,pp=1,f_P=1500,f_D=1500,f_M=1500,
            tau=0,power_w=800.,ttft_s=.1,tpot_s=.01))
    prior=dict(trace,selection_split='tuning',seed=9702)
    gates=['source_identity','profile_calibration','workload_coverage','mechanisms','energy_protocol']
    point['inputs']=dict(trace=point['trace'],system_config=put(tmp_path,'config.json',config),profiles=[profile_ref],
        offline_choice=put(tmp_path,'choice.json',choice),planning_trace=put(tmp_path,'prior.json',prior),
        qualifications=[put(tmp_path,g+'.json',dict(system='pdblend',model_id=point['model_id'],gate=g,
            formal_eligible=receipt_flags,trace_sha256=point['trace']['sha256'])) for g in gates])
    with pytest.raises(ValueError,match='formal|qualifications'):
        audit._inputs(point,{r['instance_id']:r for r in args['engine_identity']['instances']})
