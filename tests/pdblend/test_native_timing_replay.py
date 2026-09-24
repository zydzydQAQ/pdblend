"""Complete synthetic raw timing receipts; no GPU/model execution."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from pdblend.profile.collection import native_timing_replay as module
from pdblend.profile.collection.native_timing_audit import audit_window,fit_component
from pdblend.profile.collection.native_timing_plan import binding,digest
from pdblend.results.journal import payload_receipt
from pdblend.profile.query.native_timing import NativeTimingOverlay,attach_native_timing
from pdblend.profile.query.versions import LoadedVersion
from test_pdblend_native_timing import plan_fixture,component_rows
from test_comparison_acceptance import state


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True))
    return binding(path)


@pytest.fixture
def collected(tmp_path):
    plan=plan_fixture(tmp_path);plan_ref=put(tmp_path/'plan.json',plan)
    source=tmp_path/'source';source.mkdir();(source/'test.py').write_text('immutable synthetic source\n')
    files={'test.py':binding(source/'test.py')['sha256']};source_sha=digest(files)
    source_ref=put(source/'manifest.json',dict(files=files,source_sha256=source_sha))
    inventory=[dict(path='weights',bytes=1,sha256='a'*64,kind='weight'),
               dict(path='tokenizer',bytes=1,sha256='b'*64,kind='tokenizer')]
    verification=put(tmp_path/'model-verification.json',dict(all_pass=True,models={'7b':dict(
        model_id=plan['model_id'],verified=True,files=inventory)}))
    inputs=dict(schema='pdblend-native-timing-inputs-v1',system='pdblend',model_id=plan['model_id'],
        point_plan=plan_ref,source_manifest=source_ref,source_sha256=source_sha,image_digest='image',model_verification=verification)
    inputs_ref=put(tmp_path/'inputs.json',inputs)
    identity=dict(model_id=plan['model_id'],tp=1,pp=1,engine_revision='vllm-0.10.1.1',
        source_revision=source_sha,image_digest='image',
        model_hash=digest([('weights',1,'a'*64)]),tokenizer_hash=digest([('tokenizer',1,'b'*64)]))
    uuids=[f'GPU-{i}' for i in range(8)]
    attempt=tmp_path/'attempt';root=attempt/'native-timing';caps={}
    for i,iid in enumerate(module.INSTANCES):caps[iid]=dict(identity,gpu_uuids=[uuids[i]])
    def raw(point,index,base,name):
        n=point['prompt_tokens'];b=point['batch'];f=point['frequency_mhz'];count=point['output_tokens']
        clients=[];ids=[name+'-'+str(i) for i in range(b)]
        for rid in ids:
            events=[dict(token_ids=list(range(count)),token_index=count,finished=True,received_s=base+8.5)]
            clients.append(dict(submitted_s=base+3.,finished_s=base+8.6,events=events,
                **payload_receipt(events,journal_path='embedded:events',request_id=rid)))
        samples=[]
        for i in range(8 if point['role']=='decode' else 1):
            c=n+i+1 if point['role']=='decode' else n
            latency=1+b+b*c/8192 if point['role']=='decode' else 1+n/8192+(n/8192)**2
            samples.append(dict(system='pdblend',measurement_scope='runner',rank=0,tp=1,pp=1,
                role=point['role'],batch=b,prompt_lengths=[n]*b,context_lengths=[c]*b,
                scheduled_lengths=[1 if point['role']=='decode' else n]*b,request_ids=ids,
                gpu_elapsed_ms=latency,at_s=base+4.+i*.1))
        stamps=[base+3.+i*.5 for i in range(13)]
        power_source=dict(mode='instant',source_id='nvml:field:186:scope:0:mW',field_id=186,scope_id=0,value_type=1,return_code=0)
        return dict(schema='pdblend-native-timing-window-v1',system='pdblend',status='measured',point=point,
            window_id=name,capability=caps[module.INSTANCES[index]],measurement_started_s=base,
            settle_started_s=base,settle_finished_s=base+2.,start_s=base+3.,end_s=base+9.,
            measurement_start=dict(acknowledged=True),measurement_stop=dict(ranks=[dict(rank=0,acknowledged=True)]),
            cleanup_errors=[],sampler_error=None,drain=dict(state(1,base+10.,5),acknowledged=True,drained=True),
            clock_receipt=dict(acknowledged=True,success=True,requested_frequency_mhz=f,gpus=[dict(gpu_uuid=uuids[index])]),
            frequency_samples=[(t,[f]) for t in stamps],power_samples=[(t,[100.]) for t in stamps],
            power_metadata=[dict(gpus=[index],read_started_s=[t],read_finished_s=[t],
                **{k:[v] for k,v in power_source.items()}) for t in stamps],
            client_requests=clients,sample=dict(ranks=[dict(rank=0,samples=samples)]))
    now=1000.;checks=[]
    for f in (1500,2520):
        for repeat in range(3):
            p=dict(role='decode',batch=8,prompt_tokens=1024,output_tokens=64,
                   frequency_mhz=f,purpose='interference',seed=9701,repeat=repeat)
            for i,iid in enumerate(module.INSTANCES):
                name=f'{f}-{repeat}-{iid}-isolated';put(root/'interference'/(name+'.json'),raw(p,i,now,name));now+=12.
            for i,iid in enumerate(module.INSTANCES):
                name=f'{f}-{repeat}-{iid}-parallel';put(root/'interference'/(name+'.json'),raw(p,i,now,name))
                checks.append(dict(instance_id=iid,frequency_mhz=f,repeat=repeat,
                    relative_errors=dict(latency_ms=0.,power_w=0.),common_window_s=6.,passed=True))
            now+=12.
        now+=10.  # Room for the next frequency's all-peer clock/idle preparation.
    qualification=dict(qualified=True,parallel_qualified=True,mode='parallel',limit=.05,checks=checks,
        exclusive_fleet_gpu_uuids=uuids,energy_comparable=False)
    qref=put(root/'measurement-qualification.json',qualification)
    training=[];holdout=[];refs=[]
    for index,point in enumerate(plan['points']):
        for repeat in range(3):
            name=f'{digest(point)[:20]}-{repeat}';value=raw(dict(point,repeat=repeat),index%8,now,name);now+=12.
            refs.append(put(root/'samples'/(name+'.json'),value))
            rows=audit_window(value,identity=caps[module.INSTANCES[index%8]])
            (training if point['purpose']=='training' else holdout).extend(rows)
    component=fit_component(training,holdout,identity=dict(system='pdblend',**identity),raw_bindings=refs,
        measurement_qualification=dict(qualification,receipt=qref),limits=plan['holdout_limits'])
    assert component['component_qualified']
    cref=put(root/'timing-component.json',component)
    drains=[dict(instance_id=iid,received_s=now+1.,drain=dict(state(1,now+1.,5),acknowledged=True,drained=True),
                 state=state(1,now+1.,5)) for iid in module.INSTANCES]
    from dataclasses import asdict
    from pdblend_runtime.probe import NativeSpec
    specs=[NativeSpec(iid,(i,),20000+i*4,'/models/'+plan['model_id'],max_num_seqs=32,generation=1,
        extra_args=('--enforce-eager','--worker-cls',
            'pdblend.profile.collection.native_timing_worker.PDNativeTimingWorker')) for i,iid in enumerate(module.INSTANCES)]
    launch=[dict(spec=json.loads(json.dumps(asdict(s))),argv=s.command(),environment=dict(
        CUDA_VISIBLE_DEVICES=','.join(str(g) for g in s.gpus),VLLM_USE_V1='1',NCCL_CUMEM_ENABLE='0',
        NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='0')) for s in specs]
    complete=put(root/'completion.json',dict(schema='pdblend-native-timing-collection-v1',system='pdblend',
        status='passed',complete=True,hardware_executed=True,cleanup_errors=[],raw_bindings=refs,
        capabilities={iid:dict(cap,supported=True,state=state(1,900.,1)) for iid,cap in caps.items()},
        actual_launch=launch,timing_component=cref,component_qualified=True,final_drains=drains))
    payload=dict(system='pdblend',scope='native_cuda_timing_component_only',gpu_count=8,exclusive=True,
        reserve_host=True,argv=['pdblend.profile.collection.native_timing_collect'],
        source_sha256=source_sha,image_digest='image',input_manifest=inputs_ref)
    put(attempt/'manifest.json',dict(immutable=True,job_id='timing-job',attempt=1,gpu_uuids=uuids,payload=payload))
    put(attempt/'execution.json',dict(status='passed',complete=True,returncode=0,error=None,finished_s=now+2.,
        argv=['docker','run','-v',str(attempt)+':/output:rw'],receipt_sha256={'native-timing/completion.json':complete['sha256']}))
    queue=tmp_path/'queue.json';put(queue,dict(jobs={'timing-job':dict(job_id='timing-job',attempts=1,
        payload=payload,status='succeeded',lease_id=None)}))
    evidence=tmp_path/'replay-evidence.json'
    return SimpleNamespace(attempt=attempt,root=root,queue=queue,evidence=evidence,identity=identity,
        component=component,inputs_ref=inputs_ref,tmp=tmp_path,raw_factory=raw,caps=caps)


def test_completed_attempt_replays_all_actual_windows_and_rejects_any_bound_mutation(collected):
    x=collected;ref=module.capture_evidence(x.attempt,x.queue,x.evidence)
    replay=module.replay_evidence(ref)
    assert replay['replayed_windows']==354 and replay['replayed_interference_windows']==96
    assert replay['component']==x.component and not replay['formal_eligible']
    assert not replay['auxiliary_power_qualifies_power_component']
    with pytest.raises(ValueError,match='overwrite'):module.capture_evidence(x.attempt,x.queue,x.evidence)
    for path in [x.root/'completion.json',next((x.root/'samples').glob('*.json')),
                 next((x.root/'interference').glob('*.json')),x.tmp/'source/test.py',x.tmp/'inputs.json']:
        before=path.read_bytes();path.write_bytes(before+b' ')
        try:
            with pytest.raises(ValueError,match='differ|bind'):module.replay_evidence(ref)
        finally:path.write_bytes(before)


def test_running_queue_is_rejected_before_capture_even_if_a_complete_file_exists(collected):
    x=collected;value=json.loads(x.queue.read_text());value['jobs']['timing-job']['status']='running'
    put(x.queue,value)
    with pytest.raises(ValueError,match='completed immutable'):module.capture_evidence(x.attempt,x.queue,x.evidence)
    assert not x.evidence.exists()


def test_v1_rehashed_completion_cannot_change_the_actual_native32_launch(collected):
    x=collected;path=x.root/'completion.json';original=json.loads(path.read_text())
    for index,change in enumerate(('spec','argv','environment')):
        value=deepcopy(original);launch=value['actual_launch'][0]
        if change=='spec':launch['spec']['max_num_seqs']=128
        elif change=='argv':launch['argv']+=['--max-num-seqs','128']
        else:launch['environment']['NCCL_CUMEM_ENABLE']='1'
        ref=put(path,value);execution=json.loads((x.attempt/'execution.json').read_text())
        execution['receipt_sha256']['native-timing/completion.json']=ref['sha256']
        put(x.attempt/'execution.json',execution)
        evidence=module.capture_evidence(x.attempt,x.queue,x.tmp/f'launch-tamper-{index}.json')
        with pytest.raises(ValueError,match='launch'):module.replay_evidence(evidence)


def rebind_collection(x,*,samples=False):
    """Refresh every upstream hash to test semantics, not stale bindings."""
    complete=json.loads((x.root/'completion.json').read_text())
    component=json.loads((x.root/'timing-component.json').read_text())
    if samples:
        refs=[binding(row['path']) for row in complete['raw_bindings']]
        complete['raw_bindings']=component['raw_bindings']=refs
    complete['timing_component']=put(x.root/'timing-component.json',component)
    cref=put(x.root/'completion.json',complete)
    execution=json.loads((x.attempt/'execution.json').read_text())
    execution['receipt_sha256']['native-timing/completion.json']=cref['sha256']
    put(x.attempt/'execution.json',execution)


def test_forged_coefficients_cannot_replace_raw_refit(collected):
    x=collected
    # Rebind every relevant artifact to demonstrate semantic replay, rather
    # than only exercising a stale SHA after editing a result.
    path=x.root/'timing-component.json';component=json.loads(path.read_text())
    component['models'][0]['coefficients'][0]+=1.
    cref=put(path,component);complete=json.loads((x.root/'completion.json').read_text());complete['timing_component']=cref
    complete_ref=put(x.root/'completion.json',complete);execution=json.loads((x.attempt/'execution.json').read_text())
    execution['receipt_sha256']['native-timing/completion.json']=complete_ref['sha256'];put(x.attempt/'execution.json',execution)
    ref=module.capture_evidence(x.attempt,x.queue,x.evidence)
    with pytest.raises(ValueError,match='coefficients/hull/holdout'):module.replay_evidence(ref)


def test_rehashed_interference_still_requires_actual_power_clocks_and_coverage(collected):
    x=collected;path=x.root/'interference/1500-0-pd-timing-0-parallel.json'
    original=json.loads(path.read_text())
    for index,kind in enumerate(('relative_power','power_source','power_gap','frequency','client_budget')):
        raw=deepcopy(original)
        if kind=='relative_power':
            for row in raw['power_samples']:row[1][0]*=1.10
            message='raw interference replay'
        elif kind=='power_source':
            raw['power_metadata'][2]['field_id']=[999];message='source/acquisition'
        elif kind=='power_gap':
            del raw['power_samples'][2:6];del raw['power_metadata'][2:6];message='power coverage'
        elif kind=='frequency':
            raw['frequency_samples'][3][1][0]=1800;message='frequency coverage'
        else:
            raw['client_requests'][0]['events'][0]['token_ids'].pop();message='token index'
        put(path,raw)
        # capture binds this mutated raw; all old qualification passed flags
        # remain true, so the following rejection requires independent audit.
        ref=module.capture_evidence(x.attempt,x.queue,x.tmp/f'evidence-{index}.json')
        with pytest.raises(ValueError,match=message):module.replay_evidence(ref)
    put(path,original)


def test_sample_point_filename_and_client_lifetime_are_independently_checked(collected):
    x=collected;first=x.component['raw_bindings'][0]['path'];second=x.component['raw_bindings'][1]['path']
    a,b=json.loads(Path(first).read_text()),json.loads(Path(second).read_text())
    put(Path(first),b);put(Path(second),a);rebind_collection(x,samples=True)
    ref=module.capture_evidence(x.attempt,x.queue,x.evidence)
    with pytest.raises(ValueError,match='bound filename'):module.replay_evidence(ref)
    put(Path(first),a);put(Path(second),b)
    a['sample']['ranks'][0]['samples'][0]['at_s']=a['client_requests'][0]['finished_s']+.05
    put(Path(first),a);rebind_collection(x,samples=True)
    ref=module.capture_evidence(x.attempt,x.queue,x.tmp/'lifetime-evidence.json')
    with pytest.raises(ValueError,match='client lifetime'):module.replay_evidence(ref)


def test_final_drain_cannot_precede_measured_samples(collected):
    x=collected;path=x.root/'completion.json';complete=json.loads(path.read_text())
    for row in complete['final_drains']:
        row.update(received_s=2000.,state=state(1,2000.,5),
                   drain=dict(state(1,2000.,5),acknowledged=True,drained=True))
    put(path,complete);rebind_collection(x)
    ref=module.capture_evidence(x.attempt,x.queue,x.evidence)
    with pytest.raises(ValueError,match='drain/worker completion order'):module.replay_evidence(ref)


def peer_fixture():
    identities={iid:dict(tp=1,gpu_uuids=[f'GPU-{i}']) for i,iid in enumerate(module.INSTANCES)}
    peers=[];raw={}
    for index,f in enumerate((1500,2520)):
        base=1000.+index*100.
        peers.append(dict(frequency_mhz=f,drains=[dict(instance_id=iid,received_s=base,
            drain=dict(state(1,base,5),acknowledged=True,drained=True),state=state(1,base,5)) for iid in identities],
            clocks=[dict(instance_id=iid,clock=dict(acknowledged=True,success=True,requested_frequency_mhz=f,
                at_s=base+.1,gpus=[dict(gpu_uuid=identities[iid]['gpu_uuids'][0],frequency_mhz=f)])) for iid in identities],
            observations=[dict(at_s=base+.2,frequencies_mhz=[f]*8)],settle_started_s=base+.3,observed_s=base+2.4))
        raw[str(f)]=dict(point=dict(frequency_mhz=f),measurement_started_s=base+3.,drain=dict(response_at_s=base+10.))
    return dict(interference_peer_states=peers),raw,identities


def test_optional_peer_receipts_do_not_rewrite_legacy_and_replay_new_bound_completion(collected):
    report,raw,identities=peer_fixture()
    assert module.audit_interference_peers({},raw,identities)['status']=='not_recorded_legacy'
    assert module.audit_interference_peers(report,raw,identities)['replayed']
    x=collected;complete=json.loads((x.root/'completion.json').read_text())
    peers=[]
    for f in (1500,2520):
        first=json.loads((x.root/f'interference/{f}-0-pd-timing-0-isolated.json').read_text())['measurement_started_s']
        example=deepcopy(report['interference_peer_states'][0]);example['frequency_mhz']=f
        base=first-5.
        example['drains']=[dict(instance_id=iid,received_s=base,
            drain=dict(state(1,base,5),acknowledged=True,drained=True),state=state(1,base,5)) for iid in identities]
        for row in example['clocks']:
            row['clock'].update(at_s=base+.1,requested_frequency_mhz=f)
            row['clock']['gpus'][0]['frequency_mhz']=f
        example.update(observations=[dict(at_s=base+.2,frequencies_mhz=[f]*8)],
            settle_started_s=base+.3,observed_s=base+2.4)
        peers.append(example)
    complete['interference_peer_states']=peers;put(x.root/'completion.json',complete);rebind_collection(x)
    ref=module.capture_evidence(x.attempt,x.queue,x.evidence)
    result=module.replay_evidence(ref)
    assert result['interference_peer_preparation']['replayed']
    assert not result['interference_peer_preparation']['continuous_idle_clock_coverage']


@pytest.mark.parametrize('kind',['empty','duplicate_frequency','missing_drain','busy_drain','missing_clock',
    'wrong_uuid','clock_ack','unstable','missing_gpu_observation','settle_short','before_clock','after_window','overlap_previous'])
def test_peer_preparation_requires_real_fleet_identity_stability_and_time_order(kind):
    report,raw,identities=peer_fixture();peers=report['interference_peer_states'];p=peers[0]
    if kind=='empty':peers.clear()
    elif kind=='duplicate_frequency':peers[1]['frequency_mhz']=1500
    elif kind=='missing_drain':p['drains'].pop()
    elif kind=='busy_drain':p['drains'][0]['state']['running']=['busy']
    elif kind=='missing_clock':p['clocks'].pop()
    elif kind=='wrong_uuid':p['clocks'][0]['clock']['gpus'][0]['gpu_uuid']='GPU-other'
    elif kind=='clock_ack':p['clocks'][0]['clock']['acknowledged']=False
    elif kind=='unstable':p['observations'][0]['frequencies_mhz'][3]=1800
    elif kind=='missing_gpu_observation':p['observations'][0]['frequencies_mhz'].pop()
    elif kind=='settle_short':p['observed_s']=1001.
    elif kind=='before_clock':p['observations'][0]['at_s']=999.
    elif kind=='after_window':p['observed_s']=1004.
    else:raw['1500']['drain']['response_at_s']=1101.
    with pytest.raises(ValueError):module.audit_interference_peers(report,raw,identities)


def test_explicit_container_paths_and_relocated_attempt_are_resolved_without_basename_guessing(collected):
    x=collected
    for name in ('completion.json','timing-component.json'):
        path=x.root/name;value=json.loads(path.read_text())
        for row in value['raw_bindings']:row['path']=row['path'].replace(str(x.attempt),'/output')
        if name=='timing-component.json':
            value['measurement_qualification']['receipt']['path']=value['measurement_qualification']['receipt']['path'].replace(str(x.attempt),'/output')
            component=put(path,value)
        else:put(path,value)
    complete=json.loads((x.root/'completion.json').read_text());complete['timing_component']=dict(component,path='/output/native-timing/timing-component.json')
    cref=put(x.root/'completion.json',complete);execution=json.loads((x.attempt/'execution.json').read_text())
    execution['receipt_sha256']['native-timing/completion.json']=cref['sha256'];put(x.attempt/'execution.json',execution)
    ref=module.capture_evidence(x.attempt,x.queue,x.evidence)
    assert module.replay_evidence(ref)['component']['component_qualified']
    relocated=x.tmp/'relocated'
    shutil.copytree(x.tmp,relocated,ignore=shutil.ignore_patterns('relocated'))
    shutil.rmtree(x.attempt)
    replay=module.replay_evidence(ref,path_map=[(str(x.tmp),str(relocated))])
    assert replay['component']['component_qualified']
    assert replay['evidence']['path']==str(relocated/x.evidence.name)


class Base:
    system='pdblend';model='Qwen2.5-7B-Instruct';tp=1;pp=1
    runtime_components=dict(capacity=False,transfer=False)
    query_qualification=dict(index_bytes=123)
    def step_seconds(self,*args):raise AssertionError('historical timing cannot be queried')
    def prefill_seconds(self,*args):raise AssertionError('historical timing cannot be queried')
    def prefill_power_w(self,n,f):
        if n>2048:raise ValueError('missing_profile: old power domain')
        return 100.
    def decode_power_w(self,batch,f,*,ctx):
        if batch>8:raise ValueError('missing_profile: old power domain')
        return 200.
    def wake_seconds(self,*args):raise ValueError('missing_profile: runtime unqualified')


def simple_replay():
    train,hold=component_rows()
    component=fit_component(train,hold,identity=dict(system='pdblend',model_id='Qwen2.5-7B-Instruct',tp=1,pp=1),
        raw_bindings=[],measurement_qualification=dict(qualified=True),
        limits=dict(mean_relative_error=.1,p95_relative_error=.2,max_relative_error=.25))
    return dict(component=component,formal_eligible=False,evidence=dict(path='test',sha256='test'))


def test_facade_uses_native_milliseconds_as_seconds_and_preserves_power_runtime_gates():
    value=NativeTimingOverlay(Base(),simple_replay())
    assert value.step_seconds(8,1024,1500)==pytest.approx((1+8+1)/1000)
    assert value.prefill_seconds(1024,2520)==pytest.approx((1+.125+.125**2)/1000)
    assert value.prefill_marginal_seconds(1024,2520)==pytest.approx((.125+.125**2)/1000)
    assert value.token_energy_j(8,1024,1500)==pytest.approx(.01*200/8)
    assert value.prefill_energy_j(1024,2520)==pytest.approx((1+.125+.125**2)/10)
    for query in [(8,1024,1800),(33,1024,1500),(8,4097,1500),(8,63,2520),(True,1024,1500)]:
        assert not value.decode_supported(*query)
        with pytest.raises(ValueError):value.step_seconds(*query)
    with pytest.raises(ValueError,match='exact'):value.nearest_freq(1499)
    with pytest.raises(ValueError,match='power domain'):value.prefill_energy_j(4096,1500)
    with pytest.raises(ValueError,match='power domain'):value.token_energy_j(16,1024,1500)
    with pytest.raises(ValueError,match='runtime'):value.wake_seconds('parked')
    assert value.runtime_components==Base.runtime_components


def test_attachment_replays_evidence_and_cannot_promote_base_qualification(collected):
    x=collected;ref=module.capture_evidence(x.attempt,x.queue,x.evidence)
    qualification=dict(usage='development',formal_eligible=False,full_profile_qualified=False,
        missing_gates=['power','runtime','formal_workload_energy'],runtime_components=dict(capacity=False))
    loaded=LoadedVersion(Base(),dict(system='pdblend',**x.identity),{},qualification,{'existing':'preserve'})
    result=attach_native_timing(loaded,ref)
    assert result.qualification['missing_gates']==qualification['missing_gates']
    assert result.qualification['runtime_components']==qualification['runtime_components']
    assert result.qualification['native_timing_replay_passed'] and not result.qualification['formal_eligible']
    assert result.profile_key['existing']=='preserve' and loaded.profile_key=={'existing':'preserve'}
    assert not result.model.decode_supported(1,64,1500), 'actual decode contexts start after prompt prefill'
    formal=LoadedVersion(Base(),loaded.identity,{},dict(qualification,usage='formal'),{})
    with pytest.raises(ValueError,match='development-only'):attach_native_timing(formal,ref)
