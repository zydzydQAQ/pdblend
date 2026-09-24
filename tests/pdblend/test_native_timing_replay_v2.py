"""Complete TP2 timing/capacity evidence with real reducers and fitting, CPU only."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest

from pdblend.profile.collection import native_timing_replay as entry
from pdblend.profile.collection import native_timing_replay_v2 as module
from pdblend.profile.collection.native_timing_plan import binding,digest
from pdblend.profile.collection.native_timing_audit import audit_window
from pdblend.profile.collection.native_timing_capacity import (capacity_decision,unsupported_capacity_record,
    partition_windows,fit_measured_partition)
from test_native_timing_replay import collected,put
from test_native_timing_capacity import plans
from test_comparison_acceptance import state


@pytest.fixture
def collected_v2(collected,plans):
    x=collected;plan=plans['plans']['Qwen2.5-32B-Instruct'];ids=tuple(f'pd-timing-{i}' for i in range(4));tp=2
    inputs=json.loads(Path(x.inputs_ref['path']).read_text());identity=dict(x.identity,model_id=plan['model_id'],tp=tp)
    verification=json.loads(Path(inputs['model_verification']['path']).read_text())
    verification['models']['7b']['model_id']=plan['model_id']
    inputs['model_verification']=put(Path(inputs['model_verification']['path']),verification)
    inputs.update(schema='pdblend-native-timing-inputs/v2',model_id=plan['model_id'],point_plan=put(x.tmp/'v2-plan.json',plan))
    x.inputs_ref=put(Path(x.inputs_ref['path']),inputs)
    x.caps.clear();x.caps.update({iid:dict(identity,gpu_uuids=[f'GPU-{2*i}',f'GPU-{2*i+1}']) for i,iid in enumerate(ids)})
    for name in ('samples','interference'):shutil.rmtree(x.root/name)
    def raw(point,index,base,name):
        value=x.raw_factory(point,index,base,name)
        value['drain']=dict(state(tp,base+10.,5),acknowledged=True,drained=True)
        original=value['sample']['ranks'][0]['samples'];value['sample']['ranks']=[]
        for rank in range(tp):
            events=[dict(deepcopy(row),tp=tp,rank=rank) for row in original]
            value['sample']['ranks'].append(dict(rank=rank,samples=events))
        value['measurement_stop']['ranks']=[dict(rank=rank,acknowledged=True) for rank in range(tp)]
        value['clock_receipt']['gpus']=[dict(gpu_uuid=u) for u in x.caps[ids[index]]['gpu_uuids']]
        value['frequency_samples']=[(t,[v[0]]*tp) for t,v in value['frequency_samples']]
        value['power_samples']=[(t,[v[0]]*tp) for t,v in value['power_samples']]
        value['power_metadata']=[dict(gpus=list(range(index*tp,(index+1)*tp)),
            **{k:v*tp for k,v in row.items() if k!='gpus'}) for row in value['power_metadata']]
        return value
    now=1000.;checks=[];peers=[]
    for f in (1500,2520):
        peers.append(dict(frequency_mhz=f,drains=[dict(instance_id=iid,received_s=now,
            drain=dict(state(tp,now,5),acknowledged=True,drained=True),state=state(tp,now,5)) for iid in ids],
            clocks=[dict(instance_id=iid,clock=dict(acknowledged=True,success=True,requested_frequency_mhz=f,at_s=now+.1,
                gpus=[dict(gpu_uuid=u,frequency_mhz=f) for u in x.caps[iid]['gpu_uuids']])) for iid in ids],
            observations=[dict(at_s=now+.2,frequencies_mhz=[f]*8)],settle_started_s=now+.3,observed_s=now+2.4))
        now+=4.
        for repeat in range(3):
            p=dict(role='decode',batch=8,prompt_tokens=1024,output_tokens=64,
                   frequency_mhz=f,purpose='interference',seed=9701,repeat=repeat)
            for index,iid in enumerate(ids):
                name=f'{f}-{repeat}-{iid}-isolated';put(x.root/'interference'/(name+'.json'),raw(p,index,now,name));now+=12.
            for index,iid in enumerate(ids):
                name=f'{f}-{repeat}-{iid}-parallel';put(x.root/'interference'/(name+'.json'),raw(p,index,now,name))
                checks.append(dict(instance_id=iid,frequency_mhz=f,repeat=repeat,relative_errors=dict(latency_ms=0.,power_w=0.),
                    common_window_s=6.,passed=True))
            now+=12.
    uuids=[f'GPU-{i}' for i in range(8)]
    qualification=dict(qualified=True,parallel_qualified=True,mode='parallel',limit=.05,checks=checks,
        exclusive_fleet_gpu_uuids=uuids,energy_comparable=False)
    qref=put(x.root/'measurement-qualification.json',qualification);refs=[];owners=[];windows=[]
    for index,point in enumerate(plan['points']):
        owner=index%4;iid=ids[owner]
        for repeat in range(3):
            p=dict(point,repeat=repeat);name=f'{digest(point)[:20]}-{repeat}'
            idle=dict(state(tp,now,5),total_kv_tokens=16384,free_kv_tokens=16384)
            args=dict(identity=x.caps[iid],capability=dict(x.caps[iid],supported=True,state=deepcopy(idle)),
                capability_received_s=now,drain=dict(idle,acknowledged=True,drained=True),drain_received_s=now,observed_s=now+.1)
            if capacity_decision(plan,p,**args)['supported']:
                value=raw(p,owner,now,name);now+=12.
            else:
                value=unsupported_capacity_record(plan,p,**args);now+=1.
            ref=put(x.root/'samples'/(name+'.json'),value);refs.append(ref);owners.append(dict(instance_id=iid,raw=ref))
            windows.append(dict(instance_id=iid,raw=value))
    partition=partition_windows(plan,iter(windows),identities=x.caps)
    fitted=fit_measured_partition(partition,identity=dict(system='pdblend',**identity),raw_bindings=refs,
        measurement_qualification=dict(qualification,receipt=qref),limits=plan['holdout_limits'])
    from pdblend_runtime.probe import NativeSpec
    specs=[NativeSpec(iid,tuple(range(i*tp,(i+1)*tp)),20000+i*4,'/models/'+plan['model_id'],tp=tp,
        max_num_seqs=32,generation=5,extra_args=('--enforce-eager','--worker-cls',
            'pdblend.profile.collection.native_timing_worker.PDNativeTimingWorker')) for i,iid in enumerate(ids)]
    complete=dict(schema='pdblend-native-timing-collection/v2',system='pdblend',status='passed',complete=True,
        hardware_executed=True,cleanup_errors=[],raw_bindings=refs,window_owners=owners,
        capabilities={iid:dict(cap,supported=True,state=state(tp,900.,5)) for iid,cap in x.caps.items()},
        point_plan=inputs['point_plan'],capacity_policy=plan['capacity_policy'],interference_peer_states=peers,
        measurement_qualification=qref,window_partition=put(x.root/'window-partition.json',partition),
        timing_component=put(x.root/'timing-component.json',fitted),component_qualified=fitted['component_qualified'],
        unsupported_windows=len(partition['unsupported']),measured_windows=len(partition['measured']),
        actual_launch=[dict(spec=json.loads(json.dumps(asdict(s))),argv=s.command(),environment=dict(
            CUDA_VISIBLE_DEVICES=','.join(str(g) for g in s.gpus),VLLM_USE_V1='1',NCCL_CUMEM_ENABLE='0',
            NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='0')) for s in specs],
        final_drains=[dict(instance_id=iid,received_s=now+1.,drain=dict(state(tp,now+1.,5),acknowledged=True,drained=True),
            state=state(tp,now+1.,5)) for iid in ids],
        physical_cleanup=dict(passed=True,started_s=now+2.,finished_s=now+2.2,observations=[dict(at_s=now+2.1,
            devices=[dict(gpu=i,gpu_uuid=u,compute_pids=[]) for i,u in enumerate(uuids)])]),
        actual_engine_starts={iid:[dict(instance=iid,kind='start',pid=1000+i,t_s=900.)] for i,iid in enumerate(ids)},engine_loads=4)
    cref=put(x.root/'completion.json',complete)
    manifest=json.loads((x.attempt/'manifest.json').read_text());manifest['payload']['input_manifest']=x.inputs_ref
    put(x.attempt/'manifest.json',manifest)
    queue=json.loads(x.queue.read_text());queue['jobs']['timing-job']['payload']=manifest['payload'];put(x.queue,queue)
    execution=json.loads((x.attempt/'execution.json').read_text());execution.update(finished_s=now+3.,
        receipt_sha256={'native-timing/completion.json':cref['sha256']});put(x.attempt/'execution.json',execution)
    x.v2_plan=plan;x.v2_partition=partition;x.v2_fitted=fitted;x.v2_complete=complete
    return x


def rebind_completion(x,value):
    ref=put(x.root/'completion.json',value);execution=json.loads((x.attempt/'execution.json').read_text())
    execution['receipt_sha256']['native-timing/completion.json']=ref['sha256'];put(x.attempt/'execution.json',execution)


def test_tp2_complete_raw_replay_keeps_capacity_exclusions_out_of_fit_and_resolves_relocation(collected_v2):
    x=collected_v2;ref=entry.capture_evidence(x.attempt,x.queue,x.evidence)
    assert json.loads(x.evidence.read_text())['schema']==module.SCHEMA
    result=entry.replay_evidence(ref)
    assert result['identity']['tp']==2 and result['replayed_interference_windows']==48
    assert result['measured_windows']>0 and result['unsupported_windows']>0
    assert result['replayed_windows']==sum(p['repeats'] for p in x.v2_plan['points'])
    assert result['supported_fit']==x.v2_fitted and not result['formal_eligible']
    with pytest.raises(ValueError,match='overwrite'):entry.capture_evidence(x.attempt,x.queue,x.evidence)
    relocated=x.tmp/'relocated';shutil.copytree(x.tmp,relocated,ignore=shutil.ignore_patterns('relocated'))
    shutil.rmtree(x.attempt)
    assert entry.replay_evidence(ref,path_map=[(str(x.tmp),str(relocated))])['supported_fit']==result['supported_fit']


def test_rehashed_owner_partition_counts_and_cleanup_claims_cannot_replace_raw_v2_replay(collected_v2):
    x=collected_v2;original=deepcopy(x.v2_complete)
    for index,kind in enumerate(('owner','partition','count','peer','cleanup','load_count','capacity',
                                  'launch_limit','launch_argv','launch_env','launch_capability')):
        complete=deepcopy(original);restore=None
        if kind=='owner':complete['window_owners'][0]['instance_id']='pd-timing-3';message='owner inventory'
        elif kind=='partition':
            path=x.root/'window-partition.json';restore=(path,path.read_bytes());value=json.loads(path.read_text())
            value['unsupported'].pop();complete['window_partition']=put(path,value);message='partition does not reproduce'
        elif kind=='count':complete['unsupported_windows']-=1;message='counts differ'
        elif kind=='peer':complete.pop('interference_peer_states');message='peer preparation'
        elif kind=='cleanup':complete['physical_cleanup']['observations'][-1]['devices'][0]['compute_pids']=[99];message='was not empty'
        elif kind=='load_count':complete['engine_loads']=5;message='start/load inventory'
        elif kind=='launch_limit':
            complete['actual_launch'][0]['spec']['max_num_seqs']=128;message='complete native launch'
        elif kind=='launch_argv':
            complete['actual_launch'][0]['argv'].append('--enable-prefix-caching');message='complete native launch'
        elif kind=='launch_env':
            complete['actual_launch'][0]['environment']['NCCL_P2P_DISABLE']='1';message='launch environment'
        elif kind=='launch_capability':
            complete['capabilities']['pd-timing-0']['state']['max_num_seqs']=128;message='capability limits'
        else:
            owner=next(row for row in complete['window_owners'] if json.loads(Path(row['raw']['path']).read_text())['status']=='unsupported_capacity')
            path=Path(owner['raw']['path']);restore=(path,path.read_bytes());raw=json.loads(path.read_text())
            raw['capacity_decision']['reserved_tokens']-=16;ref=put(path,raw)
            complete['raw_bindings']=[ref if r['path']==str(path) else r for r in complete['raw_bindings']]
            owner['raw']=ref;message='inequality/receipt'
        rebind_completion(x,complete)
        try:
            ref=entry.capture_evidence(x.attempt,x.queue,x.tmp/f'v2-evidence-{index}.json')
            with pytest.raises(ValueError,match=message):entry.replay_evidence(ref)
        finally:
            if restore:restore[0].write_bytes(restore[1])
    rebind_completion(x,original)
