import asyncio
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import bind
import gate_evidence as g

def instances(hetero=False):
    geometry=[(1,[n]) for n in range(5)]+[(2,[6,7])] if hetero else [(1,[n]) for n in range(8)]
    return [dict(id='i'+str(n),tp=tp,gpus=gpus,role='mixed',native_kind='legacy_sync_put',scheduler_cache_observed=False,
        container=dict(id='container'+str(n),name='pdb-v2-i'+str(n),image=bind.IMAGE,StartedAt='fixed'),
        provenance=dict(instance_id='i'+str(n),pid=100+n,tp=tp,model='/models/Qwen2.5-14B-Instruct',source_files_at_import={'source.py':'sha'}))
        for n,(tp,gpus) in enumerate(geometry)]

def rank():return dict(buffered_tensors=0,buffered_gpu_bytes=0,inflight_receives=0,listener_alive=True,allocations={})
def raw(i,generation=5):
    return dict(id=i['id'],generation=generation,acknowledged_generation=generation,transport_healthy=True,timestamp=1001.,transfer_observed_s=1001.,
        active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={},transfer_buffered_tensors=0,transfer_inflight_receives=0,
        role='mixed',mode='continuous',admit_prefill=True,admit_decode=True,accepting=True)

def fixture(hetero=False,temporal_exact=True):
    ins=instances(hetero);checks=dict(complete=True,checks={},ordinary={},pd=[],requests=[],cancelled_unconsumed_kv=[]);http=[];events={i['id']:[] for i in ins}
    def add(i,n,label,rid=None,count=64,values=None):
        rid=rid or 'own-'+str(len(checks['requests']));values=values or list(range(count))
        body=dict(prompt=([9707,1879,13]*(n//3+1))[:n],max_tokens=count,temperature=0,top_p=1,ignore_eos=True,seed=0,stream=False)
        response=dict(token_ids=values,usage=dict(prompt_tokens=n,completion_tokens=count))
        row=dict(instance_id=i['id'],request_id=rid,label=label,body=body,response=response,dispatch_s=1001.,finished_s=1002.)
        checks['requests'].append(row);http.append(dict(instance=i['id'],route='/v1/completions',request_id=rid,body=body,response=response,status=200))
        return row,values
    for n in (128,7168):
        if hetero:
            a=[add(i,n,'ordinary-tp1-'+str(n))[1] for i in ins[:-1]]
            b=[add(ins[-1],n,'ordinary-tp2-local-repeat-'+str(j),values=list(range(1,65)))[1] for j in range(2)]
            checks['ordinary'][str(n)]=dict(tp1_token_ids=a,tp2_token_ids=b)
        else:checks['ordinary'][str(n)]=dict(token_ids=[add(i,n,'ordinary-'+str(n))[1] for i in ins])
    pairs=[(s,ins[-1]) for s in ins[:-1]] if hetero else [(ins[0],d) for d in ins[1:]]
    for s,d in pairs:
        for n in (128,7168):
            rid=f'pdb:n{n}:p:{s["id"]}:{d["id"]}';prefix='cross-tp-' if hetero else 'pd-'
            add(s,n,prefix+'producer',rid,count=1);vals=list(range(1,65)) if hetero else list(range(64))
            add(d,n,prefix+'consumer',rid.replace(':p:',':d:'),values=vals)
            checks['pd'].append(dict(source=s['id'],target=d['id'],prompt_length=n,token_ids=vals))
        response=dict(transfers=[rank() for _ in range(d['tp'])]);rid=f'pdb:cancel:d:{s["id"]}:{d["id"]}'
        checks['cancelled_unconsumed_kv'].append(dict(source=s['id'],target=d['id'],request_id=rid,result=response))
        http.append(dict(instance=d['id'],route='/cancel',body=dict(request_id=rid),response=response,status=200))
    if not hetero:
        refs=[add(ins[1],n,'temporal-single-reference')[1] for n in (96,192)]
        first,x=add(ins[1],96,'temporal-first');second,y=add(ins[1],192,'temporal-second',values=list(range(64)) if temporal_exact else list(range(1,65)))
        phase=dict(complete=True,reference_token_ids=refs,token_ids=[x,y],first_allocated=dict(kv_allocations={first['request_id']:1},running=1),
            held=dict(kv_allocations={},waiting=1,admit_prefill=False))
        for label,value,gen in [('close_prefill',False,2),('open_prefill',True,3)]:
            phase[label]=dict(before=dict(generation=gen-1),command=dict(generation=gen,mode='temporal',admit_prefill=value),after=dict(generation=gen,acknowledged_generation=gen))
        checks['temporal']=phase
        for row in (first,second):
            events[ins[1]['id']]+=[dict(mode='temporal',tokens=1,prefill=p,decode=1-p,request_ids=[row['request_id']]) for p in (1,0)]
    cleanup={}
    for i in ins:
        after=raw(i,6);cleanup[i['id']]=dict(complete=True,errors=[],proof=dict(drained=True,accepting=False,generation=5,
            drain_proof_type='synchronous_put_owner_barrier',transfers=[rank() for _ in range(i['tp'])]),
            resume=dict(before=raw(i),command={k:after[k] for k in ('generation','role','mode','admit_prefill','admit_decode')},after=after),after=after)
    checks['cleanup']=dict(complete=True,errors=[],instances=cleanup)
    return ins,checks,http,events

@pytest.mark.parametrize('hetero',[False,True])
def test_reconstructs_exact_requests_and_native_without_fabricated_sendcounter(hetero):
    ins,c,h,e=fixture(hetero);result,errors=g.mechanisms(c,h,e,ins,hetero)
    assert all(result.values()) and not errors
    g.native(c,ins)

def test_temporal_failure_preserves_ordinary_pd_permission():
    ins,c,h,e=fixture(temporal_exact=False);actual,errors=g.mechanisms(c,h,e,ins)
    assert actual==dict(ordinary=True,pd=True,temporal=False) and 'temporal' in errors

@pytest.mark.parametrize('kind',['response','usage','duplicate','missing_route','missing_rank','fake_temporal_owner'])
def test_raw_counterexamples_refuse_affected_mechanism(kind):
    ins,c,h,e=fixture()
    if kind=='response':c['requests'][0]['response']['token_ids'][0]=-1
    elif kind=='usage':c['requests'][0]['response']['usage']['completion_tokens']=63
    elif kind=='duplicate':c['requests'].append(copy.deepcopy(c['requests'][0]))
    elif kind=='missing_route':c['pd'].pop()
    elif kind=='missing_rank':c['cancelled_unconsumed_kv'][0]['result']['transfers']=[]
    else:e[ins[1]['id']]=[]
    if kind=='duplicate':
        with pytest.raises(RuntimeError,match='duplicate'):g.mechanisms(c,h,e,ins)
    else:
        actual,_=g.mechanisms(c,h,e,ins)
        assert not all(actual.values())

@pytest.mark.parametrize('kind',['proof','resume','rank','stale'])
def test_cleanup_headers_cannot_replace_actual_native_proof(kind):
    ins,c,_,_=fixture();r=c['cleanup']['instances']['i0']
    if kind=='proof':r['proof']['drained']=False
    elif kind=='resume':r['after']['accepting']=False
    elif kind=='rank':r['proof']['transfers'][0].pop('buffered_gpu_bytes')
    else:r['after']['transfer_observed_s']=990
    with pytest.raises(RuntimeError):g.native(c,ins)

def test_identity_foreign_pid_and_sources_refused(tmp_path):
    ins,c,_,_=fixture();rows=[]
    for i in ins:rows.append(dict(provenance=i['provenance'],runtime=raw(i),container=dict(Id=i['container']['id'],Image=bind.IMAGE,
        State=dict(StartedAt='fixed',Running=True,Pid=42),Config={},HostConfig={},Mounts=[])))
    bind.write(tmp_path/'identity.before.json',rows);bind.write(tmp_path/'identity.after.json',rows)
    g.identities(tmp_path,ins)
    rows[0]['provenance']['pid']=999
    (tmp_path/'identity.after.json').write_text(json.dumps(rows))
    with pytest.raises(RuntimeError,match='model/source/PID'):g.identities(tmp_path,copy.deepcopy(instances()))

def test_mount_enumeration_only_is_normalized(tmp_path):
    ins=instances();rows=[]
    for i in ins:rows.append(dict(provenance=i['provenance'],runtime=raw(i),container=dict(Id=i['container']['id'],Image=bind.IMAGE,
        State=dict(StartedAt='fixed',Running=True,Pid=42),Config={},HostConfig={},Mounts=[dict(Source='/a',Destination='/a',RW=False),dict(Source='/b',Destination='/b',RW=True)])))
    bind.write(tmp_path/'identity.before.json',rows)
    for r in rows:r['container']['Mounts'].reverse()
    bind.write(tmp_path/'identity.after.json',rows);g.identities(tmp_path,ins)
    rows[0]['container']['Mounts'][0]['RW']=False;(tmp_path/'identity.after.json').write_text(json.dumps(rows))
    with pytest.raises(RuntimeError,match='mounts'):g.identities(tmp_path,ins)

def test_all8_energy_and_clock_raw_are_required(tmp_path):
    power=tmp_path/'power';power.mkdir()
    with (power/'power.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_w' for i in range(8)]);w.writerows([[1000]+[100]*8,[1004]+[100]*8])
    with (power/'clocks.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)]);w.writerows([[1000]+[1500]*8,[1004]+[2520]*8])
    bind.write(power/'power_source.json',{});(power/'power_metadata.jsonl').write_text('{}\n{}\n')
    status=dict(measurement_start_s=1001.,measurement_end_s=1003.,full_operation_energy_j=1600.)
    assert g.power(tmp_path,status,lambda *a:dict(power_source_verified=True))['all8_energy_j']==1600
    status['full_operation_energy_j']=200.
    with pytest.raises(RuntimeError,match='energy differs'):g.power(tmp_path,status,lambda *a:dict(power_source_verified=True))

@pytest.mark.parametrize('hetero,temporal_ok',[(False,True),(False,False),(True,True)])
def test_complete_raw_gate_reconstruction_with_real_power_provenance(tmp_path,hetero,temporal_ok):
    from ecopadg.serving.measurement import power_evidence
    ins,c,h,e=fixture(hetero,temporal_ok);actual,_=g.mechanisms(c,h,e,ins,hetero)
    c['checks']=dict(ordinary_cross_replica_exact=actual['ordinary'],pd_exact_all_declared_pairs=actual['pd'],cancel_all_tp_ranks=actual['pd'])
    if not hetero:c['checks']['temporal_exact']=actual['temporal']
    bind.write(tmp_path/'checks/checks.json',c);(tmp_path/'checks/http.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in h))
    identities=[];refs={}
    for i in ins:
        identities.append(dict(provenance=i['provenance'],runtime=raw(i),container=dict(Id=i['container']['id'],Image=bind.IMAGE,
            State=dict(StartedAt='fixed',Running=True,Pid=42),Config={},HostConfig={},Mounts=[])))
        path=tmp_path/(i['id']+'.control.events.jsonl');path.write_text(''.join(json.dumps(x)+'\n' for x in e[i['id']]))
        refs['/original/runtime/'+path.name]=dict(offset_start=10,offset_end=10+path.stat().st_size,sha256=g.sha(path))
    bind.write(tmp_path/'identity.before.json',identities);bind.write(tmp_path/'identity.after.json',identities)
    p=tmp_path/'power';p.mkdir();source=dict(mode='instant',source_id='nvml:field:186:scope:0:mW',field_id=186,scope_id=0)
    bind.write(p/'power_source.json',source)
    with (p/'power.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_w' for i in range(8)]);w.writerows([[1000]+[100]*8,[1004]+[100]*8])
    with (p/'clocks.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)]);w.writerows([[1000]+[1500]*8,[1004]+[2520]*8])
    metadata=[]
    for t in (1000.,1004.):
        metadata.append(dict(t_s=t,gpus=list(range(8)),**{k:[v]*8 for k,v in dict(**source,value_type=1,return_code=0).items()},
            nvml_timestamp_us=[int(t*1e6)]*8,nvml_latency_us=[0]*8,read_started_s=[t]*8,read_finished_s=[t]*8))
    (p/'power_metadata.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in metadata))
    bind.write(tmp_path/'status.json',dict(complete=True,finished_s=1004.,measurement_valid=True,native_cleanup_complete=True,clock_restore_complete=True,
        cleanup_errors=[],sampling_error=None,measurement_start_s=1001.,measurement_end_s=1003.,full_operation_energy_j=1600.,events=refs,
        mechanism_gate=actual,passed=all(actual.values())))
    result,hashes=g.audit(tmp_path,ins,'distserve' if hetero else 'mixed',power_evidence,hetero)
    assert result['physical']['all8_energy_j']==1600 and len(hashes)>10
    if not hetero and not temporal_ok:
        assert not result['overall_runtime_gate_passed'] and result['verified']['ordinary']
        with pytest.raises(RuntimeError,match='required mechanism failed'):g.audit(tmp_path,ins,'ecoserve',power_evidence)

def test_dataset_and_implementation_scope():
    a=dict(model='14b',layout='resident');c=dict(model='7b',layout='resident');h=dict(model='14b',layout='distserve-longbench')
    assert bind.selected(a,None,None)==[] and bind.selected(a,'distserve',['alpaca','sharegpt'])==['alpaca','sharegpt']
    assert bind.selected(h,'distserve',['longbench'])==['longbench']
    for spec,strategy,ds in [(a,'distserve',None),(c,'dynamollm',None),(h,'mixed',['longbench']),(h,'distserve',['alpaca'])]:
        with pytest.raises(RuntimeError):bind.selected(spec,strategy,ds)

def actual_instances(spec):
    return [dict(i,container=dict(name=i['container_name'])) for i in spec['instances']]

def test_actual_A_pd_roles_and_full_dynamo_are_preserved(tmp_path):
    spec=bind.read(bind.WORKSPACE/'campaign/AC-baseline-deployment-prepared-v1/A-resident/deployment.json');ins=actual_instances(spec)
    cfgs,notes=bind.configurations(spec,ins,'distserve',['alpaca','sharegpt'],tmp_path/'dist',tmp_path/'results',{})
    assert [sum(i['role']=='prefill' for i in bind.read(cfgs[d])['instances']) for d in ('alpaca','sharegpt')]==[1,2]
    assert [bind.read(cfgs[d])['output_prior'] for d in ('alpaca','sharegpt')]==[314,512]
    cfgs,_=bind.configurations(spec,ins,'dynamollm',['alpaca'],tmp_path/'dynamo',tmp_path/'results',{})
    cfg=bind.read(cfgs['alpaca']);template=bind.read(cfg['topology']['engine_template'])
    assert cfg['strategy']=='dynamollm' and cfg['topology_costs'] and cfg['dynamo_assignments']
    assert template['observation_engine_sha256']==bind.sha(spec['source_entry'])

def test_bootstrap_is_unusable_as_performance_and_failure_writes_no_config(tmp_path,monkeypatch):
    spec=dict(model='7b',layout='resident',hostname=bind.socket.gethostname(),files={},executor_release='/unused',pdb_binding=str(tmp_path/'pdb.json'),
        instances=[],protocol_id='test',deadline_s=1,host_release='/unused')
    bind.write(tmp_path/'spec.json',spec);bind.write(tmp_path/'receipt.json',dict(complete=True,measurement_valid=True));bind.write(tmp_path/'pdb.json',dict(files={}))
    async def live(*a):return [],[]
    async def identity(*a):return []
    executor=SimpleNamespace(validate_binding=lambda b:None,identity=identity)
    monkeypatch.setattr(bind,'scope',lambda s:None);monkeypatch.setattr(bind,'live',live);monkeypatch.setattr(bind,'package_files',lambda p:{})
    monkeypatch.setattr(bind,'load',lambda *a:executor)
    a=SimpleNamespace(spec=tmp_path/'spec.json',receipt=tmp_path/'receipt.json',out=tmp_path/'bootstrap',output=None,strategy=None,gate=None,dataset=None)
    b=asyncio.run(bind.build(a));assert b['configs']=={} and b['implementation_variant']=='correctness-only' and not b['output_correctness_verified']
    assert not (a.out/'configs').exists()
    a.out=tmp_path/'bad-performance';a.strategy='mixed';a.gate=tmp_path/'gate'
    monkeypatch.setattr(bind,'audit',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('failed actual gate')))
    with pytest.raises(RuntimeError,match='failed actual gate'):asyncio.run(bind.build(a))
    assert not a.out.exists()
