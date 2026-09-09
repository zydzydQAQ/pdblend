"""Focused CPU evidence; synthetic processes/HTTP are never actual deployment."""
import asyncio
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import contracts as c
import third
spec=importlib.util.spec_from_file_location('b_static3_restore_test',ROOT/'restore.py')
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)

@pytest.fixture
def declaration(tmp_path):return c.build(tmp_path/'unused')

def test_actual_geometry_and_config_unchanged(declaration):
    s=declaration
    assert [(i['tp'],i['gpus']) for i in s['instances']+[s['third_instance']]]==[(2,[0,1]),(2,[2,3]),(2,[4,5])]
    assert len(s['expected_containers'])==2 and s['fresh_three_replica_correctness_required']
    assert not s['performance_authorized'] and not s['scale_completion_asserted']
    assert not Path(s['out']).exists()
    old=c.read(s['instances'][0]['config']);new=c.read(s['third_instance']['config'])
    for k in set(old)|set(new):
        if k not in ('id','port','kv_port','runtime_dir','generation','peers'):assert old.get(k)==new.get(k),k

def terminal_fixture():
    s=dict(model='32b',spec_sha256=c.SCALE_SPEC_SHA,release_sha256=c.RELEASE_SHA,protocol_id=c.PROTOCOL,
       deadline_s=c.DEADLINE,complete=True,phase='selected_scale_groups_finished',pid=100,started_s=1.,finished_s=3.,
       steps=[dict(pid=101,complete=True,exitcode=0)])
    p=dict(model='32b',complete=True,phase='complete',deadline_s=c.DEADLINE,started_s=.5,finished_s=4.,pid=102,
       observer_pid=103,scale_pid=100,steps=[dict(pid=104,complete=True,exitcode=0)],
       scale_spec=dict(path=str(c.SCALE_SPEC),sha256=c.SCALE_SPEC_SHA),release=dict(path=str(c.RELEASE),sha256=c.RELEASE_SHA))
    return s,p

@pytest.mark.parametrize('bad',[None,'scale_active','producer_live','observer_live','failed_step','wrong_source','late','parent_active'])
def test_terminal_boundary(bad):
    s,p=terminal_fixture();live=lambda pid:pid==({'producer_live':101,'observer_live':103}.get(bad,-1))
    if bad=='scale_active':s['complete']=False
    if bad=='failed_step':s['steps'][0]['exitcode']=1
    if bad=='wrong_source':s['spec_sha256']='0'*64
    if bad=='late':s['finished_s']=c.DEADLINE+1
    if bad=='parent_active':p['complete']=False
    if bad is None:assert c.terminal(s,p,live)
    else:
        with pytest.raises(RuntimeError):c.terminal(s,p,live)

def test_actual_incomplete_scale_before_deep_audit(monkeypatch):
    actual=c.read(c.HANDOFF/'scale/status.json')
    if actual.get('complete'):pytest.skip('actual scale now complete; synthetic refusal remains')
    monkeypatch.setattr(c,'contract',lambda:pytest.fail('incomplete predecessor must refuse before deep audit'))
    with pytest.raises(RuntimeError,match='not completely finished'):c.completed_scale()

def created_fixture(s):
    row=copy.deepcopy(next(iter(s['expected_containers'].values())))
    f=s['third_creation'];row.update(Name='/'+s['third_instance']['container_name'],Id='new-only-id',Image=f['image'],Path='python3')
    row['Config'].update(Cmd=f['entry_argv'],Env=f['env'],Labels={'pdblend.static3.spec':third.label(s)})
    row['HostConfig']=copy.deepcopy(f['host_config'])
    row['State'].update(Running=True,Pid=123,StartedAt='2026-09-08T05:00:00Z',Paused=False,Restarting=False,Dead=False)
    return row

def test_real_argv_absolute_same_v3_no_rm(declaration):
    argv=third.argv(declaration);entry=declaration['third_creation']['entry_argv']
    assert argv[-len(entry):]==entry and '-m' not in argv and 'rm' not in argv
    assert 'PYTHONPATH=/root/workspace/pdblend-next-v1/releases/io-v3-runtime/src' in argv
    assert 'CUDA_VISIBLE_DEVICES=4,5' in argv
    third.validate_created(declaration,created_fixture(declaration))

@pytest.mark.parametrize('bad',['image','label','duplicate_env','mount','hostconfig'])
def test_new_identity_rejects_differences(declaration,bad):
    row=created_fixture(declaration)
    if bad=='image':row['Image']='other'
    if bad=='label':row['Config']['Labels']={}
    if bad=='duplicate_env':row['Config']['Env']=row['Config']['Env']+[row['Config']['Env'][0]]
    if bad=='mount':row['Mounts'][0]['Source']='/other'
    if bad=='hostconfig':row['HostConfig']['AutoRemove']=True
    with pytest.raises(RuntimeError):third.validate_created(declaration,row)

@pytest.mark.parametrize('bad',[None,'duplicate_rank','missing_rank','wrong_peer','boolean_rank'])
def test_both_actual_peer_ranks(bad):
    rows=[dict(rank=i,peers=['nextv3b0','nextv3b1']) for i in (0,1)]
    if bad=='duplicate_rank':rows[1]['rank']=0
    if bad=='missing_rank':rows.pop()
    if bad=='wrong_peer':rows[1]['peers']=['other']
    if bad=='boolean_rank':rows[1]['rank']=True
    if bad is None:third.peer_reply(dict(ready=rows),'ready',dict(peers=['nextv3b0','nextv3b1']))
    else:
        with pytest.raises(RuntimeError):third.peer_reply(dict(ready=rows),'ready',dict(peers=['nextv3b0','nextv3b1']))

def test_daemon_created_after_timeout_owned_cleanup(declaration):
    s=declaration;result=dict(creation_intents=[],commands=[],errors=[]);row=created_fixture(s)
    async def command(argv,*args,**kwargs):
        if argv[1]=='run':
            assert result['creation_intents'] and Path(s['out'],'creation-intents/cap3b2.json').exists()
            raise asyncio.TimeoutError('daemon may already have created container')
        if argv[1]=='inspect':return json.dumps([row])
        assert argv==['docker','stop','--time','10',row['Id']];return row['Id']
    m=SimpleNamespace(command=command,write=r.write)
    async def run():
        with pytest.raises(asyncio.TimeoutError):await third.create_and_prepare(s,None,None,result,None,m)
        await third.stop_intents(s,None,result,m)
    asyncio.run(run())
    assert result['creation_intents'][0]['stopped_after_failure'] and not result['errors']

@pytest.mark.parametrize('bad',[None,'inflight_send','missing_budget_ack','cancel'])
def test_native_replays_actual_B_tp2_receipt(monkeypatch,bad):
    b=c.read(c.PDB);i=b['instances'][0]
    cp=c.read(next((Path(b['output'])/'checkpoints').glob('*.json')))
    assert c.sha(cp['receipt'])==cp['receipt_sha256']
    e=c.read(cp['receipt'])['restoration'][i['id']]
    before=copy.deepcopy(e['before']);paused=copy.deepcopy(e['resumed']['before']);after=copy.deepcopy(e['resumed']['after']);proof=copy.deepcopy(e['proof'])
    assert len(proof['transfers'])==2
    if bad=='inflight_send':proof['transfers'][0]['inflight_sends']=1
    if bad=='missing_budget_ack':after['scheduler_budget_effective']={}
    observations=iter([before,paused,after]);controls=[]
    async def idle(*args):return next(observations)
    async def http(session,i,route,body,limit,records):
        if route=='/drain':
            if bad=='cancel':raise asyncio.CancelledError()
            return proof
        controls.append(body);return dict(generation=body['generation'])
    monkeypatch.setattr(r,'idle',idle);monkeypatch.setattr(r,'http',http)
    common=r.load(r.COMMON,'b_static3_actual_native');result={}
    async def run():await r.native(None,i,r.Limit(time.time()+5),[],common,result)
    if bad:
        with pytest.raises(asyncio.CancelledError if bad=='cancel' else RuntimeError):asyncio.run(run())
        assert not result['complete']
        if bad in ('cancel','inflight_send'):assert not controls
    else:
        asyncio.run(run());assert result['complete']
        assert controls[0]['scheduler_budget']==dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32)

@pytest.mark.parametrize('bad',[None,'seven_power','seven_clock'])
def test_all8_energy_boundary(bad):
    n=7 if bad=='seven_power' else 8;k=7 if bad=='seven_clock' else 8
    sampler=SimpleNamespace(samples=[(0.,[100.]*n),(2.,[100.]*n)],frequency_samples=[(0.,[1500.]*k),(2.,[1500.]*k)],
        power_source='test-only',power_metadata={},error=None)
    if bad:
        with pytest.raises(RuntimeError):r.energy_evidence(sampler,.5,1.5,lambda *a:dict(power_source_verified=True))
    else:assert r.energy_evidence(sampler,.5,1.5,lambda *a:dict(power_source_verified=True))['energy_j']==800.

def test_inherited_hardware_primitives_ast_identical():
    parent=c.C/'A14B-pdb-post-scale-restore-v2/restore.py'
    assert c.sha(parent)=='19fd181be14f2447d1d053eaee1c726606d956cb647fea5d3ea0f47a0a25eb14'
    nodes=lambda p:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(p.read_text()).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    old,new=nodes(parent),nodes(ROOT/'restore.py')
    for name in ['Limit','command','http','idle','native','ready','archive_runtime','verify_prefixes','energy_evidence','static_container','verify_restarted']:
        assert new[name]==old[name],name


def test_first_cancel_during_failure_cleanup_returns_for_archival(monkeypatch):
    result=dict(start_intents=[dict(container_id='owned')],commands=[],errors=[])
    controls=[]
    async def command(*args,**kwargs):
        controls.append(args[0]);raise asyncio.CancelledError('first signal during failed cleanup')
    async def forbidden(*args):pytest.fail('no new third control after cancellation')
    monkeypatch.setattr(third,'stop_intents',forbidden)
    caught=asyncio.run(r.failure_cleanup({},None,result,SimpleNamespace(command=command)))
    assert isinstance(caught,asyncio.CancelledError) and result['cleanup_interrupted']
    assert len(controls)==1 and result['errors']
    # Caller receives the cancellation value before mandatory local archival,
    # then raises it after writing failure receipt; not a swallowed cancellation.
    source=ast.parse((ROOT/'restore.py').read_text())
    launch=next(x for x in source.body if isinstance(x,ast.AsyncFunctionDef) and x.name=='launch')
    assert 'interrupted=await failure_cleanup' in (ROOT/'restore.py').read_text()
    assert 'if failure is not None:raise failure' in (ROOT/'restore.py').read_text()


def test_bootstrap_complete_identity_isolated_from_raw(declaration,tmp_path):
    s=declaration;out=Path(s['out']);out.mkdir()
    rows=[];provenance={};native={}
    for j,i in enumerate(s['instances']):
        row=copy.deepcopy(s['expected_containers'][i['container_name']]);row['State'].update(Running=True,Pid=700+j,StartedAt='2026-09-08T05:00:00Z')
        rows.append(row);provenance[i['id']]=copy.deepcopy(i['provenance'])
        native[i['id']]=dict(complete=True,errors=[],resumed=dict(after=dict(id=i['id'],free_kv_tokens=47104)))
    row=created_fixture(s);rows.append(row);key=s['third_instance']['id']
    provenance[key]=copy.deepcopy(s['third_expected_provenance'])
    native[key]=dict(complete=True,errors=[],resumed=dict(after=dict(id=key,free_kv_tokens=47104)))
    raw=out/'containers.after.json';raw.write_text(json.dumps(rows));original=raw.read_bytes()
    spec=out/'deployment.json';spec.write_text(json.dumps(s))
    receipt=dict(complete=True,measurement_valid=True,errors=[],artifacts={str(raw):c.sha(raw)},third_identity_initial=row,
        native_after_peers=native,new_provenance=provenance)
    (out/'deployment-receipt.json').write_text(json.dumps(receipt))
    info=third.bootstrap(s,spec);b=c.read(info['path']);ident=c.read(b['identity_file'])
    assert b['configs']=={} and not b['performance_authorized']
    assert len(ident)==3 and all(set(x)=={'container','provenance','runtime'} for x in ident)
    assert b['files'][b['identity_file']]==c.sha(b['identity_file']) and raw.read_bytes()==original
    assert {x['provenance']['instance_id'] for x in ident}=={'nextv3b0','nextv3b1','cap3b2'}
