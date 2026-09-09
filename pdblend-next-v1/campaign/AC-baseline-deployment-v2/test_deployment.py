import ast
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import deploy
from build_host import build_baseline_host

def group(tmp_path):
    rows=[];out=tmp_path/'results'
    for dataset in deploy.SLOS:
        for phase,n in [('main',10),('scale',6)]:
            for j in range(n):
                cid=f'{dataset}-{phase}-{j}'
                row=dict(cell_id=cid,system='pdblend',dataset=dataset,phase=phase,trace_sha256='same-trace',
                    reuse_main_cell_id=f'{dataset}-main-{j}')
                rows.append(row);raw=out/'raw'/cid;deploy.write(raw,dict(energy_j=9,work_complete=False))
                receipt=out/'receipts'/(cid+'.json');deploy.write(receipt,dict(measurement_valid=True,child_stopped=True,
                    clock_restore_complete=True,summary=dict(work_complete=False,good_requests=0)))
                deploy.write(out/'checkpoints'/(cid+'.json'),dict(row=row,measurement_valid=True,receipt=str(receipt),
                    receipt_sha256=deploy.sha(receipt),artifacts={str(raw):deploy.sha(raw)}))
    manifest=tmp_path/'source.json';deploy.write(manifest,dict(protocol_id=deploy.PROTOCOL,cells=rows))
    binding=tmp_path/'binding.json';deploy.write(binding,dict(protocol_id=deploy.PROTOCOL,system='pdblend',output=str(out),files={str(manifest):deploy.sha(manifest)}))
    for phase in ['main','scale']:
        deploy.write(out/'invocations'/(phase+'.json'),dict(system='pdblend',phase=phase,complete=True,finished_s=2,started_s=1,pid=-1))
    return binding,manifest,out

def test_predecessor_requires_all_48_but_keeps_valid_incomplete_work(tmp_path):
    b,m,out=group(tmp_path)
    assert deploy.terminal_group(b,m)['counts']==dict(main=30,scale=18)
    assert len(deploy.terminal_group(b,m)['checkpoint_sha256'])==48

@pytest.mark.parametrize('failure',['missing_scale','changed_raw','unclean_receipt','active_runner','wrong_manifest'])
def test_predecessor_fail_closed(tmp_path,monkeypatch,failure):
    b,m,out=group(tmp_path)
    if failure=='missing_scale':(out/'checkpoints/alpaca-scale-5.json').unlink()
    elif failure=='changed_raw':(out/'raw/alpaca-main-0').write_text('changed')
    elif failure=='unclean_receipt':(out/'receipts/alpaca-main-0.json').write_text('{}')
    elif failure=='active_runner':monkeypatch.setattr(deploy,'source_process_alive',lambda *a:True)
    elif failure=='wrong_manifest':m.write_text('{}')
    with pytest.raises((RuntimeError,FileNotFoundError,KeyError)):deploy.terminal_group(b,m)

def test_docker_arguments_preserve_env_and_never_remove_old(tmp_path):
    i=dict(container_name='pdb-v2-base100ar0',environment=['PYTHONPATH=/old/src','NCCL_SHM_DISABLE=1','CUDA_VISIBLE_DEVICES=0'],
        mounts=[dict(Type='bind',Source='/root/workspace',Destination='/root/workspace',RW=True),dict(Type='bind',Source='/models',Destination='/models',RW=False)],
        image=deploy.IMAGE,engine_entry='/absolute/observation/engine.py',config='/new/cfg.json')
    argv=deploy.docker_start_arguments(i)
    assert argv[-4:]==['python3','/absolute/observation/engine.py','--config','/new/cfg.json']
    assert '-m' not in argv and 'rm' not in argv and 'PYTHONPATH=/old/src' in argv
    assert '/models:/models:ro' in argv and i['container_name'].startswith('pdb-v2-')

def test_baseline_host_only_lifecycle_entry_changes_and_actual_argv(tmp_path,monkeypatch):
    parent=deploy.WORKSPACE/'releases/five-system100-A14B-v1-runtime';out=tmp_path/'host'
    result=build_baseline_host(parent,out)
    assert result['changed_runtime_files']==['src/ecopadg/serving/topology.py']
    for path in result['files']:
        if path.endswith('.py') and path!='src/ecopadg/serving/topology.py':assert deploy.sha(out/path)==deploy.sha(parent/path)
    module=deploy.load('test_observation_lifecycle',out/'src/ecopadg/serving/topology.py')
    async def local_io(fn,*a,**k):return fn(*a,**k)
    monkeypatch.setattr(module.asyncio,'to_thread',local_io)
    entry=tmp_path/'engine.py';entry.write_text('# CPU-only entry fixture\n')
    async def invoke(override):
        template={} if not override else dict(observation_engine_entry=str(entry),observation_engine_sha256=deploy.sha(entry))
        lifecycle=module.DockerLifecycle(tmp_path/('native' if override else 'old'),deploy.IMAGE,template)
        commands=[]
        async def command(*args,**kw):commands.append(args);return 'CPU-only-id'
        lifecycle.command=command
        await lifecycle.start(module.InstanceSpec('newid',1,(0,),31000,41000),{})
        return commands[0]
    old=asyncio.run(invoke(False));new=asyncio.run(invoke(True))
    assert ('python3','-m','ecopadg.serving.engine')==old[-5:-2]
    assert new[-4:-2]==('python3',str(entry)) and '-m' not in new
    assert old[:old.index(deploy.IMAGE)]==new[:new.index(deploy.IMAGE)]
    bad=module.DockerLifecycle(tmp_path/'bad',deploy.IMAGE,dict(observation_engine_entry=str(entry),observation_engine_sha256='0'*64))
    with pytest.raises(ValueError,match='hash-bound'):
        asyncio.run(bad.start(module.InstanceSpec('bad',1,(0,),31000,41000),{}))

def test_actual_historical_engine_config_geometry_and_model_budget():
    index=deploy.read(deploy.ROOT/'inputs/engine-config-index.json')
    for node,model in [('A','14B'),('C','7B')]:
        for rec in index[node].values():
            cfg=deploy.read(deploy.ROOT/rec['copied_path'])
            assert cfg['tp']==1 and cfg['model']==f'/models/Qwen2.5-{model}-Instruct'
            assert (cfg['max_model_len'],cfg['max_num_batched_tokens'],cfg['max_num_seqs'])==(8192,8192,32)
            assert cfg.get('retained_weights')
    lb=deploy.read(deploy.ROOT/'inputs/A-distserve-longbench-tp2.json')
    # The historical engine starts mixed; the saved DistServe Controller sets
    # its real role to decode. prepare applies that preserved policy role.
    assert lb['tp']==2 and lb['role']=='mixed' and lb['model']=='/models/Qwen2.5-14B-Instruct'

def test_one_power_frame_never_opens_hardware_gate():
    sampler=SimpleNamespace(samples=[(1.,[100.]*8)],error=None,power_source='instant',power_metadata={})
    with pytest.raises(RuntimeError,match='two complete'):
        asyncio.run(deploy.await_power_ready(sampler,lambda *a:dict(power_source_verified=True),timeout_s=0))

def test_creation_intent_cleanup_covers_daemon_created_on_command_timeout(monkeypatch):
    intents=[dict(name='pdb-v2-owned-1'),dict(name='pdb-v2-owned-2')];seen=[]
    async def command(*args,**kwargs):
        seen.append(args)
        if args[1]=='inspect':return json.dumps([dict(Id='real-'+args[2])])
        if args[-1].endswith('1'):raise RuntimeError('stop failed')
        return args[-1]
    monkeypatch.setattr(deploy,'command',command)
    errors=asyncio.run(deploy.stop_creation_intents(intents,deploy.time.time()+1))
    assert len(errors)==1 and all(i['observed_container_id'].startswith('real-') for i in intents)
    assert intents[1]['stopped'] and len([a for a in seen if a[1]=='stop'])==2

def test_cleanup_shared_deadline_does_not_skip_other_owned_names(monkeypatch):
    async def command(*args,**kwargs):await asyncio.sleep(60)
    monkeypatch.setattr(deploy,'command',command)
    intents=[dict(name='pdb-v2-owned-'+str(i)) for i in range(8)]
    started=deploy.time.time()
    errors=asyncio.run(deploy.stop_creation_intents(intents,started+.02))
    assert len(errors)==8 and deploy.time.time()-started<1
