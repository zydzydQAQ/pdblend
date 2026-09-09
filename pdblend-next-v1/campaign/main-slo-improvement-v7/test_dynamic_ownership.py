import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import fcntl
import pytest
import dynamic_ownership as d

def original():
    return [dict(id='old'+str(n), tp=1, gpus=[n], port=8000+n, url='http://127.0.0.1:'+str(8000+n),
                 container=dict(id='original'+str(n))) for n in (6, 7)]

def ledger():
    initial=original()
    known={i['id']:dict(i,owner_kind='retained_original') for i in initial}
    extra=dict(id='cap-test-abc',tp=1,gpus=[0],port=9000,url='http://127.0.0.1:9000',
        owner_id='test',container_name='pdb-v2-cap-test-abc',owner_kind='created_for_cell',state='stopped')
    known[extra['id']]=extra
    return dict(schema='capacity-live-inventory-v1',pid=123,identity={'engine_image':'sha256:model'},
        initial_ids=[i['id'] for i in initial],known_instances=known,active_instances=initial,
        events=[],complete=True,transition_inflight=False)

def test_terminal_inventory_verifies_exact_owner(tmp_path):
    path=tmp_path/'inventory.json';path.write_text(json.dumps(ledger()))
    v=d.inventory(path,original(),child_pid=123,identity={'engine_image':'sha256:model'})
    d.validate_terminal(v,original())

@pytest.mark.parametrize('mutation', [
    lambda v:v.update(pid=124),
    lambda v:v['known_instances']['old6']['container'].update(id='replacement'),
    lambda v:v['known_instances']['cap-test-abc'].update(owner_id='another'),
    lambda v:v['known_instances']['cap-test-abc'].update(gpus=[6]),
    lambda v:v['known_instances']['cap-test-abc'].update(port=8006),
])
def test_replaced_or_unowned_inventory_refused(tmp_path,mutation):
    v=ledger();mutation(v);path=tmp_path/'inventory.json';path.write_text(json.dumps(v))
    with pytest.raises(ValueError):d.inventory(path,original(),child_pid=123)

@pytest.mark.parametrize('mutation', [
    lambda v:v.update(complete=False),
    lambda v:v.update(transition_inflight=True),
    lambda v:v['active_instances'].append(v['known_instances']['cap-test-abc']),
    lambda v:v['events'].append(dict(kind='rollback_failed')),
])
def test_incomplete_physical_actions_cannot_pass(mutation):
    v=ledger();mutation(v)
    with pytest.raises(ValueError):d.validate_terminal(v,original())

def test_open_file_is_not_lock_ownership(tmp_path,monkeypatch):
    lock=tmp_path/'lock';lock.touch();monkeypatch.setattr(d,'LOCK',lock)
    with lock.open('a'):
        with pytest.raises(ValueError):d.inherited_lease()

def test_real_flock_inherited_and_parent_checked(tmp_path,monkeypatch):
    lock=tmp_path/'lock';lock.touch();monkeypatch.setattr(d,'LOCK',lock)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lease=d.inherited_lease()
        source='import dynamic_ownership as d,json,sys;from pathlib import Path;d.LOCK=Path(sys.argv[1]);d.verify_inherited(json.loads(sys.argv[2]))'
        good=subprocess.run([sys.executable,'-c',source,str(lock),json.dumps(lease)],
            pass_fds=(handle.fileno(),),capture_output=True,cwd=Path(d.__file__).parent)
        assert good.returncode==0,good.stderr
        lease['holder_start_ticks']+=1
        bad=subprocess.run([sys.executable,'-c',source,str(lock),json.dumps(lease)],
            pass_fds=(handle.fileno(),),capture_output=True,cwd=Path(d.__file__).parent)
        assert bad.returncode!=0

def test_outer_cleanup_never_stops_container_with_other_owner(monkeypatch):
    calls=[]
    async def command(args,deadline):
        calls.append(args)
        return 0,json.dumps([dict(Name='/pdb-v2-cap-test-abc',Image='sha256:model',
            Config={'Labels':{'pdblend.capacity.owner':'someone-else'}},State={'Running':True,'Pid':1})]),''
    monkeypatch.setattr(d,'command',command)
    with pytest.raises(ValueError,match='ownership'):
        asyncio.run(d.remove_owned_extras(None,None,ledger(),{'identity':{'engine_image':'sha256:model'}},1,None))
    assert len(calls)==1 and calls[0][:2]==['docker','inspect']

def test_stopped_extra_requires_actual_gpu_release(monkeypatch):
    async def command(args,deadline):
        return 0,json.dumps([dict(Id='new',Name='/pdb-v2-cap-test-abc',Image='sha256:model',
            Config={'Labels':{'pdblend.capacity.owner':'test'}},State={'Running':False,'Pid':0})]),''
    monkeypatch.setattr(d,'command',command)
    hardware=SimpleNamespace(_handle=lambda g:g,_nvml=SimpleNamespace(nvmlDeviceGetComputeRunningProcesses=lambda g:[42]))
    with pytest.raises(ValueError,match='physical process'):
        asyncio.run(d.remove_owned_extras(None,None,ledger(),{'identity':{'engine_image':'sha256:model'}},1,hardware))

def test_reused_gpu_is_checked_after_all_historical_owners_stopped(monkeypatch):
    v=ledger();old=v['known_instances']['cap-test-abc']
    new=dict(old,id='cap-test-def',container_name='pdb-v2-cap-test-def',port=9001,state='started_unpublished')
    v['known_instances'][new['id']]=new
    stopped=[]
    async def command(args,deadline):
        target=args[-1]
        if args[1]=='stop':
            stopped.append(target);return 0,'',''
        running=target==new['container_name'] and target not in stopped
        return 0,json.dumps([dict(Id=target,Name='/'+target,Image='sha256:model',
            Config={'Labels':{'pdblend.capacity.owner':'test'}},State={'Running':running,'Pid':42 if running else 0})]),''
    monkeypatch.setattr(d,'command',command)
    def processes(gpu):
        assert stopped==[new['container_name']]
        return []
    hardware=SimpleNamespace(_handle=lambda g:g,_nvml=SimpleNamespace(nvmlDeviceGetComputeRunningProcesses=processes))
    result=asyncio.run(d.remove_owned_extras(None,None,v,{'identity':{'engine_image':'sha256:model'}},1,hardware))
    assert len(result)==2 and all(r['complete'] for r in result)

def test_transition_commit_without_measured_evidence_refused():
    v=ledger();v['events']=[dict(kind='physical_commit',transaction='unmeasured')]
    with pytest.raises(ValueError,match='energy evidence'):d.transition_artifacts(v)
