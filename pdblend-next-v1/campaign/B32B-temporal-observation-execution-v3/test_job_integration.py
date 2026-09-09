"""Actual parent job construction meets the unchanged child validator; no GPU/network."""
import asyncio, importlib.util, sys, time
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run as current
import common as c
from test_child import child, job, Session


def parent(tmp_path, module, started):
    _,j,_=job(tmp_path)
    op=object.__new__(module.Operation)
    op.out=tmp_path;op.state={};op.spec=j;op.diag_binding=j['binding'];op.stop=False;op.restoring=False
    c.write(tmp_path/'diagnostic-binding.json',j['binding'])
    op.limits=module.deadline_limits(started);op.deadline=op.limits['work_end_s']
    op.anchor_wall=time.time();op.anchor_mono=time.monotonic();op.terminal={};op.child=None;op.child_log=None
    op.sampler=SimpleNamespace(error=None);op.save=lambda:None;op.phase=lambda _:None
    return op


def run_actual_parent(tmp_path, monkeypatch, module, elapsed, *, real_child=False):
    now=time.time();op=parent(tmp_path,module,now-elapsed);observed={}
    async def spawn(*argv,**kwargs):
        actual=c.read(argv[-1]);observed.update(actual)
        spec=child.validate_job(actual)
        if real_child:
            state=await child.execute(argv[-1],session_factory=lambda:Session(spec))
            assert state['completed_requests']==6 and state['cleanup_complete'] and state['exact_passed']
        else:c.write(Path(actual['output_dir'])/'status.json',dict(complete=True,cleanup_complete=True))
        return SimpleNamespace(pid=99999,returncode=0)
    monkeypatch.setattr(module.asyncio,'create_subprocess_exec',spawn)
    try:result=asyncio.run(op.run_child())
    finally:
        if op.child_log:op.child_log.close()
    return op,observed,result


@pytest.mark.parametrize('elapsed',[0,30,150,200,530])
def test_actual_job_validator_early_and_late_start(tmp_path,monkeypatch,elapsed):
    op,j,_=run_actual_parent(tmp_path,monkeypatch,current,elapsed)
    issued=op.state['child_limits']['issued_s']
    assert j['work_deadline_s']==min(op.limits['work_end_s'],issued+390)
    assert j['cleanup_deadline_s']==min(op.limits['cleanup_end_s'],j['work_deadline_s']+90)
    assert op.deadline==j['work_deadline_s']
    assert 0<j['work_deadline_s']-issued<=390.001
    assert 0<j['cleanup_deadline_s']-j['work_deadline_s']<=90.001
    assert j['cleanup_deadline_s']<=op.limits['cleanup_end_s']<op.limits['restore_end_s']<op.limits['end_s']
    assert op.limits['end_s']-op.limits['started_s']==900


def test_actual_original_parent_reproduces_early_start_contract_failure(tmp_path,monkeypatch):
    path=ROOT.parent/'B32B-temporal-observation-execution-v2-r2/run.py'
    spec=importlib.util.spec_from_file_location('original_r2_run_for_contract',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    with pytest.raises(RuntimeError,match='original global/work deadline exceeded'):
        run_actual_parent(tmp_path,monkeypatch,old,30)


def test_actual_parent_job_executes_unchanged_six_request_child(tmp_path,monkeypatch):
    op,j,result=run_actual_parent(tmp_path,monkeypatch,current,30,real_child=True)
    assert result['child_exit_ok'] and result['completed_requests']==6 and op.terminal['http_child_exited']
    assert result['actual_cleanup_deadline_s']<=j['cleanup_deadline_s']


def test_expired_parent_interval_sends_no_subprocess(tmp_path,monkeypatch):
    op=parent(tmp_path,current,time.time()-541);called=[]
    async def forbidden(*args,**kwargs):called.append(args);raise AssertionError('spawned')
    monkeypatch.setattr(current.asyncio,'create_subprocess_exec',forbidden)
    with pytest.raises(RuntimeError,match='interval remains'):asyncio.run(op.run_child())
    assert not called and not (tmp_path/'child.log').exists()
