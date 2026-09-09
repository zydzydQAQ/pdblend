import copy, importlib.util, pathlib, sys
from types import SimpleNamespace
import pytest
import supervise as s

def gate():
    return (dict(complete=True,measurement_valid=True,native_cleanup_complete=True,
                 clock_restore_complete=True,cleanup_errors=[],passed=False,
                 mechanism_gate=dict(ordinary=True,pd=True,temporal=False)),
            dict(complete=True,cleanup=dict(complete=True)))

def test_temporal_failure_does_not_hide_or_block_other_complete_mechanisms():
    g,c=gate()
    for system in ('mixed','distserve','dynamollm-resident'):
        assert s.decide_system(g,c,system)[0]
    assert not s.decide_system(g,c,'ecoserve')[0]
    assert g['passed'] is False

@pytest.mark.parametrize('field',['complete','measurement_valid','native_cleanup_complete','clock_restore_complete'])
def test_partial_or_invalid_gate_never_admits(field):
    g,c=gate();g[field]=False
    with pytest.raises(RuntimeError):s.decide_system(g,c,'mixed')

def test_raw_cleanup_must_be_terminal():
    g,c=gate();c['cleanup']['complete']=False
    with pytest.raises(RuntimeError):s.decide_system(g,c,'mixed')

def test_deadline_stops_queue_400_seconds_early_and_retains_cleanup():
    stop,interrupt=s.command_deadlines(s.DEADLINE-1000,None,True)
    assert stop==s.DEADLINE-400
    assert interrupt==s.DEADLINE-125
    assert interrupt-stop==275
    assert s.command_deadlines(s.DEADLINE-2000,520,False)==(None,s.DEADLINE-1480)

def test_inherited_lease_rejected_before_file_reads(monkeypatch):
    monkeypatch.setenv('PDBLEND_NODE_LOCK_FD','99')
    monkeypatch.setattr(s,'sha',lambda p:pytest.fail('must reject before filesystem work'))
    with pytest.raises(RuntimeError,match='inherit'):s.check_files()

def test_actual_command_requests_boundary_stop_without_interrupting_cell(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'check_files',lambda:None)
    monkeypatch.setattr(s,'read',lambda p:dict(host_release='/frozen/host'))
    now=[s.DEADLINE-401]
    monkeypatch.setattr(s.time,'time',lambda:now[0])
    monkeypatch.setattr(s.time,'sleep',lambda n:now.__setitem__(0,now[0]+1))
    signals=[]
    class Child:
        pid=123;returncode=None
        def poll(self):
            if now[0]>=s.DEADLINE-398:self.returncode=0
            return self.returncode
        def send_signal(self,value):signals.append(value)
    monkeypatch.setattr(s.subprocess,'Popen',lambda *a,**k:Child())
    obj=s.Supervisor(tmp_path/'attempt');stop=tmp_path/'isolated-result/STOP'
    with pytest.raises(RuntimeError,match='boundary STOP'):
        obj.command('main',['python3','frozen.py'],stop_path=stop)
    assert stop.exists() and signals==[] and obj.active is None
    assert obj.state['steps'][0]['complete'] is True
    assert obj.state['steps'][0]['global_deadline_boundary_stop_s']==s.DEADLINE-400

def test_existing_attempt_cannot_be_overwritten(tmp_path):
    with pytest.raises(RuntimeError,match='fresh'):s.Supervisor(tmp_path)

def test_binder_v2_only_changes_main_to_take_independent_lease():
    import ast
    p=s.P
    old=ast.parse((p/'bind_baseline.py').read_text())
    new=ast.parse((p/'bind_baseline_v2.py').read_text())
    functions=lambda tree:{n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name!='main'}
    assert functions(old)==functions(new)
    main=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assert any(isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Call) and getattr(i.context_expr.func,'id','')=='node_lease' for i in n.items) for n in ast.walk(main))
