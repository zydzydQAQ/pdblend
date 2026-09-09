import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import observe as o

def setup(tmp_path):
    b=tmp_path/'binding.json';m=tmp_path/'manifest.json'
    b.write_text(json.dumps(dict(model='32b',hostname=o.HOSTNAME,system='mixed',configs={x:'/config/'+x for x in ('alpaca','sharegpt','longbench')})))
    m.write_text('{}')
    s=dict(pid=123,label='mixed-main',complete=False,argv=['python3',str(o.RUNNER),'--binding',str(b),'--manifest',str(m),'--system','mixed','--phase','main','--run'])
    return s,b,m

def test_dataset_default_uses_all_actual_binding_configs(tmp_path):
    s,b,m=setup(tmp_path);q=o.queue_step(s)
    assert q['datasets']==['alpaca','sharegpt','longbench']

def test_no_index_for_deployment_or_bootstrap():
    assert o.queue_step(dict(argv=['python3','deploy.py','--run'])) is None

def test_wrong_system_rejected(tmp_path):
    s,b,m=setup(tmp_path);s['argv'][s['argv'].index('--system')+1]='ecoserve'
    with pytest.raises(RuntimeError,match='mismatch'):o.queue_step(s)

def test_actual_full_argv_and_starttime_are_verified(tmp_path):
    s,b,m=setup(tmp_path);p=tmp_path/'123';p.mkdir()
    (p/'cmdline').write_bytes(b'\0'.join(x.encode() for x in s['argv'])+b'\0')
    # comm with spaces is legal; starttime is field 22, tail index 19.
    fields=['S']+['0']*18+['98765']+['0']*10
    (p/'stat').write_text('123 (python worker) '+' '.join(fields))
    assert o.live_identity(s,tmp_path)['proc_start_ticks']==98765
    (p/'cmdline').write_bytes(b'python3\0unrelated.py\0')
    with pytest.raises(RuntimeError,match='argv'):o.live_identity(s,tmp_path)

def test_publish_only_start_and_observed_terminal_once(tmp_path,monkeypatch):
    s,b,m=setup(tmp_path);calls=[]
    monkeypatch.setattr(o,'live_identity',lambda step:dict(pid=123,argv=step['argv'],proc_start_ticks=42))
    publisher=SimpleNamespace(update=lambda *args:calls.append(args) or dict(path='index',sha256='a'))
    obs=o.Observer(tmp_path/'out',publisher);src=dict(phase='mixed-main',steps=[s])
    obs.tick(src);obs.tick(src);assert len(calls)==1 and calls[0][-1] is None
    s.update(complete=True,exitcode=0);obs.tick(src);obs.tick(src)
    assert len(calls)==2 and calls[-1][-1]==0

def test_never_publish_unobserved_completed_pid(tmp_path):
    s,b,m=setup(tmp_path);s.update(complete=True,exitcode=0)
    obs=o.Observer(tmp_path/'out',SimpleNamespace(update=lambda *a:pytest.fail('no real PID observed')))
    obs.tick(dict(phase='mixed-main',steps=[s]));assert obs.state['publications']==[]

def test_binding_changed_after_live_observation_rejected(tmp_path,monkeypatch):
    s,b,m=setup(tmp_path);monkeypatch.setattr(o,'live_identity',lambda step:dict(pid=123,argv=step['argv'],proc_start_ticks=42))
    obs=o.Observer(tmp_path/'out',SimpleNamespace(update=lambda *a:dict(ok=True)))
    obs.tick(dict(steps=[s]));b.write_text(b.read_text()+'\n');s.update(complete=True,exitcode=0)
    with pytest.raises(RuntimeError,match='changed'):obs.tick(dict(steps=[s]))

def test_durable_creation_intent_before_popen_pid_does_not_publish(tmp_path):
    s,b,m=setup(tmp_path);s.pop('pid')
    obs=o.Observer(tmp_path/'out',SimpleNamespace(update=lambda *a:pytest.fail('not yet a process')))
    obs.tick(dict(steps=[s]));assert not obs.state['publications']

def test_process_exit_before_terminal_journal_waits_then_publishes_exit(tmp_path,monkeypatch):
    s,b,m=setup(tmp_path);calls=[]
    monkeypatch.setattr(o,'live_identity',lambda step:dict(pid=123,argv=step['argv'],proc_start_ticks=42))
    obs=o.Observer(tmp_path/'out',SimpleNamespace(update=lambda *a:calls.append(a) or dict(ok=True)))
    obs.tick(dict(steps=[s]))
    def gone(step):raise FileNotFoundError('process exited')
    monkeypatch.setattr(o,'live_identity',gone)
    obs.tick(dict(steps=[s]));assert len(calls)==1
    s.update(complete=True,exitcode=0);obs.tick(dict(steps=[s]))
    assert len(calls)==2 and calls[-1][-1]==0


def test_new_main_only_terminal_stops_observer_without_claiming_all_systems(tmp_path):
    obs=o.Observer(tmp_path/'out',SimpleNamespace(update=lambda *a:pytest.fail('no process')))
    obs.tick(dict(phase='main_incomplete_correctness',steps=[]))
    assert obs.state['complete'] and obs.state['source_phase']=='main_incomplete_correctness'
