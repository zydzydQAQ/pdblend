import copy,json
from pathlib import Path
import pytest
import launch as m

def terminal():
 return {'complete':True,'phase':'main_incomplete_correctness','pid':1,'steps':[{'pid':2,'complete':True,'exitcode':0}]}
def test_wait_for_live_main():assert m.readiness({'complete':False,'pid':1},lambda p:{'pid':p}) is False
def test_disappeared_not_terminal():
 with pytest.raises(RuntimeError,match='disappeared'):m.readiness({'complete':False,'pid':1},lambda p:None)
def test_failed_main_never_launches():
 s=terminal();s['phase']='failed'
 with pytest.raises(RuntimeError,match='failed'):m.readiness(s,lambda p:None)
def test_terminal_status_still_live_waits():assert m.readiness(terminal(),lambda p:{'pid':p} if p==1 else None) is False
def test_stage_process_must_exit():assert m.readiness(terminal(),lambda p:{'pid':p} if p==2 else None) is False
def test_only_actual_zero_exit():
 for bad in (False,True,1,None):
  s=terminal();s['steps'][0]['exitcode']=bad
  with pytest.raises(RuntimeError,match='cleanly'):m.readiness(s,lambda p:None)
def test_eligible_three_groups_does_not_require_false_eco():assert m.readiness(terminal(),lambda p:None) is True

def setup_index(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'CAMPAIGN',tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path/'launcher');monkeypatch.setattr(m,'ATTEMPT',tmp_path/'attempt');m.ATTEMPT.mkdir()
 m.write(m.ATTEMPT/'spec.json',{'x':1});m.write(tmp_path/'current-experiment.json',{'active_system':'distserve','queue_running':False,'queue_pid':4});(tmp_path/'CURRENT_EXPERIMENT.md').write_text('original')
 return {'diagnostic_instance':{'config':str(m.ATTEMPT/'engine.json')}}
def test_wrong_pid_cannot_publish_or_archive(tmp_path,monkeypatch):
 spec=setup_index(tmp_path,monkeypatch);original=(tmp_path/'current-experiment.json').read_bytes();monkeypatch.setattr(m,'proc',lambda p:None)
 with pytest.raises(RuntimeError,match='process changed'):m.index(spec,5,{'start_ticks':4,'argv':[]},'preflight',False,{})
 assert (tmp_path/'current-experiment.json').read_bytes()==original;assert not (tmp_path/'current-experiment-history').exists()
def test_live_exact_argv_and_terminal_record(tmp_path,monkeypatch):
 spec=setup_index(tmp_path,monkeypatch);handle={'start_ticks':4,'argv':['python',str(m.PACKAGE/'run.py'),'--spec',str(m.ATTEMPT/'spec.json')]}
 monkeypatch.setattr(m,'proc',lambda p:copy.deepcopy(handle));m.index(spec,5,handle,'preflight',False,{})
 first=m.read(tmp_path/'current-experiment.json');assert first['queue_running'] and first['active_system']=='diagnostic' and not first['performance_evidence']
 with pytest.raises(RuntimeError,match='actual process exit'):m.index(spec,5,handle,'terminal',True,{'exitcode':0})
 monkeypatch.setattr(m,'proc',lambda p:None);m.index(spec,5,handle,'terminal',True,{'exitcode':0});last=m.read(tmp_path/'current-experiment.json');assert not last['queue_running'] and not last['output_correctness_verified'] and last['fresh_correctness_and_binding_required_after_restore']
 assert len(list((tmp_path/'current-experiment-history').iterdir()))==2


def test_producer_receipt_exactly_bound_to_actual_spec():
 before={'actual_main_producers':90,'files':{'a':'x'},'records':[1]};m.bind_producers(before,copy.deepcopy(before),{'files':{'a':'x'}})
 with pytest.raises(RuntimeError,match='does not freeze'):m.bind_producers(before,before,{'files':{'a':'wrong'}})
 with pytest.raises(RuntimeError,match='changed during prepare'):m.bind_producers(before,dict(before,records=[2]),{'files':{'a':'x'}})


def test_other_actual_launcher_must_exit(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'CAMPAIGN',tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path/'B32B-temporal-observation-launch-v4')
 old=tmp_path/'B32B-temporal-observation-launch-v2/attempt-001/status.json';m.write(old,{'pid':123,'complete':True})
 monkeypatch.setattr(m,'proc',lambda p:{'pid':p})
 with pytest.raises(RuntimeError,match='another diagnostic launcher'):m.no_other_launcher()
 monkeypatch.setattr(m,'proc',lambda p:None);m.no_other_launcher()
