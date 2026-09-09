import copy,json,time
from pathlib import Path
import pytest
import handoff as h

def actual_schema():
 launch=h.read(h.MAIN/'launch.json')
 b=h.read(h.BIND);p=next((Path(b['output'])/'invocations').glob('*.json'));inv=h.read(p)
 # Synthetic terminal fields over the actual producer/schema; these are never written as experiment output.
 rows=[r['cell_id'] for r in h.read(h.SOURCE)['cells'] if r['system']=='ecoserve' and r['phase']=='main']
 inv.update(complete=True,completed=rows,finished_s=launch['started_s']+6000);inv.pop('error',None);inv.pop('skipped',None)
 terminal=dict(launch,exit_code=0,finished_s=launch['started_s']+6001)
 return launch,terminal,[inv]

def test_actual_producer_schema_missing_skipped_field_can_finish():
 a,b,c=actual_schema();assert h.ready(a,b,c,lambda _:None)

def test_live_actual_main_is_never_interrupted_or_treated_finished():
 a,b,c=actual_schema();assert not h.ready(a,b,c,lambda pid:dict(argv=a['argv']) if pid==a['pid'] else None)

def test_wrong_pid_argv_rejected():
 a,b,c=actual_schema()
 with pytest.raises(RuntimeError,match='wrong actual'):h.ready(a,b,c,lambda pid:dict(argv=['other']) if pid==a['pid'] else None)

def test_monitor_must_finish_old_terminal_index_publication():
 a,b,c=actual_schema();assert not h.ready(a,b,c,lambda pid:dict(argv=['monitor']) if pid==a['monitor_pid'] else None)

@pytest.mark.parametrize('case',['failed_exit','incomplete','skipped','binding','duplicate','foreign_id','time'])
def test_bad_actual_terminal_does_not_authorize_proof(case):
 a,b,c=actual_schema()
 if case=='failed_exit':b['exit_code']=1
 if case=='incomplete':c[0]['complete']=False
 if case=='skipped':c[0]['skipped']=[c[0]['completed'].pop()]
 if case=='binding':c[0]['binding_sha256']='0'*64
 if case=='duplicate':c[0]['completed'][-1]=c[0]['completed'][0]
 if case=='foreign_id':c[0]['completed'][-1]='unrelated-unique-cell'
 if case=='time':c[0]['finished_s']=b['finished_s']+1
 with pytest.raises(RuntimeError):h.ready(a,b,c,lambda _:None)

def test_missing_terminal_with_no_live_monitor_fails_closed():
 a,_,c=actual_schema()
 with pytest.raises(RuntimeError,match='without terminal'):h.ready(a,None,c,lambda _:None)

def test_stop_prevents_even_cpu_successor(tmp_path,monkeypatch):
 s=h.Handoff(tmp_path);(tmp_path/'STOP').write_text('operator stop')
 monkeypatch.setattr(h.subprocess,'Popen',lambda *a,**k:pytest.fail('no process after STOP'))
 with pytest.raises(RuntimeError,match='STOP'):s.command('proof',['unused'],{})

def test_deadline_does_not_reset_at_new_phase(tmp_path,monkeypatch):
 s=h.Handoff(tmp_path);monkeypatch.setattr(h.time,'time',lambda:h.DEADLINE-599)
 monkeypatch.setattr(h.subprocess,'Popen',lambda *a,**k:pytest.fail('no process past reserve'))
 with pytest.raises(RuntimeError,match='original deadline'):s.command('proof',['unused'],{})


def test_observer_exit_does_not_signal_performance(tmp_path):
 class Observer:
  returncode=1
  def poll(self):return self.returncode
 class Performance:
  def send_signal(self,*_):pytest.fail('metadata failure must not signal performance')
 s=h.Handoff(tmp_path);s.state={};s.observer=Observer();s.child=Performance()
 s.observe_exit()
 assert s.state['observer_exitcode']==1 and s.state['observer_error']
 assert 'scale_boundary_stop_sent_s' not in s.state
