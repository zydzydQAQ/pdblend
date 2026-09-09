import json,sys
from pathlib import Path
import pytest
import supervise as s

def fixture(tmp_path,n=30,q_failed=False):
    rows=[];b=dict(system='mixed',output=str(tmp_path/'results'))
    root=Path(b['output']);(root/'checkpoints').mkdir(parents=True)
    for i in range(30):
        row=dict(cell_id='cell-%02d'%i,system='mixed',phase='main',seed=701,trace_duration_s=100)
        rows.append(row)
        if i<n:
            p=root/('receipt-%d.json'%i);p.write_text(json.dumps(dict(measurement_valid=True,finished_s=1,summary={'q':0 if q_failed else 1})))
            cp=dict(row=row,receipt=str(p),receipt_sha256=s.sha(p),measurement_valid=True,work_complete=not q_failed,artifacts={str(p):s.sha(p)})
            (root/'checkpoints'/(row['cell_id']+'.json')).write_text(json.dumps(cp))
    return b,{'cells':rows}

def test_completed_prefix_resume_and_valid_failure_preserved(tmp_path):
    b,m=fixture(tmp_path,23,True);v=s.verify_phase(b,m,'main',False)
    assert v['completed']==23 and not v['complete']
    assert all(r['work_complete'] is False for r in v['records'])
    with pytest.raises(RuntimeError):s.verify_phase(b,m,'main')

def test_changed_raw_not_accepted_for_resume(tmp_path):
    b,m=fixture(tmp_path,1);(Path(b['output'])/'receipt-0.json').write_text('{}')
    with pytest.raises(RuntimeError):s.verify_phase(b,m,'main',False)

def test_checkpoint_hole_rejected(tmp_path):
    b,m=fixture(tmp_path,3);(Path(b['output'])/'checkpoints/cell-01.json').unlink()
    with pytest.raises(RuntimeError):s.verify_phase(b,m,'main',False)

def old_status():return dict(pid=100,phase='stopped',finished_s=1,steps=[dict(pid=101,complete=True,exitcode=0,label='mixed-main')])

def test_old_live_owner_or_unfinished_stage_blocks_transition():
    assert s.old_terminal(old_status(),lambda pid:False)
    with pytest.raises(RuntimeError):s.old_terminal(old_status(),lambda pid:pid==101)
    bad=old_status();bad['steps'][0]['complete']=False
    with pytest.raises(RuntimeError):s.old_terminal(bad,lambda pid:False)

def test_owned_stop_archived_but_unrelated_stop_preserved(tmp_path):
    b=tmp_path/'binding.json';out=tmp_path/'results';out.mkdir();b.write_text(json.dumps({'output':str(out)}))
    stop=out/'STOP';stop.write_text('unrelated user STOP')
    with pytest.raises(RuntimeError):s.archive_owned_queue_stop(b,tmp_path/'archive.STOP')
    assert stop.exists()
    stop.write_text(s.EXPECTED_STOP);h=s.sha(stop)
    receipt=s.archive_owned_queue_stop(b,tmp_path/'archive.STOP')
    assert not stop.exists() and receipt['sha256']==h and s.sha(tmp_path/'archive.STOP')==h

def test_new_supervisor_reuses_exact_frozen_command_and_has_no_scale_dispatch():
    assert s.Supervisor.command is s.old.Supervisor.command
    src=Path(s.__file__).read_text()
    assert "'--phase','main'" in src and "'--phase','scale'" not in src
    assert 'main_incomplete_correctness' in src and 'global_main_barrier_verified=False' in src

def test_same_original_deadline_and_stop_lead():
    stop,interrupt=s.old.command_deadlines(0,None,True)
    assert stop==s.DEADLINE-400 and interrupt==s.DEADLINE-125
