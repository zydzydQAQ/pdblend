import json,time,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
import archive as a

def fixture(tmp_path):
    instances=[];stopped=[];native={'instances':{}}
    for n in range(4):
        iid='b'+str(n);cfg=tmp_path/(iid+'.json');c.write(cfg,dict(runtime_dir=str(tmp_path)))
        state=dict(generation=10,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
        c.write(tmp_path/(iid+'.control.json'),state)
        (tmp_path/(iid+'.control.events.jsonl')).write_bytes(b'{"old":1}\n{"old":2}\n')
        for rank in (0,1):(tmp_path/(iid+'.control.json.kv.'+str(rank)+'.jsonl')).write_bytes(b'{"old_kv":1}\n')
        instances.append(dict(id=iid,tp=2,engine_config=str(cfg),container={'id':'cid'+str(n)}))
        stopped.append(dict(Id='cid'+str(n),State=dict(Running=False,Pid=0)))
        native['instances'][iid]={'after':state}
    return {'instances':instances},stopped,native

def test_full_original_control_and_event_prefix_preserved(tmp_path):
    b,stopped,native=fixture(tmp_path);initial=a.capture(b,tmp_path/'initial',time.time()+20)
    for group in a.paths(b).values():
        with group['events'].open('ab') as f:f.write(b'{"during_drain":3}\n')
        c.write(group['control'],dict(native['instances']['b0']['after'],generation=11))
    final=a.capture(b,tmp_path/'final',time.time()+20,stopped=stopped,initial=initial)
    assert final['complete'] and final['original_processes_stopped']
    for iid,group in a.paths(b).items():
        old=final['instances'][iid]
        assert Path(old['events']['archive']).read_bytes()==group['events'].read_bytes()
        assert c.read(old['control']['archive'])['generation']==11
        # Real restart replaces the control JSON and appends to the timeline.
        c.write(group['control'],native['instances'][iid]['after'])
        with group['events'].open('ab') as f:f.write(b'{"new_process":1}\n')
    result=a.verify_prefix_after_restore(b,final,native,time.time()+20)
    assert result['complete'] and all(x['appended_event_bytes']>0 and x['stopped_control_sha256']!=x['restored_control_sha256'] for x in result['instances'].values())

@pytest.mark.parametrize('bad',['running','pid','set'])
def test_final_archive_requires_all_original_processes_stopped(tmp_path,bad):
    b,stopped,_=fixture(tmp_path)
    if bad=='running':stopped[0]['State']['Running']=True
    if bad=='pid':stopped[0]['State']['Pid']=99
    if bad=='set':stopped[0]['Id']='other'
    with pytest.raises(RuntimeError):a.capture(b,tmp_path/'final',time.time()+10,stopped=stopped)
    assert not (tmp_path/'final').exists()

@pytest.mark.parametrize('bad',['truncated','same_size_mutated'])
def test_prefix_change_during_stop_or_restore_rejected(tmp_path,bad):
    b,stopped,native=fixture(tmp_path);initial=a.capture(b,tmp_path/'initial',time.time()+20)
    path=a.paths(b)['b0']['events'];original=path.read_bytes()
    path.write_bytes(original[:-1] if bad=='truncated' else original.replace(b'1',b'9'))
    with pytest.raises(RuntimeError):a.capture(b,tmp_path/'final',time.time()+20,stopped=stopped,initial=initial)
    path.write_bytes(original);final=a.capture(b,tmp_path/'final2',time.time()+20,stopped=stopped,initial=initial)
    path.write_bytes(original[:2] if bad=='truncated' else original.replace(b'1',b'9'))
    with pytest.raises(RuntimeError):a.verify_prefix_after_restore(b,final,native,time.time()+20)


def test_missing_or_changed_archived_control_not_ignored(tmp_path):
    b,stopped,native=fixture(tmp_path);final=a.capture(b,tmp_path/'final',time.time()+20,stopped=stopped)
    Path(final['instances']['b0']['control']['archive']).write_bytes(b'{}')
    with pytest.raises(RuntimeError,match='archive changed'):a.verify_prefix_after_restore(b,final,native,time.time()+20)


def test_control_reset_must_match_actual_final_ack(tmp_path):
    b,stopped,native=fixture(tmp_path);final=a.capture(b,tmp_path/'final',time.time()+20,stopped=stopped)
    c.write(a.paths(b)['b0']['control'],dict(generation=0))
    with pytest.raises(RuntimeError,match='native ACK'):a.verify_prefix_after_restore(b,final,native,time.time()+20)


def test_large_runtime_is_not_silently_truncated(tmp_path,monkeypatch):
    b,stopped,native=fixture(tmp_path);monkeypatch.setattr(a,'MAX_TOTAL_BYTES',1)
    with pytest.raises(RuntimeError,match='never trim'):a.capture(b,tmp_path/'final',time.time()+20,stopped=stopped)


def test_archive_deadline_is_enforced_before_io(tmp_path):
    b,stopped,native=fixture(tmp_path)
    with pytest.raises(RuntimeError,match='deadline'):a.capture(b,tmp_path/'final',time.time()-1,stopped=stopped)
    assert not (tmp_path/'final').exists()


@pytest.mark.parametrize('kind',['kv_rank_0','kv_rank_1'])
def test_original_rank_event_logs_are_not_left_unprotected(tmp_path,kind):
    b,stopped,native=fixture(tmp_path);initial=a.capture(b,tmp_path/'initial',time.time()+20)
    final=a.capture(b,tmp_path/'final',time.time()+20,stopped=stopped,initial=initial)
    rank=a.paths(b)['b0'][kind]
    assert Path(final['instances']['b0'][kind]['archive']).read_bytes()==rank.read_bytes()
    rank.write_bytes(b'{"changed":1}\n')
    with pytest.raises(RuntimeError,match='KV event prefix'):a.verify_prefix_after_restore(b,final,native,time.time()+20)
