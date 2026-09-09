import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ecopadg.serving import budget, campaign, campaign_pipeline


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(tmp_path, *, started=100):
    root=tmp_path/'campaign';root.mkdir()
    save(root/'budget.json',dict(started_s=started,limit_s=86400,stage='old'))
    auth=root/'authorization.json'
    save(auth,dict(authorized_at_s=started+10,instruction='Extend until the approved plan is complete',
        scope='All declared work, preserving consumed time and fixed workload',original_started_s=started,
        original_limit_s=86400,original_deadline_s=started+86400))
    return root,auth


def extend(root,auth,limit=345600,now=200):
    return budget.append_extension(root,authorization_path=auth,limit_s=limit,reason='finite next full-plan envelope',now=now)


def test_legacy_read_is_compatible_and_does_not_create_files(tmp_path):
    root,_=fixture(tmp_path);before=(root/'budget.json').read_bytes()
    value=budget.read_budget(root,now=200)
    assert value['deadline_s']==86500 and value['remaining_s']==86300
    assert value['revision_seq']==0 and value['revision_artifacts']=={}
    assert not (root/'.budget-ledger').exists() and before==(root/'budget.json').read_bytes()


def test_extension_preserves_origin_elapsed_and_mutable_progress(tmp_path):
    root,auth=fixture(tmp_path);before=(root/'budget.json').read_bytes()
    value=extend(root,auth)
    assert value['started_s']==100 and value['elapsed_s']==100
    assert value['limit_s']==345600 and value['remaining_s']==345500
    assert value['original_deadline_s']==86500 and value['deadline_s']==345700
    assert value['revision_seq']==1 and len(value['revision_artifacts'])==3
    assert before==(root/'budget.json').read_bytes()


def test_exact_immutable_prefix_survives_further_authorized_extension(tmp_path):
    root,auth=fixture(tmp_path);one=extend(root,auth)
    frozen=dict(one['revision_artifacts']);two=extend(root,auth,432000,201)
    assert two['revision_seq']==2 and len(two['revision_artifacts'])==4
    assert all(budget._sha(p)==h for p,h in frozen.items())
    assert two['started_s']==100 and two['original_limit_s']==86400


def test_authorization_snapshot_is_independent_of_later_status_edits(tmp_path):
    root,auth=fixture(tmp_path);one=extend(root,auth)
    data=json.loads(auth.read_text());data['status']='applied';save(auth,data)
    assert budget.read_budget(root,now=201)['authorization_sha256']==one['authorization_sha256']


@pytest.mark.parametrize('limit',[86400,10,float('inf'),float('nan'),True])
def test_extension_must_be_explicit_finite_and_increasing(tmp_path,limit):
    root,auth=fixture(tmp_path)
    with pytest.raises(ValueError):extend(root,auth,limit)
    assert not (root/'.budget-ledger/origin.json').exists()


def test_mutable_limit_alone_cannot_extend_budget(tmp_path):
    root,_=fixture(tmp_path);save(root/'budget.json',dict(started_s=100,limit_s=345600))
    with pytest.raises(ValueError,match='requires explicit'):budget.read_budget(root,now=200)


def test_authorization_must_bind_original_start(tmp_path):
    root,auth=fixture(tmp_path);data=json.loads(auth.read_text());data['original_started_s']=101;save(auth,data)
    with pytest.raises(ValueError,match='bind'):extend(root,auth)


@pytest.mark.parametrize('damage',['start','gap','chain','snapshot'])
def test_ledger_identity_and_chain_fail_closed(tmp_path,damage):
    root,auth=fixture(tmp_path);extend(root,auth);extend(root,auth,432000,201)
    ledger=root/'.budget-ledger'
    if damage=='start':save(root/'budget.json',dict(started_s=101,limit_s=86400))
    elif damage=='gap':(ledger/'00000001.json').unlink()
    elif damage=='chain':
        data=json.loads((ledger/'00000002.json').read_text());data['previous_sha256']='0'*64;save(ledger/'00000002.json',data)
    else:next((ledger/'authorizations').glob('*.json')).write_text('{}')
    with pytest.raises(ValueError):budget.read_budget(root,now=202)


def test_extension_cannot_compete_with_active_node_owner(tmp_path):
    root,auth=fixture(tmp_path)
    with (root.parent/'node-experiment.lock').open('a') as lock:
        budget.fcntl.flock(lock,budget.fcntl.LOCK_EX|budget.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):extend(root,auth)
    assert not (root/'.budget-ledger').exists()


def test_hard_ceiling_cannot_be_bypassed_by_older_unbounded_authorization(tmp_path):
    root,auth=fixture(tmp_path);old=root/'old-unbounded.json';old.write_bytes(auth.read_bytes())
    value=json.loads(auth.read_text());value['hard_deadline_s']=400000;save(auth,value)
    first=extend(root,auth,300000)
    assert first['hard_deadline_s']==400000
    with pytest.raises(ValueError,match='hard deadline'):extend(root,old,432000,201)
    last=extend(root,old,399900,201)
    assert last['deadline_s']==400000 and last['hard_deadline_s']==400000
    assert last['started_s']==100 and last['elapsed_s']==101


def test_first_extension_cannot_exceed_new_user_hard_ceiling(tmp_path):
    root,auth=fixture(tmp_path);value=json.loads(auth.read_text());value['hard_deadline_s']=200000;save(auth,value)
    with pytest.raises(ValueError,match='hard deadline'):extend(root,auth)
    assert not (root/'.budget-ledger').exists()


def test_old_manifest_cannot_shrink_authorized_deadline_or_stage_cap(tmp_path,monkeypatch):
    root,auth=fixture(tmp_path);extend(root,auth);monkeypatch.setattr(campaign.time,'time',lambda:90000)
    waits=[]
    monkeypatch.setattr(campaign.subprocess,'Popen',lambda *a,**k:SimpleNamespace(wait=lambda timeout:waits.append(timeout) or 0))
    node=campaign.Campaign(root,86400)
    try:
        assert node.state['started_s']==100 and node.state['limit_s']==345600
        node.run('same-short-stage',['synthetic-cpu'],7,gpu=False)
        assert waits==[7] and node.state['last_exit_code']==0 and node.state['last_error'] is None
    finally:node.close()
    assert not (root/'deadline_cleanup.json').exists()
    saved=json.loads((root/'budget.json').read_text())
    assert saved['elapsed_s']==89900 and saved['limit_s']==345600


def cli(tmp_path, child, *, limit=5):
    root=tmp_path/'node';root.mkdir()
    save(root/'budget.json',dict(started_s=time.time()-10,limit_s=86400))
    manifest=tmp_path/'once.campaign.json'
    save(manifest,dict(output=str(root),budget_s=86400,stages=[dict(name='cpu',gpu=False,limit_s=limit,
        argv=[sys.executable,'-c',child])]))
    env=dict(os.environ,PYTHONPATH=str(Path(campaign.__file__).parents[2]))
    argv=[sys.executable,'-m','ecopadg.serving.campaign','--manifest',str(manifest)]
    return root,manifest,env,argv


@pytest.mark.parametrize('code',[0,7])
def test_real_cpu_cli_records_actual_stage_exit_and_refuses_replay(tmp_path,code):
    root,manifest,env,argv=cli(tmp_path,f'raise SystemExit({code})')
    result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==(0 if code==0 else 1),result.stderr
    receipt=manifest.with_suffix('.execution-result.json');raw=json.loads(receipt.read_text())
    assert raw['stage_outcomes'][0]['exit_code']==code and raw['complete']==(code==0)
    assert raw['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
    state=json.loads((root/'budget.json').read_text());assert state['last_exit_code']==code
    before=receipt.read_bytes();again=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=10)
    assert again.returncode!=0 and receipt.read_bytes()==before


def test_cancel_after_successful_child_cleanup_is_not_queue_success(tmp_path):
    marker=tmp_path/'ready'
    child=("import signal,time\nfrom pathlib import Path\n"
           "def stop(*a):raise SystemExit(0)\n"
           "signal.signal(signal.SIGTERM,stop)\n"
           f"Path({str(marker)!r}).write_text('ready')\n"
           "while True:time.sleep(.01)\n")
    root,manifest,env,argv=cli(tmp_path,child,limit=30)
    process=subprocess.Popen(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        end=time.monotonic()+5
        while not marker.exists() and time.monotonic()<end:time.sleep(.01)
        assert marker.exists()
        process.send_signal(signal.SIGINT)  # Only this newly created CPU test PID.
        process.communicate(timeout=10)
        state=json.loads((root/'budget.json').read_text())
        assert process.returncode==130 and state['last_exit_code']==0
        assert state['last_interrupted'] is True and 'KeyboardInterrupt' in state['last_error']
        receipt=json.loads(manifest.with_suffix('.execution-result.json').read_text())
        assert receipt['complete'] is False and receipt['stage_outcomes'][0]['exit_code']==0
        with pytest.raises(ValueError,match='last completed'):
            campaign_pipeline.queue_completion({'active_campaign':str(manifest)},state)
    finally:
        if process.poll() is None:process.kill();process.communicate()
