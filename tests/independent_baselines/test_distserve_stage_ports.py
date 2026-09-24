"""Real CPU TCP reservations, immutable lease binding and cross-member failures."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket

import pytest

from pdblend_baselines.distserve.stage_collect import specs_for
from pdblend_baselines.distserve import stage_ports as p


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


_used=set()
def free_port(count=1):
    for _ in range(100):
        held=[]
        try:
            s=socket.socket();held.append(s);s.bind(('0.0.0.0',0));base=s.getsockname()[1]
            if base+count>=65536 or any(base+i in _used for i in range(count)):continue
            for offset in range(1,count):
                s=socket.socket();held.append(s);s.bind(('0.0.0.0',base+offset))
            _used.update(base+i for i in range(count));return base
        except OSError:pass
        finally:
            for s in held:s.close()
    raise RuntimeError('no free CPU test ports')


def test_exact_actual_native_tp_rank_inventory_has_no_implicit_control_listener():
    a=specs_for('/models/Qwen2.5-7B-Instruct',1,[0,1],10600)
    b=specs_for('/models/Qwen2.5-32B-Instruct',2,[0,1,2,3],10200)
    rows=p.inventory(a+b)
    assert sorted(r['port'] for r in rows)==[10200,10216,10600,10616,30200,30201,30216,30217,30600,30616]
    assert all(r['reservation_bind_address']=='0.0.0.0' for r in rows)
    with pytest.raises(ValueError,match='overlap'):p.inventory(a+a)
    with pytest.raises(ValueError,match='side-channel'):p.inventory([replace(a[0],side_channel_port=25000)])


def test_http_wildcard_and_explicit_kv_base_are_enumerated_from_real_argv():
    spec=specs_for('/models/Qwen2.5-32B-Instruct',2,[0,1,2,3],10200)[0]
    rows=p.inventory([replace(spec,kv_port=31000)])
    assert [r['port'] for r in rows]==[10200,31000,31001]
    assert rows[0]['engine_bind_address']=='127.0.0.1'


def registrations():
    members=['seven','fourteen','thirtytwo'];rows=[]
    for index,count in enumerate((2,2,4)):
        rows.append(dict(contract=p.CONTRACT,status='reserved',lease=dict(cohort_id='cohort',
            member=members[index],expected_members=members,lease_id='lease'+str(index),
            gpu_uuids=[f'GPU-{index}-{i}' for i in range(count)]),
            ports=[dict(port=15000+index*100+j) for j in range(count+2)]))
    return members,rows


@pytest.mark.parametrize('fault',['port','gpu','lease','member','contract','failed'])
def test_complete_cross_job_inventory_rejects_overlap_and_unbound_members(fault):
    members,rows=registrations();assert p.validate_registrations(rows,members,'cohort')['status']=='passed'
    if fault=='port':rows[2]['ports'][0]['port']=rows[0]['ports'][0]['port']
    elif fault=='gpu':rows[2]['lease']['gpu_uuids'][0]=rows[0]['lease']['gpu_uuids'][0]
    elif fault=='lease':rows[2]['lease']['lease_id']=rows[0]['lease']['lease_id']
    elif fault=='member':rows[2]['lease']['expected_members']=['other']
    elif fault=='contract':rows[2]['contract']='other'
    else:rows[2]['status']='failed'
    with pytest.raises(ValueError):p.validate_registrations(rows,members,'cohort')


def make_guard(tmp_path,monkeypatch,index=0,members=None):
    members=members or ['seven','fourteen','thirtytwo'];member=members[index]
    tp=2 if index==2 else 1
    root=tmp_path/'coord';root.mkdir(exist_ok=True)
    write(root/'wave.json',dict(cohort_id='cohort',members=members,startup_port_contract=p.CONTRACT))
    model=f'/models/Qwen2.5-{32 if tp==2 else 7}B-Instruct'
    specs=[replace(s,port=free_port(),kv_port=free_port(tp)) for s in specs_for(model,tp,list(range(tp*2)),12000)]
    gpu=[f'GPU-{index}-{i}' for i in range(tp*2)]
    manifest=write(tmp_path/member/'manifest.json',dict(immutable=True,lease_id='lease'+str(index),
        job_id='job'+str(index),gpu_uuids=gpu,payload=dict(sampling_cohort='cohort',cohort_member=member,gpu_count=tp*2)))
    environment=write(tmp_path/member/'concurrency.json',dict(lease_id='lease'+str(index),
        allocated_gpu_uuids=gpu,lease_manifest_file='manifest.json',lease_manifest_sha256=p.sha(manifest)))
    monkeypatch.setenv('PDBLEND_GPU_UUIDS',','.join(gpu))
    return p.PortReservations(root,member,specs,tmp_path/member/'receipts',environment_path=environment,
                              timeout_s=.4,proc_root='/proc')


def test_all_members_reserve_before_any_model_load_and_handoff_only_its_engine(tmp_path,monkeypatch):
    guards=[make_guard(tmp_path,monkeypatch,i) for i in range(3)]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            values=list(pool.map(lambda g:g.reserve(),guards))
        assert all(v['status']=='passed' for v in values)
        for guard in guards:
            assert len(guard.sockets)==len(guard.ports)
            for port in guard.sockets:
                with socket.socket() as other:
                    other.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                    with pytest.raises(OSError):other.bind(('127.0.0.1',port))
        first=guards[0];spec=first.specs[0]
        first.before_engine_start(spec)
        assert len(first.sockets)==len(first.ports)//2
        with socket.socket() as other:other.bind(('127.0.0.1',spec.port))
        with pytest.raises(ValueError,match='exactly once'):first.before_engine_start(spec)
    finally:
        for guard in guards:guard.close()
    assert all(g.released and not g.sockets for g in guards)


@pytest.mark.parametrize('reuse',[False,True])
def test_existing_loopback_listener_is_not_stolen_and_failure_retains_pid_evidence(tmp_path,monkeypatch,reuse):
    guard=make_guard(tmp_path,monkeypatch)
    with socket.socket() as other:
        other.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,int(reuse))
        other.bind(('127.0.0.1',guard.ports[-1]['port']));other.listen(1)
        with pytest.raises(OSError):guard.reserve()
        assert guard.released and not guard.sockets
        failure=next(row for row in guard.events if row['kind']=='failed')
        assert any(r['port']==guard.ports[-1]['port'] for r in failure['snapshot']['sockets'])
        assert any(r['pid']==os.getpid() for r in failure['snapshot']['processes'])
        assert other.fileno()>=0


def test_peer_failure_and_missing_member_timeout_release_every_reservation(tmp_path,monkeypatch):
    guard=make_guard(tmp_path,monkeypatch);guard.timeout=.01
    with pytest.raises(ValueError,match='timeout'):guard.reserve()
    assert guard.released and not guard.sockets
    assert (guard.directory/(guard.member+'.failed.json')).is_file()


def test_corrupted_queue_lease_or_wrong_physical_gpu_rejected_before_binding(tmp_path,monkeypatch):
    guard=make_guard(tmp_path,monkeypatch);guard.close()
    environment=tmp_path/'seven/concurrency.json'
    data=json.loads(environment.read_text());data['lease_id']='other';write(environment,data)
    with pytest.raises(ValueError,match='real lease'):
        p.lease_binding(environment,member='seven',cohort_id='cohort',members=guard.members)


def test_actual_listening_socket_is_joined_to_observed_pid_not_container_pid_guess():
    with socket.socket() as owned:
        owned.bind(('127.0.0.1',0));owned.listen(1);port=owned.getsockname()[1]
        snapshot=p.socket_snapshot([port])
        receipt=p.listener_ownership(snapshot,os.getpid())
        assert receipt['passed'] and receipt['engine_observer_pid']==os.getpid()
        foreign=deepcopy(snapshot)
        for row in foreign['processes']:row['pid_namespace']='pid:[other]'
        with pytest.raises(ValueError,match='namespace'):p.listener_ownership(foreign,os.getpid())
