import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def transport():
    pytest.importorskip('torch')
    pytest.importorskip('vllm')
    path=Path('/root/workspace/vllm-pd-fork/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py')
    spec=importlib.util.spec_from_file_location('pdb_test_transport',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine=module.P2pNcclEngine.__new__(module.P2pNcclEngine)
    engine.send_type='PUT'
    engine.config=SimpleNamespace(get_from_extra_config=lambda k,d:d)
    engine.recv_store_cv=threading.Condition()
    engine.pool_lock=threading.Lock()
    engine.device='cpu'
    engine.buffer_size=0
    engine.rank=0
    engine.cancelled_requests={}
    engine._listener_thread=SimpleNamespace(is_alive=lambda:True)
    return engine


@pytest.mark.parametrize('load_fails',[False,True])
def test_pinned_staging_reclaimed_after_import_even_on_copy_failure(transport,load_fails):
    freed=[]
    result=object()
    def load(*args):
        if load_fails:
            raise RuntimeError('copy failed')
        return result
    transport.pool=SimpleNamespace(load_tensor=load,free=freed.append)
    transport.recv_store={'nonce#k':(123,'dtype',(4,))}
    transport.recv_request_id_to_tensor_ids={'nonce':{'nonce#k'}}
    if load_fails:
        with pytest.raises(RuntimeError,match='copy failed'):
            transport.recv_tensor('nonce#k')
    else:
        assert transport.recv_tensor('nonce#k') is result
    assert freed==[123]
    assert not transport.recv_store and not transport.recv_request_id_to_tensor_ids


def test_dead_listener_or_cancelled_transfer_fails_without_waiting(transport):
    transport.recv_store={}
    transport._listener_thread=SimpleNamespace(is_alive=lambda:False)
    with pytest.raises(RuntimeError,match='listener'):
        transport.recv_tensor('nonce#k')
    transport._listener_thread=SimpleNamespace(is_alive=lambda:True)
    transport.cancelled_requests={'nonce':None}
    with pytest.raises(RuntimeError,match='cancelled'):
        transport.recv_tensor('nonce#k')


def test_transfer_staging_capacity_is_distinct_from_decode_kv():
    from dataclasses import replace
    from test_planner import system
    planner,snapshot,request=system()
    snapshot=replace(snapshot,instances=tuple(replace(i,free_transfer_bytes=1)
                        for i in snapshot.instances if i.role!='mixed'))
    assert not planner.plan(snapshot,(request,),now=10).feasible
