"""Private gateway RPC failure contracts; CPU oracles never qualify serving."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('vllm')
from fastapi import HTTPException
from pdblend_baselines.dynamollm import stationary_target_service as service
from pdblend_baselines.dynamollm.stationary_service import StationaryContext
from pdblend_baselines.dynamollm.stationary_target_bootstrap import TargetBootstrap
from pdblend_baselines.dynamollm.stationary_ipc import process_identity
from tests.independent_baselines.test_dynamo_stationary_target_bootstrap import config,bound
from tests.independent_baselines.test_dynamo_stationary_service import NativeOracle
from tests.independent_baselines.test_dynamo_stationary_kv import drained


@pytest.mark.parametrize('fault',[None,'wrong_uuid','wrong_epoch','dead_target','rpc_failure'])
def test_source_only_exports_to_exact_live_bound_target_after_actual_fresh_drain(tmp_path,monkeypatch,fault):
    c=config(tmp_path);bootstrap=TargetBootstrap(bound(tmp_path,'bootstrap.json',c))
    ready=bootstrap.publish_ready(gpu_uuid='GPU-0',device_index=0)
    if fault=='wrong_epoch':ready['target_generation']=1
    ready_ref=bound(tmp_path,'ready.json',ready)
    native=NativeOracle(tp=1);native.generation=0
    p=dict(transaction_id='tx',expected_generation=0,expected_gpu_uuids=['GPU-0'])
    context=StationaryContext();context.phase='released';context.transaction=deepcopy(p)
    called=[]
    async def ranks(method,*,operation,payload):
        assert native.drains and native.drains[-1]['accepting'] is False
        called.append(operation)
        if fault=='rpc_failure':raise RuntimeError('real source RPC lost')
        return [dict(rank=0,generation=0,transaction_id='tx',gpu_uuid='GPU-other' if fault=='wrong_uuid' else 'GPU-0',
            packet=dict(consumer=ready['process'],plan_sha256=ready['plan_sha256']))]
    native.ranks=ranks
    if fault=='dead_target':monkeypatch.setattr(service,'process_matches',lambda _:False)
    coordinator=service.TargetAwareSourceCoordinator(context,native)
    if fault is None:
        result=asyncio.run(coordinator.execute('export_to_target',dict(p,target_ready_ref=ready_ref)))
        assert result['packet']['consumer']==ready['process'] and context.phase=='released'
        assert called==['export'] and not result['formal_eligible']
    else:
        with pytest.raises(HTTPException) as caught:
            asyncio.run(coordinator.execute('export_to_target',dict(p,target_ready_ref=ready_ref)))
        assert caught.value.status_code==503 and context.phase=='quarantined' and not native.accepting
        if fault in ('wrong_epoch','dead_target'):assert not called


@pytest.mark.parametrize('fault',[None,'rank','epoch','uuid','consumer','unsynchronized'])
def test_target_close_requires_exact_real_worker_identity_and_consumer_clean_ack(tmp_path,monkeypatch,fault):
    from pdblend_runtime import serve
    c=config(tmp_path);bootstrap=TargetBootstrap(bound(tmp_path,'bootstrap.json',c));identity=process_identity()
    app=NS(state=NS(dynamo_target_bootstrap=bootstrap,dynamo_target_lock=asyncio.Lock()))
    async def payload():return dict(transaction_id='tx',expected_generation=0)
    request=NS(app=app,json=payload)
    async def drain(request,payload):
        value=drained(1);value['generation']=0
        for row in value['ranks']:row['generation']=0
        return value
    async def workers(request,method,*,operation,payload):
        row=dict(rank=0,generation=0,gpu_uuid='GPU-0',process=identity)
        if operation=='status':return [row]
        ack=dict(consumer=identity,gpu_uuid='GPU-0',cuda_synchronized=True,views_released=True)
        row.update(views_released=True,release_ack=ack)
        if fault=='rank':row['rank']=1
        elif fault=='epoch':row['generation']=1
        elif fault=='uuid':row['gpu_uuid']='GPU-other'
        elif fault=='consumer':ack['consumer']=dict(identity,pid=identity['pid']+1)
        elif fault=='unsynchronized':ack['cuda_synchronized']=False
        return [row]
    monkeypatch.setattr(serve,'drain_engine',drain);monkeypatch.setattr(serve,'workers',workers)
    if fault is None:
        result=asyncio.run(service.target_operation(request,'close'))
        assert result['target_process_exit_required'] and not result['formal_eligible']
    else:
        with pytest.raises(HTTPException) as caught:asyncio.run(service.target_operation(request,'close'))
        assert caught.value.status_code==409
