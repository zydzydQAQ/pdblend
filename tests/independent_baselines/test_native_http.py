"""Real ASGI serialization/admission tests; no GPU qualification is implied."""
import importlib.util
import time
from types import SimpleNamespace

import pytest

pytest.importorskip('fastapi')
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pdblend_runtime.serve import router


class Engine:
    def __init__(self):
        self.engine_core=self
        self.generation=0
        self.accepting=True
        self.worker_generation=0
        self.calls=[]
        self.partial_load=False
    async def call_utility_async(self,method,operation,payload):
        self.calls.append((method,operation,payload))
        if operation=='control':
            self.generation=payload.get('generation',self.generation)
            self.accepting=payload.get('accepting',self.accepting)
        return dict(generation=self.generation,acknowledged_generation=self.generation,
            native_evidence_complete=True,all_queue=[],running=[],waiting=[],tp=2,pp=1,
            accepting=self.accepting,acknowledged=True,native_at_s=time.time())
    async def collective_rpc(self,method,kwargs):
        self.calls.append((method,kwargs))
        if method=='native_generation_set':self.worker_generation=kwargs['generation']
        rows=[dict(rank=i,generation=self.worker_generation,acknowledged=True,
             native_evidence_complete=True,transfer_allocations={},pending_transfers=0,
             healthy=True,retained_kv_supported=True,at_s=time.time()) for i in range(2)]
        if method=='native_kv_operation':
            for row in rows:
                row.update(transaction_id='tx',target_request_id='kv',expected_layers=28,loaded_layers=28)
            if self.partial_load:rows[1]['loaded_layers']=27
        return rows


@pytest.mark.asyncio
async def test_native_generation_installs_real_ranks_and_stale_request_does_not_mutate():
    app=FastAPI();app.include_router(router);engine=Engine()
    app.state.engine_client=engine;app.state.native_tp=2;app.state.native_pp=1
    async with AsyncClient(transport=ASGITransport(app),base_url='http://test') as client:
        response=await client.post('/baseline/control',json={'generation':2})
        assert response.status_code==200,response.text
        assert response.json()['ranks'][1]['generation']==2
        assert engine.generation==engine.worker_generation==2 and engine.accepting
        count=len([r for r in engine.calls if r[0]=='native_generation_set'])
        response=await client.post('/baseline/control',json={'generation':1})
        assert response.status_code==409
        assert len([r for r in engine.calls if r[0]=='native_generation_set'])==count
        await client.post('/baseline/control',json={'accepting':False})
        assert (await client.post('/baseline/generate',json={})).status_code==409


@pytest.mark.asyncio
async def test_native_load_ack_requires_every_rank_every_layer():
    app=FastAPI();app.include_router(router);engine=Engine()
    app.state.engine_client=engine;app.state.native_tp=2;app.state.native_pp=1
    async with AsyncClient(transport=ASGITransport(app),base_url='http://test') as client:
        payload=dict(target_request_id='kv',transaction_id='tx',generation=0)
        response=await client.post('/baseline/distserve/load_ack',json=payload)
        assert response.status_code==200,response.text
        engine.partial_load=True
        response=await client.post('/baseline/distserve/load_ack',json=payload)
        assert response.status_code==503
