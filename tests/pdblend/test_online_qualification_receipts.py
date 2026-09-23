import json
from types import SimpleNamespace as NS

import pytest

from pdblend.bench import online_qualification as run
from pdblend.online.router import RequestRecord
from pdblend.results.receipts import request_record_receipt


@pytest.mark.asyncio
async def test_live_qualification_returns_json_array_ownership_without_mutating_record(tmp_path,monkeypatch):
    record=RequestRecord('r','PD','a','b',512,32,1.25,first_token_s=2.5,finished_s=3.75,
                         completion_tokens=32,engine_instances={'b','a'},terminal_state='completed')
    closed=[]
    class Native:
        def __init__(self,*args,**kwargs):pass
        async def drain(self,iid,timeout):return dict(instance_id=iid,acknowledged=True,drained=True)
        async def cancel(self,*args,**kwargs):raise AssertionError('receipt test does not invoke engine work')
    class Controller:
        def __init__(self,*args,**kwargs):self.transition_events=[dict(phase='drain',duration_s=.125)]
        async def execute(self,plan):pass
    class Runner:
        async def cleanup(self):closed.append(True)
    async def serve(*args):return Runner()
    async def qualify(proxy,*args,**kwargs):
        proxy.router.records.append(record)
        return dict(functional_passed=True)
    monkeypatch.setattr(run,'NativeControl',Native)
    monkeypatch.setattr(run,'Controller',Controller)
    monkeypatch.setattr(run,'PoolPlanner',lambda *args:None)
    monkeypatch.setattr(run,'_serve_proxy',serve)
    monkeypatch.setattr(run,'qualify_live_proxy',qualify)
    spec=lambda iid:NS(tp=1,pp=1,generation=0,pool_id='pool',profile_key='profile',
                       model='Qwen2.5-7B-Instruct',base_url='http://localhost/'+iid)
    fleet=NS(instances={iid:NS(spec=spec(iid)) for iid in ('a','b')})
    result=await run.qualify(fleet,None,NS(freqs=[900,1500]),tmp_path,12345)
    # No default=str escape hatch: the actual returned object is JSON-safe.
    assert json.loads(json.dumps(result))==result
    assert result['routes'][0]['engine_instances']==['a','b']
    assert result['routes'][0]['completion_tokens']==32 and result['routes'][0]['finished_s']==3.75
    assert record.engine_instances=={'a','b'} and closed==[True]
    assert result['routes'][0]==request_record_receipt(record)
