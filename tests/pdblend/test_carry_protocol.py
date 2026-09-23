import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp import web

from pdblend.engine.carry import (CarryProtocolError,SSEEvents,combined_usage,decode_body,encode_event,
                                 extract_first,first_event,golden,validate_request)
from pdblend.engine.client import EngineClient,PDTransfer,pd_complete
from pdblend.proxy.router import ResidentRouter,Router
from pdblend.proxy.server import Proxy
from test_e2e import FakeEngineServer


def request():
    return dict(prompt=[10,11,12],max_tokens=4,temperature=0.,ignore_eos=True,seed=701)


def prefill(text='t0'):
    return dict(choices=[dict(index=0,text=text,finish_reason='length',logprobs=dict(tokens=['token_id:9000']))],
                usage=dict(prompt_tokens=3,completion_tokens=1,total_tokens=4))


def test_exact_id_carry_usage_and_strict_golden_never_retokenize():
    body=request();saved=copy.deepcopy(body)
    first=extract_first(prefill(),prompt_tokens=3,received_s=1.)
    d=decode_body(body,first)
    assert body==saved and d['prompt']==[10,11,12,9000] and d['max_tokens']==3 and d['seed']==701
    assert 'logprobs' not in d
    assert decode_body(body,first,diagnostics=True)['return_tokens_as_token_ids'] is True
    assert combined_usage(dict(prompt_tokens=4,completion_tokens=3,total_tokens=7),original_prompt_tokens=3,max_tokens=4)==dict(prompt_tokens=3,completion_tokens=4,total_tokens=7)
    g=golden([1,2,3],[1,2,3],[1,8,3]);assert not g['passed'] and g['first_mismatch_index']==1 and g['pd_token_id']==8
    assert not golden([1,2],[1,3],[1,2])['passed']


@pytest.mark.parametrize('change',[{'prompt':'raw text'},{'prompt':[[1,2]]},{'prompt':[True]},
    {'ignore_eos':False},{'stop':['END']},{'logprobs':0},{'n':2},{'echo':True},{'temperature':.5},
    {'presence_penalty':1},{'min_tokens':4},{'guided_regex':'.*'}])
def test_unsupported_sampling_fails_before_any_engine_work(change):
    body=request();body.update(change)
    with pytest.raises(CarryProtocolError):validate_request(body)


@pytest.mark.parametrize('change',['missing_ids','text_as_id','multiple_ids','wrong_usage','eos','unicode','empty'])
def test_ambiguous_prefill_is_not_accepted(change):
    data=prefill()
    if change=='missing_ids':data['choices'][0].pop('logprobs')
    if change=='text_as_id':data['choices'][0]['logprobs']['tokens']=['t0']
    if change=='multiple_ids':data['choices'][0]['logprobs']['tokens'].append('token_id:2')
    if change=='wrong_usage':data['usage']['completion_tokens']=2
    if change=='eos':data['choices'][0]['finish_reason']='stop'
    if change=='unicode':data['choices'][0]['text']='\ufffd'
    if change=='empty':data['choices'][0]['text']=''
    with pytest.raises(CarryProtocolError):extract_first(data,prompt_tokens=3,received_s=1.)
    if change in ('unicode','empty'):
        assert extract_first(data,prompt_tokens=3,received_s=1.,allow_unsafe_text=True).token_id==9000


def test_sse_fragmentation_unicode_and_final_usage():
    events=[dict(choices=[dict(text='中',index=0)]),dict(choices=[],usage=dict(completion_tokens=4))]
    data=b''.join(encode_event(e) for e in events)+b'data: [DONE]\n\n'
    for split in range(1,len(data)):
        parser=SSEEvents();received=[]
        received.extend(parser.feed(data[:split]));received.extend(parser.feed(data[split:]))
        assert received==events and parser.done
    with pytest.raises(CarryProtocolError):combined_usage(dict(prompt_tokens=3,completion_tokens=3),original_prompt_tokens=3,max_tokens=4)


@pytest.mark.parametrize('special',[151643,151644,151645])
def test_ignore_eos_true_carries_special_ids_and_counts_empty_first_sse(special):
    from pdblend.proxy.sse import StreamScan
    data=prefill('');data['choices'][0]['logprobs']['tokens']=[f'token_id:{special}']
    first=extract_first(data,prompt_tokens=3,received_s=1.)
    decoded=decode_body(request(),first)
    assert decoded['prompt'][-1]==special and decoded['ignore_eos'] is True
    event=first_event(first,request_id='r',model='m',created=1)
    assert event['pdblend_generated_tokens']==1 and event['choices'][0]['text']==''
    encoded=encode_event(event)
    for split in range(1,len(encoded)):
        scan=StreamScan()
        count=scan.feed(encoded[:split])[1]+scan.feed(encoded[split:])[1]
        assert count==1 and scan.tokens==1
    ordinary=first_event(extract_first(prefill(),prompt_tokens=3,received_s=1.),request_id='r',model='m',created=1)
    assert 'pdblend_generated_tokens' not in ordinary
    assert StreamScan().feed(encode_event(ordinary))[1]==1


@asynccontextmanager
async def engines():
    servers=[FakeEngineServer(i) for i in range(2)];runners=[];urls={}
    try:
        for i,server in enumerate(servers):
            runner=web.AppRunner(server.app);await runner.setup();runners.append(runner)
            site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
            urls[f'i{i}']=f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
        yield servers,urls
    finally:
        for runner in runners:await runner.cleanup()


def transfer():return PDTransfer('P2pNcclConnector',{'i0':'127.0.0.1:30100','i1':'127.0.0.1:30200'})


@pytest.mark.asyncio
@pytest.mark.parametrize('diagnostics',[False,True])
async def test_public_client_carries_real_first_token_and_keeps_both_clocks(diagnostics):
    async with engines() as (servers,urls):
        async with EngineClient('i0',urls['i0']) as p,EngineClient('i1',urls['i1']) as d:
            ordinary=await d.complete([10,11,12],4,'ordinary',token_diagnostics=True)
            pre,combined=await pd_complete(transfer(),p,d,[10,11,12],4,'pd',seed=701,token_diagnostics=diagnostics)
        assert not pre.error and not combined.error
        assert pre.token_ids==[9000] and combined.completion_tokens==4 and combined.prompt_tokens==3
        assert combined.text=='t0t1t2t3'
        assert combined.first_token_s==pre.first_token_s and combined.submitted_s==pre.submitted_s
        assert combined.decode_first_token_s>=combined.first_token_s
        assert combined.decode_completion_tokens==3 and len(combined.token_times_s)==4
        assert combined.token_ids==ordinary.token_ids if diagnostics else combined.token_ids is None
        assert servers[0].seen_bodies[-1]['seed']==servers[1].seen_bodies[-1]['seed']==701
        assert servers[1].seen_bodies[-1]['prompt']==[10,11,12,9000]
        assert servers[1].seen_bodies[-1]['max_tokens']==3
        assert servers[0].seen_bodies[-1]['logprobs']==0
        assert ('logprobs' in servers[1].seen_bodies[-1])==diagnostics


@pytest.mark.asyncio
async def test_one_token_client_never_creates_remote_kv():
    async with engines() as (servers,urls):
        async with EngineClient('i0',urls['i0']) as p,EngineClient('i1',urls['i1']) as d:
            pre,result=await pd_complete(transfer(),p,d,[10,11,12],1,'single',token_diagnostics=True)
        assert pre is result and not result.error and result.token_ids==[9000]
        assert result.pd_protocol=='single_engine_no_remote_kv'
        assert servers[0].seen_request_ids==['single'] and not servers[1].seen_request_ids
        assert servers[0].seen_bodies[0]['stream'] and 'kv_transfer_params' not in servers[0].seen_bodies[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('max_tokens,diagnostics',[(1,False),(1,True),(4,False),(4,True)])
@pytest.mark.parametrize('resident',[False,True])
async def test_proxy_merges_first_sse_usage_router_counts_and_diagnostics(max_tokens,diagnostics,resident):
    async with engines() as (servers,urls):
        router=Router(list(urls));router.set_roles({'i0':'P','i1':'D'})
        if resident:
            for iid in urls:
                router.set_instance_metadata(iid,tp=1,pool_id='pool',model_id='Qwen2.5-7B-Instruct',profile_key='test-profile')
            model=SimpleNamespace(freqs=(2100,),kv_capacity_tokens=100000,
                prefill_seconds=lambda n,f:.001,step_seconds=lambda b,c,f:.01)
            router=ResidentRouter({'pool':router},{'pool':model})
        proxy=Proxy(urls,router,transfer=transfer());runner=web.AppRunner(proxy.app)
        await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        url=f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
        try:
            async with EngineClient('proxy',url) as client:
                result=await client.complete([10,11,12],max_tokens,'via-proxy',seed=701,
                    token_diagnostics=diagnostics,pdblend_token_diagnostics=diagnostics)
            assert not result.error,result.error
            assert result.prompt_tokens==3 and result.completion_tokens==max_tokens
            assert result.text==''.join(f't{i}' for i in range(max_tokens))
            assert result.token_ids==list(range(9000,9000+max_tokens)) if diagnostics else result.token_ids is None
            record=router.records[-1]
            assert record.first_token_s is not None and record.completion_tokens==max_tokens
            assert all(x.inflight_prefill_tokens==x.inflight_seqs==0 for x in router.loads.values())
            assert not any(router.active.values())
            if resident:assert not router.selector.active
            if max_tokens==1:
                assert record.path=='P_ONLY' and record.prefill_instance==record.decode_instance=='i0'
                assert not servers[1].seen_bodies and not servers[0].seen_request_ids[0].startswith('___')
            else:
                assert record.path=='PD' and record.tokens_so_far==max_tokens
                assert servers[1].seen_bodies[-1]['max_tokens']==max_tokens-1
                assert ('logprobs' in servers[1].seen_bodies[-1])==diagnostics
        finally:await runner.cleanup()


@pytest.mark.asyncio
async def test_downstream_cancellation_is_not_a_native_terminal_ack(monkeypatch):
    from unittest.mock import AsyncMock
    child=Router(['i0'],instance_metadata={'i0':dict(tp=1,pool_id='pool',model_id='Qwen2.5-7B-Instruct')})
    model=SimpleNamespace(freqs=(2100,),kv_capacity_tokens=100000,
                         prefill_seconds=lambda n,f:.001,step_seconds=lambda b,c,f:.01)
    router=ResidentRouter({'pool':child},{'pool':model})
    proxy=Proxy({'i0':'http://unused'},router)
    response=SimpleNamespace(prepared=True,prepare=AsyncMock(),write=AsyncMock(),write_eof=AsyncMock())
    monkeypatch.setattr('pdblend.proxy.server.web.StreamResponse',lambda **kwargs:response)
    proxy._stream_leg=AsyncMock(side_effect=asyncio.CancelledError())
    req=SimpleNamespace(json=AsyncMock(return_value=dict(request(),request_id='cancelled')))
    with pytest.raises(asyncio.CancelledError):await proxy.completions(req)
    assert router.records[-1].error=='client_cancelled_without_native_ack'
    assert 'cancelled' in router.selector.active and router.quarantined=={'i0'}
