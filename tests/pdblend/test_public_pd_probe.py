"""The whole public protocol probe against real CPU HTTP/SSE endpoints."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from pdblend.engine.client import Completion,EngineClient
from pdblend_runtime.public_pd_probe import check_completion,run
from test_carry_protocol import engines


@dataclass
class Spec:
    instance_id:str
    base_url:str
    zmq_address:str
    port:int
    model:str='Qwen2.5-7B-Instruct'
    tp:int=1
    pp:int=1
    gpus:tuple=()


def specs(urls,public_port):
    # Public probe reserves p.port+48. Fake engine URLs have separately bound
    # ephemeral ports; no GPU/model process or live-queue port is touched.
    return [Spec('i0',urls['i0'],'127.0.0.1:30100',public_port-48,gpus=(0,)),
            Spec('i1',urls['i1'],'127.0.0.1:30200',public_port-32,gpus=(1,))]


def test_incomplete_receipt_never_passes_from_token_ids_alone():
    value=Completion('r','i0',0,first_token_s=1,finished_s=2,prompt_tokens=128,
                     completion_tokens=1,token_ids=[151645])
    checks=check_completion(value,[151645],prompt_tokens=128,output_tokens=1)
    assert not checks['stream_done'] and not checks['usage_received']
    value.stream_done=value.usage_received=True
    assert all(check_completion(value,[151645],prompt_tokens=128,output_tokens=1).values())


@pytest.mark.asyncio
async def test_whole_probe_covers_three_lengths_pd_mixed_and_proxy_single(tmp_path,unused_tcp_port):
    async with engines() as (servers,urls):
        output=tmp_path/'public.json'
        result=await run(specs(urls,unused_tcp_port),output)
        assert result['complete'] and result['status']=='passed',result.get('error')
        assert [r['input_tokens'] for r in result['rows']]==[512,2048,7168]
        for row in result['rows']:
            assert all(row['client_checks'].values())
            assert all(row['proxy_PD']['checks'].values()) and all(row['proxy_M']['checks'].values())
        single=result['single_token']
        assert all(single['reference_checks'].values()) and all(single['checks'].values())
        assert all(single['proxy']['checks'].values()) and single['proxy']['record']['path']=='P_ONLY'
        assert len(result['router_records'])==7
        assert json.loads(output.read_text())==result
        # Both one-token client and proxy calls are untagged ordinary P work.
        assert servers[0].seen_request_ids[-2]=='public-single-client'
        assert not servers[0].seen_request_ids[-1].startswith('___')
        assert all(body['max_tokens']>0 for server in servers for body in server.seen_bodies)


@pytest.mark.asyncio
@pytest.mark.parametrize('target,field,value',[
    ('public-single-reference','error','stream failed after first token'),
    ('public-single-reference','stream_done',False),
    ('public-single-reference','usage_received',False),
    ('public-proxy-single','usage_received',False),
])
async def test_faults_in_single_reference_or_proxy_do_not_pass(tmp_path,unused_tcp_port,monkeypatch,target,field,value):
    original=EngineClient.complete
    async def altered(self,prompt,max_tokens,request_id,**kwargs):
        completion=await original(self,prompt,max_tokens,request_id,**kwargs)
        if request_id==target:setattr(completion,field,value)
        return completion
    monkeypatch.setattr(EngineClient,'complete',altered)
    async with engines() as (_,urls):
        result=await run(specs(urls,unused_tcp_port),tmp_path/'failed.json')
    assert not result['complete'] and result['status']=='failed'
    if target=='public-single-reference':
        assert 'reference is incomplete' in result['error']
        assert not all(result['single_token']['reference_checks'].values())
    else:
        assert 'proxy P_ONLY path failed' in result['error']
        assert not result['single_token']['proxy']['checks']['usage_received']


@pytest.mark.asyncio
async def test_gate_and_profiler_use_real_client_second_output_and_reject_legacy_resume(monkeypatch,unused_tcp_port):
    from pdblend.bench.gates import _gate_kv
    from pdblend.engine.client import PDTransfer
    from pdblend.engine.handoff_timing import PROTOCOL
    from pdblend.profile import profiler as profiler_module
    async with engines() as (servers,urls):
        layout=specs(urls,unused_tcp_port)
        fleet=SimpleNamespace(instances={s.instance_id:SimpleNamespace(spec=s) for s in layout})
        transfer=PDTransfer('P2pNcclConnector',{s.instance_id:s.zmq_address for s in layout})
        gate=await _gate_kv(fleet,transfer,[512],1,4,1024)
        assert gate['summary']['512']['token_ids_match_all']
        assert gate['summary']['512']['physical_bandwidth'] is False
        row=gate['rows'][0]
        assert row['handoff_timing']['protocol']==PROTOCOL and not row['handoff_timing']['physical_copy_time']
        assert row['mixed_token_ids']==row['pd_token_ids']==[9000,9001,9002,9003]
        # Execute the production collector method without constructing any
        # Profiler/model/GPU object. Its HTTP requests still use real clients.
        profiler=object.__new__(profiler_module.Profiler)
        profiler.raw={'transfer':[]};profiler.transfer=transfer;profiler.parallel_layout={'test':'cpu_http'}
        checkpoints=[];profiler._checkpoint=lambda:checkpoints.append(len(profiler.raw['transfer']))
        monkeypatch.setattr(profiler_module,'TRANSFER_INPUTS',(512,))
        p,d=list(fleet.instances.values())
        await profiler._transfer(p,d)
        point=profiler.raw['transfer'][0]
        assert point['protocol']==PROTOCOL and point['runs']==3 and len(point['timing_repeats'])==3
        assert point['measured_gpu_ids']==[0,1] and checkpoints==[1]
        assert all(r['physical_copy_time'] is False for r in point['timing_repeats'])
        requests=sum(len(server.seen_bodies) for server in servers)
        await profiler._transfer(p,d)
        assert requests==sum(len(server.seen_bodies) for server in servers)
        profiler.raw['transfer']=[dict(input_tokens=512,overhead_s=.01)]
        with pytest.raises(ValueError,match='legacy transfer timing'):
            await profiler._transfer(p,d)
