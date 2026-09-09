import asyncio
import json
import time
import aiohttp
from aiohttp import web
from ecopadg.serving.controller import Controller
from benchmarks.scripts.bench_vllm import send_request


def test_nonstream_client_updates_live_slack_and_receives_every_token(tmp_path):
    async def run():
        release=asyncio.Event()
        async def state(request):
            return web.json_response(dict(role='mixed',generation=0,timestamp=time.time(),
                free_kv_tokens=4096,running=0,waiting=0,accepting=True))
        async def completion(request):
            body=await request.json()
            assert body['stream'] is True
            count=body['max_tokens']
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            for index in range(count):
                event=dict(id='upstream',token_ids=[10+index],choices=[dict(text='' if index==0 else 'x',index=0)],
                    usage=dict(prompt_tokens=1,completion_tokens=count,total_tokens=count+1) if index==count-1 else None)
                await response.write(('data: '+json.dumps(event)+'\n\n').encode())
                if index==0: await release.wait()
            await response.write(b'data: [DONE]\n\n')
            return response
        engine=web.Application();engine.router.add_get('/runtime',state)
        engine.router.add_post('/v1/completions',completion)
        upstream=web.AppRunner(engine);await upstream.setup()
        site=web.TCPSite(upstream,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        c=Controller(dict(strategy='mixed',journal=str(tmp_path/'events'),manage_clocks=False,
            prepare_peers=False,slo_ttft_s=2,slo_tpot_s=.1,
            instances=[dict(id='m',tp=1,role='mixed',gpus=[0],url=f'http://127.0.0.1:{port}')]))
        proxy=web.AppRunner(c.application());await proxy.setup()
        site=web.TCPSite(proxy,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession(trust_env=False) as session:
                async def client(count):
                    async with session.post(f'http://127.0.0.1:{port}/v1/completions',
                        json=dict(prompt=[1],max_tokens=count,stream=False)) as response:
                        assert response.status==200
                        return await response.json()
                tasks=[asyncio.create_task(client(count)) for count in (3,5)]
                deadline=time.monotonic()+2
                while sum(a['budget'].emitted==1 for a in c.active.values())!=2:
                    assert time.monotonic()<deadline
                    await asyncio.sleep(.005)
                assert all(not task.done() for task in tasks)
                budgets=[a['budget'] for a in c.active.values()]
                assert all(b.first_token_s is not None for b in budgets)
                # Same visible input and completed history, different stopping
                # bounds: neither bound may become an output prediction.
                assert {b.predicted_output for b in budgets}=={256}
                assert {b.output_limit for b in budgets}=={3,5}
                assert not c.predictor.history[c.predictor.bucket(1)]
                release.set()
                for count,result in zip((3,5),await asyncio.gather(*tasks)):
                    assert result['token_ids']==list(range(10,10+count))
                    assert result['choices'][0]['text']=='x'*(count-1)
                    assert result['usage']['completion_tokens']==count
                assert sorted(c.predictor.history[c.predictor.bucket(1)])==[3,5]
        finally:
            release.set()
            await proxy.cleanup();await upstream.cleanup()
    asyncio.run(run())


def test_deadline_and_queue_refusals_are_identified_without_engine_execution(tmp_path):
    async def run():
        async def state(request):
            return web.json_response(dict(role='mixed',generation=0,timestamp=time.time(),
                free_kv_tokens=0,running=0,waiting=0,accepting=True))
        engine=web.Application();engine.router.add_get('/runtime',state)
        upstream=web.AppRunner(engine);await upstream.setup()
        site=web.TCPSite(upstream,'127.0.0.1',0);await site.start()
        engine_port=site._server.sockets[0].getsockname()[1]
        c=Controller(dict(strategy='mixed',journal=str(tmp_path/'events'),manage_clocks=False,
            prepare_peers=False,slo_ttft_s=.1,slo_tpot_s=.1,max_pending=1,
            instances=[dict(id='m',tp=1,role='mixed',gpus=[0],url=f'http://127.0.0.1:{engine_port}')]))
        proxy=web.AppRunner(c.application());await proxy.setup()
        site=web.TCPSite(proxy,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        task=None
        try:
            async with aiohttp.ClientSession(trust_env=False) as session:
                url=f'http://127.0.0.1:{port}'
                task=asyncio.create_task(send_request(session,url,'model',[1],3,request_id='0'))
                deadline=time.monotonic()+1
                while not c.pending.full():
                    assert time.monotonic()<deadline
                    await asyncio.sleep(.001)
                second=await send_request(session,url,'model',[1],3,request_id='1')
                first=await task
                assert first['admission_rejection']=='admission_deadline'
                assert second['admission_rejection']=='admission_queue_full'
                for result in (first,second):
                    assert result['http_status']==429 and not result['success']
                    assert result['generated_tokens']==0 and result['token_ids']==[]
                assert not c.state.reservations and c.failure is None
        finally:
            if task and not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
            await proxy.cleanup();await upstream.cleanup()
    asyncio.run(run())
