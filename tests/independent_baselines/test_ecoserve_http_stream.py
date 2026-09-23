"""Exercise actual SSE sockets through both EcoServe transport entry points."""
import asyncio

import pytest
from aiohttp import web

from pdblend_baselines.ecoserve.runtime import HttpEcoServeTransport, MappedEcoServeTransport


@pytest.mark.asyncio
@pytest.mark.parametrize('mapped', [False, True])
@pytest.mark.parametrize('close_early', [False, True])
async def test_incremental_stream_and_consumer_cleanup(mapped, close_early):
    received = asyncio.Event()
    disconnected = asyncio.Event()
    terminal = asyncio.Event()

    async def generate(request):
        payload = await request.json()
        assert payload['instance_id'] == 'e0'
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(b'data: {"token_ids": [42], "finished": false}\n\n')
        await asyncio.wait_for(received.wait(), 3)
        if close_early:
            try:
                while True:
                    await response.write(b': heartbeat\n\n')
                    await asyncio.sleep(.01)
            except (ConnectionResetError, RuntimeError):
                disconnected.set()
        else:
            terminal.set()
            await response.write(b'data: {"token_ids": [43], "finished": true}\n\n')
            await response.write(b'data: [DONE]\n\n')
        return response

    app = web.Application()
    app.router.add_post('/baseline/generate', generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    url = 'http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1])
    client = MappedEcoServeTransport({'e0': url}) if mapped else HttpEcoServeTransport(url)
    stream = client.stream('e0', {'request_id': 'r', 'prompt': [1], 'max_tokens': 2})
    try:
        first = await asyncio.wait_for(anext(stream), 3)
        assert first['token_ids'] == [42]
        assert not terminal.is_set()
        received.set()
        if close_early:
            await stream.aclose()
            await asyncio.wait_for(disconnected.wait(), 3)
        else:
            second = await asyncio.wait_for(anext(stream), 3)
            assert second['finished'] and second['token_ids'] == [43]
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(stream), 3)
    finally:
        received.set()
        await stream.aclose()
        await runner.cleanup()
