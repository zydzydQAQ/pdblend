"""CPU proof that DistServe HTTP generation forwards SSE events incrementally."""
import asyncio
import time

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web

from pdblend_baselines.distserve.runtime import HttpDistServeTransport


@pytest.mark.asyncio
async def test_generate_yields_first_event_before_server_terminal_event():
    terminal = asyncio.Event()
    client_received = asyncio.Event()

    async def generate(request):
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"token_index": 1}\n\n')
        await asyncio.wait_for(client_received.wait(), timeout=1.0)
        assert not terminal.is_set()
        await response.write(b'data: {"token_index": 2, "finished": true}\n\n')
        terminal.set()
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/baseline/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        transport = HttpDistServeTransport(f"http://127.0.0.1:{port}")
        started = time.monotonic()
        events = []
        first_at = None
        async for event in transport.generate({"request_id": "stream-test"}):
            if first_at is None:
                first_at = time.monotonic()
                assert not terminal.is_set()
                client_received.set()
            events.append(event)
        assert [e["token_index"] for e in events] == [1, 2]
        assert first_at is not None
        assert terminal.is_set()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_generate_early_close_disconnects_server_and_closes_client():
    disconnected = asyncio.Event()
    client_received = asyncio.Event()

    async def generate(request):
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"token_index": 1}\n\n')
        client_received.set()
        try:
            while True:
                await asyncio.sleep(0.01)
                await response.write(b": keepalive\n\n")
        except (ConnectionResetError, RuntimeError):
            # aiohttp reports a closed peer either from write() or the
            # transport cleanup path, depending on event-loop timing.
            pass
        finally:
            disconnected.set()
        return response

    app = web.Application()
    app.router.add_post("/baseline/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        transport = HttpDistServeTransport(f"http://127.0.0.1:{port}")
        stream = transport.generate({"request_id": "early-close"})
        first = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert first["token_index"] == 1
        await stream.aclose()
        await asyncio.wait_for(disconnected.wait(), timeout=1.0)
    finally:
        await runner.cleanup()
