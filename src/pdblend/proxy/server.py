"""OpenAI-compatible proxy: forwards completions to engines along the router's chosen path."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

import aiohttp
from aiohttp import web

from ..engine.client import PDTransfer
from ..engine.launcher import SERVED_NAME
from .router import RequestRecord, Router
from .sse import StreamScan


class Proxy:
    def __init__(self, instances: dict[str, str], router: Optional[Router] = None,
                 pd_threshold_tokens: int = 0, transfer: Optional[PDTransfer] = None):
        """instances: instance_id -> base_url."""
        self.urls = dict(instances)
        self.router = router or Router(self.urls, pd_threshold_tokens)
        self.transfer = transfer or PDTransfer("NixlConnector")
        self.session: Optional[aiohttp.ClientSession] = None
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_post("/v1/completions", self.completions)
        self.app.router.add_get("/admin/roles", self.get_roles)
        self.app.router.add_post("/admin/roles", self.post_roles)
        self.app.router.add_get("/admin/stats", self.stats)
        self.app.router.add_get("/health", self.health)
        self.app.on_startup.append(self._startup)
        self.app.on_cleanup.append(self._cleanup)

    async def _startup(self, app):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=600), connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=3.0))

    async def _cleanup(self, app):
        if self.session:
            await self.session.close()

    async def health(self, request):
        return web.json_response(dict(ok=True))

    async def get_roles(self, request):
        return web.json_response(dict(roles=self.router.roles(), pd_threshold_tokens=self.router.pd_threshold_tokens))

    async def post_roles(self, request):
        body = await request.json()
        self.router.set_roles(body.get("roles", {}), body.get("pd_threshold_tokens"))
        return await self.get_roles(request)

    async def stats(self, request):
        loads = {i: dict(role=l.role, inflight_prefill_tokens=l.inflight_prefill_tokens,
                         inflight_seqs=l.inflight_seqs) for i, l in self.router.loads.items()}
        return web.json_response(dict(loads=loads, rejected=self.router.rejected,
                                      records=len(self.router.records)))

    async def completions(self, request):
        body = await request.json()
        prompt = body.get("prompt")
        input_tokens = len(prompt) if isinstance(prompt, list) else body.get("prompt_tokens") or 0
        max_tokens = int(body.get("max_tokens", 16))
        request_id = body.pop("request_id", None) or f"req-{uuid.uuid4().hex[:12]}"
        record = self.router.dispatch(request_id, input_tokens, max_tokens)
        if record is None:
            return web.json_response(dict(error="no instance accepting requests"), status=503)
        engine_body = dict(body, model=SERVED_NAME, stream=True, stream_options=dict(include_usage=True))
        engine_id = request_id
        if record.path == "PD":
            engine_id = self.transfer.request_id(record.prefill_instance, record.decode_instance, request_id)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                               "X-PDBlend-Path": record.path,
                                               "X-PDBlend-Prefill": record.prefill_instance,
                                               "X-PDBlend-Decode": record.decode_instance})
        completion_tokens, error = 0, None
        try:
            if record.path == "PD":
                handoff = await self._prefill_leg(record, dict(engine_body), engine_id)
                params = self.transfer.decode_params(handoff)
                if params is not None:
                    engine_body["kv_transfer_params"] = params
            await response.prepare(request)
            completion_tokens = await self._stream_leg(record, engine_body, engine_id, response)
        except Exception as exc:  # engine failure surfaces as an SSE error event
            error = repr(exc)
            if not response.prepared:
                await response.prepare(request)
            await response.write(f"data: {json.dumps(dict(error=error))}\n\n".encode())
        finally:
            self.router.finish(record, completion_tokens, error)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def _prefill_leg(self, record: RequestRecord, body: dict, engine_id: str) -> Optional[dict]:
        body.update(max_tokens=1, stream=False)
        body.pop("stream_options", None)
        params = self.transfer.prefill_params()
        if params is not None:
            body["kv_transfer_params"] = params
        async with self.session.post(self.urls[record.prefill_instance] + "/v1/completions", json=body,
                                     headers={"X-Request-Id": engine_id}) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"prefill leg {resp.status}: {text[:300]}")
            handoff = json.loads(text).get("kv_transfer_params")
            if params is not None and not handoff:
                raise RuntimeError("prefill leg returned no kv_transfer_params")
            return handoff

    async def _stream_leg(self, record: RequestRecord, body: dict, engine_id: str,
                          response: web.StreamResponse) -> int:
        scan, first = StreamScan(), False
        async with self.session.post(self.urls[record.decode_instance] + "/v1/completions", json=body,
                                     headers={"X-Request-Id": engine_id}) as resp:
            if resp.status != 200:
                raise RuntimeError(f"decode leg {resp.status}: {(await resp.text())[:300]}")
            async for chunk in resp.content.iter_any():
                fwd, n = scan.feed(chunk)
                if n:
                    if not first:
                        self.router.first_token(record)
                        first = True
                    record.tokens_so_far = scan.tokens
                if fwd:
                    await response.write(fwd)
                if scan.done:
                    break
        return scan.completion_tokens()


def run(instances: dict[str, str], host: str = "127.0.0.1", port: int = 8000, **kw) -> None:
    proxy = Proxy(instances, **kw)
    web.run_app(proxy.app, host=host, port=port, print=None)
