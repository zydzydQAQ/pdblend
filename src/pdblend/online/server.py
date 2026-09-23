"""OpenAI-compatible proxy: forwards completions to engines along the router's chosen path."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

import aiohttp
from aiohttp import web

from pdblend.engine.client import PDTransfer
from pdblend.engine.carry import CarryProtocolError, SSEEvents, combined_usage, decode_body, diagnostic_body, encode_event, extract_first, first_event, token_ids, validate_request
from pdblend.engine.launcher import SERVED_NAME
from pdblend.online.router import DuplicateRequestError, RequestRecord, Router
from pdblend.online.sse import StreamScan, TerminalStream


class Proxy:
    def __init__(self, instances: dict[str, str], router: Optional[Router] = None,
                 pd_threshold_tokens: int = 0, transfer: Optional[PDTransfer] = None,
                 native_cancel=None, cancel_timeout_s: float = 15.0):
        """instances: instance_id -> base_url."""
        self.urls = dict(instances)
        self.router = router or Router(self.urls, pd_threshold_tokens)
        self.transfer = transfer or PDTransfer("NixlConnector")
        self.native_cancel = native_cancel
        self.cancel_timeout_s = cancel_timeout_s
        self._cleanup_tasks: set[asyncio.Task] = set()
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
        if self._cleanup_tasks:
            await asyncio.gather(*tuple(self._cleanup_tasks), return_exceptions=True)
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
        diagnostics = body.pop("pdblend_token_diagnostics", False)
        if type(diagnostics) is not bool:
            return web.json_response(dict(error="pdblend_token_diagnostics must be boolean"), status=400)
        if diagnostics and body.get("logprobs") == 0 and body.get("return_tokens_as_token_ids") is True:
            body.pop("logprobs"); body.pop("return_tokens_as_token_ids")
        prompt = body.get("prompt")
        input_tokens = len(prompt) if isinstance(prompt, list) else body.get("prompt_tokens") or 0
        max_tokens = body.get("max_tokens", 16)
        if type(max_tokens) is not int or max_tokens < 1:
            return web.json_response(dict(error="max_tokens must be a positive integer"), status=400)
        body.setdefault("max_tokens", max_tokens)
        request_id = (body.pop("request_id", None) or getattr(request, 'headers', {}).get('X-Request-Id')
                      or f"req-{uuid.uuid4().hex[:12]}")
        if not isinstance(request_id, str):
            return web.json_response(dict(error="request_id must be a string"), status=400)
        try:
            record = self.router.dispatch(request_id, input_tokens, max_tokens)
        except DuplicateRequestError as exc:
            return web.json_response(dict(error=str(exc)), status=409)
        if record is None:
            return web.json_response(dict(error="no instance accepting requests"), status=503)
        if record.path == "PD":
            try:
                if max_tokens > 1:
                    validate_request(body)
            except CarryProtocolError as exc:
                # No request reached an engine: release the reservation rather
                # than quarantining a healthy pool for an unsupported input.
                self.router.finish(record, 0, str(exc), terminal_state="rejected_before_engine")
                return web.json_response(dict(error=str(exc)), status=400)
            if max_tokens == 1:
                # Ordinary untagged generation on the already chosen P engine.
                # Move sequence accounting before any engine work; retained
                # resident-pool reservations are still released by request ID.
                p, d = record.prefill_instance, record.decode_instance
                self.router.loads[d].inflight_seqs -= 1
                self.router.loads[p].inflight_seqs += 1
                self.router.active[d].remove(record)
                self.router.active[p].append(record)
                record.decode_instance = p
                record.path = "P_ONLY"
                record.route_reason = "single_token_no_remote_kv"
        engine_body = dict(body, model=SERVED_NAME, stream=True, stream_options=dict(include_usage=True))
        if diagnostics and record.path != "PD":
            engine_body = diagnostic_body(engine_body)
        engine_id = request_id
        if record.path == "PD":
            engine_id = self.transfer.request_id(record.prefill_instance, record.decode_instance, request_id)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                               "X-PDBlend-Path": record.path,
                                               "X-PDBlend-Prefill": record.prefill_instance,
                                               "X-PDBlend-Decode": record.decode_instance,
                                               "X-PDBlend-Token-Diagnostics": str(diagnostics).lower(),
                                               "X-PDBlend-PD-Protocol": "carry_first_token" if record.path == "PD" else "single_engine"})
        completion_tokens, error = 0, None
        terminal_state = "uncertain"
        try:
            if record.path == "PD":
                first, handoff = await self._prefill_leg(record, dict(engine_body), engine_id, diagnostics=diagnostics)
                engine_body = decode_body(engine_body, first, diagnostics=diagnostics)
                params = self.transfer.decode_params(handoff)
                if params is not None:
                    engine_body["kv_transfer_params"] = params
                await response.prepare(request)
                await response.write(encode_event(first_event(first, request_id=request_id, model=SERVED_NAME,
                    created=int(record.submitted_s), diagnostics=diagnostics)))
                self.router.first_token(record)
                record.tokens_so_far = completion_tokens = 1
                completion_tokens = await self._stream_pd_leg(record, engine_body, engine_id, response, diagnostics=diagnostics)
            else:
                await response.prepare(request)
                completion_tokens = await self._stream_leg(record, engine_body, engine_id, response)
            terminal_state = "completed"
        except asyncio.CancelledError:
            # An interrupted downstream stream does not acknowledge native
            # KV release. Let resident routing preserve its failure quarantine.
            error = "client_cancelled_without_native_ack"
            raise
        except Exception as exc:  # engine failure surfaces as an SSE error event
            error = repr(exc)
            if not response.prepared:
                await response.prepare(request)
            await response.write(f"data: {json.dumps(dict(error=error))}\n\n".encode())
        finally:
            # Failures before any submission own no native work. Cancelled
            # streaming operations remain uncertain even if a transport hook
            # failed before recording the submission locally.
            if error and not record.engine_instances and error != "client_cancelled_without_native_ack":
                terminal_state = "rejected_before_engine"
            self.router.finish(record, completion_tokens or record.tokens_so_far, error,
                               terminal_state=terminal_state)
            if terminal_state == "uncertain" and self.native_cancel is not None:
                task = asyncio.create_task(self._recover_cancel(record, engine_id))
                self._cleanup_tasks.add(task)
                task.add_done_callback(self._cleanup_tasks.discard)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def _recover_cancel(self, record, engine_id):
        try:
            instances = tuple(sorted(record.engine_instances or
                                     {record.prefill_instance, record.decode_instance}))
            receipts = await asyncio.wait_for(self.native_cancel(record, engine_id, instances),
                                              timeout=self.cancel_timeout_s)
            self.router.recover_cancel(record, receipts, engine_request_id=engine_id)
        except Exception as exc:
            # Cleanup failure preserves both reservation and quarantine. Keep
            # the original request error and attach recovery evidence separately.
            record.route_estimate['cancel_recovery_error'] = repr(exc)

    async def _prefill_leg(self, record: RequestRecord, body: dict, engine_id: str, *, diagnostics=False):
        body.update(max_tokens=1, stream=False)
        body.pop("stream_options", None)
        body = diagnostic_body(body)
        params = self.transfer.prefill_params()
        if params is not None:
            body["kv_transfer_params"] = params
        record.engine_instances.add(record.prefill_instance)
        async with self.session.post(self.urls[record.prefill_instance] + "/v1/completions", json=body,
                                     headers={"X-Request-Id": engine_id}) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"prefill leg {resp.status}: {text[:300]}")
            data = json.loads(text)
            first = extract_first(data, prompt_tokens=record.input_tokens, received_s=time.time(),
                                  allow_unsafe_text=diagnostics, special_token_ids=self.transfer.special_token_ids)
            handoff = data.get("kv_transfer_params")
            if params is not None and not handoff:
                raise RuntimeError("prefill leg returned no kv_transfer_params")
            return first, handoff

    async def _stream_pd_leg(self, record, body, engine_id, response, *, diagnostics=False):
        parser = SSEEvents()
        usage = None
        observed_ids = 0
        record.engine_instances.add(record.decode_instance)
        async with self.session.post(self.urls[record.decode_instance] + "/v1/completions", json=body,
                                     headers={"X-Request-Id": engine_id}) as resp:
            if resp.status != 200:
                raise RuntimeError(f"decode leg {resp.status}: {(await resp.text())[:300]}")
            async for chunk in resp.content.iter_any():
                for event in parser.feed(chunk):
                    if event.get("error"):
                        raise RuntimeError(f"decode leg error: {event['error']}")
                    event["id"] = record.request_id
                    if len(event.get("choices", ())) > 1:
                        raise CarryProtocolError("carry stream must return one choice")
                    for choice in event.get("choices", ()):
                        if choice.get("index", 0) != 0:
                            raise CarryProtocolError("carry stream returned another choice")
                        if diagnostics:
                            ids = token_ids(choice)
                            observed_ids += len(ids)
                            self.router.token(record, count=len(ids))
                        elif choice.get("text"):
                            self.router.token(record)
                    if event.get("usage") is not None:
                        usage = combined_usage(event["usage"], original_prompt_tokens=record.input_tokens,
                                               max_tokens=record.max_tokens)
                        event["usage"] = usage
                    await response.write(encode_event(event))
                if parser.done:
                    break
        if not parser.done or usage is None or usage["completion_tokens"] != record.max_tokens:
            raise CarryProtocolError("carry stream lacks terminal DONE and exact combined usage")
        if diagnostics and observed_ids + 1 != usage["completion_tokens"]:
            raise CarryProtocolError("carry stream token IDs disagree with combined usage")
        record.tokens_so_far = usage["completion_tokens"]
        return usage["completion_tokens"]

    async def _stream_leg(self, record: RequestRecord, body: dict, engine_id: str,
                          response: web.StreamResponse) -> int:
        scan = StreamScan()
        terminal = TerminalStream(record.max_tokens,
                                  prompt_tokens=record.input_tokens if isinstance(body.get('prompt'), list) else None,
                                  choices=int(body.get('n', 1)))
        record.engine_instances.add(record.decode_instance)
        async with self.session.post(self.urls[record.decode_instance] + "/v1/completions", json=body,
                                     headers={"X-Request-Id": engine_id}) as resp:
            if resp.status != 200:
                raise RuntimeError(f"decode leg {resp.status}: {(await resp.text())[:300]}")
            async for chunk in resp.content.iter_any():
                terminal.feed(chunk)
                fwd, n = scan.feed(chunk)
                if n:
                    self.router.token(record, count=n)
                if fwd:
                    await response.write(fwd)
                if scan.done:
                    break
        return terminal.completion_tokens()


def run(instances: dict[str, str], host: str = "127.0.0.1", port: int = 8000, **kw) -> None:
    proxy = Proxy(instances, **kw)
    web.run_app(proxy.app, host=host, port=port, print=None)
