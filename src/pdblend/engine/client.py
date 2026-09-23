"""Async HTTP client for one vLLM V1 instance: mixed, remote-prefill and remote-decode requests."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import aiohttp

from .launcher import SERVED_NAME
from .carry import (CarryProtocolError, QWEN25_SPECIAL_TOKEN_IDS, diagnostic_body, extract_first,
                    token_ids, validate_request)


@dataclass
class Completion:
    request_id: str
    instance_id: str
    submitted_s: float
    first_token_s: Optional[float] = None
    finished_s: Optional[float] = None
    token_times_s: list = field(default_factory=list)
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    kv_transfer_params: Optional[dict] = None
    error: Optional[str] = None
    token_ids: Optional[list[int]] = None
    token_diagnostics: bool = False
    usage_received: bool = False
    stream_done: bool = False
    pd_protocol: Optional[str] = None
    decode_submitted_s: Optional[float] = None
    decode_first_token_s: Optional[float] = None
    decode_completion_tokens: Optional[int] = None
    text_is_diagnostic: bool = False

    @property
    def ttft_s(self) -> Optional[float]:
        return None if self.first_token_s is None else self.first_token_s - self.submitted_s

    @property
    def tpot_s(self) -> Optional[float]:
        if self.first_token_s is None or self.finished_s is None or self.completion_tokens < 2:
            return None
        return (self.finished_s - self.first_token_s) / (self.completion_tokens - 1)


def remote_decode_params() -> dict:
    """Nixl kv_transfer_params for the prefill leg: produce KV, hand off to a remote decoder."""
    return dict(do_remote_decode=True, do_remote_prefill=False, remote_engine_id=None,
                remote_block_ids=None, remote_host=None, remote_port=None)


def remote_prefill_params(handoff: dict) -> dict:
    """Nixl kv_transfer_params for the decode leg, built from the prefill response."""
    params = dict(handoff)
    params.update(do_remote_prefill=True, do_remote_decode=False)
    return params


def p2p_request_id(prefill_zmq: str, decode_zmq: str, tag: str) -> str:
    return f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{tag}"


class PDTransfer:
    """How a prefill→decode handoff is expressed for a connector.

    Nixl: handoff travels in kv_transfer_params (prefill response → decode request).
    P2pNccl: both legs share a request id that names the two engines' ZMQ addresses;
    the engines route the KV themselves and the bodies carry no kv_transfer_params.
    """

    def __init__(self, connector: Optional[str], zmq_addresses: Optional[Mapping[str, str]] = None,
                 *, special_token_ids: Sequence[int] = QWEN25_SPECIAL_TOKEN_IDS):
        self.connector = connector
        self.zmq = dict(zmq_addresses or {})
        self.special_token_ids = tuple(special_token_ids)
        if connector == "P2pNcclConnector" and not self.zmq:
            raise ValueError("P2pNcclConnector needs instance_id -> zmq_address")

    @property
    def p2p(self) -> bool:
        return self.connector == "P2pNcclConnector"

    def request_id(self, prefill_instance: str, decode_instance: str, tag: str) -> str:
        if self.p2p:
            return p2p_request_id(self.zmq[prefill_instance], self.zmq[decode_instance], tag)
        return tag

    def prefill_params(self) -> Optional[dict]:
        return None if self.p2p else remote_decode_params()

    def decode_params(self, handoff: Optional[dict]) -> Optional[dict]:
        return None if self.p2p else remote_prefill_params(handoff)


class EngineClient:
    def __init__(self, instance_id: str, base_url: str, session: Optional[aiohttp.ClientSession] = None,
                 timeout_s: float = 600.0):
        self.instance_id = instance_id
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._own_session = session is None
        self.timeout = aiohttp.ClientTimeout(total=timeout_s, sock_read=timeout_s)

    def _new_session(self) -> aiohttp.ClientSession:
        # uvicorn drops idle keep-alive connections after 5 s; expire ours first so we never reuse a dead one
        return aiohttp.ClientSession(timeout=self.timeout, connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=3.0))

    async def __aenter__(self):
        if self._session is None:
            self._session = self._new_session()
        return self

    async def __aexit__(self, *exc):
        if self._own_session and self._session is not None:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = self._new_session()
        return self._session

    def _body(self, prompt: Sequence[int] | str, max_tokens: int, stream: bool,
              kv_transfer_params: Optional[dict], **sampling) -> dict:
        body: dict[str, Any] = dict(model=SERVED_NAME, prompt=list(prompt) if not isinstance(prompt, str) else prompt,
                                    max_tokens=max_tokens, temperature=0.0, ignore_eos=True, stream=stream)
        if stream:
            body["stream_options"] = dict(include_usage=True)
        if kv_transfer_params is not None:
            body["kv_transfer_params"] = kv_transfer_params
        body.update(sampling)
        return body

    async def _stream(self, body: dict, request_id: str, result: Completion,
                      on_token: Optional[Callable[[Completion, float], None]] = None) -> None:
        async with self.session.post(self.base_url + "/v1/completions", json=body,
                                     headers={"X-Request-Id": request_id}) as resp:
            if resp.status != 200:
                result.error = f"{resp.status}: {(await resp.text())[:300]}"
                return
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    result.stream_done = True
                    break
                event = json.loads(payload)
                now = time.time()
                if event.get("error"):
                    result.error = str(event["error"])
                    return
                for choice in event.get("choices", ()):
                    text = choice.get("text", "")
                    result.text += text
                    ids = token_ids(choice) if result.token_diagnostics else None
                    if ids is not None:
                        result.token_ids.extend(ids)
                    if ids or (ids is None and (text or choice.get("finish_reason") is None)):
                        if result.first_token_s is None:
                            result.first_token_s = now
                        result.token_times_s.extend([now] * (len(ids) if ids is not None else 1))
                        if on_token is not None:
                            on_token(result, now)
                usage = event.get("usage")
                if usage:
                    result.usage_received = True
                    result.prompt_tokens = usage.get("prompt_tokens", 0)
                    result.completion_tokens = usage.get("completion_tokens", 0)
                if event.get("kv_transfer_params"):
                    result.kv_transfer_params = event["kv_transfer_params"]

    async def complete(self, prompt: Sequence[int] | str, max_tokens: int, request_id: str,
                       kv_transfer_params: Optional[dict] = None,
                       on_token: Optional[Callable[[Completion, float], None]] = None,
                       token_diagnostics: bool = False,
                       **sampling) -> Completion:
        """Streamed completion; records per-token arrival times."""
        result = Completion(request_id, self.instance_id, time.time())
        result.token_diagnostics = token_diagnostics
        result.token_ids = [] if token_diagnostics else None
        body = self._body(prompt, max_tokens, True, kv_transfer_params, **sampling)
        if token_diagnostics:
            body = diagnostic_body(body)
        for attempt in range(2):
            try:
                if on_token is None:
                    await self._stream(body, request_id, result)
                else:
                    await self._stream(body, request_id, result, on_token=on_token)
                break
            except (aiohttp.ClientError, TimeoutError, CarryProtocolError, ValueError) as exc:
                # a keep-alive connection closed by the idle server surfaces here before any byte is read; retry once
                if attempt == 0 and result.first_token_s is None and isinstance(exc, (aiohttp.ClientOSError,
                                                                                      aiohttp.ServerDisconnectedError)):
                    result.submitted_s = time.time()
                    continue
                result.error = repr(exc)
        result.finished_s = time.time()
        if not result.completion_tokens:
            result.completion_tokens = len(result.token_times_s)
        if token_diagnostics and (not result.stream_done or not result.usage_received or len(result.token_ids) != result.completion_tokens):
            result.error = result.error or "token diagnostics disagree with final usage"
        return result

    async def prefill_remote(self, prompt: Sequence[int] | str, request_id: str,
                             kv_transfer_params: Optional[dict] = None, token_diagnostics: bool = False,
                             special_token_ids: Sequence[int] = QWEN25_SPECIAL_TOKEN_IDS,
                             **sampling) -> Completion:
        """Prefill leg: one token, KV kept for a remote decoder. Non-streamed."""
        result = Completion(request_id, self.instance_id, time.time())
        body = self._body(prompt, 1, False, kv_transfer_params, **sampling)
        try:
            validate_request(body)
            body = diagnostic_body(body)
            async with self.session.post(self.base_url + "/v1/completions", json=body,
                                         headers={"X-Request-Id": request_id}) as resp:
                text = await resp.text()
                result.finished_s = result.first_token_s = time.time()
                if resp.status != 200:
                    result.error = f"{resp.status}: {text[:300]}"
                    return result
                data = json.loads(text)
                result.kv_transfer_params = data.get("kv_transfer_params")
                usage = data.get("usage") or {}
                result.usage_received = bool(usage)
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)
                first = extract_first(data, prompt_tokens=len(prompt), received_s=result.first_token_s,
                                      allow_unsafe_text=token_diagnostics, special_token_ids=special_token_ids)
                result.text = first.text
                result.token_ids = [first.token_id]
                result.token_times_s = [first.received_s]
                if kv_transfer_params is not None and result.kv_transfer_params is None:
                    result.error = "prefill response carried no kv_transfer_params"
        except (aiohttp.ClientError, TimeoutError, CarryProtocolError, ValueError) as exc:
            result.error = repr(exc)
            result.finished_s = time.time()
        return result

    async def metrics(self) -> dict:
        """Parse the Prometheus endpoint into the few gauges the controller needs."""
        async with self.session.get(self.base_url + "/metrics") as resp:
            text = await resp.text()
        wanted = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
                  "vllm:kv_cache_usage_perc": "kv_usage", "vllm:gpu_cache_usage_perc": "kv_usage"}
        out = dict(running=0.0, waiting=0.0, kv_usage=0.0, t_s=time.time())
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in wanted:
                try:
                    out[wanted[name]] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        return out

    async def cancel(self, request_id: str) -> dict:
        """Ask the engine to cancel a request and return the native HTTP receipt.

        vLLM releases differ in cancellation endpoint support; a 404 is kept as
        ``unsupported_engine`` by callers rather than being treated as a
        successful cancellation.
        """
        try:
            async with self.session.delete(self.base_url + f"/v1/completions/{request_id}") as resp:
                body = (await resp.text())[:300]
                return {"status": resp.status, "supported": resp.status not in (404, 405), "body": body}
        except (aiohttp.ClientError, TimeoutError) as exc:
            return {"status": None, "supported": False, "error": repr(exc)}

    async def health(self) -> bool:
        try:
            async with self.session.get(self.base_url + "/health",
                                        timeout=aiohttp.ClientTimeout(total=2)) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, TimeoutError, OSError):
            return False

    async def sleep(self, level: int = 1) -> None:
        async with self.session.post(self.base_url + f"/sleep?level={level}") as resp:
            resp.raise_for_status()

    async def wake_up(self) -> None:
        async with self.session.post(self.base_url + "/wake_up") as resp:
            resp.raise_for_status()


async def pd_complete(transfer: PDTransfer, prefill: EngineClient, decode: EngineClient,
                      prompt: Sequence[int] | str, max_tokens: int, tag: str,
                      token_diagnostics: bool = False,
                      **sampling) -> tuple[Completion, Optional[Completion]]:
    """Return (prefill leg, combined logical completion), preserving the real P token.

    Raw D arrival time remains in ``decode_first_token_s``. The combined TTFT is
    the P token's arrival; it must not be interpreted as KV transfer duration.
    """
    body = prefill._body(prompt, max_tokens, True, None, **sampling)
    try:
        validate_request(body)
    except CarryProtocolError as exc:
        return Completion(tag, prefill.instance_id, time.time(), error=str(exc)), None
    if max_tokens == 1:
        direct = await prefill.complete(prompt, 1, tag, token_diagnostics=token_diagnostics, **sampling)
        direct.pd_protocol = "single_engine_no_remote_kv"
        return direct, direct
    request_id = transfer.request_id(prefill.instance_id, decode.instance_id, tag)
    pre = await prefill.prefill_remote(prompt, request_id, transfer.prefill_params(),
                                      token_diagnostics=token_diagnostics,
                                      special_token_ids=transfer.special_token_ids, **sampling)
    if pre.error:
        return pre, None
    dec = await decode.complete([*prompt, pre.token_ids[0]], max_tokens-1, request_id,
        kv_transfer_params=transfer.decode_params(pre.kv_transfer_params), token_diagnostics=token_diagnostics, **sampling)
    dec.decode_submitted_s, dec.decode_first_token_s = dec.submitted_s, dec.first_token_s
    dec.decode_completion_tokens = dec.completion_tokens
    if not dec.error and (not dec.stream_done or not dec.usage_received or dec.prompt_tokens != len(prompt)+1 or dec.completion_tokens != max_tokens-1):
        dec.error = "carry decode usage must report the extended prompt and exact remaining output budget"
    dec.submitted_s, dec.first_token_s = pre.submitted_s, pre.first_token_s
    dec.token_times_s = [pre.first_token_s, *dec.token_times_s]
    dec.text = pre.text + dec.text
    dec.prompt_tokens, dec.completion_tokens = len(prompt), 1+dec.completion_tokens
    dec.token_ids = [pre.token_ids[0], *dec.token_ids] if token_diagnostics else None
    dec.pd_protocol = "carry_first_token"
    dec.text_is_diagnostic = True
    return pre, dec
