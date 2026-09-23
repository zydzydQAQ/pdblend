"""Independent DistServe runtime adapter.

The schedulers in :mod:`policy` make placement decisions, while this module
owns the deliberately small native-v1 contract.  A missing capability or KV
acknowledgement is an error; the adapter never treats an HTTP response as an
execution receipt by itself.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.request import Request as URLRequest, urlopen

from .policy import DecodeScheduler, PrefillScheduler, Request


from .request_runtime import (DistServeCapabilityError, DistServeTransport,
                              DistServeResult, DistServeRuntime, _ack)


class HttpDistServeTransport:
    """Small standalone HTTP transport for the native-v1 service."""
    def __init__(self, base_url: str): self.base_url = base_url.rstrip("/")

    async def _call(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        def call():
            data = None if body is None else json.dumps(body).encode()
            req = URLRequest(self.base_url + path, data=data, method=method,
                             headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read())
        try:
            return await asyncio.to_thread(call)
        except Exception as exc:
            raise DistServeCapabilityError(f"native DistServe endpoint unavailable: {exc}") from exc

    async def capability(self):
        state = await self.state()
        return {"supported": state.get("native_evidence_complete") is True,
                "tp": state.get("tp", 1), "pp": state.get("pp", 1)}
    async def state(self): return await self._call("GET", "/baseline/state")
    async def prefill(self, request): return await self._call("POST", "/baseline/distserve/prefill", {"request_id": request.request_id, "prompt_token_ids": request.payload.get("prompt_token_ids", list(range(request.input_tokens)))})
    async def transfer(self, retained_handle, target_request_id, target_address, target_tp, generation):
        return await self._call("POST", "/baseline/distserve/transfer", {"held_request_id":retained_handle,"target_request_id":target_request_id,"target_address":target_address,"target_tp":target_tp,"generation":generation})
    async def release(self, retained_handle): return await self._call("POST", "/baseline/distserve/release", {"held_request_id":retained_handle})
    async def decode(self, request): return await self._call("POST", "/baseline/control", {"operation": "decode", "request_id": request.request_id, "kv_handle": request.kv_handle, "max_tokens": request.output_tokens})
    async def cancel(self, request_id): return await self._call("POST", "/baseline/cancel", {"request_id": request_id})

    async def _generate(self, payload):
        # Keep the response open and yield each SSE event as soon as it is
        # received.  Buffering ``list(lines())`` delays TTFT until generation
        # completes and leaves a reader thread alive when the caller cancels.
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise DistServeCapabilityError("aiohttp is required for streaming generation") from exc
        timeout = aiohttp.ClientTimeout(total=300)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.base_url + "/baseline/generate",
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    if response.status != 200:
                        body = await response.text()
                        raise DistServeCapabilityError(
                            f"native DistServe generate returned HTTP {response.status}: {body[:500]}"
                        )
                    async for raw in response.content:
                        line = raw.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "":
                            continue
                        if data == "[DONE]":
                            break
                        yield json.loads(data)
        except asyncio.CancelledError:
            raise
        except DistServeCapabilityError:
            raise
        except Exception as exc:
            raise DistServeCapabilityError(f"native DistServe streaming endpoint unavailable: {exc}") from exc

    def generate(self, payload):
        return self._generate(payload)


class MappedDistServeTransport:
    """Explicit P/D transport; no shared planner or implicit endpoint choice."""
    def __init__(self, prefill_url: str, decode_url: str, *, prefill_address=None, decode_address=None):
        self.prefill_url=prefill_url.rstrip('/'); self.decode_url=decode_url.rstrip('/')
        self.prefill_address, self.decode_address = prefill_address, decode_address
    async def _call(self, base, method, path, body=None):
        return await HttpDistServeTransport(base)._call(method,path,body)
    async def capability(self, role): return await self._call(self.prefill_url if role=='P' else self.decode_url,'GET','/baseline/capability')
    async def state(self, role): return await self._call(self.prefill_url if role=='P' else self.decode_url,'GET','/baseline/state')
    async def prefill(self, request): return await self._call(self.prefill_url,'POST','/baseline/distserve/prefill',request)
    async def transfer(self, payload): return await self._call(self.prefill_url,'POST','/baseline/distserve/transfer',payload)
    async def expect_load(self, payload): return await self._call(self.decode_url,'POST','/baseline/distserve/expect_load',payload)
    def generate_decode(self, payload):
        return HttpDistServeTransport(self.decode_url).generate(payload)
    async def load_ack(self, payload): return await self._call(self.decode_url,'POST','/baseline/distserve/load_ack',payload)
    async def release(self, payload): return await self._call(self.prefill_url,'POST','/baseline/distserve/release',payload)
    async def cancel(self, role, payload): return await self._call(self.prefill_url if role=='P' else self.decode_url,'POST','/baseline/cancel',payload)


def main(argv=None):
    from .run_native import main as run
    return run(argv)


if __name__ == "__main__": raise SystemExit(main())
