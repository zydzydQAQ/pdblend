"""Standalone EcoServe runtime and native-v1 transport.

This is an adapter around the independent author controller.  It deliberately
requires native state/event/control acknowledgements and never substitutes the
shared PDblend planner when a field is missing.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
import json
from typing import Any, Protocol
from urllib.request import Request as URLRequest, urlopen

from .controller import EcoServeController


class EcoServeCapabilityError(RuntimeError):
    pass


class EcoServeTransport(Protocol):
    async def state(self, identifier: str) -> dict[str, Any]: ...
    async def events(self, identifier: str, after_seq: int = 0) -> dict[str, Any]: ...
    async def json(self, identifier: str, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def clock(self, gpus: list[int], frequency: int) -> dict[str, Any]: ...
    async def park(self, gpus: list[int]) -> dict[str, Any]: ...
    async def cancel(self, identifier: str, request_id: str) -> dict[str, Any]: ...
    async def stream(self, identifier: str, payload: dict[str, Any]): ...
    def generate(self, payload: dict[str, Any]): ...


def validate_native_state(state: dict[str, Any], *, block_size: int = 16) -> None:
    required = ("free_kv_tokens", "total_kv_tokens", "block_size", "generation",
                "acknowledged_generation", "native_at_s", "native_evidence_complete")
    if not isinstance(state, dict) or any(k not in state for k in required):
        raise EcoServeCapabilityError("native-v1 EcoServe state fields are incomplete")
    if state["block_size"] != block_size or state["native_evidence_complete"] is not True:
        raise EcoServeCapabilityError("native-v1 EcoServe evidence is incomplete")
    if state.get("error") or state.get("runtime_error"):
        raise EcoServeCapabilityError("native-v1 EcoServe state is unhealthy")
    if state["acknowledged_generation"] != state["generation"]:
        raise EcoServeCapabilityError("native-v1 EcoServe generation is not acknowledged")


class EcoServeRuntime:
    def __init__(self, config: dict[str, Any], transport: EcoServeTransport, journal):
        self.config = config
        self.transport = transport
        self.controller = EcoServeController(config, transport, journal)
        if hasattr(transport, 'bind_specs'):
            transport.bind_specs(self.controller.specs)
        self.started = False

    async def start(self):
        for spec in self.controller.specs.values():
            validate_native_state(await self.transport.state(spec["id"]))
            events = await self.transport.events(spec["id"], after_seq=0)
            if not isinstance(events, dict) or "events" not in events or "next_seq" not in events:
                raise EcoServeCapabilityError("native-v1 EcoServe event receipt is incomplete")
        await self.controller.startup()
        self.started = True

    async def handle(self, payload: dict[str, Any], request_id: str):
        if not self.started:
            await self.start()
        async for event in self.controller.handle(payload, request_id):
            yield event

    async def close(self):
        await self.controller.close()
        self.started = False


class HttpEcoServeTransport:
    def __init__(self, base_url: str): self.base_url = base_url.rstrip("/")

    async def _call(self, method: str, path: str, body=None):
        def call():
            data = None if body is None else json.dumps(body).encode()
            req = URLRequest(self.base_url + path, data=data, method=method,
                             headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=30) as response: return json.loads(response.read())
        try: return await asyncio.to_thread(call)
        except Exception as exc: raise EcoServeCapabilityError(f"native-v1 endpoint unavailable: {exc}") from exc

    async def state(self, identifier): return await self._call("GET", f"/baseline/state?instance_id={identifier}")
    async def events(self, identifier, after_seq=0):
        batch=await self._call("GET", f"/baseline/events?instance_id={identifier}&after_seq={after_seq}")
        return self._native_events(batch)

    @staticmethod
    def _native_events(batch):
        # The author journal's first argument is named kind. Preserve the
        # native event type without colliding with that positional argument.
        rows=[]
        for event in batch['events']:
            row=dict(event)
            if 'kind' in row:row['native_kind']=row.pop('kind')
            rows.append(row)
        return dict(batch,events=rows)
    async def json(self, identifier, path, payload): return await self._call("POST", f"/baseline{path}?instance_id={identifier}", payload)
    async def clock(self, gpus, frequency): return await self._call("POST", "/baseline/clock", {"gpus": gpus, "frequency": frequency})
    async def park(self, gpus): return await self._call("POST", "/baseline/park", {"gpus": gpus})
    async def cancel(self, identifier, request_id): return await self._call("POST", "/baseline/cancel", {"instance_id": identifier, "request_id": request_id})

    async def stream(self, identifier, payload):
        payload = dict(payload, instance_id=identifier)
        async with aclosing(self.generate(payload)) as stream:
            async for event in stream:
                yield event

    async def _generate(self, payload):
        import aiohttp
        # An async response belongs to the request lifetime. A cancelled or
        # closed consumer must not leave a blocking reader thread generating
        # tokens and retaining native KV after the controller has moved on.
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                async with session.post(self.base_url+'/baseline/generate', json=payload,
                                        headers={'Accept': 'text/event-stream'}) as response:
                    if response.status != 200:
                        raise EcoServeCapabilityError(
                            f'native generate HTTP {response.status}: {(await response.text())[:500]}')
                    async for raw in response.content:
                        line = raw.decode().strip()
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        if data == '[DONE]':
                            break
                        if data:
                            yield json.loads(data)
        except asyncio.CancelledError:
            raise
        except EcoServeCapabilityError:
            raise
        except Exception as exc:
            raise EcoServeCapabilityError(f'native streaming endpoint unavailable: {exc}') from exc

    def generate(self, payload): return self._generate(payload)


class MappedEcoServeTransport(HttpEcoServeTransport):
    """Transport for fixed TP instances at separate native service URLs."""
    def __init__(self, endpoints: dict[str, str]):
        self.endpoints = {str(k): str(v).rstrip('/') for k, v in endpoints.items()}
        if not self.endpoints: raise ValueError('at least one EcoServe endpoint required')
        super().__init__(next(iter(self.endpoints.values())))
        self.clients = {identifier: HttpEcoServeTransport(url) for identifier, url in self.endpoints.items()}
        self.gpu_groups = {}

    def bind_specs(self, specs):
        if set(specs) != set(self.endpoints):
            raise EcoServeCapabilityError('instance specifications differ from native endpoints')
        groups = {identifier: tuple(spec['gpus']) for identifier, spec in specs.items()}
        flat = [gpu for group in groups.values() for gpu in group]
        if any(not group for group in groups.values()) or len(flat) != len(set(flat)):
            raise EcoServeCapabilityError('EcoServe GPU groups must be nonempty and disjoint')
        self.gpu_groups = groups

    def _base_for(self, identifier):
        try: return self.endpoints[identifier]
        except KeyError as exc: raise EcoServeCapabilityError('missing endpoint for '+str(identifier)) from exc

    async def _instance_call(self, identifier, method, path, body=None):
        self._base_for(identifier)
        return await self.clients[identifier]._call(method, path, body)

    async def state(self, identifier): return await self._instance_call(identifier, 'GET', '/baseline/state')
    async def events(self, identifier, after_seq=0):
        return self._native_events(await self._instance_call(identifier, 'GET', f'/baseline/events?after_seq={after_seq}'))
    async def json(self, identifier, path, payload):
        return await self._instance_call(identifier, 'POST', '/baseline'+path if not path.startswith('/baseline/') else path, payload)
    async def cancel(self, identifier, request_id): return await self._instance_call(identifier, 'POST', '/baseline/cancel', {'request_id':request_id})
    async def stream(self, identifier, payload):
        self._base_for(identifier)
        async with aclosing(self.clients[identifier]._generate(dict(payload, instance_id=identifier))) as stream:
            async for event in stream:
                yield event

    async def _group_operation(self, gpus, path, payload):
        requested = set(gpus)
        groups = [identifier for identifier, group in self.gpu_groups.items() if set(group) <= requested]
        covered = {gpu for identifier in groups for gpu in self.gpu_groups[identifier]}
        if not requested or len(gpus) != len(requested) or covered != requested:
            raise EcoServeCapabilityError('clock/park requires exactly configured whole TP groups')
        receipts = await asyncio.gather(*[self._instance_call(identifier, 'POST', path, payload) for identifier in groups])
        if not all(row.get('acknowledged') is True for row in receipts):
            raise EcoServeCapabilityError('missing native GPU group clock receipt')
        return dict(success=True, acknowledged=True, instances=groups, receipts=receipts)

    async def clock(self, gpus, frequency):
        return await self._group_operation(gpus, '/baseline/clock', dict(frequency_mhz=frequency))

    async def park(self, gpus):
        return await self._group_operation(gpus, '/baseline/park', {})


def main(argv=None):
    parser = argparse.ArgumentParser(description="EcoServe native-v1 PP1 capability probe")
    parser.add_argument("--url", required=True)
    args = parser.parse_args(argv)
    async def run():
        # A service-specific config is required for actual admission. This
        # command only confirms that the native state endpoint is reachable.
        result = await HttpEcoServeTransport(args.url)._call("GET", "/baseline/state")
        validate_native_state(result)
        print(json.dumps({"supported": True, "block_size": result["block_size"]}, sort_keys=True))
    asyncio.run(run())


if __name__ == "__main__": main()
