"""Validated, policy-neutral native control for online parking and recovery."""
from __future__ import annotations

import asyncio
import time
from typing import Mapping

import aiohttp


class NativeControlError(RuntimeError):
    pass


def validate_state(value: dict, *, generation: int, tp: int, pp: int = 1,
                   drained: bool = False, request_id: str | None = None,
                   observed_after_s: float | None = None) -> dict:
    """Never substitute proxy counts or HTTP success for native evidence."""
    if (not isinstance(value, dict) or value.get('generation') != generation
            or (value.get('tp'), value.get('pp')) != (tp, pp)
            or value.get('native_evidence_complete') is not True
            or value.get('transport_healthy') is not True):
        raise NativeControlError('native identity, generation or transport evidence missing')
    if observed_after_s is not None and value.get('native_at_s', 0) < observed_after_s:
        raise NativeControlError('stale native scheduler evidence')
    ranks = value.get('ranks')
    if (not isinstance(ranks, list) or len(ranks) != tp * pp
            or {r.get('rank') for r in ranks} != set(range(tp * pp))
            or any(r.get('generation') != generation or r.get('native_evidence_complete') is not True
                   or r.get('healthy') is not True for r in ranks)):
        raise NativeControlError('missing or stale native rank evidence')
    for field in ('all_queue', 'running', 'waiting', 'retained_kv_requests'):
        if not isinstance(value.get(field), list):
            raise NativeControlError('native inventory missing: ' + field)
        if drained and value[field] or request_id is not None and request_id in value[field]:
            raise NativeControlError('native request still owns ' + field)
    if (value.get('pending_transfers') != 0 or value.get('transfer_allocations') != {}
            or any(r.get('pending_transfers') != 0 or r.get('transfer_allocations') != {} for r in ranks)):
        raise NativeControlError('native transfers have not been released')
    allocations = value.get('kv_allocations')
    if not isinstance(allocations, dict):
        raise NativeControlError('native KV inventory missing')
    if drained:
        total, free, reserved = (value.get(k) for k in ('total_blocks', 'free_blocks', 'reserved_blocks'))
        if (allocations or any(type(x) is not int for x in (total, free, reserved))
                or total <= 0 or reserved < 0 or free < total - reserved):
            raise NativeControlError('native KV blocks not fully returned')
    elif request_id is not None and request_id in allocations:
        raise NativeControlError('cancelled request retains KV blocks')
    return value


class NativeControl:
    def __init__(self, specs: Mapping[str, object], *, timeout_s: float = 30.):
        self.specs = dict(specs)
        self.timeout_s = timeout_s

    async def _request(self, iid, method, endpoint, payload=None, timeout_s=None):
        timeout = aiohttp.ClientTimeout(total=(timeout_s or self.timeout_s) + 5.)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, self.specs[iid].base_url + '/baseline/' + endpoint,
                                       json=payload) as response:
                if response.status != 200:
                    raise NativeControlError(f'{iid} {endpoint}: {response.status} {(await response.text())[:300]}')
                value = await response.json()
                if not isinstance(value, dict):
                    raise NativeControlError('native control did not return an object')
                return value

    def _validate(self, iid, value, **kwargs):
        spec = self.specs[iid]
        return validate_state(value, generation=spec.generation, tp=spec.tp, pp=spec.pp, **kwargs)

    async def drain(self, iid, timeout_s=None):
        started = time.time()
        receipt = await self._request(iid, 'POST', 'drain',
            dict(timeout_s=timeout_s or self.timeout_s), timeout_s)
        self._validate(iid, receipt, drained=True, observed_after_s=started)
        if receipt.get('acknowledged') is not True or receipt.get('drained') is not True:
            raise NativeControlError('native drain ACK absent')
        return receipt

    async def resume(self, iid, role):
        started = time.time()
        receipt = await self._request(iid, 'POST', 'control',
            dict(generation=self.specs[iid].generation, accepting=True,
                 admit_prefill=True, admit_decode=True, role=role))
        if receipt.get('acknowledged') is not True or receipt.get('generation') != self.specs[iid].generation:
            raise NativeControlError('native resume ACK absent or stale')
        # Read the ranks after control, not a cached pre-control snapshot.
        state = await self._request(iid, 'GET', 'state')
        self._validate(iid, state, drained=True, observed_after_s=started)
        if state.get('accepting') is not True:
            raise NativeControlError('native resume did not open admission')
        return dict(control=receipt, state=state)

    async def cancel(self, record, engine_id, instance_ids):
        async def one(iid):
            started = time.time()
            receipt = await self._request(iid, 'POST', 'cancel',
                dict(request_id=engine_id, timeout_s=self.timeout_s))
            if (receipt.get('request_id') != engine_id or receipt.get('generation') != record.generation
                    or receipt.get('acknowledged') is not True or receipt.get('cancelled') is not True):
                raise NativeControlError('native cancel request/generation ACK absent')
            self._validate(iid, receipt.get('native_state'), request_id=engine_id, observed_after_s=started)
            return iid, dict(receipt, instance_id=iid)
        return dict(await asyncio.gather(*(one(iid) for iid in sorted(set(instance_ids)))))
