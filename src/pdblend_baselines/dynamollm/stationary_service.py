"""Opt-in private Dynamo source-KV RPC gateway; frozen native server is reused.

This gateway supplies real drain evidence itself. Caller-provided drain receipts
are forbidden. No target activation, missing-fragment transport, or TP change is
implemented by this source-only transaction.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import time

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import JSONResponse

from .native_state import validate_state
from .stationary_ipc import need


PREFIX = '/baseline/dynamollm/stationary/'
router = APIRouter()


class StationaryContext:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.phase = 'idle'
        self.public_inflight = 0
        self.transaction = None
        self.completed_ids = set()
        self.released_ranks = set()
        self.journal = []

    def record(self, event, **fields):
        row = dict(at_s=time.time(), event=event, phase=self.phase, **fields)
        self.journal.append(row)
        return row


class AdmissionFence:
    """Drain previously admitted HTTP operations before native quiescence.

    Counting the entire ASGI lifetime also covers tokenization and streaming:
    neither may submit a late request after a supposedly empty scheduler read.
    Other server processes/instances have their own context and remain active.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') in ('GET', 'HEAD', 'OPTIONS') or scope['path'].startswith(PREFIX):
            return await self.app(scope, receive, send)
        context = scope['app'].state.dynamo_stationary
        # No await separates this check and count increment.
        if context.phase != 'idle':
            return await JSONResponse(dict(error='Dynamo stationary transaction fences public mutations',
                phase=context.phase, transaction_id=context.transaction and context.transaction['transaction_id']),
                status_code=409)(scope, receive, send)
        context.public_inflight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            context.public_inflight -= 1


class RequestNative:
    def __init__(self, request):
        self.request = request
        from pdblend_runtime import serve
        self.serve = serve

    def identity(self):
        state = self.request.app.state
        return dict(identity=deepcopy(state.native_identity), tp=state.native_tp, pp=state.native_pp)

    async def drain(self):
        return await self.serve.drain_engine(self.request, dict(timeout_s=30))

    async def state(self):
        return await self.serve.state(self.request)

    async def control(self, payload):
        return await self.serve.scheduler(self.request, 'control', payload)

    async def ranks(self, method, **kwargs):
        return await self.serve.workers(self.request, method, **kwargs)


class StationaryCoordinator:
    def __init__(self, context, native, *, http_drain_timeout_s=30.):
        self.context, self.native = context, native
        self.http_drain_timeout_s = http_drain_timeout_s

    def _epoch(self, payload):
        current = self.native.identity()
        identity = current['identity']
        need(current['pp'] == 1 and current['tp'] in (1, 2, 4)
             and len(identity['gpu_uuids']) == current['tp'], 'private stationary native topology differs')
        need(type(payload.get('expected_generation')) is int and payload['expected_generation'] >= 0,
             'explicit native generation required')
        need(payload.get('expected_gpu_uuids') == identity['gpu_uuids'], 'actual source UUID order differs')
        need(all(identity.get(k) for k in ('model_id', 'model_hash', 'tokenizer_hash',
             'engine_revision', 'source_revision', 'image_digest')), 'native source identity missing')
        return current

    async def _drain(self, payload):
        identity = self._epoch(payload)
        started = time.time()
        receipt = await asyncio.wait_for(self.native.drain(), timeout=35.)
        validate_state(receipt, generation=payload['expected_generation'], tp=identity['tp'],
                       pp=1, drained=True, observed_after_s=max(started, time.time() - .5))
        need(receipt.get('acknowledged') is True and receipt.get('drained') is True
             and receipt.get('accepting') is False, 'actual source drain/admission ACK missing')
        self.context.record('native_drain', receipt=receipt)
        return receipt

    def _rank_ack(self, rows, payload, *, phase=None, selected=None):
        current = self._epoch(payload)
        need(len(rows) == current['tp'] and {r.get('rank') for r in rows} == set(range(current['tp'])),
             'private stationary ACK lacks exact all-rank inventory')
        for row in rows:
            rank = row['rank']
            need(row.get('generation') == payload['expected_generation']
                 and row.get('transaction_id') == payload['transaction_id'], 'private worker epoch/transaction differs')
            if selected is not None and rank != selected:
                need(row.get('participating') is False, 'nonselected worker unexpectedly mutated')
                continue
            uuid = row.get('gpu_uuid', row.get('original_storage_preserved', {}).get('gpu_uuid'))
            need(uuid == current['identity']['gpu_uuids'][rank], 'private worker physical UUID differs')
            if phase is not None:
                need(row.get('status') == phase and row.get('requires_process_isolation') is False,
                     'private worker KV phase differs or requires isolation')

    async def _workers(self, operation, payload):
        rows = await asyncio.wait_for(self.native.ranks('dynamo_stationary_operation',
            operation=operation, payload=payload), timeout=35.)
        self.context.record('worker_receipts', operation=operation, receipts=rows)
        return rows

    async def _quarantine(self, error):
        context = self.context
        context.phase = 'quarantined'
        receipt = dict(error=repr(error), inference_fenced=True, process_isolation_required=True,
                       native_admission_closed=False, worker_completion_known=False)
        try:
            ack = await asyncio.wait_for(self.native.control(dict(accepting=False, admit_prefill=True,
                admit_decode=True)), timeout=5.)
            state = await asyncio.wait_for(self.native.state(), timeout=5.)
            receipt.update(quiesce_ack=ack, native_state=state,
                native_admission_closed=ack.get('acknowledged') is True
                    and ack.get('accepting') is False and state.get('accepting') is False)
        except BaseException as failure:
            receipt['quiesce_error'] = repr(failure)
        context.record('quarantine', receipt=receipt)
        return receipt

    async def execute(self, operation, payload):
        context = self.context
        need(isinstance(payload, dict) and 'native_scheduler_drain' not in payload,
             'caller-supplied native drain evidence is forbidden')
        async with context.lock:
            if operation == 'status':
                return self.receipt()
            need(context.phase != 'quarantined', 'private source is quarantined; owned-process isolation required')
            identity = self._epoch(payload)
            if operation == 'describe':
                rows = await asyncio.wait_for(self.native.ranks('dynamo_stationary_operation', operation='describe',
                    payload=dict(expected_generation=payload['expected_generation'])), timeout=35.)
                need(len(rows) == identity['tp'] and {r.get('rank') for r in rows} == set(range(identity['tp']))
                     and all(r.get('generation') == payload['expected_generation']
                         and r.get('gpu_uuid') == identity['identity']['gpu_uuids'][r['rank']] for r in rows),
                     'native source description rank/epoch/UUID differs')
                return dict(ranks=rows, identity=identity, formal_eligible=False)
            tx = payload.get('transaction_id')
            need(isinstance(tx, str) and 0 < len(tx) <= 128, 'bounded private transaction ID required')
            if operation == 'pin':
                need(context.phase == 'idle' and tx not in context.completed_ids,
                     'one new private stationary transaction required')
                need(payload['plan']['source_gpu_uuids'] == identity['identity']['gpu_uuids'],
                     'source tensor plan UUID order differs')
                need(payload['plan'].get('target_gpu_uuids') == identity['identity']['gpu_uuids']
                     and payload['plan'].get('planned_transfer_bytes') == 0,
                     'this source-only service cannot prepare a target TP layout or weight transport')
            else:
                need(context.transaction is not None and all(payload.get(k) == context.transaction[k]
                    for k in ('transaction_id', 'expected_generation', 'expected_gpu_uuids')),
                    'private request does not match pinned transaction')
            try:
                if operation == 'pin':
                    context.transaction = {k: deepcopy(payload[k]) for k in
                        ('transaction_id', 'expected_generation', 'expected_gpu_uuids')}
                    context.released_ranks = set()
                    context.phase = 'quiescing'
                    deadline = time.monotonic() + self.http_drain_timeout_s
                    while context.public_inflight:
                        need(time.monotonic() < deadline, 'pre-existing HTTP requests did not drain')
                        await asyncio.sleep(.01)
                    drain = await self._drain(payload)
                    rows = await self._workers('pin', dict(payload, native_scheduler_drain=drain))
                    self._rank_ack(rows, payload)
                    need(all(r.get('plan_sha256') == payload['plan']['plan_sha256'] for r in rows),
                         'pinned native tensor plan differs')
                    context.phase = 'pinned'
                elif operation in ('release_kv', 'restore_kv', 'close_kv_workspace'):
                    before, after = {'release_kv': ('pinned', 'released'),
                        'restore_kv': ('released', 'restored'),
                        'close_kv_workspace': ('restored', 'closed')}[operation]
                    need(context.phase == before, 'source KV operation is out of order')
                    drain = await self._drain(payload)
                    rows = await self._workers(operation, dict(payload, native_scheduler_drain=drain))
                    self._rank_ack(rows, payload, phase=after)
                    context.phase = after
                elif operation == 'release':
                    need(context.phase == 'closed', 'source KV must be restored before releasing owner leases')
                    rank = payload.get('source_rank')
                    need(type(rank) is int and 0 <= rank < identity['tp'] and rank not in context.released_ranks,
                         'each actual owner rank must be released exactly once')
                    rows = await self._workers('release', payload)
                    self._rank_ack(rows, payload, selected=rank)
                    selected = next(r for r in rows if r['rank'] == rank)
                    need(selected.get('released') is True and selected.get('clean_consumer_release') is True
                         and selected.get('owner_exit_required') is False, 'source owner release is uncertain')
                    context.released_ranks.add(rank)
                    if len(context.released_ranks) == identity['tp']:
                        context.phase = 'owners_released'
                elif operation == 'abort_pin':
                    need(context.phase == 'pinned', 'only an unmodified pinned source can abort before KV detach')
                    await self._drain(payload)
                    for rank in range(identity['tp']):
                        rows = await self._workers('release', dict(payload, source_rank=rank,
                            consumer_processes_gone=[]))
                        self._rank_ack(rows, payload, selected=rank)
                        selected = next(r for r in rows if r['rank'] == rank)
                        need(selected.get('released') is True and selected.get('clean_consumer_release') is True
                             and selected.get('owner_exit_required') is False, 'pinned owner abort is uncertain')
                        context.released_ranks.add(rank)
                    context.phase = 'owners_released'
                elif operation == 'resume':
                    need(context.phase == 'owners_released', 'all source owners must cleanly release before resume')
                    await self._drain(payload)
                    generation = payload['expected_generation'] + 1
                    rows = await asyncio.wait_for(self.native.ranks('native_generation_set',
                        generation=generation), timeout=35.)
                    need(len(rows) == identity['tp'] and {r.get('rank') for r in rows} == set(range(identity['tp']))
                         and all(r.get('acknowledged') is True and r.get('generation') == generation for r in rows),
                         'native restored epoch lacks all-rank installation ACK')
                    ack = await asyncio.wait_for(self.native.control(dict(generation=generation, accepting=False,
                        admit_prefill=True, admit_decode=True)), timeout=5.)
                    need(ack.get('acknowledged') is True and ack.get('generation') == generation
                         and ack.get('accepting') is False, 'native scheduler epoch installation differs')
                    final = await self._drain(dict(payload, expected_generation=generation))
                    ack = await asyncio.wait_for(self.native.control(dict(accepting=True,
                        admit_prefill=True, admit_decode=True)), timeout=5.)
                    need(ack.get('acknowledged') is True and ack.get('accepting') is True
                         and ack.get('generation') == generation, 'native source resume ACK differs')
                    context.record('source_restored', drain=final, epoch_ranks=rows, resume=ack)
                    context.completed_ids.add(tx)
                    context.phase = 'idle'
                else:
                    raise ValueError('private source-only operation is not implemented')
                context.record('completed_operation', operation=operation, transaction_id=tx)
                return self.receipt()
            except BaseException as error:
                quarantine = await self._quarantine(error)
                # The fixed vLLM HTTP exception handler requires message:str.
                raise HTTPException(503, detail=json.dumps(dict(error=repr(error), quarantine=quarantine,
                    stationary=self.receipt()), allow_nan=False)) from error

    def receipt(self):
        c = self.context
        return dict(schema='dynamo-source-kv-service-transaction/v1', phase=c.phase,
            transaction=deepcopy(c.transaction), public_inflight=c.public_inflight,
            released_owner_ranks=sorted(c.released_ranks), journal=deepcopy(c.journal),
            source_only=True, target_engine_activated=False, missing_fragment_transport_qualified=False,
            full_tp_switch_qualified=False, original_dynamo_mechanism_qualified=False, formal_eligible=False)


@router.post(PREFIX + '{operation}')
async def stationary_operation(request: Request, operation: str):
    try:
        return await StationaryCoordinator(request.app.state.dynamo_stationary, RequestNative(request)).execute(
            operation, await request.json())
    except ValueError as error:
        raise HTTPException(409, detail=str(error)) from error


def main():
    import sys
    from vllm.entrypoints.openai import api_server
    from pdblend_runtime import serve
    extension = 'pdblend_baselines.dynamollm.stationary_kv_worker.DynamoStationaryKvWorkerExtension'
    argv = list(sys.argv[1:])
    if '--worker-extension-cls' in argv:
        need(argv[argv.index('--worker-extension-cls') + 1] == extension, 'private stationary extension differs')
    else:
        argv += ['--worker-extension-cls', extension]
    need('--enable-sleep-mode' not in argv and '--kv-transfer-config' not in argv,
         'stationary source gateway does not support sleep or KV connector')
    if '--enforce-eager' not in argv:
        argv += ['--enforce-eager']
    original = api_server.build_app
    def build_app(args):
        app = original(args)
        app.state.dynamo_stationary = StationaryContext()
        app.add_middleware(AdmissionFence)
        app.include_router(router)
        return app
    api_server.build_app = build_app
    sys.argv = [sys.argv[0], *argv]
    serve.main()


if __name__ == '__main__':
    main()
