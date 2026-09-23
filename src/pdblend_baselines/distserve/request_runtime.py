"""Independent PP1 DistServe queues bound to explicit native P/D services.

The author schedulers choose admission and bridge order. Native vLLM controls
its executed continuous batches: a Python admission cohort is not evidence of
an identical GPU batch. No latency/profile or predictor is shared with PDBlend.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import math
import time
from typing import Any, Protocol
import uuid

from .policy import DecodeScheduler, PrefillScheduler, Request


class DistServeCapabilityError(RuntimeError):
    pass


class DistServeTransport(Protocol):
    prefill_address: str
    decode_address: str
    async def capability(self, role): ...
    async def state(self, role): ...
    async def prefill(self, payload): ...
    async def expect_load(self, payload): ...
    async def transfer(self, payload): ...
    def generate_decode(self, payload): ...
    async def load_ack(self, payload): ...
    async def release(self, payload): ...
    async def cancel(self, role, payload): ...


@dataclass
class DistServeResult:
    request_id: str
    status: str
    tokens: int = 0
    receipts: list[dict[str, Any]] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    cleanup_errors: list[str] = field(default_factory=list)
    formal_eligible: bool = False
    energy_comparable: bool = False


def _ack(receipt, name):
    if not isinstance(receipt, dict) or receipt.get('acknowledged') is not True:
        raise DistServeCapabilityError('missing native '+name+' acknowledgement')
    return receipt


_DONE = object()


class DistServeRuntime:
    """Asynchronous retained-KV pipeline on one symmetric fixed-TP P/D pair."""
    def __init__(self, transport: DistServeTransport, *, tp=1, pp=1, max_batch_size=8,
                 max_tokens_per_batch=8192, num_gpu_blocks=4096, block_size=16,
                 request_timeout_s=180., poll_s=.01, journal=None):
        if type(tp) is not int or tp < 1 or pp != 1:
            raise DistServeCapabilityError('native DistServe runtime requires symmetric TP and PP1')
        if any(not math.isfinite(value) or value <= 0 for value in (request_timeout_s, poll_s)):
            raise ValueError('positive finite runtime timeouts required')
        self.transport, self.tp, self.pp, self.block_size = transport, tp, pp, block_size
        limits = dict(max_batch_size=max_batch_size, max_tokens_per_batch=max_tokens_per_batch,
                      num_gpu_blocks=num_gpu_blocks, block_size=block_size)
        self.prefill, self.decode = PrefillScheduler(**limits), DecodeScheduler(**limits)
        self.request_timeout, self.poll, self.journal = request_timeout_s, poll_s, journal
        self.requests, self.results, self.contexts, self.queues = {}, {}, {}, {}
        self.prefill_tasks, self.decode_tasks = {}, {}
        self.quarantined, self.failure, self.ready, self.closed = set(), None, False, False
        self.capabilities, self.generations, self.states = {}, {}, {}
        self.runner, self.step_lock, self.start_lock = None, asyncio.Lock(), asyncio.Lock()

    async def emit(self, event, **fields):
        if self.journal:
            value = self.journal(event, at_s=time.time(), **fields)
            if inspect.isawaitable(value): await value

    def _state(self, role, state, *, initial=False):
        required = ('all_queue', 'kv_allocations', 'pending_transfers', 'transfer_allocations',
                    'free_kv_tokens', 'total_kv_tokens', 'block_size', 'generation',
                    'acknowledged_generation', 'native_at_s', 'max_num_seqs',
                    'max_num_batched_tokens', 'max_model_len')
        if (not isinstance(state, dict) or any(k not in state for k in required)
                or state.get('native_evidence_complete') is not True
                or state.get('transport_healthy') is not True or state.get('error') or state.get('runtime_error')):
            raise DistServeCapabilityError(role+' native capacity/health evidence incomplete')
        stamp = state['native_at_s']
        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                or not -.05 <= time.time()-stamp <= 2):
            raise DistServeCapabilityError(role+' native state is stale')
        if (type(state['generation']) is not int or state['generation'] < 0
                or state['generation'] != state['acknowledged_generation']
                or not initial and state['generation'] != self.generations[role]):
            raise DistServeCapabilityError(role+' generation differs from verified deployment')
        ranks = state.get('ranks', [])
        if (len(ranks) != self.tp or {row.get('rank') for row in ranks} != set(range(self.tp))
                or any(row.get('generation') != state['generation'] or row.get('native_evidence_complete') is not True
                       or row.get('healthy') is not True for row in ranks)):
            raise DistServeCapabilityError(role+' state lacks all native rank health/generation ACKs')
        if (state['block_size'] != self.block_size or any(type(state[key]) is not int or state[key] <= 0
                for key in ('total_kv_tokens', 'max_num_seqs', 'max_num_batched_tokens', 'max_model_len'))
                or type(state['free_kv_tokens']) is not int or not 0 <= state['free_kv_tokens'] <= state['total_kv_tokens']):
            raise DistServeCapabilityError(role+' actual KV/batch capacity invalid')
        return state

    @staticmethod
    def _empty(state):
        return (not state['all_queue'] and not state['kv_allocations'] and not state['pending_transfers']
                and not state['transfer_allocations'] and state['free_kv_tokens'] == state['total_kv_tokens'])

    async def _observe_states(self):
        async def observe(role):
            # Validate each response on arrival. A slower peer does not make
            # an already fresh reply retroactively stale; observations retain
            # their original owner timestamps and are not atomic fleet state.
            return role, self._state(role, await self.transport.state(role))
        return dict(await asyncio.gather(*(observe(role) for role in ('P', 'D'))))

    async def start(self):
        async with self.start_lock:
            if self.ready: return self.capabilities
            if self.closed: raise RuntimeError('DistServe runtime is closed')
            for name in ('prefill_address', 'decode_address'):
                address = getattr(self.transport, name, None)
                if not isinstance(address, str) or ':' not in address:
                    raise DistServeCapabilityError('explicit P/D KV addresses required: '+name)
            if self.transport.prefill_address == self.transport.decode_address:
                raise DistServeCapabilityError('P/D KV addresses must identify distinct deployments')
            caps, states = {}, {}
            for role, sched in (('P', self.prefill), ('D', self.decode)):
                cap = await self.transport.capability(role)
                if (cap.get('supported') is not True or cap.get('tp') != self.tp or cap.get('pp') != 1
                        or cap.get('engine_revision') != 'vllm-0.10.1.1'):
                    raise DistServeCapabilityError(role+' model/topology/native capability mismatch')
                state = self._state(role, await self.transport.state(role), initial=True)
                if not self._empty(state) or state.get('accepting') is not True:
                    raise DistServeCapabilityError(role+' startup requires an idle accepting deployment')
                caps[role], states[role] = cap, state
                sched.num_gpu_blocks = min(sched.num_gpu_blocks, state['total_kv_tokens']//self.block_size)
                sched.max_batch_size = min(sched.max_batch_size, state['max_num_seqs'])
                sched.max_tokens_per_batch = min(sched.max_tokens_per_batch, state['max_num_batched_tokens'])
            for key in ('model_id', 'model_hash', 'tokenizer_hash', 'image_digest', 'source_revision'):
                if not caps['P'].get(key) or caps['P'][key] != caps['D'].get(key):
                    raise DistServeCapabilityError('P/D execution identity differs: '+key)
            for role, cap in caps.items():
                uuids = cap.get('gpu_uuids', [])
                if len(uuids) != self.tp or len(set(uuids)) != self.tp:
                    raise DistServeCapabilityError(role+' exact GPU UUID topology missing')
            if set(caps['P']['gpu_uuids']) & set(caps['D']['gpu_uuids']):
                raise DistServeCapabilityError('P/D deployments overlap physical GPU UUIDs')
            if states['P']['generation'] != states['D']['generation']:
                raise DistServeCapabilityError('symmetric P/D generation mismatch')
            self.capabilities, self.states = caps, states
            self.generations = {role: value['generation'] for role, value in states.items()}
            self.ready = True
            await self.emit('distserve_runtime_started', capabilities=caps, states=states,
                gpu_batch_equivalence_qualified=False, formal_eligible=False, energy_comparable=False)
            self.runner = asyncio.create_task(self._serve())
            return caps

    async def submit(self, request_id, input_tokens, output_tokens, payload=None):
        if not self.ready or self.closed or self.quarantined:
            raise DistServeCapabilityError('DistServe runtime is unavailable or quarantined')
        if not isinstance(request_id, str) or not request_id or '___' in request_id or '#' in request_id:
            raise ValueError('nonempty request ID without reserved KV protocol markers required')
        if request_id in self.requests: raise ValueError('duplicate request identifier')
        payload = dict(payload or {})
        prompt = payload.get('prompt_token_ids', payload.get('prompt'))
        if (not isinstance(prompt, list) or len(prompt) != input_tokens
                or any(type(token) is not int or token < 0 for token in prompt)):
            raise ValueError('explicit model-tokenized prompt matching input_tokens required')
        request = Request(request_id, input_tokens, output_tokens, payload)
        if input_tokens+output_tokens > min(state['max_model_len'] for state in self.states.values()):
            raise ValueError('request exceeds deployment context capacity')
        if (input_tokens > self.prefill.max_tokens_per_batch or
                request.blocks(self.block_size, prompt_only=True) > min(self.prefill.num_gpu_blocks, self.decode.num_gpu_blocks)):
            raise DistServeCapabilityError('request cannot fit author admission constraints')
        if payload.get('ignore_eos', True) is not True or payload.get('temperature', 0) != 0:
            raise ValueError('this fixed-work carry adapter requires greedy ignore_eos requests')
        source_id = ('distserve-hold-'+request_id+'___prefill_addr_'+self.transport.prefill_address+
                     '___decode_addr_127.0.0.1:0_'+request_id)
        self.requests[request_id] = request
        self.contexts[request_id] = dict(source_id=source_id, target_id=None, transaction_id=uuid.uuid4().hex,
            receipts=[], events=[], token_ids=[], started_s=time.monotonic(), source_submitted=False,
            released=False, load_registered=False, cleanup_done=False)
        self.queues[request_id] = asyncio.Queue()
        self.prefill.add(request)
        await self.emit('distserve_request_queued', request_id=request_id, input_tokens=input_tokens,
                        output_tokens=output_tokens, payload=payload)

    def _ranks(self, receipt, name, *, role, identity=None):
        _ack(receipt, name)
        rows = receipt.get('ranks', [])
        if (len(rows) != self.tp or {row.get('rank') for row in rows} != set(range(self.tp))
                or any(row.get('generation') != self.generations[role] for row in rows)):
            raise DistServeCapabilityError(name+' lacks every rank/generation ACK')
        if identity and any(any(row.get(key) != value for key, value in identity.items()) for row in rows):
            raise DistServeCapabilityError(name+' transaction/request identity differs')
        return rows

    async def _receipt(self, request, step, value):
        self.contexts[request.request_id]['receipts'].append(dict(step=step, receipt=value))
        await self.emit('distserve_native_receipt', request_id=request.request_id, step=step, receipt=value)

    async def _output(self, request, row, *, first=False):
        context = self.contexts[request.request_id]
        tokens = row.get('token_ids')
        if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
            raise DistServeCapabilityError('native SSE lacks authoritative token IDs')
        before = len(context['token_ids'])
        expected_index = before+len(tokens)-(0 if first else 1)
        if row.get('token_index') != expected_index:
            raise DistServeCapabilityError('native SSE output index is discontinuous')
        context['token_ids'].extend(tokens)
        request.generated = len(context['token_ids'])
        if request.generated > request.output_tokens:
            raise DistServeCapabilityError('native output exceeds fixed work')
        event = dict(row, request_id=request.request_id, native_request_id=row.get('request_id'),
                     token_index=request.generated, received_s=time.time())
        if first and request.output_tokens > 1:
            event.update(finished=False, finish_reason=None,
                         choices=[dict(choice, finish_reason=None) for choice in row.get('choices', [])])
        event['usage'] = dict(prompt_tokens=request.input_tokens, completion_tokens=request.generated)
        context['events'].append(event); self.queues[request.request_id].put_nowait(event)
        await self.emit('distserve_client_sse', request_id=request.request_id, payload=event)

    async def _prefill(self, request):
        context = self.contexts[request.request_id]
        context['source_submitted'] = True
        response = await self.transport.prefill(dict(request_id=request.request_id,
            prompt_token_ids=request.payload.get('prompt_token_ids', request.payload.get('prompt')),
            seed=request.payload.get('seed', 701), ignore_eos=True))
        await self._receipt(request, 'prefill', response)
        ranks = self._ranks(response, 'prefill retain', role='P')
        if (response.get('retained_handle') != context['source_id']
                or response.get('source_address') != self.transport.prefill_address
                or any(context['source_id'] not in row.get('held_requests', []) for row in ranks)):
            raise DistServeCapabilityError('prefill retained identity or source address differs')
        outputs = response.get('outputs', [])
        if (len(outputs) != 1 or len(outputs[0].get('token_ids', [])) != 1
                or outputs[0].get('finished') is not True):
            raise DistServeCapabilityError('prefill must return exactly one real terminal first token')
        request.kv_handle = context['source_id']
        context['first_output'] = outputs[0]
        if request.output_tokens > 1:
            await self._output(request, outputs[0], first=True)
        return response

    async def _release(self, request):
        context = self.contexts[request.request_id]
        receipt = await self.transport.release(dict(held_request_id=context['source_id']))
        await self._receipt(request, 'release', receipt)
        ranks = self._ranks(receipt, 'release', role='P')
        if receipt.get('released') is not True or any(row.get('released') is not True for row in ranks):
            raise DistServeCapabilityError('source KV release was not acknowledged')
        context['released'] = True

    async def _decode(self, request):
        context = self.contexts[request.request_id]
        if request.output_tokens == 1:
            await self._release(request)
            await self._output(request, context['first_output'], first=True)
            return
        tx = context['transaction_id']
        target_id = ('distserve-runtime-'+tx+'___prefill_addr_'+self.transport.prefill_address+
                     '___decode_addr_'+self.transport.decode_address+'_decode')
        context['target_id'] = target_id
        identity = dict(target_request_id=target_id, transaction_id=tx, generation=self.generations['D'])
        context['load_registered'] = True
        receipt = await self.transport.expect_load(dict(identity, source_tokens=request.input_tokens))
        await self._receipt(request, 'expect_load', receipt)
        rows = self._ranks(receipt, 'expect_load', role='D', identity=identity)
        if any(row.get('acknowledged') is not True for row in rows):
            raise DistServeCapabilityError('receive registration missing rank ACK')
        receipt = await self.transport.transfer(dict(identity, held_request_id=context['source_id'],
            target_address=self.transport.decode_address, target_tp=self.tp))
        await self._receipt(request, 'transfer', receipt)
        rows = self._ranks(receipt, 'transfer', role='P', identity=identity)
        if any(row.get('acknowledged') is not True or row.get('layers', 0) <= 0 for row in rows):
            raise DistServeCapabilityError('source layer transfer ACK missing')
        prompt = request.payload.get('prompt_token_ids', request.payload.get('prompt'))
        finished, terminal = False, None
        async for row in self.transport.generate_decode(dict(request_id=target_id,
                prompt=[*prompt, context['token_ids'][0]], max_tokens=request.output_tokens-1,
                seed=request.payload.get('seed', 701), ignore_eos=True, temperature=0)):
            if finished: raise DistServeCapabilityError('native SSE contains output after terminal')
            await self.emit('distserve_native_sse', request_id=request.request_id, payload=row)
            finished = bool(row.get('finished'))
            if finished: terminal = row
            else: await self._output(request, row)
        if not finished or request.generated+len(terminal.get('token_ids', [])) != request.output_tokens:
            raise DistServeCapabilityError('native decode output is truncated')
        receipt = await self.transport.load_ack(identity)
        await self._receipt(request, 'load_ack', receipt)
        rows = self._ranks(receipt, 'load', role='D', identity=identity)
        if (receipt.get('generation') != identity['generation'] or receipt.get('transaction_id') != tx
                or receipt.get('request_id') != target_id or any(row.get('acknowledged') is not True
                    or row.get('expected_layers', 0) <= 0 or row.get('loaded_layers') != row.get('expected_layers') for row in rows)):
            raise DistServeCapabilityError('complete layer/request load ACK missing')
        await self._release(request)
        # Expose a successful terminal packet only after actual rank/layer
        # load and source-release receipts, retaining native SSE separately.
        await self._output(request, terminal)

    async def _cleanup(self, request):
        context = self.contexts[request.request_id]
        if context['cleanup_done']: return []
        failures = []
        # Target first: uncertain receives remain quarantined by the native
        # service instead of being relabelled as released source KV.
        for role, rid in (('D', context['target_id']),
                          ('P', context['source_id'] if context['source_submitted'] and not context['released'] else None)):
            if rid is None: continue
            try:
                receipt = _ack(await self.transport.cancel(role, dict(request_id=rid)), role+' cancel')
                if receipt.get('generation') != self.generations[role]:
                    raise DistServeCapabilityError(role+' cancel generation mismatch')
                await self._receipt(request, 'cancel_'+role, receipt)
            except Exception as exc:
                failures.append(role+': '+repr(exc)); self.quarantined.add(role)
        context['cleanup_done'] = not failures
        return failures

    async def _finish(self, request, status, *, error=None, cleanup_errors=()):
        rid = request.request_id
        if rid in self.results: return self.results[rid]
        context = self.contexts[rid]
        self.prefill.processing.pop(rid, None); self.prefill.release(rid); self.decode.finish(rid)
        result = DistServeResult(rid, status, request.generated, list(context['receipts']),
                                list(context['token_ids']), list(context['events']), error, list(cleanup_errors))
        self.results[rid] = result
        self.queues[rid].put_nowait(_DONE)
        await self.emit('distserve_request_finished', request_id=rid, status=status, error=error,
                        cleanup_errors=list(cleanup_errors), generated=request.generated)
        return result

    async def step(self):
        if not self.ready: raise DistServeCapabilityError('runtime is not started')
        async with self.step_lock:
            completed = []
            for tasks, phase in ((self.prefill_tasks, 'prefill'), (self.decode_tasks, 'decode')):
                for rid, task in list(tasks.items()):
                    if not task.done(): continue
                    tasks.pop(rid); request = self.requests[rid]
                    if rid in self.results: continue
                    try:
                        task.result()
                        if phase == 'prefill':
                            self.prefill.complete((request,)); self.decode.add_bridge(request)
                            await self.emit('distserve_bridge_ready', request_id=rid, retained_handle=request.kv_handle)
                        else:
                            completed.append(await self._finish(request, 'completed'))
                    except (Exception, asyncio.CancelledError) as exc:
                        self.quarantined.update(('P', 'D'))
                        errors = await self._cleanup(request)
                        completed.append(await self._finish(request, 'failed', error=repr(exc), cleanup_errors=errors))
            for rid, context in self.contexts.items():
                if rid not in self.results and time.monotonic()-context['started_s'] > self.request_timeout:
                    await self._cancel(rid, status='failed', error='request deadline exceeded')
            if self.quarantined: return tuple(completed)
            states = await self._observe_states()
            self.states = states
            batch = self.prefill.next_batch(free_gpu_blocks=states['P']['free_kv_tokens']//self.block_size)
            if batch:
                await self.emit('distserve_prefill_admission', request_ids=[r.request_id for r in batch],
                    free_kv_tokens=states['P']['free_kv_tokens'], gpu_batch_equivalence_qualified=False)
            for request in batch:
                self.prefill_tasks[request.request_id] = asyncio.create_task(self._prefill(request))
            available = states['D']['free_kv_tokens']//self.block_size
            while (request := self.decode.accept_next(free_gpu_blocks=available)) is not None:
                available -= request.blocks(self.block_size, prompt_only=True)
                await self.emit('distserve_decode_selected', request_id=request.request_id, generation=self.generations['D'])
            for request in self.decode.next_batch():
                if request.request_id not in self.decode_tasks:
                    self.decode_tasks[request.request_id] = asyncio.create_task(self._decode(request))
            return tuple(completed)

    async def _serve(self):
        try:
            while not self.closed:
                await self.step(); await asyncio.sleep(self.poll)
        except asyncio.CancelledError: raise
        except Exception as exc:
            self.failure = repr(exc); self.quarantined.update(('P', 'D'))
            await self.emit('distserve_runtime_failure', error=self.failure)
            async with self.step_lock:
                for rid in list(self.requests):
                    if rid not in self.results: await self._cancel(rid, status='failed', error=self.failure)

    async def _cancel(self, request_id, *, status='cancelled', error=None):
        request = self.requests[request_id]
        if request_id in self.results: return self.results[request_id]
        request.cancelled = True; self.prefill.cancel(request_id)
        tasks = [table.pop(request_id) for table in (self.prefill_tasks, self.decode_tasks) if request_id in table]
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        errors = await self._cleanup(request)
        if errors: status = 'quarantined'
        return await self._finish(request, status, error=error, cleanup_errors=errors)

    async def cancel(self, request_id):
        async with self.step_lock:
            result = await self._cancel(request_id)
        return dict(acknowledged=not result.cleanup_errors, request_id=request_id,
                    status=result.status, receipts=result.receipts, cleanup_errors=result.cleanup_errors)

    async def handle(self, payload, request_id):
        if not self.ready: await self.start()
        prompt = payload.get('prompt_token_ids', payload.get('prompt'))
        await self.submit(request_id, len(prompt or ()), payload['max_tokens'], payload)
        complete = False
        try:
            while True:
                row = await self.queues[request_id].get()
                if row is _DONE:
                    result = self.results[request_id]
                    if result.status != 'completed': raise DistServeCapabilityError(result.error or result.status)
                    complete = True; break
                yield row
        finally:
            if not complete and request_id not in self.results: await self.cancel(request_id)

    async def recover(self):
        async with self.step_lock:
            if self.closed or not self.ready:
                raise DistServeCapabilityError('recovery requires an initialized open runtime')
            if any(rid not in self.results for rid in self.requests) or self.prefill_tasks or self.decode_tasks:
                raise DistServeCapabilityError('finish/cancel owned requests before recovery')
            values = await self._observe_states()
            for role, value in values.items():
                if not self._empty(value) or value.get('accepting') is not True:
                    raise DistServeCapabilityError(role+' recovery lacks actual empty accepting native state')
            self.quarantined.clear(); self.failure = None; self.states = values
            if self.runner is None or self.runner.done(): self.runner = asyncio.create_task(self._serve())
            await self.emit('distserve_recovered', states=values, generation=dict(self.generations))
            return dict(acknowledged=True, states=values)

    async def close(self):
        if self.closed: return
        self.closed = True
        if self.runner:
            self.runner.cancel(); await asyncio.gather(self.runner, return_exceptions=True)
        for rid in list(self.requests):
            if rid not in self.results: await self.cancel(rid)
        values = await self._observe_states()
        for role, value in values.items():
            if not self._empty(value):
                self.quarantined.add(role)
                raise DistServeCapabilityError(role+' close lacks actual released requests/KV')
        self.ready = False
        await self.emit('distserve_runtime_closed', states=values, quarantined=sorted(self.quarantined))
