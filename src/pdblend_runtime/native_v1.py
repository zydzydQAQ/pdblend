"""Owner-thread scheduler control and real CUDA measurements for vLLM 0.10.1.1.

No planner, predictor, baseline policy or fitted profile lives in this module.
The scheduler RPC is installed when vLLM imports the explicit scheduler class.
It executes on EngineCore's request thread, including while the engine is idle.
"""
from __future__ import annotations

import os
import time
from collections import deque

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu_worker import Worker


class NativeScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_generation = int(os.environ.get('DYNAMO_GENERATION', '0'))
        self.native_role = 'mixed'
        self.native_mode = 'temporal'
        self.native_accepting = os.environ.get('DYNAMO_DUMMY', '0') != '1'
        self.native_admit_prefill = self.native_accepting
        self.native_admit_decode = self.native_accepting
        self.native_events = deque(maxlen=100000)
        self.native_seq = 0
        self.native_last_schedule = []
        self.native_prefill_mode = False
        self.native_held = set()
        self.native_max_batch = self.max_num_running_reqs
        self.native_max_tokens = self.max_num_scheduled_tokens
        self._native_event('initialized')

    def _native_event(self, kind, **fields):
        self.native_seq += 1
        now = time.time()
        row = dict(seq=self.native_seq, kind=kind, timestamp=now, at_s=now,
                   generation=self.native_generation, block_size=self.block_size,
                   free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                   prefill_mode=self.native_prefill_mode, schedule_queue=list(self.native_last_schedule),
                   all_queue=list(self.requests))
        row.update(fields)
        self.native_events.append(row)
        return row

    def native_state(self):
        pool = self.kv_cache_manager.block_pool
        free = pool.get_num_free_blocks()
        total = self.cache_config.num_gpu_blocks
        allocations = {rid: [list(group) for group in self.kv_cache_manager.get_block_ids(rid)]
                       for rid in self.requests}
        return dict(role=self.native_role, mode=self.native_mode, accepting=self.native_accepting,
                    generation=self.native_generation, acknowledged_generation=self.native_generation,
                    running=[r.request_id for r in self.running], waiting=[r.request_id for r in self.waiting],
                    schedule_queue=list(self.native_last_schedule), all_queue=list(self.requests),
                    prefill_mode=self.native_prefill_mode, free_blocks=free, total_blocks=total,
                    free_kv_tokens=free*self.block_size, total_kv_tokens=(total-1)*self.block_size,
                    reserved_blocks=1,
                    max_num_seqs=self.native_max_batch, max_num_batched_tokens=self.native_max_tokens,
                    max_model_len=self.max_model_len,
                    block_size=self.block_size, kv_allocations=allocations,
                    retained_kv_requests=sorted(self.native_held),
                    tp=self.parallel_config.tensor_parallel_size, pp=self.parallel_config.pipeline_parallel_size,
                    native_at_s=time.time(), healthy=True, native_evidence_complete=True,
                    next_seq=self.native_seq, admit_prefill=self.native_admit_prefill,
                    admit_decode=self.native_admit_decode)

    def native_operation(self, operation, payload=None):
        payload = payload or {}
        if operation == 'state':
            return self.native_state()
        if operation == 'events':
            after = int(payload.get('after_seq', 0))
            first = self.native_events[0]['seq'] if self.native_events else self.native_seq+1
            return dict(events=[e for e in self.native_events if e['seq'] > after],
                        next_seq=self.native_seq, first_seq=first, gap=after < first-1)
        if operation == 'control':
            generation = int(payload.get('generation', self.native_generation))
            if generation < self.native_generation:
                raise ValueError('stale scheduler generation')
            role = payload.get('role', self.native_role)
            if role not in ('mixed', 'prefill', 'decode', 'M', 'P', 'D'):
                raise ValueError('invalid native role')
            batch = int(payload.get('max_batch_size', self.native_max_batch))
            tokens = int(payload.get('max_tokens_per_batch', self.native_max_tokens))
            if not 1 <= batch <= self.native_max_batch or not 1 <= tokens <= self.native_max_tokens:
                raise ValueError('native batch budget outside engine capacity')
            self.native_role = {'M':'mixed','P':'prefill','D':'decode'}.get(role, role)
            self.native_mode = str(payload.get('mode', self.native_mode))
            self.native_generation = generation
            self.native_accepting = bool(payload.get('accepting', self.native_accepting))
            self.native_admit_prefill = bool(payload.get('admit_prefill', self.native_admit_prefill))
            self.native_admit_decode = bool(payload.get('admit_decode', self.native_admit_decode))
            self.max_num_running_reqs, self.max_num_scheduled_tokens = batch, tokens
            event = self._native_event('control_ack', control=dict(payload))
            return dict(acknowledged=True, generation=generation, acknowledged_generation=generation,
                        applied=event, state=self.native_state())
        if operation == 'release':
            rid = payload['request_id']
            if rid not in self.native_held:
                raise ValueError('request has no retained source KV')
            request = self.requests.get(rid)
            if request is None or not request.is_finished():
                raise ValueError('cannot release active prefill request')
            self.native_held.remove(rid)
            self._free_blocks(request)
            event = self._native_event('kv_released', request_id=rid)
            return dict(acknowledged=True, released=True, request_id=rid,
                        generation=self.native_generation, event=event)
        raise ValueError(f'unsupported native scheduler operation: {operation}')

    def _connector_finished(self, request):
        if request.request_id.startswith('distserve-hold-') and self.connector is not None:
            # Only a completed one-token prefill can own a retained source.
            # Abort also reaches this hook, including before any GPU step;
            # retaining that request prevents cancel/drain from ever clearing.
            if (request.status == RequestStatus.FINISHED_LENGTH_CAPPED and
                    request.num_computed_tokens >= request.num_prompt_tokens):
                self.native_held.add(request.request_id)
                self._native_event('kv_retained', request_id=request.request_id)
                return True, dict(native_retained_request=request.request_id)
            self.native_held.discard(request.request_id)
        return super()._connector_finished(request)

    def schedule(self):
        # Temporarily remove held request classes; their KV blocks and request
        # identities stay resident. Restore queues even when scheduling fails.
        held_running = [r for r in self.running
                        if not (self.native_admit_prefill if r.num_computed_tokens < r.num_prompt_tokens
                                else self.native_admit_decode)]
        self.running = [r for r in self.running if r not in held_running]
        original_waiting = None
        if not self.native_admit_prefill and not (self.native_role == 'decode' and self.native_admit_decode):
            original_waiting, self.waiting = self.waiting, create_request_queue(self.policy)
        try:
            output = super().schedule()
        finally:
            self.running = held_running + self.running
            if original_waiting is not None:
                # No new engine inputs are processed during this owner-thread
                # call. Preserve any newly preempted waiting requests as well.
                for request in self.waiting:
                    original_waiting.add_request(request)
                self.waiting = original_waiting
        self.native_last_schedule = list(output.num_scheduled_tokens)
        self.native_prefill_mode = any(output.num_scheduled_tokens[rid] > 1 for rid in self.native_last_schedule)
        if self.native_last_schedule:
            self._native_event('schedule', schedule_queue=list(self.native_last_schedule),
                               all_queue=list(self.requests), prefill_mode=self.native_prefill_mode,
                               free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                               block_size=self.block_size,
                               num_batched_tokens=output.total_num_scheduled_tokens)
        return output

    def finish_requests(self, request_ids, finished_status):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        super().finish_requests(ids, finished_status)
        for rid in ids:
            self._native_event('request_terminated', request_id=rid, status=str(finished_status))

    def update_from_output(self, *args, **kwargs):
        output = super().update_from_output(*args, **kwargs)
        self.native_last_schedule = [rid for rid in self.native_last_schedule if rid in self.requests]
        self._native_event('step_completed')
        return output


def _engine_native_operation(self, operation, payload=None):
    return self.scheduler.native_operation(operation, payload)


# EngineCore is already defined when the engine resolves scheduler_cls.
from vllm.v1.engine.core import EngineCore
EngineCore.native_operation = _engine_native_operation


class NativeWorker(Worker):
    """Adds optional CUDA-event instrumentation without changing batch policy."""
    def load_model(self):
        super().load_model()
        self._native_scope = None
        self._native_system = None
        self._native_pending = []
        self._native_requests = {}
        self._native_current_shape = None
        self._native_forward_hooks = []
        self._native_generation = int(os.environ.get('DYNAMO_GENERATION', '0'))
        original = self.model_runner.execute_model

        def measured(scheduler_output, *args, **kwargs):
            shape = self._native_shape(scheduler_output)
            self._native_current_shape = shape
            pair = self._native_begin(shape) if self._native_scope == 'runner' and shape['batch'] else None
            try:
                output = original(scheduler_output, *args, **kwargs)
            except BaseException:
                if pair:
                    pair[0]['failed'] = True
                raise
            finally:
                if pair:
                    pair[2].record()
                    self._native_pending.append(pair)
                self._native_current_shape = None
            return output
        self.model_runner.execute_model = measured
        from vllm.distributed.kv_transfer import has_kv_transfer_group
        if has_kv_transfer_group():
            from .kv import install
            install(self)

    def native_kv_operation(self, operation, payload):
        from .kv import operation as kv_operation
        result = kv_operation(self, operation, payload)
        result.update(rank=self.rank, tp=self.parallel_config.tensor_parallel_size,
                      pp=self.parallel_config.pipeline_parallel_size,
                      generation=self._native_generation)
        return result

    def native_generation_set(self, generation):
        if type(generation) is not int or generation < self._native_generation:
            raise ValueError('stale native worker generation')
        import torch
        torch.cuda.synchronize()
        self._native_generation = generation
        return dict(rank=self.rank, generation=self._native_generation, acknowledged=True)

    def native_measurement_stop(self):
        for hook in self._native_forward_hooks:
            hook.remove()
        self._native_forward_hooks = []
        self._native_scope = self._native_system = None
        return dict(rank=self.rank, acknowledged=True)

    def _native_shape(self, output):
        for rid in output.finished_req_ids:
            self._native_requests.pop(rid, None)
        for req in output.scheduled_new_reqs:
            self._native_requests[req.req_id] = dict(prompt=len(req.prompt_token_ids), computed=req.num_computed_tokens)
        cached = output.scheduled_cached_reqs
        for rid, computed in zip(cached.req_ids, cached.num_computed_tokens):
            if rid in self._native_requests:
                self._native_requests[rid]['computed'] = computed
        contexts, phases, prompts = [], [], []
        for rid, scheduled in output.num_scheduled_tokens.items():
            row = self._native_requests.get(rid)
            if row is None:
                raise RuntimeError('native timing is missing worker request metadata')
            contexts.append(row['computed'] + scheduled)
            prompts.append(row['prompt'])
            phases.append('prefill' if row['computed'] < row['prompt'] else 'decode')
            row['computed'] += scheduled
        role = phases[0] if phases and len(set(phases)) == 1 else 'mixed'
        return dict(role=role, batch=len(phases), input_tokens=output.total_num_scheduled_tokens,
                    max_input_tokens=max(prompts, default=0),
                    context_tokens=max(contexts, default=0), max_context_tokens=max(contexts, default=0),
                    request_ids=list(output.num_scheduled_tokens))

    def _native_begin(self, shape):
        import torch
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        return (dict(shape, at_s=time.time(), measurement_scope=self._native_scope, system=self._native_system), start, end)

    def native_measurement_start(self, scope, system):
        if scope not in ('runner', 'forward') or system not in ('distserve', 'ecoserve', 'dynamollm', 'pdblend'):
            raise ValueError('unsupported measurement identity/scope')
        if scope == 'forward' and not self.model_config.enforce_eager:
            raise ValueError('forward measurement requires enforce_eager; CUDA graph capture is not a profile sample')
        for hook in self._native_forward_hooks:
            hook.remove()
        self._native_forward_hooks = []
        self._native_pending = []
        self._native_scope, self._native_system = scope, system
        if scope == 'forward':
            def before(module, args):
                shape = self._native_current_shape
                self._native_forward_pair = self._native_begin(shape) if shape and shape['batch'] else None
            def after(module, args, output):
                pair = self._native_forward_pair
                if pair:
                    pair[2].record()
                    self._native_pending.append(pair)
                self._native_forward_pair = None
            self._native_forward_hooks = [self.model_runner.model.register_forward_pre_hook(before),
                                          self.model_runner.model.register_forward_hook(after)]
        return dict(rank=self.rank, tp=self.parallel_config.tensor_parallel_size,
                    pp=self.parallel_config.pipeline_parallel_size, scope=scope, system=system, acknowledged=True)

    def native_measurement_samples(self):
        from vllm.distributed import get_tp_group, get_pp_group
        pending, self._native_pending = self._native_pending, []
        rows = []
        for row, start, end in pending:
            end.synchronize()
            row.update(gpu_elapsed_ms=start.elapsed_time(end), rank=self.rank,
                       tp_rank=get_tp_group().rank_in_group, pp_rank=get_pp_group().rank_in_group,
                       tp=self.parallel_config.tensor_parallel_size, pp=self.parallel_config.pipeline_parallel_size)
            rows.append(row)
        return dict(rank=self.rank, tp=self.parallel_config.tensor_parallel_size,
                    pp=self.parallel_config.pipeline_parallel_size, samples=rows)

    def native_worker_state(self):
        import torch
        from vllm.distributed.kv_transfer import has_kv_transfer_group, get_kv_transfer_group
        torch.cuda.synchronize()
        state = dict(rank=self.rank, generation=self._native_generation, at_s=time.time(), healthy=True, transfer_allocations={},
                     native_evidence_complete=True, pending_transfers=0,
                     retained_kv_supported=hasattr(self, '_pdblend_kv_connector'),
                     dynamo_weights_ready=getattr(self, '_dynamo_weights_ready', os.environ.get('DYNAMO_DUMMY', '0') != '1'))
        if has_kv_transfer_group():
            connector = get_kv_transfer_group()
            engine = getattr(connector, 'p2p_nccl_engine', None)
            if engine is None:
                state.update(native_evidence_complete=False, unsupported_connector=type(connector).__name__)
            else:
                with engine.recv_store_cv:
                    received = list(engine.recv_store)
                with engine.send_queue_cv:
                    queued = len(engine.send_queue)
                from .kv import operation
                retained = operation(self, 'state', {})
                state.update(transfer_allocations={name: 'received' for name in received},
                             pending_transfers=queued + retained['receiving_transactions'], retained=retained)
        return state
