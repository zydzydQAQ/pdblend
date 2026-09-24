import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS
import time

import pytest
pytest.importorskip('fastapi')
from fastapi import HTTPException

from pdblend_baselines.dynamollm.stationary_service import (
    AdmissionFence, PREFIX, StationaryContext, StationaryCoordinator)
from tests.independent_baselines.test_dynamo_stationary_kv import drained


class NativeOracle:
    """Protocol fault injector only; no hardware or native serving claim."""
    def __init__(self, tp=2):
        self.tp, self.generation, self.accepting = tp, 5, True
        self.calls, self.drains = [], []
        self.fault = None
    def identity(self):
        return dict(tp=self.tp, pp=1, identity=dict(gpu_uuids=[f'GPU-{i}' for i in range(self.tp)],
            model_id='Qwen2.5-7B-Instruct', model_hash='m', tokenizer_hash='t', engine_revision='vllm-test',
            source_revision='source', image_digest='image'))
    async def drain(self):
        self.accepting = False
        value = drained(self.tp)
        value['generation'] = self.generation
        for row in value['ranks']: row['generation'] = self.generation
        self.drains.append(value)
        return value
    async def state(self):
        value = drained(self.tp)
        value.update(generation=self.generation, accepting=self.accepting)
        return value
    async def control(self, payload):
        self.calls.append(('control', deepcopy(payload)))
        self.generation = payload.get('generation', self.generation)
        self.accepting = payload.get('accepting', self.accepting)
        return dict(acknowledged=True, generation=self.generation, accepting=self.accepting)
    async def ranks(self, method, **kwargs):
        self.calls.append((method, deepcopy(kwargs)))
        if method == 'native_generation_set':
            return [dict(rank=r, generation=kwargs['generation'], acknowledged=True) for r in range(self.tp)]
        op, p = kwargs['operation'], kwargs['payload']
        if self.fault == 'rpc' and op == 'release_kv': raise RuntimeError('rank1 RPC died after rank0 mutation')
        if op in ('pin', 'release_kv', 'restore_kv', 'close_kv_workspace'):
            assert any(p['native_scheduler_drain'] is d for d in self.drains)
        rows = []
        for rank in range(self.tp):
            row = dict(rank=rank, generation=self.generation, transaction_id=p.get('transaction_id'),
                gpu_uuid=f'GPU-{rank}', requires_process_isolation=False)
            if op == 'pin': row['plan_sha256'] = p['plan']['plan_sha256']
            elif op in ('release_kv', 'restore_kv', 'close_kv_workspace'):
                row['status'] = {'release_kv':'released', 'restore_kv':'restored', 'close_kv_workspace':'closed'}[op]
            elif op == 'release':
                if p['source_rank'] == rank:
                    row.update(released=True, clean_consumer_release=True, owner_exit_required=False)
                else: row['participating'] = False
            rows.append(row)
        if op == 'release_kv':
            if self.fault == 'uuid': rows[-1]['gpu_uuid'] = 'GPU-other'
            elif self.fault == 'epoch': rows[-1]['generation'] += 1
            elif self.fault == 'phase': rows[-1]['status'] = 'failed'
            elif self.fault == 'missing_rank': rows.pop()
            elif self.fault == 'isolation': rows[-1]['requires_process_isolation'] = True
        return rows


def payload():
    return dict(expected_generation=5, expected_gpu_uuids=['GPU-0', 'GPU-1'], transaction_id='tx',
                plan=dict(source_gpu_uuids=['GPU-0', 'GPU-1'], target_gpu_uuids=['GPU-0','GPU-1'],
                    planned_transfer_bytes=0, plan_sha256='plan'))


def test_actual_drain_is_injected_and_full_restore_installs_new_epoch_before_resume():
    async def run():
        context, native = StationaryContext(), NativeOracle()
        coordinator = StationaryCoordinator(context, native)
        p = payload()
        for op, phase in [('pin','pinned'), ('release_kv','released'), ('restore_kv','restored'),
                          ('close_kv_workspace','closed')]:
            receipt = await coordinator.execute(op, p)
            assert receipt['phase'] == phase and not receipt['full_tp_switch_qualified']
            assert not native.accepting
        for rank in range(2):
            await coordinator.execute('release', dict(p, source_rank=rank, consumer_processes_gone=[]))
        result = await coordinator.execute('resume', p)
        assert result['phase'] == 'idle' and native.accepting and native.generation == 6
        assert result['released_owner_ranks'] == [0,1]
        assert all(not r.get('formal_eligible', False) for r in [result])
        assert len(native.drains) == 6
        assert context.completed_ids == {'tx'}
    asyncio.run(run())


@pytest.mark.parametrize('fault', ['rpc', 'uuid', 'epoch', 'phase', 'missing_rank', 'isolation'])
def test_any_rank_failure_is_fenced_and_quarantined_without_fake_restore(fault):
    async def run():
        context, native = StationaryContext(), NativeOracle()
        coordinator = StationaryCoordinator(context, native)
        await coordinator.execute('pin', payload())
        native.fault = fault
        with pytest.raises(HTTPException) as caught: await coordinator.execute('release_kv', payload())
        assert caught.value.status_code == 503
        assert context.phase == 'quarantined' and not native.accepting
        q = json.loads(caught.value.detail)['quarantine']
        assert q['inference_fenced'] and q['process_isolation_required'] and q['native_admission_closed']
        assert q['worker_completion_known'] is False
        with pytest.raises(ValueError, match='quarantined'): await coordinator.execute('resume', payload())
        assert (await coordinator.execute('status', {}))['phase'] == 'quarantined'
    asyncio.run(run())


def test_synthetic_caller_drain_and_wrong_uuid_are_rejected_without_touching_engine():
    async def run():
        context, native = StationaryContext(), NativeOracle()
        coordinator = StationaryCoordinator(context, native)
        with pytest.raises(ValueError, match='caller-supplied'):
            await coordinator.execute('pin', dict(payload(), native_scheduler_drain=drained()))
        with pytest.raises(ValueError, match='UUID'):
            await coordinator.execute('pin', dict(payload(), expected_gpu_uuids=['GPU-x','GPU-1']))
        assert not native.calls and not native.drains and context.phase == 'idle'
    asyncio.run(run())


def test_fence_accounts_for_preexisting_streams_and_does_not_pause_another_instance():
    async def run():
        context, native = StationaryContext(), NativeOracle()
        other = StationaryContext()
        entered, finish = asyncio.Event(), asyncio.Event()
        async def app(scope, receive, send):
            entered.set()
            await finish.wait()
        fence = AdmissionFence(app)
        scope = dict(type='http', method='POST', path='/v1/completions',
                     app=NS(state=NS(dynamo_stationary=context)))
        async def receive(): return dict(type='http.request', body=b'')
        output=[]
        async def send(row): output.append(row)
        stream = asyncio.create_task(fence(scope, receive, send))
        await entered.wait()
        assert context.public_inflight == 1
        coordinator = StationaryCoordinator(context, native)
        task = asyncio.create_task(coordinator.execute('pin', payload()))
        await asyncio.sleep(.02)
        assert context.phase == 'quiescing' and not native.drains
        await fence(scope, receive, send)
        assert output[0]['status'] == 409
        assert context.public_inflight == 1
        # The other server context remains independent and can still admit.
        other_task = asyncio.create_task(fence(dict(scope,app=NS(state=NS(dynamo_stationary=other))),receive,send))
        await asyncio.sleep(0)
        assert other.public_inflight == 1 and other.phase == 'idle'
        finish.set()
        await asyncio.gather(stream, other_task)
        assert (await task)['phase'] == 'pinned'
        assert context.public_inflight == other.public_inflight == 0
    asyncio.run(run())


def test_http_drain_timeout_quarantines_without_pinning_any_weight():
    async def run():
        context, native = StationaryContext(), NativeOracle()
        context.public_inflight = 1
        coordinator = StationaryCoordinator(context, native, http_drain_timeout_s=.001)
        with pytest.raises(HTTPException): await coordinator.execute('pin', payload())
        assert context.phase == 'quarantined'
        assert not any(c[0] == 'dynamo_stationary_operation' for c in native.calls)
    asyncio.run(run())
