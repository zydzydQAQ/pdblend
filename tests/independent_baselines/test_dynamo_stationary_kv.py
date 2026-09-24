"""Real CPU tensor lifetime oracle; no CUDA KV/serving qualification claims."""
from dataclasses import dataclass
from types import SimpleNamespace as NS
import time
import weakref

import pytest
import torch

from pdblend_baselines.dynamollm.stationary_kv import SourceKvWorkspace, allocation_blocks
from pdblend_baselines.dynamollm.stationary_tensors import _tensor_identity


@dataclass
class FullAttentionSpec:
    block_size: int = 16


@dataclass
class CacheTensor:
    size: int
    shared_by: list


@dataclass
class CacheGroup:
    layer_names: list
    kv_cache_spec: FullAttentionSpec


@dataclass
class CacheConfig:
    num_blocks: int
    kv_cache_tensors: list
    kv_cache_groups: list


class Runner:
    def __init__(self):
        self.model = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.shared_kv_cache_layers = {}
        self.encoder_cache = {}
        self.kv_cache_config = CacheConfig(16, [CacheTensor(64, ['model.layers.0.self_attn'])],
            [CacheGroup(['model.layers.0.self_attn'], FullAttentionSpec())])
        self.compilation_config = NS(static_forward_context={'model.layers.0.self_attn': NS(kv_cache=[])})
        self.kv_caches = []
        self.calls = []
        self.fail_rebuild = False
        self.initialize_kv_cache_tensors(self.kv_cache_config)

    def initialize_kv_cache_tensors(self, config):
        assert not self.kv_caches
        self.calls.append('allocate_kv_only')
        cache = torch.zeros(64, dtype=torch.uint8)
        self.kv_caches.append(cache)
        self.compilation_config.static_forward_context['model.layers.0.self_attn'].kv_cache = [cache]
        if self.fail_rebuild:
            raise RuntimeError('partial KV allocation failed')
        return {'model.layers.0.self_attn': cache}

    def execute_model(self):
        self.calls.append('execute')
        return 42

    def profile_run(self):
        self.calls.append('profile')

    def capture_model(self):
        self.calls.append('graph')


class OracleBackend:
    is_cpu_oracle = True

    def __init__(self, runner):
        self.runner = runner
        self.original = weakref.ref(runner.kv_caches[0])
        self.original_ptr = runner.kv_caches[0].untyped_storage().data_ptr()
        self.uncertain = False
        self.free_bytes = 1024 ** 3

    def synchronize(self):
        pass

    def collect(self):
        import gc
        gc.collect()

    def observe(self):
        value = self.original()
        blocks = []
        if value is not None or self.uncertain:
            blocks.append(dict(address=self.original_ptr, size=64,
                state='active_allocated' if value is not None else 'active_awaiting_free'))
        return dict(gpu_uuid='GPU-oracle', free_bytes=self.free_bytes, total_bytes=2 * 1024 ** 3,
            blocks=blocks, cpu_oracle=True)


def setup():
    runner = Runner()
    worker = NS(model_runner=runner, _native_generation=5, rank=0,
        model_config=NS(enforce_eager=True, enable_sleep_mode=False, hf_config=NS(model_type='qwen2')),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        vllm_config=NS(kv_transfer_config=None, speculative_config=None, lora_config=None))
    weight_before = _tensor_identity(runner.model.weight)
    def pinned():
        assert _tensor_identity(runner.model.weight) == weight_before
        return dict(cpu_oracle=True)
    owner = NS(closed=False, quarantined=False, codec=NS(uuid='GPU-oracle', device=0),
        lease=NS(generation=5, source_rank=0, receipt=pinned),
        identity=dict(pid=123, start_ticks='oracle', boot_id='oracle'),
        plan=dict(source_gpu_uuids=['GPU-oracle'], plan_sha256='oracle-plan'),
        exports={}, alive=lambda p: p['alive'])
    backend = OracleBackend(runner)
    return worker, owner, backend


def drained(tp=1):
    return dict(generation=5, tp=tp, pp=1, native_evidence_complete=True, transport_healthy=True,
        native_at_s=time.time(), ranks=[dict(rank=r, generation=5, native_evidence_complete=True,
            healthy=True, pending_transfers=0, transfer_allocations={}) for r in range(tp)],
        all_queue=[], running=[], waiting=[], retained_kv_requests=[], pending_transfers=0,
        transfer_allocations={}, kv_allocations={}, total_blocks=16, free_blocks=16, reserved_blocks=0,
        accepting=False, acknowledged=True, drained=True)


def test_cpu_oracle_releases_real_tensor_owners_preserves_weights_and_rebuilds_only_kv():
    worker, owner, backend = setup()
    before = _tensor_identity(worker.model_runner.model.weight)
    workspace = SourceKvWorkspace.cpu_oracle(worker, owner, backend)
    released = workspace.release(drained())
    assert backend.original() is None
    assert released['status'] == 'released' and released['source_execution_blocked']
    assert not worker.model_runner.kv_caches
    assert not released['hardware_qualified'] and not released['target_engine_activated']
    assert released['driver_free_memory_credit_bytes'] == 0
    with pytest.raises(ValueError, match='execution blocked'): worker.model_runner.execute_model()
    with pytest.raises(ValueError, match='execution blocked'): worker.model_runner.profile_run()
    with pytest.raises(ValueError, match='execution blocked'): worker.model_runner.capture_model()
    with pytest.raises(ValueError, match='execution blocked'):
        worker.model_runner.model(torch.ones(1, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match='cannot drop guards'): workspace.close()
    restored = workspace.restore(drained())
    assert restored['status'] == 'restored' and not restored['source_admission_reopened']
    assert _tensor_identity(worker.model_runner.model.weight) == before
    assert worker.model_runner.calls == ['allocate_kv_only', 'allocate_kv_only']
    assert worker.model_runner.execute_model() == 42
    workspace.close()
    assert not workspace.receipt()['source_execution_blocked']
    assert worker.model_runner.execute_model() == 42
    assert not owner.closed


@pytest.mark.parametrize('damage', ['graphs', 'sleep', 'connector', 'speculation', 'shared_kv',
    'lora', 'encoder', 'transport', 'measurement', 'pending_measurement', 'owner_quarantine'])
def test_unsupported_owners_are_rejected_before_detach(damage):
    worker, owner, backend = setup()
    if damage == 'graphs': worker.model_config.enforce_eager = False
    elif damage == 'sleep': worker.model_config.enable_sleep_mode = True
    elif damage == 'connector': worker.vllm_config.kv_transfer_config = object()
    elif damage == 'speculation': worker.vllm_config.speculative_config = object()
    elif damage == 'shared_kv': worker.model_runner.shared_kv_cache_layers = {'a': 'b'}
    elif damage == 'lora': worker.vllm_config.lora_config = object()
    elif damage == 'encoder': worker.model_runner.encoder_cache = {'request': torch.zeros(1)}
    elif damage == 'transport': worker._dynamo_weight_sessions = {'tx': object()}
    elif damage == 'measurement': worker._native_scope = 'runner'
    elif damage == 'pending_measurement': worker._native_pending = [object()]
    else: owner.quarantined = True
    with pytest.raises(ValueError): SourceKvWorkspace.cpu_oracle(worker, owner, backend)
    assert backend.original() is worker.model_runner.kv_caches[0]
    assert worker.model_runner.execute_model() == 42


@pytest.mark.parametrize('damage', ['admission', 'stale', 'generation', 'rank_generation',
    'active_request', 'kv_blocks', 'partial_tp2', 'wrong_config', 'different_binding', 'unknown_allocator'])
def test_bad_drain_or_inventory_cannot_detach(damage):
    worker, owner, backend = setup()
    workspace = SourceKvWorkspace.cpu_oracle(worker, owner, backend)
    state = drained()
    if damage == 'admission': state['accepting'] = True
    elif damage == 'stale': state['native_at_s'] -= 1
    elif damage == 'generation': worker._native_generation += 1
    elif damage == 'rank_generation': state['ranks'][0]['generation'] += 1
    elif damage == 'active_request': state['all_queue'] = ['request']
    elif damage == 'kv_blocks': state['free_blocks'] = 2
    elif damage == 'partial_tp2': worker.parallel_config.tensor_parallel_size = 2; state['tp'] = 2
    elif damage == 'wrong_config': worker.model_runner.kv_cache_config.num_blocks += 1
    elif damage == 'different_binding':
        worker.model_runner.compilation_config.static_forward_context['model.layers.0.self_attn'].kv_cache = [torch.zeros(64)]
    elif damage == 'unknown_allocator': backend.observe = lambda: dict(blocks=[])
    with pytest.raises((ValueError, RuntimeError)): workspace.release(state)
    assert workspace.status == 'resident'
    assert backend.original() is not None


@pytest.mark.parametrize('damage', ['tensor_owner', 'storage_view_owner', 'allocator_pending'])
def test_uncertain_release_keeps_execution_blocked_and_requires_isolation(damage):
    worker, owner, backend = setup()
    workspace = SourceKvWorkspace.cpu_oracle(worker, owner, backend)
    extra = None
    if damage == 'tensor_owner': extra = worker.model_runner.kv_caches[0]
    elif damage == 'storage_view_owner':
        extra = worker.model_runner.kv_caches[0].view(-1)
        backend.uncertain = True  # CPU oracle models allocator's surviving storage record.
    else: backend.uncertain = True
    with pytest.raises(ValueError): workspace.release(drained())
    assert workspace.receipt()['requires_process_isolation']
    with pytest.raises(ValueError): workspace.restore(drained())
    with pytest.raises(ValueError): workspace.close()
    with pytest.raises(ValueError): worker.model_runner.execute_model()
    assert not owner.closed
    del extra


@pytest.mark.parametrize('damage', ['consumer_alive', 'unclean_consumer', 'budget', 'restore_oom', 'weights_changed'])
def test_restore_does_not_race_consumers_invent_budget_or_replace_weights(damage):
    worker, owner, backend = setup()
    workspace = SourceKvWorkspace.cpu_oracle(worker, owner, backend)
    workspace.release(drained())
    if damage in ('consumer_alive', 'unclean_consumer'):
        owner.exports = {'x': dict(release_ack={} if damage == 'consumer_alive' else None,
            packet=dict(consumer=dict(alive=damage == 'consumer_alive')))}
    elif damage == 'budget': backend.free_bytes = 64
    elif damage == 'restore_oom': worker.model_runner.fail_rebuild = True
    elif damage == 'weights_changed':
        with torch.no_grad(): worker.model_runner.model.weight.add_(1)
    with pytest.raises((ValueError, RuntimeError, AssertionError)): workspace.restore(drained())
    assert workspace.status in ('released', 'failed')
    assert workspace.receipt()['source_execution_blocked']
    assert worker.model_runner.calls.count('allocate_kv_only') == (2 if damage == 'restore_oom' else 1)
    assert not worker.model_runner.kv_caches


def test_allocator_parser_retains_reserved_segment_and_pending_block_distinctions():
    rows = allocation_blocks([dict(device=0, address=100, total_size=96, blocks=[
        dict(size=32, state='active_allocated'), dict(size=64, state='inactive')])], 0)
    assert rows[1] == dict(address=132, size=64, state='inactive', segment_address=100, segment_bytes=96)
    with pytest.raises(ValueError):
        allocation_blocks([dict(device=0, address=100, total_size=96,
            blocks=[dict(address=101, size=96, state='active_allocated')])], 0)


def test_opt_in_worker_extension_dispatches_real_workspace_and_blocks_premature_owner_release(monkeypatch):
    from pdblend_baselines.dynamollm.stationary_kv_worker import DynamoStationaryKvWorkerExtension
    worker, owner, backend = setup()
    extension = DynamoStationaryKvWorkerExtension()
    vars(extension).update(vars(worker))
    extension._dynamo_stationary_owners = {'tx': owner}
    monkeypatch.setattr(SourceKvWorkspace, 'for_native_worker',
        lambda w, o: SourceKvWorkspace.cpu_oracle(w, o, backend))
    payload = dict(transaction_id='tx', expected_generation=5, native_scheduler_drain=drained())
    result = extension.dynamo_stationary_operation('release_kv', payload)
    assert result['status'] == 'released' and result['rank'] == 0 and result['transaction_id'] == 'tx'
    with pytest.raises(ValueError, match='restore KV'):
        extension.dynamo_stationary_operation('release', payload)
    with pytest.raises(ValueError, match='generation'):
        extension.dynamo_stationary_operation('restore_kv', dict(payload, expected_generation=6))
    result = extension.dynamo_stationary_operation('restore_kv', dict(payload, native_scheduler_drain=drained()))
    assert result['status'] == 'restored'
    assert extension.dynamo_stationary_operation('close_kv_workspace', payload)['status'] == 'closed'
    assert not extension.dynamo_stationary_operation('kv_workspace_status', payload)['source_execution_blocked']
