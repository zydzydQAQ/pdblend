"""Actual BF16 worker transfer operations with only NCCL/CUDA replaced.

All rank workers execute in separate CPU threads. Point-to-point messages copy
real Torch tensors and wait for matching peer receives, exercising the worker's
real parameter traversal, transfer order, slice copies, byte counts and cleanup.
"""
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from pdblend_baselines.dynamollm.gpu_weights import DynamoWorkerExtension


GEOMETRY = dict(hidden_size=8, num_attention_heads=4, num_key_value_heads=4, intermediate_size=16)
SHAPES = {
    'model.embed_tokens.weight': (32, 8),
    'model.layers.0.self_attn.qkv_proj.weight': (24, 8),
    'model.layers.0.self_attn.qkv_proj.bias': (24,),
    'model.layers.0.self_attn.o_proj.weight': (8, 8),
    'model.layers.0.mlp.gate_up_proj.weight': (32, 8),
    'model.layers.0.mlp.down_proj.weight': (8, 16),
    'model.layers.0.input_layernorm.weight': (8,),
    'model.layers.0.post_attention_layernorm.weight': (8,),
    'model.norm.weight': (8,),
    'lm_head.weight': (32, 8),
}


def expected_shard(name, full, tp, rank):
    # Independent model-construction rules; do not call transfer_pieces/_segments.
    if name.endswith(('input_layernorm.weight', 'post_attention_layernorm.weight', 'norm.weight')):
        return full.clone()
    if '.qkv_proj.' in name:
        return torch.cat([part.chunk(tp, dim=0)[rank] for part in full.split(8, dim=0)], dim=0).clone()
    if '.gate_up_proj.' in name:
        return torch.cat([part.chunk(tp, dim=0)[rank] for part in full.split(16, dim=0)], dim=0).clone()
    axis = 1 if name.endswith(('o_proj.weight', 'down_proj.weight')) else 0
    return full.chunk(tp, dim=axis)[rank].clone()


def worker(tp, rank, *, target=False):
    model = torch.nn.Module()
    expected = {}
    for index, (name, shape) in enumerate(SHAPES.items()):
        full = (torch.arange(torch.Size(shape).numel()).reshape(shape)+index*512).to(torch.bfloat16)
        value = expected_shard(name, full, tp, rank)
        expected[name] = value.clone()
        if target:
            value.fill_(-1)
        parent = model
        *parts, leaf = name.split('.')
        for part in parts:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent._modules[part]
        parent.register_parameter(leaf, torch.nn.Parameter(value))
    value = DynamoWorkerExtension()
    value.rank, value._native_generation = rank, 1 if target else 0
    value.parallel_config = SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1, data_parallel_size=1)
    value.model_runner = SimpleNamespace(model=model)
    value.model_config = SimpleNamespace(model='/models/Qwen2.5-7B-Instruct', hf_config=SimpleNamespace(**GEOMETRY))
    return value, expected


@pytest.fixture
def wire(monkeypatch):
    import torch.distributed as dist
    channels, groups, barrier = {}, {}, None
    lock = threading.Lock()
    def channel(source, target):
        with lock:
            return channels.setdefault((source, target), queue.Queue())
    class Work:
        def __init__(self, action=lambda: None): self.action = action
        def wait(self): return self.action()
    class Group:
        def __init__(self, store, rank, world, timeout):
            nonlocal barrier
            assert timeout.total_seconds() == 60
            self.rank, self.closed, self.fail_close = rank, False, False
            groups[rank] = self
            if barrier is None:
                barrier = threading.Barrier(world)
        def allreduce(self, tensors):
            assert tensors[0].numel() == 1
            return Work(lambda: barrier.wait(10))
        def send(self, tensors, peer, tag):
            assert tag == 0 and len(tensors) == 1 and tensors[0].is_contiguous()
            consumed = threading.Event()
            channel(self.rank, peer).put((tensors[0].clone(), consumed))
            def wait():
                assert consumed.wait(10), 'peer never received actual tensor payload'
            return Work(wait)
        def recv(self, tensors, peer, tag):
            assert tag == 0 and len(tensors) == 1
            def copy():
                tensor, consumed = channel(peer, self.rank).get(timeout=10)
                assert tensor.shape == tensors[0].shape and tensor.dtype == tensors[0].dtype
                tensors[0].copy_(tensor)
                consumed.set()
            return Work(copy)
        def shutdown(self):
            if self.fail_close:
                raise RuntimeError('injected NCCL shutdown failure')
            self.closed = True
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(dist, 'TCPStore', lambda *args, **kwargs: object())
    monkeypatch.setattr(dist, 'PrefixStore', lambda session, store: store)
    monkeypatch.setattr(dist, 'ProcessGroupNCCL', Group)
    return SimpleNamespace(groups=groups, channels=channels)


@pytest.mark.parametrize('source_tp,target_tp', [(1, 2), (2, 4), (4, 2)])
def test_real_worker_tensor_transfer_all_parameters_and_generations(wire, source_tp, target_tp):
    sources = [worker(source_tp, rank)[0] for rank in range(source_tp)]
    targets, expected = zip(*(worker(target_tp, rank, target=True) for rank in range(target_tp)))
    all_workers = [*sources, *targets]
    common = dict(transaction_id='cpu-tx', session_id='cpu-session', store_host='127.0.0.1', store_port=19080,
                  world_size=len(all_workers), gpu_ids=list(range(len(all_workers))))
    for index, value in enumerate(all_workers):
        receipt = value.dynamo_operation('open', dict(common, rank_offset=0 if index < source_tp else source_tp,
                                                     expected_generation=value._native_generation))
        assert receipt['rank'] == value.rank and receipt['generation'] == value._native_generation
    source_meta = sources[0].dynamo_operation('describe', {'expected_generation': 0})
    target_meta = targets[0].dynamo_operation('describe', {'expected_generation': 1})
    transfer = dict(common, operation_id='cpu-tx', source_ranks=list(range(source_tp)),
                    target_ranks=list(range(source_tp, len(all_workers))), geometry=GEOMETRY,
                    source_shapes=source_meta['parameters'], target_shapes=target_meta['parameters'],
                    compare_target_before_copy=False)
    def execute(value):
        return value.dynamo_operation('transfer', dict(transfer, expected_generation=value._native_generation))
    with ThreadPoolExecutor(max_workers=len(all_workers)) as pool:
        results = list(pool.map(execute, all_workers))
    sent = sum(row['sent_bytes'] for row in results)
    received = sum(row['received_bytes'] for row in results)
    assert sent == received == sum(p.numel()*p.element_size() for value in targets for p in value.model_runner.model.parameters())
    assert all(row['parameter_count'] == len(SHAPES) for row in results)
    assert all(row['target_complete'] for row in results[source_tp:])
    for target, wanted in zip(targets, expected):
        assert all(torch.equal(value, wanted[name]) for name, value in target.model_runner.model.named_parameters())
        with pytest.raises(ValueError, match='verified complete'):
            target.dynamo_operation('mark_ready', dict(operation_id='cpu-tx', transaction_id='cpu-tx', verified=False))
        assert target.dynamo_operation('mark_ready', dict(operation_id='cpu-tx', transaction_id='cpu-tx', verified=True))['weights_ready']
    # Replay is acknowledged from the stored operation without new sends.
    assert execute(targets[0])['received_bytes'] == results[source_tp]['received_bytes']
    for value in all_workers:
        closed = value.dynamo_operation('close', dict(transaction_id='cpu-tx', session_id='cpu-session'))
        assert closed['closed'] and not value._dynamo_weight_sessions
        assert value.dynamo_operation('drain', {})['active_weight_sessions'] == 0
    assert all(group.closed for group in wire.groups.values())
    assert all(channel.empty() for channel in wire.channels.values())
    assert not torch.cuda.is_initialized()


def test_failed_shutdown_keeps_uncertain_worker_session_for_isolation(wire):
    value, _ = worker(1, 0)
    common = dict(transaction_id='cpu-tx', session_id='cpu-session', store_host='127.0.0.1', store_port=19080,
                  world_size=1, gpu_ids=[0], rank_offset=0)
    value.dynamo_operation('open', common)
    wire.groups[0].fail_close = True
    with pytest.raises(RuntimeError, match='shutdown failure'):
        value.dynamo_operation('close', common)
    assert 'cpu-session' in value._dynamo_weight_sessions
    drained = value.dynamo_operation('drain', {})
    assert not drained['drained'] and drained['active_weight_sessions'] == 1
    assert not torch.cuda.is_initialized()
