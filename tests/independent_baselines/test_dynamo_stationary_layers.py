"""Small real Torch CPU oracles; never initialize CUDA or qualify serving."""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from pdblend_baselines.dynamollm.stationary_binding import MetaTargetBinding, build_meta_target
from pdblend_baselines.dynamollm.stationary_ipc import digest
from pdblend_baselines.dynamollm.stationary_layers import (
    BorrowableStationaryConsumer, FragmentInventory, FragmentParameter, SegmentedQuantMethod,
)
from pdblend_baselines.dynamollm.stationary_tensors import tensor_plan
from tests.independent_baselines.test_dynamo_stationary_tensors import GEOMETRY, shapes, shard
from tests.independent_baselines.test_dynamo_stationary_ipc import setup_owner, CONSUMER


def fixture(source=2, target=1, rank=0, *, dtype=torch.float64):
    plan = tensor_plan(source_gpus=['GPU-' + str(i) for i in range(source)],
        target_gpus=['GPU-' + str(i) for i in range(target)], source_shapes=shapes(source),
        target_shapes=shapes(target), geometry=GEOMETRY)
    owners = {(sr, name): torch.tensor(shard(name, source, sr).astype('float64') / 100., dtype=dtype)
              for sr in range(source) for name in shapes(source)}
    fragments = []
    for piece in plan['pieces']:
        if piece['target_rank'] != rank:
            continue
        original = owners[piece['source_rank'], piece['parameter']]
        view = original if piece['axis'] is None else original.narrow(piece['axis'], piece['source_offset'], piece['length'])
        # Incoming-only test buffers model a distinct device. Never clone the
        # retained case, and never describe this CPU construction as transport.
        if piece['kind'] == 'direct_gpu_transfer':
            view = view.clone()
        fragments.append((piece, view))
    return plan, owners, fragments, FragmentInventory.cpu_oracle(plan, rank, fragments)


@pytest.mark.parametrize('source,target', [(1, 2), (2, 1), (2, 4), (4, 2)])
def test_exact_qkv_gate_up_row_and_vocab_arithmetic_without_materializing_weights(source, target, monkeypatch):
    for rank in range(target):
        plan, owners, fragments, inventory = fixture(source, target, rank)
        before = {key: value.untyped_storage().data_ptr() for key, value in owners.items()}
        for piece, view in fragments:
            if piece['kind'] == 'retain_on_gpu':
                assert view.untyped_storage().data_ptr() == before[piece['source_rank'], piece['parameter']]
        # The independent target constructor exists only in this oracle.
        references = {name: torch.tensor(shard(name, target, rank).astype('float64') / 100.)
                      for name in shapes(target)}
        def no_weight_copy(*args, **kwargs):
            raise AssertionError('retained weight materialization attempted')
        with monkeypatch.context() as patch:
            patch.setattr(torch, 'cat', no_weight_copy)
            patch.setattr(torch.Tensor, 'clone', no_weight_copy)
            patch.setattr(torch.Tensor, 'contiguous', no_weight_copy)
            for name, param in inventory.parameters.items():
                if len(param.shape) != 2:
                    continue
                x = torch.arange(2 * 3 * param.shape[1], dtype=torch.float64).reshape(2, 3, -1) / 19
                bias_name = name.replace('.weight', '.bias')
                bias = inventory.parameters.get(bias_name)
                actual = param.linear(x, bias)
                expected = F.linear(x, references[name], references.get(bias_name))
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-12)
        receipt = inventory.receipt()
        assert receipt['cpu_oracle'] and not receipt['hardware_qualified']
        assert receipt['retained_fragment_copy_bytes'] == receipt['dense_target_weight_allocated_bytes'] == 0
        assert {key: value.untyped_storage().data_ptr() for key, value in owners.items()} == before


@pytest.mark.parametrize('source,target,rank', [(2, 1, 0), (1, 2, 0), (1, 2, 1), (4, 2, 1)])
def test_embedding_boundary_tokens_padding_and_lm_head_use_same_fragments(source, target, rank):
    _, _, _, inventory = fixture(source, target, rank)
    name = 'model.embed_tokens.weight'
    p = inventory.parameters[name]
    ids = torch.arange(p.shape[0], dtype=torch.int64).reshape(2, -1)
    expected = torch.tensor(shard(name, target, rank).astype('float64') / 100.)
    method = SegmentedQuantMethod(inventory, name)
    torch.testing.assert_close(method.embedding(None, ids), F.embedding(ids, expected))
    hidden = torch.arange(3 * p.shape[1], dtype=torch.float64).reshape(3, -1) / 15
    torch.testing.assert_close(method.apply(None, hidden), F.linear(hidden, expected))
    for invalid in (-1, p.shape[0]):
        with pytest.raises(ValueError, match='outside vocabulary'):
            p.embedding(torch.tensor([invalid]))
    with pytest.raises(ValueError, match='bias'):
        method.apply(None, hidden, torch.zeros(p.shape[0]))


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'foreign', 'plan', 'mutated', 'stride', 'device', 'grad'])
def test_fragment_inventory_rejects_unsafe_or_incomplete_binding(fault):
    plan, owners, fragments, inventory = fixture()
    if fault == 'mutated':
        owners[0, 'model.norm.weight'].add_(1)
        with pytest.raises(ValueError, match='version changed'):
            inventory.check()
        return
    if fault == 'missing':
        fragments.pop()
    elif fault == 'duplicate':
        fragments.append(fragments[0])
    elif fault == 'foreign':
        fragments[0] = (dict(fragments[0][0], source_gpu_uuid='GPU-OTHER'), fragments[0][1])
    elif fault == 'plan':
        plan['planned_retained_bytes'] += 2
    else:
        index = next(i for i, (_, v) in enumerate(fragments) if v.ndim == 2)
        piece, v = fragments[index]
        if fault == 'stride':
            v = torch.empty(v.shape[1], v.shape[0], dtype=v.dtype).t()
        elif fault == 'device':
            v = torch.empty(v.shape, dtype=v.dtype, device='meta')
        elif fault == 'grad':
            v = v.detach().requires_grad_()
        fragments[index] = piece, v
    with pytest.raises(ValueError):
        FragmentInventory.cpu_oracle(plan, 0, fragments)


def test_cuda_path_rejects_cpu_arithmetic_and_unvalidated_device():
    plan, _, fragments, _ = fixture()
    with pytest.raises(ValueError, match='CUDA BF16'):
        FragmentInventory(plan, 0, fragments, consumers=[], cpu_oracle=False)
    with pytest.raises(ValueError, match='physical UUID'):
        FragmentInventory.from_ipc(plan, 0, target_codec=object(), consumers=[], missing=[])


def test_borrowed_ipc_consumer_cannot_acknowledge_release_until_all_layers_drop_views():
    # Existing genuine CPU alias codec exercises the same production process /
    # generation/packet validator, without pretending to initialize CUDA.
    plan, _, alive, codec, owner = setup_owner()
    packet = owner.export(CONSUMER, target_rank=0)
    consumer = BorrowableStationaryConsumer(packet, plan=plan, codec=codec, expected_generation=3,
        consumer_identity=CONSUMER, process_alive=lambda identity: digest(identity) in alive)
    a, b = object(), object()
    consumer.borrow(a)
    consumer.borrow(b)
    with pytest.raises(ValueError, match='still borrow'):
        consumer.close()
    consumer.return_borrow(a)
    with pytest.raises(ValueError, match='still borrow'):
        consumer.close()
    consumer.return_borrow(b)
    assert consumer.close()['views_released']


class DenseOracleMethod:
    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)
    def embedding(self, layer, ids):
        return F.embedding(ids, layer.weight)


class OracleLayer(torch.nn.Module):
    def __init__(self, shape, kind):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(shape, dtype=torch.float64), requires_grad=False)
        self.kind = kind
        self.quant_method = DenseOracleMethod()
        self.bias = None
        self.skip_bias_add = False
    def forward(self, x):
        if self.kind == 'norm':
            return x * self.weight
        if self.kind == 'embedding':
            return self.quant_method.embedding(self, x)
        return self.quant_method.apply(self, x, self.bias), None


def oracle_model(tp=1):
    model, kinds = torch.nn.Module(), {}
    for name, shape in shapes(tp).items():
        if name.endswith('.bias'):
            continue
        parent = model
        parts = name.split('.')
        for part in parts[:-2]:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        kind = ('norm' if name.endswith('norm.weight') else 'embedding' if 'embed_tokens' in name
                else 'row' if name.endswith(('o_proj.weight', 'down_proj.weight')) else 'column')
        layer = OracleLayer(shape, kind)
        parent.add_module(parts[-2], layer)
        kinds['.'.join(parts[:-1])] = kind
        if 'qkv_proj.weight' in name:
            layer.bias = torch.nn.Parameter(torch.empty(shape[0], dtype=torch.float64), requires_grad=False)
    # Nonweight buffers need a separate target initialization hook; binding must
    # leave this visible rather than silently copy source CUDA graph/KV state.
    model.register_buffer('rope_cache', torch.empty((3, 2), dtype=torch.float64))
    model.oracle_kinds = kinds
    return model


def test_meta_construction_and_layer_binding_allocate_no_target_weight_storage(monkeypatch):
    _, _, fragments, inventory = fixture()
    model = build_meta_target(oracle_model)
    assert all(p.is_meta for p in model.parameters())
    original = dict(model.named_parameters())
    qkv = model.model.layers.get_submodule('0').self_attn.qkv_proj
    original_forward = qkv.forward.__func__
    binding = MetaTargetBinding(model, inventory, cpu_oracle_kinds=model.oracle_kinds)
    assert qkv.weight is None and qkv.bias is None
    assert qkv.forward.__func__ is original_forward
    assert model.model.norm.weight.untyped_storage().data_ptr() == inventory.parameters['model.norm.weight'].replicated_view().untyped_storage().data_ptr()
    x = torch.arange(32, dtype=torch.float64).reshape(2, 16) / 17
    expected_w = torch.tensor(shard('model.layers.0.self_attn.qkv_proj.weight', 1, 0).astype('float64') / 100)
    expected_b = torch.tensor(shard('model.layers.0.self_attn.qkv_proj.bias', 1, 0).astype('float64') / 100)
    torch.testing.assert_close(qkv(x)[0], F.linear(x, expected_w, expected_b))
    torch.testing.assert_close(model.model.norm(x), x * inventory.parameters['model.norm.weight'].replicated_view())
    for module in (model, qkv, model.model.norm):
        with pytest.raises(RuntimeError, match='cannot move'):
            module.to(dtype=torch.float32)
        with pytest.raises(RuntimeError, match='cannot move'):
            module.load_state_dict({})
    with pytest.raises(ValueError, match='still bound'):
        inventory.close()
    receipt = binding.receipt()
    assert receipt['meta_buffers'] == ['rope_cache']
    assert not receipt['hardware_qualified'] and not receipt['target_engine_activated']
    with pytest.raises(RuntimeError, match='not qualified'):
        binding.require_activation()
    binding.close()
    assert all(p is original[name] and p.is_meta for name, p in model.named_parameters())
    assert qkv.quant_method.__class__ is DenseOracleMethod
    inventory.close()
    assert inventory.closed


@pytest.mark.parametrize('fault', ['real_weight', 'real_buffer', 'missing', 'wrong_shape', 'quantized', 'deferred_bias', 'unsupported', 'tied'])
def test_target_preflight_rejects_before_modifying_any_layer(fault):
    _, _, _, inventory = fixture()
    model = build_meta_target(oracle_model)
    qkv = model.model.layers.get_submodule('0').self_attn.qkv_proj
    kinds = dict(model.oracle_kinds)
    if fault == 'real_weight':
        qkv.weight = torch.nn.Parameter(torch.zeros(qkv.weight.shape, dtype=torch.float64), requires_grad=False)
    elif fault == 'real_buffer':
        model.rope_cache = torch.empty(3, 2)
    elif fault == 'missing':
        model.model.norm.weight = None
    elif fault == 'wrong_shape':
        qkv.weight = torch.nn.Parameter(torch.empty(1, 1, device='meta', dtype=torch.float64))
    elif fault == 'quantized':
        # Real entrypoint refuses CPU fake classes instead of interpreting them
        # as an arbitrary third-party quantized vLLM layer.
        kinds = None
    elif fault == 'deferred_bias':
        qkv.skip_bias_add = True
    elif fault == 'unsupported':
        kinds['model.norm'] = 'unknown'
    elif fault == 'tied':
        model.model.norm.weight = qkv.bias
    original = qkv.quant_method
    with pytest.raises(ValueError):
        MetaTargetBinding(model, inventory, cpu_oracle_kinds=kinds)
    assert qkv.quant_method is original and not inventory._bindings


def test_binding_failure_rolls_back_all_meta_parameters_and_storage_borrows(monkeypatch):
    _, _, _, inventory = fixture()
    model = build_meta_target(oracle_model)
    original = dict(model.named_parameters())
    real = MetaTargetBinding._remember
    def fail_after_one_layer(self, obj, attr):
        if attr == 'weight' and self._methods:
            raise RuntimeError('injected second layer binding failure')
        return real(self, obj, attr)
    monkeypatch.setattr(MetaTargetBinding, '_remember', fail_after_one_layer)
    with pytest.raises(RuntimeError, match='injected'):
        MetaTargetBinding(model, inventory, cpu_oracle_kinds=model.oracle_kinds)
    assert all(p is original[name] for name, p in model.named_parameters())
    assert not inventory._bindings
    inventory.close()


def test_meta_factory_rejects_explicit_real_allocation_and_no_cuda_was_initialized():
    def bad_factory():
        return torch.nn.Linear(2, 2, device='cpu')
    with pytest.raises(ValueError, match='allocated real'):
        build_meta_target(bad_factory)
    assert not torch.cuda.is_initialized()
