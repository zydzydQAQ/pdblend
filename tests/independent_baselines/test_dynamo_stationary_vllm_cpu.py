"""Pinned-image layer ABI check, CPU only; no TP group or CUDA qualification.

Run inside the fixed image with --runtime=runc, NVIDIA_VISIBLE_DEVICES=void.
TP rank/size lookups and collectives use declared one-rank CPU oracle geometry.
Actual vLLM constructors, forwards and logits entrypoint run without model load.
"""
import pytest
import torch
import torch.nn.functional as F

vllm = pytest.importorskip('vllm')

from pdblend_baselines.dynamollm.stationary_binding import MetaTargetBinding, build_meta_target
from pdblend_baselines.dynamollm.stationary_layers import FragmentInventory
from pdblend_baselines.dynamollm.stationary_tensors import tensor_plan
from tests.independent_baselines.test_dynamo_stationary_tensors import GEOMETRY, shapes, shard


def test_actual_pinned_vllm_meta_layers_and_logits_with_no_cuda(monkeypatch):
    assert vllm.__version__ == '0.10.1.1'
    from vllm.config import VllmConfig, DeviceConfig, set_current_vllm_config
    from vllm.model_executor.layers import linear as L, vocab_parallel_embedding as E
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    for module in (L, E):
        monkeypatch.setattr(module, 'get_tensor_model_parallel_world_size', lambda: 1)
        monkeypatch.setattr(module, 'get_tensor_model_parallel_rank', lambda: 0)
    # Distribution collectives are outside this one-rank layer ABI test.
    monkeypatch.setattr(LogitsProcessor, '_gather_logits', lambda self, logits: logits)
    embedding_collectives = []
    def embedding_reduce(output):
        embedding_collectives.append(output.shape)
        return output
    monkeypatch.setattr(E, 'tensor_model_parallel_all_reduce', embedding_reduce)
    source_shapes = dict(shapes(2), **{'lm_head.weight': [16, 16]})
    target_shapes = dict(shapes(1), **{'lm_head.weight': [32, 16]})
    plan = tensor_plan(source_gpus=['GPU-0', 'GPU-1'], target_gpus=['GPU-0'],
        source_shapes=source_shapes, target_shapes=target_shapes, geometry=GEOMETRY)
    references = {name: torch.tensor(shard('model.embed_tokens.weight' if name == 'lm_head.weight' else name, 1, 0).astype('float64') / 100)
                  for name in target_shapes}
    owners = {(rank, name): torch.tensor(shard('model.embed_tokens.weight' if name == 'lm_head.weight' else name, 2, rank).astype('float64') / 100)
              for rank in range(2) for name in source_shapes}
    fragments = []
    for piece in plan['pieces']:
        old = owners[piece['source_rank'], piece['parameter']]
        view = old if piece['axis'] is None else old.narrow(piece['axis'], piece['source_offset'], piece['length'])
        fragments.append((piece, view if piece['kind'] == 'retain_on_gpu' else view.clone()))
    inventory = FragmentInventory.cpu_oracle(plan, 0, fragments)
    def factory():
        root = torch.nn.Module()
        root.model = torch.nn.Module()
        root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = root.model.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = L.QKVParallelLinear(16, 2, 8, 4, params_dtype=torch.float64)
        layer.self_attn.o_proj = L.RowParallelLinear(16, 16, bias=False, params_dtype=torch.float64)
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = L.MergedColumnParallelLinear(16, [32, 32], bias=False, params_dtype=torch.float64)
        layer.mlp.down_proj = L.RowParallelLinear(32, 16, bias=False, params_dtype=torch.float64)
        root.model.embed_tokens = E.VocabParallelEmbedding(32, 16, padding_size=1, params_dtype=torch.float64)
        root.model.norm = RMSNorm(16, dtype=torch.float64)
        root.lm_head = E.ParallelLMHead(32, 16, padding_size=1, params_dtype=torch.float64)
        root.logits_processor = LogitsProcessor(32)
        return root
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device='cpu'))):
        model = build_meta_target(factory)
        binding = MetaTargetBinding(model, inventory)
        x = torch.arange(48, dtype=torch.float64).reshape(3, 16) / 30
        for name, layer in model.named_modules():
            key = name + '.weight'
            if key not in references or name.endswith(('norm', 'embed_tokens', 'lm_head')):
                continue
            input_ = x if references[key].shape[1] == 16 else torch.cat([x, x], dim=-1)
            actual = layer(input_)[0]
            torch.testing.assert_close(actual, F.linear(input_, references[key], references.get(name + '.bias')))
        ids = torch.tensor([0, 15, 16, 31])
        torch.testing.assert_close(model.model.embed_tokens(ids), F.embedding(ids, references['model.embed_tokens.weight']))
        assert embedding_collectives == [torch.Size([4, 16])]
        torch.testing.assert_close(model.logits_processor._get_logits(x, model.lm_head, None), F.linear(x, references['lm_head.weight']))
        assert model.model.norm.weight.untyped_storage().data_ptr() == inventory.parameters['model.norm.weight'].replicated_view().untyped_storage().data_ptr()
        assert not binding.receipt()['hardware_qualified']
        binding.close()
        inventory.close()
    assert not torch.cuda.is_initialized()
