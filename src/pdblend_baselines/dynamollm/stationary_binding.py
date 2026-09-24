"""Dynamo target parameter binding without allocating complete dummy weights.

The factory constructs metadata on torch's meta device. Only a proven complete
fragment inventory can replace those placeholders. This is a per-target hook,
not a public vLLM patch or a working KV/TP reconfiguration lifecycle.
"""
from __future__ import annotations

from types import MethodType

import torch

from .stationary_ipc import need
from .stationary_layers import SegmentedQuantMethod


_LINEAR_MODULE = 'vllm.model_executor.layers.linear'
_VOCAB_MODULE = 'vllm.model_executor.layers.vocab_parallel_embedding'
_KINDS = {
    (_LINEAR_MODULE, 'QKVParallelLinear'): 'column',
    (_LINEAR_MODULE, 'MergedColumnParallelLinear'): 'column',
    (_LINEAR_MODULE, 'RowParallelLinear'): 'row',
    (_VOCAB_MODULE, 'VocabParallelEmbedding'): 'embedding',
    (_VOCAB_MODULE, 'ParallelLMHead'): 'lm_head',
    ('vllm.model_executor.layers.layernorm', 'RMSNorm'): 'norm',
}


def build_meta_target(factory, *args, **kwargs):
    """Call an isolated target factory with zero allocated parameter storage.

    A constructor that explicitly ignores the meta device is rejected. This
    postcondition cannot undo its side effects; use only the pinned, reviewed
    Qwen factory, never an arbitrary live engine factory.
    """
    with torch.device('meta'):
        model = factory(*args, **kwargs)
    need(isinstance(model, torch.nn.Module), 'target factory must return a torch model')
    need(all(p.is_meta for p in model.parameters()) and all(b.is_meta for b in model.buffers()),
         'target factory allocated real parameter/buffer storage before fragment binding')
    return model


def build_qwen_meta_target(vllm_config, inventory):
    """Pinned native Qwen constructor hook; caller must supply isolated TP state.

    This does not initialize a worker, allocate KV, release source KV, load a
    checkpoint, initialize model buffers, or publish a routing endpoint.
    """
    need(not inventory.is_cpu_oracle, 'native Qwen binding requires actual CUDA fragment inventory')
    inventory.check()
    mc, pc = vllm_config.model_config, vllm_config.parallel_config
    hf = mc.hf_config
    need(vllm_config.quant_config is None and vllm_config.lora_config is None
         and mc.dtype == torch.bfloat16 and not hf.tie_word_embeddings,
         'native stationary hook requires untied unquantized BF16 Qwen without LoRA')
    need(getattr(hf, 'model_type', None) == 'qwen2' and hf.architectures == ['Qwen2ForCausalLM'],
         'only pinned Qwen2 causal model is supported')
    need(pc.pipeline_parallel_size == 1 and getattr(pc, 'data_parallel_size', 1) == 1
         and pc.tensor_parallel_size == len(inventory.plan['target_gpu_uuids']), 'native target topology differs')
    need(all(getattr(hf, key) == value for key, value in inventory.plan['geometry'].items()),
         'native target Qwen geometry differs from retained tensor plan')
    import vllm
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    need(vllm.__version__ == '0.10.1.1', 'native target vLLM revision differs')
    need(get_tensor_model_parallel_rank() == inventory.target_rank
         and get_tensor_model_parallel_world_size() == pc.tensor_parallel_size,
         'target TP group is absent or belongs to another rank/layout')
    with set_current_vllm_config(vllm_config):
        # vLLM's loader normally owns the default dtype context. This direct
        # constructor intentionally skips that loader and restores the context.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = build_meta_target(Qwen2ForCausalLM, vllm_config=vllm_config)
        finally:
            torch.set_default_dtype(previous_dtype)
    return MetaTargetBinding(model, inventory)


def _forbid_storage_move(self, *args, **kwargs):
    raise RuntimeError('stationary target cannot move/convert/load dense weights; retire the binding first')


class MetaTargetBinding:
    """Transactional fragment binding into one inactive Qwen target model.

    The unchanged Column/Row/Embedding forwards retain vLLM's TP collectives.
    There is no torch.cat, retained clone, dense load_state_dict or weight copy.
    close() restores metadata, drops layer references, then releases the borrow.
    """

    def __init__(self, model, inventory, *, cpu_oracle_kinds=None):
        inventory.check()
        need(cpu_oracle_kinds is None or inventory.is_cpu_oracle, 'oracle layer mapping forbidden for CUDA binding')
        parameters = dict(model.named_parameters(remove_duplicate=False))
        need(set(parameters) == set(inventory.parameters), 'target model parameter inventory differs')
        need(len({id(v) for v in parameters.values()}) == len(parameters),
             'tied/aliased target parameters require a separate qualified binding')
        need(all(p.is_meta for p in parameters.values()) and all(b.is_meta for b in model.buffers()),
             'target must contain only meta placeholders before binding')
        self.model, self.inventory, self.closed = model, inventory, False
        self._undo, self._methods, self._norm_handles = [], [], []
        actions = []
        covered = set()
        for module_name, module in model.named_modules():
            weight_name = module_name + '.weight' if module_name else 'weight'
            if weight_name not in parameters:
                continue
            kind = (_KINDS.get((type(module).__module__, type(module).__name__))
                    if cpu_oracle_kinds is None else cpu_oracle_kinds.get(module_name))
            need(kind in ('column', 'row', 'embedding', 'lm_head', 'norm'),
                 'unsupported target weight owner: ' + module_name)
            param = inventory.parameters[weight_name]
            need(tuple(module.weight.shape) == param.shape and module.weight.dtype == param.dtype,
                 'target weight metadata differs')
            bias_name = module_name + '.bias' if module_name else 'bias'
            bias_name = bias_name if bias_name in parameters else None
            if kind == 'norm':
                need(param.axis is None and bias_name is None, 'only replicated bias-free RMSNorm is supported')
            else:
                need(param.axis == (1 if kind == 'row' else 0), 'target execution partition axis differs')
                need(not getattr(module, 'skip_bias_add', False), 'deferred bias path is not qualified')
                if cpu_oracle_kinds is None:
                    method = getattr(module, 'quant_method', None)
                    need(method is not None and type(method).__name__ in
                         ('UnquantizedLinearMethod', 'UnquantizedEmbeddingMethod'),
                         'only pinned unquantized BF16 target is supported')
                if bias_name is not None:
                    need(kind == 'column' and weight_name.endswith('qkv_proj.weight'),
                         'only Qwen QKV column bias has qualified fragment semantics')
                    bp = inventory.parameters[bias_name]
                    need(bp.shape == (param.shape[0],) and bp.axis == 0
                         and module.bias.dtype == bp.dtype, 'QKV bias fragment metadata differs')
            actions.append((module, kind, weight_name, bias_name))
            covered.add(weight_name)
            if bias_name:
                covered.add(bias_name)
        need(covered == set(parameters), 'target includes an unsupported standalone parameter')
        inventory.borrow_binding(self)
        try:
            for module, kind, weight_name, bias_name in actions:
                self._remember(module, 'weight')
                if kind == 'norm':
                    view = inventory.parameters[weight_name].replicated_view()
                    module.weight = torch.nn.Parameter(view, requires_grad=False)
                    need(module.weight.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
                         and module.weight.storage_offset() == view.storage_offset(), 'norm binding copied storage')
                    # Norm bypasses quant_method, so guard its use explicitly.
                    handle = module.register_forward_pre_hook(self._check_norm)
                    self._norm_handles.append(handle)
                else:
                    self._remember(module, 'quant_method')
                    method = SegmentedQuantMethod(inventory, weight_name, bias_name)
                    self._methods.append(method)
                    module.quant_method = method
                    # vLLM Column/Row forwards still call quant_method.apply.
                    # None prevents accidental dense access outside that path.
                    module.weight = None
                    if bias_name is not None:
                        self._remember(module, 'bias')
                        module.bias = None
            # Device moves and generic checkpoint loads would copy aliases.
            # Guard every module, including a caller acting on a child directly.
            for module in model.modules():
                for attr in ('_apply', 'load_state_dict'):
                    self._remember(module, attr)
                    setattr(module, attr, MethodType(_forbid_storage_move, module))
        except BaseException:
            self.close()
            raise

    def _remember(self, obj, attr):
        # Registered parameters are owned by _parameters, not __dict__.
        registered = attr in obj._parameters
        own = registered or attr in obj.__dict__
        self._undo.append((obj, attr, own, getattr(obj, attr, None)))

    def _check_norm(self, module, args):
        need(not self.closed, 'stationary model binding is closed')
        self.inventory.check()

    def receipt(self):
        self.inventory.check()
        need(not self.closed, 'stationary model binding is closed')
        return dict(schema='dynamo-meta-target-binding/v1',
            plan_sha256=self.inventory.plan['plan_sha256'], target_rank=self.inventory.target_rank,
            cpu_oracle=self.inventory.is_cpu_oracle, dense_target_weight_allocated_bytes=0,
            retained_fragment_copy_bytes=0, host_weight_staging_bytes=0,
            meta_buffers=[name for name, b in self.model.named_buffers() if b.is_meta],
            segmented_layers=len(self._methods), tp_collectives='unchanged target layer forward',
            remaining_gates=['native_KV_teardown_budget_rebuild', 'target_TP_group_isolation',
                             'nonweight_buffer_initialization', 'missing_fragment_transport_acceptance',
                             'fragment_aware_native_metadata_and_original_owner_donor_graph',
                             'BF16_ordinary_token_goldens', 'atomic_routing_and_failure_recovery'],
            hardware_qualified=False, target_engine_activated=False, formal_eligible=False)

    def require_activation(self):
        raise RuntimeError('stationary parameter binding has not qualified KV/TP/buffers/transport/goldens/routing')

    def close(self):
        if self.closed:
            return
        for method in self._methods:
            method.closed = True
        for handle in self._norm_handles:
            handle.remove()
        for obj, attr, own, value in reversed(self._undo):
            if own:
                setattr(obj, attr, value)
            elif attr in obj.__dict__:
                delattr(obj, attr)
        self._undo.clear()
        self._methods.clear()
        self._norm_handles.clear()
        self.inventory.return_binding(self)
        self.closed = True
