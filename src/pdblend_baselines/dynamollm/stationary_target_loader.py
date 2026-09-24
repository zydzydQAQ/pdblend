"""Private vLLM loader: original IPC fragments, no checkpoint or dummy weights.

This module only constructs the target model and nonweight RoPE buffers. The
target worker must still initialize real KV and run native golden requests.
"""
from __future__ import annotations

import torch
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from .stationary_binding import build_qwen_meta_target
from .stationary_ipc import TorchCudaIpcCodec,need
from .stationary_layers import BorrowableStationaryConsumer,FragmentInventory
from .stationary_target_bootstrap import TargetBootstrap


LOAD_FORMAT='dynamo_stationary_same_tp_v1'


def initialize_rope_buffers(model,device,*,cpu_oracle=False):
    """Initialize only the pinned standard Qwen RoPE cache, never move the model."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    device=torch.device(device)
    need(device.type==('cpu' if cpu_oracle else 'cuda'), 'explicit CPU oracle or actual CUDA buffer device required')
    pending=[]
    for module_name,module in model.named_modules():
        for name,value in module._buffers.items():
            if value is None:continue
            need(value.is_meta, 'target nonweight buffer was allocated before private initialization')
            need(type(module) is RotaryEmbedding and name=='cos_sin_cache',
                 'unreviewed target nonweight buffer: '+module_name+'.'+name)
            pending.append((module_name,module,name,value))
    created=[]
    try:
        for module_name,module,name,old in pending:
            with torch.device(device):
                value=module._compute_cos_sin_cache().to(module.dtype)
            need(tuple(value.shape)==tuple(old.shape) and value.dtype==old.dtype and value.device==device,
                 'real target RoPE cache differs from model metadata')
            module._buffers[name]=value
            created.append((module,name,old))
        need(all(not b.is_meta and b.device==device for b in model.buffers()), 'uninitialized target nonweight buffer')
    except BaseException:
        for module,name,old in reversed(created):module._buffers[name]=old
        raise
    return [dict(module=name,buffer=key,shape=list(old.shape),dtype=str(old.dtype),
        bytes=module._buffers[key].numel()*module._buffers[key].element_size())
        for name,module,key,old in pending]


def initialize_attention_constants(model,device,*,cpu_oracle=False):
    """Pinned Attention stores these nonweight tensors outside registered buffers.

    Only the fixed nonquantized auto-KV defaults are constructed. No tensor is
    moved from a meta parameter, no checkpoint scale is guessed, and the native
    KV allocator retains responsibility for its empty initial placeholders.
    """
    from vllm import envs
    from vllm.attention.layer import Attention
    device=torch.device(device)
    need(device.type==('cpu' if cpu_oracle else 'cuda'),'actual target device or explicit CPU oracle required')
    constants=dict(_q_scale=1.,_k_scale=1.,_v_scale=1.,_prob_scale=1.,
        q_range=envs.Q_SCALE_CONSTANT,k_range=envs.K_SCALE_CONSTANT,v_range=envs.V_SCALE_CONSTANT)
    pending=[]
    for module_name,module in model.named_modules():
        tensors={name:value for name,value in vars(module).items() if isinstance(value,torch.Tensor)}
        if type(module) is not Attention:
            need(not tensors,'unreviewed unregistered target tensors: '+module_name)
            continue
        need(module.kv_cache_dtype=='auto' and module.calculate_kv_scales is False
             and set(tensors)==set(constants) and module._k_scale_float==module._v_scale_float==1.,
             'only fixed unquantized Attention nonweight constants are supported')
        need(len(module.kv_cache)==1 and all(v.is_meta and v.numel()==0 for v in module.kv_cache),
             'actual target KV must be initialized later by the native worker')
        for name,value in tensors.items():
            need(value.is_meta and value.shape==torch.Size([]) and value.dtype==torch.float32,
                 'Attention scalar differs from pinned meta constructor')
            pending.append((module_name,module,name,value))
    created=[]
    try:
        for _,module,name,old in pending:
            setattr(module,name,torch.tensor(constants[name],dtype=old.dtype,device=device))
            created.append((module,name,old))
    except BaseException:
        for module,name,old in reversed(created):setattr(module,name,old)
        raise
    return [dict(module=module_name,attribute=name,shape=[],dtype=str(old.dtype),bytes=4,
                 constructor_default=constants[name],registered_buffer=False)
            for module_name,_,name,old in pending]


class TargetModelRuntime:
    def __init__(self,bootstrap,codec):
        self.bootstrap,self.codec=bootstrap,codec
        self.consumer=self.inventory=self.binding=None
        self.buffers=[];self.closed=False;self.release_ack=None

    def load(self,vllm_config):
        from vllm.distributed import get_tensor_model_parallel_world_size,get_tensor_model_parallel_rank
        c=self.bootstrap.config;pc=vllm_config.parallel_config;mc=vllm_config.model_config
        need(pc.tensor_parallel_size==pc.pipeline_parallel_size==1
             and get_tensor_model_parallel_world_size()==1 and get_tensor_model_parallel_rank()==0,
             'same-TP target must own its independent native TP1 group')
        need(mc.enforce_eager and not mc.enable_sleep_mode and getattr(mc.hf_config,'rope_scaling',None) is None
             and vllm_config.kv_transfer_config is None and vllm_config.speculative_config is None,
             'initial target requires eager standard-RoPE Qwen, no sleep/connector/speculation')
        self.bootstrap.publish_ready(gpu_uuid=self.codec.uuid,device_index=self.codec.device)
        try:
            packet=self.bootstrap.wait_packet()
            self.consumer=BorrowableStationaryConsumer(packet,plan=c['plan'],codec=self.codec,
                expected_generation=c['source_generation'])
            self.inventory=FragmentInventory.from_ipc(c['plan'],0,target_codec=self.codec,
                consumers=[self.consumer],missing=[])
            self.binding=build_qwen_meta_target(vllm_config,self.inventory)
            self.buffers=initialize_rope_buffers(self.binding.model,torch.device('cuda',self.codec.device))
            self.buffers+=initialize_attention_constants(self.binding.model,torch.device('cuda',self.codec.device))
            self.binding.model.eval()
            self.binding.model._dynamo_stationary_target_runtime=self
            self.bootstrap.record('model-bound',binding=self.binding.receipt(),buffers=self.buffers,
                target_KV_initialized=False,target_served=False,formal_eligible=False)
            return self.binding.model
        except BaseException as error:
            cleanup_error=None
            try:self.close()
            except BaseException as cleanup:cleanup_error=repr(cleanup)
            self.bootstrap.record('load-failed',error=repr(error),cleanup_error=cleanup_error,
                release_ack=self.release_ack,target_served=False,formal_eligible=False)
            raise

    def close(self):
        if self.closed:return self.release_ack
        if self.binding is not None:self.binding.close()
        if self.inventory is not None:self.inventory.close()
        if self.consumer is not None:self.release_ack=self.consumer.close()
        self.closed=True
        self.bootstrap.record('views-released',release_ack=self.release_ack,target_process_exit_required=True,
            source_owner_must_remain_alive=True,formal_eligible=False)
        return self.release_ack


@register_model_loader(LOAD_FORMAT)
class StationaryTargetLoader(BaseModelLoader):
    def download_model(self,model_config):
        raise RuntimeError('stationary target never downloads or loads checkpoint weights')

    def load_weights(self,model,model_config):
        raise RuntimeError('stationary target cannot reload or reconstruct dense weights')

    def load_model(self,vllm_config,model_config):
        extra=self.load_config.model_loader_extra_config
        need(set(extra)=={'bootstrap_ref'}, 'explicit immutable target bootstrap binding required')
        bootstrap=TargetBootstrap(extra['bootstrap_ref'])
        c=bootstrap.config
        need(model_config is vllm_config.model_config
             and vllm_config.cache_config.gpu_memory_utilization==c['target_gpu_memory_utilization'],
             'target native memory request or model config differs from bootstrap')
        codec=TorchCudaIpcCodec(gpu_uuid=c['plan']['target_gpu_uuids'][0],device_index=torch.cuda.current_device())
        return TargetModelRuntime(bootstrap,codec).load(vllm_config)
