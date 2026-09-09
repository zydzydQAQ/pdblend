"""Qwen2 BF16 weight retention in CPU cache and bounded asynchronous TP reload.

This is a slow transaction facility. Existing GPU weights remain live until
source instances drain and are stopped; the reusable CPU shards survive that
stop. It does not claim in-place GPU resharding or KV migration.
"""
from collections import deque
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import torch
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


def regions(groups,source_tp,target_tp,target_rank):
    """Intersect global ranges of each packed Q/K/V or gate/up component."""
    if source_tp not in (1,2,4,8) or target_tp not in (1,2,4,8) or not 0<=target_rank<target_tp:
        raise ValueError('unsupported tensor parallelism')
    source_offset=target_offset=0
    result=[]
    for size in groups:
        if size<=0 or size%source_tp or size%target_tp:
            raise ValueError('packed component not divisible by TP')
        sw,tw=size//source_tp,size//target_tp
        left,right=target_rank*tw,(target_rank+1)*tw
        for rank in range(source_tp):
            lo,hi=max(left,rank*sw),min(right,(rank+1)*sw)
            if lo<hi:
                result.append((rank,source_offset+lo-rank*sw,target_offset+lo-left,hi-lo))
        source_offset+=sw
        target_offset+=tw
    return tuple(result)


def parameter_spec(module,name,parameter,tp):
    from vllm.model_executor.layers.linear import QKVParallelLinear,MergedColumnParallelLinear,RowParallelLinear
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from vllm.model_executor.layers.layernorm import RMSNorm
    if isinstance(module,(QKVParallelLinear,MergedColumnParallelLinear)):
        if isinstance(module,QKVParallelLinear) and module.num_kv_head_replicas!=1:
            raise ValueError('replicated GQA heads are not supported by retained-weight loader')
        axis,groups=0,list(module.output_sizes)
    elif isinstance(module,RowParallelLinear) and name=='weight':
        axis,groups=1,[parameter.shape[1]*tp]
    elif isinstance(module,VocabParallelEmbedding) and name=='weight':
        if module.num_added_embeddings:
            raise ValueError('LoRA vocabulary expansion is not supported')
        axis,groups=0,[module.num_embeddings_padded]
    elif isinstance(module,RMSNorm) or (isinstance(module,RowParallelLinear) and name=='bias'):
        axis,groups=None,[]
    else:
        raise ValueError(f'unvalidated retained parameter: {type(module).__name__}.{name}')
    return dict(axis=axis,groups=groups,dtype=str(parameter.dtype))


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def export_weights(worker,directory):
    from safetensors.torch import save_file
    from vllm.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size
    model=worker.model_runner.model
    config=worker.model_config
    if config.hf_config.model_type!='qwen2' or config.quantization or config.dtype!=torch.bfloat16:
        raise ValueError('weight retention currently validated only for unquantized Qwen2 BF16')
    rank,tp=get_tensor_model_parallel_rank(),get_tensor_model_parallel_world_size()
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    dest=root/f'rank-{rank}.safetensors'
    if dest.exists():
        raise FileExistsError('retained source shards are immutable')
    started=time.time()
    modules=dict(model.named_modules())
    specs={};weights={}
    for key,param in model.named_parameters():
        module_name,name=key.rsplit('.',1)
        specs[key]=parameter_spec(modules[module_name],name,param,tp)
        weights[key]=param.detach().to(device='cpu').contiguous()
    save_file(weights,str(dest.with_suffix('.tmp')))
    dest.with_suffix('.tmp').replace(dest)
    identity=hashlib.sha256(config.hf_config.to_json_string().encode()).hexdigest()
    result=dict(rank=rank,tp=tp,file=dest.name,sha256=file_sha256(dest),
        parameter_specs=specs,model_config_sha256=identity,
        bytes=sum(t.numel()*t.element_size() for t in weights.values()),
        started_s=started,finished_s=time.time())
    (root/f'rank-{rank}.json').write_text(json.dumps(result))
    return result


class RetainedWeightLoader(BaseModelLoader):
    """Read only intersecting source slices, copy with bounded pinned buffers."""
    def download_model(self,model_config):
        return None

    def load_weights(self,model,model_config):
        from safetensors import safe_open
        from vllm.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size
        extra=self.load_config.model_loader_extra_config or {}
        root=Path(extra['retained_weights'])
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest.get('schema')!=1 or not manifest.get('complete'):
            raise ValueError('incomplete retained weight transaction')
        identity=hashlib.sha256(model_config.hf_config.to_json_string().encode()).hexdigest()
        if identity!=manifest['model_config_sha256'] or model_config.quantization:
            raise ValueError('retained weights belong to a different model')
        source_tp=manifest['tp'];target_tp=get_tensor_model_parallel_world_size()
        target_rank=get_tensor_model_parallel_rank()
        if len(manifest['ranks'])!=source_tp or sorted(r['rank'] for r in manifest['ranks'])!=list(range(source_tp)):
            raise ValueError('retained cache lacks complete source rank coverage')
        specs=manifest['ranks'][0]['parameter_specs']
        if any(r['parameter_specs']!=specs for r in manifest['ranks']):
            raise ValueError('source ranks disagree on tensor partitioning')
        modules=dict(model.named_modules())
        parameters=dict(model.named_parameters())
        if set(parameters)!=set(specs):
            raise ValueError('retained model parameter set mismatch')
        stream=torch.cuda.Stream();pending=deque();copied=0
        started=time.time()
        with ExitStack() as stack, torch.no_grad(), torch.cuda.stream(stream):
            readers=[]
            for rank in manifest['ranks']:
                path=root/rank['file']
                if path.parent!=root or not path.is_file():
                    raise ValueError('invalid retained shard path')
                if file_sha256(path)!=rank['sha256']:
                    raise ValueError('retained source shard fingerprint changed')
                readers.append(stack.enter_context(safe_open(str(path),framework='pt',device='cpu')))
            for key,param in parameters.items():
                module_name,name=key.rsplit('.',1)
                spec=parameter_spec(modules[module_name],name,param,target_tp)
                if spec!=specs[key]:
                    raise ValueError(f'target partition mismatch: {key}')
                axis=spec['axis']
                spans=((0,0,0,param.shape[0]),) if axis is None else regions(spec['groups'],source_tp,target_tp,target_rank)
                for rank,src_offset,dst_offset,length in spans:
                    selected=readers[rank].get_slice(key)
                    source_shape=selected.get_shape()
                    if len(source_shape)!=param.ndim or any(source_shape[d]!=param.shape[d]
                        for d in range(param.ndim) if d!=axis):
                        raise ValueError(f'source shape mismatch: {key}')
                    if axis is not None and source_shape[axis]!=sum(spec['groups'])//source_tp:
                        raise ValueError(f'incomplete source tensor: {key}')
                    dim=0 if axis is None else axis
                    per_unit=param.numel()*param.element_size()//param.shape[dim]
                    step=max(1,(64*1024**2)//per_unit)
                    for offset in range(0,length,step):
                        count=min(step,length-offset)
                        slices=[slice(None)]*param.ndim
                        slices[dim]=slice(src_offset+offset,src_offset+offset+count)
                        cpu=selected[tuple(slices)].contiguous().pin_memory()
                        param.data.narrow(dim,dst_offset+offset,count).copy_(cpu,non_blocking=True)
                        copied+=cpu.numel()*cpu.element_size()
                        pending.append(cpu)
                        if len(pending)>=2:
                            stream.synchronize();pending.clear()
            stream.synchronize();pending.clear()
        torch.cuda.current_stream().wait_stream(stream)
        self.retained_load_evidence=dict(source_tp=source_tp,target_tp=target_tp,target_rank=target_rank,
            copied_bytes=copied,started_s=started,finished_s=time.time(),max_pinned_inflight_bytes=128*1024**2)
        (root/f'load-tp{target_tp}-rank{target_rank}-{time.time_ns()}.json').write_text(
            json.dumps(self.retained_load_evidence))
