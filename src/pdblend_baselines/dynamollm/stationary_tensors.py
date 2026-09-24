"""Exact Qwen shard geometry and a same-worker retained-storage primitive.

This consumes real parameter shapes instead of treating every weight as one
uniform shard. It is deliberately disconnected from the production relay:
retained views are not an activated target vLLM engine. A future independent
worker handoff must preserve the owner lifetime, bind target parameters, rebuild
TP execution groups, and pass ordinary output/drain tests before activation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from .gpu_weights import transfer_pieces


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_plan(*, source_gpus, target_gpus, source_shapes, target_shapes, geometry):
    """Translate an already matched rank placement into exact BF16 byte ranges.

    Each layout is one complete model instance. Replicated norms may use any
    resident rank on the target GPU; the old copy-only transport's rank-zero
    donor convention must not erase their actual local residency.
    """
    source_gpus, target_gpus = list(source_gpus), list(target_gpus)
    for layout in (source_gpus, target_gpus):
        if (len(layout) not in (1,2,4) or len(set(layout)) != len(layout)
                or any(not isinstance(g, str) or not g.startswith('GPU-') for g in layout)):
            raise ValueError('explicit unique physical UUIDs and legal Qwen TP1/2/4 required')
    if not source_shapes or set(source_shapes) != set(target_shapes):
        raise ValueError('same complete parameter inventory required')
    for shapes in (source_shapes,target_shapes):
        if any(not shape or any(type(n) is not int or n < 1 for n in shape) for shape in shapes.values()):
            raise ValueError('positive integer parameter dimensions required')
    pieces=[]
    for tr,target_gpu in enumerate(target_gpus):
        for name,target_shape in sorted(target_shapes.items()):
            source_shape=source_shapes[name]
            probe=transfer_pieces(name,source_shape,target_shape,len(source_gpus),0,
                                  len(target_gpus),tr,geometry)
            replicated=bool(probe and probe[0]['axis'] is None)
            donors=list(range(len(source_gpus)))
            if replicated:
                donors=[source_gpus.index(target_gpu) if target_gpu in source_gpus else 0]
            parameter=[]
            for sr in donors:
                entries=transfer_pieces(name,source_shape,target_shape,len(source_gpus),
                    0 if replicated else sr,len(target_gpus),tr,geometry)
                for piece in entries:
                    axis=piece['axis']
                    elements=piece['length'] if axis is None else piece['length']*math.prod(
                        n for index,n in enumerate(target_shape) if index != axis)
                    parameter.append(dict(parameter=name,source_rank=sr,target_rank=tr,
                        source_gpu_uuid=source_gpus[sr],target_gpu_uuid=target_gpu,
                        kind='retain_on_gpu' if source_gpus[sr] == target_gpu else 'direct_gpu_transfer',
                        source_shape=list(source_shape),target_shape=list(target_shape),
                        elements=elements,bytes=elements*2,**piece))
            if sum(p['elements'] for p in parameter) != math.prod(target_shape):
                raise ValueError('target parameter is not covered exactly: '+name)
            if not replicated:
                cursor=0
                for p in sorted(parameter,key=lambda row:row['target_offset']):
                    if p['target_offset'] != cursor:
                        raise ValueError('overlapping or missing target slice: '+name)
                    cursor += p['length']
                if cursor != target_shape[parameter[0]['axis']]:
                    raise ValueError('incomplete target axis: '+name)
            pieces.extend(parameter)
    retained=sum(p['bytes'] for p in pieces if p['kind'] == 'retain_on_gpu')
    transferred=sum(p['bytes'] for p in pieces if p['kind'] == 'direct_gpu_transfer')
    result=dict(schema='dynamo-exact-stationary-tensor-plan/v1',source_gpu_uuids=source_gpus,
        target_gpu_uuids=target_gpus,source_shapes=deepcopy(source_shapes),target_shapes=deepcopy(target_shapes),
        geometry=deepcopy(geometry),pieces=pieces,planned_retained_bytes=retained,
        planned_transfer_bytes=transferred,planned_target_bytes=retained+transferred,
        executed=False,formal_eligible=False,original_weight_retention_implemented=False)
    result['plan_sha256']=_digest(result)
    return result


def _tensor_identity(tensor):
    return dict(object_id=id(tensor),device=str(tensor.device),shape=list(tensor.shape),
        stride=list(tensor.stride()),storage_offset=tensor.storage_offset(),data_ptr=tensor.data_ptr(),
        storage_ptr=tensor.untyped_storage().data_ptr(),version=tensor._version)


class RetainedStorageLease:
    """Keep original CUDA parameter storage alive using views, with no copy.

    This implements only ownership inside the source worker. It never exports
    CUDA IPC handles or changes a live vLLM parameter. Fused QKV/gate-up slices
    remain separate views; concatenating them would allocate new storage and is
    not silently treated as retaining the same allocation.
    """
    def __init__(self, plan, *, source_rank, parameters, gpu_uuid, generation, expected_generation):
        if (type(generation) is not int or generation < 0 or generation != expected_generation
                or type(source_rank) is not int or not 0 <= source_rank < len(plan['source_gpu_uuids'])
                or plan['source_gpu_uuids'][source_rank] != gpu_uuid):
            raise ValueError('resident owner physical GPU/rank/generation differs')
        payload={k:v for k,v in plan.items() if k != 'plan_sha256'}
        if _digest(payload) != plan.get('plan_sha256'):
            raise ValueError('stationary tensor plan changed')
        if set(parameters) != set(plan['source_shapes']):
            raise ValueError('actual parameter inventory differs from plan')
        self.owner=parameters
        self.plan_sha256,self.source_rank,self.gpu_uuid,self.generation=plan['plan_sha256'],source_rank,gpu_uuid,generation
        self.before={};self.views=[];self.closed=False
        selected=[p for p in plan['pieces'] if p['source_rank'] == source_rank and p['kind'] == 'retain_on_gpu']
        for piece in selected:
            tensor=parameters[piece['parameter']]
            if (getattr(tensor,'is_cuda',False) is not True or str(tensor.dtype) != 'torch.bfloat16'
                    or list(tensor.shape) != piece['source_shape'] or tensor.element_size() != 2):
                raise ValueError('actual CUDA BF16 source parameter differs')
            self.before.setdefault(piece['parameter'],_tensor_identity(tensor))
            view=tensor if piece['axis'] is None else tensor.narrow(
                piece['axis'],piece['source_offset'],piece['length'])
            if view.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr():
                raise RuntimeError('retained slice allocated another storage')
            self.views.append((deepcopy(piece),view))

    def receipt(self):
        if self.closed:
            raise RuntimeError('retained storage lease already released')
        for name,before in self.before.items():
            if _tensor_identity(self.owner[name]) != before:
                raise RuntimeError('retained source parameter storage/value version changed: '+name)
        return dict(schema='dynamo-retained-storage-lease/v1',plan_sha256=self.plan_sha256,
            source_rank=self.source_rank,gpu_uuid=self.gpu_uuid,generation=self.generation,
            views=[dict(piece=p,view=_tensor_identity(v)) for p,v in self.views],
            held_view_bytes=sum(p['bytes'] for p,_ in self.views),source_storage=self.before,
            path='same-worker CUDA views; no tensor copy or host weight staging',
            target_parameter_binding_ready=False,target_engine_activated=False,
            hardware_qualified=False,formal_eligible=False,original_weight_retention_implemented=False)

    def release(self):
        receipt=self.receipt()
        self.views.clear();self.owner={};self.closed=True
        return dict(receipt,released=True)
