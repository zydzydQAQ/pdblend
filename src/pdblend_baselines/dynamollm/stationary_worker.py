"""Dynamo-specific owner-worker extension. No shared NativeWorker edits.

An isolated server must prove scheduler drain before pinning. This extension
can export actual existing CUDA parameter views, but never activates a target.
"""
from __future__ import annotations

import os
import time

from .gpu_weights import DynamoWorkerExtension,_metadata
from .stationary_ipc import StationaryOwner,TorchCudaIpcCodec,need
from pdblend.online.native_control import validate_state


class DynamoStationaryWorkerExtension(DynamoWorkerExtension):
    def dynamo_operation(self,operation,payload=None):
        held=[key for key,value in getattr(self,'_dynamo_stationary_owners',{}).items()
              if not value.closed or value.quarantined]
        need(not held or operation in ('describe','drain'),
             'stationary storage is pinned/quarantined; weight writes and topology changes are blocked')
        result=super().dynamo_operation(operation,payload)
        if operation=='drain':
            result['stationary_owner_transactions']=held
            result['active_weight_sessions']+=len(held)
            result['drained']=result['drained'] and not held
        return result

    def dynamo_stationary_operation(self,operation,payload=None):
        import torch
        payload=dict(payload or {})
        generation=getattr(self,'_native_generation',int(os.environ.get('DYNAMO_GENERATION','0')))
        need(type(generation) is int and type(payload.get('expected_generation')) is int
             and payload['expected_generation']==generation,
             'stationary worker actual generation differs')
        need(self.parallel_config.pipeline_parallel_size==1
             and getattr(self.parallel_config,'data_parallel_size',1)==1,'stationary owner requires PP1 and DP1')
        owners=getattr(self,'_dynamo_stationary_owners',None)
        if owners is None:owners={};self._dynamo_stationary_owners=owners
        if operation=='describe':
            parameters,shapes,geometry=_metadata(self)
            free,total=torch.cuda.mem_get_info()
            return dict(rank=self.rank,generation=generation,shapes=shapes,geometry=geometry,
                model_weight_bytes=sum(p.numel()*p.element_size() for p in parameters.values()),
                observed_free_bytes=free,total_bytes=total,allocator_reserved_bytes=torch.cuda.memory_reserved(),
                allocator_allocated_bytes=torch.cuda.memory_allocated(),at_s=time.time(),
                source_kv_release_implemented=False,target_binding_implemented=False,
                original_weight_retention_implemented=False,formal_eligible=False)
        key=payload.get('transaction_id')
        need(isinstance(key,str) and 0<len(key)<=128,'bounded stationary transaction required')
        if operation=='pin':
            need(key not in owners and not any(not o.closed or o.quarantined for o in owners.values()),
                 'one clean owner transition at a time; quarantined owner must exit')
            drained=payload['native_scheduler_drain']
            validate_state(drained,generation=generation,tp=self.parallel_config.tensor_parallel_size,
                           pp=1,drained=True,observed_after_s=time.time()-.5)
            need(drained.get('accepting') is False and drained.get('acknowledged') is True
                 and drained.get('drained') is True,'source owner must be natively drained and closed')
            plan=payload['plan'];parameters,shapes,geometry=_metadata(self)
            need(plan['source_shapes']==shapes and plan['geometry']==geometry,'stationary plan differs from actual model')
            device=torch.cuda.current_device();gpu=plan['source_gpu_uuids'][self.rank]
            codec=TorchCudaIpcCodec(gpu_uuid=gpu,device_index=device)
            owners[key]=StationaryOwner(plan,source_rank=self.rank,parameters=parameters,gpu_uuid=gpu,
                                        generation=generation,codec=codec)
            result=owners[key].lease.receipt()
        elif operation=='export':
            need(key in owners,'source views were not pinned')
            target=payload['target_rank'];owner=owners[key]
            if owner.plan['target_gpu_uuids'][target]!=owner.codec.uuid:
                result=dict(participating=False,reason='no retained storage on target physical GPU')
            else:
                result=owner.export(payload['consumer'],target_rank=target)
        elif operation=='consumer_release_ack':
            need(type(payload.get('source_rank')) is int
                 and 0<=payload['source_rank']<self.parallel_config.tensor_parallel_size,'explicit source rank required')
            if payload['source_rank']!=self.rank:
                return dict(rank=self.rank,generation=generation,participating=False,formal_eligible=False)
            need(key in owners,'unknown stationary owner')
            owners[key].acknowledge_release(payload['receipt']);result=dict(acknowledged=True)
        elif operation=='release':
            need(type(payload.get('source_rank')) is int
                 and 0<=payload['source_rank']<self.parallel_config.tensor_parallel_size,'explicit source rank required')
            if payload['source_rank']!=self.rank:
                return dict(rank=self.rank,generation=generation,participating=False,formal_eligible=False)
            need(key in owners,'unknown stationary owner')
            result=owners[key].release(consumer_processes_gone=payload['consumer_processes_gone'])
        elif operation in ('release_kv','bind_target','activate'):
            raise RuntimeError('native KV-release/dense-to-segmented target serving integration is not implemented')
        else:raise ValueError('unknown stationary worker operation')
        return dict(result,rank=self.rank,generation=generation,transaction_id=key,
                    serving_qualified=False,formal_eligible=False)
