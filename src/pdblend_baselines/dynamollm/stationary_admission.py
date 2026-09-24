"""Dynamo stationary layout/memory contracts, independent of serving policy."""
from __future__ import annotations

from copy import deepcopy
import math

from .stationary_ipc import digest,need


def target_bindings(plan,target_rank):
    need(plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'}),'tensor plan changed')
    need(type(target_rank) is int and 0<=target_rank<len(plan['target_gpu_uuids']),'target rank outside layout')
    rows=[]
    for name,shape in sorted(plan['target_shapes'].items()):
        pieces=[p for p in plan['pieces'] if p['target_rank']==target_rank and p['parameter']==name]
        need(pieces,'parameter lacks target pieces')
        same_owner=len({p['source_rank'] for p in pieces})==1
        local=all(p['kind']=='retain_on_gpu' for p in pieces)
        axis=pieces[0]['axis']
        offsets={p['source_offset']-p['target_offset'] for p in pieces}
        affine=local and same_owner and (axis is None or len(offsets)==1)
        rows.append(dict(parameter=name,target_shape=list(shape),axis=axis,pieces=deepcopy(pieces),
                         binding='original_storage_view' if affine else 'segmented_execution_required',
                         dense_weight_materialization_allowed=False,
                         local_logical_bytes=sum(p['bytes'] for p in pieces if p['kind']=='retain_on_gpu'),
                         remote_fragment_bytes=sum(p['bytes'] for p in pieces if p['kind']=='direct_gpu_transfer')))
    return dict(plan_sha256=plan['plan_sha256'],target_rank=target_rank,
        target_gpu_uuid=plan['target_gpu_uuids'][target_rank],parameters=rows,
        existing_dense_vllm_compatible=all(r['binding']=='original_storage_view' for r in rows),
        ready_for_serving=False,formal_eligible=False)


def memory_admission(*,free_bytes,target_remote_fragment_bytes,target_kv_bytes,target_workspace_bytes,
                     target_context_bytes,target_tp_communicator_bytes,guard_bytes,
                     pinned_allocation_bytes,logical_retained_bytes,source_kv_allocated_bytes):
    """Use measured CURRENT free memory, never credit a hypothetical KV release.

    Pinned source allocations and existing KV are already excluded from free.
    Listing them makes allocation retention auditable; do not subtract twice.
    All quantities must come from the same physical GPU's actual observation.
    This arithmetic is a CPU plan, not a hardware residency receipt.
    """
    values=locals().copy()
    need(all(type(v) is int and v>=0 for v in values.values()),'explicit nonnegative byte counts required')
    need(pinned_allocation_bytes>=logical_retained_bytes,'logical slices cannot exceed pinned allocator backing')
    required=sum((target_remote_fragment_bytes,target_kv_bytes,target_workspace_bytes,
                  target_context_bytes,target_tp_communicator_bytes,guard_bytes))
    return dict(**values,required_additional_bytes=required,headroom_bytes=free_bytes-required,
                capacity_feasible=free_bytes>=required,credited_source_kv_release_bytes=0,
                observed_source_kv_still_allocated=source_kv_allocated_bytes>0,
                hardware_verified=False,target_activation_authorized=False,formal_eligible=False)


def segmented_linear_reference(x,fragments,*,target_shape,axis,bias=None):
    """CPU semantic oracle; never used to claim a CUDA serving implementation.

    The fragments are original arrays/views; only activation outputs allocate.
    The row-partition sum has a different floating-point reduction order from
    a dense GEMM. A future CUDA layer therefore requires exact output goldens.
    """
    import numpy as np
    need(isinstance(x,np.ndarray) and x.ndim>=1 and len(target_shape)==2 and axis in (0,1),
         'explicit CPU linear reference shape required')
    need(x.shape[-1]==target_shape[1],'input dimension differs')
    rows=sorted(fragments,key=lambda row:row['target_offset']);cursor=0
    for row in rows:
        value=row['view'];length=value.shape[axis]
        need(row['target_offset']==cursor and value.ndim==2,'fragment gap/overlap')
        need(value.shape[1-axis]==target_shape[1-axis],'nonpartitioned dimension differs')
        cursor+=length
    need(cursor==target_shape[axis],'incomplete parameter coverage')
    out=np.zeros((*x.shape[:-1],target_shape[0]),dtype=np.result_type(x,*[r['view'] for r in rows]))
    for row in rows:
        start=row['target_offset'];weight=row['view']
        if axis==0:out[...,start:start+weight.shape[0]]=x@weight.T
        else:out+=x[...,start:start+weight.shape[1]]@weight.T
    if bias is not None:
        need(bias.shape==(target_shape[0],),'bias dimension differs');out+=bias
    return out


def require_serving_integration(bindings):
    """No caller-supplied boolean may turn a storage primitive into vLLM."""
    names=[r['parameter'] for r in bindings['parameters'] if r['binding']=='segmented_execution_required']
    raise RuntimeError('stationary serving not implemented: segmented linear/embedding binding, native KV release/rebuild '
                       'and target TP group activation/goldens are required; segmented parameters='+','.join(names))
