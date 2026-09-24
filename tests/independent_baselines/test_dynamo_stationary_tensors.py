"""Exact reconstructed weights and genuine CPU aliasing; no GPU qualification."""
from types import SimpleNamespace

import numpy as np
import pytest

from pdblend_baselines.dynamollm.stationary_tensors import tensor_plan, RetainedStorageLease


GEOMETRY=dict(hidden_size=16,num_attention_heads=8,num_key_value_heads=4,intermediate_size=32)


def shapes(tp):
    return {'model.layers.0.self_attn.qkv_proj.weight': [32//tp,16],
            'model.layers.0.self_attn.qkv_proj.bias': [32//tp],
            'model.layers.0.mlp.gate_up_proj.weight': [64//tp,16],
            'model.layers.0.self_attn.o_proj.weight': [16,16//tp],
            'model.layers.0.mlp.down_proj.weight': [16,32//tp],
            'model.embed_tokens.weight': [32//tp,16], 'model.norm.weight': [16]}


def shard(name, tp, rank):
    full=np.arange(np.prod(shapes(1)[name]),dtype=np.uint16).reshape(shapes(1)[name])
    # Independent model constructor, without the planner's segment helper.
    if name.endswith('norm.weight'):return full.copy()
    if '.qkv_proj.' in name:
        return np.concatenate([np.split(part,tp,axis=0)[rank]
                               for part in np.split(full,[16,24],axis=0)],axis=0).copy()
    if '.gate_up_proj.' in name:
        return np.concatenate([np.split(part,tp,axis=0)[rank]
                               for part in np.split(full,2,axis=0)],axis=0).copy()
    axis=1 if name.endswith(('o_proj.weight','down_proj.weight')) else 0
    return np.split(full,tp,axis=axis)[rank].copy()


def narrow(value,axis,offset,length):
    if axis is None:return value
    indices=[slice(None)]*value.ndim;indices[axis]=slice(offset,offset+length)
    return value[tuple(indices)]


@pytest.mark.parametrize('source,target', [
    (['GPU-0'],['GPU-0','GPU-1']),
    (['GPU-0','GPU-1'],['GPU-1']),
    (['GPU-0','GPU-1','GPU-2','GPU-3'],['GPU-0','GPU-2']),
    (['GPU-0','GPU-1'],['GPU-0','GPU-2','GPU-1','GPU-3']),
    (['GPU-0','GPU-1'],['GPU-2','GPU-3'])])
def test_exact_qkv_gate_up_row_vocab_and_norm_reconstruct_all_target_weights(source,target):
    plan=tensor_plan(source_gpus=source,target_gpus=target,source_shapes=shapes(len(source)),
                     target_shapes=shapes(len(target)),geometry=GEOMETRY)
    actual={(tr,name):np.zeros(shape,dtype=np.uint16) for tr in range(len(target))
            for name,shape in shapes(len(target)).items()}
    for piece in plan['pieces']:
        old=shard(piece['parameter'],len(source),piece['source_rank'])
        src=narrow(old,piece['axis'],piece['source_offset'],piece['length'])
        dst=narrow(actual[(piece['target_rank'],piece['parameter'])],
                   piece['axis'],piece['target_offset'],piece['length'])
        assert src.size*2 == piece['bytes']
        dst[...] = src
        assert (piece['kind'] == 'retain_on_gpu') is (piece['source_gpu_uuid'] == piece['target_gpu_uuid'])
    for (rank,name),value in actual.items():
        np.testing.assert_array_equal(value,shard(name,len(target),rank))
    assert sum(a.nbytes for a in actual.values()) == plan['planned_target_bytes']
    assert not plan['executed'] and not plan['original_weight_retention_implemented']
    assert bool(plan['planned_retained_bytes']) is bool(set(source)&set(target))


def test_replicated_norm_is_retained_from_its_actual_local_rank():
    plan=tensor_plan(source_gpus=['GPU-0','GPU-1'],target_gpus=['GPU-1'],
                     source_shapes=shapes(2),target_shapes=shapes(1),geometry=GEOMETRY)
    norms=[p for p in plan['pieces'] if p['parameter'] == 'model.norm.weight']
    assert len(norms) == 1 and norms[0]['source_rank'] == 1 and norms[0]['kind'] == 'retain_on_gpu'


class CpuAliasTensor:
    """Exercise object/storage lifetime; the production lease still reports unqualified."""
    is_cuda=True
    dtype='torch.bfloat16'
    device='cuda:0'
    def __init__(self,array,root=None):
        self.array=array
        self.root=root or self
        if root is None:self.version=0
    @property
    def _version(self):return self.root.version
    @property
    def shape(self):return self.array.shape
    def stride(self):return tuple(n//2 for n in self.array.strides)
    def storage_offset(self):return (self.data_ptr()-self.root.data_ptr())//2
    def data_ptr(self):return self.array.__array_interface__['data'][0]
    def untyped_storage(self):return SimpleNamespace(data_ptr=self.root.data_ptr)
    def element_size(self):return 2
    def narrow(self,axis,offset,length):return CpuAliasTensor(narrow(self.array,axis,offset,length),self.root)


def lease_fixture():
    plan=tensor_plan(source_gpus=['GPU-0'],target_gpus=['GPU-0','GPU-1'],
                     source_shapes=shapes(1),target_shapes=shapes(2),geometry=GEOMETRY)
    parameters={name:CpuAliasTensor(shard(name,1,0)) for name in shapes(1)}
    lease=RetainedStorageLease(plan,source_rank=0,parameters=parameters,gpu_uuid='GPU-0',
                               generation=3,expected_generation=3)
    return plan,parameters,lease


def test_retained_storage_uses_aliases_and_cannot_claim_engine_activation():
    plan,parameters,lease=lease_fixture()
    for piece,view in lease.views:
        assert np.shares_memory(view.array,parameters[piece['parameter']].array)
    receipt=lease.receipt()
    assert receipt['held_view_bytes'] == plan['planned_retained_bytes']
    assert not receipt['target_parameter_binding_ready'] and not receipt['target_engine_activated']
    assert not receipt['hardware_qualified'] and not receipt['original_weight_retention_implemented']
    assert lease.release()['released']
    assert not lease.views and not lease.owner
    with pytest.raises(RuntimeError,match='released'):lease.receipt()


@pytest.mark.parametrize('fault', ['generation','gpu','plan','device','shape'])
def test_owner_mismatch_or_non_cuda_parameter_is_refused(fault):
    plan,parameters,lease=lease_fixture();lease.release()
    generation,expected,gpu=3,3,'GPU-0'
    if fault == 'generation':expected=4
    elif fault == 'gpu':gpu='GPU-1'
    elif fault == 'plan':plan['planned_retained_bytes'] += 2
    elif fault == 'device':next(iter(parameters.values())).is_cuda=False
    elif fault == 'shape':next(iter(parameters.values())).array=np.ones(1,dtype=np.uint16)
    with pytest.raises(ValueError):
        RetainedStorageLease(plan,source_rank=0,parameters=parameters,gpu_uuid=gpu,
                             generation=generation,expected_generation=expected)


@pytest.mark.parametrize('fault', ['replacement','mutation'])
def test_changed_source_allocation_or_version_cannot_produce_clean_receipt(fault):
    _,parameters,lease=lease_fixture();name=next(iter(parameters))
    if fault == 'replacement':parameters[name]=CpuAliasTensor(parameters[name].array.copy())
    else:parameters[name].version += 1
    with pytest.raises(RuntimeError,match='storage/value version changed'):lease.receipt()
