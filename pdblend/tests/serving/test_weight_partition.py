import importlib.util
from pathlib import Path
import torch
import pytest

path=Path('/root/workspace/vllm-pd-fork/vllm/pdblend_weights.py')
spec=importlib.util.spec_from_file_location('pdblend_weights_test',path)
weights=importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)


def shard(full,groups,tp,rank,axis):
    parts=[];offset=0
    for size in groups:
        parts.append(full.narrow(axis,offset+rank*size//tp,size//tp))
        offset+=size
    return torch.cat(parts,dim=axis)


@pytest.mark.parametrize('axis,groups',[(0,[40,8,8]),(0,[32,32]),(1,[64]),(0,[128])])
def test_packed_weight_repartition_matches_independent_global_reference(axis,groups):
    shape=[sum(groups),3] if axis==0 else [3,sum(groups)]
    full=torch.arange(shape[0]*shape[1]).reshape(shape)
    for source_tp in (1,2,4,8):
        sources=[shard(full,groups,source_tp,r,axis) for r in range(source_tp)]
        for target_tp in (1,2,4,8):
            for rank in range(target_tp):
                expected=shard(full,groups,target_tp,rank,axis)
                actual=torch.full_like(expected,-1)
                for source,lo,dest,n in weights.regions(groups,source_tp,target_tp,rank):
                    actual.narrow(axis,dest,n).copy_(sources[source].narrow(axis,lo,n))
                assert torch.equal(actual,expected)


def test_repartition_rejects_unsupported_head_replication_and_rank():
    with pytest.raises(ValueError): weights.regions([4],4,8,0)
    with pytest.raises(ValueError): weights.regions([8],1,2,2)
    with pytest.raises(ValueError): weights.regions([9],1,3,0)
