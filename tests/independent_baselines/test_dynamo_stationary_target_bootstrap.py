"""Actual metadata and IPC serialization contracts without CUDA operations."""
from copy import deepcopy
import json

import pytest

from pdblend_baselines.dynamollm.stationary_ipc import digest,process_identity
from pdblend_baselines.dynamollm.stationary_tensors import tensor_plan
from pdblend_baselines.dynamollm.stationary_target_bootstrap import TargetBootstrap,publish,packet_from_worker_receipt
from tests.independent_baselines.test_dynamo_stationary_owner_graph import bound
from tests.independent_baselines.test_dynamo_stationary_probe import probe
from tests.independent_baselines.test_dynamo_stationary_ipc import setup_owner,CONSUMER


def config(tmp_path):
    p=tensor_plan(source_gpus=['GPU-0'],target_gpus=['GPU-0'],source_shapes=probe.shapes(1),
        target_shapes=probe.shapes(1),geometry=probe.GEOMETRY)
    return dict(schema='dynamo-same-tp-target-bootstrap/v1',transaction_id='tx',plan=p,
        source_generation=0,target_generation=0,target_gpu_memory_utilization=.25,
        target_public_admission=False,rendezvous_dir=str(tmp_path/'exchange'),
        model_path='/models/Qwen2.5-7B-Instruct',target_source_sha256='a'*64,image_digest='image')


def test_atomic_publication_refuses_overwrite_and_exposes_complete_json(tmp_path):
    path=tmp_path/'x.json';publish(path,dict(original=True))
    with pytest.raises(FileExistsError):publish(path,dict(replaced=True))
    assert json.loads(path.read_text())==dict(original=True)
    assert list(tmp_path.iterdir())==[path]


@pytest.mark.parametrize('fault',['tp_change','epoch','public','memory'])
def test_bootstrap_rejects_unbound_topology_epoch_or_admission(tmp_path,fault):
    c=config(tmp_path)
    if fault=='tp_change':
        c['plan']['target_gpu_uuids']=['GPU-0','GPU-1']
        c['plan']['plan_sha256']=digest({k:v for k,v in c['plan'].items() if k!='plan_sha256'})
    elif fault=='epoch':c['source_generation']=True
    elif fault=='public':c['target_public_admission']=True
    else:c['target_gpu_memory_utilization']=0
    with pytest.raises(ValueError):TargetBootstrap(bound(tmp_path,'c.json',c))


def test_bootstrap_requires_real_ready_identity_and_bound_original_packet(tmp_path):
    c=config(tmp_path);ref=bound(tmp_path,'config.json',c);b=TargetBootstrap(ref)
    with pytest.raises(ValueError,match='publish'):b.wait_packet(0)
    b.publish_ready(gpu_uuid='GPU-0',device_index=0)
    with pytest.raises(ValueError,match='timed out'):b.wait_packet(0)
    packet=dict(consumer=process_identity(),target_rank=0,plan_sha256=c['plan']['plan_sha256'],generation=0,gpu_uuid='GPU-0')
    packet['packet_sha256']=digest(packet)
    pref=bound(tmp_path,'packet.json',packet)
    publish(b.directory/'target-rank-0-packet.json',dict(bootstrap_ref=ref,consumer=process_identity(),packet_ref=pref))
    assert b.wait_packet(0)==packet


def test_real_owner_packet_survives_collective_rpc_wrapper_without_forging_its_hash():
    _,_,_,_,owner=setup_owner();packet=owner.export(CONSUMER,target_rank=0)
    wrapper=dict(packet,rank=0,transaction_id='tx',serving_qualified=False,formal_eligible=False)
    assert packet_from_worker_receipt(wrapper)==packet
    wrapper['source_storage']=deepcopy(wrapper['source_storage'])
    next(iter(wrapper['source_storage'].values()))['storage_ptr']+=2
    with pytest.raises(ValueError,match='mutated'):packet_from_worker_receipt(wrapper)
