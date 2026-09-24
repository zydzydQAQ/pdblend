"""CPU lifecycle/geometry tests; CUDA IPC and vLLM activation stay unqualified."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pdblend_baselines.dynamollm.stationary_admission import (
    memory_admission, require_serving_integration, segmented_linear_reference, target_bindings,
)
from pdblend_baselines.dynamollm.stationary_ipc import (
    StationaryOwner, StationaryConsumer, digest, process_identity, process_matches, validate_descriptor,
)
from pdblend_baselines.dynamollm.stationary_tensors import tensor_plan
from tests.independent_baselines.test_dynamo_stationary_tensors import GEOMETRY, CpuAliasTensor, narrow, shapes, shard


OWNER=dict(pid=101,start_ticks=1,boot_id='cpu-test')
CONSUMER=dict(pid=102,start_ticks=2,boot_id='cpu-test')


class AliasCodec:
    """Keep NumPy aliases, never pretend these are hardware IPC receipts."""
    uuid='GPU-0'
    def __init__(self):self.views={};self.syncs=0;self.collects=0
    def synchronize(self):self.syncs+=1
    def collect(self):self.collects+=1
    def export_view(self,view):
        key=str(len(self.views));self.views[key]=view
        return dict(schema='dynamo-cuda-ipc-descriptor/v1',gpu_uuid=self.uuid,dtype='bfloat16',
            shape=list(view.shape),stride=list(view.stride()),tensor_offset=view.storage_offset(),
            storage_size_bytes=view.root.array.nbytes,storage_offset_bytes=0,
            allocation_bytes=view.root.array.nbytes,logical_bytes=view.array.nbytes,
            source_storage_ptr=view.root.data_ptr(),allocation_identity=str(view.root.data_ptr()),handle=key)
    def import_view(self,descriptor):return self.views[descriptor['handle']]


def setup_owner():
    plan=tensor_plan(source_gpus=['GPU-0'],target_gpus=['GPU-0','GPU-1'],
                     source_shapes=shapes(1),target_shapes=shapes(2),geometry=GEOMETRY)
    parameters={name:CpuAliasTensor(shard(name,1,0)) for name in shapes(1)}
    alive={digest(OWNER),digest(CONSUMER)};codec=AliasCodec()
    owner=StationaryOwner(plan,source_rank=0,parameters=parameters,gpu_uuid='GPU-0',
                          generation=3,codec=codec,owner_identity=OWNER,
                          process_alive=lambda identity:digest(identity) in alive)
    return plan,parameters,alive,codec,owner


def import_consumer(plan,codec,packet,alive,**kwargs):
    return StationaryConsumer(packet,plan=plan,codec=codec,expected_generation=kwargs.pop('generation',3),
        consumer_identity=CONSUMER,process_alive=lambda identity:digest(identity) in alive,**kwargs)


def test_original_views_are_shared_and_owner_outlives_consumer_even_after_clean_ack():
    plan,parameters,alive,codec,owner=setup_owner()
    packet=owner.export(CONSUMER,target_rank=0)
    consumer=import_consumer(plan,codec,packet,alive)
    for piece,view in consumer.views:
        assert np.shares_memory(view.array,parameters[piece['parameter']].array)
        expected=narrow(shard(piece['parameter'],2,0),piece['axis'],piece['target_offset'],piece['length'])
        np.testing.assert_array_equal(view.array,expected)
    receipt=consumer.receipt()
    assert receipt['tensor_data_copied_bytes']==receipt['host_weight_staging_bytes']==0
    assert not receipt['target_engine_activated'] and not receipt['formal_eligible']
    with pytest.raises(RuntimeError,match='do not supply'):consumer.require_inactive()
    owner.acknowledge_release(consumer.close())
    with pytest.raises(ValueError,match='outlive'):owner.release(consumer_processes_gone=[CONSUMER])
    alive.remove(digest(CONSUMER))
    result=owner.release(consumer_processes_gone=[CONSUMER])
    assert result['clean_consumer_release'] and not result['owner_exit_required']
    assert result['pinned_cuda_allocations_bytes']==sum(p.array.nbytes for p in parameters.values())
    assert not result['source_gpu_memory_reclaimed'] and not result['formal_eligible']
    assert owner.closed and not owner.quarantined


def test_duplicate_export_wrong_gpu_or_dead_consumer_is_rejected_before_refcounts_change():
    _,_,alive,codec,owner=setup_owner()
    with pytest.raises(ValueError,match='physical GPU'):owner.export(CONSUMER,target_rank=1)
    with pytest.raises(ValueError):owner.export(CONSUMER,target_rank=-1)
    owner.export(CONSUMER,target_rank=0);count=len(codec.views)
    with pytest.raises(ValueError,match='already issued'):owner.export(CONSUMER,target_rank=0)
    alive.remove(digest(CONSUMER))
    with pytest.raises(ValueError,match='live owned'):owner.export(CONSUMER,target_rank=0)
    assert len(codec.views)==count


def test_consumer_crash_quarantines_owner_without_claiming_cuda_allocation_reclaimed():
    plan,_,alive,codec,owner=setup_owner();packet=owner.export(CONSUMER,target_rank=0)
    import_consumer(plan,codec,packet,alive)
    alive.remove(digest(CONSUMER))
    result=owner.release(consumer_processes_gone=[CONSUMER])
    assert result['owner_exit_required'] and owner.quarantined
    assert not result['clean_consumer_release'] and not result['source_gpu_memory_reclaimed']
    with pytest.raises(ValueError,match='not exportable'):owner.export(CONSUMER,target_rank=0)


def test_partial_export_is_pinned_and_requires_consumer_isolation(monkeypatch):
    _,_,alive,codec,owner=setup_owner();real=codec.export_view
    def export(view):
        if codec.views:raise RuntimeError('allocator/refcounter failure')
        return real(view)
    monkeypatch.setattr(codec,'export_view',export)
    with pytest.raises(RuntimeError,match='allocator'):owner.export(CONSUMER,target_rank=0)
    assert owner.quarantined and not owner.closed and owner.lease.views
    with pytest.raises(ValueError,match='outlive'):owner.release(consumer_processes_gone=[])
    alive.remove(digest(CONSUMER))
    assert owner.release(consumer_processes_gone=[CONSUMER])['owner_exit_required']


def test_partial_import_collects_views_without_synthesizing_release_ack(monkeypatch):
    plan,_,alive,codec,owner=setup_owner();packet=owner.export(CONSUMER,target_rank=0)
    calls=[];real=codec.import_view
    def load(descriptor):
        calls.append(descriptor)
        if len(calls)==2:raise RuntimeError('CUDA import failed')
        return real(descriptor)
    monkeypatch.setattr(codec,'import_view',load)
    with pytest.raises(RuntimeError,match='import failed'):import_consumer(plan,codec,packet,alive)
    assert codec.collects==1 and not owner.closed
    assert all(v['release_ack'] is None for v in owner.exports.values())


@pytest.mark.parametrize('fault',['packet','plan','generation','owner','consumer','piece','view_shape','view_bounds','gpu','wrong_shard'])
def test_identity_or_geometry_mismatch_is_rejected_before_import(fault,monkeypatch):
    plan,_,alive,codec,owner=setup_owner();packet=owner.export(CONSUMER,target_rank=0)
    generation=3
    if fault=='packet':packet['generation']+=1
    elif fault=='plan':plan['planned_retained_bytes']+=2
    elif fault=='generation':generation=4
    elif fault=='owner':alive.remove(digest(OWNER))
    elif fault=='consumer':packet['consumer']['start_ticks']+=1
    elif fault=='piece':packet['views'][0]['piece']['target_offset']+=1
    elif fault=='view_shape':packet['views'][0]['descriptor']['shape'][0]+=1
    elif fault=='view_bounds':packet['views'][0]['descriptor']['tensor_offset']=10**9
    elif fault=='gpu':packet['views'][0]['descriptor']['gpu_uuid']='GPU-1'
    elif fault=='wrong_shard':packet['views'][0]['descriptor']['stride']=[0]*len(packet['views'][0]['descriptor']['shape'])
    if fault!='packet':packet['packet_sha256']=digest({k:v for k,v in packet.items() if k!='packet_sha256'})
    seen=[];monkeypatch.setattr(codec,'import_view',lambda d:seen.append(d))
    with pytest.raises(ValueError):import_consumer(plan,codec,packet,alive,generation=generation)
    assert not seen


def test_owner_death_or_parameter_mutation_cannot_produce_clean_receipt():
    plan,parameters,alive,codec,owner=setup_owner();packet=owner.export(CONSUMER,target_rank=0)
    consumer=import_consumer(plan,codec,packet,alive);alive.remove(digest(OWNER))
    with pytest.raises(ValueError,match='owner is absent'):consumer.receipt()
    next(iter(parameters.values())).version+=1
    with pytest.raises(RuntimeError,match='version changed'):owner.lease.receipt()


def test_release_ack_binds_exact_process_and_packet():
    plan,_,alive,codec,owner=setup_owner();packet=owner.export(CONSUMER,target_rank=0)
    ack=import_consumer(plan,codec,packet,alive).close();ack['consumer']=dict(CONSUMER,start_ticks=100)
    with pytest.raises(ValueError,match='identity'):owner.acknowledge_release(ack)


def test_process_identity_uses_boot_and_start_ticks_not_pid_alone(monkeypatch):
    identity=process_identity();assert process_matches(identity)
    assert not process_matches(dict(identity,start_ticks=identity['start_ticks']+1))
    assert not process_matches({})


def test_fused_qkv_and_gate_up_require_segmented_binding_even_when_all_target_bytes_local():
    plan,_,_,_,_=setup_owner();bindings=target_bindings(plan,0)
    byname={row['parameter']:row for row in bindings['parameters']}
    assert byname['model.layers.0.self_attn.qkv_proj.weight']['binding']=='segmented_execution_required'
    assert byname['model.layers.0.mlp.gate_up_proj.weight']['binding']=='segmented_execution_required'
    assert byname['model.layers.0.self_attn.o_proj.weight']['binding']=='original_storage_view'
    assert not bindings['existing_dense_vllm_compatible']
    with pytest.raises(RuntimeError,match='not implemented'):require_serving_integration(bindings)


@pytest.mark.parametrize('source_tp,target_tp',[(1,2),(2,1),(2,4),(4,2)])
def test_segmented_qkv_gate_up_and_row_math_matches_independent_dense_target(source_tp,target_tp):
    source=[f'GPU-{i}' for i in range(source_tp)];target=[f'GPU-{i}' for i in range(target_tp)]
    plan=tensor_plan(source_gpus=source,target_gpus=target,source_shapes=shapes(source_tp),
                     target_shapes=shapes(target_tp),geometry=GEOMETRY)
    for rank in range(target_tp):
        for name,shape in shapes(target_tp).items():
            if len(shape)!=2:continue
            rows=[p for p in plan['pieces'] if p['parameter']==name and p['target_rank']==rank]
            arrays={r:shard(name,source_tp,r).astype(np.float64) for r in range(source_tp)}
            fragments=[dict(target_offset=p['target_offset'],
                view=narrow(arrays[p['source_rank']],p['axis'],p['source_offset'],p['length'])) for p in rows]
            x=np.arange(2*shape[1],dtype=np.float64).reshape(2,shape[1])
            got=segmented_linear_reference(x,fragments,target_shape=shape,axis=rows[0]['axis'])
            np.testing.assert_array_equal(got,x@shard(name,target_tp,rank).astype(np.float64).T)
            assert all(np.shares_memory(fragment['view'],arrays[p['source_rank']])
                       for p,fragment in zip(rows,fragments))


def test_memory_budget_does_not_credit_hypothetical_source_kv_release():
    values=dict(free_bytes=100,target_remote_fragment_bytes=20,target_kv_bytes=60,
                target_workspace_bytes=10,target_context_bytes=10,target_tp_communicator_bytes=10,
                guard_bytes=5,pinned_allocation_bytes=900,logical_retained_bytes=200,
                source_kv_allocated_bytes=1000)
    receipt=memory_admission(**values)
    assert receipt['required_additional_bytes']==115 and receipt['headroom_bytes']==-15
    assert not receipt['capacity_feasible'] and receipt['credited_source_kv_release_bytes']==0
    assert not receipt['hardware_verified'] and not receipt['target_activation_authorized']
    with pytest.raises(ValueError,match='backing'):memory_admission(**dict(values,pinned_allocation_bytes=1))


def test_segmented_math_rejects_gap_and_overlap():
    view=np.ones((2,4));x=np.ones((1,4))
    for offset in (1,-1):
        with pytest.raises(ValueError,match='gap/overlap'):
            segmented_linear_reference(x,[dict(view=view,target_offset=offset)],target_shape=(2,4),axis=0)


def native_drain(generation=3,tp=1):
    import time
    return dict(generation=generation,tp=tp,pp=1,native_evidence_complete=True,transport_healthy=True,
        native_at_s=time.time(),ranks=[dict(rank=r,generation=generation,native_evidence_complete=True,
            healthy=True,pending_transfers=0,transfer_allocations={}) for r in range(tp)],
        all_queue=[],running=[],waiting=[],retained_kv_requests=[],pending_transfers=0,
        transfer_allocations={},kv_allocations={},total_blocks=100,free_blocks=100,reserved_blocks=0,
        accepting=False,acknowledged=True,drained=True)


def worker_fixture(monkeypatch):
    import sys
    from pdblend_baselines.dynamollm import stationary_worker as module
    _,parameters,_,_,_=setup_owner()
    worker=module.DynamoStationaryWorkerExtension()
    worker.rank=0;worker._native_generation=3
    worker.parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1)
    worker.model_config=SimpleNamespace(hf_config=SimpleNamespace(**GEOMETRY))
    worker.model_runner=SimpleNamespace(model=SimpleNamespace(named_parameters=lambda:parameters.items()))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(current_device=lambda:0)))
    monkeypatch.setattr(module,'TorchCudaIpcCodec',lambda **kwargs:AliasCodec())
    return worker


@pytest.mark.parametrize('fault',['busy','stale','rank','generation','admission','dp'])
def test_worker_uses_real_native_validator_and_actual_generation_before_pinning(monkeypatch,fault):
    from pdblend_baselines.dynamollm.native_state import NativeControlError
    plan,_,_,_,_=setup_owner();worker=worker_fixture(monkeypatch);drain=native_drain()
    expected=3
    if fault=='busy':drain['all_queue']=['still-serving']
    elif fault=='stale':drain['native_at_s']-=10
    elif fault=='rank':drain['ranks']=[]
    elif fault=='generation':expected=2
    elif fault=='admission':drain['accepting']=True
    elif fault=='dp':worker.parallel_config.data_parallel_size=2
    with pytest.raises((NativeControlError,ValueError)):
        worker.dynamo_stationary_operation('pin',dict(expected_generation=expected,transaction_id='cpu-pin',
                                            plan=plan,native_scheduler_drain=drain))
    assert not getattr(worker,'_dynamo_stationary_owners',{})


def test_worker_retained_owner_blocks_weight_writes_and_full_drain_until_clean_release(monkeypatch):
    from pdblend_baselines.dynamollm.gpu_weights import DynamoWorkerExtension
    plan,_,_,_,_=setup_owner();worker=worker_fixture(monkeypatch)
    payload=dict(expected_generation=3,transaction_id='cpu-pin',plan=plan,native_scheduler_drain=native_drain())
    receipt=worker.dynamo_stationary_operation('pin',payload)
    assert receipt['held_view_bytes']>0 and not receipt['serving_qualified']
    with pytest.raises(ValueError,match='blocked'):worker.dynamo_operation('transfer',{})
    monkeypatch.setattr(DynamoWorkerExtension,'dynamo_operation',
                        lambda *_:dict(drained=True,active_weight_sessions=0))
    blocked=worker.dynamo_operation('drain',{})
    assert not blocked['drained'] and blocked['active_weight_sessions']==1
    released=worker.dynamo_stationary_operation('release',dict(payload,source_rank=0,consumer_processes_gone=[]))
    assert released['released'] and not released['owner_exit_required']
    assert worker.dynamo_operation('drain',{})['drained']


def test_worker_quarantine_blocks_future_transition_even_after_references_released(monkeypatch):
    worker=worker_fixture(monkeypatch)
    worker._dynamo_stationary_owners={'crashed':SimpleNamespace(closed=True,quarantined=True)}
    with pytest.raises(ValueError,match='blocked'):worker.dynamo_operation('open',{})
    with pytest.raises(ValueError,match='quarantined owner must exit'):
        worker.dynamo_stationary_operation('pin',dict(expected_generation=3,transaction_id='next'))


@pytest.mark.parametrize('operation',['release_kv','bind_target','activate'])
def test_worker_cannot_promote_primitive_into_target_serving(monkeypatch,operation):
    worker=worker_fixture(monkeypatch)
    with pytest.raises(RuntimeError,match='not implemented'):
        worker.dynamo_stationary_operation(operation,dict(expected_generation=3,transaction_id='cpu'))


def test_worker_release_rpc_does_not_apply_another_source_ranks_consumer_list(monkeypatch):
    worker=worker_fixture(monkeypatch);worker.parallel_config.tensor_parallel_size=2
    receipt=worker.dynamo_stationary_operation('release',dict(expected_generation=3,transaction_id='cpu',
                                            source_rank=1,consumer_processes_gone=[CONSUMER]))
    assert receipt['participating'] is False and not receipt['formal_eligible']
