"""Original producer graph CPU contracts; synthetic receipts never qualify GPUs."""
from copy import deepcopy
import hashlib
import json
import math

import pytest

from pdblend_baselines.dynamollm.stationary_ipc import digest
from pdblend_baselines.dynamollm.stationary_owner_graph import OriginalOwnerGraph,read_bound
from tests.independent_baselines.test_dynamo_stationary_memory import fixture,kv_receipt,packet
from tests.independent_baselines.test_dynamo_stationary_tensors import shapes


def bound(tmp_path,name,value):
    path=tmp_path/name;path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def setup(tmp_path,tp=1):
    plan,owners,_=fixture(tp,2 if tp==1 else 1)
    refs=[]
    for owner in owners:
        receipt=kv_receipt(plan,owner);receipt['generation']=0
        for identity in receipt['original_weight_identity'].values():
            identity.update(stride=[math.prod(identity['shape'][i+1:]) for i in range(len(identity['shape']))],
                storage_offset=0,data_ptr=identity['storage_ptr'],version=0)
        ref=bound(tmp_path,'owner-'+str(owner['source_rank'])+'.json',dict(raw=receipt))
        refs.append(dict(ref,json_path=['raw']))
    graph=OriginalOwnerGraph(plan,refs,allow_cpu_oracle=True)
    alive={digest(o['process']) for o in owners}
    return graph,plan,owners,alive,lambda p:digest(p) in alive


def export(tmp_path,graph,plan,owner):
    value=packet(plan,owner);value.update(export_id='export0',generation=0)
    value['packet_sha256']=digest({k:v for k,v in value.items() if k!='packet_sha256'})
    graph.register_export(bound(tmp_path,'packet.json',value),bound(tmp_path,'plan.json',plan))
    return value


@pytest.mark.parametrize('tp,target',[(1,1),(1,2),(2,1),(2,4),(4,2)])
def test_exact_canonical_coverage_retains_original_owner_and_keeps_full_allocation_roots(tmp_path,tp,target):
    graph,_,owners,_,alive=setup(tmp_path,tp)
    before=deepcopy(graph.roots)
    plan=graph.plan_handoff([f'GPU-{r}' for r in range(target)],shapes(target),process_alive=alive)
    assert sum(r['bytes'] for r in plan['routes'])==sum(math.prod(s)*2*target for s in shapes(target).values())
    for rank in range(target):
        for name,shape in shapes(target).items():
            routes=[r for r in plan['routes'] if r['target_rank']==rank and r['parameter']==name]
            assert sum(r['bytes'] for r in routes)==math.prod(shape)*2
            assert all(r['source_owner'] in [o['process'] for o in owners] for r in routes)
    assert graph.roots==before
    assert plan['checks']['fresh_owner_tensor_revalidation_required']
    assert not plan['checks']['hardware_qualified'] and not plan['target_served'] and not plan['executed']


def test_source_artifact_hash_and_os_liveness_are_independently_required(tmp_path):
    graph,_,owners,live,alive=setup(tmp_path)
    live.clear()
    with pytest.raises(ValueError,match='exited or restarted'):
        graph.plan_handoff(['GPU-0'],shapes(1),process_alive=alive)
    live.add(digest(dict(owners[0]['process'],start_ticks=2)))
    with pytest.raises(ValueError,match='exited or restarted'):
        graph.plan_handoff(['GPU-0'],shapes(1),process_alive=alive)
    live.add(digest(owners[0]['process']))
    path=graph.base_refs[0]['path']
    from pathlib import Path
    Path(path).write_bytes(Path(path).read_bytes()+b' ')
    with pytest.raises(ValueError,match='evidence bytes changed'):
        graph.plan_handoff(['GPU-0'],shapes(1),process_alive=alive)


def test_consumer_ack_does_not_release_owner_before_actual_consumer_exit(tmp_path):
    graph,plan,owners,live,alive=setup(tmp_path)
    value=export(tmp_path,graph,plan,owners[0]);consumer=value['consumer'];live.add(digest(consumer))
    ack=dict(export_id=value['export_id'],packet_sha256=value['packet_sha256'],consumer=consumer,
             gpu_uuid=value['gpu_uuid'],cuda_synchronized=True,views_released=True)
    graph.acknowledge_release(bound(tmp_path,'ack.json',ack))
    with pytest.raises(ValueError,match='actually exit'):graph.retire_consumer(consumer,process_alive=alive)
    with pytest.raises(ValueError,match='all target consumers'):graph.require_owner_restore(owners[0]['process'],process_alive=alive)
    live.remove(digest(consumer));graph.retire_consumer(consumer,process_alive=alive)
    result=graph.require_owner_restore(owners[0]['process'],process_alive=alive)
    assert result['graph_release_preconditions_passed'] and result['physical_owner_weight_release_credit_bytes']==0
    assert not result['source_KV_restored'] and not result['formal_eligible']


def test_consumer_crash_quarantines_original_owner_and_blocks_rollback(tmp_path):
    graph,plan,owners,_,alive=setup(tmp_path);value=export(tmp_path,graph,plan,owners[0])
    result=graph.retire_consumer(value['consumer'],process_alive=alive)
    assert result['owner_process_isolation_required'] and not result['clean']
    with pytest.raises(ValueError,match='quarantined'):
        graph.require_owner_restore(owners[0]['process'],process_alive=alive)


@pytest.mark.parametrize('fault',['consumer_as_owner','wrong_stride','missing_view','generation','root_allocation'])
def test_exports_cannot_forward_imported_aliases_or_change_original_geometry(tmp_path,fault):
    graph,plan,owners,_,_=setup(tmp_path);value=packet(plan,owners[0]);value.update(export_id='x',generation=0)
    if fault=='consumer_as_owner':value['owner']=dict(pid=777,start_ticks=1,boot_id='cpu-only')
    elif fault=='wrong_stride':
        v=value['views'][0];v['descriptor']['stride'][-1]=0
        value['source_storage'][v['piece']['parameter']]['stride'][-1]=0
    elif fault=='missing_view':value['views'].pop()
    elif fault=='generation':value['generation']=1
    else:value['views'][0]['descriptor']['allocation_bytes']+=512
    value['packet_sha256']=digest({k:v for k,v in value.items() if k!='packet_sha256'})
    with pytest.raises(ValueError):graph.register_export(bound(tmp_path,'bad.json',value),bound(tmp_path,'p.json',plan))
    assert not graph.exports


def receive_root(tmp_path,graph,route,handoff_ref,process,index=0):
    shape=list(route['target_shape'])
    if route['axis'] is not None:shape[route['axis']]=route['length']
    ptr=100000+index*10000
    receipt=dict(schema='dynamo-direct-receive-owner-root/v1',ownership='direct_cuda_receive_allocation',
        ipc_imported=False,completed=True,cuda_synchronized=True,host_weight_staging_bytes=0,
        retained_fragment_copy_bytes=0,cpu_oracle=True,handoff_ref=handoff_ref,route=route,
        gpu_uuid=route['target_gpu_uuid'],process=process,generation=1,
        tensor=dict(shape=shape,stride=[math.prod(shape[i+1:]) for i in range(len(shape))],
            storage_offset=0,storage_ptr=ptr,data_ptr=ptr,version=0,device='cpu'),
        allocation=dict(storage_ptr=ptr,storage_bytes=route['bytes'],segment_address=ptr,
                        allocation_bytes=route['bytes']))
    return receipt


def test_multi_round_handoff_keeps_new_owned_receive_roots_and_never_promotes_ipc_alias(tmp_path):
    graph,_,owners,live,alive=setup(tmp_path)
    target_owner=dict(pid=302,start_ticks=2,boot_id='cpu-only');live.add(digest(target_owner))
    first=graph.plan_handoff(['GPU-0','GPU-1'],shapes(2),process_alive=alive)
    ref=bound(tmp_path,'handoff.json',first)
    original_count=len(graph.roots)
    for i,route in enumerate(r for r in first['routes'] if r['kind']=='direct_missing_fragment_transfer'):
        value=receive_root(tmp_path,graph,route,ref,target_owner,i)
        graph.add_received_root(bound(tmp_path,f'received-{i}.json',value))
    assert len(graph.roots)>original_count
    second=graph.plan_handoff(['GPU-1'],shapes(1),process_alive=alive)
    local=[r for r in second['routes'] if r['kind']=='original_owner_ipc_export']
    assert local and all(r['source_owner']==target_owner for r in local)
    assert second['planned_retained_bytes']>0 and second['planned_transfer_bytes']>0
    assert any(r['source_owner']==owners[0]['process'] for r in second['routes'])
    snapshot=bound(tmp_path,'snapshot.json',graph.snapshot())
    restored=OriginalOwnerGraph.restore_snapshot(snapshot,allow_cpu_oracle=True,process_alive=alive)
    assert digest(restored.snapshot())==digest(graph.snapshot())
    assert restored.plan_handoff(['GPU-1'],shapes(1),process_alive=alive)['routes']==second['routes']


@pytest.mark.parametrize('fault',['ipc_alias','shape','canonical','copied_retained','handoff_hash'])
def test_received_root_requires_direct_ownership_and_exact_parent_route(tmp_path,fault):
    graph,_,_,_,alive=setup(tmp_path)
    h=graph.plan_handoff(['GPU-0','GPU-1'],shapes(2),process_alive=alive)
    route=next(r for r in h['routes'] if r['kind']=='direct_missing_fragment_transfer')
    ref=bound(tmp_path,'h.json',h)
    row=receive_root(tmp_path,graph,route,ref,dict(pid=302,start_ticks=2,boot_id='cpu-only'))
    if fault=='ipc_alias':row['ipc_imported']=True
    elif fault=='shape':row['tensor']['shape'][0]+=1
    elif fault=='canonical':row['route']=dict(route,canonical_offset=route['canonical_offset']+1)
    elif fault=='copied_retained':row['retained_fragment_copy_bytes']=2
    else:row['handoff_ref']=dict(ref,sha256='0'*64)
    with pytest.raises(ValueError):graph.add_received_root(bound(tmp_path,'bad-recv.json',row))


def test_snapshot_cannot_omit_an_original_root_even_when_no_route_currently_uses_it(tmp_path):
    graph,_,_,_,alive=setup(tmp_path)
    value=graph.snapshot();value['roots'].pop(next(iter(value['roots'])))
    with pytest.raises(ValueError,match='lost roots'):
        OriginalOwnerGraph.restore_snapshot(bound(tmp_path,'missing.json',value),allow_cpu_oracle=True,process_alive=alive)


def test_received_owner_exit_requires_bound_cleanup_before_source_rollback(tmp_path):
    graph,_,owners,live,alive=setup(tmp_path)
    h=graph.plan_handoff(['GPU-0','GPU-1'],shapes(2),process_alive=alive)
    route=next(r for r in h['routes'] if r['kind']=='direct_missing_fragment_transfer')
    target=dict(pid=302,start_ticks=2,boot_id='cpu-only');live.add(digest(target))
    graph.add_received_root(bound(tmp_path,'receive.json',receive_root(tmp_path,graph,route,
        bound(tmp_path,'handoff.json',h),target)))
    cleanup=dict(schema='dynamo-owned-process-isolation/v1',process=target,process_gone=True,
        compute_pid_absent=True,gpu_uuids=['GPU-1'],host_pid_binding_ref=bound(tmp_path,'pid-map.json',dict(
            schema='dynamo-nvml-host-pid-binding/v1',process=target,host_pid=90302,namespace_pid_chain=[90302,302],
            observer_pid_namespace='host',host_proc_start_ticks=2,host_boot_id='cpu-only')),
        observations=[dict(source='NVML_compute_processes',pid_namespace='host',gpu_uuid='GPU-1',compute_pids=[])])
    ref=bound(tmp_path,'cleanup.json',cleanup)
    with pytest.raises(ValueError,match='actually exited'):graph.retire_received_owner(ref,process_alive=alive)
    live.remove(digest(target))
    with pytest.raises(ValueError,match='exited or restarted'):graph.require_owner_restore(owners[0]['process'],process_alive=alive)
    graph.retire_received_owner(ref,process_alive=alive)
    assert graph.require_owner_restore(owners[0]['process'],process_alive=alive)['graph_release_preconditions_passed']
    final=graph.plan_handoff(['GPU-0'],shapes(1),process_alive=alive)
    assert final['planned_transfer_bytes']==0
    assert all(r['source_owner']==owners[0]['process'] for r in final['routes'])


def test_cpu_receipts_never_enter_default_non_oracle_graph(tmp_path):
    graph,plan,_,_,_=setup(tmp_path)
    with pytest.raises(ValueError,match='CPU oracle'):
        OriginalOwnerGraph(plan,graph.base_refs)


@pytest.mark.parametrize('name',['model.norm.weight','model.embed_tokens.weight'])
def test_target_shape_cannot_silently_drop_original_parameter_coordinates(tmp_path,name):
    graph,_,_,_,alive=setup(tmp_path);target=shapes(1);target[name]=list(target[name]);target[name][-1]-=1
    with pytest.raises(ValueError):graph.plan_handoff(['GPU-0'],target,process_alive=alive)


@pytest.mark.parametrize('fault',['packet','consumer','synchronization','views'])
def test_clean_release_ack_cannot_be_substituted_by_an_unrelated_or_partial_ack(tmp_path,fault):
    graph,plan,owners,_,_=setup(tmp_path);value=export(tmp_path,graph,plan,owners[0])
    ack=dict(export_id=value['export_id'],packet_sha256=value['packet_sha256'],consumer=value['consumer'],
             gpu_uuid=value['gpu_uuid'],cuda_synchronized=True,views_released=True)
    if fault=='packet':ack['packet_sha256']='0'*64
    elif fault=='consumer':ack['consumer']=dict(value['consumer'],start_ticks=2)
    elif fault=='synchronization':ack['cuda_synchronized']=False
    else:ack['views_released']=False
    with pytest.raises(ValueError,match='clean ACK binding differs'):
        graph.acknowledge_release(bound(tmp_path,'bad-ack.json',ack))
    assert not graph.acks


@pytest.mark.parametrize('fault',['float_address','bool_bytes','negative_address','storage_overlap','segment_conflict'])
def test_receive_root_physical_bounds_cannot_be_inferred_from_logical_fragment_bytes(tmp_path,fault):
    graph,_,_,_,alive=setup(tmp_path)
    h=graph.plan_handoff(['GPU-0','GPU-1'],shapes(2),process_alive=alive)
    route=next(r for r in h['routes'] if r['kind']=='direct_missing_fragment_transfer')
    target=dict(pid=302,start_ticks=2,boot_id='cpu-only')
    value=receive_root(tmp_path,graph,route,bound(tmp_path,'handoff.json',h),target)
    if fault=='float_address':value['allocation']['storage_ptr']=float(value['allocation']['storage_ptr'])
    elif fault=='bool_bytes':value['allocation']['storage_bytes']=True
    elif fault=='negative_address':value['allocation']['segment_address']=-1
    else:
        value['allocation']['allocation_bytes']+=1000
        graph.add_received_root(bound(tmp_path,'original-receive.json',value))
        value=deepcopy(value)
        if fault=='storage_overlap':
            value['tensor']['storage_ptr']+=2;value['tensor']['data_ptr']+=2;value['allocation']['storage_ptr']+=2
        else:
            value['tensor']['storage_ptr']+=route['bytes'];value['tensor']['data_ptr']+=route['bytes']
            value['allocation']['storage_ptr']+=route['bytes'];value['allocation']['allocation_bytes']+=route['bytes']
    with pytest.raises(ValueError):graph.add_received_root(bound(tmp_path,'bad.json',value))


def test_nvml_cleanup_cannot_confuse_container_pid_with_still_live_host_cuda_pid(tmp_path):
    graph,_,_,_,alive=setup(tmp_path)
    h=graph.plan_handoff(['GPU-0','GPU-1'],shapes(2),process_alive=alive)
    route=next(r for r in h['routes'] if r['kind']=='direct_missing_fragment_transfer')
    target=dict(pid=302,start_ticks=2,boot_id='cpu-only')
    graph.add_received_root(bound(tmp_path,'receive.json',receive_root(tmp_path,graph,route,
        bound(tmp_path,'handoff.json',h),target)))
    mapping=dict(schema='dynamo-nvml-host-pid-binding/v1',process=target,host_pid=90302,
        namespace_pid_chain=[90302,302],observer_pid_namespace='host',host_proc_start_ticks=2,host_boot_id='cpu-only')
    cleanup=dict(schema='dynamo-owned-process-isolation/v1',process=target,process_gone=True,
        compute_pid_absent=True,gpu_uuids=['GPU-1'],host_pid_binding_ref=bound(tmp_path,'map.json',mapping),
        observations=[dict(source='NVML_compute_processes',pid_namespace='host',gpu_uuid='GPU-1',compute_pids=[90302])])
    with pytest.raises(ValueError,match='compute PID'):
        graph.retire_received_owner(bound(tmp_path,'cleanup.json',cleanup),process_alive=alive)
