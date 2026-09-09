from dataclasses import replace
from ecopadg.serving.dynamo_topology import DynamoTopologyPlanner,TopologyCost,PoolConfiguration
from ecopadg.serving.profiles import ProfilePoint,ProfileStore
from ecopadg.serving.types import InstanceState,RuntimeSnapshot


def test_batch_capacity_counts_every_prefill_before_shared_decode():
    point=ProfilePoint('mixed',1,2520,7168,7680,8,1,.04,200,60,0,3,'measured')
    assert point.batch_service_s(2)==8.04
    profiles=ProfileStore([point],idle_unallocated_gpu_w=35)
    planner=DynamoTopologyPlanner(profiles,())
    demand=dict(rate=2,input_tokens=7168,output_tokens=2,ttft_s=5,tpot_s=.1)
    assert not planner.configurations('LS',demand,1)


def test_measured_batch_ceiling_supports_rates_between_batch_grid_points():
    points=[ProfilePoint('mixed',1,2520,128,640,b,.02,.04,220,60,0,3,'measured') for b in (1,8,32)]
    planner=DynamoTopologyPlanner(ProfileStore(points,idle_unallocated_gpu_w=35),())
    # Above batch-8 capacity and below 97% of batch-32 capacity: this is a
    # supported workload, not a reason to reject the measured batch ceiling.
    demand=dict(rate=1,input_tokens=128,output_tokens=512,ttft_s=5,tpot_s=.1)
    choices=planner.configurations('SL',demand,1)
    assert choices and choices[0].capacity_rps>=1
    assert choices[0].batch_sizes==(32,)


def test_zero_load_diagnostic_keeps_maximum_capacity_when_residency_power_ties():
    points=[ProfilePoint('mixed',1,2520,128,640,b,.02,.04,220,60,0,3,'measured') for b in (1,8,32)]
    planner=DynamoTopologyPlanner(ProfileStore(points,idle_unallocated_gpu_w=35),())
    demand=dict(rate=0,input_tokens=128,output_tokens=512,ttft_s=5,tpot_s=.1)
    choice=planner.configurations('SL',demand,1)[0]
    assert choice.batch_sizes==(32,)
    assert choice.capacity_rps==max(p.batch/p.batch_service_s(512) for p in points)


def test_slow_pool_search_uses_only_measured_tp_and_amortized_cost():
    points=[ProfilePoint('mixed',tp,2520,1024,2048,1,.1,.04,w,30*tp,.05,3,'measured')
            for tp,w in ((1,120),(2,350))]
    profiles=ProfileStore(points,idle_unallocated_gpu_w=35)
    state=RuntimeSnapshot(1,10,(InstanceState('a', 'mixed',2,(0,1),10,0,2520,10000,0,0),))
    demand={'SS':dict(rate=.1,input_tokens=128,output_tokens=64,ttft_s=5,tpot_s=.1)}
    planner=DynamoTopologyPlanner(profiles,())
    assert planner.choose(state,{'a':'SS'},demand,'ScaleShard') is None
    planner=DynamoTopologyPlanner(profiles,[TopologyCost((2,),(1,),1,1,'measured')])
    assert planner.choose(state,{'a':'SS'},demand,'ScaleShard') is None
    result=planner.choose(state,{'a':'SS'},demand,'ScaleInst')
    assert result and result['target_tps']==(1,)
    assert result['savings_lower_j']>result['cost_upper_j']
    assert all(set(c.tps)<={1,2} for c in planner.configurations('SS',demand['SS'],8))


def test_fragmented_short_long_request_never_spills_to_long_short_pool():
    planner=DynamoTopologyPlanner(ProfileStore(()),())
    request=dict(rate=1,input_tokens=128,output_tokens=1024,ttft_s=5,tpot_s=.1)
    assert planner.pool_demand({'SL':request},{'0':'MS','1':'LL'})=={'LL':request}
    assert planner.pool_demand({'SL':request},{'0':'MS'}) is None


def test_overload_proposes_measured_capacity_recovery_without_invented_savings():
    points=[ProfilePoint('mixed',tp,2520,1024,2048,1,.1,iteration,w,30*tp,0,3,'measured')
            for tp,w,iteration in ((1,120,.09),(2,350,.02))]
    profiles=ProfileStore(points,idle_unallocated_gpu_w=35)
    state=RuntimeSnapshot(1,10,(InstanceState('a','mixed',1,(0,),10,0,2520,10000,0,0),))
    demand={'SS':dict(rate=.3,input_tokens=128,output_tokens=64,ttft_s=5,tpot_s=.1)}
    planner=DynamoTopologyPlanner(profiles,[TopologyCost((1,),(2,),10,1000,'measured')])
    proposal=planner.choose(state,{'a':'SS'},demand,'ScaleInst')
    assert proposal and proposal['target_tps']==(2,)
    assert proposal['savings_lower_j']==0
    recovery=proposal['capacity_recovery']
    assert recovery['current_rps']<recovery['required_rps']<=recovery['target_rps']


def small_pool(costs,instances,rate=.1):
    planner=DynamoTopologyPlanner(ProfileStore(()),costs)
    def choices(shape,demand,budget):
        return [PoolConfiguration(shape,(1,)*n,n,100*n,(1,)*n) for n in (1,2)
                if n<=budget and n>=demand['rate']]
    planner.configurations=choices
    state=RuntimeSnapshot(1,10,tuple(instances))
    demand={'SS':dict(rate=rate,input_tokens=128,output_tokens=64,ttft_s=5,tpot_s=.1)}
    return planner,state,demand


def test_shrink_keeps_live_matching_replica_and_measures_only_removed_difference():
    idle=InstanceState('a','mixed',1,(0,),10,0,2520,10000,0,0)
    live=replace(idle,instance_id='b',gpus=(1,),running=2)
    planner,state,demand=small_pool([TopologyCost((1,),(),1,10,'real-delta',True)],[idle,live])
    result=planner.choose(state,{'a':'SS','b':'SS'},demand,'ScaleInst',cached_weights=True)
    assert result['retained_ids']==('b',) and result['remove_ids']==('a',)
    assert result['target_tps']==(1,) and result['add_tps']==()
    assert result['source_cost'].source_tps==(1,) and result['source_cost'].target_tps==()


def test_missing_difference_cost_uses_only_explicit_measured_whole_pool_fallback():
    a=InstanceState('a','mixed',1,(0,),10,0,2520,10000,0,0)
    b=replace(a,instance_id='b',gpus=(1,),running=1)
    planner,state,demand=small_pool([],[a,b])
    assert planner.choose(state,{'a':'SS','b':'SS'},demand,'ScaleInst',cached_weights=True) is None
    planner.costs=(TopologyCost((1,1),(1,),20,100,'whole-pool'),)
    result=planner.choose(state,{'a':'SS','b':'SS'},demand,'ScaleInst',cached_weights=True)
    assert result['retained_ids']==() and set(result['remove_ids'])=={'a','b'}
    assert result['add_tps']==result['target_tps']==(1,)
    assert result['cost_upper_j']==100


def test_pure_expansion_requires_measured_empty_source_cost_and_cached_weights():
    a=InstanceState('a','mixed',1,(0,),10,0,2520,10000,1,0)
    planner,state,demand=small_pool([],[a],rate=1.5)
    assert planner.choose(state,{'a':'SS'},demand,'ScaleInst',cached_weights=True) is None
    planner.costs=(TopologyCost((),(1,),10,100,'measured-add',True),)
    assert planner.choose(state,{'a':'SS'},demand,'ScaleInst',cached_weights=False) is None
    result=planner.choose(state,{'a':'SS'},demand,'ScaleInst',cached_weights=True)
    assert result['retained_ids']==('a',) and result['remove_ids']==()
    assert result['target_tps']==(1,1) and result['add_tps']==(1,)
    assert result['capacity_recovery'] and result['savings_lower_j']==0


def test_expansion_does_not_borrow_another_pool_or_budget_outside_eight_gpus():
    instances=[InstanceState(str(n),'mixed',1,(n,),10,0,2520,10000,1,0) for n in range(8)]
    planner,state,demand=small_pool([TopologyCost((),(1,),10,100,'measured-add',True)],instances,rate=1.5)
    mapping={str(n):'SS' if n==0 else 'LL' for n in range(8)}
    assert planner.choose(state,mapping,demand,'ScaleInst',cached_weights=True) is None


def test_pure_expansion_commit_preserves_explicit_native_pool_shape():
    import asyncio
    from types import SimpleNamespace
    from ecopadg.serving.runtime import Controller
    from ecopadg.serving.topology import InstanceSpec
    async def run():
        controller=Controller.__new__(Controller)
        controller.action_lock=asyncio.Lock();controller.active={}
        controller._dynamo_transaction_shape='MS'
        controller.dynamo_scheduler=SimpleNamespace(assignments={'live':'MS','catchall':'LL'},version=0)
        controller.backend=SimpleNamespace(instances={'live':dict(gpus=[0]),'catchall':dict(gpus=[1])},clocks=None)
        def replace_instances(ids,added):
            controller.backend.instances={k:v for k,v in controller.backend.instances.items() if k not in ids}
            controller.backend.instances.update({v['id']:v for v in added})
        controller.backend.replace_instances=replace_instances
        async def refresh():pass
        controller.refresh=refresh
        await controller.commit_topology((),(InstanceSpec('added',1,(2,),18002,19200),))
        assert controller.dynamo_scheduler.assignments=={'live':'MS','catchall':'LL','added':'MS'}
    asyncio.run(run())


def test_runtime_creates_only_incremental_target_specs_and_records_retained_ids(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from ecopadg.serving.runtime import Controller
    from ecopadg.serving.topology import InstanceSpec
    from ecopadg.serving.state import StateManager
    async def run():
        for expand in (False,True):
            states=[InstanceState('live','mixed',1,(0,),10,0,2520,10000,1,0)]
            if not expand:states.append(replace(states[0],instance_id='idle',gpus=(1,),running=0))
            controller=Controller(dict(strategy='mixed',journal=str(tmp_path/'journal')))
            controller.state=StateManager(RuntimeSnapshot(1,10,tuple(states)))
            controller.retained_weights='cache'
            controller.dynamo_scheduler=SimpleNamespace(assignments={i.instance_id:'MS' for i in states},
                forecast=lambda now:{},hierarchy=SimpleNamespace(PERIODS={'ScaleInst':1800}))
            proposal=dict(shape='MS',retained_ids=('live',),remove_ids=() if expand else ('idle',),
                add_tps=(1,) if expand else (),target_tps=(1,1) if expand else (1,),
                savings_lower_j=100,cost_upper_j=10)
            controller.dynamo_topology=SimpleNamespace(choose=lambda *args,**kwargs:proposal)
            specs={i.instance_id:InstanceSpec(i.instance_id,1,i.gpus,18000+j,19000+j*16) for j,i in enumerate(states)}
            calls=[];events=[]
            async def reconfigure(removed,added,**kwargs):
                calls.append((removed,added))
                assert controller._dynamo_transaction_shape=='MS'
                return dict(retained_weights='cache',committed=True)
            controller.topology_manager=SimpleNamespace(node_gpus=tuple(range(8)),specs=specs,version=0,reconfigure=reconfigure)
            async def emit(event):events.append(event)
            async def reassign(now):pass
            controller.journal=SimpleNamespace(emit=emit);controller.dynamo_reassign=reassign
            await controller.dynamo_slow('ScaleInst',10)
            assert len(calls)==1 and len(calls[0][1])==(1 if expand else 0)
            assert 'live' not in calls[0][0]
            assert controller._dynamo_transaction_shape is None
            assert events[-1]['executed'] and events[-1]['retained_ids']==('live',)
    asyncio.run(run())


def dormant_pool(costs=None):
    profiles=ProfileStore((),idle_unallocated_gpu_w=35,parked_residency_w_by_tp={1:85})
    costs=[TopologyCost([1],[],2,1000,'measured-delete',True)] if costs is None else costs
    planner=DynamoTopologyPlanner(profiles,costs)
    instances=tuple(InstanceState(name,'mixed',1,(gpu,),1800,0,900,10000,0,0)
        for name,gpu in (('a',0),('b',1),('catchall',2)))
    state=RuntimeSnapshot(1,1800,instances)
    return planner,state,{'a':'LM','b':'LM','catchall':'LL'}


def test_scaleinst_reclaims_dormant_duplicate_with_measured_parked_savings():
    planner,state,assignments=dormant_pool()
    result=planner.choose(state,assignments,{},'ScaleInst',cached_weights=True)
    assert result['shape']=='LM' and len(result['remove_ids'])==1
    assert result['target_tps']==(1,) and result['add_tps']==()
    assert set(result['remove_ids'])|set(result['retained_ids'])=={'a','b'}
    assert 'catchall' not in result['remove_ids']
    assert result['forecast']['rate']==0 and result['capacity_recovery'] is None
    assert result['savings_lower_j']==(85-35)*.7*(1800-2)
    assert result['savings_lower_j']>result['cost_upper_j']==1000
    assert planner.choose(state,assignments,{},'ScaleShard',cached_weights=True) is None


def test_dormant_shrink_requires_actual_cost_and_positive_amortization():
    for costs in ([],[TopologyCost((1,),(),2,100000,'too-expensive',True)],
                  [TopologyCost((1,),(),2,1000,'',True)],
                  [TopologyCost((1,1),(1,),2,1000,'whole-rebuild-only',True)]):
        planner,state,assignments=dormant_pool(costs)
        assert planner.choose(state,assignments,{},'ScaleInst',cached_weights=True) is None
    planner,state,assignments=dormant_pool()
    assert planner.choose(state,assignments,{},'ScaleInst',cached_weights=False) is None
    planner.profiles.parked_residency_w_by_tp[1]=35
    assert planner.choose(state,assignments,{},'ScaleInst',cached_weights=True) is None


def test_dormant_shrink_preserves_active_reserved_stale_or_unmatched_pool():
    from ecopadg.serving.types import RequestBudget
    changes=[dict(running=1),dict(waiting=1),dict(accepting=False),dict(timestamp_s=1798),
        dict(timestamp_s=1801),dict(reserved_kv_tokens=1),dict(reserved_transfer_bytes=1),
        dict(kv_allocations=(('r',1),)),dict(transfer_allocations=(('r',1),)),
        dict(requests=(RequestBudget('r',1700,128,64,5,.1),)),dict(tp=2)]
    for change in changes:
        planner,state,assignments=dormant_pool()
        state=replace(state,instances=(replace(state.instances[0],**change),*state.instances[1:]))
        assert planner.choose(state,assignments,{},'ScaleInst',cached_weights=True) is None
    planner,state,assignments=dormant_pool()
    # One resident instance is already the per-pool warm floor.
    state=replace(state,instances=(state.instances[0],state.instances[2]))
    assert planner.choose(state,assignments,{},'ScaleInst',cached_weights=True) is None


def test_dormant_shrink_does_not_override_a_causal_positive_demand():
    planner,state,assignments=dormant_pool()
    # No usable active-capacity measurements: positive demand must not be
    # treated as zero simply because the loaded replicas currently look idle.
    demand={'LS':dict(rate=.1,input_tokens=7168,output_tokens=64,ttft_s=5,tpot_s=.1)}
    assert planner.choose(state,assignments,demand,'ScaleInst',cached_weights=True) is None


def test_aged_unrouted_controller_request_blocks_dormant_shrink():
    planner,state,assignments=dormant_pool()
    # The 300-second forecast may omit an older request still in controller
    # admission queues; no physical target has reserved that request yet.
    assert planner.choose(state,assignments,{},'ScaleInst',cached_weights=True,
        has_unrouted_requests=True) is None


def test_runtime_passes_unrouted_work_guard_before_offloading_scaleinst():
    import asyncio
    from types import SimpleNamespace
    from ecopadg.serving.runtime import Controller
    async def run():
        for active,blocked in (({},False),({'r':dict(route=None)},True),
                               ({'r':dict(route=object())},False)):
            controller=Controller.__new__(Controller)
            controller.state=SimpleNamespace(snapshot=RuntimeSnapshot(0,1800,()))
            controller.dynamo_scheduler=SimpleNamespace(assignments={},forecast=lambda now:{},
                hierarchy=SimpleNamespace(PERIODS={'ScaleInst':1800}))
            calls=[]
            def choose(*args,**kwargs):calls.append(kwargs);return None
            controller.dynamo_topology=SimpleNamespace(choose=choose)
            controller.retained_weights='measured-cache';controller.active=active
            controller.control_stop=asyncio.Event()
            async def emit(event):pass
            async def reassign(now):pass
            controller.journal=SimpleNamespace(emit=emit);controller.dynamo_reassign=reassign
            await controller.dynamo_slow('ScaleInst',1800)
            assert calls==[dict(cached_weights=True,has_unrouted_requests=blocked)]
    asyncio.run(run())


def test_dynamo_guard_does_not_change_pd_topology_call_signature():
    import asyncio
    from types import SimpleNamespace
    from ecopadg.serving.runtime import Controller
    from ecopadg.serving.pd_topology import PDBlendTopologyPlanner,MeasuredCapacity
    from ecopadg.serving.planner import JointPlanner
    from ecopadg.serving.forecast import RoleForecast
    async def run():
        controller=Controller.__new__(Controller)
        controller.state=SimpleNamespace(snapshot=RuntimeSnapshot(0,1800,()))
        controller.pd_topology=PDBlendTopologyPlanner(JointPlanner(ProfileStore(())),
            [TopologyCost((2,),(1,),1,1,'measured')],[MeasuredCapacity(1,1000,1024,1,'measured')])
        controller.retained_weights='cache';controller.active={'pending':dict(route=None)}
        controller.control_stop=asyncio.Event();events=[]
        async def emit(event):events.append(event)
        controller.journal=SimpleNamespace(emit=emit)
        await controller.pd_slow(RoleForecast((),1,300,0,0,0))
        assert [e['kind'] for e in events]==['pd_topology_epoch']
        assert events[0]['proposal'] is None
    asyncio.run(run())
