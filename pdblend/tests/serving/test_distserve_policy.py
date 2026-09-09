from dataclasses import replace

import pytest

from ecopadg.serving.distserve import DistServeScheduler
from ecopadg.serving.baselines import DistServeSearch
from ecopadg.serving.planner import TransferCost
from ecopadg.serving.profiles import ProfilePoint,ProfileStore
from test_planner import system


def test_spatial_policy_keeps_full_clocks_and_independent_stage_limits():
    planner,snapshot,r=system()
    policy=DistServeScheduler(planner.profiles,planner.transfers,prefill_batch=1,decode_batch=4)
    plan=policy.plan(snapshot,(r,),now=10)
    assert plan.feasible and plan.routes[0].prefill_id!=plan.routes[0].decode_id
    assert all(a.frequency_mhz==2520 for a in plan.frequencies)
    busy=replace(snapshot,instances=tuple(replace(i,requests=(replace(r,request_id='old'),))
                 if i.role=='prefill' else i for i in snapshot.instances))
    assert not policy.plan(busy,(r,),now=10).feasible


def test_search_rejects_unvalidated_transfer_and_respects_gpu_and_kv_limits():
    planner,snapshot,r=system()
    search=DistServeSearch(planner.profiles,planner.transfers,gpu_count=8)
    choices=search.search(128,64,2,.1,1,{1:100000})
    assert choices and all(sum(map(len,c.gpus))<=8 for c in choices)
    assert not search.search(128,64,2,.1,1,{1:1})
    search=DistServeSearch(planner.profiles,(),gpu_count=8)
    assert not search.search(128,64,2,.1,1,{1:100000})


def measured(role,*,tp=1,n=128,context=129,batch=1,prefill=.1,iteration=.001,source='measured'):
    return ProfilePoint(role,tp,2520,n,context,batch,prefill,iteration,200,30,0,1,source)


@pytest.mark.parametrize('old_context',[129,640])
def test_search_cannot_bypass_fresh_duplicate_or_more_specific_prefill(old_context):
    fresh=measured('prefill',prefill=1,source='fresh-first')
    old=measured('prefill',context=old_context,source='old-faster')
    decode=measured('decode',context=640)
    link=TransferCost(1,1,128,.01,0,'transfer',True)
    store=ProfileStore([fresh,old,decode])
    assert store.lookup('prefill',1,2520,128,129,1)==fresh
    assert not DistServeSearch(store,[link],gpu_count=2).search(128,64,.5,.1,5,{1:100000})
    assert DistServeSearch(ProfileStore([old,decode]),[link],gpu_count=2).search(
        128,64,.5,.1,5,{1:100000})


def test_decode_search_requires_requested_input_as_well_as_total_context():
    prefill=measured('prefill',n=512,context=513)
    short=measured('decode',n=128,context=640)
    actual=measured('decode',n=512,context=1024,iteration=.2)
    store=ProfileStore([prefill,short,actual])
    assert store.lookup('decode',1,2520,512,576,1)==actual
    link=TransferCost(1,1,512,.01,0,'transfer',True)
    assert not DistServeSearch(store,[link],gpu_count=2).search(512,64,2,.1,1,{1:100000})


def test_lookup_keeps_independent_measured_stage_tp_and_batch_choices():
    store=ProfileStore([measured('prefill',tp=2,batch=4,prefill=.2),
                        measured('decode',tp=1,batch=8,context=640,iteration=.03)])
    link=TransferCost(2,1,128,.01,0,'transfer',True,profile_batch=4)
    choices=DistServeSearch(store,[link],gpu_count=3).search(128,64,2,.1,1,{1:100000})
    assert choices
    assert all((c.prefill_tp,c.decode_tp,c.prefill_batch,c.decode_batch)==(2,1,4,8) for c in choices)


def test_search_uses_full_frequency_transfer_from_the_online_shape_bucket():
    store=ProfileStore([measured('prefill'),measured('decode',context=640)])
    links=[TransferCost(1,1,n,seconds,0,f'{n}-{frequency}',True,decode_frequency_mhz=frequency)
           for n,frequency,seconds in [(128,2520,.2),(128,900,.01),(512,2520,.001)]]
    search=DistServeSearch(store,links,gpu_count=2)
    choices=search.search(128,64,1,.1,1,{1:100000})
    assert choices and all(c.transfer_upper_s==.2 for c in choices)
    assert not search.search(128,64,.15,.1,1,{1:100000})
