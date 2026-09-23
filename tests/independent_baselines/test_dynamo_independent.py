import asyncio
import importlib
import math
from pathlib import Path

import pytest

from pdblend_baselines.dynamollm.policy import (
    Request, Replica, DynamoPolicy, WeeklyLoadTemplate, shard_milp,
    classify, allocate_pools, Epochs, Configuration,
)
from pdblend_baselines.dynamollm.profiles import PaperProfiles, CoverageError
from pdblend_baselines.dynamollm.reconfiguration import (
    Transition, Reconfiguration, weight_transfer_plan,
)
from .independence_contract import assert_independent_imports


def profiles():
    return PaperProfiles([dict(role='mixed', tp=tp, frequency_mhz=f,
        input_tokens=n, context_tokens=c, batch=b, prefill_s=n / 10000,
        iteration_s=.01 * 2520/f, power_w=tp * f/10, samples=1,
        source_sha256='a'*64) for tp in (1,2) for f in (1260,2520)
        for n in (16,32) for c in (64,128) for b in (1,4)])


def test_independent_package_imports_only_reviewed_measurement_infrastructure():
    package=Path(importlib.import_module('pdblend_baselines.dynamollm').__file__).parent
    for path in package.glob('*.py'):
        assert_independent_imports(path.read_text(), 'dynamollm/' + path.name)


def test_nine_classes_exact_boundaries_and_real_periods():
    assert {classify(n,o) for n in (255,256,1024) for o in (99,100,350)} == {
        a+b for a in 'SML' for b in 'SML'}
    clocks=Epochs(10)
    assert clocks.due(14.99)==()
    assert clocks.due(15)==('ScaleFreq',)
    assert clocks.due(310)==('ScaleShard','ScaleFreq')
    assert clocks.due(1810)==('ScaleInst','ScaleShard','ScaleFreq')


def test_profile_interpolates_interior_without_pdblend_dominating_bucket():
    p=profiles().query(1,2520,24,96,2)
    assert p.prefill_s == pytest.approx(.0024)
    assert p.decode_s == pytest.approx(.01)
    with pytest.raises(CoverageError): profiles().query(1,2520,33,96,2)
    with pytest.raises(CoverageError): profiles().query(4,2520,24,96,2)


def test_minimum_energy_same_pool_before_spilling_and_no_future_output():
    policy=DynamoPolicy(profiles())
    q=Request('q',24,40,0,1,.1)
    replicas=[Replica('hot',(0,),1,'SS',2520,free_kv_tokens=1000),
              Replica('cool',(1,),1,'SS',1260,free_kv_tokens=1000),
              Replica('large',(2,),1,'LL',1260,free_kv_tokens=1000)]
    assert policy.route(q,replicas,0).instance_id=='cool'
    replicas[0].accepting=False;replicas[1].free_kv_tokens=0
    assert policy.route(q,replicas,0).instance_id=='large'
    replicas[2].shape='MS'
    q.predicted_output=350
    assert policy.route(q,replicas,0) is None


def test_milp_exact_budget_and_capacity_optimizes_whole_pool():
    choices=[Configuration(1,1,1.,100.), Configuration(2,1,3.,150.)]
    result=shard_milp(choices,4,5.)
    assert sum(c.tp*n for c,n in result)==4
    assert sum(c.capacity_rps*n for c,n in result)>=5
    assert sum(c.power_w*n for c,n in result)==300
    with pytest.raises(CoverageError): shard_milp(choices,4,7.)


def test_scaleinst_floor_carries_fraction_to_next_larger_pool():
    result=allocate_pools({'SS':2.5,'LL':.5}, {'SS':2.,'SM':2.,'SL':2.,
        'ML':2.,'LL':2.}, reference_tp=2, gpu_budget=8)
    assert result['SS']['gpus']==2
    assert sum(v['gpus'] for v in result.values())==4
    assert result['LL']['rate_rps']==1.


def test_scaleinst_budget_caps_replicas_and_reserves_largest_pool():
    # Demand far exceeds the fixed host. Pools are sized down to the budget
    # instead of raising, and the largest (LL) pool stays allocated as the
    # universal sink so every request class remains routable.
    result=allocate_pools({'SS':20.}, {'SS':5.,'SM':5.,'SL':5.,'ML':5.,'LL':5.},
        reference_tp=4, gpu_budget=8)
    assert set(result)=={'SS','LL'}
    assert result['SS']==dict(gpus=4,rate_rps=5.)
    assert result['LL']==dict(gpus=4,rate_rps=5.)
    assert sum(p['gpus'] for p in result.values())==8


def test_weekly_template_uses_prior_weeks_and_rejects_future_training():
    week=604800;monday=345600
    rows=[dict(at_s=monday+day*86400+k+1,input_tokens=16,output_tokens=40)
          for day,count in enumerate((1,2,3,4,100,10,20)) for k in range(count)]
    template=WeeklyLoadTemplate.fit(rows,start_s=monday,end_s=monday+week,slot_s=300)
    assert template.forecast(monday+week,300)['SS']==pytest.approx(3/300)
    assert template.forecast(monday+week+5*86400,300)['SS']==pytest.approx(15/300)
    assert template.forecast(monday+week+300,300)['SS']==0
    with pytest.raises(ValueError): WeeklyLoadTemplate.fit(rows+[dict(rows[0],at_s=monday+week)],
        start_s=monday,end_s=monday+week)


def test_weight_graph_mapping_minimizes_transfers_both_directions():
    plan=weight_transfer_plan([(0,1,2,3)],[(0,1),(2,3)],8)
    assert plan['model_units']==8
    assert plan['retained_units']==8
    assert sum(t['units'] for t in plan['transfers'])==8
    assert all(t['source_gpu']!=t['target_gpu'] for t in plan['transfers'])
    reverse=weight_transfer_plan([(0,1),(2,3)],[(0,1,2,3)],8)
    assert reverse['retained_units']==8
    assert reverse['transfers']==[]


@pytest.mark.asyncio
async def test_overlap_prepares_before_source_freeze_and_commits_only_after_ack():
    events=[]
    class Hooks:
        async def prepare(self,t,plan): events.append('prepare');return {'id':'prepared'}
        async def verify(self,t,prepared): events.append('verify');return {'ready':True,'outputs_valid':True}
        async def freeze(self,t):events.append('freeze')
        async def drain(self,t):events.append('drain')
        async def activate(self,t,prepared):events.append('activate');return {'activated':True}
        async def retire(self,t):events.append('retire')
        async def abort(self,t,prepared):events.append('abort');return {'target_stopped':True}
        async def restore(self,t):events.append('restore')
        async def isolate(self,t):events.append('isolate')
    tx=Reconfiguration(Hooks(),lambda event,**fields: None)
    t=Transition('t',('source',),((0,1),),((0,),(1,)),overlap_memory_qualified=True)
    result=await tx.execute(t,commit=lambda _: events.append('commit'))
    assert events==['prepare','verify','freeze','drain','activate','commit','retire']
    assert result['phase']=='complete'
    assert await tx.execute(t,commit=lambda _:events.append('duplicate')) == result


@pytest.mark.asyncio
async def test_uncertain_target_failure_isolates_and_never_restores_source():
    events=[]
    class Hooks:
        async def prepare(self,*a):return {}
        async def verify(self,*a):raise RuntimeError('target CUDA unknown')
        async def abort(self,*a):return {'target_stopped':False}
        async def isolate(self,*a):events.append('isolate')
        async def restore(self,*a):events.append('restore')
    tx=Reconfiguration(Hooks(),lambda event,**fields:None)
    t=Transition('bad',('source',),((0,1),),((0,),(1,)),overlap_memory_qualified=True)
    with pytest.raises(RuntimeError):await tx.execute(t,commit=lambda _:None)
    assert events==['isolate']
    with pytest.raises(RuntimeError,match='quarantined'):
        await tx.execute(Transition('another',('source',),((0,1),),((0,),(1,)),
            overlap_memory_qualified=True),commit=lambda _:None)


@pytest.mark.asyncio
async def test_prepare_partial_retirement_restores_even_before_global_freeze():
    events=[]
    class Hooks:
        async def prepare(self,*args):events.append('retire-overlap');raise RuntimeError('target initialization failed')
        async def abort(self,*args):return dict(target_stopped=True,source_restore_required=True)
        async def restore(self,*args):events.append('restore-retired')
        async def isolate(self,*args):events.append('isolate')
    tx=Reconfiguration(Hooks(),lambda event,**fields:None)
    with pytest.raises(RuntimeError):
        await tx.execute(Transition('partial',('a','b'),((0,),(1,)),((1,2),),overlap_memory_qualified=True),commit=lambda _:None)
    assert events==['retire-overlap','restore-retired']


def test_sparse_profile_requires_all_interpolation_corners():
    rows=[dict(role='mixed',tp=1,frequency_mhz=2520,input_tokens=n,context_tokens=c,batch=1,
        prefill_s=.01,iteration_s=.02,power_w=100,samples=1,source_sha256='a'*64)
        for n,c in ((16,64),(32,128))]
    with pytest.raises(CoverageError,match='corners'):PaperProfiles(rows).query(1,2520,24,96,1)
