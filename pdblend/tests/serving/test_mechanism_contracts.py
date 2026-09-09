"""Executed CPU contracts. None certifies hardware batching or KV boundaries."""
import ast
from dataclasses import replace
from pathlib import Path

from ecopadg.serving.baselines import DistServeSearch,DynamoLLMPolicy
from ecopadg.serving.dynamo import DynamoScheduler,SHAPES,dominates
from ecopadg.serving.ecoserve import EcoServeScheduler
from ecopadg.serving.profiles import ProfileStore,OutputPredictor
from ecopadg.serving.planner import JointPlanner,TransferCost
from test_planner import system


def test_distserve_independent_tp_and_instance_search():
    planner,_,_=system()
    p=replace(planner.profiles.points[0],role='prefill',tp=1,frequency_mhz=2520,batch=1,prefill_s=.05)
    d=replace(p,role='decode',tp=2,batch=4,iteration_s=.04)
    search=DistServeSearch(ProfileStore([p,d]),[TransferCost(1,2,4096,.01,1,'cpu-fixture',True,profile_batch=4)])
    choices=search.search(128,64,5,.1,0,{2:100000})
    assert choices and {(c.prefill_tp,c.decode_tp) for c in choices}=={(1,2)}
    assert len({c.prefill_count for c in choices})>1 and len({c.decode_count for c in choices})>1
    assert any(c.prefill_count!=c.decode_count for c in choices)
    assert all(c.prefill_batch==1 and c.decode_batch==4 and sum(map(len,c.gpus))<=8 for c in choices)


def test_dynamo_causal_length_prediction():
    predictor=OutputPredictor(256)
    assert predictor.predict(128)==256
    predictor.observe_completed(128,64)
    assert predictor.predict(128)==64 and predictor.predict(8192)==256
    # Only completed observations enter the predictor; future arrival records
    # and realized lengths have no online API argument.
    import inspect
    assert tuple(inspect.signature(predictor.predict).parameters)==('input_tokens',)
    from ecopadg.serving import profiles
    tree=ast.parse((Path(profiles.__file__).parent/'runtime.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='predictor' and n.func.attr=='predict']
    assert len(calls)==1 and len(calls[0].args)==1
    assert ast.dump(calls[0].args[0])==ast.dump(ast.parse('input_tokens',mode='eval').body)
    policy=DynamoLLMPolicy();policy.observe_arrival(20,128,64)
    assert policy.forecast(19)['SS']==0 and policy.forecast(20)['SS']>0


def test_dynamo_nine_logical_pools_and_fragmentation():
    planner,snapshot,request=system()
    assert set(SHAPES)=={a+b for a in 'SML' for b in 'SML'}
    classifier=DynamoLLMPolicy()
    assert {classifier.classify(i,o) for i in (32,512,4096) for o in (32,256,1024)}==set(SHAPES)
    snapshot=replace(snapshot,instances=tuple(replace(i,role='mixed') for i in snapshot.instances))
    policy=DynamoScheduler(planner.profiles,{'0':'SS','1':'MM','2':'LL'},clock_settle_s=0)
    saturated=replace(snapshot,instances=(replace(snapshot.instances[0],free_kv_tokens=0),*snapshot.instances[1:]))
    plan=policy.plan(saturated,(request,),now=10)
    assert plan.feasible and plan.routes[0].decode_id in ('1','2')
    assert not dominates('LS','SL') and dominates('LL','SL')
    busy=replace(snapshot,instances=(replace(snapshot.instances[0],requests=(request,),running=1),*snapshot.instances[1:]))
    updated=policy.resident_reassignment(busy,10)
    assert updated['0']=='SS' and set(updated)==set(policy.assignments) and 'LL' in updated.values()


def test_eco_joint_ttft_credit_and_kv_constraints():
    planner,snapshot,request=system()
    policy=EcoServeScheduler(planner.profiles,['0'],lower=1,upper=2)
    older=(replace(request,request_id='a',arrival_s=0,first_token_s=9,emitted=10),
        replace(request,request_id='b',arrival_s=0,first_token_s=9,emitted=30))
    instance=replace(snapshot.instances[0],requests=older)
    assert policy.feasible(instance,request,10,9) is not None
    assert policy.feasible(instance,replace(request,ttft_s=.001),10,9) is None
    assert policy.feasible(replace(instance,free_kv_tokens=0),request,10,9) is None
    no_credit=replace(instance,requests=tuple(replace(r,emitted=1) for r in older))
    assert policy.feasible(no_credit,request,10,9) is None


def test_mixed_frequency_uses_feasible_energy_minimum():
    planner,snapshot,request=system()
    planner=JointPlanner(planner.profiles,allow_pd=False)
    snapshot=replace(snapshot,instances=(snapshot.instances[0],))
    plan=planner.plan(snapshot,(request,),now=10,joint=False)
    assert plan.feasible and any(a.frequency_mhz==1500 for a in plan.frequencies)
    assert all(a.frequency_mhz!=900 for a in plan.frequencies)
    late=replace(request,ttft_s=.0001)
    assert not planner.plan(snapshot,(late,),now=10,joint=False).feasible
