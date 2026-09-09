from dataclasses import replace

from ecopadg.serving.dynamo import DynamoScheduler,dominates
from test_planner import system


def setup():
    planner,snapshot,request=system()
    snapshot=replace(snapshot,instances=tuple(replace(i,role='mixed') for i in snapshot.instances))
    policy=DynamoScheduler(planner.profiles,{'0':'SS','1':'MM','2':'LL'},clock_settle_s=0)
    return policy,snapshot,request


def test_shape_routing_keeps_manager_frequency_and_spills_only_larger():
    policy,snapshot,request=setup()
    plan=policy.plan(snapshot,(request,),now=10)
    assert plan.feasible and plan.routes[0].decode_id=='0'
    assert all(a.frequency_mhz==2520 for a in plan.frequencies)
    saturated=replace(snapshot,instances=tuple(replace(i,free_kv_tokens=0)
                      if i.instance_id=='0' else i for i in snapshot.instances))
    plan=policy.plan(saturated,(request,),now=10)
    assert plan.feasible and plan.routes[0].decode_id in ('1','2')
    assert dominates('ML','SL') and not dominates('MS','SL')


def test_forecast_is_causal_and_membership_does_not_move_live_requests():
    policy,snapshot,request=setup()
    policy.arrival(replace(request,arrival_s=20))
    assert policy.forecast(19)=={}
    assert policy.forecast(20)['SS']['rate']==.1
    busy=replace(snapshot,instances=(replace(snapshot.instances[0],requests=(request,),running=1),
                                      *snapshot.instances[1:]))
    assignments=policy.resident_reassignment(busy,10)
    assert assignments['0']=='SS'
    assert set(assignments)==set(policy.assignments)
    assert 'LL' in assignments.values()


def test_scale_frequency_checks_individual_deadlines_and_restores_capacity():
    policy,snapshot,request=setup()
    late=replace(request,emitted=2,first_token_s=9,last_token_s=9.1)
    busy=replace(snapshot,instances=(replace(snapshot.instances[0],requests=(late,),running=1),))
    plan=policy.frequency_plan(busy,10)
    assert plan.frequencies[0].frequency_mhz==2520
