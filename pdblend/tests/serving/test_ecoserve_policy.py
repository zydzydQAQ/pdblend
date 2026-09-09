from dataclasses import replace

from ecopadg.serving.ecoserve import EcoServeScheduler
from test_planner import system


def test_rolling_activation_stays_then_advances_and_closes_previous_window():
    planner,snapshot,r=system()
    states=tuple(replace(i,role='mixed',mode='temporal',admit_prefill=i.instance_id=='0') for i in snapshot.instances)
    snapshot=replace(snapshot,instances=states)
    policy=EcoServeScheduler(planner.profiles,['0','1','2'])
    first=policy.plan(snapshot,(r,),now=10)
    assert first.routes[0].decode_id=='0'
    policy.committed(first,10)
    assert policy.plan(snapshot,(replace(r,request_id='next'),),now=10.01).routes[0].decode_id=='0'
    snapshot=replace(snapshot,instances=(replace(states[0],free_kv_tokens=0),*states[1:]))
    second=policy.plan(snapshot,(r,),now=10.1)
    assert second.routes[0].decode_id=='1'
    assert {(a.instance_id,a.admit_prefill) for a in second.windows}=={('0',False),('1',True)}


def test_paper_uses_mean_saved_tpot_while_pdblend_checks_each_request():
    planner,snapshot,r=system()
    existing=(replace(r,request_id='a',arrival_s=0,first_token_s=9,emitted=10),
              replace(r,request_id='b',arrival_s=0,first_token_s=9,emitted=30))
    i=replace(snapshot.instances[0],role='mixed',requests=existing)
    policy=EcoServeScheduler(planner.profiles,['0'],lower=1,upper=2)
    assert policy.feasible(i,r,10,9) is not None
    point=planner.point(i,r,2520,3)
    assert not planner.safe_for_existing(i,point,10,prefill_delay=.1)


def test_mitosis_moves_handles_without_changing_running_requests():
    planner,snapshot,r=system()
    states=tuple(replace(snapshot.instances[0],instance_id=str(n),gpus=(n,),
                         requests=(r,) if n==0 else ()) for n in range(4))
    snapshot=replace(snapshot,instances=states)
    policy=EcoServeScheduler(planner.profiles,['0','1','2'],lower=2,upper=3)
    policy.add_instance('3')
    assert policy.groups==[('0','1'),('2','3')]
    assert policy.remove_idle_instance(snapshot)=='1'
    assert policy.groups==[('0','2','3')]
    assert snapshot.instances[0].requests==(r,)


def test_merge_closes_the_second_prefill_window_immediately():
    planner,snapshot,r=system()
    states=tuple(replace(snapshot.instances[0],instance_id=str(n),gpus=(n,),
        role='mixed',mode='temporal',admit_prefill=n in (0,2)) for n in range(4))
    snapshot=replace(snapshot,instances=states)
    policy=EcoServeScheduler(planner.profiles,['0','1','2','3'],lower=2,upper=3)
    policy.remove_idle_instance(snapshot)
    plan=policy.membership_plan(snapshot,10)
    assert any(a.instance_id=='2' and not a.admit_prefill for a in plan.windows)
