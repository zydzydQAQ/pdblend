import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
from capacity_executor import sha
from capacity_runtime import CapacityService,load_planner,bounded_policy


def module():
    path=Path(__file__).resolve().parent/'capacity_planner.py'
    return load_planner(dict(path=str(path),sha256=sha(path)))


def test_capacity_deficit_cannot_expose_spare_before_actual_minimum_off(monkeypatch):
    import capacity_runtime
    m=module();identity=m.Identity('1'*64,'2'*64,'sha256:'+'3'*64,'4'*64,2)
    now=[100.]
    monkeypatch.setattr(capacity_runtime.time,'time',lambda:now[0])
    service=object.__new__(CapacityService)
    service.module=m;service.identity=identity;service.last_spares={}
    service.planner=SimpleNamespace(policy=m.Policy(min_residents=2))
    service.controller=SimpleNamespace(state=SimpleNamespace(snapshot=SimpleNamespace(instances=[])),
        backend=SimpleNamespace(instances={},topology_version=0))
    async def gpu_state(free):
        return [dict(gpu=g,at_s=now[0],free_bytes=100,process_pids=[]) for g in free]
    service.backend=SimpleNamespace(gpu_state=gpu_state)
    service.inventory=SimpleNamespace(value=dict(transition_inflight=False))
    assert not asyncio.run(service.snapshot()).spares
    now[0]=129.99
    assert not asyncio.run(service.snapshot()).spares
    now[0]=130.
    assert len(asyncio.run(service.snapshot()).spares)==4


def test_shrink_payback_cannot_include_savings_past_deadline_or_restore_reserve():
    m=module();planner=SimpleNamespace(policy=m.Policy(min_residents=2),
        transitions=[SimpleNamespace(operation='restore_cold',duration_upper_s=50.)])
    assert bounded_policy(planner,1000.,100.,120.).amortization_horizon_s==600.
    assert bounded_policy(planner,1000.,700.,120.).amortization_horizon_s==130.
    assert bounded_policy(planner,1000.,830.,120.) is None
    assert planner.policy.amortization_horizon_s==600.


def test_empty_history_never_invents_domain_and_unknown_arrival_resets_previous():
    m=module();service=object.__new__(CapacityService);service.module=m
    service.binding=dict(rate_observation_window_s=60.,demand_domains=[dict(sha256='a'*64,
        max_input_tokens=512,max_output_limit=512,max_context_tokens=1024,slo_ttft_s=1.,slo_tpot_s=.1)])
    c=SimpleNamespace(arrival_history=[],active={},history_started_s=0.)
    service.controller=c
    assert service.demand(60.).domain_sha256=='0'*64
    request=SimpleNamespace(arrival_s=60.,input_tokens=100,output_limit=100,ttft_s=1.,tpot_s=.1)
    c.arrival_history=[request]
    assert service.demand(60.).domain_sha256=='a'*64
    c.arrival_history=[]
    idle=service.demand(121.)
    assert idle.domain_sha256=='a'*64 and idle.rate_lower_rps==idle.rate_upper_rps==0
    request.ttft_s=99.;request.arrival_s=122.;c.arrival_history=[request]
    assert service.demand(122.).domain_sha256=='0'*64
    c.arrival_history=[]
    assert service.demand(183.).domain_sha256=='0'*64


def test_revalidation_recomputes_latest_rate_domain_and_payback(monkeypatch):
    import capacity_runtime
    from dataclasses import replace
    m=module();now=1000.;monkeypatch.setattr(capacity_runtime.time,'time',lambda:now)
    identity=m.Identity('1'*64,'2'*64,'sha256:'+'3'*64,'4'*64,1)
    evidence=m.Evidence(identity,'6'*64);domain='5'*64
    current=((0,),(1,),(2,));target=((0,),(1,))
    planner=m.CapacityPlanner(identity,[m.LayoutBound(current,domain,15.,evidence),
        m.LayoutBound(target,domain,10.,evidence)],
        [m.TransitionBound('remove',(2,),1.,100.,evidence),
         m.TransitionBound('restore_cold',(2,),30.,5000.,evidence,1000)],
        [m.SavingsBound(current,target,domain,0.,6.,100.,evidence)],m.Policy(min_residents=2))
    snapshot=m.Snapshot(identity,9,tuple(m.Resident(f'r{i}',(i,),now,7,0.) for i in range(3)),())
    demand=[m.Demand(now,120.,1.,2.,domain)];state=m.State(low_since_s=now-100,layout_groups=current)
    proposal=planner.choose(snapshot,demand[0],state,now).proposal
    assert proposal.action=='remove' and m.revalidate(proposal,snapshot,now)
    service=object.__new__(CapacityService);service.module=m;service.identity=identity;service.planner=planner
    service.state=state;service.previous_domain=domain;service.binding=dict(deadline_s=5000.)
    service.executor=SimpleNamespace(cleanup_reserve_s=120.)
    async def observe():return snapshot
    service.snapshot=observe;service.demand=lambda now:demand[0]
    assert asyncio.run(service.revalidate(proposal))
    demand[0]=replace(demand[0],rate_upper_rps=14.)
    assert not asyncio.run(service.revalidate(proposal))
    demand[0]=replace(demand[0],rate_upper_rps=2.,domain_sha256='a'*64)
    assert not asyncio.run(service.revalidate(proposal))
    demand[0]=replace(demand[0],domain_sha256=domain);service.binding['deadline_s']=1170.
    assert not asyncio.run(service.revalidate(proposal))


def test_cold_readiness_risk_uses_only_actual_pending_deadline():
    m=module();service=object.__new__(CapacityService);service.module=m
    service.controller=SimpleNamespace(active={'pending':dict(route=None,
        budget=SimpleNamespace(arrival_s=990.,ttft_s=15.),future=SimpleNamespace(done=lambda:False))})
    service.planner=SimpleNamespace(capacity=lambda layout,d:(10.,None),
        transitions=[SimpleNamespace(operation='restore_cold',gpus=(2,),duration_upper_s=50.)])
    snapshot=SimpleNamespace(residents=[SimpleNamespace(gpus=(0,)),SimpleNamespace(gpus=(1,))],
        spares=[SimpleNamespace(gpus=(2,))])
    risk=service.startup_risk(snapshot,SimpleNamespace(rate_upper_rps=12.),1000.)
    assert risk['already_expired_ttft']==0 and risk['ttft_deadline_before_earliest_cold_ready']==1
    assert risk['earliest_qualified_cold_ready_estimate_s']==1050. and not risk['slo_recovery_guaranteed']
