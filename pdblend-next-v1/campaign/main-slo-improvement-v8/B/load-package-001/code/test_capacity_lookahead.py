"""Synthetic causal forecasts; no future trace or measured claims in fixtures."""
import sys
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import capacity_planner as p
from capacity_runtime import CapacityService

def fixture(now=1000.):
 identity=p.Identity('1'*64,'2'*64,'sha256:'+'3'*64,'4'*64,1);e=p.Evidence(identity,'5'*64);domain='6'*64
 planner=p.CapacityPlanner(identity,[p.LayoutBound(((0,),(1,)),domain,2.5,e),p.LayoutBound(((0,),(1,),(2,)),domain,3.5,e)],
   [p.TransitionBound('restore_cold',(2,),49.,30000.,e,1000)],[],p.Policy(min_residents=2))
 snapshot=p.Snapshot(identity,1,tuple(p.Resident(str(i),(i,),now,1,1.,removable=False) for i in (0,1)),(p.Spare((2,),now,now-31,10000),))
 return planner,snapshot,domain,now

def test_upper_only_single_arrival_spike_does_not_start_extra():
 planner,snapshot,domain,now=fixture();service=object.__new__(CapacityService);service.module=p
 request=SimpleNamespace(arrival_s=.5,input_tokens=10,output_limit=10,ttft_s=1.,tpot_s=.1)
 service.controller=SimpleNamespace(arrival_history=[request],active={},history_started_s=0.)
 service.binding=dict(demand_domains=[dict(sha256=domain,max_input_tokens=10,max_output_limit=10,max_context_tokens=20,slo_ttft_s=1.,slo_tpot_s=.1)])
 state=p.State()
 for elapsed in [1.,2.,3.,6.]:
  demand=service.demand(elapsed);demand=replace(demand,observed_at_s=now)
  result=planner.choose(snapshot,demand,state,now);state=result.state
  assert result.proposal is None
  assert not planner.growth_signals(snapshot,demand,2.5,now)['capacity_deficit']
  assert demand.rate_trend_lower_rps2==0.


def test_observed_rising_rate_uses_real_cold_bound_before_present_headroom_threshold():
 planner,snapshot,domain,now=fixture()
 demand=p.Demand(now,60.,1.2,1.6,domain,recent_rate_lower_rps=1.5,rate_trend_lower_rps2=.03)
 assert demand.rate_upper_rps < planner.policy.up_utilization*2.5
 first=planner.choose(snapshot,demand,p.State(),now)
 assert first.reason=='growth_hysteresis' and first.state.high_since_s==now
 later=replace(snapshot,residents=tuple(replace(i,observed_at_s=now+5) for i in snapshot.residents),spares=tuple(replace(i,observed_at_s=now+5) for i in snapshot.spares))
 result=planner.choose(later,replace(demand,observed_at_s=now+5),first.state,now+5)
 assert result.proposal.reason=='cold_start_headroom' and result.proposal.slo_during_transition_guaranteed is False
 fast=p.CapacityPlanner(planner.identity,planner.layouts,[replace(planner.transitions[0],duration_upper_s=1.)],[],planner.policy)
 assert fast.choose(snapshot,demand,p.State(),now).proposal is None
 assert fast.choose(snapshot,demand,p.State(),now).state.high_since_s is None


def test_capacity_lower_or_persistent_queue_bypasses_only_cooldown_not_minimum_off():
 planner,snapshot,domain,now=fixture();d=p.Demand(now,60.,3.,4.,domain)
 assert planner.choose(snapshot,d,p.State(last_change_s=now-1),now).proposal
 young=replace(snapshot,spares=(replace(snapshot.spares[0],unallocated_since_s=now-29.99),))
 assert planner.choose(young,d,p.State(),now).proposal is None
 one=p.Demand(now,60.,0.,1.,domain,1,.1,pending_ttft_remaining_s=.01)
 assert not planner.growth_signals(snapshot,one,2.5,now)['pending_deadline_risk']
 two=replace(one,queued_requests=2)
 result=planner.choose(snapshot,two,p.State(last_change_s=now-1),now)
 assert result.proposal.reason=='pending_deadline_risk'
 assert planner.choose(young,two,p.State(),now).proposal is None


def test_recent_trend_uses_only_two_completed_past_intervals_not_future_arrivals():
 _,_,domain,_=fixture();service=object.__new__(CapacityService);service.module=p
 def req(t):return SimpleNamespace(arrival_s=t,input_tokens=10,output_limit=10,ttft_s=1.,tpot_s=.1)
 service.controller=SimpleNamespace(arrival_history=[req(92.)]+[req(96.+i*.1) for i in range(25)]+[req(105.)],active={},history_started_s=0.)
 service.binding=dict(demand_domains=[dict(sha256=domain,max_input_tokens=10,max_output_limit=10,max_context_tokens=20,slo_ttft_s=1.,slo_tpot_s=.1)])
 d=service.demand(100.);assert d.recent_rate_lower_rps>0 and d.rate_trend_lower_rps2>0
 service.controller.arrival_history.pop();assert service.demand(100.)==d
