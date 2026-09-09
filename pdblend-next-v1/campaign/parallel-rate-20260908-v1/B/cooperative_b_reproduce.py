"""Bounded CPU proof using exact frozen dispatch/queue/clock/state method ASTs.
No Controller initialization, vLLM, HTTP, GPU, real clock owner or serving source edit.
"""
import ast,asyncio,hashlib,importlib.util,json,time,types
from collections import Counter
from dataclasses import replace,asdict
from pathlib import Path
from types import SimpleNamespace as N

ROOT=Path('/root/workspace/pdblend-next-v1/campaign/cooperative-admission-yield-v1')
SOURCE=ROOT.parents[1]/'releases/five-system100-B32B-v1-runtime/src/ecopadg/serving'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(p,name):
    import sys
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def method(path,cls,name,namespace):
    tree=ast.parse(path.read_text());c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    fn=next(x for x in c.body if isinstance(x,ast.AsyncFunctionDef) and x.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]

async def case(queue_path,*,calls=1500,pending=160,cost_s=.00010):
    qmod=module(queue_path,'cooperative_queue_'+str(time.time_ns()));t=module(SOURCE/'types.py','cooperative_types_'+str(time.time_ns()))
    obs=module(SOURCE/'observe.py','cooperative_observe_'+str(time.time_ns()))
    class StalePlan(RuntimeError):pass
    class ExpiredPlan(StalePlan):pass
    ns=dict(asyncio=asyncio,time=time,replace=replace,asdict=asdict,ExpiredPlan=ExpiredPlan,StalePlan=StalePlan,
        AdmissionRejected=qmod.AdmissionRejected)
    dispatch=method(SOURCE/'runtime.py','Controller','dispatch',ns)
    apply_clocks=method(SOURCE/'state.py','StateManager','apply_clocks',dict(asyncio=asyncio,replace=replace))
    set_clock=method(SOURCE/'backend.py','ClockOwner','set',dict(asyncio=asyncio,time=time))
    backend_ns=dict(time=time,ExpiredPlan=ExpiredPlan)
    execute=method(SOURCE/'backend.py','HttpEngineBackend','execute',backend_ns)
    execute_inner=method(SOURCE/'backend.py','HttpEngineBackend','_execute',backend_ns)
    now=time.time();states=tuple(t.InstanceState('i'+str(j),'mixed',1,(j,),now,1,2520,100000,0,0) for j in range(8))
    state=N(snapshot=t.RuntimeSnapshot(0,now,states),lock=asyncio.Lock())
    state.apply_clocks=types.MethodType(apply_clocks,state)
    clock=N(gpus=tuple(range(8)),epochs=Counter(),lock=asyncio.Lock(),applied={j:2520 for j in range(8)},
        deferred={},fallbacks={},settle_timeout_s=.5)
    clock.set=types.MethodType(set_clock,clock)
    backend=N(clocks=clock,instances={i.instance_id:{'gpus':i.gpus} for i in states},frequency={i.instance_id:2520 for i in states},
        parked=set(),dvfs_resume_s={},frequency_outcomes=[],inflight_actions=0,last_action_finished_s=0.)
    backend.execute=types.MethodType(execute,backend);backend._execute=types.MethodType(execute_inner,backend)
    count=0;heartbeats=[];started=time.monotonic();cancelled=False
    class Policy:
        def plan(self,snapshot,pending,*,now):
            nonlocal count
            count+=1
            if count>calls:raise asyncio.CancelledError('bounded CPU reproducer complete')
            # Synthetic 100us CPU cost, compared with observed 76us planning avg.
            until=time.perf_counter()+cost_s
            while time.perf_counter()<until:pass
            return t.ControlPlan(snapshot.version,now,now+1,feasible=False,
                frequencies=tuple(t.FrequencyAction(i.instance_id,2520) for i in states),reason='CPU fixture: infeasible/no route')
    c=N(pending=qmod.AdmissionQueue(256),active={},evaluation_v3=True,strategy='dynamollm-resident',
        state=state,backend=backend,action_lock=asyncio.Lock(),eco_scheduler=None,dynamo_scheduler=Policy(),
        distserve_scheduler=None,planning_stats=obs.PlanningStats(),config={'allow_unprofiled_fallback':False},failure=None)
    async def recovery(*a):return None
    c.recovery_plan=recovery
    c.record_idle_tail_fallback=types.MethodType(method(SOURCE/'runtime.py','Controller','record_idle_tail_fallback',dict(time=time,asdict=asdict)),c)
    for j in range(pending):
        rid=str(j);budget=t.RequestBudget(rid,now,38,314,1.,.1,hard_deadline_s=now+30)
        c.active[rid]={'budget':budget,'future':asyncio.get_running_loop().create_future(),'route':None}
        c.pending.put_nowait(rid)
    async def heartbeat():
        while True:
            await asyncio.sleep(.001);heartbeats.append(time.monotonic()-started)
    h=asyncio.create_task(heartbeat());await asyncio.sleep(0)
    try:await dispatch(c)
    except asyncio.CancelledError:cancelled=True
    elapsed=time.monotonic()-started;h.cancel();await asyncio.gather(h,return_exceptions=True)
    return dict(calls=count-1,pending=pending,synthetic_cpu_cost_s=cost_s,elapsed_s=elapsed,
        heartbeat_count_before_dispatch_return=len(heartbeats),first_heartbeat_s=heartbeats[0] if heartbeats else None,
        max_heartbeat_gap_s=max((b-a for a,b in zip([0.]+heartbeats,heartbeats+[elapsed])),default=elapsed),
        actual_clock_changes=0,all_current_clock_commands=2520,bounded_stop=cancelled,
        original_dispatch_failure=c.failure,completed_plans=c.planning_stats.counts['completed_calls'],
        frozen_method_files={str(SOURCE/n):sha(SOURCE/n) for n in ('runtime.py','backend.py','state.py','types.py','observe.py')},
        queue_path=str(queue_path),queue_sha256=sha(queue_path),hardware_executed=False)

def main():
    old=asyncio.run(case(SOURCE/'admission.py'));new=asyncio.run(case(ROOT/'admission.py'))
    result=dict(schema=1,original=old,candidate=new,scope='CPU starvation counterexample, not a replay of actual B request scheduling')
    print(json.dumps(result,indent=2));(Path(__file__).resolve().parent/'cooperative-b-cpu-reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
