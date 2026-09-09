"""Actual production scheduler and Controller drain guard, CPU only."""
import asyncio, ast, collections, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
C=Path(__file__).resolve().parent; REPO=C.parents[2]
HOST=REPO/'releases/five-system100-A14B-baseline-eco-drain-v1-runtime'
sys.path[:0]=[str(HOST/'src'),str(HOST),str(REPO/'tests/serving')]
from ecopadg.serving.runtime import Controller
from ecopadg.serving.ecoserve import EcoServeScheduler
from ecopadg.serving.state import StateManager, ExpiredPlan, StalePlan
from ecopadg.serving.types import WindowAction
from test_planner import system

async def checks():
    p,s,r=system();now=time.time();r=replace(r,arrival_s=now,hard_deadline_s=now+120)
    old=replace(r,request_id='accepted-old')
    states=tuple(replace(i,role='mixed',mode='temporal',timestamp_s=now,
        admit_prefill=i.instance_id=='0') for i in s.instances)
    states=(replace(states[0],free_kv_tokens=0,waiting=1,requests=(old,)),*states[1:])
    s=replace(s,timestamp_s=now,instances=states)
    eco=EcoServeScheduler(p.profiles,['0','1','2']);plan=eco.plan(s,(r,),now=now)
    assert plan.feasible and plan.routes[0].decode_id=='1'
    assert any(a.instance_id=='0' and not a.admit_prefill for a in plan.windows)
    ctrl=object.__new__(Controller);ctrl.state=StateManager(s)
    ctrl.action_lock=asyncio.Lock();ctrl.planning_stats=SimpleNamespace(counts=collections.Counter())
    passed=['original_scheduler_reproduces_unsafe_off']
    async def test(name,snap,expect,selected=plan):
        ctrl.state.snapshot=snap;before=ctrl.state.snapshot
        try:
            async with ctrl.action_lock:await ctrl.eco_require_prefill_drained(selected)
            actual=True
        except ExpiredPlan:actual=False
        assert actual is expect,name
        assert ctrl.state.snapshot==before and r.hard_deadline_s==now+120
        assert not ctrl.action_lock.locked()
        passed.append(name)
    await test('accepted_prefill_blocks_off_without_reservation',s,False)
    await test('native_waiting_with_prior_first_token_blocks_off',replace(s,instances=(replace(states[0],requests=(replace(old,first_token_s=now,emitted=1),)),*states[1:])),False)
    await test('first_token_absent_blocks_even_native_waiting_zero',replace(s,instances=(replace(states[0],waiting=0),*states[1:])),False)
    drained=replace(s,instances=(replace(states[0],waiting=0,requests=(replace(old,first_token_s=now,emitted=1),)),*states[1:]))
    await test('drained_prefill_allows_original_switch_and_decode',drained,True)
    await test('empty_old_window_allows_switch',replace(s,instances=(replace(states[0],waiting=0,requests=()),*states[1:])),True)
    await test('stale_state_fails_closed',replace(drained,instances=(replace(drained.instances[0],timestamp_s=now-2),*states[1:])),False)
    await test('future_state_fails_closed',replace(drained,instances=(replace(drained.instances[0],timestamp_s=now+2),*states[1:])),False)
    await test('missing_instance_fails_closed',replace(s,instances=states[1:]),False)
    await test('opening_window_has_no_new_restriction',s,True,replace(plan,windows=(WindowAction('1',0,True),)))
    await test('no_window_action_unchanged',s,True,replace(plan,windows=()))
    assert issubclass(ExpiredPlan,StalePlan);passed.append('uses_original_stale_plan_requeue')
    src=(HOST/'src/ecopadg/serving/runtime.py').read_text();tree=ast.parse(src)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Controller')
    hooks=[]
    for method in cls.body:
        for node in ast.walk(method):
            if isinstance(node,ast.AsyncWith) and 'self.action_lock' in ast.unparse(node.items[0].context_expr):
                text=ast.unparse(node)
                if 'self.eco_require_prefill_drained(plan)' in text:hooks.append((method.name,text))
    assert len(hooks)==2
    admission=next(t for n,t in hooks if n!='eco_resize')
    assert admission.index('self.eco_require_prefill_drained(plan)')<admission.index('self.state.reserve(')
    resize=next(t for n,t in hooks if n=='eco_resize')
    assert resize.index('try:')<resize.index('self.eco_require_prefill_drained(plan)')<resize.index('self.backend.execute(plan)')
    assert 'except ExpiredPlan:' in resize and 'self.eco_scheduler.groups = list(before)' in resize and 'previous_selected' in resize
    passed.extend(['admission_action_lock_before_reservation','membership_try_and_original_rollback'])
    manifest=json.loads((HOST/'manifest.json').read_text());parent=Path(manifest['parent_release'])
    changed=[]
    for name,h in manifest['files'].items():
        assert hashlib.sha256((HOST/name).read_bytes()).hexdigest()==h
        if (HOST/name).read_bytes()!=(parent/name).read_bytes():changed.append(name)
    assert changed==['src/ecopadg/serving/runtime.py'];passed.append('all_policy_profile_native_backend_source_bytes_unchanged')
    return passed

if __name__=='__main__':
    results=asyncio.run(checks());print(json.dumps({'passed':True,'count':len(results),'checks':results,'host':str(HOST),'manifest_sha256':hashlib.sha256((HOST/'manifest.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
