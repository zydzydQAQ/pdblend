import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
from capacity_executor import sha
from capacity_runtime import CapacityService,load_planner,bounded_policy


def module():
    path=Path(__file__).resolve().parents[2]/'capacity-controller-v2/planner.py'
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
