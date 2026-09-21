import asyncio
import time
from types import SimpleNamespace

import pytest

from pdblend2.control.controller import Controller
from pdblend2.control.planner import SLO, Plan, PlannerConfig, PoolPlanner
from pdblend2.control.shield import Shield
from pdblend2.proxy.router import RequestRecord, Router

from synthetic import synthetic_model


class FakeInstance:
    def __init__(self, iid, gpu):
        self.spec = SimpleNamespace(instance_id=iid, gpus=(gpu,))
        self.state = "ready"
        self.calls = []

    def start(self): self.calls.append("start"); self.state = "ready"
    def wait_ready(self): return 0.1
    def stop(self, timeout_s=30.0): self.calls.append("stop"); self.state = "off"
    def sleep(self, level=1): self.calls.append(f"sleep{level}"); self.state = f"L{level}"; return 0.1
    def wake_up(self): self.calls.append("wake"); self.state = "ready"; return 0.1


class FakeFleet:
    def __init__(self, n):
        self.instances = {f"i{k}": FakeInstance(f"i{k}", k) for k in range(n)}

    def __getitem__(self, iid): return self.instances[iid]


class FakeGpus:
    def __init__(self): self.clocks, self.parked = {}, set()
    def set_clock(self, g, mhz): self.clocks[g] = mhz
    def reset_clock(self, g): self.clocks[g] = None
    def park(self, g): self.parked.add(g)
    def unpark(self, g): self.parked.discard(g)


def make_controller(n=4, shield=True, **cfg):
    fleet = FakeFleet(n)
    router = Router(list(fleet.instances))
    planner = PoolPlanner(synthetic_model(), PlannerConfig(n, SLO(5.0, 0.15), **cfg))
    ctl = Controller(fleet, router, FakeGpus(), planner, Shield(SLO(5.0, 0.15)) if shield else None,
                     period_s=0.05, tick_s=0.01, drain_timeout_s=0.2)
    return ctl, fleet, router


def test_execute_parks_and_sets_clocks():
    ctl, fleet, router = make_controller()
    plan = Plan({"P": 1, "D": 1, "M": 0, "L1": 1, "off": 1}, 2520, 1200, 2520, 0, 300.0, 0.5, 0.05)
    asyncio.run(ctl.execute(plan))
    assert sorted(ctl.roles.values()) == ["D", "L1", "P", "off"]
    roles = router.roles()
    assert sorted(roles.values()) == ["D", "P", "parked", "parked"]
    d = next(i for i, r in ctl.roles.items() if r == "D")
    assert ctl.gpus.clocks[fleet[d].spec.gpus[0]] == 1200
    assert next(fleet[i].spec.gpus[0] for i, r in ctl.roles.items() if r == "L1") in ctl.gpus.parked
    assert "stop" in next(fleet[i].calls for i, r in ctl.roles.items() if r == "off")


def test_execute_wakes_from_off_and_reroutes():
    ctl, fleet, router = make_controller()
    asyncio.run(ctl.execute(Plan({"M": 2, "off": 2}, 2520, 2520, 1500, 0, 1, 1, 1)))
    asyncio.run(ctl.execute(Plan({"P": 1, "D": 3}, 2520, 1800, 2520, 0, 1, 1, 1)))
    assert sorted(ctl.roles.values()) == ["D", "D", "D", "P"]
    assert all(fleet[i].state == "ready" for i in fleet.instances)
    assert sum("start" in fleet[i].calls for i in fleet.instances) == 2
    assert any(r["kind"] == "reroute" for r in ctl._log)


def test_run_loop_replans_and_stops():
    ctl, fleet, router = make_controller(shield=False)
    for _ in range(20):
        ctl.forecaster.arrive(512)

    async def go():
        stop = asyncio.Event()
        task = asyncio.create_task(ctl.run(stop))
        await asyncio.sleep(0.3)
        stop.set()
        await task
    asyncio.run(go())
    kinds = ctl.summary()["events"]
    assert kinds.get("plan", 0) >= 1 and kinds.get("forecast", 0) >= 2 and kinds.get("stop") == 1


def test_run_loop_fails_open_with_empty_forecast():
    ctl, fleet, router = make_controller()

    async def go():
        stop = asyncio.Event()
        task = asyncio.create_task(ctl.run(stop))
        await asyncio.sleep(0.12)               # several replan periods, still zero arrivals
        stop.set()
        await task
    asyncio.run(go())
    assert sorted(ctl.roles.values()) == ["M"] * 4
    assert all(mhz == 2520 for mhz in ctl.freqs.values())
    plans = [r for r in ctl._log if r["kind"] == "plan"]
    assert plans and all(r.get("cold_start") for r in plans)


def test_run_loop_uses_planner_once_traffic_seen():
    ctl, fleet, router = make_controller(shield=False)
    ctl.min_warm_s, ctl.min_warm_samples = 0.05, 5
    for _ in range(20):
        ctl.forecaster.arrive(512)

    async def go():
        stop = asyncio.Event()
        task = asyncio.create_task(ctl.run(stop))
        await asyncio.sleep(0.12)
        stop.set()
        await task
    asyncio.run(go())
    plans = [r for r in ctl._log if r["kind"] == "plan"]
    assert plans and plans[0].get("cold_start")            # the loop always opens fail-open
    assert not plans[-1].get("cold_start")                 # then hands over to the planner


def test_run_loop_stays_fail_open_during_warmup_window():
    ctl, fleet, router = make_controller(shield=False)   # defaults: 20 s window, 30 samples
    for _ in range(100):
        ctl.forecaster.arrive(512)

    async def go():
        stop = asyncio.Event()
        task = asyncio.create_task(ctl.run(stop))
        await asyncio.sleep(0.12)
        stop.set()
        await task
    asyncio.run(go())
    plans = [r for r in ctl._log if r["kind"] == "plan"]
    assert plans and all(r.get("cold_start") for r in plans)
    assert all(r["counts"] == {"M": 4} for r in plans)


def test_shield_escalates_on_slow_ttft_and_decays():
    slo = SLO(1.0, 0.1)
    shield = Shield(slo, cooldown_s=5.0)
    now = 1000.0
    recs = [RequestRecord(f"r{i}", "M", "i0", "i0", 100, 16, now - 2.0, first_token_s=now - 1.1,
                          finished_s=now - 0.5, completion_tokens=16) for i in range(10)]
    p = shield.observe(recs, now)
    assert p.prefill and not p.decode
    assert shield.update(p, now) == 1
    assert shield.update(p, now + 1) == 1       # rate limited
    assert shield.update(p, now + 3) == 2
    plan = Plan({"M": 2, "L1": 1, "off": 1}, 2520, 1500, 1200, 0, 1, 1, 1)
    boosted = shield.apply(plan, p, 2520)
    assert boosted.counts["M"] == 3 and boosted.counts.get("L1", 0) == 0 and boosted.f_M == 2520
    calm = shield.observe([], now + 4)
    assert shield.update(calm, now + 4) == 2    # still within cooldown
    assert shield.update(calm, now + 9) == 1    # decays one step per quiet cooldown
    assert shield.update(calm, now + 14) == 0   # reaching zero starts the hold-down probe (floor 3)
    assert shield.floor_active == 3


def _hot(t, n=10):
    return [RequestRecord(f"r{t}-{i}", "M", "i0", "i0", 100, 16, t - 2.0, first_token_s=t - 1.1,
                          finished_s=t - 0.5, completion_tokens=16) for i in range(n)]


def test_shield_holddown_floor_ratchets_down_and_restores_on_pressure():
    slo = SLO(1.0, 0.1)
    shield = Shield(slo, cooldown_s=5.0)
    now = 1000.0
    p = shield.observe(_hot(now), now)
    assert shield.update(p, now) == 1
    assert shield.update(p, now + 2) == 2
    plan = Plan({"M": 2, "L1": 1, "off": 1}, 2520, 1500, 1200, 0, 1, 1, 1)
    assert shield.apply(plan, p, 2520).counts["M"] == 3     # floor learned: 3 active
    calm = shield.observe([], now + 3)
    assert shield.update(calm, now + 7) == 1
    assert shield.update(calm, now + 12) == 0               # probe window starts
    lean = Plan({"M": 1, "L1": 3}, 2520, 1500, 900, 0, 1, 1, 1)
    held = shield.apply(lean, calm, 2520)
    assert held.counts["M"] == 3 and held.f_M == 900        # floor kept, planner clocks kept
    assert shield.update(calm, now + 17) == 0               # quiet probe: floor 3 -> 2
    assert shield.apply(lean, calm, 2520).counts["M"] == 2
    p2 = shield.observe(_hot(now + 18), now + 18)           # probe fails
    assert shield.update(p2, now + 18) == 1                 # floor restored, backoff x2
    assert shield.floor_active == 3 and shield.probe_windows == 2.0
    calm2 = shield.observe([], now + 19)
    assert shield.update(calm2, now + 23) == 0
    assert shield.update(calm2, now + 32) == 0              # backed-off 10 s window still holds
    assert shield.update(calm2, now + 33) == 0              # window over: floor 3 -> 2
    assert shield.floor_active == 2


def test_shield_probe_pressure_at_episode_peak_adds_no_backoff():
    slo = SLO(1.0, 0.1)
    shield = Shield(slo, cooldown_s=5.0)
    now = 1000.0
    p = shield.observe(_hot(now), now)
    assert shield.update(p, now) == 1
    shield.apply(Plan({"M": 2, "L1": 2}, 2520, 1500, 1200, 0, 1, 1, 1), p, 2520)
    assert shield.floor_active == 2 and shield.peak_active == 2
    calm = shield.observe([], now + 1)
    assert shield.update(calm, now + 5) == 0                # probe starts at floor == peak
    p2 = shield.observe(_hot(now + 6), now + 6)             # pressure at the peak capacity itself
    assert shield.update(p2, now + 6) == 1
    assert shield.floor_active == 2 and shield.probe_windows == 1.0


def test_shield_in_flight_tpot_pressure():
    slo = SLO(5.0, 0.1)
    shield = Shield(slo)
    now = 1000.0
    rec = RequestRecord("r", "PD", "i0", "i1", 100, 64, now - 3.0, first_token_s=now - 2.0)
    rec.tokens_so_far = 11                      # 10 gaps in 2 s -> 0.2 s/token
    p = shield.observe([rec], now)
    assert p.decode and not p.prefill
    plan = Plan({"P": 1, "D": 1, "off": 2}, 2520, 900, 2520, 0, 1, 1, 1)
    shield.update(p, now)
    shield.update(p, now + 2)
    boosted = shield.apply(plan, p, 2520)
    assert boosted.counts["D"] == 2 and boosted.counts["P"] == 1 and boosted.f_D == 2520
