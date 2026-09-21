"""Control loop: observe -> forecast -> plan (every period) -> shield (every second) -> act.

Actions are role-table rewrites (free), clock changes (~100 ms), HBM/SM clock parking (L1) and process
stop/start (off). Everything is logged as JSONL so a run can be audited afterwards.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..bench.metering import Gpus
from ..engine.launcher import Fleet
from ..proxy.router import Router
from .forecast import Forecaster
from .planner import ACTIVE, PARKED, Plan, PoolPlanner, assign_roles
from .shield import Shield

ROUTER_ROLE = {"P": "P", "D": "D", "M": "M", "idle": "parked", "L1": "parked", "off": "parked"}


@dataclass
class Controller:
    fleet: Fleet
    router: Router
    gpus: Gpus
    planner: PoolPlanner
    shield: Optional[Shield] = None
    forecaster: Forecaster = field(default_factory=Forecaster)
    period_s: float = 10.0
    tick_s: float = 1.0
    drain_timeout_s: float = 30.0
    log_path: Optional[Path] = None
    initial_plan: Optional[Plan] = None
    freeze: bool = False                     # static policies: execute initial_plan once, never replan
    min_warm_s: float = 20.0                 # fail-open until this long after first traffic ...
    min_warm_samples: int = 30               # ... and at least this many arrivals were seen
    roles: dict = field(default_factory=dict)
    freqs: dict = field(default_factory=dict)
    plan_now: Optional[Plan] = None
    _log: list = field(default_factory=list)
    _first_traffic_s: Optional[float] = None

    def __post_init__(self):
        for iid in self.fleet.instances:
            self.roles.setdefault(iid, "M")
            self.freqs.setdefault(iid, None)
        self.router.listeners.append(self.forecaster)
        self.max_freq = max(self.planner.cfg.freqs)

    # ---- logging -----------------------------------------------------------------------------
    def log(self, kind: str, **data) -> None:
        row = dict(t=time.time(), kind=kind, **data)
        self._log.append(row)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

    # ---- actions -----------------------------------------------------------------------------
    def _set_clock(self, iid: str, mhz: Optional[int]) -> None:
        if self.freqs.get(iid) == mhz:
            return
        for g in self.fleet[iid].spec.gpus:
            if mhz is None:
                self.gpus.reset_clock(g)
            else:
                self.gpus.set_clock(g, mhz)
        self.freqs[iid] = mhz

    async def _drain(self, iid: str) -> bool:
        deadline = time.time() + self.drain_timeout_s
        while self.router.loads[iid].inflight_seqs > 0 and time.time() < deadline:
            await asyncio.sleep(0.2)
        return self.router.loads[iid].inflight_seqs == 0

    async def _park(self, iid: str, level: str) -> None:
        started = time.time()
        self.router.set_accepting(iid, False)
        self.router.set_roles({iid: "parked"})
        drained = await self._drain(iid)
        inst = self.fleet[iid]
        if level == "off":
            await asyncio.to_thread(inst.stop)
        self._set_clock(iid, None)
        if level == "L1":
            for g in inst.spec.gpus:
                self.gpus.park(g)
        self.roles[iid] = level
        self.log("park", instance=iid, level=level, drained=drained, seconds=time.time() - started)

    async def _wake(self, iid: str, role: str, mhz: int) -> None:
        started = time.time()
        inst = self.fleet[iid]
        prev = self.roles[iid]
        if prev == "off":
            await asyncio.to_thread(inst.start)
            await asyncio.to_thread(inst.wait_ready)
        elif prev == "L1":
            for g in inst.spec.gpus:
                self.gpus.unpark(g)
        self._set_clock(iid, mhz)
        self.roles[iid] = role
        self.router.set_roles({iid: ROUTER_ROLE[role]})
        self.router.set_accepting(iid, True)
        self.log("wake", instance=iid, from_state=prev, role=role, mhz=mhz, seconds=time.time() - started)

    def _role_freq(self, plan: Plan, role: str) -> int:
        return {"P": plan.f_P, "D": plan.f_D, "M": plan.f_M}[role]

    async def execute(self, plan: Plan) -> None:
        target = assign_roles(self.roles, plan.counts, self.router.inflight())
        tasks = []
        for iid, role in target.items():
            prev = self.roles[iid]
            if role in ACTIVE:
                mhz = self._role_freq(plan, role)
                if prev in PARKED:
                    tasks.append(self._wake(iid, role, mhz))
                else:
                    if prev != role:
                        self.log("reroute", instance=iid, from_role=prev, to_role=role)
                    self.roles[iid] = role
                    self.router.set_roles({iid: role})
                    self.router.set_accepting(iid, True)
                    self._set_clock(iid, mhz)
            elif prev in ACTIVE:
                tasks.append(self._park(iid, role))
            elif prev != role:
                # Deeper or shallower parking without an intermediate active phase.
                tasks.append(self._repark(iid, prev, role))
        self.router.set_roles({}, plan.tau)
        if tasks:
            await asyncio.gather(*tasks)
        self.plan_now = plan
        self.log("plan", counts=plan.counts, f_P=plan.f_P, f_D=plan.f_D, f_M=plan.f_M, tau=plan.tau,
                 power_w=plan.power_w, ttft_s=plan.ttft_s, tpot_s=plan.tpot_s,
                 shield_level=plan.detail.get("shield_level", 0), fallback=plan.detail.get("fallback", False),
                 cold_start=plan.detail.get("cold_start", False), roles=dict(self.roles))

    async def _repark(self, iid: str, prev: str, level: str) -> None:
        inst = self.fleet[iid]
        if prev == "L1":
            for g in inst.spec.gpus:
                self.gpus.unpark(g)
        if prev == "off":
            await asyncio.to_thread(inst.start)
            await asyncio.to_thread(inst.wait_ready)
        if level == "off":
            await asyncio.to_thread(inst.stop)
        elif level == "L1":
            for g in inst.spec.gpus:
                self.gpus.park(g)
        self.roles[iid] = level
        self.log("repark", instance=iid, from_state=prev, level=level)

    def _fail_open_plan(self) -> Plan:
        """Zero requests seen means zero information: keep everything active at max clock.
        Parking down is decided from the first informed forecast on."""
        n = len(self.fleet.instances)
        return Plan({"M": n}, self.max_freq, self.max_freq, self.max_freq, 0, 0.0, 0.0, 0.0,
                    dict(cold_start=True))

    def _informed(self, fc, now: float) -> bool:
        """The EWMA rate needs a window of traffic before it is anywhere near the true rate;
        planning on the early underestimate slashes capacity exactly when load is ramping."""
        if not fc.samples:
            return False
        if self._first_traffic_s is None:
            self._first_traffic_s = now
        return fc.samples >= self.min_warm_samples and now - self._first_traffic_s >= self.min_warm_s

    # ---- loop --------------------------------------------------------------------------------
    async def run(self, stop: asyncio.Event) -> None:
        now = time.time()
        fc = self.forecaster.forecast(now)
        plan = self.initial_plan or (self.planner.plan(fc) if self._informed(fc, now) else self._fail_open_plan())
        await self.execute(plan)
        next_plan_at = now + self.period_s
        last_level = 0
        while not stop.is_set():
            await asyncio.sleep(self.tick_s)
            now = time.time()
            level = 0
            pressure = None
            if self.shield is not None:
                pressure = self.shield.observe(self.router.recent(60.0, now), now)
                level = self.shield.update(pressure, now)
            replan = (not self.freeze) and (now >= next_plan_at or level != last_level)
            if replan:
                fc = self.forecaster.forecast(now)
                if self._informed(fc, now):
                    plan = await asyncio.to_thread(self.planner.plan, fc, self.plan_now)
                else:
                    plan = self._fail_open_plan()
                if self.shield is not None and (level or self.shield.floor_active):
                    plan = self.shield.apply(plan, pressure, self.max_freq)
                self.log("forecast", rate_rps=fc.rate_rps, trend_rps=fc.trend_rps, input_mean=fc.input_mean,
                         input_p95=fc.input_p95, output_mean=fc.output_mean, inflight=fc.inflight,
                         shield_level=level, pressure=None if pressure is None else vars(pressure))
                if self.plan_now is None or plan.key() != self.plan_now.key():
                    await self.execute(plan)
                next_plan_at = time.time() + self.period_s
            last_level = level
        self.log("stop", roles=dict(self.roles))

    def summary(self) -> dict:
        kinds = {}
        for row in self._log:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        return dict(events=kinds, shield_events=list(self.shield.events) if self.shield else [], final_roles=dict(self.roles))
