"""Control loop: observe -> forecast -> plan (every period) -> shield (every second) -> act.

Actions are role-table rewrites (free), clock changes (~100 ms), mem/SM clock parking (L1) and process
stop/start (off). Everything is logged as JSONL so a run can be audited afterwards.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from pdblend.bench.metering import Gpus
from pdblend.engine.launcher import Fleet
from pdblend.online.router import Router
from pdblend.planner.forecast import Forecaster
from pdblend.planner.pool import ACTIVE, PARKED, Plan, PoolPlanner, assign_roles
from pdblend.online.shield import Shield
from pdblend.online.gpu_actions import gpu_action
from pdblend.online.observations import backlog_snapshot

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
    hold_initial: bool = False                 # hold initial_plan until informed instead of fail-open
    freeze: bool = False                     # static policies: execute initial_plan once, never replan
    min_warm_s: float = 20.0                 # fail-open until this long after first traffic ...
    min_warm_samples: int = 30               # ... and at least this many arrivals were seen
    roles: dict = field(default_factory=dict)
    freqs: dict = field(default_factory=dict)
    plan_now: Optional[Plan] = None
    _log: list = field(default_factory=list)
    _first_traffic_s: Optional[float] = None
    min_plan_hold_s: float = 0.0
    down_plan_votes: int = 1
    home_margin: float = 0.0         # >0: pull back to initial_plan when feasible and this fraction cheaper
    _last_plan_change_s: Optional[float] = None
    _down_candidate_key: Optional[tuple] = None
    _down_votes: int = 0
    # PDblend-only controls; zero/false preserve all baseline policies.
    dynamic_m_floor: bool = False
    base_m_floor: int = 0
    low_load_m_floor: int = 2
    m_floor_pressure_enter: float = 0.75
    m_floor_pressure_exit: float = 0.55
    m_floor_stable_windows: int = 2
    m_floor_hold_s: float = 30.0
    pd_pressure_enter: float = 0.75
    pd_pressure_exit: float = 0.55
    pd_route_hold_s: float = 30.0
    pd_route_stable_windows: int = 2
    shield_protect_s: float = 0.0
    transition_cooldown_s: float = 0.0
    _m_floor: int = 0
    _m_floor_votes: int = 0
    _m_floor_last_change_s: Optional[float] = None
    _m_floor_last_window: Optional[int] = None
    _risk_windows: int = 0
    _quiet_windows: int = 0
    _last_pressure: dict = field(default_factory=dict)
    native_control: object | None = None
    transition_events: list = field(default_factory=list)

    def __post_init__(self):
        for iid in self.fleet.instances:
            self.roles.setdefault(iid, "M")
            self.freqs.setdefault(iid, None)
        self.router.listeners.append(self.forecaster)
        self.max_freq = max(self.planner.cfg.freqs)
        self._m_floor = max(self.base_m_floor, int(self.planner.cfg.min_m_instances))
        if self.dynamic_m_floor:
            self.base_m_floor = min(self.planner.cfg.slots, max(self._m_floor, 4))
            self.planner.cfg.pressure_controls = True
            self._m_floor = max(self._m_floor, int(self.low_load_m_floor))
            if hasattr(self.router, "configure_pressure_gate"):
                self.router.configure_pressure_gate(
                    enter=self.pd_pressure_enter, exit=self.pd_pressure_exit,
                    hold_s=self.pd_route_hold_s, stable_windows=self.pd_route_stable_windows,
                    min_input_tokens=1024)
                if self.planner.cfg.pd_pressure_active:
                    self.router.set_pressure_state(m_pressure=self.pd_pressure_enter,
                                                   reason='warm_start_pressure')

    # ---- logging -----------------------------------------------------------------------------
    def log(self, kind: str, **data) -> None:
        row = dict(t=time.time(), kind=kind, **data)
        self._log.append(row)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

    # ---- actions -----------------------------------------------------------------------------
    async def _phase(self, iid, operation, action):
        started, tick = time.time(), time.monotonic()
        row = dict(instance=iid, operation=operation, gpus=list(self.fleet[iid].spec.gpus),
                   started_s=started, generation=getattr(self.fleet[iid].spec, 'generation', 0),
                   source_role=self.roles.get(iid), source_frequency_mhz=self.freqs.get(iid),
                   transition_id=getattr(self, '_active_transition_id', None))
        try:
            result = await action()
            row['status'] = 'passed'
            if operation.startswith('native_'):
                row['native_receipt'] = result
            return result
        except BaseException as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            row.update(finished_s=time.time(), duration_s=time.monotonic()-tick,
                       energy_status='awaiting_common_sampler_integration')
            self.transition_events.append(row)
            self.log('transition_phase', **row)

    async def _hardware(self, iid, operation, action):
        return await self._phase(iid, operation,
            lambda: gpu_action(self.fleet[iid].spec.gpus, action))

    async def _publish(self, iid, role, accepting):
        async def apply():
            self.router.set_roles({iid: ROUTER_ROLE[role]})
            self.router.set_accepting(iid, accepting)
        await self._phase(iid, 'route_publish', apply)

    async def _set_clock(self, iid: str, mhz: Optional[int]) -> None:
        if self.freqs.get(iid) == mhz:
            return
        def apply():
            for g in self.fleet[iid].spec.gpus:
                if mhz is None:
                    self.gpus.reset_clock(g)
                else:
                    self.gpus.set_clock(g, mhz)
        await self._hardware(iid, 'clock_reset' if mhz is None else 'clock_set', apply)
        self.freqs[iid] = mhz

    async def _drain(self, iid: str) -> bool:
        deadline = time.monotonic() + self.drain_timeout_s
        def idle():
            load = self.router.loads[iid]
            return load.inflight_seqs == 0 and load.inflight_prefill_tokens == 0
        while not idle() and time.monotonic() < deadline:
            await asyncio.sleep(min(0.2, max(deadline - time.monotonic(), 0.0)))
        return idle()

    async def _park(self, iid: str, level: str) -> None:
        started = time.time()
        await self._publish(iid, level, False)
        drained = await self._phase(iid, 'proxy_drain', lambda: self._drain(iid))
        if not drained:
            # A prefill source may have zero decode sequences while still
            # serving a prompt/KV handoff. Never stop or reset its clocks on
            # timeout, and keep new admission disabled until recovery.
            self.log("park_failed", instance=iid, level=level, drained=False,
                     drain_scope="proxy_requests_only", seconds=time.time() - started)
            raise TimeoutError(f"{iid}: prefill/decode requests did not drain before parking")
        if self.native_control is not None:
            await self._phase(iid, 'native_drain',
                              lambda: self.native_control.drain(iid, self.drain_timeout_s))
        inst = self.fleet[iid]
        if level == "off":
            await self._phase(iid, 'stop', lambda: asyncio.to_thread(inst.stop))
        await self._set_clock(iid, None)
        if level == "L1":
            await self._hardware(iid, 'park', lambda: [self.gpus.park(g) for g in inst.spec.gpus])
        self.roles[iid] = level
        self.log("park", instance=iid, level=level, drained=drained,
                 drain_scope="native_and_proxy" if self.native_control else "proxy_requests_only",
                 seconds=time.time() - started)

    async def _wake(self, iid: str, role: str, mhz: int) -> None:
        started = time.time()
        inst = self.fleet[iid]
        prev = self.roles[iid]
        if prev == "off":
            await self._phase(iid, 'start', lambda: asyncio.to_thread(inst.start))
            await self._phase(iid, 'ready', lambda: asyncio.to_thread(inst.wait_ready))
        elif prev == "L1":
            await self._hardware(iid, 'unpark', lambda: [self.gpus.unpark(g) for g in inst.spec.gpus])
        await self._set_clock(iid, mhz)
        if self.native_control is not None:
            await self._phase(iid, 'native_resume', lambda: self.native_control.resume(iid, role))
        self.roles[iid] = role
        await self._publish(iid, role, True)
        self.log("wake", instance=iid, from_state=prev, role=role, mhz=mhz, seconds=time.time() - started)

    def _role_freq(self, plan: Plan, role: str) -> int:
        return {"P": plan.f_P, "D": plan.f_D, "M": plan.f_M}[role]

    async def execute(self, plan: Plan) -> None:
        started = time.time()
        self._active_transition_id = f'{started:.9f}'
        target = assign_roles(self.roles, plan.counts, self.router.inflight())
        tasks = []
        affected = []
        for iid, role in target.items():
            prev = self.roles[iid]
            if role in ACTIVE:
                mhz = self._role_freq(plan, role)
                if prev in PARKED:
                    affected.append(iid)
                    tasks.append(asyncio.create_task(self._wake(iid, role, mhz)))
                else:
                    if prev != role or self.freqs.get(iid) != mhz:
                        affected.append(iid)
                        tasks.append(asyncio.create_task(self._activate(iid, role, mhz)))
            elif prev in ACTIVE:
                affected.append(iid)
                tasks.append(asyncio.create_task(self._park(iid, role)))
            elif prev != role:
                # Deeper or shallower parking without an intermediate active phase.
                affected.append(iid)
                tasks.append(asyncio.create_task(self._repark(iid, prev, role)))
        if tasks:
            # Wait for all actions before recovery; failed clock RPCs must not
            # leave another thread modifying the hardware behind quarantine.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [result for result in results if isinstance(result, BaseException)]
            if failures:
                for iid in affected:
                    self.router.set_accepting(iid, False)
                self.log('transition_failed', affected=affected, errors=[repr(e) for e in failures],
                         started_s=started, finished_s=time.time(), partial_state=dict(self.roles))
                raise failures[0]
        self.router.set_roles({}, plan.tau)
        self.plan_now = plan
        self._last_plan_change_s = time.time()
        self._down_candidate_key, self._down_votes = None, 0
        self.log("plan", counts=plan.counts, f_P=plan.f_P, f_D=plan.f_D, f_M=plan.f_M, tau=plan.tau,
                 power_w=plan.power_w, ttft_s=plan.ttft_s, tpot_s=plan.tpot_s,
                 shield_level=plan.detail.get("shield_level", 0), fallback=plan.detail.get("fallback", False),
                 cold_start=plan.detail.get("cold_start", False), roles=dict(self.roles),
                 profile_key=getattr(self.planner.model, 'profile_key', {}),
                 calibration_identity=getattr(self.planner.model, 'calibration_identity', {}),
                 calibration_coverage=getattr(self.planner.model, 'calibration_coverage', {}),
                 query_results=plan.detail)
        self.log('transition_complete', started_s=started, finished_s=time.time(),
                 critical_path_s=time.time()-started, affected=affected,
                 transition_id=self._active_transition_id,
                 validation_scope='native_drain_resume_and_proxy_publication',
                 formal_eligible=False)

    async def _activate(self, iid, role, mhz):
        previous = self.roles[iid]
        await self._set_clock(iid, mhz)
        self.roles[iid] = role
        await self._publish(iid, role, True)
        if previous != role:
            self.log('reroute', instance=iid, from_role=previous, to_role=role,
                     existing_requests_pinned=True)

    async def _repark(self, iid: str, prev: str, level: str) -> None:
        inst = self.fleet[iid]
        if prev == "L1":
            await self._hardware(iid, 'unpark', lambda: [self.gpus.unpark(g) for g in inst.spec.gpus])
        if prev == "off":
            await self._phase(iid, 'start', lambda: asyncio.to_thread(inst.start))
            await self._phase(iid, 'ready', lambda: asyncio.to_thread(inst.wait_ready))
            if self.native_control is not None:
                await self._phase(iid, 'native_drain', lambda: self.native_control.drain(iid, self.drain_timeout_s))
        if level == "off":
            await self._phase(iid, 'stop', lambda: asyncio.to_thread(inst.stop))
        elif level == "L1":
            await self._hardware(iid, 'park', lambda: [self.gpus.park(g) for g in inst.spec.gpus])
        else:
            await self._set_clock(iid, None)
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

    @staticmethod
    def _is_downshift(current: Plan, candidate: Plan) -> bool:
        if candidate.active() != current.active():
            return candidate.active() < current.active()
        # Ignore placeholder clocks for roles with zero instances.
        return any(current.counts.get(role, 0) and candidate.counts.get(role, 0)
                   and getattr(candidate, f"f_{role}") < getattr(current, f"f_{role}")
                   for role in ACTIVE)

    def _anchor_candidate(self, fc, candidate: Plan) -> Optional[Plan]:
        """Warm-start home anchor: a noise-driven upshift sticks because the hysteresis margin
        blocks the small saving of returning. When the offline home plan is feasible again and
        cheaper than the candidate by more than home_margin, offer it instead (the result still
        goes through dwell/vote gating, and the shield can override it under real pressure)."""
        home = self.initial_plan
        if self.dynamic_m_floor and self.planner.cfg.pd_pressure_active:
            return None
        if self.home_margin <= 0.0 or home is None or self.plan_now is None:
            return None
        if home.key() == candidate.key() or home.key() == self.plan_now.key():
            return None
        home_ev = self.planner.evaluate(home.counts, home.f_P, home.f_D, home.f_M, home.tau, fc)
        if home_ev is None:
            return None
        if home_ev.power_w < candidate.power_w * (1.0 - self.home_margin):
            return home_ev
        return None

    def _m_pressure(self, plan: Optional[Plan]) -> float:
        """Normalize the current M-pool pressure for routing and floor control."""
        if plan is None or plan.counts.get("M", 0) <= 0:
            return 0.0
        return float(self._last_pressure.get('pressure', 2.0))

    def _update_m_floor(self, fc, pressure, level: int, now: float, stable_window: bool) -> bool:
        """Move the PDblend M floor down one instance at a time only after quiet windows."""
        if not self.dynamic_m_floor:
            return False
        top = self.base_m_floor
        low = max(1, min(int(self.low_load_m_floor), top))
        m_pressure = self._m_pressure(self.plan_now)
        emergency = (level > 0 or (self.shield is not None and (self.shield.floor_active > 0
                        or self.shield.protection_active(now)))
                     or (pressure is not None and (pressure.prefill or pressure.decode)))
        safe = (not emergency and m_pressure <= self.m_floor_pressure_exit
                and fc.samples >= self.min_warm_samples and not self.router._pd_pressure_active)
        changed = False
        if emergency or m_pressure >= self.m_floor_pressure_enter:
            self._m_floor_votes = 0
            self._m_floor_last_window = None
            if self._m_floor != top:
                self._m_floor = top
                self._m_floor_last_change_s = now
                changed = True
        elif stable_window and safe:
            window = int(now // max(self.period_s, 1.0))
            if window != self._m_floor_last_window:
                self._m_floor_last_window = window
                self._m_floor_votes += 1
            if (self._m_floor_votes >= max(1, self.m_floor_stable_windows)
                    and self._m_floor > low
                    and (self._m_floor_last_change_s is None
                         or now - self._m_floor_last_change_s >= self.m_floor_hold_s)):
                # Validate the *smaller* pool at a 25% load reserve before
                # releasing the floor. A quiet current M4 does not prove M2.
                reserve = replace(fc, rate_rps=fc.rate_rps * 1.25)
                target = self._m_floor - 1
                if any(self.planner.mixed_pressure(reserve, target, f)['pressure'] <= self.m_floor_pressure_exit
                       for f in self.planner.cfg.freqs):
                    self._m_floor -= 1
                    self._m_floor_votes = 0
                    self._m_floor_last_change_s = now
                    changed = True
        elif stable_window:
            self._m_floor_votes = 0
        self.planner.cfg.min_m_instances = self._m_floor
        return changed

    def _update_strategy(self, fc, pressure, level: int, now: float, scheduled: bool) -> tuple[bool, bool]:
        if not self.dynamic_m_floor or self.plan_now is None:
            return False, False
        plan = self.plan_now
        self._last_pressure = self.planner.mixed_pressure(fc, plan.counts.get('M', 0), plan.f_M)
        observed = bool(pressure and (pressure.prefill or pressure.decode))
        if scheduled:
            self._risk_windows = self._risk_windows + 1 if observed else 0
            self._quiet_windows = 0 if observed or level else self._quiet_windows + 1
        has_long = fc.input_p95 >= self.planner.cfg.pd_min_input_tokens
        mode_changed = self.router.set_pressure_state(
            m_pressure=self._last_pressure['pressure'] if has_long else 0.0,
            decode_risk=has_long and self._risk_windows >= 2,
            shield_active=bool(level), now=now, stable_window=scheduled,
            reason='sustained_slo_pressure' if self._risk_windows >= 2 else 'predicted_m_pressure')
        self.planner.cfg.pd_pressure_active = self.router._pd_pressure_active
        previous = self._m_floor
        floor_changed = self._update_m_floor(fc, pressure, level, now, scheduled)
        urgent = self._m_floor > previous or (mode_changed and self.router._pd_pressure_active)
        return floor_changed or mode_changed, urgent

    def _gate_plan_change(self, candidate: Plan, now: float, *, scheduled: bool = True,
                          shield_protected: bool = False) -> tuple[Plan, str]:
        """PDblend-only dwell and shrink confirmation; defaults preserve baseline decisions."""
        current = self.plan_now
        if current is None or candidate.key() == current.key():
            self._down_candidate_key, self._down_votes = None, 0
            return candidate, "initial" if current is None else "unchanged"
        if self._last_plan_change_s is not None and now - self._last_plan_change_s < self.min_plan_hold_s:
            self._down_candidate_key, self._down_votes = None, 0
            return current, "minimum_hold"
        if self._is_downshift(current, candidate):
            if shield_protected:
                self._down_candidate_key, self._down_votes = None, 0
                return current, "shield_protection"
            if self.dynamic_m_floor and self._quiet_windows < 2:
                self._down_candidate_key, self._down_votes = None, 0
                return current, "slo_quiet_confirmation"
            if (self.transition_cooldown_s > 0.0 and self._last_plan_change_s is not None
                    and now - self._last_plan_change_s < self.transition_cooldown_s):
                self._down_candidate_key, self._down_votes = None, 0
                return current, "transition_cooldown"
        if self.down_plan_votes > 1 and self._is_downshift(current, candidate):
            key = candidate.key()
            if key != self._down_candidate_key:
                self._down_candidate_key, self._down_votes = key, 0
            # Shield ticks are not additional independent forecast windows.
            if scheduled:
                self._down_votes += 1
            if self._down_votes < self.down_plan_votes:
                return current, "downshift_confirmation"
            reason = "confirmed_downshift"
        else:
            reason = "planner_change"
        self._down_candidate_key, self._down_votes = None, 0
        return candidate, reason

    # ---- loop --------------------------------------------------------------------------------
    async def run(self, stop: asyncio.Event) -> None:
        now = time.time()
        self.forecaster.set_backlog(backlog_snapshot(self.router))
        fc = self.forecaster.forecast(now)
        plan = self.initial_plan or (self.planner.plan(fc) if self._informed(fc, now) else self._fail_open_plan())
        await self.execute(plan)
        self._m_floor_last_change_s = time.time()
        next_plan_at = now + self.period_s
        last_level = 0
        while not stop.is_set():
            await asyncio.sleep(self.tick_s)
            now = time.time()
            self.forecaster.set_backlog(backlog_snapshot(self.router))
            level = 0
            pressure = None
            if self.shield is not None:
                observations = (self.router.observation_records(60.0, now)
                                if hasattr(self.router, 'observation_records') else self.router.recent(60.0, now))
                pressure = self.shield.observe(observations, now)
                level = self.shield.update(pressure, now)
            scheduled = now >= next_plan_at
            fc_now = self.forecaster.forecast(now) if self.dynamic_m_floor else None
            strategy_changed, urgent = self._update_strategy(fc_now, pressure, level, now, scheduled)
            m_pressure = self._m_pressure(self.plan_now) if self.dynamic_m_floor else 0.0
            replan = (not self.freeze) and (scheduled or level != last_level or strategy_changed)
            if replan:
                fc = fc_now if fc_now is not None else self.forecaster.forecast(now)
                if self._informed(fc, now) or (urgent and self.dynamic_m_floor):
                    plan = await asyncio.to_thread(self.planner.plan, fc, self.plan_now)
                elif self.hold_initial and self.plan_now is not None:
                    plan = self.plan_now
                else:
                    plan = self._fail_open_plan()
                candidate = plan
                anchor = self._anchor_candidate(fc, candidate)
                if anchor is not None:
                    candidate = plan = anchor
                protected = bool(self.shield and self.shield.protection_active(now))
                if urgent:
                    reason = 'pressure_safety_override'
                else:
                    plan, reason = self._gate_plan_change(plan, now, scheduled=scheduled,
                                                           shield_protected=protected)
                if self.shield is not None and (level or self.shield.floor_active):
                    # Apply safety *after* ordinary-change gating so escalation
                    # can immediately raise clocks/wake instances during a hold.
                    if not urgent and level > last_level and (self.min_plan_hold_s or self.down_plan_votes > 1):
                        plan = self.plan_now or plan
                    plan = self.shield.apply(plan, pressure, self.max_freq)
                    if self.plan_now is None or plan.key() != self.plan_now.key():
                        reason = "shield_override"
                    if level:
                        self._down_candidate_key, self._down_votes = None, 0
                self.log("forecast", rate_rps=fc.rate_rps, trend_rps=fc.trend_rps, input_mean=fc.input_mean,
                         input_p95=fc.input_p95, output_mean=fc.output_mean, inflight=fc.inflight,
                         pending_prefill_tokens=fc.pending_prefill_tokens,
                         remaining_decode_tokens=fc.remaining_decode_tokens,
                         occupied_kv_tokens=fc.occupied_kv_tokens,
                         paired_length_samples=len(fc.length_pairs),
                         recent_rate_rps=fc.recent_rate_rps, decision_reason=reason,
                         down_votes=self._down_votes, candidate_counts=candidate.counts,
                         candidate_clocks=dict(P=candidate.f_P, D=candidate.f_D, M=candidate.f_M),
                         shield_level=level, pressure=None if pressure is None else vars(pressure),
                         m_floor=self._m_floor, m_pressure=m_pressure,
                         model_pressure=self._last_pressure, risk_windows=self._risk_windows,
                         route_pressure=self.router.pressure_state() if hasattr(self.router, "pressure_state") else None)
                if self.plan_now is None or plan.key() != self.plan_now.key():
                    await self.execute(plan)
                if scheduled or not self.dynamic_m_floor:
                    next_plan_at = time.time() + self.period_s
            last_level = level
        self.log("stop", roles=dict(self.roles))

    def summary(self) -> dict:
        kinds = {}
        for row in self._log:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        return dict(events=kinds, shield_events=list(self.shield.events) if self.shield else [],
                    final_roles=dict(self.roles), transition_phases=self.transition_events)
