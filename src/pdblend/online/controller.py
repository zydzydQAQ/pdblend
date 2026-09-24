"""Control loop: observe -> forecast -> plan (every period) -> shield (every second) -> act.

Actions are role-table rewrites (free), clock changes (~100 ms), mem/SM clock parking (L1) and process
stop/start (off). Everything is logged as JSONL so a run can be audited afterwards.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
import inspect
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
    _clock_known: set = field(default_factory=set, init=False)
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
    safety_recovery: bool = False          # opt in for PDblend; baseline control remains unchanged
    startup_safety: bool = False
    deadline_safety: bool = False
    safety_max_freq: Optional[int] = None
    frequency_snapshot: object | None = None
    _clock_locks: dict = field(default_factory=dict, init=False)
    _execute_lock: object = field(default_factory=asyncio.Lock, init=False)
    _executing: bool = field(default=False, init=False)
    _transition_open: bool = field(default=False, init=False)
    _transition_target_roles: dict = field(default_factory=dict, init=False)
    _urgent_clock_task: object | None = field(default=None, init=False)
    _deadline_floor_mhz: int = field(default=0, init=False)
    _deadline_clear_votes: int = field(default=0, init=False)
    _startup_validated: bool = field(default=False, init=False)
    _frequency_tasks: set = field(default_factory=set, init=False)
    _frequency_snapshot_error: object | None = field(default=None, init=False)
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
        self._clock_known.update(iid for iid, mhz in self.freqs.items() if mhz is not None)
        for iid in self.fleet.instances:
            self.roles.setdefault(iid, "M")
            self.freqs.setdefault(iid, None)
        self.router.listeners.append(self.forecaster)
        self.max_freq = max(self.planner.cfg.freqs)
        if self.safety_max_freq is not None:
            allowed = [f for f in self.planner.cfg.freqs if f <= self.safety_max_freq]
            if not allowed:
                raise ValueError('safety frequency ceiling excludes every planner frequency')
            self.max_freq = max(allowed)
        if self.startup_safety:
            self.min_warm_s = max(20., self.min_warm_s)
            self.min_warm_samples = max(30, self.min_warm_samples)
            self.down_plan_votes = max(2, self.down_plan_votes)
        if self.deadline_safety:
            self.router.configure_deadline_safety(enabled=True,
                actuation_s=max(.05, float(getattr(self.planner.model, 'freq_switch_s', .1))))
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
        if operation in {"park", "unpark"}:
            self._clock_known.discard(iid)
        return await self._phase(iid, operation,
            lambda: gpu_action(self.fleet[iid].spec.gpus, action))

    async def _publish(self, iid, role, accepting):
        async def apply():
            self.router.set_roles({iid: ROUTER_ROLE[role]})
            self.router.set_accepting(iid, accepting)
        await self._phase(iid, 'route_publish', apply)

    async def _set_clock(self, iid: str, mhz: Optional[int], *, expected_role=None,
                         expected_generation=None, expected_transition_id=None) -> None:
        # Only clock RPCs own this lock: loading a stopped engine must never
        # prevent an urgent clock change on already serving instances.
        lock = self._clock_locks.setdefault(iid, asyncio.Lock())
        async with lock:
            if expected_role is not None:
                # A deadline decision can wait behind a normal clock RPC.
                # Recheck ownership after acquiring the lock; the instance
                # may now be draining or changing from M to P/D.
                if (self.roles.get(iid) != expected_role
                        or self.router.loads[iid].role != expected_role
                        or not self.router.loads[iid].accepting
                        or getattr(self.fleet[iid].spec, 'generation', 0) != expected_generation
                        or not self._transition_open
                        or self._active_transition_id != expected_transition_id
                        or self._transition_target_roles.get(iid) != expected_role):
                    return
            if mhz is not None:
                mhz = min(mhz, self.max_freq)
                intended_role = (self._transition_target_roles.get(iid, self.roles.get(iid))
                                 if self._transition_open else self.roles.get(iid))
                if intended_role == 'M' and self._deadline_floor_mhz:
                    mhz = max(mhz, self._deadline_floor_mhz)
            if iid in self._clock_known and self.freqs.get(iid) == mhz:
                return
            self._clock_known.discard(iid)
            def apply():
                for g in self.fleet[iid].spec.gpus:
                    if mhz is None:
                        self.gpus.reset_clock(g)
                    else:
                        self.gpus.set_clock(g, mhz)
            await self._hardware(iid, 'clock_reset' if mhz is None else 'clock_set', apply)
            self.freqs[iid] = mhz
            self._clock_known.add(iid)

    def requested_frequencies(self):
        return {gpu: (210 if self.roles[iid] == 'L1' else self.freqs.get(iid)
                      if self.roles[iid] in ACTIVE else None)
                for iid, inst in self.fleet.instances.items() for gpu in inst.spec.gpus}

    async def _capture_frequency(self, reason):
        if self.frequency_snapshot is None:
            return
        task = asyncio.create_task(asyncio.to_thread(self.frequency_snapshot,
            requested=self.requested_frequencies(), reason=reason))
        try:
            rows = await asyncio.shield(task)
        except asyncio.CancelledError:
            # A cancelled settled-snapshot task must not leave a thread
            # appending observations after the sampler has been finalized.
            await task
            raise
        if inspect.isawaitable(rows):
            rows = await rows
        self.log('physical_frequency_snapshot', reason=reason, readings=rows)
        return rows

    async def _collective_probe_available(self, pressure):
        """Unknown clock evidence preserves the ordinary Shield probe.

        A requested/set clock alone is insufficient. Read every GPU in each
        affected, accepting instance while its ownership stays unchanged.
        This is used only for short-gap edges, never sustained/deadline risk.
        """
        if self.frequency_snapshot is None or self._transition_open:
            return None
        roles = self.shield._pressure_roles(pressure)
        instances = [iid for iid, role in self.roles.items() if role in roles]
        if not instances or not roles <= {self.roles[iid] for iid in instances}:
            return None

        def ownership():
            return tuple((iid, self.roles.get(iid), self.freqs.get(iid), iid in self._clock_known,
                          self.router.loads[iid].role, self.router.loads[iid].accepting,
                          getattr(self.fleet[iid].spec, 'generation', 0)) for iid in instances)

        before = ownership()
        if any(freq != self.max_freq or not known or role != routed or not accepting
               for _, role, freq, known, routed, accepting, _ in before):
            return None
        started = time.time()
        rows = await self._capture_frequency('collective_probe_headroom')
        finished = time.time()
        if self._transition_open or before != ownership() or not isinstance(rows, list):
            return None
        expected = {gpu for iid in instances for gpu in self.fleet[iid].spec.gpus}
        observed = {}
        for row in rows:
            if not isinstance(row, dict) or row.get('gpu') not in expected:
                continue
            gpu = row['gpu']
            values = [row.get(key) for key in ('read_started_s', 'read_finished_s', 'observed_mhz')]
            if (gpu in observed or row.get('error') is not None
                    or row.get('requested_mhz') != self.max_freq
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in values)):
                return None
            read_start, read_end, mhz = values
            if not (started <= read_start <= read_end <= finished
                    and finished - read_start <= 1.0 and abs(mhz - self.max_freq) <= 30):
                return None
            observed[gpu] = mhz
        return False if observed.keys() == expected else None

    async def _settled_frequency(self, transition_id):
        await asyncio.sleep(1.)
        await self._capture_frequency('settled:' + transition_id)

    def _frequency_snapshot_done(self, task):
        self._frequency_tasks.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                self._frequency_snapshot_error = error

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
        await self._set_clock(iid, max(mhz, self._deadline_floor_mhz) if role == 'M' else mhz)
        if self.native_control is not None:
            await self._phase(iid, 'native_resume', lambda: self.native_control.resume(iid, role))
        self.roles[iid] = role
        await self._publish(iid, role, True)
        self.log("wake", instance=iid, from_state=prev, role=role, mhz=mhz, seconds=time.time() - started)

    def _role_freq(self, plan: Plan, role: str) -> int:
        return {"P": plan.f_P, "D": plan.f_D, "M": plan.f_M}[role]

    async def execute(self, plan: Plan) -> None:
        async with self._execute_lock:
            self._executing = True
            try:
                await self._execute(plan)
            finally:
                self._executing = False
                self._transition_open = False
                self._transition_target_roles = {}

    async def _execute(self, plan: Plan) -> None:
        await self._capture_frequency('before_transition')
        # Normalize clocks before deciding which instances require actions.
        # Updating only the final plan would claim an unapplied clock floor.
        plan = replace(plan, f_P=min(plan.f_P, self.max_freq),
                       f_D=min(plan.f_D, self.max_freq),
                       f_M=min(self.max_freq, max(plan.f_M, self._deadline_floor_mhz)
                               if plan.counts.get('M', 0) else plan.f_M))
        started = time.time()
        self._active_transition_id = f'{started:.9f}'
        self._transition_open = True
        target = assign_roles(self.roles, plan.counts, self.router.inflight())
        self._transition_target_roles = dict(target)
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
        if self._urgent_clock_task is not None:
            await self._urgent_clock_task
        if self._deadline_floor_mhz and plan.counts.get('M', 0):
            plan = replace(plan, f_M=max(plan.f_M, self._deadline_floor_mhz))
        plan = replace(plan, f_P=min(plan.f_P, self.max_freq),
                       f_D=min(plan.f_D, self.max_freq), f_M=min(plan.f_M, self.max_freq))
        if self.startup_safety or self.deadline_safety:
            self.forecaster.set_backlog(backlog_snapshot(self.router))
            plan = self.planner.refresh_estimate(plan, self.forecaster.forecast(time.time()))
        # A deadline action may have raised active clocks while another
        # instance was loading inside this same physical transition.
        affected = sorted(set(affected) | {r['instance'] for r in self.transition_events
            if r.get('transition_id') == self._active_transition_id})
        self.router.set_roles({}, plan.tau)
        self.plan_now = plan
        self._last_plan_change_s = time.time()
        self._down_candidate_key, self._down_votes = None, 0
        self.log("plan", counts=plan.counts, f_P=plan.f_P, f_D=plan.f_D, f_M=plan.f_M, tau=plan.tau,
                 plan_identity=dict(tp=plan.tp, pp=plan.pp, pool_id=plan.pool_id,
                                    generation=plan.generation, profile_key=plan.profile_key),
                 power_w=plan.power_w, ttft_s=plan.ttft_s, tpot_s=plan.tpot_s,
                 shield_level=plan.detail.get("shield_level", 0), fallback=plan.detail.get("fallback", False),
                 cold_start=plan.detail.get("cold_start", False), roles=dict(self.roles),
                 profile_key=getattr(self.planner.model, 'profile_key', {}),
                 calibration_identity=getattr(self.planner.model, 'calibration_identity', {}),
                 calibration_coverage=getattr(self.planner.model, 'calibration_coverage', {}),
                 query_results=plan.detail)
        self._transition_open = False
        self.log('transition_complete', started_s=started, finished_s=time.time(),
                 critical_path_s=time.time()-started, affected=affected,
                 transition_id=self._active_transition_id,
                 validation_scope='native_drain_resume_and_proxy_publication',
                 formal_eligible=False)
        await self._capture_frequency('after_transition')
        if self.frequency_snapshot is not None:
            task = asyncio.create_task(self._settled_frequency(self._active_transition_id))
            self._frequency_tasks.add(task)
            task.add_done_callback(self._frequency_snapshot_done)

    async def _activate(self, iid, role, mhz):
        previous = self.roles[iid]
        await self._set_clock(iid, max(mhz, self._deadline_floor_mhz) if role == 'M' else mhz)
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
        if self.initial_plan is not None:
            return replace(self.initial_plan, counts={"M": n}, f_P=self.max_freq,
                           f_D=self.max_freq, f_M=self.max_freq, tau=0,
                           power_w=0.0, ttft_s=0.0, tpot_s=0.0, detail=dict(cold_start=True))
        model = self.planner.model
        spec = next(iter(self.fleet.instances.values())).spec
        return Plan({"M": n}, self.max_freq, self.max_freq, self.max_freq, 0, 0.0, 0.0, 0.0,
                    dict(cold_start=True), tp=model.tp, pp=model.pp,
                    pool_id=getattr(spec, 'pool_id', ''), generation=getattr(spec, 'generation', 0),
                    profile_key=json.dumps(model.profile_key, sort_keys=True, separators=(',', ':'))
                    if model.profile_key else '')

    def _informed(self, fc, now: float) -> bool:
        """The EWMA rate needs a window of traffic before it is anywhere near the true rate;
        planning on the early underestimate slashes capacity exactly when load is ramping."""
        if self.startup_safety:
            first = self.router.first_arrival_s
            return (first is not None and self.router.admitted_requests >= self.min_warm_samples
                    and now-first >= self.min_warm_s)
        if not fc.samples:
            return False
        if self._first_traffic_s is None:
            self._first_traffic_s = now
        return fc.samples >= self.min_warm_samples and now - self._first_traffic_s >= self.min_warm_s

    def _startup_plan(self, plan):
        if not self.startup_safety or self.freeze:
            return plan
        return replace(plan, f_P=self.max_freq, f_D=self.max_freq, f_M=self.max_freq,
                       detail=dict(plan.detail, startup_safety=True,
                                   startup_evidence='awaiting_real_requests'))

    def _startup_observed_safe(self, now):
        if not self.startup_safety:
            return True
        records = self.router.observation_records(5., now)
        measured = [r for r in records if r.first_token_s is not None]
        if not measured:
            return False
        slo = self.planner.cfg.slo
        for r in records:
            if r.error:
                return False
            if r.first_token_s is None:
                if now-r.submitted_s >= .8*slo.ttft_s:
                    return False
            elif r.first_token_s-r.submitted_s > slo.ttft_s:
                return False
            if r.finished_s is not None and r.completion_tokens > 1:
                end = r.last_token_s if r.last_token_s is not None else r.finished_s
                if (end-r.first_token_s)/(r.completion_tokens-1) > slo.tpot_s:
                    return False
            elif r.finished_s is None and r.first_token_s is not None and r.max_tokens > 1:
                if r.tokens_so_far < 2 or r.last_token_s is None:
                    return False
                if (r.last_token_s-r.first_token_s)/(r.tokens_so_far-1) > slo.tpot_s:
                    return False
        return True

    async def _deadline_clock_action(self, risk):
        if self.plan_now is None or not self.plan_now.counts.get('M', 0):
            return
        current = max((self.freqs.get(iid) or 0 for iid, role in self.roles.items() if role == 'M'), default=0)
        target = self.max_freq
        for frequency in sorted(self.planner.cfg.freqs):
            if not current < frequency <= self.max_freq:
                continue
            candidate = self.router.deadline_risk_snapshot(frequency=frequency, include_saturation=False)
            if not candidate['risk'] and not candidate['unavailable']:
                target = frequency
                break
        old_floor = self._deadline_floor_mhz
        self._deadline_floor_mhz = max(self._deadline_floor_mhz, target)
        self._deadline_clear_votes = 0
        needs_change = any((self.freqs.get(iid) or 0) < target
                           for iid, role in self.roles.items() if role == 'M')
        if old_floor != self._deadline_floor_mhz or needs_change:
            self.log('deadline_risk', target_frequency_mhz=target, pressure=risk,
                     action='mixed_clock_only', counts=dict(self.plan_now.counts))
        if not needs_change:
            return
        # Snapshot hooks may yield before a transition is opened or just
        # after it closes. Wait only for that short boundary, never queue the
        # emergency behind the subsequent engine readiness wait.
        while self._executing and not self._transition_open:
            await asyncio.sleep(.001)
        if self._transition_open:
            # The topology action can be waiting for off->ready for seconds.
            # Raise only currently serving M clocks within that transition;
            # its final plan is refreshed from the resulting configuration.
            async def raise_active():
                transition_id = self._active_transition_id
                candidates = [(iid, getattr(self.fleet[iid].spec, 'generation', 0))
                    for iid, role in tuple(self.roles.items())
                    if role == self._transition_target_roles.get(iid) == 'M'
                    and self.router.loads[iid].role == 'M' and self.router.loads[iid].accepting
                    and (self.freqs.get(iid) or 0) < target]
                await asyncio.gather(*(self._set_clock(iid, target, expected_role='M',
                    expected_generation=generation, expected_transition_id=transition_id)
                    for iid, generation in candidates))
            self._urgent_clock_task = asyncio.create_task(raise_active())
            await self._urgent_clock_task
        else:
            plan = replace(self.plan_now, f_M=max(self.plan_now.f_M, target),
                           detail=dict(self.plan_now.detail, deadline_safety=risk))
            await self.execute(plan)

    async def _deadline_worker(self, stop):
        last_evaluation = None
        while not stop.is_set():
            await self.router.deadline_event.wait()
            if last_evaluation is not None:
                delay = .05 - (time.monotonic()-last_evaluation)
                if delay > 0:
                    await asyncio.sleep(delay)
            self.router.deadline_event.clear()
            if stop.is_set():
                break
            last_evaluation = time.monotonic()
            risk = self.router.deadline_risk_snapshot()
            if risk['risk']:
                await self._deadline_clock_action(risk)

    def _deadline_hold(self, plan, now, scheduled):
        if not self.deadline_safety or not self._deadline_floor_mhz:
            return plan
        risk = self.router.deadline_risk_snapshot(now=now)
        if risk['risk'] or risk['unavailable']:
            self._deadline_clear_votes = 0
        elif scheduled:
            self._deadline_clear_votes += 1
            if self._deadline_clear_votes >= max(2, self.down_plan_votes):
                self._deadline_floor_mhz = 0
                self.log('deadline_release', safe_windows=self._deadline_clear_votes)
        if self._deadline_floor_mhz:
            if self.plan_now is not None and plan.counts.get('M', 0) < self.plan_now.counts.get('M', 0):
                plan = self.plan_now
            return replace(plan, f_M=max(plan.f_M, self._deadline_floor_mhz),
                           detail=dict(plan.detail, deadline_frequency_floor=self._deadline_floor_mhz))
        return plan

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
        if self.startup_safety and not self._startup_validated:
            return None
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
        collective = self._collective_clock_only(pressure)
        emergency = not collective and (level > 0 or (self.shield is not None and (self.shield.floor_active > 0
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

    def _collective_clock_only(self, pressure):
        classify = getattr(self.shield, '_collective_only', None)
        return bool(pressure is not None and self.shield is not None
                    and getattr(self.shield, 'mode', None) == 'budget_aware'
                    and callable(classify) and classify(pressure))

    def _update_strategy(self, fc, pressure, level: int, now: float, scheduled: bool) -> tuple[bool, bool]:
        if not self.dynamic_m_floor or self.plan_now is None:
            return False, False
        plan = self.plan_now
        self._last_pressure = self.planner.mixed_pressure(fc, plan.counts.get('M', 0), plan.f_M)
        collective = self._collective_clock_only(pressure)
        observed = bool(pressure and (pressure.prefill or pressure.decode) and not collective)
        if scheduled:
            self._risk_windows = self._risk_windows + 1 if observed else 0
            self._quiet_windows = 0 if observed or level else self._quiet_windows + 1
        has_long = fc.input_p95 >= self.planner.cfg.pd_min_input_tokens
        mode_changed = self.router.set_pressure_state(
            m_pressure=self._last_pressure['pressure'] if has_long else 0.0,
            decode_risk=has_long and self._risk_windows >= 2,
            shield_active=bool(level) and not collective, now=now, stable_window=scheduled,
            reason='sustained_slo_pressure' if self._risk_windows >= 2 else 'predicted_m_pressure')
        self.planner.cfg.pd_pressure_active = self.router._pd_pressure_active
        previous = self._m_floor
        floor_changed = self._update_m_floor(fc, pressure, level, now, scheduled)
        urgent = self._m_floor > previous or (mode_changed and self.router._pd_pressure_active)
        return floor_changed or mode_changed, urgent

    def _is_safety_recovery(self, candidate: Plan, pressure=None, forecast=None) -> bool:
        """Recognize capacity increases and model-supported rescue of a pressured role.

        Energy-only changes still obey the ordinary hold/vote policy. A role
        redistribution additionally needs finite SLO predictions and cannot
        remove any branch that still owns requests.
        """
        current = self.plan_now
        if not self.safety_recovery or current is None or candidate.key() == current.key():
            return False
        if candidate.active() < current.active():
            return False
        owned = ({w.branch for w in forecast.backlog} if forecast is not None else
                 ({"PD"} if current.counts.get("P", 0) else set())
                 | ({"M"} if current.counts.get("M", 0) else set()))
        if ((owned & {"PD", "P_ONLY"} and not
                (candidate.counts.get("P", 0) and candidate.counts.get("D", 0)))
                or "M" in owned and not candidate.counts.get("M", 0)):
            return False
        if any(current.counts.get(role, 0) and candidate.counts.get(role, 0)
               and getattr(candidate, f"f_{role}") < getattr(current, f"f_{role}")
               for role in ACTIVE):
            return False
        monotone = all(candidate.counts.get(role, 0) >= current.counts.get(role, 0)
                       for role in ACTIVE)
        increased = any(candidate.counts.get(role, 0) > current.counts.get(role, 0)
                        or (current.counts.get(role, 0) and candidate.counts.get(role, 0)
                            and getattr(candidate, f"f_{role}") > getattr(current, f"f_{role}"))
                        for role in ACTIVE)
        if monotone and increased and candidate.tau == current.tau:
            return True
        if (pressure is None or not (pressure.prefill or pressure.decode)
                or candidate.detail.get("capacity_insufficient")
                or not all(math.isfinite(v) for v in
                           (candidate.power_w, candidate.ttft_s, candidate.tpot_s))):
            return False
        slo = self.planner.cfg.slo
        if candidate.ttft_s > slo.ttft_s * slo.safety or candidate.tpot_s > slo.tpot_s * slo.safety:
            return False
        pressured = set()
        for stage, dedicated in (("prefill", "P"), ("decode", "D")):
            if not getattr(pressure, stage):
                continue
            paths = getattr(pressure, stage + "_paths", ())
            if paths:
                pressured.update("M" if path == "M" else dedicated for path in paths)
            else:
                pressured.add(dedicated if current.counts.get("P", 0) else "M")
        return bool(pressured) and all(candidate.counts.get(role, 0) >= current.counts.get(role, 0)
                                      for role in pressured) and any(
            candidate.counts.get(role, 0) > current.counts.get(role, 0) for role in pressured)

    def _gate_plan_change(self, candidate: Plan, now: float, *, scheduled: bool = True,
                          shield_protected: bool = False, pressure=None, forecast=None) -> tuple[Plan, str]:
        """PDblend-only dwell and shrink confirmation; defaults preserve baseline decisions."""
        current = self.plan_now
        if current is None or candidate.key() == current.key():
            self._down_candidate_key, self._down_votes = None, 0
            return candidate, "initial" if current is None else "unchanged"
        if self._is_safety_recovery(candidate, pressure, forecast):
            self._down_candidate_key, self._down_votes = None, 0
            return candidate, "safety_recovery"
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
        worker = asyncio.create_task(self._deadline_worker(stop)) if self.deadline_safety else None
        self._deadline_worker_task = worker
        completed = False
        try:
            await self._run_control(stop)
            completed = True
        finally:
            try:
                if worker is not None:
                    if completed:
                        # A normal cohort stop still owes a complete receipt
                        # for a clock action already in flight. Wake an idle
                        # worker and join it instead of cancelling its execute.
                        stop.set()
                        self.router.deadline_event.set()
                        await worker
                    else:
                        worker.cancel()
                        try:
                            await worker
                        except asyncio.CancelledError:
                            pass
            finally:
                for task in tuple(self._frequency_tasks):
                    task.cancel()
                if self._frequency_tasks:
                    await asyncio.gather(*self._frequency_tasks, return_exceptions=True)
        self.log('stop', roles=dict(self.roles))

    async def _run_control(self, stop: asyncio.Event) -> None:
        now = time.time()
        self.forecaster.set_backlog(backlog_snapshot(self.router))
        fc = self.forecaster.forecast(now)
        plan = self.initial_plan or (self.planner.plan(fc) if self._informed(fc, now) else self._fail_open_plan())
        plan = self._startup_plan(plan)
        if self.planner.capacity_reserve_enabled:
            plan = self.planner.enforce_capacity_floor(plan, fc)
        await self.execute(plan)
        self._m_floor_last_change_s = time.time()
        next_plan_at = now + self.period_s
        last_level = 0
        while not stop.is_set():
            await asyncio.sleep(self.tick_s)
            if self._frequency_snapshot_error is not None:
                raise self._frequency_snapshot_error
            if self._deadline_worker_task is not None and self._deadline_worker_task.done():
                await self._deadline_worker_task
            now = time.time()
            self.forecaster.set_backlog(backlog_snapshot(self.router))
            level = 0
            pressure = None
            if self.shield is not None:
                observations = (self.router.observation_records(60.0, now)
                                if hasattr(self.router, 'observation_records') else self.router.recent(60.0, now))
                pressure = self.shield.observe(observations, now)
                if self.shield.needs_collective_clock_probe(pressure, now):
                    available = await self._collective_probe_available(pressure)
                    level = self.shield.update(pressure, now, clock_probe_available=available)
                else:
                    level = self.shield.update(pressure, now)
            scheduled = now >= next_plan_at
            fc_now = self.forecaster.forecast(now) if (self.dynamic_m_floor or self.planner.capacity_reserve_enabled) else None
            strategy_changed, urgent = self._update_strategy(fc_now, pressure, level, now, scheduled)
            m_pressure = self._m_pressure(self.plan_now) if self.dynamic_m_floor else 0.0
            floor_plan = None
            if (self.planner.capacity_reserve_enabled and self.planner.cfg.capacity_floor_reserve_canonical
                    and self.plan_now is not None):
                # The planner owns the exact count/clock/workload domain. A
                # qualified low clock need not restore capacity; treating every
                # non-maximal clock as an exit would starve periodic planning.
                floor_plan = self.planner.enforce_capacity_floor(self.plan_now, fc_now)
            floor_restore = bool(floor_plan is not None and
                ({role:n for role,n in floor_plan.counts.items() if n}
                 != {role:n for role,n in self.plan_now.counts.items() if n}
                 or floor_plan.f_M != self.plan_now.f_M))
            replan = (not self.freeze) and (scheduled or level != last_level or strategy_changed or floor_restore)
            if replan:
                fc = fc_now if fc_now is not None else self.forecaster.forecast(now)
                if (self.startup_safety and not self._startup_validated
                        and self._informed(fc, now) and self._startup_observed_safe(now)):
                    self._startup_validated = True
                    self.log('startup_validated', real_requests=self.router.admitted_requests,
                             real_traffic_s=now-self.router.first_arrival_s)
                planner_invoked = False
                if floor_restore:
                    # Qualification exit first restores the current layout's
                    # reserved M capacity. Ordinary P/D changes wait for a later
                    # planning tick and cannot hide inside this safety action.
                    plan = floor_plan
                elif self.startup_safety and not self._startup_validated:
                    plan = self._startup_plan(self.plan_now or self.initial_plan)
                elif self._informed(fc, now) or (urgent and self.dynamic_m_floor):
                    planner_invoked = True
                    plan = await asyncio.to_thread(self.planner.plan, fc, self.plan_now)
                elif self.hold_initial and self.plan_now is not None:
                    plan = self.plan_now
                else:
                    plan = self._fail_open_plan()
                planner_candidate = plan
                candidate = plan
                anchor = (None if floor_restore or self._is_safety_recovery(candidate, pressure, fc)
                          else self._anchor_candidate(fc, candidate))
                if anchor is not None:
                    candidate = plan = anchor
                protected = bool(self.shield and self.shield.protection_active(now))
                if floor_restore:
                    reason = 'capacity_floor_restore'
                elif urgent:
                    reason = 'pressure_safety_override'
                else:
                    plan, reason = self._gate_plan_change(plan, now, scheduled=scheduled,
                                                           shield_protected=protected,
                                                           pressure=pressure, forecast=fc)
                if self.shield is not None and (level or self.shield.floor_active):
                    # Apply safety *after* ordinary-change gating so escalation
                    # can immediately raise clocks/wake instances during a hold.
                    if (not urgent and not floor_restore and level > last_level
                            and (self.min_plan_hold_s or self.down_plan_votes > 1)
                            and not self._is_safety_recovery(plan, pressure, fc)):
                        plan = self.plan_now or plan
                    clock_probe_only = (getattr(self.shield, 'collective_clock_probe_only', False)
                                        and not self._deadline_floor_mhz)
                    if self.planner.capacity_reserve_enabled and not clock_probe_only:
                        # Reserved slots become M before Shield allocates any
                        # additional P/D capacity. A collective-only clock probe
                        # has no capacity evidence, including its quiet ticks.
                        # The unconditional check below still rejects any clock
                        # or workload change outside the exact low-M evidence.
                        plan = self.planner.enforce_capacity_floor(plan, fc, force_canonical=True,
                                                                  preserve_restoration=floor_restore)
                    plan = self.shield.apply(plan, pressure, self.max_freq)
                    if ((self.plan_now is None or plan.key() != self.plan_now.key())
                            and reason != "safety_recovery"):
                        reason = "shield_override"
                    if level:
                        self._down_candidate_key, self._down_votes = None, 0
                if self.planner.capacity_reserve_enabled:
                    # Holds and Shield may reuse an older plan. Recheck its
                    # qualification against this tick's complete workload.
                    plan = self.planner.enforce_capacity_floor(plan, fc, preserve_restoration=True)
                    if (plan.detail.get('capacity_floor_restoration')
                            and (self.plan_now is None or plan.key() != self.plan_now.key())):
                        reason = 'capacity_floor_restore'
                plan = self._deadline_hold(plan, now, scheduled)
                if self.startup_safety or self.deadline_safety:
                    plan = self.planner.refresh_estimate(plan, fc)
                self.log("forecast", rate_rps=fc.rate_rps, trend_rps=fc.trend_rps, input_mean=fc.input_mean,
                         input_p95=fc.input_p95, output_mean=fc.output_mean, inflight=fc.inflight,
                         pending_prefill_tokens=fc.pending_prefill_tokens,
                         remaining_decode_tokens=fc.remaining_decode_tokens,
                         occupied_kv_tokens=fc.occupied_kv_tokens,
                         paired_length_samples=len(fc.length_pairs),
                         recent_rate_rps=fc.recent_rate_rps, decision_reason=reason,
                         planner_invoked=planner_invoked,
                         candidate_fallback=bool(planner_invoked and planner_candidate.detail.get('fallback')),
                         candidate_fallback_reason=(planner_candidate.detail.get('fallback_reason', 'planner_no_feasible_candidate')
                             if planner_invoked and planner_candidate.detail.get('fallback') else None),
                         candidate_capacity_estimate_available=(planner_candidate.detail.get('capacity_estimate_available')
                             if planner_invoked else None),
                         down_votes=self._down_votes, candidate_counts=candidate.counts,
                         candidate_clocks=dict(P=candidate.f_P, D=candidate.f_D, M=candidate.f_M),
                         shield_level=level, pressure=None if pressure is None else vars(pressure),
                         m_floor=self._m_floor, m_pressure=m_pressure,
                         model_pressure=self._last_pressure, risk_windows=self._risk_windows,
                         route_pressure=self.router.pressure_state() if hasattr(self.router, "pressure_state") else None)
                if self.plan_now is None or plan.key() != self.plan_now.key():
                    await self.execute(plan)
                if scheduled or (not self.safety_recovery and not self.dynamic_m_floor):
                    next_plan_at = time.time() + self.period_s
            last_level = level

    def summary(self) -> dict:
        kinds = {}
        for row in self._log:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        return dict(events=kinds, shield_events=list(self.shield.events) if self.shield else [],
                    final_roles=dict(self.roles), transition_phases=self.transition_events)
