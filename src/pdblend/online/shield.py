"""Fast SLO shield: rule table that escalates capacity when observed latency approaches the SLO.

The planner runs every ~10 s on forecasts; the shield runs every second on observations and can only
make the deployment more capable (raise clocks, add active instances). Its escalation level decays
one step per quiet cooldown so the planner regains control gradually once the pressure is gone.
In budget-aware mode, capacity is an absolute target advanced by an observed
escalation, never by repeatedly applying the same level to its own output. Short
collective gaps get a bounded clock probe, with no implied capacity shortage.

Hold-down: during an escalation episode the shield records the largest active-instance count it
needed (the floor). After the level decays to zero the floor is released one instance at a time,
each step only after a quiet probe window; a step that brings pressure back restores the previous
floor and doubles the next window (capped at 8x). This stops the planner from walking straight back
into the capacity shortage that caused the episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from pdblend.planner.forecast import percentile
from pdblend.planner.pool import ACTIVE, PARKED, Plan, SLO


@dataclass
class Pressure:
    prefill: bool = False        # TTFT approaching SLO or requests waiting for a first token too long
    decode: bool = False         # TPOT approaching SLO
    ttft_p90: float = 0.0
    tpot_p90: float = 0.0
    stuck: int = 0
    decode_stalled: int = 0
    longest_token_gap_s: float = 0.0
    decode_active: int = 0
    decode_stalled_fraction: float = 0.0
    decode_sustained_stalls: int = 0
    decode_budget_risks: int = 0
    decode_short_output_risks: int = 0
    prefill_paths: tuple[str, ...] = ()
    decode_paths: tuple[str, ...] = ()
    mode: str = "legacy"
    criteria: dict = field(default_factory=dict)


@dataclass
class Shield:
    slo: SLO
    threshold: float = 0.8       # fraction of SLO that triggers escalation
    cooldown_s: float = 30.0     # quiet window per decay step; base length of a probe window
    window_s: float = 5.0
    level: int = 0               # 0 none, 1 max clocks, k>=2 max clocks + (k-1) extra active instances
    since_s: float = 0.0
    last_escalation_s: float = 0.0
    floor_active: int = 0        # hold-down floor: active slots currently required
    peak_active: int = 0         # largest floor of the current episode
    probe_windows: float = 1.0   # probe-window length in units of cooldown_s (backoff on failure)
    release_s: float = 0.0       # start of the current probe window (level reached 0)
    events: list = field(default_factory=list)
    protect_s: float = 0.0       # PDblend-only protection lease after an escalation
    protect_until: float = 0.0
    mode: str = "legacy"
    sustained_gap_s: float = 1.0
    stalled_fraction: float = 0.25
    stalled_min_requests: int = 2
    escalation_sequence: int = 0
    capacity_target_active: int = 0
    _target_initialized: bool = field(default=False, repr=False)
    _pending_capacity: list[Pressure] = field(default_factory=list, repr=False)
    _capacity_pool: str = field(default="M", repr=False)
    _clock_roles: set[str] = field(default_factory=set, repr=False)
    _collective_active: bool = field(default=False, repr=False)
    _collective_probe_only: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.mode not in {"legacy", "budget_aware"}:
            raise ValueError("unknown Shield mode")
        if (not math.isfinite(self.sustained_gap_s) or self.sustained_gap_s <= 0
                or not 0 < self.stalled_fraction <= 1
                or type(self.stalled_min_requests) is not int or self.stalled_min_requests < 2):
            raise ValueError("invalid Shield sustained/collective stall thresholds")

    def protection_active(self, now: float) -> bool:
        return self.protect_s > 0.0 and now < self.protect_until

    @property
    def collective_clock_probe_only(self) -> bool:
        """This entire episode has only clock evidence, including quiet ticks.

        A caller must still enforce its exact clock/workload qualification and
        any independent deadline recovery. Existing capacity targets/floors
        must never be erased by a later short collective observation.
        """
        return (self.mode == 'budget_aware' and self.level == 1
                and self._collective_probe_only and self.floor_active == 0
                and self.capacity_target_active == 0 and not self._pending_capacity)

    def observe(self, records, now: float) -> Pressure:
        ttfts, tpots, stuck, stalled, longest_gap = [], [], 0, 0, 0.0
        active = sustained = budget_risks = short_risks = 0
        prefill_paths, decode_paths, gap_paths = set(), set(), set()
        limit = self.threshold * self.slo.tpot_s
        for r in records:
            if r.error:
                continue
            path = "PD" if r.path in {"PD", "P_ONLY"} else "M"
            if r.first_token_s is None:
                if now - r.submitted_s > self.threshold * self.slo.ttft_s:
                    stuck += 1
                    prefill_paths.add(path)
            else:
                if r.finished_s is None and (self.mode == "legacy" or r.tokens_so_far < r.max_tokens):
                    active += int(r.max_tokens > 1)
                    last = getattr(r, 'last_token_s', None)
                    last = r.first_token_s if last is None else last
                    gap = max(0.0, now - last)
                    longest_gap = max(longest_gap, gap)
                    if gap > limit and (self.mode == "legacy" or r.max_tokens > 1):
                        stalled += 1
                        gap_paths.add(path)
                    if r.max_tokens > 1 and gap >= self.sustained_gap_s:
                        sustained += 1
                        decode_paths.add(path)
                    # A two-token response has exactly one TPOT interval. A
                    # carry token followed by a delayed D token cannot amortise it.
                    if r.max_tokens == 2 and r.tokens_so_far < 2 and gap > limit:
                        short_risks += 1
                        decode_paths.add(path)
                    # Even instantaneous future tokens could not recover this
                    # output budget. Otherwise a lone gap is classified separately
                    # from completed inter-token timing and sustained stalls.
                    if r.max_tokens > 1 and (now - r.first_token_s) / (r.max_tokens - 1) > limit:
                        budget_risks += 1
                        decode_paths.add(path)
                if r.first_token_s >= now - self.window_s:
                    ttfts.append(r.first_token_s - r.submitted_s)
                    if ttfts[-1] > self.threshold * self.slo.ttft_s:
                        prefill_paths.add(path)
                end = r.finished_s or now
                tokens = r.completion_tokens if r.finished_s else r.tokens_so_far
                if tokens and tokens >= 2 and end >= now - self.window_s:
                    measured_end = end
                    if self.mode == "budget_aware":
                        measured_end = r.last_token_s if r.last_token_s is not None else end
                    tpots.append((measured_end - r.first_token_s) / (tokens - 1))
                    if tpots[-1] > limit:
                        decode_paths.add(path)
        fraction = stalled / active if active else 0.0
        p = Pressure(ttft_p90=percentile(ttfts, 0.9), tpot_p90=percentile(tpots, 0.9), stuck=stuck,
                     decode_stalled=stalled, longest_token_gap_s=longest_gap,
                     decode_active=active, decode_stalled_fraction=fraction,
                     decode_sustained_stalls=sustained, decode_budget_risks=budget_risks,
                     decode_short_output_risks=short_risks, mode=self.mode,
                     criteria=dict(threshold=self.threshold, sustained_gap_s=self.sustained_gap_s,
                                   stalled_fraction=self.stalled_fraction,
                                   stalled_min_requests=self.stalled_min_requests))
        p.prefill = stuck > 0 or p.ttft_p90 > self.threshold * self.slo.ttft_s
        collective = stalled >= self.stalled_min_requests and fraction >= self.stalled_fraction
        p.decode = (stalled > 0 or p.tpot_p90 > limit) if self.mode == "legacy" else (
            sustained > 0 or budget_risks > 0 or short_risks > 0 or collective or p.tpot_p90 > limit)
        if self.mode == "legacy" or collective:
            decode_paths.update(gap_paths)
        p.prefill_paths = tuple(sorted(prefill_paths)) if p.prefill else ()
        p.decode_paths = tuple(sorted(decode_paths)) if p.decode else ()
        return p

    def needs_collective_clock_probe(self, pressure: Pressure, now: float) -> bool:
        return (self.mode == 'budget_aware' and self._collective_only(pressure)
                and not self._collective_active and self.level == 0
                and now - self.last_escalation_s >= 2.0)

    def update(self, pressure: Pressure, now: float, *, clock_probe_available=None) -> int:
        if self.mode == "budget_aware":
            return self._update_budget(pressure, now, clock_probe_available=clock_probe_available)
        if pressure.prefill or pressure.decode:
            if self.level == 0 and self.floor_active:
                # A probe at reduced capacity brought the pressure back: step the floor back up
                # (unless pressure reappeared at the episode peak itself) and back off.
                if self.floor_active < self.peak_active:
                    self.floor_active += 1
                    self.probe_windows = min(self.probe_windows * 2, 8.0)
                self.release_s = now
                self.events.append(dict(t=now, level=0, floor=self.floor_active, probe="failed"))
            if now - self.last_escalation_s >= 2.0:       # at most one step every 2 s
                self.level += 1
                self.last_escalation_s = now
                if self.protect_s > 0.0:
                    self.protect_until = max(self.protect_until, now + self.protect_s)
                self.events.append(dict(t=now, level=self.level, ttft_p90=pressure.ttft_p90,
                                        tpot_p90=pressure.tpot_p90, stuck=pressure.stuck,
                                        decode_stalled=pressure.decode_stalled,
                                        longest_token_gap_s=pressure.longest_token_gap_s,
                                        pressure=vars(pressure),
                                        protect_until=self.protect_until))
            self.since_s = now
        elif self.level:
            if now - self.since_s >= self.cooldown_s:     # decay one step per quiet cooldown
                self.level -= 1
                self.since_s = now
                self.events.append(dict(t=now, level=self.level))
                if self.level == 0 and self.floor_active:
                    self.release_s = now
                    self.events.append(dict(t=now, level=0, floor=self.floor_active, probe="start"))
        elif self.floor_active and now - self.release_s >= self.probe_windows * self.cooldown_s:
            self.floor_active -= 1                        # quiet probe: release one slot
            self.probe_windows = max(1.0, self.probe_windows / 2)
            self.release_s = now
            if self.floor_active == 0:
                self.peak_active = 0
                self.probe_windows = 1.0
            self.events.append(dict(t=now, level=0, floor=self.floor_active, probe="step"))
        return self.level

    def _collective_only(self, pressure: Pressure) -> bool:
        """Short correlated gaps justify a clock probe, not a capacity claim."""
        return (pressure.decode and not pressure.prefill
                and pressure.decode_stalled >= self.stalled_min_requests
                and pressure.decode_stalled_fraction >= self.stalled_fraction
                and not pressure.decode_sustained_stalls
                and not pressure.decode_budget_risks
                and not pressure.decode_short_output_risks
                and pressure.tpot_p90 <= self.threshold * self.slo.tpot_s)

    @staticmethod
    def _pressure_roles(pressure: Pressure) -> set[str]:
        roles = set()
        for enabled, paths, pd_role in ((pressure.prefill, pressure.prefill_paths, "P"),
                                        (pressure.decode, pressure.decode_paths, "D")):
            if enabled:
                if not paths:  # Older callers have no path attribution.
                    roles.update(ACTIVE)
                else:
                    roles.update("M" if path == "M" else pd_role for path in paths)
        return roles

    def _update_budget(self, pressure: Pressure, now: float, *, clock_probe_available=None) -> int:
        collective = self._collective_only(pressure)
        collective_edge = collective and not self._collective_active
        self._collective_active = collective
        strong = (pressure.prefill or pressure.decode) and not collective
        if strong:
            # Even within the two-second escalation throttle, actual pressure
            # ends the collective-only exemption immediately.
            self._collective_probe_only = False
            if self.level == 0 and self.floor_active:
                if self.floor_active < self.peak_active:
                    self.floor_active += 1
                    self.probe_windows = min(self.probe_windows * 2, 8.0)
                self.release_s = now
                self.events.append(dict(t=now, level=0, floor=self.floor_active, probe="failed"))
            if now - self.last_escalation_s >= 2.0:
                self.level += 1
                self.last_escalation_s = now
                self.escalation_sequence += 1
                self._clock_roles.update(self._pressure_roles(pressure))
                if self.level >= 2:
                    self._pending_capacity.append(replace(pressure))
                if self.protect_s > 0.0:
                    self.protect_until = max(self.protect_until, now + self.protect_s)
                self.events.append(dict(t=now, level=self.level,
                    escalation_sequence=self.escalation_sequence,
                    capacity_escalation=self.level >= 2, pressure=vars(pressure),
                    ttft_p90=pressure.ttft_p90, tpot_p90=pressure.tpot_p90,
                    stuck=pressure.stuck, decode_stalled=pressure.decode_stalled,
                    longest_token_gap_s=pressure.longest_token_gap_s,
                    protect_until=self.protect_until))
            self.since_s = now
        elif collective_edge and self.level == 0 and now - self.last_escalation_s >= 2.0:
            if clock_probe_available is False:
                # Only a fresh physical snapshot of every affected GPU can
                # suppress this no-op. Real latency/budget pressure takes the
                # strong branch above, irrespective of clock headroom.
                self.events.append(dict(t=now, level=0, capacity_escalation=False,
                    collective_clock_probe=False, clock_probe_noop=True,
                    reason='already_at_measured_ceiling', pressure=vars(pressure)))
                return self.level
            # An entire collective episode gets one bounded frequency probe.
            # Repeated short gaps neither extend this clock probe nor restore a
            # released floor; only actual latency/budget pressure can do that.
            self.level = 1
            self._collective_probe_only = True
            self.last_escalation_s = self.since_s = now
            self.escalation_sequence += 1
            self._clock_roles = self._pressure_roles(pressure)
            self.events.append(dict(t=now, level=1, escalation_sequence=self.escalation_sequence,
                capacity_escalation=False, collective_clock_probe=True, pressure=vars(pressure),
                ttft_p90=pressure.ttft_p90, tpot_p90=pressure.tpot_p90,
                stuck=pressure.stuck, decode_stalled=pressure.decode_stalled,
                longest_token_gap_s=pressure.longest_token_gap_s,
                protect_until=self.protect_until))
        elif self.level:
            if now - self.since_s >= self.cooldown_s:
                self.level -= 1
                self.since_s = now
                self.events.append(dict(t=now, level=self.level))
                if self.level == 0:
                    self._collective_probe_only = False
                    self.capacity_target_active = 0
                    self._target_initialized = False
                    self._pending_capacity.clear()
                    self._clock_roles.clear()
                    if self.floor_active:
                        self.release_s = now
                        self.events.append(dict(t=now, level=0, floor=self.floor_active, probe="start"))
        elif self.floor_active and now - self.release_s >= self.probe_windows * self.cooldown_s:
            self.floor_active -= 1
            self.probe_windows = max(1.0, self.probe_windows / 2)
            self.release_s = now
            if self.floor_active == 0:
                self.peak_active = 0
                self.probe_windows = 1.0
            self.events.append(dict(t=now, level=0, floor=self.floor_active, probe="step"))
        return self.level

    def _pressured_pool(self, counts: dict, pressure: Pressure) -> str:
        roles = self._pressure_roles(pressure)
        if not roles:
            return self._capacity_pool
        if roles == {"M"} or not (counts.get("P", 0) and counts.get("D", 0)):
            return "M"
        if "M" in roles and ("M" in pressure.prefill_paths or "M" in pressure.decode_paths):
            return "M"
        if roles == {"P"}:
            return "P"
        if roles == {"D"}:
            return "D"
        return "D" if counts.get("D", 0) <= counts.get("P", 0) else "P"

    @staticmethod
    def _wake_to_target(counts: dict, target: int, pool: str) -> None:
        extra = target - sum(counts.get(role, 0) for role in ACTIVE)
        for role in PARKED:
            while extra > 0 and counts.get(role, 0) > 0:
                counts[role] -= 1
                counts[pool] = counts.get(pool, 0) + 1
                extra -= 1

    def _apply_budget(self, plan: Plan, pressure: Pressure, max_freq: int) -> Plan:
        counts = dict(plan.counts)
        active = sum(counts.get(role, 0) for role in ACTIVE)
        slots = sum(counts.values())
        if self.level and not self.collective_clock_probe_only:
            if not self._target_initialized:
                self.capacity_target_active = max(active, self.floor_active)
                # Preserve explicit construction at a nonzero level while
                # making its first application the sole anchoring operation.
                if not self.escalation_sequence and not self._pending_capacity:
                    self.capacity_target_active = min(slots, self.capacity_target_active + max(0, self.level - 1))
                self._target_initialized = True
                self._capacity_pool = self._pressured_pool(counts, pressure)
            for observed in self._pending_capacity:
                active = sum(counts.get(role, 0) for role in ACTIVE)
                self.capacity_target_active = min(slots, max(self.capacity_target_active, active) + 1)
                self._capacity_pool = self._pressured_pool(counts, observed)
                self._wake_to_target(counts, self.capacity_target_active, self._capacity_pool)
            self._pending_capacity.clear()
            target = max(self.capacity_target_active, self.floor_active)
        else:
            target = self.floor_active
        self._wake_to_target(counts, target, self._capacity_pool)
        counts = {k: v for k, v in counts.items() if v or k in ACTIVE}
        detail = dict(plan.detail, shield_level=self.level, shield_floor=self.floor_active,
                      shield_escalation_sequence=self.escalation_sequence,
                      shield_capacity_target_active=self.capacity_target_active,
                      shield_capacity_pool=self._capacity_pool)
        if self.level:
            if self.level >= 2:
                achieved = sum(counts.get(role, 0) for role in ACTIVE)
                self.floor_active = max(self.floor_active, min(achieved, self.capacity_target_active))
                self.peak_active = max(self.peak_active, self.floor_active)
                detail["shield_floor"] = self.floor_active
            roles = self._clock_roles or self._pressure_roles(pressure) or {self._capacity_pool}
            detail["shield_clock_roles"] = sorted(roles)
            clocks = {"f_" + role: max_freq for role in roles}
            return replace(plan, counts=counts, detail=detail, **clocks)
        detail["shield_clock_roles"] = []
        return replace(plan, counts=counts, detail=detail)

    def apply(self, plan: Plan, pressure: Pressure, max_freq: int) -> Plan:
        """Return a more capable copy of `plan`: escalation level plus the hold-down floor."""
        if self.mode == "budget_aware":
            return self._apply_budget(plan, pressure, max_freq)
        counts = dict(plan.counts)
        active = sum(counts.get(r, 0) for r in ACTIVE)
        target = max(active + max(0, self.level - 1), self.floor_active)
        extra = target - active
        # Wake the shallowest parked slots into the pressured pool; P/D pressure grows its own pool,
        # a mixed-only deployment grows M.
        pool = "M"
        if counts.get("P", 0) and counts.get("D", 0):
            if pressure.prefill and not pressure.decode:
                pool = "P"
            elif pressure.decode and not pressure.prefill:
                pool = "D"
            else:
                pool = "D" if counts.get("D", 0) <= counts.get("P", 0) else "P"
            if counts.get("M", 0) and pool == "M":
                pool = "M"
        for role in PARKED:
            while extra > 0 and counts.get(role, 0) > 0:
                counts[role] -= 1
                counts[pool] = counts.get(pool, 0) + 1
                extra -= 1
        counts = {k: v for k, v in counts.items() if v or k in ACTIVE}
        if self.level:
            achieved = sum(counts.get(r, 0) for r in ACTIVE)
            self.floor_active = max(self.floor_active, achieved)
            self.peak_active = max(self.peak_active, achieved)
            detail = dict(plan.detail, shield_level=self.level)
            return replace(plan, counts=counts, f_P=max_freq, f_D=max_freq, f_M=max_freq,
                           detail=detail)
        if counts == plan.counts:
            return plan
        detail = dict(plan.detail, shield_floor=self.floor_active)
        return replace(plan, counts=counts, detail=detail)
