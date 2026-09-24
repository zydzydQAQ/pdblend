"""Fast SLO shield: rule table that escalates capacity when observed latency approaches the SLO.

The planner runs every ~10 s on forecasts; the shield runs every second on observations and can only
make the deployment more capable (raise clocks, add active instances). Its escalation level decays
one step per quiet cooldown so the planner regains control gradually once the pressure is gone.

Hold-down: during an escalation episode the shield records the largest active-instance count it
needed (the floor). After the level decays to zero the floor is released one instance at a time,
each step only after a quiet probe window; a step that brings pressure back restores the previous
floor and doubles the next window (capped at 8x). This stops the planner from walking straight back
into the capacity shortage that caused the episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

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

    def protection_active(self, now: float) -> bool:
        return self.protect_s > 0.0 and now < self.protect_until

    def observe(self, records, now: float) -> Pressure:
        ttfts, tpots, stuck, stalled, longest_gap = [], [], 0, 0, 0.0
        for r in records:
            if r.error:
                continue
            if r.first_token_s is None:
                if now - r.submitted_s > self.threshold * self.slo.ttft_s:
                    stuck += 1
            else:
                if r.finished_s is None:
                    last = getattr(r, 'last_token_s', None)
                    last = r.first_token_s if last is None else last
                    gap = max(0.0, now - last)
                    longest_gap = max(longest_gap, gap)
                    if gap > self.threshold * self.slo.tpot_s:
                        stalled += 1
                if r.first_token_s >= now - self.window_s:
                    ttfts.append(r.first_token_s - r.submitted_s)
                end = r.finished_s or now
                tokens = r.completion_tokens if r.finished_s else r.tokens_so_far
                if tokens and tokens >= 2 and end >= now - self.window_s:
                    tpots.append((end - r.first_token_s) / (tokens - 1))
        p = Pressure(ttft_p90=percentile(ttfts, 0.9), tpot_p90=percentile(tpots, 0.9), stuck=stuck,
                     decode_stalled=stalled, longest_token_gap_s=longest_gap)
        p.prefill = stuck > 0 or p.ttft_p90 > self.threshold * self.slo.ttft_s
        p.decode = stalled > 0 or p.tpot_p90 > self.threshold * self.slo.tpot_s
        return p

    def update(self, pressure: Pressure, now: float) -> int:
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

    def apply(self, plan: Plan, pressure: Pressure, max_freq: int) -> Plan:
        """Return a more capable copy of `plan`: escalation level plus the hold-down floor."""
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
