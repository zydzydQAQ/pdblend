"""Periodic pool planner: choose pool sizes, parking depth and per-pool clocks minimising predicted
node power subject to queueing-model TTFT/TPOT constraints. Pure enumeration, milliseconds in Python.

Roles per slot: P (prefill), D (decode), M (mixed), idle (drained, reset clock), L1 (weights resident,
mem/SM clocks pinned to their floor), off (process stopped).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

from ..profile.model import PerfModel
from .forecast import Forecast

ACTIVE = ("P", "D", "M")
PARKED = ("idle", "L1", "off")
PARK_STATE = {"idle": "active_idle_reset", "L1": "parked", "off": "off"}
PREFILL_FREQS = (2100, 2520)
TAUS = (0, 1024, 4096)          # PD/M split thresholds when both pool kinds exist


@dataclass(frozen=True)
class SLO:
    ttft_s: float
    tpot_s: float
    safety: float = 0.85         # planner targets safety * SLO so the 90% joint attainment has headroom


@dataclass
class Plan:
    counts: dict                 # role -> number of slots
    f_P: int
    f_D: int
    f_M: int
    tau: int
    power_w: float
    ttft_s: float
    tpot_s: float
    detail: dict = field(default_factory=dict)
    tp: int = 1
    pp: int = 1
    pool_id: str = ""
    generation: int = 0
    profile_key: str = ""

    def active(self) -> int:
        return sum(self.counts.get(r, 0) for r in ACTIVE)

    def key(self):
        return (tuple(sorted(self.counts.items())), self.f_P, self.f_D, self.f_M, self.tau,
                self.tp, self.pp, self.pool_id, self.generation, self.profile_key)


@dataclass
class PlannerConfig:
    slots: int
    slo: SLO
    freqs: tuple = ()
    rho_max: float = 0.7
    dwell_s: float = 60.0        # amortisation window for wake/switch energy
    margin: float = 0.03         # relative saving required before changing a plan
    allow_pd: bool = True
    allow_park: tuple = ("L1", "off")
    allow_dvfs: bool = True
    fixed_mixed: bool = False    # Mixed / Mixed+DVFS baselines: every slot is M
    min_active: int = 1
    max_num_seqs: int = 256
    peak_batch_cap: int = 160  # decode model grid ends at B=64; beta>=0 extrapolation is conservative up to ~B=160 (m3-validated)
    max_num_batched_tokens: int = 8192
    rho_decode: float = 0.92     # decode pools: offered tokens/s over peak tokens/s at the clock
    tail_target: float = 0.9     # mixed pools: predicted share of requests meeting TPOT despite prefill stalls
    # Below this prompt size the fixed per-request cost of disaggregation (connector handshake,
    # receive-side KV handling on D, two-hop proxying) outweighs the prefill compute that PD
    # parallelises; measured PD TTFT is seconds while the queueing model predicts milliseconds.
    min_pd_input_tokens: float = 256.0
    min_m_instances: int = 0       # empirical safety floor; enabled only for PDblend candidate policy
    pressure_controls: bool = False
    pd_pressure_active: bool = False
    pd_min_input_tokens: int = 1024
    pure_pd_min_input_tokens: int = 2048


def mdc_wait(rate: float, service_s: float, servers: int) -> Optional[float]:
    """Mean queueing delay for Poisson arrivals split JSQ-style over `servers` M/D/1 queues."""
    if servers <= 0:
        return None
    rho = rate * service_s / servers
    if rho >= 1.0:
        return None
    return rho * service_s / (2.0 * (1.0 - rho))


def poisson_tail(mean: float, k: int) -> float:
    """P(N >= k) for N ~ Poisson(mean)."""
    if k <= 0:
        return 1.0
    if k > mean + 8.0 * math.sqrt(mean) + 12.0:
        return 0.0
    p = math.exp(-mean)
    cdf = p
    for i in range(1, k):
        p *= mean / i
        cdf += p
    return max(0.0, 1.0 - cdf)


def quantiles(values, q: int) -> list:
    values = sorted(values)
    return [values[min(len(values) - 1, int((j + 0.5) / q * len(values)))] for j in range(q)] if values else []


class PoolPlanner:
    def __init__(self, model: PerfModel, config: PlannerConfig):
        self.model = model
        self.cfg = replace(config, freqs=config.freqs or model.freqs)
        self._peak_cache: dict = {}
        self._quantile_cache: tuple = (None, ([], []))
        self._split_cache: dict = {}

    def mixed_pressure(self, fc: Forecast, n: int, f: int) -> dict:
        """Full offered load on M, including decode, KV and short-output stall risk.

        This is telemetry for the experimental policy. It does not change the
        additive performance model or baseline feasibility decisions.
        """
        if n <= 0:
            return dict(pressure=2.0, feasible=False)
        m = self._mixed_pool(fc.rate_rps, fc, fc.input_mean, fc.input_p95, n, f)
        if m is None:
            return dict(pressure=2.0, feasible=False)
        ctx = fc.input_mean + fc.output_mean / 2
        peak = self._peak_decode_tps(ctx, f)
        decode = fc.rate_rps * fc.output_mean / max(n * peak * (1 - m['busy']), 1e-9)
        kv = m['batch'] * (fc.input_mean + fc.output_mean) / max(self.model.kv_capacity_tokens * .9, 1)
        prefill = m['busy'] / self.cfg.rho_max
        tail = m['tpot_miss'] / max(1 - self.cfg.tail_target, .01)
        latency = max(m['ttft_s'] / (self.cfg.slo.ttft_s * self.cfg.slo.safety),
                      m['tpot_s'] / (self.cfg.slo.tpot_s * self.cfg.slo.safety))
        pressure = max(prefill, decode / self.cfg.rho_decode, kv, tail, latency)
        return dict(pressure=pressure, feasible=pressure <= 1, decode_utilization=decode,
                    prefill_utilization=prefill, kv_utilization=kv, tail_risk=tail,
                    capacity_margin=1 - pressure, **m)

    # ---- pool models -------------------------------------------------------------------------
    def _decode_batch(self, rate: float, out_mean: float, ctx: float, servers: int, f: int,
                      dilution: float = 0.0) -> float:
        """Little's law fixed point: L = rate * out_mean * tpot(B), B = L / servers, tpot = step(B)/(1-dilution)."""
        b, cap = 1.0, 4.0 * self.cfg.max_num_seqs
        for _ in range(32):
            # Fractional B is time-average occupancy. A non-idle decode step
            # still executes at least one sequence; idle duty cycle is modeled
            # separately below. Never extrapolate a bounded profile below B1.
            active_batch = max(1.0, b) if (self.model.bounded_coverage or self.model.decode_power_overrides) else b
            if not self.model.decode_supported(active_batch, ctx, f):
                return float('inf')
            tpot = self.model.step_seconds(active_batch, ctx, f) / max(1.0 - dilution, 1e-6)
            nxt = max(rate * out_mean * tpot / servers, 1e-6)
            if nxt > cap:
                return nxt
            if abs(nxt - b) < 1e-3 * max(b, 1.0):
                return nxt
            b = nxt
        return b

    def _prefill_pool(self, rate: float, fc: Forecast, in_mean: float, in_p95: float, n: int, f: int):
        if n == 0:
            return None
        # The engine prefills every waiting prompt in one step, so throughput is bound by tokens
        # (marginal cost) while the per-step intercept is paid once per step, not per request.
        lam = rate / n
        s_full, s_m = self.model.prefill_seconds(int(in_mean), f), self.model.prefill_marginal_seconds(int(in_mean), f)
        u = lam * s_m
        if u >= self.cfg.rho_max:
            return None
        period = max(s_full - s_m, 0.0) / (1.0 - u)
        wait = min(mdc_wait(lam, s_full, 1) or float("inf"), period)
        busy = 1.0 - math.exp(-lam * s_full)
        tokens_per_step = in_mean * max(1.0, lam * period)
        idle = self.model.static_power_w("active_idle", f)
        p_dyn = self.model.prefill_power_w(int(tokens_per_step), f)
        power = n * (idle + busy * max(p_dyn - idle, 0.0))
        ttft = wait + max(self.model.prefill_seconds(int(in_p95), f), period) + self.model.transfer_seconds(int(in_p95))
        return dict(power_w=power, ttft_s=ttft, busy=u)

    def _peak_decode_tps(self, ctx: float, f: int) -> float:
        key = (round(ctx), f)
        if key not in self._peak_cache:
            self._peak_cache[key] = max(
                (b / self.model.step_seconds(b, ctx, f)
                 for b in range(8, min(self.cfg.max_num_seqs, self.cfg.peak_batch_cap) + 1, 8)
                 if self.model.decode_supported(b, ctx, f)), default=0.0)
        return self._peak_cache[key]

    def _length_quantiles(self, fc: Forecast) -> tuple[list, list]:
        held, cached = self._quantile_cache
        if held is None or held[0] is not fc.inputs or held[1] is not fc.outputs:
            cached = (quantiles(fc.inputs, 6), quantiles(fc.outputs, 6))
            self._quantile_cache = ((fc.inputs, fc.outputs), cached)
        return cached

    def _split(self, fc: Forecast, tau: int) -> tuple[float, float, float]:
        """fc.split over thousands of samples is the dominant enumeration cost; one result per (forecast, tau)."""
        cache = self._split_cache
        if cache.get("ref") is not fc.inputs:   # holds a reference, so the id cannot be recycled
            cache.clear()
            cache["ref"] = fc.inputs
        if tau not in cache:
            cache[tau] = fc.split(tau)
        return cache[tau]

    def _decode_pool(self, rate: float, fc: Forecast, in_mean: float, n: int, f: int):
        if n == 0:
            return None
        ctx = in_mean + fc.output_mean / 2.0
        if rate * fc.output_mean / n > self.cfg.rho_decode * self._peak_decode_tps(ctx, f):
            return None
        b = self._decode_batch(rate, fc.output_mean, ctx, n, f)
        active_batch = max(1.0, b) if (self.model.bounded_coverage or self.model.decode_power_overrides) else b
        if (not self.model.decode_supported(active_batch, ctx, f) or
                not self.model.decode_power_supported(active_batch, ctx, f)):
            return None
        if b > self.cfg.peak_batch_cap or b * (in_mean + fc.output_mean) > self.model.kv_capacity_tokens * 0.9:
            return None
        step = self.model.step_seconds(active_batch, ctx, f)
        idle = self.model.static_power_w("active_idle", f)
        if b >= 1.0:
            power = self.model.decode_power_w(b, f, ctx=ctx)
        else:
            power = idle + b * max(self.model.decode_power_w(1.0, f, ctx=ctx) - idle, 0.0)
        return dict(power_w=n * power, tpot_s=step, first_step_s=step, batch=b)

    def _mixed_pool(self, rate: float, fc: Forecast, in_mean: float, in_p95: float, n: int, f: int):
        if n == 0:
            return None
        # Chunked prefill piggybacks on decode steps: a prompt adds only its marginal token cost to the
        # step it rides in, so the busy fraction excludes the per-step intercept.
        s_p = self.model.prefill_marginal_seconds(int(in_mean), f)
        u_p = rate * s_p / n
        if u_p >= self.cfg.rho_max:
            return None
        ctx = in_mean + fc.output_mean / 2.0
        b = self._decode_batch(rate, fc.output_mean, ctx, n, f, dilution=u_p)
        active_batch = max(1.0, b) if (self.model.bounded_coverage or self.model.decode_power_overrides) else b
        if (not self.model.decode_supported(active_batch, ctx, f) or
                not self.model.decode_power_supported(active_batch, ctx, f)):
            return None
        if b > self.cfg.peak_batch_cap or b * (in_mean + fc.output_mean) > self.model.kv_capacity_tokens * 0.9:
            return None
        if rate * fc.output_mean / n > self.cfg.rho_decode * (1.0 - u_p) * self._peak_decode_tps(ctx, f):
            return None
        step = self.model.step_seconds(active_batch, ctx, f)
        tpot = step / (1.0 - u_p)
        wait = mdc_wait(rate, s_p, n) or 0.0
        ttft = wait + self.model.prefill_seconds(int(in_p95), f) + tpot
        idle = self.model.static_power_w("active_idle", f)
        p_pre = self.model.prefill_power_w(int(in_mean), f)
        p_dec = self.model.decode_power_w(b, f, ctx=ctx) if b >= 1.0 else idle + b * max(self.model.decode_power_w(1.0, f, ctx=ctx) - idle, 0.0)
        power = u_p * p_pre + (1.0 - u_p) * p_dec
        return dict(power_w=n * power, ttft_s=ttft, tpot_s=tpot, batch=b, busy=u_p,
                    tpot_miss=self._stall_miss(rate / n, fc, step, f))

    def _stall_miss(self, lam: float, fc: Forecast, step: float, f: int) -> float:
        """Share of requests whose mean TPOT misses the SLO because another prompt was prefilled in one
        step while they were decoding. A prompt arriving during a request's own prefill shares that
        step up to the token budget and the remainder spills into the next one; a prompt arriving
        during its decode stalls it for the whole prefill. Few output tokens cannot amortise a stall."""
        ins, outs = self._length_quantiles(fc)
        room = self.cfg.slo.tpot_s - step
        if not ins or not outs:
            return 0.0
        if room <= 0.0:
            return 1.0
        budget = self.cfg.max_num_batched_tokens
        stall_dec = sum(self.model.prefill_marginal_seconds(j, f) for j in ins) / len(ins)
        miss = 0.0
        for i in ins:
            stall_pre = sum(self.model.prefill_marginal_seconds(max(j - (budget - i), 0), f) for j in ins) / len(ins)
            w_pre = self.model.prefill_seconds(i, f)
            for o in outs:
                k = max(o - 1, 1)
                w_dec = k * step
                stall = (w_pre * stall_pre + w_dec * stall_dec) / (w_pre + w_dec)
                if stall > 0.0:
                    miss += poisson_tail(lam * (w_pre + w_dec), math.floor(room * k / stall) + 1)
        return miss / (len(ins) * len(outs))

    # ---- evaluation --------------------------------------------------------------------------
    def evaluate(self, counts: dict, f_P: int, f_D: int, f_M: int, tau: int, fc: Forecast,
                 strict: bool = True) -> Optional[Plan]:
        """Predicted plan for a layout; None if the queues are unstable or (strict) the SLO is missed."""
        slo = self.cfg.slo
        n_P, n_D, n_M = counts.get("P", 0), counts.get("D", 0), counts.get("M", 0)
        if 0 < n_M < min(self.cfg.min_m_instances, self.cfg.slots):
            return None
        has_pd, has_m = n_P > 0 and n_D > 0, n_M > 0
        if (n_P > 0) != (n_D > 0) or not (has_pd or has_m):
            return None
        if self.cfg.pressure_controls and has_pd:
            if not self.cfg.pd_pressure_active:
                return None
            if not has_m and fc.input_p95 < self.cfg.pure_pd_min_input_tokens:
                return None
            if has_m and tau != self.cfg.pd_min_input_tokens:
                return None
        if has_pd and fc.input_p95 < self.cfg.min_pd_input_tokens:
            return None     # even the longest branch cannot amortise PD fixed costs
        if has_pd and has_m:
            share, in_pd, in_m = self._split(fc, tau)
            if share <= 0.0 or share >= 1.0:
                return None
        elif has_pd:
            share, in_pd, in_m = 1.0, fc.input_mean, fc.input_mean
        else:
            share, in_pd, in_m = 0.0, fc.input_mean, fc.input_mean
        rate_pd, rate_m = fc.rate_rps * share, fc.rate_rps * (1.0 - share)
        if has_pd and in_pd < self.cfg.min_pd_input_tokens:
            return None
        power = ttft = tpot = 0.0
        detail = {}
        if has_pd:
            p = self._prefill_pool(rate_pd, fc, in_pd, fc.input_p95, n_P, f_P)
            d = self._decode_pool(rate_pd, fc, in_pd, n_D, f_D)
            if p is None or d is None:
                return None
            ttft_pd = p["ttft_s"] + d["first_step_s"]
            if strict and (ttft_pd > slo.ttft_s * slo.safety or d["tpot_s"] > slo.tpot_s * slo.safety):
                return None
            power += p["power_w"] + d["power_w"]
            ttft, tpot = max(ttft, ttft_pd), max(tpot, d["tpot_s"])
            detail.update(P=p, D=d)
        if has_m:
            if self.cfg.pressure_controls and n_M < 4:
                reserved = replace(fc, rate_rps=rate_m * 1.25)
                if self.mixed_pressure(reserved, n_M, f_M)['pressure'] > .55:
                    return None
            m = self._mixed_pool(rate_m, fc, in_m, fc.input_p95, n_M, f_M)
            if m is None or (strict and (m["ttft_s"] > slo.ttft_s * slo.safety or m["tpot_s"] > slo.tpot_s * slo.safety
                                         or m["tpot_miss"] > 1.0 - self.cfg.tail_target)):
                return None
            power += m["power_w"]
            ttft, tpot = max(ttft, m["ttft_s"]), max(tpot, m["tpot_s"])
            detail["M"] = m
        for role in PARKED:
            power += counts.get(role, 0) * self.model.static_power_w(PARK_STATE[role])
        return Plan(dict(counts), f_P, f_D, f_M, tau, power, ttft, tpot, detail)

    # ---- enumeration -------------------------------------------------------------------------
    def _count_options(self) -> Iterable[dict]:
        N = self.cfg.slots
        parks = [r for r in PARKED if r in self.cfg.allow_park]
        if self.cfg.fixed_mixed:
            yield {"M": N}
            return
        for active in range(self.cfg.min_active, N + 1):
            parked = N - active
            park_splits = [{}] if parked == 0 else [
                dict(zip(parks, c)) for c in itertools.product(range(parked + 1), repeat=len(parks)) if sum(c) == parked]
            if not parks and parked:
                continue
            for n_M in range(active + 1):
                # The experimental pressure policy treats pure P/D as an
                # escape hatch. Legacy PDblend keeps its original M-floor
                # enumeration and therefore cannot silently change results.
                if (n_M and n_M < min(self.cfg.min_m_instances, N)) or (
                        not n_M and self.cfg.min_m_instances and not self.cfg.pressure_controls):
                    continue
                rest = active - n_M
                pd_splits = [(0, 0)] if rest == 0 else ([(p, rest - p) for p in range(1, rest)] if self.cfg.allow_pd else [])
                for n_P, n_D in pd_splits:
                    for ps in park_splits:
                        yield {"P": n_P, "D": n_D, "M": n_M, **{k: v for k, v in ps.items() if v}}

    def candidates(self, fc: Forecast) -> list[Plan]:
        freqs = self.cfg.freqs if self.cfg.allow_dvfs else (max(self.cfg.freqs),)
        f_ps = [f for f in PREFILL_FREQS if f in self.cfg.freqs] or [max(self.cfg.freqs)]
        plans = []
        for counts in self._count_options():
            has_pd, has_m = counts.get("P", 0) > 0, counts.get("M", 0) > 0
            taus = ((self.cfg.pd_min_input_tokens,) if self.cfg.pressure_controls else TAUS) if (has_pd and has_m) else (0,)
            for tau in taus:
                for f_P in (f_ps if has_pd else [f_ps[-1]]):
                    for f_D in (freqs if has_pd else [freqs[-1]]):
                        for f_M in (freqs if has_m else [freqs[-1]]):
                            plan = self.evaluate(counts, f_P, f_D, f_M, tau, fc)
                            if plan is not None:
                                plans.append(plan)
        plans.sort(key=lambda p: (p.power_w, -p.active()))
        if self.cfg.pressure_controls and self.cfg.pd_pressure_active:
            pd = [p for p in plans if p.counts.get('P', 0) and p.counts.get('D', 0)]
            if pd:
                return pd
        return plans

    def switch_energy_j(self, current: Optional[Plan], new: Plan) -> float:
        """Energy to move from `current` to `new`: wake-ups at active power plus clock switches."""
        if current is None:
            return 0.0
        joules = 0.0
        active_now, active_new = current.active(), new.active()
        woken = max(0, active_new - active_now)
        # Wake the cheapest-to-wake parked slots first (idle before L1 before off).
        pool = []
        for role in PARKED:
            pool += [role] * current.counts.get(role, 0)
        for role in pool[:woken]:
            joules += self.model.wake_seconds(PARK_STATE[role]) * self.model.static_power_w("active_idle", max(self.cfg.freqs))
        if (current.f_D, current.f_M, current.f_P) != (new.f_D, new.f_M, new.f_P):
            joules += self.model.freq_switch_s * self.model.static_power_w("active_idle", max(self.cfg.freqs)) * active_new
        return joules

    def plan(self, fc: Forecast, current: Optional[Plan] = None) -> Plan:
        """Best feasible plan with hysteresis; falls back to the most capable plan if nothing is feasible."""
        plans = self.candidates(fc)
        if not plans:
            return self.fallback(fc)
        best = plans[0]
        if current is None:
            return best
        if (self.cfg.pressure_controls and self.cfg.pd_pressure_active
                and best.counts.get('P', 0) and not current.counts.get('P', 0)):
            return best
        cur = self.evaluate(current.counts, current.f_P, current.f_D, current.f_M, current.tau, fc)
        if cur is None:
            return best
        saving_w = cur.power_w - best.power_w
        cost_w = self.switch_energy_j(cur, best) / self.cfg.dwell_s
        if saving_w > cost_w + self.cfg.margin * cur.power_w:
            return best
        return cur

    def fallback(self, fc: Forecast) -> Plan:
        """Everything active at max clock; PD only if it lowers predicted TTFT/TPOT violation."""
        N, f = self.cfg.slots, max(self.cfg.freqs)
        counts = {"M": N}
        m = self._mixed_pool(fc.rate_rps, fc, fc.input_mean, fc.input_p95, N, f)
        if m is None and self.model.decode_power_overrides:
            ctx = fc.input_mean + fc.output_mean / 2
            u = fc.rate_rps * self.model.prefill_marginal_seconds(int(fc.input_mean), f) / N
            batch = self._decode_batch(fc.rate_rps, fc.output_mean, ctx, N, f, dilution=u)
            if not self.model.decode_power_supported(max(1., batch), ctx, f):
                from ..profile.power_table import PowerCoverageError
                raise PowerCoverageError('missing_profile: fallback layout has no measured decode power coverage')
        m = m or dict(power_w=float("inf"), ttft_s=float("inf"), tpot_s=float("inf"))
        return Plan(counts, f, f, f, 0, m["power_w"], m["ttft_s"], m["tpot_s"], dict(fallback=True, M=m))


def assign_roles(current: dict, counts: dict, load: Optional[dict] = None) -> dict:
    """Map role counts onto instances, changing as few instances as possible.

    current: instance_id -> role; load: instance_id -> in-flight sequences (lower is preferred for parking).
    Parked roles are ordered so instances already asleep stay asleep in preference to waking them.
    """
    load = load or {}
    remaining = {r: counts.get(r, 0) for r in ACTIVE + PARKED}
    result = {}
    for iid, role in current.items():
        if remaining.get(role, 0) > 0:
            result[iid] = role
            remaining[role] -= 1
    free = [iid for iid in current if iid not in result]
    # Wake order: instances that are already active first, then L1, L2, off.
    depth = {"P": 0, "D": 0, "M": 0, "idle": 1, "L1": 2, "off": 3}
    free.sort(key=lambda i: (depth[current[i]], load.get(i, 0)))
    for role in ACTIVE:
        while remaining[role] > 0 and free:
            result[free.pop(0)] = role
            remaining[role] -= 1
    free.sort(key=lambda i: (-depth[current[i]], load.get(i, 0)))
    for role in reversed(PARKED):
        while remaining[role] > 0 and free:
            result[free.pop(0)] = role
            remaining[role] -= 1
    for iid in free:
        result[iid] = current[iid]
    return result
