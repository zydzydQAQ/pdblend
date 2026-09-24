"""Periodic pool planner: choose pool sizes, parking depth and per-pool clocks minimising predicted
node power subject to queueing-model TTFT/TPOT constraints. Pure enumeration, milliseconds in Python.

Roles per slot: P (prefill), D (decode), M (mixed), idle (drained, reset clock), L1 (weights resident,
mem/SM clocks pinned to their floor), off (process stopped).
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Optional

from pdblend.profile.query.model import PerfModel
from pdblend.profile.query.runtime import RuntimeProfile, require_planner_components
from pdblend.planner.forecast import Forecast

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


@dataclass(frozen=True)
class QualifiedCapacityFloor:
    """Explicit GPU acceptance domain; an empty/unqualified domain cannot relax policy."""
    model_id: str
    tp: int
    pp: int
    min_m_instances: int
    rate_range: tuple[float, float]
    input_range: tuple[float, float]
    output_range: tuple[float, float]
    evidence: str
    qualified: bool = False
    profile_key: str = ""
    accepted_slo: tuple[float, float] | None = None
    version: int = 1
    frequency_mhz: int | None = None
    context: dict = field(default_factory=dict)

    def matches(self, model: PerfModel, fc: Forecast, slo: SLO | None = None, **kwargs) -> bool:
        return self.rejection_reason(model, fc, slo, **kwargs) is None

    def rejection_reason(self, model: PerfModel, fc: Forecast, slo: SLO | None = None, *,
                         context=None, n_m=None, frequency_mhz=None) -> str | None:
        identity = model.profile_key.get("model_id", model.model)
        if not self.qualified or not self.evidence or self.min_m_instances < 1:
            return "missing_qualified_acceptance"
        if self.model_id != identity or (self.tp, self.pp) != (model.tp, model.pp):
            return "model_or_topology_mismatch"
        if self.profile_key and self.profile_key != json.dumps(model.profile_key, sort_keys=True, separators=(",", ":")):
            return "profile_mismatch"
        if self.version == 2:
            if not self.context or context != self.context:
                return "source_workload_or_recovery_mismatch"
            if n_m is not None and n_m != self.min_m_instances:
                return "mixed_count_outside_accepted_domain"
            if frequency_mhz is not None and frequency_mhz != self.frequency_mhz:
                return "frequency_outside_accepted_domain"
            if self.accepted_slo is None or slo is None or (slo.ttft_s, slo.tpot_s) != tuple(self.accepted_slo):
                return "slo_outside_accepted_domain"
        elif self.version != 1:
            return "unsupported_capacity_floor_version"
        if self.accepted_slo is not None and (slo is None or slo.ttft_s < self.accepted_slo[0]
                                              or slo.tpot_s < self.accepted_slo[1]):
            return "slo_stricter_than_accepted"
        inputs = fc.inputs or (fc.input_mean, fc.input_p95)
        outputs = tuple(o for _, o in fc.length_pairs) or fc.outputs or (fc.output_mean,)
        if not self.rate_range[0] <= fc.rate_rps <= self.rate_range[1]:
            return "rate_outside_accepted_domain"
        if not self.input_range[0] <= min(inputs) <= max(inputs) <= self.input_range[1]:
            return "input_outside_accepted_domain"
        if not self.output_range[0] <= min(outputs) <= max(outputs) <= self.output_range[1]:
            return "output_outside_accepted_domain"
        if not all(self.input_range[0] <= w.input_tokens <= self.input_range[1]
                   and w.remaining_output_tokens <= self.output_range[1] for w in fc.backlog):
            return "backlog_outside_accepted_domain"
        return None


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
    capacity_floors: tuple[QualifiedCapacityFloor, ...] = ()
    capacity_floor_reserve_canonical: bool = False  # comparison keeps parked M restoration capacity
    capacity_floor_context: dict = field(default_factory=dict)
    memoize_roles: bool = True
    preserve_overload_capacity: bool = False  # PDblend recovery; legacy baseline fallback is unchanged
    transition_estimator: Callable[[Plan, Plan], float | dict | None] | None = None


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
    def __init__(self, model: RuntimeProfile, config: PlannerConfig):
        require_planner_components(model, allow_pd=config.allow_pd, allow_dvfs=config.allow_dvfs)
        self.model = model
        self.cfg = replace(config, freqs=config.freqs or model.freqs)
        self._peak_cache: dict = {}
        self._quantile_cache: tuple = (None, ([], []))
        self._split_cache: dict = {}
        self._role_cache: dict | None = None
        self._branch_cache: dict = {}
        self._floor: int | None = None
        self._capacity_decision_cache = None

    @property
    def capacity_reserve_enabled(self) -> bool:
        """Whether controller safety must restore the canonical M reserve."""
        return bool(self.cfg.capacity_floors)

    def mixed_floor(self, fc: Forecast) -> int:
        qualified = [q.min_m_instances for q in self.cfg.capacity_floors if q.matches(
            self.model, fc, self.cfg.slo, context=self.cfg.capacity_floor_context)]
        return min(qualified) if qualified else self.cfg.min_m_instances

    def capacity_floor_decision(self, fc: Forecast, *, n_m=None, frequency_mhz=None) -> dict | None:
        """Bind the exact workload and qualification used to relax the M reserve."""
        if not self.cfg.capacity_floors:
            return None
        if (n_m is None and frequency_mhz is None and self._capacity_decision_cache is not None
                and self._capacity_decision_cache[0] is fc):
            return self._capacity_decision_cache[1]
        from .capacity import capacity_floor_decision
        return capacity_floor_decision(self.cfg.capacity_floors, self.model, fc, self.cfg.slo,
                                       canonical_floor=self.cfg.min_m_instances,
                                       context=self.cfg.capacity_floor_context,
                                       n_m=n_m, frequency_mhz=frequency_mhz)

    def _bind_capacity_floor(self, plan: Plan, fc: Forecast) -> Plan:
        decision = self.capacity_floor_decision(fc, n_m=plan.counts.get('M', 0), frequency_mhz=plan.f_M)
        if decision is not None:
            plan.detail["capacity_floor"] = decision
        return plan

    def enforce_capacity_floor(self, plan: Plan, fc: Forecast, *, force_canonical=False,
                               preserve_restoration=False) -> Plan:
        """Refresh evidence and wake reserved slots without reducing P/D capacity.

        Comparison candidates reserve this space before selection. During Shield
        escalation it becomes M capacity before Shield may allocate other spares.
        """
        decision = self.capacity_floor_decision(fc, n_m=plan.counts.get('M', 0), frequency_mhz=plan.f_M)
        if decision is None:
            return plan
        detail = dict(plan.detail, capacity_floor=decision)
        if not preserve_restoration:
            detail.pop('capacity_floor_restoration', None)
        if not self.cfg.capacity_floor_reserve_canonical:
            return replace(plan, detail=detail)
        canonical = min(self.cfg.min_m_instances, self.cfg.slots)
        counts = dict(plan.counts)
        if (sum(counts.values()) != self.cfg.slots
                or counts.get('P', 0) + counts.get('D', 0) > self.cfg.slots - canonical):
            raise ValueError('capacity-floor plan lacks reserved canonical M restoration slots')
        unsupported_clock = (counts.get('M', 0) < canonical and plan.f_M != max(self.cfg.freqs)
            and not any(q.version == 2 and q.matches(self.model, fc, self.cfg.slo,
                context=self.cfg.capacity_floor_context, n_m=counts.get('M', 0), frequency_mhz=plan.f_M)
                for q in self.cfg.capacity_floors))
        unsupported_layout = (counts.get('M',0) < canonical and any(q.version == 2 for q in self.cfg.capacity_floors)
            and (any(n for role,n in counts.items() if role not in ('M','L1')) or plan.tau != 0))
        required = canonical if force_canonical or unsupported_clock or unsupported_layout else min(decision['effective_floor'], self.cfg.slots)
        needed = max(0, required - counts.get('M', 0))
        if not needed:
            return replace(plan, detail=detail)
        restored = needed
        for role in PARKED:
            take = min(needed, counts.get(role, 0))
            counts[role] = counts.get(role, 0) - take
            counts['M'] = counts.get('M', 0) + take
            needed -= take
        if needed:
            raise ValueError('capacity-floor restoration cannot reduce owned P/D capacity')
        counts = {role: n for role, n in counts.items() if n or role in ACTIVE}
        detail['capacity_floor_restoration'] = dict(from_counts=dict(plan.counts), to_counts=dict(counts),
            restored_instances=restored, required_floor=required,
            reason=('shield_capacity_reserve' if force_canonical else
                    'unqualified_low_m_frequency' if unsupported_clock else
                    'unqualified_low_m_layout' if unsupported_layout else 'qualified_domain_exit'))
        detail.update(capacity_estimate_available=False, capacity_insufficient=True)
        restored_plan = replace(plan, counts=counts, f_M=max(self.cfg.freqs), detail=detail,
                       power_w=float('inf'), ttft_s=float('inf'), tpot_s=float('inf'))
        return self._bind_capacity_floor(restored_plan, fc)

    def refresh_estimate(self, plan: Plan, forecast: Forecast) -> Plan:
        """Estimate the final applied layout without changing its deployment identity."""
        try:
            fresh = self.evaluate(plan.counts, plan.f_P, plan.f_D, plan.f_M, plan.tau,
                                  forecast, strict=False)
        except ValueError as exc:
            if 'outside measured coverage' not in str(exc) and 'missing_profile:' not in str(exc):
                raise
            fresh = None
        detail = {key: value for key, value in plan.detail.items()
                  if key not in ('P', 'D', 'M', 'forecast', 'prediction_unavailable')}
        if fresh is None or not all(math.isfinite(value) for value in (fresh.power_w,fresh.ttft_s,fresh.tpot_s)):
            detail.update(prediction_unavailable='final_layout_outside_model_or_capacity_domain',
                          capacity_estimate_available=False)
            result = replace(plan, power_w=float('inf'), ttft_s=float('inf'),
                             tpot_s=float('inf'), detail=detail)
        else:
            detail.update(fresh.detail)
            detail['capacity_estimate_available'] = True
            result = replace(plan, power_w=fresh.power_w, ttft_s=fresh.ttft_s,
                             tpot_s=fresh.tpot_s, detail=detail)
        return self._bind_capacity_floor(result, forecast)

    def _branches(self, fc: Forecast, tau: int) -> tuple[Forecast, Forecast]:
        if self._branch_cache.get("ref") is not fc:
            self._branch_cache = {"ref": fc}
        if tau not in self._branch_cache:
            self._branch_cache[tau] = fc.split_forecasts(tau)
        return self._branch_cache[tau]

    def _role(self, role: str, fc: Forecast, n: int, f: int):
        key = (role, id(fc), n, f)
        if self._role_cache is not None and key in self._role_cache:
            return self._role_cache[key]
        try:
            if role == "P":
                result = self._prefill_pool(fc.rate_rps, fc, fc.input_mean, fc.input_p95, n, f)
            elif role == "D":
                result = self._decode_pool(fc.rate_rps, fc, fc.input_mean, n, f)
            else:
                result = self._mixed_pool(fc.rate_rps, fc, fc.input_mean, fc.input_p95, n, f)
        except ValueError as exc:
            if "outside measured coverage" not in str(exc) and "missing_profile:" not in str(exc):
                raise
            result = None
        if self._role_cache is not None:
            self._role_cache[key] = result
        return result

    def _prefill_rate(self, rate: float, fc: Forecast, in_mean: float) -> float:
        return rate + fc.pending_prefill_tokens / max(self.cfg.dwell_s * in_mean, 1e-9)

    def _decode_rate(self, rate: float, fc: Forecast) -> float:
        return rate + fc.remaining_decode_tokens / max(self.cfg.dwell_s * fc.output_mean, 1e-9)

    def _backlog_wait(self, fc: Forecast, in_mean: float, n: int, f: int) -> float:
        return (fc.pending_prefill_tokens / max(in_mean * n, 1)
                * self.model.prefill_marginal_seconds(int(in_mean), f))

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
        lam = self._prefill_rate(rate, fc, in_mean) / n
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
        ttft += self._backlog_wait(fc, in_mean, n, f)
        return dict(power_w=power, ttft_s=ttft, busy=u,
                    pending_prefill_tokens=fc.pending_prefill_tokens)

    def _peak_decode_tps(self, ctx: float, f: int) -> float:
        key = (ctx, f)
        if key not in self._peak_cache:
            self._peak_cache[key] = max(
                (b / self.model.step_seconds(b, ctx, f)
                 # Long contexts can fit only 1..7 requests in physical KV.
                 # Sampling multiples of eight incorrectly reports zero
                 # capacity even though valid smaller batches can serve.
                 for b in range(1, min(self.cfg.max_num_seqs, self.cfg.peak_batch_cap) + 1)
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
        ctx = max(in_mean + fc.output_mean / 2.0,
                  fc.occupied_kv_tokens / max(len(fc.backlog), 1))
        rate = self._decode_rate(rate, fc)
        if rate * fc.output_mean / n > self.cfg.rho_decode * self._peak_decode_tps(ctx, f):
            return None
        b = max(self._decode_batch(rate, fc.output_mean, ctx, n, f), len(fc.backlog) / n)
        active_batch = max(1.0, b) if (self.model.bounded_coverage or self.model.decode_power_overrides) else b
        if (not self.model.decode_supported(active_batch, ctx, f) or
                not self.model.decode_power_supported(active_batch, ctx, f)):
            return None
        if b > self.cfg.peak_batch_cap or max(b * (in_mean + fc.output_mean), fc.occupied_kv_tokens / n) > self.model.kv_capacity_tokens * 0.9:
            return None
        step = self.model.step_seconds(active_batch, ctx, f)
        idle = self.model.static_power_w("active_idle", f)
        if b >= 1.0:
            power = self.model.decode_power_w(b, f, ctx=ctx)
        else:
            power = idle + b * max(self.model.decode_power_w(1.0, f, ctx=ctx) - idle, 0.0)
        return dict(power_w=n * power, tpot_s=step, first_step_s=step, batch=b,
                    remaining_decode_tokens=fc.remaining_decode_tokens,
                    occupied_kv_tokens=fc.occupied_kv_tokens)

    def _mixed_pool(self, rate: float, fc: Forecast, in_mean: float, in_p95: float, n: int, f: int):
        if n == 0:
            return None
        # Chunked prefill piggybacks on decode steps: a prompt adds only its marginal token cost to the
        # step it rides in, so the busy fraction excludes the per-step intercept.
        s_p = self.model.prefill_marginal_seconds(int(in_mean), f)
        prefill_rate = self._prefill_rate(rate, fc, in_mean)
        decode_rate = self._decode_rate(rate, fc)
        u_p = prefill_rate * s_p / n
        if u_p >= self.cfg.rho_max:
            return None
        ctx = max(in_mean + fc.output_mean / 2.0,
                  fc.occupied_kv_tokens / max(len(fc.backlog), 1))
        b = max(self._decode_batch(decode_rate, fc.output_mean, ctx, n, f, dilution=u_p), len(fc.backlog) / n)
        active_batch = max(1.0, b) if (self.model.bounded_coverage or self.model.decode_power_overrides) else b
        if (not self.model.decode_supported(active_batch, ctx, f) or
                not self.model.decode_power_supported(active_batch, ctx, f)):
            return None
        if b > self.cfg.peak_batch_cap or max(b * (in_mean + fc.output_mean), fc.occupied_kv_tokens / n) > self.model.kv_capacity_tokens * 0.9:
            return None
        if decode_rate * fc.output_mean / n > self.cfg.rho_decode * (1.0 - u_p) * self._peak_decode_tps(ctx, f):
            return None
        step = self.model.step_seconds(active_batch, ctx, f)
        tpot = step / (1.0 - u_p)
        wait = (mdc_wait(prefill_rate, s_p, n) or 0.0) + self._backlog_wait(fc, in_mean, n, f)
        ttft = wait + self.model.prefill_seconds(int(in_p95), f) + tpot
        idle = self.model.static_power_w("active_idle", f)
        p_pre = self.model.prefill_power_w(int(in_mean), f)
        p_dec = self.model.decode_power_w(b, f, ctx=ctx) if b >= 1.0 else idle + b * max(self.model.decode_power_w(1.0, f, ctx=ctx) - idle, 0.0)
        power = u_p * p_pre + (1.0 - u_p) * p_dec
        energy_model = "composed_unqualified"
        supported = getattr(self.model, "mixed_power_supported", None)
        if supported and supported(active_batch, ctx, f, chunk_tokens=int(in_mean),
                                   prefill_rate_rps=prefill_rate / n):
            power = self.model.mixed_power_w(active_batch, ctx, f, chunk_tokens=int(in_mean),
                                            prefill_rate_rps=prefill_rate / n)
            energy_model = "measured_mixed_window"
        return dict(power_w=n * power, ttft_s=ttft, tpot_s=tpot, batch=b, busy=u_p,
                    tpot_miss=self._stall_miss(prefill_rate / n, fc, step, f),
                    pending_prefill_tokens=fc.pending_prefill_tokens,
                    remaining_decode_tokens=fc.remaining_decode_tokens,
                    occupied_kv_tokens=fc.occupied_kv_tokens, energy_model=energy_model)

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
        if fc.length_pairs:
            pairs = quantiles(fc.length_pairs, 36)
        else:
            pairs = [(i, o) for i in ins for o in outs]
        for i, o in pairs:
            stall_pre = sum(self.model.prefill_marginal_seconds(max(j - (budget - i), 0), f) for j in ins) / len(ins)
            w_pre = self.model.prefill_seconds(i, f)
            k = max(o - 1, 1)
            w_dec = k * step
            stall = (w_pre * stall_pre + w_dec * stall_dec) / (w_pre + w_dec)
            if stall > 0.0:
                miss += poisson_tail(lam * (w_pre + w_dec), math.floor(room * k / stall) + 1)
        return miss / len(pairs)

    # ---- evaluation --------------------------------------------------------------------------
    def evaluate(self, counts: dict, f_P: int, f_D: int, f_M: int, tau: int, fc: Forecast,
                 strict: bool = True) -> Optional[Plan]:
        """Predicted plan for a layout; None if the queues are unstable or (strict) the SLO is missed."""
        slo = self.cfg.slo
        n_P, n_D, n_M = counts.get("P", 0), counts.get("D", 0), counts.get("M", 0)
        if (n_M < self.cfg.min_m_instances and any(q.version == 2 for q in self.cfg.capacity_floors)
                and (sum(counts.values()) != self.cfg.slots or tau != 0
                     or any(n for role,n in counts.items() if role not in ('M','L1')))):
            return None
        floor = self.mixed_floor(fc) if self._floor is None else self._floor
        decision = self.capacity_floor_decision(fc, n_m=n_M, frequency_mhz=f_M)
        if decision is not None:
            floor = decision['effective_floor']
        if self.cfg.capacity_floors and self.cfg.capacity_floor_reserve_canonical:
            canonical = min(self.cfg.min_m_instances, self.cfg.slots)
            if (n_P + n_D > self.cfg.slots - canonical or n_M < min(floor, self.cfg.slots)
                    or (n_M < canonical and f_M != max(self.cfg.freqs)
                        and not any(q.version == 2 and q.matches(self.model, fc, slo,
                            context=self.cfg.capacity_floor_context, n_m=n_M, frequency_mhz=f_M)
                            for q in self.cfg.capacity_floors))):
                return None
        if 0 < n_M < min(floor, self.cfg.slots):
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
        if fc.backlog and ((not has_pd and any(w.branch in {"PD", "P_ONLY"} for w in fc.backlog))
                           or (not has_m and any(w.branch == "M" for w in fc.backlog))):
            return None
        if has_pd and has_m:
            share, in_pd, in_m = self._split(fc, tau)
        elif has_pd:
            share, in_pd, in_m = 1.0, fc.input_mean, fc.input_mean
        else:
            share, in_pd, in_m = 0.0, fc.input_mean, fc.input_mean
        if has_pd and has_m:
            fc_pd, fc_m = self._branches(fc, tau)
            if (share <= 0 and not fc_pd.backlog) or (share >= 1 and not fc_m.backlog):
                return None
        else:
            fc_pd = fc_m = fc
        rate_pd, rate_m = fc_pd.rate_rps, fc_m.rate_rps
        if has_pd and fc_pd.input_mean < self.cfg.min_pd_input_tokens:
            return None
        power = ttft = tpot = 0.0
        detail = {}
        if has_pd:
            p = self._role("P", fc_pd, n_P, f_P)
            d = self._role("D", fc_pd, n_D, f_D)
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
                reserved = replace(fc_m, rate_rps=rate_m * 1.25)
                if self.mixed_pressure(reserved, n_M, f_M)['pressure'] > .55:
                    return None
            m = self._role("M", fc_m, n_M, f_M)
            if m is None or (strict and (m["ttft_s"] > slo.ttft_s * slo.safety or m["tpot_s"] > slo.tpot_s * slo.safety
                                         or m["tpot_miss"] > 1.0 - self.cfg.tail_target)):
                return None
            power += m["power_w"]
            ttft, tpot = max(ttft, m["ttft_s"]), max(tpot, m["tpot_s"])
            detail["M"] = m
        for role in PARKED:
            power += counts.get(role, 0) * self.model.static_power_w(PARK_STATE[role])
        detail["forecast"] = dict(pd_output_mean=fc_pd.output_mean if has_pd else None,
                                  mixed_output_mean=fc_m.output_mean if has_m else None,
                                  pending_prefill_tokens=fc.pending_prefill_tokens,
                                  remaining_decode_tokens=fc.remaining_decode_tokens,
                                  occupied_kv_tokens=fc.occupied_kv_tokens,
                                  mixed_floor=floor)
        return self._bind_capacity_floor(Plan(dict(counts), f_P, f_D, f_M, tau, power, ttft, tpot, detail,
                    tp=self.model.tp, pp=self.model.pp,
                    profile_key=json.dumps(self.model.profile_key, sort_keys=True, separators=(",", ":"))
                    if self.model.profile_key else ""), fc)

    # ---- enumeration -------------------------------------------------------------------------
    def _count_options(self) -> Iterable[dict]:
        N = self.cfg.slots
        floor = self.cfg.min_m_instances if self._floor is None else self._floor
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
                if (n_M and n_M < min(floor, N)) or (
                        not n_M and floor and not self.cfg.pressure_controls):
                    continue
                rest = active - n_M
                pd_splits = [(0, 0)] if rest == 0 else ([(p, rest - p) for p in range(1, rest)] if self.cfg.allow_pd else [])
                for n_P, n_D in pd_splits:
                    for ps in park_splits:
                        yield {"P": n_P, "D": n_D, "M": n_M, **{k: v for k, v in ps.items() if v}}

    def candidates(self, fc: Forecast) -> list[Plan]:
        # Each call owns the memo. No stale result survives model/config or backlog changes.
        self._role_cache = {} if self.cfg.memoize_roles else None
        self._peak_cache.clear()
        self._branch_cache.clear()
        self._split_cache.clear()
        self._floor = self.mixed_floor(fc)
        self._capacity_decision_cache = (fc, self.capacity_floor_decision(fc))
        try:
            return self._enumerate_candidates(fc)
        finally:
            self._role_cache = None
            self._floor = None
            self._capacity_decision_cache = None

    def _enumerate_candidates(self, fc: Forecast) -> list[Plan]:
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
        """Extra transition energy, with measured provenance or explicit legacy fallback."""
        if current is None:
            return 0.0
        from .transitions import signature
        if signature(current) == signature(new):
            new.detail["transition_cost"] = dict(provenance="no_state_change", energy_j=0.0, qualified=True)
            return 0.0
        audit = dict(qualified=False, provenance="legacy_wake_and_clock_estimate")
        if self.cfg.transition_estimator is not None:
            result = self.cfg.transition_estimator(current, new)
            if isinstance(result, dict):
                audit = dict(result)
                value = result.get("incremental_energy_j") if result.get("qualified") else None
                if value is None and result.get("qualified_only"):
                    new.detail["transition_cost"] = dict(audit, policy="skipped_unqualified")
                    return float("inf")
            else:
                value = result
                audit = dict(qualified=False, provenance="external_numeric_estimate")
            if value is not None:
                measured = float(value)
                if not math.isfinite(measured) or measured < 0:
                    raise ValueError("transition estimator must return finite nonnegative joules")
                new.detail["transition_cost"] = dict(audit, energy_j=measured)
                return measured
            audit["fallback"] = "legacy_wake_and_clock_estimate"
        estimated = self._legacy_switch_energy_j(current, new)
        new.detail["transition_cost"] = dict(audit, energy_j=estimated)
        return estimated

    def _legacy_switch_energy_j(self, current: Plan, new: Plan) -> float:
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
            # Specialized/legacy planners may implement the original one-arg
            # fallback. Only the explicitly enabled recovery passes ownership.
            fallback = (self.fallback(fc, current) if self.cfg.preserve_overload_capacity
                        else self.fallback(fc))
            if current is not None:
                fallback.pool_id, fallback.generation = current.pool_id, current.generation
            return fallback
        if current is None:
            return plans[0]
        horizon = self.cfg.dwell_s
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError("planning horizon must be positive and finite")
        # A cheap steady-state winner can be an expensive transition. Rank every
        # candidate with its own transition cost before applying hysteresis.
        for candidate in plans:
            candidate.pool_id, candidate.generation = current.pool_id, current.generation
        scored = [(horizon * p.power_w + self.switch_energy_j(current, p), index, p)
                  for index, p in enumerate(plans)]
        scored = [entry for entry in scored if math.isfinite(entry[0])]
        cur = self.evaluate(current.counts, current.f_P, current.f_D, current.f_M, current.tau, fc)
        if cur is not None:
            # Retain topology identity on a reevaluated current plan.
            cur.tp, cur.pp, cur.pool_id = current.tp, current.pp, current.pool_id
            cur.generation, cur.profile_key = current.generation, current.profile_key
        if not scored:
            if cur is not None:
                cur.detail["objective"] = dict(horizon_s=horizon, transition_energy_j=0.0,
                                               total_energy_j=horizon * cur.power_w, retained_current=True,
                                               reason="no_qualified_transition")
                return cur
            raise ValueError("missing_profile: no transition-qualified feasible candidate")
        cost, _, best = min(scored, key=lambda item: (item[0], item[1]))
        forced_pd = (self.cfg.pressure_controls and self.cfg.pd_pressure_active
                     and best.counts.get("P", 0) and not current.counts.get("P", 0))
        if cur is not None and not forced_pd and cost >= horizon * cur.power_w * (1.0 - self.cfg.margin):
            cur.detail["objective"] = dict(horizon_s=horizon, transition_energy_j=0.0,
                                           total_energy_j=horizon * cur.power_w, retained_current=True)
            return cur
        best.detail["objective"] = dict(horizon_s=horizon,
                                        transition_energy_j=cost - horizon * best.power_w,
                                        total_energy_j=cost, retained_current=False)
        return best

    def _capacity_preserving_fallback(self, fc: Forecast, current: Plan) -> Plan:
        """Retain the known deployment when the model cannot certify a replacement.

        An empty feasible set is not evidence that P can be reduced to one.
        Without a measured bottleneck, every existing role keeps its capacity
        and its routing threshold. Spare slots may augment those roles only
        when the resulting full-clock layout has a finite capacity estimate.
        """
        f, slots = max(self.cfg.freqs), self.cfg.slots
        counts = dict(current.counts)
        if sum(counts.values()) != slots:
            raise ValueError("current fallback plan does not match the instance inventory")
        if (any(w.branch in {"PD", "P_ONLY"} for w in fc.backlog)
                and not (counts.get("P", 0) and counts.get("D", 0))
                or any(w.branch == "M" for w in fc.backlog) and not counts.get("M", 0)):
            raise ValueError("current fallback plan does not preserve backlog ownership")
        roles = [role for role in ACTIVE if counts.get(role, 0)]
        spare = slots - current.active()
        options = [counts]
        if spare and roles:
            for additions in itertools.product(range(spare + 1), repeat=len(roles)):
                if sum(additions) == spare:
                    options.append({role: counts.get(role, 0) + extra
                                    for role, extra in zip(roles, additions)})
        finite = []
        for layout in options:
            candidate = self.evaluate(layout, f, f, f, current.tau, fc, strict=False)
            if candidate is not None and all(math.isfinite(value) for value in
                    (candidate.power_w, candidate.ttft_s, candidate.tpot_s)):
                finite.append(candidate)
        if finite:
            slo = self.cfg.slo
            best = min(finite, key=lambda p: (max(p.ttft_s / slo.ttft_s,
                                                 p.tpot_s / slo.tpot_s), p.power_w))
            best.tp, best.pp, best.pool_id = current.tp, current.pp, current.pool_id
            best.generation, best.profile_key = current.generation, current.profile_key
            best.detail.update(fallback=True, fallback_reason="capacity_preserving_full_clock",
                               capacity_insufficient=True, capacity_estimate_available=True)
            return best
        # These infinities explicitly mean "no capacity prediction"; unlike
        # the legacy fallback they never authorize a new role split or tau.
        return replace(current, counts=counts, f_P=f, f_D=f, f_M=f,
                       power_w=float("inf"), ttft_s=float("inf"), tpot_s=float("inf"),
                       detail=dict(fallback=True, fallback_reason="retain_uncertified_capacity",
                                   capacity_insufficient=True, capacity_estimate_available=False,
                                   preserved_backlog_branches=True))

    def fallback(self, fc: Forecast, current: Optional[Plan] = None) -> Plan:
        """Opt-in capacity-preserving recovery, or the legacy full-clock fallback."""
        return self.enforce_capacity_floor(self._fallback(fc, current), fc)

    def _fallback(self, fc: Forecast, current: Optional[Plan] = None) -> Plan:
        if self.cfg.preserve_overload_capacity and current is not None:
            return self._capacity_preserving_fallback(fc, current)
        N, f = self.cfg.slots, max(self.cfg.freqs)
        has_pd_work = any(w.branch in {"PD", "P_ONLY"} for w in fc.backlog)
        has_m_work = any(w.branch == "M" for w in fc.backlog)
        if has_pd_work and N >= (3 if has_m_work else 2):
            n_m = max(1, min(self.mixed_floor(fc), N - 2)) if has_m_work else 0
            counts = {"P": 1, "D": N - 1 - n_m, "M": n_m}
            # Emergency fallback must keep every currently owned path alive.
            # The controller still drains physical instances before any role change.
            tau = self.cfg.pd_min_input_tokens if has_m_work else 0
            candidate = self.evaluate(counts, f, f, f, tau, fc, strict=False)
            if candidate is not None:
                candidate.detail["fallback"] = True
                return candidate
            return Plan(counts, f, f, f, tau, float("inf"), float("inf"), float("inf"),
                        dict(fallback=True, preserved_backlog_branches=True),
                        tp=self.model.tp, pp=self.model.pp,
                        profile_key=json.dumps(self.model.profile_key, sort_keys=True, separators=(",", ":"))
                        if self.model.profile_key else "")
        counts = {"M": N}
        m = self._mixed_pool(fc.rate_rps, fc, fc.input_mean, fc.input_p95, N, f)
        if m is None and self.model.decode_power_overrides:
            ctx = fc.input_mean + fc.output_mean / 2
            u = fc.rate_rps * self.model.prefill_marginal_seconds(int(fc.input_mean), f) / N
            batch = self._decode_batch(fc.rate_rps, fc.output_mean, ctx, N, f, dilution=u)
            if not self.model.decode_power_supported(max(1., batch), ctx, f):
                from pdblend.profile.query.power_table import PowerCoverageError
                raise PowerCoverageError('missing_profile: fallback layout has no measured decode power coverage')
        m = m or dict(power_w=float("inf"), ttft_s=float("inf"), tpot_s=float("inf"))
        return Plan(counts, f, f, f, 0, m["power_w"], m["ttft_s"], m["tpot_s"], dict(fallback=True, M=m),
                    tp=self.model.tp, pp=self.model.pp,
                    profile_key=json.dumps(self.model.profile_key, sort_keys=True, separators=(",", ":"))
                    if self.model.profile_key else "")


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
