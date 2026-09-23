"""Published baselines re-expressed as decisions on the shared engine/proxy stack.

DistServe (static): the goodput-maximising prefill/decode split for the GPU budget, all clocks at max,
every request disaggregated. Placement search is the upstream rule (max feasible rate per config);
the runtime is our kv_both engines, so the KV path is host-staged like every other PD policy here.

DynamoLLM: the three-level hierarchy with its published periods. ScaleInst (1800 s) sizes the fleet
from a load template built on observed history — the max completed 60 s arrival bin over the trailing
epoch — and turns the rest off; the history comes from an unmeasured pre-window replay of the same
stationary workload (standing in for the paper's previous-week templates), and the fleet runs
fail-open (all on) until the first bin completes. ScaleShard is a no-op because TP is fixed per run;
ScaleFreq (5 s) picks the lowest clock whose predicted TTFT/TPOT meets the SLO, max clock when none
does (the paper's emergency stage). Shape pools collapse to one mixed pool: with one model size per
run there is no shard heterogeneity to exploit, and the paper's 9-bucket request-shape routing
(BERT output-length proxy) is not reproduced. Performance predictions use this repo's profile model.

EcoServe: macro groups of 2-3 mixed instances with one rotating prefill instance (temporal P/D
multiplexing, author admission rule), instance count scaled every 5 s from mean TTFT and saved TPOT
credit, drained instances parked at reset clock, fixed max clock (the artifact has no DVFS).
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Optional

from pdblend.online.router import Router
from pdblend.planner.forecast import Forecast
from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner

DYNAMO_PERIODS = dict(inst=1800.0, freq=5.0)
ECO_PERIOD_S = 5.0
ECO_MACRO = (2, 3)


def capacity_rps(planner: PoolPlanner, fc: Forecast, counts: dict, f_P: int, f_D: int, f_M: int, tau: int,
                 upper: float = 64.0, eps: float = 0.02) -> float:
    """Largest arrival rate (bisection) at which the layout still meets the planner's SLO model."""
    lo, hi = 0.0, upper
    while hi - lo > eps:
        mid = (lo + hi) / 2
        if planner.evaluate(counts, f_P, f_D, f_M, tau, replace(fc, rate_rps=mid)) is not None:
            lo = mid
        else:
            hi = mid
    return lo


def forced_plan(planner: PoolPlanner, counts: dict, f_P: int, f_D: int, f_M: int, tau: int, fc: Forecast) -> Plan:
    """The layout as a Plan even when the model predicts an SLO miss (baselines do not bail out)."""
    plan = planner.evaluate(counts, f_P, f_D, f_M, tau, fc, strict=False)
    if plan is None:
        plan = Plan(dict(counts), f_P, f_D, f_M, tau, float("inf"), float("inf"), float("inf"), dict(unstable=True))
    return plan


# ---- DistServe ---------------------------------------------------------------------------------
def distserve_plan(planner: PoolPlanner, fc: Forecast) -> Plan:
    N, f = planner.cfg.slots, max(planner.cfg.freqs)
    best, best_cap = None, -1.0
    for n_P in range(1, N):
        counts = {"P": n_P, "D": N - n_P}
        cap = capacity_rps(planner, fc, counts, f, f, f, 0)
        if cap > best_cap:
            best, best_cap = counts, cap
    plan = forced_plan(planner, best, f, f, f, 0, fc)
    plan.detail["capacity_rps"] = best_cap
    return plan


# ---- DynamoLLM ---------------------------------------------------------------------------------
class DynamoPlanner(PoolPlanner):
    def __init__(self, planner: PoolPlanner):
        super().__init__(planner.model, replace(planner.cfg, allow_pd=False, allow_park=("off",)))
        self.t0: Optional[float] = None
        self._latched_epoch: Optional[int] = None
        self.n_active: Optional[int] = None
        self._cap: dict = {}

    def _capacity_one(self, fc: Forecast) -> float:
        key = (round(fc.input_mean, -1), round(fc.input_p95, -1), round(fc.output_mean, -1))
        if key not in self._cap:
            f = max(self.cfg.freqs)
            self._cap[key] = capacity_rps(self, fc, {"M": 1}, f, f, f, 0)
        return self._cap[key]

    def scale_inst(self, fc: Forecast) -> int:
        """ceil(PL/ML) with PL = max observed 60 s bin rate over the trailing epoch (the load template)."""
        cap = self._capacity_one(fc)
        n = math.ceil(fc.peak_rps / cap) if cap > 0 else self.cfg.slots
        return max(1, min(self.cfg.slots, n))

    def plan(self, fc: Forecast, current: Optional[Plan] = None) -> Plan:
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        epoch = int((now - self.t0) // DYNAMO_PERIODS["inst"])
        if fc.completed_bins and (epoch != self._latched_epoch or epoch == 0):
            # Fail-open with no template; epoch 0 re-evaluates as bins complete (peak only grows),
            # from epoch 1 on ScaleInst fires only on epoch boundaries (the paper's 30 min period).
            self.n_active = self.scale_inst(fc)
            self._latched_epoch = epoch
        n = self.n_active if self.n_active is not None else self.cfg.slots
        counts = {"M": n}
        if n < self.cfg.slots:
            counts["off"] = self.cfg.slots - n
        f_max = max(self.cfg.freqs)
        feasible = [p for p in (self.evaluate(counts, f_max, f_max, f, 0, fc) for f in self.cfg.freqs) if p is not None]
        if feasible:
            return min(feasible, key=lambda p: (p.power_w, p.f_M))
        plan = forced_plan(self, counts, f_max, f_max, f_max, 0, fc)
        plan.detail["emergency"] = True
        return plan


# ---- EcoServe ----------------------------------------------------------------------------------
class EcoRouter(Router):
    """Author admission: keep sending prompts to the macro's prefill instance while the TTFT budgets of
    its resident requests absorb the predicted prefill time; otherwise rotate to the next member."""

    def __init__(self, instance_ids, model, slo, lower: int = ECO_MACRO[0], upper: int = ECO_MACRO[1]):
        super().__init__(instance_ids)
        self.model, self.slo, self.lower, self.upper = model, slo, lower, upper
        self.groups: list[list[str]] = []
        self.cursor: dict[int, int] = {}
        self.since: dict[str, float] = {i: time.time() for i in instance_ids}   # upstream instance.schedule_time
        self.f_max = max(model.freqs)

    def _regroup(self, active: list[str]) -> None:
        current = [i for g in self.groups for i in g]
        if sorted(current) == sorted(active):
            return
        layout: list[list[str]] = []
        for iid in active:
            if not layout:
                layout.append([iid])
                continue
            idx = next((j for j, g in enumerate(layout) if len(g) < self.upper), 0)
            layout[idx].append(iid)
            if len(layout[idx]) > self.upper:
                g = layout[idx]
                layout[idx:idx + 1] = [g[:-self.lower], g[-self.lower:]]
        self.groups = layout
        self.cursor = {j: 0 for j in range(len(layout))}
        now = time.time()
        for g in layout:
            self.since[g[0]] = now

    def _slack_ms(self, record, at_s: float) -> float:
        # upstream credit: measured TTFT + iterations * TPOT - (now - arrival)
        ttft_ms = (record.first_token_s - record.submitted_s) * 1000.0
        return ttft_ms + record.tokens_so_far * self.slo.tpot_s * 1000.0 - (at_s - record.submitted_s) * 1000.0

    def _fits(self, group: list[str], iid: str, input_tokens: int, now: float) -> bool:
        ttft_ms = self.slo.ttft_s * 1000.0
        need = self.model.prefill_seconds(input_tokens, self.f_max) * 1000.0
        pending, saved = [], []
        for r in self.active[iid]:
            if r.first_token_s is None:
                pending.append(ttft_ms)
                need += self.model.prefill_seconds(r.input_tokens, self.f_max) * 1000.0
            else:
                saved.append(self._slack_ms(r, self.since.get(iid, now)))
        remaining = pending + ([max(saved)] if saved else [])
        if not remaining or min(remaining) > need:
            return True
        if need > ttft_ms:
            return False
        nxt = group[(group.index(iid) + 1) % len(group)]
        saved_next = [self._slack_ms(r, now) for r in self.active[nxt] if r.first_token_s is not None]
        return bool(saved_next) and max(saved_next) < ttft_ms * (len(group) - 1) / len(group)

    def choose(self, input_tokens: int):
        active = self._pool("M")
        if not active:
            return None
        self._regroup(active)
        now = time.time()
        j = min(range(len(self.groups)), key=lambda k: sum(self.loads[i].inflight_seqs for i in self.groups[k]))
        group = self.groups[j]
        iid = group[self.cursor[j] % len(group)]
        if len(group) > 1 and not self._fits(group, iid, input_tokens, now):
            self.cursor[j] = (self.cursor[j] + 1) % len(group)
            iid = group[self.cursor[j]]
            self.since[iid] = now
        return "M", iid, iid


class EcoPlanner(PoolPlanner):
    def __init__(self, planner: PoolPlanner, router: Router, lower: int = ECO_MACRO[0]):
        super().__init__(planner.model, replace(planner.cfg, allow_pd=False, allow_dvfs=False, allow_park=("idle",)))
        self.router, self.lower = router, lower

    def plan(self, fc: Forecast, current: Optional[Plan] = None) -> Plan:
        N = self.cfg.slots
        n = current.counts.get("M", N) if current is not None else N
        now = time.time()
        ttfts = [r.ttft_s for r in self.router.recent(60.0, now) if r.ttft_s is not None]
        if ttfts and sum(ttfts) / len(ttfts) > self.cfg.slo.ttft_s:
            n = min(N, n + 1)
        else:
            groups = getattr(self.router, "groups", None) or [list(self.router.active)]
            for g in sorted(groups, key=lambda g: (len(g) <= self.lower, len(g), g)):
                credits = [(r.first_token_s - r.submitted_s) + r.tokens_so_far * self.cfg.slo.tpot_s - (now - r.submitted_s)
                           for i in g for r in self.router.active.get(i, []) if r.first_token_s is not None]
                if credits and sum(credits) / len(credits) > self.cfg.slo.ttft_s * (len(g) + 1) / len(g):
                    if n > self.lower:
                        n -= 1
                    break
        counts = {"M": n}
        if n < N:
            counts["idle"] = N - n
        f = max(self.cfg.freqs)
        return forced_plan(self, counts, f, f, f, 0, fc)


def build_control(policy_name: str, planner: PoolPlanner, router: Router, fc: Forecast):
    """(planner, initial_plan, freeze, period_s) for the ported baselines; None for planner-table policies."""
    if policy_name == "distserve_static":
        return planner, distserve_plan(planner, fc), True, None
    if policy_name == "dynamollm":
        return DynamoPlanner(planner), None, False, DYNAMO_PERIODS["freq"]
    if policy_name == "ecoserve":
        return EcoPlanner(planner, router), None, False, ECO_PERIOD_S
    return None
