"""G5: replay the ported baseline decisions against the frozen pdblend_baselines code on identical CPU state.

Three decision points are shared between the ports in control/policies/baselines.py and the frozen package:
EcoServe admission (OfficialMacro._check_constraints vs EcoRouter._fits), EcoServe scaling
(EcoServeController.scale_once vs EcoPlanner.plan) and DynamoLLM ScaleFreq (DynamoPolicy.frequency vs the
minimum-energy feasible clock on the same performance estimates). DistServe placement is structurally different
(upstream replicates fixed 1P+1D units and searches parallelism; the port searches the P:D split) and is
reported as a capacity comparison, not a decision match.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

from ..control.forecast import Forecast
from ..control.planner import SLO, PlannerConfig, PoolPlanner
from ..control.policies.baselines import DynamoPlanner, EcoPlanner, EcoRouter, capacity_rps, distserve_plan
from ..profile.model import PerfModel
from ..proxy.router import RequestRecord


class LegacyPrefill:
    """PerfModel view with the integer-ms prefill lookup upstream EcoServe uses (PrefillProfile.predict_ms)."""

    def __init__(self, model: PerfModel):
        self.model, self.freqs = model, model.freqs

    def predict_ms(self, n: int) -> int:
        return int(self.model.prefill_seconds(n, max(self.freqs)) * 1000.0)

    def prefill_seconds(self, n: int, f: int) -> float:
        return self.predict_ms(n) / 1000.0


def _eco_state(rng: random.Random, n_res: int, now: float, ttft_s: float, tpot_s: float, allow_pending: bool):
    rows = []
    for k in range(n_res):
        pending = allow_pending and rng.random() < 0.3
        n = rng.choice((64, 256, 512, 1024, 2048))
        arrival = now - rng.uniform(0.05, 20.0)
        if pending:
            rows.append(dict(rid=f"r{k}", n=n, arrival=arrival, ttft=None, iters=0))
        else:
            ttft = min(rng.uniform(0.05, 1.5) * ttft_s, now - arrival)
            max_iters = int((now - arrival - ttft) / max(tpot_s * 0.5, 1e-3))
            rows.append(dict(rid=f"r{k}", n=n, arrival=arrival, ttft=ttft, iters=rng.randint(1, max(1, max_iters))))
    return rows


def _legacy_requests(rows, ttft_ms: float, predict_ms):
    from pdblend_baselines.ecoserve.policy import RequestState
    out = []
    for r in rows:
        ttft = ttft_ms if r["ttft"] is None else r["ttft"] * 1000.0
        out.append(RequestState(r["rid"], r["arrival"] * 1000.0, r["iters"], ttft, predict_ms(r["n"]), -1,
                                (r["n"] + 16) // 16))
    return out


def _new_records(rows, iid: str):
    out = []
    for r in rows:
        rec = RequestRecord(r["rid"], "M", iid, iid, r["n"], 256, r["arrival"])
        if r["ttft"] is not None:
            rec.first_token_s = r["arrival"] + r["ttft"]
            rec.tokens_so_far = r["iters"]
        out.append(rec)
    return out


def eco_admission(model: PerfModel, slo: SLO, trials: int, seed: int) -> dict:
    from pdblend_baselines.ecoserve.policy import OfficialMacro
    rng = random.Random(seed)
    ids = ["i0", "i1", "i2"]
    legacy_model = LegacyPrefill(model)
    ttft_ms, tpot_ms = slo.ttft_s * 1000.0, slo.tpot_s * 1000.0
    agree, rows = 0, []
    for t in range(trials):
        now = 1_000_000.0 + t
        since = now - rng.uniform(0.0, 2.0)
        cur = _eco_state(rng, rng.randint(0, 6), now, slo.ttft_s, slo.tpot_s, True)
        nxt = _eco_state(rng, rng.randint(0, 6), now, slo.ttft_s, slo.tpot_s, False)
        n_new = rng.choice((32, 128, 512, 1024, 2048, 4096))

        macro = OfficialMacro(ids, legacy_model, ttft_ms, tpot_ms, now_ms=lambda: now * 1000.0)
        st0, st1 = macro.instance_states[0], macro.instance_states[1]
        st0.requests.extend(_legacy_requests(cur, ttft_ms, legacy_model.predict_ms))
        st0.waiting_queue = [r["rid"] for r in cur if r["ttft"] is None]
        st0.schedule_time, st0.free_blocks = since * 1000.0, 10**9
        st1.requests.extend(_legacy_requests(nxt, ttft_ms, legacy_model.predict_ms))
        st1.free_blocks = 10**9
        legacy = macro._check_constraints((n_new + 16) // 16, legacy_model.predict_ms(n_new))

        router = EcoRouter(ids, legacy_model, slo)
        router.set_roles({i: "M" for i in ids})
        router.groups, router.cursor = [ids], {0: 0}
        router.since["i0"] = since
        router.active["i0"] = _new_records(cur, "i0")
        router.active["i1"] = _new_records(nxt, "i1")
        new = router._fits(ids, "i0", n_new, now)

        agree += legacy == new
        if legacy != new and len(rows) < 20:
            rows.append(dict(trial=t, legacy=legacy, new=new, n_new=n_new, current=cur, next=nxt))
    return dict(trials=trials, agree=agree, agreement=agree / trials, mismatches=rows)


class _Transport:
    def __init__(self, ids):
        self.states = {i: dict(id=i, generation=0, acknowledged_generation=0, role="mixed", mode="continuous",
                               admit_prefill=True, admit_decode=True, running=0, waiting=0, active=0,
                               kv_allocations={}, transfer_allocations={}, free_kv_tokens=16000,
                               total_kv_tokens=16000, accepting=True, timestamp=time.time()) for i in ids}

    async def state(self, i):
        return dict(self.states[i], timestamp=time.time())

    async def json(self, i, path, payload=None):
        self.states[i].update(payload, acknowledged_generation=payload["generation"])
        return self.states[i].copy()

    async def clock(self, gpus, frequency):
        return dict(acknowledged=True, frequency_mhz=frequency)

    async def park(self, gpus):
        return dict(acknowledged=True, operation="reset_locked_clocks")

    async def events(self, i, after_seq=0):
        return dict(events=[], next_seq=0)

    async def cancel(self, i, request_id):
        return None


async def _legacy_scale(tmp: Path, N: int, n: int, ttfts, per_instance, slo: SLO, model: PerfModel) -> int:
    from pdblend_baselines.ecoserve import build_controller
    csv = tmp / "prefill.csv"
    lp = LegacyPrefill(model)
    csv.write_text(f"Length,Prefill Time\n16,{lp.predict_ms(16)}\n4096,{lp.predict_ms(4096)}\n")
    ids = [str(i) for i in range(N)]
    cfg = dict(instances=[dict(id=i, gpus=[int(i)], tp=1, url=f"http://127.0.0.1:{9000 + int(i)}", role="mixed") for i in ids],
               eco_prefill_csv=str(csv), slo_ttft_s=slo.ttft_s, slo_tpot_s=slo.tpot_s, eco_scale_period_s=60,
               eco_macro_lower=2, eco_macro_upper=3, eco_initial_instances=n, eco_state_poll_s=0.002)
    controller = build_controller(cfg, _Transport(ids), lambda *a, **k: None)
    await controller.startup()
    try:
        now = time.time()
        controller.ttft_history.extend((now - 1.0, v) for v in ttfts)
        for i, rows in per_instance.items():
            controller.members[i].requests.extend(_legacy_requests(rows, slo.ttft_s * 1000.0, lp.predict_ms))
        await controller.scale_once()
        return len(controller.assigned)
    finally:
        await controller.close()


def eco_scaling(model: PerfModel, slo: SLO, trials: int, seed: int, tmp: Path) -> dict:
    rng = random.Random(seed)
    N = 4
    agree, rows = 0, []
    for t in range(trials):
        n = rng.randint(2, N)
        ttfts = [rng.uniform(0.2, 1.8) * slo.ttft_s for _ in range(rng.randint(0, 6))]
        now = time.time()
        per_instance = {str(i): _eco_state(rng, rng.randint(0, 5), now, slo.ttft_s, slo.tpot_s, False) for i in range(n)}
        legacy = asyncio.run(_legacy_scale(tmp, N, n, ttfts, per_instance, slo, model))

        ids = [str(i) for i in range(N)]
        router = EcoRouter(ids, model, slo)
        router.set_roles({i: ("M" if int(i) < n else "parked") for i in ids})
        router._regroup(ids[:n])
        for i, rws in per_instance.items():
            router.active[i] = _new_records(rws, i)
        for v in ttfts:
            rec = RequestRecord(f"h{v}", "M", "0", "0", 128, 64, now - 1.0 - v)
            rec.first_token_s = now - 1.0
            rec.finished_s = now - 0.5
            router.records.append(rec)
        planner = PoolPlanner(model, PlannerConfig(slots=N, slo=slo, freqs=model.freqs))
        eco = EcoPlanner(planner, router)
        from ..control.planner import Plan
        current = Plan({"M": n, **({"idle": N - n} if n < N else {})}, 2520, 2520, 2520, 0, 0.0, 0.0, 0.0, {})
        new = eco.plan(Forecast(1.0, 0.0, 512, 1024, 128, 0, (512,) * 10), current).counts["M"]

        agree += legacy == new
        if legacy != new and len(rows) < 20:
            rows.append(dict(trial=t, n=n, legacy=legacy, new=new, ttfts=ttfts, per_instance=per_instance))
    return dict(trials=trials, agree=agree, agreement=agree / trials, mismatches=rows)


def _paper_profiles(model: PerfModel):
    from pdblend_baselines.dynamollm.profiles import PaperProfiles
    inputs, ctxs, batches = (32, 128, 512, 1024, 2048, 4096, 8192), (64, 256, 1024, 2048, 4096, 8192, 16384), (1, 4, 16, 32, 64)
    rows = []
    for f in model.freqs:
        for n in inputs:
            for c in ctxs:
                for b in batches:
                    rows.append(dict(role="mixed", tp=1, frequency_mhz=f, input_tokens=n, context_tokens=c, batch=b,
                                     prefill_s=model.prefill_seconds(n, f), iteration_s=model.step_seconds(b, c, f),
                                     prefill_power_w=model.prefill_power_w(n, f), decode_power_w=model.decode_power_w(b, f),
                                     samples=1, source_sha256="perfmodel"))
    return PaperProfiles(rows)


def dynamo_freq(model: PerfModel, slo: SLO, trials: int, seed: int) -> dict:
    from pdblend_baselines.dynamollm.policy import DynamoPolicy, Replica, Request
    rng = random.Random(seed)
    policy = DynamoPolicy(_paper_profiles(model))
    agree, feasible_agree, rows = 0, 0, []
    for t in range(trials):
        B = rng.choice((1, 2, 4, 8, 16, 24, 32, 48, 64))
        n = rng.choice((64, 128, 256, 512, 1024, 2048, 4096))
        o = rng.choice((32, 64, 128, 256, 512))
        emitted = rng.randint(1, max(1, o - 1))
        now = 1_000_000.0
        reqs = [Request(f"r{k}", n, o, now - 5.0, slo.ttft_s, slo.tpot_s, emitted=emitted,
                        first_token_s=now - 1.0, last_token_s=now, started=True) for k in range(B)]
        replica = Replica("i0", (0,), 1, "MM", max(model.freqs), requests=reqs)
        legacy = policy.frequency(replica, now)

        ctx = n + max(o, emitted + 1)
        remaining = max(1, o - emitted)
        feasible = [f for f in model.freqs if model.step_seconds(B, ctx, f) <= slo.tpot_s]
        legacy_feasible = [f for f in model.freqs if policy.feasible(replica, reqs, now, f) is not None]
        feasible_agree += feasible == legacy_feasible
        if feasible:
            new = min(feasible, key=lambda f: (remaining * model.step_seconds(B, ctx, f) * model.decode_power_w(B, f), f))
        else:
            new = max(model.freqs)
        agree += legacy == new
        if legacy != new and len(rows) < 20:
            rows.append(dict(trial=t, B=B, input=n, output=o, emitted=emitted, legacy=legacy, new=new,
                             feasible=feasible, legacy_feasible=legacy_feasible))
    return dict(trials=trials, agree=agree, agreement=agree / trials, feasible_set_agreement=feasible_agree / trials,
                mismatches=rows)


def distserve_capacity(model: PerfModel, slo: SLO, slots: int, fc: Forecast) -> dict:
    planner = PoolPlanner(model, PlannerConfig(slots=slots, slo=slo, freqs=model.freqs))
    f = max(model.freqs)
    port = distserve_plan(planner, fc)
    upstream = {"P": slots // 2, "D": slots // 2}
    return dict(slots=slots, port_counts=port.counts, port_capacity_rps=port.detail.get("capacity_rps"),
                upstream_unit_counts=upstream, upstream_unit_capacity_rps=capacity_rps(planner, fc, upstream, f, f, f, 0),
                note="upstream DistServe replicates 1P+1D units (parallelism search only); the port searches the P:D "
                     "split, so its capacity is >= the upstream layout by construction")


def gate_g5(profile: Path, out: Path, trials: int = 200, seed: int = 7, dataset: str = "sharegpt") -> dict:
    from . import client as bc
    model = PerfModel.load(profile)
    slo = SLO(*bc.SLOS[dataset])
    tmp = out.parent / "g5-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    fc = Forecast(4.0, 0.0, 512.0, 1500.0, 200.0, 0, (512,) * 20)
    result = dict(
        gate="G5", profile=str(profile), dataset=dataset, slo=dict(ttft_s=slo.ttft_s, tpot_s=slo.tpot_s), trials=trials,
        eco_admission=eco_admission(model, slo, trials, seed),
        eco_scaling=eco_scaling(model, slo, max(20, trials // 5), seed, tmp),
        dynamo_scale_freq=dynamo_freq(model, slo, trials, seed),
        distserve=[distserve_capacity(model, slo, n, fc) for n in (4, 8)],
        differences=[
            "EcoServe admission: the port has no free-block (KV) check; upstream rejects when need_blocks > free_blocks.",
            "EcoServe scaling: upstream removes the last member of the first group whose mean credit exceeds "
            "TTFT*(|g|+1)/|g|; the port shrinks the M count by one and lets the router regroup.",
            "DynamoLLM ScaleInst: upstream sizes 9 shape pools from a weekly forecast; the port sizes one mixed pool "
            "from the epoch's observed peak (oracle stand-in). ScaleFreq compares only the per-replica clock rule.",
            "DistServe: upstream searches parallelism with fixed 1P+1D units (with TP1/PP1 that is P=D=N/2); the port "
            "searches the P:D split at max clocks, a strictly stronger static baseline.",
        ])
    out.write_text(json.dumps(result, indent=1, default=str))
    return result
