#!/usr/bin/env python3
"""CPU-only, reproducible figure arithmetic from PDblend's synthetic test profile.

Run from any directory with /home/pdblend/.venv/bin/python /absolute/path/to/this/file.
This is an explanatory example, not a measured GPU benchmark.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/pdblend")]

from synthetic import synthetic_model  # noqa: E402
from pdblend.control.forecast import Forecast  # noqa: E402
from pdblend.control.planner import PoolPlanner, PlannerConfig, SLO  # noqa: E402


def demand(output_tokens: int) -> Forecast:
    return Forecast(
        rate_rps=12.0, trend_rps=0.0, input_mean=1728.0,
        input_p95=4096.0, output_mean=float(output_tokens), inflight=0,
        inputs=(256, 512, 2048, 4096) * 16, outputs=(output_tokens,) * 64,
    )


def planner(min_m: int) -> PoolPlanner:
    return PoolPlanner(synthetic_model(), PlannerConfig(
        slots=8, slo=SLO(ttft_s=1.0, tpot_s=0.020, safety=0.85),
        dwell_s=60.0, margin=0.05, min_m_instances=min_m,
        allow_park=("L1",),
    ))


SPECS = {
    "A": ({"M": 5, "L1": 3}, 2520, 2520, 2520, 0),
    "B": ({"M": 4, "L1": 4}, 2520, 2520, 2520, 0),
    "C": ({"P": 2, "D": 1, "M": 2, "L1": 3}, 2100, 2520, 2520, 1024),
    "D": ({"P": 2, "D": 2, "M": 2, "L1": 2}, 2100, 2520, 2520, 1024),
    "E_tau4096": ({"P": 2, "D": 1, "M": 2, "L1": 3}, 2100, 2520, 2520, 4096),
}


def same_layout(a, b) -> bool:
    def nonzero(c):
        return {k: v for k, v in c.items() if v}
    return (nonzero(a.counts), a.f_P, a.f_D, a.f_M, a.tau) == (
        nonzero(b.counts), b.f_P, b.f_D, b.f_M, b.tau)


def evaluated(p, fc):
    rows = {}
    for label, spec in SPECS.items():
        relaxed = p.evaluate(*spec, fc, strict=False)
        strict = p.evaluate(*spec, fc, strict=True)
        assert relaxed is not None
        violations = []
        if relaxed.ttft_s > p.cfg.slo.ttft_s * p.cfg.slo.safety:
            violations.append("TTFT > 850 ms")
        if relaxed.tpot_s > p.cfg.slo.tpot_s * p.cfg.slo.safety:
            violations.append("TPOT > 17 ms")
        if relaxed.detail.get("M", {}).get("tpot_miss", 0) > 1 - p.cfg.tail_target:
            violations.append("mixed TPOT miss fraction > 10%")
        rows[label] = {
            "feasible": strict is not None, "violations": violations,
            "plan": asdict(relaxed),
        }
    return rows


def transition(p, old, new):
    switch = p.switch_energy_j(old, new)
    old_energy = p.cfg.dwell_s * old.power_w
    new_energy = p.cfg.dwell_s * new.power_w + switch
    return {
        "horizon_s": p.cfg.dwell_s,
        "switch_j": switch,
        "old_horizon_j": old_energy,
        "new_horizon_including_switch_j": new_energy,
        "saving_j": old_energy - new_energy,
        "gain_fraction": (old_energy - new_energy) / old_energy,
        "saving_threshold_j": p.cfg.margin * old_energy,
        "energy_hysteresis_passed": (old_energy - new_energy) > p.cfg.margin * old_energy,
    }


def main():
    p = planner(2)
    initial, changed = demand(128), demand(256)
    initial_rows, changed_rows = evaluated(p, initial), evaluated(p, changed)
    a = p.evaluate(*SPECS["A"], initial)
    c = p.evaluate(*SPECS["C"], initial)
    d = p.evaluate(*SPECS["D"], changed)
    c_under_changed = p.evaluate(*SPECS["C"], changed, strict=False)
    assert a is not None and c is not None and d is not None and c_under_changed is not None

    initial_candidates, changed_candidates = p.candidates(initial), p.candidates(changed)
    chosen_initial = p.plan(initial, current=a)
    chosen_changed = p.plan(changed, current=c)
    assert same_layout(initial_candidates[0], c)
    assert same_layout(changed_candidates[0], d)
    assert same_layout(chosen_initial, c)
    assert same_layout(chosen_changed, d)
    assert p.evaluate(*SPECS["C"], changed) is None
    assert initial_rows["A"]["feasible"] and initial_rows["C"]["feasible"]
    assert not initial_rows["B"]["feasible"] and not initial_rows["E_tau4096"]["feasible"]

    first_transition = transition(p, a, c)
    second_transition = transition(p, c_under_changed, d)
    assert abs(first_transition["switch_j"] - 34.0) < 1e-9
    assert abs(second_transition["switch_j"] - 3.4) < 1e-9
    assert first_transition["energy_hysteresis_passed"]
    assert not second_transition["energy_hysteresis_passed"]

    # The existing implementation has no separately fitted incremental KV-transfer
    # energy term. This bound is an algebraic sensitivity limit, not a measurement.
    extra_kv_limit_j = first_transition["saving_j"] - first_transition["saving_threshold_j"]
    pd_count_in_horizon = initial.rate_rps * initial.split(1024)[0] * p.cfg.dwell_s
    first_transition["unmodeled_extra_energy_strict_upper_bound_j"] = extra_kv_limit_j
    first_transition["unmodeled_extra_energy_strict_upper_bound_average_w"] = extra_kv_limit_j / p.cfg.dwell_s
    first_transition["unmodeled_extra_energy_strict_upper_bound_per_pd_request_j"] = extra_kv_limit_j / pd_count_in_horizon
    second_transition["decision_reason"] = "Current layout violates TPOT safety target; planner returns the new best feasible candidate before applying energy hysteresis."

    unrestricted = planner(0)
    audit = {
        "min_m_instances": 0,
        "initial_global_best": asdict(unrestricted.plan(initial)),
        "changed_global_best": asdict(unrestricted.plan(changed)),
        "purpose": "Appendix only: demonstrates why min_M=2 is an explicit figure configuration, not an unqualified global optimum claim.",
    }
    hashes = {}
    for rel in ("src/pdblend/control/planner.py", "src/pdblend/control/forecast.py", "src/pdblend/profile/model.py", "tests/pdblend/synthetic.py"):
        hashes[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()

    model = p.model
    data = {
        "label": "CPU synthetic worked example; not measured hardware performance",
        "config": asdict(p.cfg),
        "figure_constraints": "8 one-GPU TP1/PP1 instances; min_M=2 explicitly chosen for this example; parked state L1 only; pressure_controls=False.",
        "initial_workload": asdict(initial),
        "changed_workload": asdict(changed),
        "split_tau1024": {"pd_fraction": 0.5, "pd_rps": 6.0, "mixed_rps": 6.0, "mean_pd_input_tokens": 3072.0, "mean_mixed_input_tokens": 384.0},
        "split_tau4096": {"pd_fraction": 0.25, "pd_rps": 3.0, "mixed_rps": 9.0, "mean_pd_input_tokens": 4096.0, "mean_mixed_input_tokens": (256+512+2048)/3},
        "initial_candidates_displayed": initial_rows,
        "changed_candidates_displayed": changed_rows,
        "full_enumeration": {
            "initial_feasible_count": len(initial_candidates),
            "changed_feasible_count": len(changed_candidates),
            "initial_chosen_with_current_A": asdict(chosen_initial),
            "changed_chosen_with_current_C": asdict(chosen_changed),
        },
        "transition_A_to_C": first_transition,
        "transition_C_to_D_after_output_doubles": second_transition,
        "transfer_model_examples": {
            str(tokens): {"kv_bytes": tokens * model.kv_bytes_per_token, "transfer_ms": 1000 * model.transfer_seconds(tokens)}
            for tokens in (2048, 3072, 4096)
        },
        "switch_model": {
            "active_idle_power_at_2520_mhz_w": model.static_power_w("active_idle", 2520),
            "clock_switch_s": model.freq_switch_s,
            "L1_wake_s": model.wake_seconds("parked"),
            "A_to_C": "No wake; frequency tuple changes: 0.1 s × 68 W × 5 active GPUs = 34 J. The implementation charges all new active instances for any pool-frequency tuple change.",
            "C_to_D": "One L1 instance wakes; no frequency tuple change: 0.05 s × 68 W = 3.4 J.",
        },
        "energy_model_scope": "Current planner uses H*P + switch energy and strict gain > margin for a feasible current layout. It does not separately calibrate/charge unknown incremental KV-transfer energy. The extra-energy limit is a sensitivity bound, not a measured zero cost.",
        "appendix_min_m_zero": audit,
        "source_sha256": hashes,
        "assertions_passed": True,
    }
    (HERE / "example.json").write_text(json.dumps(data, indent=2) + "\n")

    lines = [
        "# Planner figure: CPU synthetic worked example", "",
        "This is a calculation from the repository synthetic profile, not a GPU measurement.",
        "8 GPUs, TP=PP=1; min_M=2 (explicit figure configuration); parked=L1; H=60 s; margin=5%.",
        "Arrival rate 12 req/s; input lengths 256/512/2048/4096 equally likely (mean 1728, p95 4096); output 128 initially, then 256.",
        "SLO: TTFT 1 s, TPOT 20 ms; safety 0.85 -> feasibility thresholds 850 ms / 17 ms.", "",
        "| Output | Candidate | P/D/M/L1 | tau | Power W | TTFT ms | TPOT ms | M miss % | Feasible |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for output, rows in ((128, initial_rows), (256, changed_rows)):
        for name, row in rows.items():
            v = row["plan"]
            counts = v["counts"]
            role_text = "/".join(str(counts.get(r, 0)) for r in ("P", "D", "M", "L1"))
            lines.append(f"| {output} | {name} | {role_text} | {v['tau']} | {v['power_w']:.6f} | {1000*v['ttft_s']:.6f} | {1000*v['tpot_s']:.6f} | {100*v['detail'].get('M',{}).get('tpot_miss',0):.6f} | {row['feasible']} |")
    lines += ["", "Full enumeration chooses C initially and D after output length doubles.",
        f"Initial feasible candidates: {len(initial_candidates)}; changed feasible candidates: {len(changed_candidates)}.", "",
        "## Initial A -> C", "",
        f"Esw=34 J; E_A={first_transition['old_horizon_j']:.6f} J; E_C including switch={first_transition['new_horizon_including_switch_j']:.6f} J.",
        f"Saving={first_transition['saving_j']:.6f} J; gain={100*first_transition['gain_fraction']:.6f}% > 5%: switch.",
        f"Additional unmodeled energy must be strictly below {extra_kv_limit_j:.6f} J/60s ({extra_kv_limit_j/60:.6f} W average; {extra_kv_limit_j/pd_count_in_horizon:.6f} J per PD request) for gain to remain strictly >5%.",
        "This is an algebraic sensitivity bound. No separately measured incremental KV-energy term exists in this synthetic planner score.", "",
        "## C -> D after output doubles", "",
        f"Re-evaluated C: {c_under_changed.power_w:.6f} W, {c_under_changed.tpot_s*1000:.6f} ms TPOT >17 ms target (also >20 ms SLO).",
        f"D: {d.power_w:.6f} W, {d.tpot_s*1000:.6f} ms TPOT; Esw=3.4 J.",
        f"Current C horizon={second_transition['old_horizon_j']:.6f} J; D with switch={second_transition['new_horizon_including_switch_j']:.6f} J; gain={100*second_transition['gain_fraction']:.6f}%.",
        "C is infeasible, so planner changes to D before checking the energy-saving hysteresis. Extra D capacity is required even though power rises.", "",
        "## Appendix: min_M=0 audit", "",
        "Removing the M-floor changes the full optimum: initially pure P2/D2/M0/L1×4 at 663.121615 W; after output doubles pure P2/D3/M0/L1×3 at 777.544566 W. Thus the min_M=2 constraint must be visible in the figure.", "",
        "All assertions passed. See example.json for exact intermediate pool power, utilization, batch, transfer arithmetic, and source hashes.",
    ]
    (HERE / "AUDIT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
