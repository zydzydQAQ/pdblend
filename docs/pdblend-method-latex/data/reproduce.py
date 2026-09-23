#!/usr/bin/env python3
"""Reproduce the method figures with the repository's synthetic CPU model.

Run with a Python environment containing NumPy, for example:
    /home/pdblend/.venv/bin/python data/reproduce.py
    /home/pdblend/.venv/bin/python data/reproduce.py --repo /path/to/pdblend

The script discovers the repository among its parent directories, imports only
the model/planner/forecast and synthetic fixture, and writes examples.json next
to this script. It does not launch engines, inspect GPUs, or run a GPU workload.
All candidate comparisons are illustrative, not an exhaustive optimum search.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import sys


# Reproduction should not write bytecode into the source repository.
sys.dont_write_bytecode = True

SOURCE_PATHS = (
    "tests/pdblend/synthetic.py",
    "src/pdblend/profile/model.py",
    "src/pdblend/control/forecast.py",
    "src/pdblend/control/planner.py",
)


def find_repo(explicit: Path | None) -> Path:
    candidates = [explicit.resolve()] if explicit else Path(__file__).resolve().parents
    for root in candidates:
        if all((root / relative).is_file() for relative in SOURCE_PATHS):
            return root
    raise SystemExit("Cannot find the PDblend source repository; pass --repo /path/to/pdblend.")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded(value: float, digits: int) -> str:
    """Use the figure's fixed decimal precision, preserving trailing zeros."""
    return f"{value:.{digits}f}"


def reproduce(repo: Path) -> dict:
    sys.path[:0] = [str(repo / "src"), str(repo / "tests/pdblend")]
    from synthetic import synthetic_model
    from pdblend.control.forecast import Forecast
    from pdblend.control.planner import PlannerConfig, PoolPlanner, SLO

    assert Path(inspect.getfile(synthetic_model)).resolve() == (repo / SOURCE_PATHS[0]).resolve()
    model = synthetic_model()
    forecast = Forecast(12, 0, 1728, 4096, 128, 0,
                        (256, 512, 2048, 4096), (128,) * 4)
    horizon = 60.0
    margin = 0.08
    slo = SLO(1.0, 0.02, safety=0.85)
    config = PlannerConfig(8, slo, min_m_instances=0, dwell_s=horizon, margin=margin)
    planner = PoolPlanner(model, config)

    # Named candidates are a displayed subset; no claim of a global optimum.
    specs = {
        "A": ({"M": 5, "off": 3}, 2520, 2520, 2520, 0),
        "B": ({"M": 4, "off": 4}, 2520, 2520, 2520, 0),
        "C": ({"P": 2, "D": 1, "M": 2, "off": 3}, 2100, 2520, 2520, 1024),
    }
    plans = {name: planner.evaluate(*spec, forecast, strict=False)
             for name, spec in specs.items()}
    assert all(plan is not None for plan in plans.values()), "A displayed layout became unstable."
    current = plans["A"]
    current_energy = current.power_w * horizon
    guard_ttft = slo.ttft_s * slo.safety
    guard_tpot = slo.tpot_s * slo.safety

    candidates = {}
    for name, spec in specs.items():
        plan = plans[name]
        checked = planner.evaluate(*spec, forecast, strict=True)
        components = {role: detail["power_w"]
                      for role, detail in plan.detail.items() if "power_w" in detail}
        components["off"] = plan.counts.get("off", 0) * model.static_power_w("off")
        assert math.isclose(sum(components.values()), plan.power_w, abs_tol=1e-9)
        switch = planner.switch_energy_j(current, plan)
        operating_energy = plan.power_w * horizon
        lower_bound = operating_energy + switch
        reasons = []
        if plan.ttft_s > guard_ttft:
            reasons.append("latency_proxy_above_guard")
        if plan.tpot_s > guard_tpot:
            reasons.append("tpot_above_guard")
        if plan.detail.get("M", {}).get("tpot_miss", 0.0) > 1 - config.tail_target:
            reasons.append("mixed_tpot_miss_probability_above_limit")
        candidates[name] = {
            "counts": plan.counts,
            "frequencies_mhz": {"P": plan.f_P, "D": plan.f_D, "M": plan.f_M},
            "threshold_tokens": plan.tau,
            "power_w": plan.power_w,
            "power_components_w": components,
            "latency_proxy_s": plan.ttft_s,
            "latency_proxy_ms": 1000 * plan.ttft_s,
            "tpot_s": plan.tpot_s,
            "tpot_ms": 1000 * plan.tpot_s,
            "strict_model_feasible": checked is not None,
            "rejection_reasons": reasons,
            "operating_and_standby_energy_j": operating_energy,
            "switch_energy_j_from_A": switch,
            "window_energy_lower_bound_j": lower_bound,
            "normalized_energy_lower_bound_A100": 100 * lower_bound / current_energy,
            "unmodeled_extra_handoff_energy_j": None if name == "C" else 0.0,
            "pool_details": plan.detail,
        }

    share, long_mean, short_mean = forecast.split(1024)
    pd_requests = horizon * forecast.rate_rps * share
    best_case_gain = current_energy - candidates["C"]["window_energy_lower_bound_j"]
    required_gain = margin * current_energy
    comparison = {
        "scope": "C versus A only; other candidates are not decided",
        "horizon_s": horizon,
        "margin_fraction": margin,
        "normalization_A": 100.0,
        "accept_C_only_below_normalized_cost": 100 * (1 - margin),
        "required_gain_j": required_gain,
        "best_case_C_gain_j": best_case_gain,
        "best_case_C_gain_fraction": best_case_gain / current_energy,
        "C_extra_handoff_energy_lower_bound_j": 0.0,
        "C_extra_handoff_energy_formula": "360 * e joules, where e >= 0 J per PD request is unknown",
        "pairwise_decision": "keep_A",
        "reason": "Even zero extra handoff energy leaves the saving below the 8% margin.",
    }

    input_tokens, output_tokens, batch = 2048, 128, 16
    context = input_tokens + output_tokens / 2
    prefill_grid = [
        {"frequency_mhz": frequency, "input_tokens": length,
         "time_ms": 1000 * model.prefill_seconds(length, frequency),
         "power_w": model.prefill_power_w(length, frequency)}
        for frequency in model.freqs for length in (512, 2048, 4096)
    ]
    decode_grid = [
        {"frequency_mhz": frequency, "batch": batch, "context_tokens": context,
         "step_ms": 1000 * model.step_seconds(batch, context, frequency),
         "power_w": model.decode_power_w(batch, frequency, ctx=context),
         "energy_per_token_j": model.token_energy_j(batch, context, frequency)}
        for frequency in model.freqs
    ]
    query = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "assumed_decode_batch": batch,
        "representative_context_tokens": context,
        "frequency_mhz": 2520,
        "prefill_ms": 1000 * model.prefill_seconds(input_tokens, 2520),
        "prefill_power_w": model.prefill_power_w(input_tokens, 2520),
        "decode_step_ms": 1000 * model.step_seconds(batch, context, 2520),
        "decode_power_w": model.decode_power_w(batch, 2520, ctx=context),
        "decode_energy_per_token_j": model.token_energy_j(batch, context, 2520),
        "handoff_estimate_ms": 1000 * model.transfer_seconds(input_tokens),
        "handoff_is_physical_copy_measurement": False,
        "prefill_grid": prefill_grid,
        "decode_query_by_frequency": decode_grid,
    }

    # Exact anchors and the precision displayed in the two method figures.
    for key, expected in {
        "prefill_ms": 96.114304,
        "decode_step_ms": 16.26496,
        "decode_power_w": 127.0,
        "handoff_estimate_ms": 28.4881024,
    }.items():
        assert math.isclose(query[key], expected, rel_tol=0, abs_tol=1e-10), (key, query[key])
    expected_rows = {
        "A": ("769.14", "214.19", "16.03", True),
        "B": ("709.55", "218.07", "17.50", False),
        "C": ("708.28", "319.17", "15.54", True),
    }
    display = {}
    for name, expected in expected_rows.items():
        row = candidates[name]
        observed = (rounded(row["power_w"], 2), rounded(row["latency_proxy_ms"], 2),
                    rounded(row["tpot_ms"], 2), row["strict_model_feasible"])
        assert observed == expected, (name, observed, expected)
        display[name] = dict(zip(("power_w", "latency_proxy_ms", "tpot_ms", "feasible"), observed))
    assert candidates["B"]["rejection_reasons"] == ["tpot_above_guard"]
    assert candidates["C"]["switch_energy_j_from_A"] == 34.0
    assert (share, long_mean, short_mean, pd_requests) == (0.5, 3072.0, 384.0, 360.0)
    assert best_case_gain < required_gain
    energy_display = {
        "A_energy_kj": rounded(current_energy / 1000, 3),
        "C_operating_energy_kj": rounded(candidates["C"]["operating_and_standby_energy_j"] / 1000, 3),
        "C_switch_kj": rounded(candidates["C"]["switch_energy_j_from_A"] / 1000, 3),
        "C_minimum_normalized_cost": rounded(candidates["C"]["normalized_energy_lower_bound_A100"], 2),
        "required_gain_kj": rounded(required_gain / 1000, 3),
        "best_case_gain_kj": rounded(best_case_gain / 1000, 3),
    }
    assert energy_display == {
        "A_energy_kj": "46.148", "C_operating_energy_kj": "42.497",
        "C_switch_kj": "0.034", "C_minimum_normalized_cost": "92.16",
        "required_gain_kj": "3.692", "best_case_gain_kj": "3.617",
    }
    decode_by_freq = {row["frequency_mhz"]: row for row in decode_grid}
    assert rounded(decode_by_freq[900]["step_ms"], 3) == "45.542"
    assert rounded(decode_by_freq[900]["power_w"], 3) == "106.429"
    assert rounded(decode_by_freq[900]["energy_per_token_j"], 5) == "0.30293"
    assert rounded(query["decode_energy_per_token_j"], 5) == "0.12910"
    assert [rounded(row["time_ms"], 3) for row in prefill_grid if row["frequency_mhz"] == 2520] == [
        "30.742", "96.114", "190.617"]
    gpu_frameworks = sorted(set(sys.modules) & {"torch", "vllm", "pynvml", "cupy"})
    assert not gpu_frameworks, f"Unexpected GPU-framework imports: {gpu_frameworks}"

    return {
        "schema_version": 1,
        "status": "synthetic CPU example; not a GPU measurement or full-space optimum",
        "python_version": platform.python_version(),
        "gpu_frameworks_imported": gpu_frameworks,
        "source_sha256": {relative: sha256(repo / relative) for relative in SOURCE_PATHS},
        "reproduction_script_sha256": sha256(Path(__file__).resolve()),
        "planning_inputs": {
            "forecast": asdict(forecast), "gpu_count": 8, "tp": 1, "pp": 1,
            "horizon_s": horizon, "slo": asdict(slo),
            "guarded_latency_proxy_ms": guard_ttft * 1000,
            "guarded_tpot_ms": guard_tpot * 1000,
            "min_m_instances": config.min_m_instances,
            "fixed_M_floor_disabled_for_target_example": True,
        },
        "traffic_split": {
            "threshold_tokens": 1024, "PD_share": share,
            "PD_rate_rps": forecast.rate_rps * share,
            "M_rate_rps": forecast.rate_rps * (1 - share),
            "PD_mean_input_tokens": long_mean, "M_mean_input_tokens": short_mean,
            "PD_requests_per_horizon": pd_requests,
        },
        "candidates": candidates,
        "pairwise_comparison": comparison,
        "model_query": query,
        "figure_display_values": {"candidate_rows": display, "energy": energy_display},
        "interpretation": [
            "The latency proxy includes PD handoff and a first D step; it is not measured carry-protocol TTFT.",
            "Synthetic formulas do not establish measured profile coverage or hardware qualification.",
            "Candidate C uses M2 and is outside the default policy's fixed M-floor enumeration.",
            "C's unknown incremental handoff energy remains symbolic and nonnegative, not assumed measured zero.",
            "The lower bound alone rejects replacing A by C at an 8% margin; other candidates are not decided.",
        ],
        "assertions_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="PDblend repository root; otherwise search script parents")
    args = parser.parse_args()
    result = reproduce(find_repo(args.repo))
    output = Path(__file__).resolve().with_name("examples.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}; all numerical and display assertions passed.")
    for name, row in result["figure_display_values"]["candidate_rows"].items():
        print(f"{name}: {row['power_w']} W, proxy {row['latency_proxy_ms']} ms, "
              f"TPOT {row['tpot_ms']} ms, feasible={row['feasible']}")
    print("Pairwise result: C cannot replace A at the 8% margin, even with zero extra handoff energy.")


if __name__ == "__main__":
    main()
