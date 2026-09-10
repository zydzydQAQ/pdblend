"""Read-only source arithmetic; writes only this new analysis directory."""
import csv
import hashlib
import json
import math
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
N = OUT.parents[1]
BASELINES = ["mixed", "distserve", "dynamollm", "ecoserve"]
METRICS = ["n_expected", "good_requests", "slo_attainment", "energy_j",
           "energy_per_good_request_j", "goodput_measurement_rps", "measurement_duration_s"]


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def main():
    source = N / "reports/final-001/results.json"
    source_ref = ref(source)
    report = read(source)
    rows = report["observations"]
    assert len(rows) == 93
    b_reference_path = N / "B/reference.json"
    b_reference = read(b_reference_path)
    assert ref(Path(b_reference["source"]["path"])) == b_reference["source"]
    old_b = {row["cell_id"]: row for row in read(b_reference["source"]["path"])["observations"]}
    provenance_checks, observation_index = [], {}
    for row in rows:
        assert row["measurement_host"] in ("A", "B", "C")
        assert row["measurement_valid"] and row["strict_slo_recomputed"]
        assert row["energy_measured_gpu_count"] == 8 and row["energy_j"] > 0
        assert row["good_requests"] > 0 and row["measurement_duration_s"] > 0
        assert math.isclose(row["slo_attainment"], row["good_requests"] / row["n_expected"], abs_tol=1e-12)
        assert math.isclose(row["energy_per_good_request_j"], row["energy_j"] / row["good_requests"], rel_tol=1e-12)
        assert math.isclose(row["goodput_measurement_rps"], row["good_requests"] / row["measurement_duration_s"], rel_tol=1e-12)
        if row["measurement_host"] == "B":
            assert row["cell_id"] in b_reference["cell_ids"]
            original = old_b[row["cell_id"]]
            provenance = dict(kind="frozen_old_B_reference", source=b_reference["source"],
                new_measurement=False, full_qualification_context_replayed=False)
        else:
            reference = row["audit_reference"]
            assert ref(Path(reference["path"])) == reference
            original = read(reference["path"])
            provenance = dict(kind="new_A_C_frozen_audit", source=reference, new_measurement=True)
        for field in METRICS + ["trace_sha256", "system", "rate_rps", "repeat"]:
            assert row[field] == original[field], (row["cell_id"], field)
        provenance_checks.append(dict(cell_id=row["cell_id"], passed=True, **provenance))
        observation_index[row["cell_id"]] = dict(measurement_host=row["measurement_host"],
            reference_only=row["measurement_host"] == "B", system=row["system"],
            rate_rps=row["rate_rps"], repeat=row["repeat"], slo_scale=row["slo_scale"],
            trace_sha256=row["trace_sha256"], work_complete=row["work_complete"],
            metrics={field: row[field] for field in METRICS}, provenance=provenance)
    lookup = {(row["measurement_host"], row["system"], row["rate_rps"], row["repeat"]): row for row in rows}
    regular_pairs, confirmation_pairs = [], []
    for host, count in (("A", 4), ("B", 6), ("C", 8)):
        pdbs = sorted([r for r in rows if r["measurement_host"] == host and r["system"] == "pdblend"],
                      key=lambda r: (r["rate_rps"], r["repeat"]))
        assert sum(row["repeat"] == 1 for row in pdbs) == count
        assert sum(row["repeat"] == 2 for row in pdbs) == 1
        for pdb in pdbs:
            for system in BASELINES:
                baseline = lookup[host, system, pdb["rate_rps"], 1]
                assert pdb["trace_sha256"] == baseline["trace_sha256"]
                for key in ("n_expected", "slo_ttft_s", "slo_tpot_s", "arrival_window_s", "seed", "sampling_seed"):
                    assert pdb[key] == baseline[key], key
                pair = dict(measurement_host=host, reference_only=host == "B", slo_scale=pdb["slo_scale"],
                    rate_rps=pdb["rate_rps"], pdb_repeat=pdb["repeat"], baseline_repeat=1,
                    comparison_role="regular" if pdb["repeat"] == 1 else "boundary_confirmation_only",
                    baseline=system, pdb_cell_id=pdb["cell_id"], baseline_cell_id=baseline["cell_id"],
                    trace_sha256=pdb["trace_sha256"], n_expected=int(pdb["n_expected"]),
                    pdb_good_requests=int(pdb["good_requests"]), baseline_good_requests=int(baseline["good_requests"]),
                    pdb_slo_pct=100*pdb["slo_attainment"], baseline_slo_pct=100*baseline["slo_attainment"],
                    slo_delta_pp=100*(pdb["slo_attainment"]-baseline["slo_attainment"]),
                    pdb_energy_j=pdb["energy_j"], baseline_energy_j=baseline["energy_j"],
                    energy_reduction_pct=100*(1-pdb["energy_j"]/baseline["energy_j"]),
                    pdb_Jgood=pdb["energy_per_good_request_j"], baseline_Jgood=baseline["energy_per_good_request_j"],
                    Jgood_reduction_pct=100*(1-pdb["energy_per_good_request_j"]/baseline["energy_per_good_request_j"]),
                    pdb_goodput=pdb["goodput_measurement_rps"], baseline_goodput=baseline["goodput_measurement_rps"],
                    goodput_change_pct=100*(pdb["goodput_measurement_rps"]/baseline["goodput_measurement_rps"]-1),
                    pdb_meets_90=pdb["slo_attainment"]>=.9, baseline_meets_90=baseline["slo_attainment"]>=.9,
                    slo_equal=pdb["good_requests"]==baseline["good_requests"],
                    slo_not_worse=pdb["good_requests"]>=baseline["good_requests"],
                    energy_lower=pdb["energy_j"]<baseline["energy_j"],
                    Jgood_lower=pdb["energy_per_good_request_j"]<baseline["energy_per_good_request_j"])
                pair["slo_not_worse_and_energy_lower"] = pair["slo_not_worse"] and pair["energy_lower"]
                (regular_pairs if pdb["repeat"] == 1 else confirmation_pairs).append(pair)
    assert len(regular_pairs) == 72 and len(confirmation_pairs) == 12
    summaries = []
    for host in ("A", "B", "C"):
        for baseline in BASELINES:
            candidates = [p for p in regular_pairs if p["measurement_host"] == host and p["baseline"] == baseline]
            for scope in ("all_regular", "PDB_at_least_90_regular"):
                selected = [p for p in candidates if scope == "all_regular" or p["pdb_meets_90"]]
                summary = dict(measurement_host=host, reference_only=host=="B", slo_scale=selected[0]["slo_scale"],
                    baseline=baseline, subset=scope, paired_points=len(selected),
                    rates_rps=";".join(str(p["rate_rps"]) for p in selected),
                    slo_not_worse_and_energy_lower_count=sum(p["slo_not_worse_and_energy_lower"] for p in selected),
                    slo_equal_and_energy_lower_count=sum(p["slo_equal"] and p["energy_lower"] for p in selected),
                    slo_better_and_energy_lower_count=sum(p["slo_not_worse"] and not p["slo_equal"] and p["energy_lower"] for p in selected),
                    slo_worse_count=sum(not p["slo_not_worse"] for p in selected),
                    energy_lower_count=sum(p["energy_lower"] for p in selected),
                    Jgood_lower_count=sum(p["Jgood_lower"] for p in selected),
                    pdb_meets_90_count=sum(p["pdb_meets_90"] for p in selected),
                    baseline_meets_90_count=sum(p["baseline_meets_90"] for p in selected))
                for field in ("energy_reduction_pct", "slo_delta_pp", "Jgood_reduction_pct", "goodput_change_pct"):
                    for direction, operation in (("min", min), ("max", max)):
                        extreme = operation(p[field] for p in selected)
                        summary[field + "_" + direction] = extreme
                        summary[field + "_" + direction + "_rates"] = ";".join(str(p["rate_rps"]) for p in selected if p[field] == extreme)
                summaries.append(summary)
    highlights = []
    for host, rate in (("A", .25), ("B", .75), ("B", 1.25), ("C", 1.5)):
        pairs = [p for p in regular_pairs if p["measurement_host"] == host and p["rate_rps"] == rate]
        highlights.append(dict(measurement_host=host, reference_only=host=="B", rate_rps=rate,
            equal_quality_energy_savings=[p for p in pairs if p["slo_equal"] and p["energy_lower"]],
            better_quality_energy_savings=[p for p in pairs if p["slo_not_worse"] and not p["slo_equal"] and p["energy_lower"]],
            lower_quality_tradeoffs=[p for p in pairs if not p["slo_not_worse"]]))
    for name, values in (("quantitative.csv", summaries), ("regular-pairs.csv", regular_pairs),
                         ("confirmation-pairs.csv", confirmation_pairs)):
        with (OUT/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    evidence = dict(schema="slo-rate-baseline-comparison-quantitative-v1", created_s=time.time(), passed=True,
        source_report=source_ref, B_frozen_reference=ref(b_reference_path), source_builder=ref(Path(__file__)),
        scope="Same-host same-rate 14B ShareGPT comparisons; A4 new, B6 old reference, C8 new regular PDB points.",
        regular_PDB_points=18, regular_baseline_pairs=72, summaries_count=24,
        boundary_confirmations=3, confirmation_baseline_pairs=12,
        formulas=dict(energy_reduction_pct="100*(1-E_PDB/E_baseline)",
            slo_delta_pp="100*(good_PDB/N-good_baseline/N)",
            Jgood_reduction_pct="100*(1-(E_PDB/good_PDB)/(E_baseline/good_baseline))",
            goodput_change_pct="100*((good_PDB/D_PDB)/(good_baseline/D_baseline)-1)"),
        counting_semantics="SLO equality and ordering use integer good counts after asserting identical N and trace. No work-incomplete point is dropped when its audited service outcome is valid.",
        subset_semantics="Only repeat1 points. PDB>=90 subset uses that point's own first observation; B1.5 passing repeat2 cannot enter it.",
        range_semantics="Unweighted min/max across the listed rates within one host and one baseline; endpoints retained. No cross-host weighted or pooled ratio.",
        confirmation_semantics="Repeat2 observations listed separately, compared descriptively with the existing same-rate baseline repeat1. They are not new baseline repetitions and do not contribute to any regular range/count.",
        quality_scope="Strict joint request SLO attainment, not a claim that all latency distributions are equal when attainment is equal.",
        energy_scope="Whole eight-GPU primary measurement window, 100s arrivals plus actual drain/control tail. Different systems may have different durations. Setup/qualification/failed-attempt energy is separate.",
        provenance_checks=provenance_checks, observation_index=observation_index,
        regular_pairs=regular_pairs, summaries=summaries, separate_confirmation_pairs=confirmation_pairs,
        requested_highlights=highlights, independent_seed_CI=False,
        statistical_significance_asserted=False, cross_host_aggregate_ratio=False,
        causal_mechanism_established_by_these_statistics=False,
        GPU_executed=False, raw_modified=False, prior_report_or_runtime_modified=False,
        limitations=["One arrival seed and sampling seed; repeat2 is same-trace boundary confirmation.",
            "B is a frozen old reference, not newly rerun data; full historical qualification context is not replayed here.",
            "Host and SLO scale are confounded, including C PHB topology; the within-host comparisons are kept separate.",
            "Nominal rates have different offered request counts; aggregate total energy across rates is not an equal-work comparison.",
            "Energy/Jgood savings alone do not establish capacity superiority, latency distribution equivalence, or mechanism causality."],
        outputs={name: ref(OUT/name) for name in ("quantitative.csv", "regular-pairs.csv", "confirmation-pairs.csv")})
    assert ref(source) == source_ref
    (OUT / "quantitative-evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(passed=True, summaries=len(summaries), regular_pairs=72, confirmation_pairs=12,
        outputs={name:ref(OUT/name) for name in ("quantitative.csv", "quantitative-evidence.json")}), ensure_ascii=False))
    for summary in summaries:
        if summary["subset"] == "all_regular":
            print(summary["measurement_host"], summary["baseline"], "n", summary["paired_points"],
                "E", round(summary["energy_reduction_pct_min"],2), round(summary["energy_reduction_pct_max"],2),
                "SLOpp", round(summary["slo_delta_pp_min"],2), round(summary["slo_delta_pp_max"],2),
                "Jgood", round(summary["Jgood_reduction_pct_min"],2), round(summary["Jgood_reduction_pct_max"],2),
                "notworse+lessE", summary["slo_not_worse_and_energy_lower_count"])


if __name__ == "__main__":
    main()
