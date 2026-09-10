import json

import pytest

from ecopadg.scalability.report import (build_report, _capacity_plot, _control_plot, _cpu_aggregate,
                                      _target, _targets, _weak_scaling, CPU_SEEDS, CPU_SCALES)
from ecopadg.scalability.protocol import FORMAL_SEEDS


def test_no_measurements_do_not_create_positive_figures(tmp_path):
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "capacity.svg").write_text("previous report curve")
    result = build_report([], tmp_path / "report")
    assert result["formal_valid_runs"] == 0 and result["plots"] == []
    assert "尚无正式容量测量" in (tmp_path / "report" / "REPORT.md").read_text()
    assert len(list((tmp_path / "report").glob("*.csv"))) == 8
    assert result["target_status_counts"]["pass"] == 0
    assert result["target_status_counts"]["incomplete"] > 0
    assert not (tmp_path / "report" / "capacity.svg").exists()


def test_duplicate_explicit_input_is_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="duplicate"):
        build_report([path, path], tmp_path / "report")


def test_formal_capacity_paired_scales_and_source_mixing(tmp_path, monkeypatch):
    rows, paths = {}, []
    for n in (4, 8):
        for seed in range(5):
            for passed, rate in ((True, n), (False, n * 1.04)):
                path = tmp_path / f"{n}-{seed}-{passed}.json"
                path.write_text(json.dumps(dict(scope="gpu_serving")))
                paths.append(path)
                rows[str(path)] = dict(scope="gpu_serving", dataset="sharegpt", system="pdblend-joint",
                    n_gpus=n, seed=seed, rate_rps=rate, stage="capacity", formal_eligible=True,
                    measurement_valid=True, capacity_pass=passed, model="14b", host_id="host-a",
                    profile_sha256="profile", source_config_sha256="config", source_hashes={"source.py": "hash"})
    monkeypatch.setattr("ecopadg.scalability.report.audit_run", lambda path: rows[str(path)])
    result = build_report(paths, tmp_path / "report")
    assert result["capacity_rows"] == 10 and result["efficiency_rows"] == 1
    assert (tmp_path / "report" / "scaling_efficiency.pdf").exists()
    rows[str(paths[0])]["profile_sha256"] = "different"
    with pytest.raises(ValueError, match="cannot mix"):
        build_report(paths, tmp_path / "bad")


def test_cpu_smoke_is_separate_and_has_no_formal_figure(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(dict(scope="control_plane_replay", n_instances=128, layout="pd",
        measurement_valid=True, formal_evidence=False, planner=dict(latency=dict(p99_ms=3.)))))
    index = tmp_path / "index.json"
    index.write_text(json.dumps(dict(scope="control_plane_replay", cells=[dict(summary=str(summary))])))
    result = build_report([index], tmp_path / "report")
    assert result["control_plane_rows"] == 1 and result["gpu_runs"] == 0 and not result["plots"]


def test_energy_pairs_require_identical_trace_and_rate(tmp_path, monkeypatch):
    rows, paths = {}, []
    for system, trace in (("pdblend-joint", "one"), ("mixed", "other")):
        path = tmp_path / f"{system}.json"
        path.write_text(json.dumps(dict(scope="gpu_serving")))
        paths.append(path)
        rows[str(path)] = dict(scope="gpu_serving", dataset="sharegpt", system=system, n_gpus=4, seed=1,
            rate_rps=1., stage="weak", formal_eligible=True, measurement_valid=True,
            model="14b", host_id="host-a", profile_sha256="profile", source_config_sha256="config",
            source_hashes={"source.py": "hash"}, trace_sha256=trace, joules_per_good_request=10.)
    monkeypatch.setattr("ecopadg.scalability.report.audit_run", lambda path: rows[str(path)])
    assert build_report(paths, tmp_path / "report")["weak_energy_rows"] == 0
    rows[str(paths[1])]["trace_sha256"] = "one"
    assert build_report(paths, tmp_path / "paired")["weak_energy_rows"] == 1


def cpu_row(n, seed, *, mode="planner", layout="selective", active_mode="per_instance4", latency=10.):
    row = dict(scope="control_plane_replay", mode=mode, n_instances=n, seed=seed, layout=layout,
        active_mode=active_mode, formal_evidence=True, protocol_sampling_complete=True, measurement_valid=True,
        cpu=dict(model="test-cpu", physical_core_count=2, planning_workers=1, logical_cpu_ids=[0, 1]),
        source_hashes={"planner.py": "source-sha"}, provenance=dict(input_hashes={"profiles": {"sha256": "profile-sha"}}),
        parameters=dict(planner_calls=1000, concurrent_seconds=300, input_tokens=128, output_tokens=128),
        planner=dict(latency=dict(count=1000, p99_ms=latency), budget_fallback_ratio=.2, no_candidate_ratio=0))
    if mode == "concurrent":
        row.update(arrival_window_s=300, control_latency=dict(p99_ms=latency * 2),
            input_arrival_rate_rps=10., successful_commit_throughput_rps=10., stale_retry_ratio=.02,
            counts=dict(arrived=3000, successful_commits=3000, pending_at_window_end=0),
            audited_backlog_stability=dict(valid=True, slope_rps=0.))
    return row


def test_cpu_groups_seed_metrics_before_plotting_and_keeps_modes_separate():
    import matplotlib.pyplot as plt
    rows = [cpu_row(n, seed, latency=n + index) for n in (4, 8)
            for index, seed in enumerate(CPU_SEEDS)]
    rows += [cpu_row(4, seed, mode="concurrent", latency=20.) for seed in CPU_SEEDS]
    aggregate = _cpu_aggregate(rows)
    planner = [r for r in aggregate if r["mode"] == "planner" and r["metric"] == "planning_p99_ms"]
    assert [(r["n_instances"], r["estimate"], r["seed_count"]) for r in planner] == [(4, 6., 5), (8, 10., 5)]
    absent = [r for r in aggregate if r["mode"] == "planner" and r["metric"] == "successful_commit_throughput_rps"]
    assert all(r["estimate"] is None and r["seed_count"] == 0 for r in absent)
    fig, ax = plt.subplots()
    _control_plot(ax, aggregate)
    series = {line.get_label(): line for line in ax.lines}
    assert list(series["planner/selective/per_instance4"].get_xdata()) == [4, 8]
    assert list(series["concurrent/selective/per_instance4"].get_xdata()) == [4]
    plt.close(fig)


def test_cpu_source_and_duplicate_seed_cannot_be_mixed():
    rows = [cpu_row(4, seed) for seed in CPU_SEEDS]
    with pytest.raises(ValueError, match="duplicate"):
        _cpu_aggregate(rows + rows[:1])
    rows[-1]["source_hashes"] = {"planner.py": "changed-source"}
    with pytest.raises(ValueError, match="cannot mix CPU"):
        _cpu_aggregate(rows)


def test_one_complete_cpu_submatrix_is_not_the_whole_matrix_or_full_control():
    rows = [cpu_row(n, seed, latency=60. if n == 128 else 10.) for n in CPU_SCALES for seed in CPU_SEEDS]
    targets = _targets([], [], [], [], _cpu_aggregate(rows))
    subsets = [t for t in targets if t["target"] == "cpu_submatrix_coverage" and t["status"] == "pass"]
    assert len(subsets) == 1 and subsets[0]["mode"] == "planner"
    assert next(t for t in targets if t["target"] == "cpu_full_matrix_coverage")["status"] == "incomplete"
    assert all(t["status"] == "incomplete" for t in targets if t["target"] == "cpu_full_control")
    failed = [t for t in targets if t["target"] == "cpu_planning_p99" and t["status"] == "fail"]
    assert len(failed) == 1 and failed[0]["n_instances"] == 128


def test_target_uses_registered_point_criterion_and_separate_ci_support():
    result = _target("energy", .08, .10, "<=", True, ci_low=.01, ci_high=.14)
    assert result["status"] == "pass" and result["ci_support"] == "overlaps"
    assert _target("energy", .08, .10, "<=", False)["status"] == "incomplete"


def test_cpu_actual_arrivals_and_bounded_drain_drive_commit_target_not_nominal_lambda():
    rows = [cpu_row(4, seed, mode="concurrent") for seed in CPU_SEEDS]
    for row in rows:
        row.update(input_arrival_rate_rps=10., successful_commit_throughput_rps=8.9,
                   counts=dict(arrived=2700, successful_commits=2700, pending_at_window_end=30))
    targets = _targets([], [], [], [], _cpu_aggregate(rows))
    target = next(t for t in targets if t["target"] == "cpu_commit_meets_actual_arrivals"
                  and t.get("layout") == "selective" and t.get("active_mode") == "per_instance4" and t["n_instances"] == 4)
    assert target["status"] == "pass" and target["estimate"] == 1.
    for row in rows:
        row.pop("counts")
    targets = _targets([], [], [], [], _cpu_aggregate(rows))
    target = next(t for t in targets if t["target"] == "cpu_commit_meets_actual_arrivals"
                  and t.get("layout") == "selective" and t.get("active_mode") == "per_instance4" and t["n_instances"] == 4)
    assert target["status"] == "incomplete"


def test_energy_inflation_is_same_seed_q_and_generator_pair():
    rows = [dict(dataset="sharegpt", system="pdblend", stage="weak", formal_eligible=True,
        measurement_valid=True, n_gpus=n, seed=seed, q=.25, rate_rps=n * .25,
        trace_generation_identity="frozen-pool-and-generator", slo_ttft_s=5, slo_tpot_s=.15,
        joules_per_good_request=10 if n == 4 else 10.8) for n in (4, 8) for seed in FORMAL_SEEDS]
    result = _weak_scaling(rows)
    assert len(result) == 1 and result[0]["five_seed_complete"]
    assert result[0]["energy_inflation"] == pytest.approx(.08)
    assert result[0]["inflation_ci95_low"] == pytest.approx(.08)
    rows[-1]["q"] = .3
    assert not _weak_scaling(rows)[0]["five_seed_complete"]
    for row in rows:
        if row["n_gpus"] == 8:
            row["trace_generation_identity"] = "different-pool"
    assert _weak_scaling(rows) == []


def test_capacity_reference_lines_only_connect_comparable_observed_scales():
    import matplotlib.pyplot as plt
    rows = [dict(dataset="sharegpt", system="pdblend", n_gpus=n, seed=seed,
                 capacity_lower_rps=n / 4, inconsistent=False) for n in (3, 4, 6, 8) for seed in FORMAL_SEEDS]
    fig, ax = plt.subplots()
    _capacity_plot(ax, rows)
    assert {tuple(line.get_xdata()) for line in ax.lines} == {(3, 6), (4, 8)}
    assert all(line.get_ydata()[1] == 2 * line.get_ydata()[0] for line in ax.lines)
    plt.close(fig)


def test_gpu_slo_and_observed_kv_span_figures_without_fabricated_bytes(tmp_path, monkeypatch):
    rows, paths = {}, []
    for n in (4, 8):
        for seed in FORMAL_SEEDS:
            path = tmp_path / f"n{n}-s{seed}.json"
            path.write_text(json.dumps(dict(scope="gpu_serving")))
            paths.append(path)
            rows[str(path)] = dict(scope="gpu_serving", dataset="sharegpt", system="pdblend", n_gpus=n,
                seed=seed, q=.25, rate_rps=n * .25, stage="weak", formal_eligible=True, measurement_valid=True,
                model="14b", host_id="host-a", profile_sha256="profile", source_config_sha256="config",
                source_hashes={"source.py": "hash"}, trace_sha256=f"trace-{n}-{seed}",
                trace_generation_identity="same-pool", joules_per_good_request=10., offered_requests=100,
                ttft_s=dict(p99=.1), tpot_s=dict(p99=.01), slo_attainment=.95, slo_ttft_s=5., slo_tpot_s=.15,
                kv_transfer_s=2., kv_transfer_count=10, kv_send_s=1., kv_receive_s=1.5, kv_transfer_bytes=None)
    monkeypatch.setattr("ecopadg.scalability.report.audit_run", lambda path: rows[str(path)])
    result = build_report(paths, tmp_path / "report")
    assert result["weak_energy_scaling_rows"] == 1
    for name in ("gpu_ttft", "gpu_tpot", "gpu_attainment", "gpu_kv_span", "weak_energy_scaling"):
        assert (tmp_path / "report" / f"{name}.svg").exists()
    assert not (tmp_path / "report" / "gpu_kv_bytes.svg").exists()


def test_cli_manifest_list_array_and_direct_inputs_reject_duplicates(tmp_path, monkeypatch):
    from ecopadg.scalability.report import main
    manifest = tmp_path / "cpu.json"
    manifest.write_text(json.dumps(dict(scope="control_plane_replay", rows=[])))
    path_list = tmp_path / "paths.json"
    path_list.write_text(json.dumps([manifest.name]))
    monkeypatch.setattr("sys.argv", ["report", "--manifest-list", str(path_list), "--out", str(tmp_path / "report")])
    main()
    assert (tmp_path / "report" / "REPORT.md").exists()
    monkeypatch.setattr("sys.argv", ["report", "--manifest-list", str(path_list), "--manifests", str(manifest),
                                    "--out", str(tmp_path / "report2")])
    with pytest.raises(ValueError, match="duplicate"):
        main()
