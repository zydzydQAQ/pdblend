import json

from pdblend2.bench.report import compare, load_points, report


def point(tmp, name, policy, dataset, scale, energy, slo, success=1.0):
    d = tmp / name
    d.mkdir()
    s = dict(model="m", policy=dict(name=policy), trace_meta=dict(dataset=dataset, scale=scale, seed=701),
             trace=dict(mean_rps=scale * 10), slo=dict(offered=100, success_rate=success, joint_slo_rate=slo,
             ttft_p50=0.1, ttft_p90=0.2, tpot_p50=0.01, tpot_p90=0.02, output_tokens=1000),
             energy_j=energy, window_energy_j=energy * 0.9, mean_power_w=energy / 100, j_per_request=energy / 100,
             j_per_token=energy / 1000, tail_s=2.0, controller=dict(events=dict(plan=3), shield_events=[]))
    (d / "summary.json").write_text(json.dumps(s))


def test_compare_and_report(tmp_path):
    point(tmp_path, "a", "static_best", "sharegpt", 0.5, 1000, 0.95)
    point(tmp_path, "b", "pdblend", "sharegpt", 0.5, 800, 0.94)
    point(tmp_path, "c", "static_best", "sharegpt", 1.0, 2000, 0.97)
    point(tmp_path, "d", "pdblend", "sharegpt", 1.0, 1700, 0.90)      # degraded by > 1pp
    point(tmp_path, "e", "static_best", "alpaca", 0.5, 1000, 0.80)     # reference infeasible
    point(tmp_path, "f", "pdblend", "alpaca", 0.5, 700, 0.95)
    point(tmp_path, "g", "static_best", "longbench", 0.5, 1000, 0.95)
    point(tmp_path, "h", "pdblend", "longbench", 0.5, 900, 0.95)
    result = report(tmp_path)
    summ = result["summary"]["pdblend"]
    assert summ["per_dataset"]["sharegpt"]["valid"] == 1
    assert abs(summ["per_dataset"]["sharegpt"]["mean_saving"] - 0.2) < 1e-9
    assert summ["per_dataset"]["sharegpt"]["statuses"]["service_degraded"] == 1
    assert summ["per_dataset"]["alpaca"]["statuses"]["inconclusive_reference_infeasible"] == 1
    assert abs(summ["overall_saving"] - (0.2 + 0.1) / 2) < 1e-9
    assert not summ["all_points_valid"]
    assert (tmp_path / "comparisons.csv").exists() and (tmp_path / "points.csv").exists()
