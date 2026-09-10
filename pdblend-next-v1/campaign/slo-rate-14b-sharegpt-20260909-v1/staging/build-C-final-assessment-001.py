"""CPU-only final C assessment from existing terminal evidence; no execution imports."""
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time

N = Path(__file__).resolve().parents[1]
OUT = N / "reports"
SYSTEMS = ["pdblend", "mixed", "distserve", "dynamollm", "ecoserve"]
NAMES = dict(pdblend="PDBlend", mixed="Mixed", distserve="DistServe",
             dynamollm="DynamoLLM", ecoserve="EcoServe")
RATES = [.25, .5, .75, 1., 1.25, 1.5, 1.75, 2.]


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    prior = [OUT / name for name in ("C-assessment-partial.md", "C-comparison-partial.csv",
        "C-assessment-partial-evidence.json", "C-assessment-partial-input.json")]
    prior_refs = [ref(path) for path in prior]
    report_path = OUT / "current/results.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    rows = [r for r in report["observations"] if r["measurement_host"] == "C"]
    acceptance = report["completion_acceptance"]["nodes"]["C"]
    status_path = N / "C/run-002/status.json"
    status = read(status_path)
    assert len(rows) == 41 and acceptance["complete"], "C report not complete"
    assert status["complete"] and status["five_system_complete"] and status["phase"] == "complete"
    assert status["node_lease_held"] is False and status["child"]["exitcode"] == 0
    assert status["finished_s"] >= status["started_s"]
    assert ref(status_path) == acceptance["supervisor"]
    mirror_path = N / "staging/full-mirror-C-final-001/status.json"
    mirror = read(mirror_path)
    assert mirror["complete"] and mirror["scope_complete"] and not mirror["errors"]
    assert not mirror["unavailable"]
    assert ref(Path(mirror["verified_files"]["path"])) == mirror["verified_files"]
    assert ref(status_path) == mirror["terminal"]
    expected = {(system, rate, repeat) for system in SYSTEMS for rate in RATES
                for repeat in ([1, 2] if system == "pdblend" and rate == 2 else [1])}
    assert {(r["system"], r["rate_rps"], r["repeat"]) for r in rows} == expected
    assert len({r["cell_id"] for r in rows}) == 41
    spec = importlib.util.spec_from_file_location("C_final_assessment_crosscheck", OUT / "crosscheck.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    rows.sort(key=lambda r: (r["rate_rps"], SYSTEMS.index(r["system"]), r["repeat"]))
    with (OUT / "C-comparison-partial.csv").open() as stream:
        fields = csv.DictReader(stream).fieldnames
    comparisons, replays = [], []
    for row in rows:
        assert row["measurement_valid"] and row["service_terminal_valid"]
        assert row["strict_slo_recomputed"] and row["unknown_error_count"] == 0
        replay = checker.check_row(row)
        with open(row["raw_requests"]["path"]) as stream:
            requests = list(csv.DictReader(stream))
        counts = dict(good=0, complete_ttft_only_fail=0, complete_tpot_only_fail=0,
                      complete_both_fail=0, incomplete_timeout=0, incomplete_capacity=0)
        failures = {item["request_id"]: item["classification"]
                    for item in row["service_failures"]["failures"]}
        for request in requests:
            done = (checker.truth(request["success"]) and not request["error"] and
                request["token_count_source"] == "server_usage" and
                checker.truth(request["token_ids_verified"]) and
                int(request["input_tokens"]) == int(request["prompt_len"]) and
                int(request["generated_tokens"]) == int(request["output_len"]) and
                not checker.truth(request["request_timeout"]))
            if done:
                ttft = float(request["ttft_s"]) >= 10.
                tpot = float(request["tpot_s"]) >= .3
                category = ("complete_both_fail" if ttft and tpot else
                    "complete_ttft_only_fail" if ttft else
                    "complete_tpot_only_fail" if tpot else "good")
            else:
                assert failures[request["request_id"]] == "request_hard_timeout"
                assert request["error"] == "request_hard_timeout" and checker.truth(request["request_timeout"])
                category = "incomplete_timeout"
            counts[category] += 1
        assert sum(counts.values()) == row["n_expected"] and counts["good"] == row["good_requests"]
        result = {field: row.get(field, "") for field in fields}
        result.update(node="C", status="valid_terminal", role="boundary_confirmation" if row["repeat"] == 2 else "grid",
            slo_attainment_pct=100 * row["slo_attainment"], attainment_target=.9,
            meets_attainment_target=row["slo_attainment"] >= .9, energy_kj=row["energy_j"] / 1000,
            **{key: value for key, value in counts.items() if key != "good"})
        for field in ("n_expected", "completed_work_requests", "good_requests", "energy_measured_gpu_count"):
            result[field] = int(row[field])
        for prefix, key in (("audit", "audit_reference"), ("bench", "raw_requests"), ("power", "raw_power")):
            result[prefix + "_path"] = row[key]["path"]
            result[prefix + "_sha256"] = row[key]["sha256"]
        comparisons.append(result)
        replays.append(dict(cell_id=row["cell_id"], passed=True, recomputed=replay,
            mutually_exclusive_partition=counts, references={key: row[key] for key in
            ("audit_reference", "raw_requests", "raw_power", "checkpoint", "receipt", "summary", "binding", "trace_reference")}))
    for rate in RATES:
        assert len({row["trace_sha256"] for row in rows if row["rate_rps"] == rate}) == 1
    snapshot_path = OUT / "C-assessment-input.json"
    save(snapshot_path, dict(schema="slo-rate-C-final-assessment-input-v1", source_path_at_read=str(report_path),
        source_sha256_at_read=hashlib.sha256(report_bytes).hexdigest(), source_created_s=report["created_s"],
        observations=rows, completion_acceptance=acceptance, terminal=ref(status_path), mirror=ref(mirror_path)))
    csv_path = OUT / "C-comparison.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)

    def get(system, rate, repeat=1):
        return next(row for row in comparisons if (row["system"], row["rate_rps"], row["repeat"]) == (system, rate, repeat))

    paired = []
    for rate in (1.75, 2.):
        pdb = get("pdblend", rate)
        for system in SYSTEMS[1:]:
            baseline = get(system, rate)
            paired.append(dict(rate_rps=rate, pdb_repeat=1, baseline=system,
                quality_delta_pp=pdb["slo_attainment_pct"] - baseline["slo_attainment_pct"],
                energy_reduction_pct=100 * (1 - pdb["energy_j"] / baseline["energy_j"]),
                goodput_change_pct=100 * (pdb["goodput_measurement_rps"] / baseline["goodput_measurement_rps"] - 1),
                Jgood_reduction_pct=100 * (1 - pdb["energy_per_good_request_j"] / baseline["energy_per_good_request_j"])))
    first_losses = {system: next((rate for rate in RATES if get(system, rate)["slo_attainment"] < .9), None)
                    for system in SYSTEMS}
    assert first_losses["pdblend"] == 2. and first_losses["dynamollm"] == 1.5
    eco2 = get("ecoserve", 2.)
    evidence_path = OUT / "C-assessment-evidence.json"
    evidence = dict(schema="slo-rate-C-final-assessment-evidence-v1", created_s=time.time(), passed=True,
        node="C", report_status="final", observed_measurements=41, expected_measurements=41,
        pending_measurements=[], frozen_C_only_input=ref(snapshot_path), comparison_csv=ref(csv_path),
        checker=ref(OUT / "crosscheck.py"), assessment_builder=ref(Path(__file__)),
        fresh_core_replay_count=41, fresh_core_replays=replays, paired_comparisons=paired,
        first_observed_loss_rates=first_losses, terminal=ref(status_path), C_completion_acceptance=acceptance,
        full_context_mirror_status_reference=ref(mirror_path), verified_files=mirror["verified_files"],
        nonweight_mirror_scope_complete=True, retained_weight_reconstruction_complete=mirror["weight_reconstruction_complete"],
        full_qualification_replayed=False, same_rate_same_trace_verified=True,
        all_request_denominators_preserved=True, C_only_new_measurements=True,
        old_B_reference_eligibility_not_used=True, prior_partial_artifacts_preserved=prior_refs,
        C_physical_interconnect=ref(N / "C/baseline-002/qualification/frequency/interconnect.txt"),
        independent_seed_CI=False, cross_host_ratios=False, GPU_executed=False, raw_modified=False,
        final_eco2_explicitly_checked=dict(audit_reference=next(row["audit_reference"] for row in rows
            if row["system"] == "ecoserve" and row["rate_rps"] == 2.),
            n_expected=eco2["n_expected"], completed_work_requests=eco2["completed_work_requests"],
            good_requests=eco2["good_requests"], slo_attainment=eco2["slo_attainment"]))
    finished = datetime.datetime.fromtimestamp(status["finished_s"], datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+8")
    lines = [
        f'C 机 14B ShareGPT、SLO scale 2.0 的五系统实验已完成：0.25 至 2.0 rps 每隔 0.25 测一档，五系统各八档，另有 PDBlend 的 2.0 rps 同 trace 确认，共 41/41 次。最后监督器于 {finished} 完成，已释放节点锁；完成门槛通过。本次分析重新复算全部 41 份请求 CSV 和八卡功耗 CSV，均匹配冻结审计。',
        '',
        f'完整非权重证据镜像已完成并通过 SHA 核验；最后一轮补齐 {mirror["downloaded_files"]} 个文件、{mirror["downloaded_bytes"]:,} 字节，无缺失或冲突。保留权重二进制的本地重建不在此次镜像范围。这里的核心指标复算和完整非权重镜像验收各有独立证据；本次分析未重新执行物理资格验证或任何 GPU 测量。',
        '',
        'PDBlend 的最后一个达标档位是 1.75 rps：135/144（93.75%），全部请求完整输出，9 个未达标请求只因 TTFT 未达要求。2.0 rps 首测为 133/164（81.10%），确认为 134/164（81.71%），首次低于 90% 的封顶档位保持为 2.0 rps。两次各有 163 个完整请求和 1 个已独立审计的 120 s 硬超时，超时仍留在 164 个请求的分母中。另有 30、29 个完整请求只因 TTFT 失分；完整请求没有仅 TPOT 或两项同时失分。这是本次已测网格的服务边界，不代表连续负载轴上的精确饱和点。',
        '',
        '每个 good 请求须完整输出，且同时满足 TTFT < 10 s、TPOT < 0.3 s。SLO 达成率以全部 trace 请求为分母。每档五系统使用完全相同的冻结 trace；rate 是目标到达率，100 s 窗口的实际请求数依次为 24、48、59、81、102、122、144、164。八卡能耗 E 包含到达窗口及实际排空／控制尾段；goodput = good / 实际主测量时长 D，J/good = E / good，各系统 D 可不同。部署、资格验证及失败尝试的能耗在独立台账。确认点单独保留，不与首测取最优或平均。',
        '',
        '| 目标负载 (rps) | 系统 | 达标 / 全部 | 完整请求 | SLO 达成率 | D (s) | 八卡 E (kJ) | goodput (req/s) | J/good |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        label = NAMES[row["system"]] + ('（确认）' if row["repeat"] == 2 else '')
        lines.append(f'| {row["rate_rps"]:g} | {label} | {row["good_requests"]}/{row["n_expected"]} | {row["completed_work_requests"]} | {row["slo_attainment_pct"]:.2f}% | {row["measurement_duration_s"]:.2f} | {row["energy_kj"]:.3f} | {row["goodput_measurement_rps"]:.5f} | {row["energy_per_good_request_j"]:.2f} |')
    lines += ['',
        '在最后一个达标档位 1.75 rps，PDBlend 为 135/144（93.75%）、85.011 kJ、0.97302 goodput、629.71 J/good。Mixed 与 EcoServe 均为 144/144（100%）；DistServe 为 142/144（98.61%）；DynamoLLM 为 120/144（83.33%）。PDBlend 的能耗和 J/good 在五系统中最低，但相对 Mixed、DistServe、EcoServe 的达成率分别低 6.25、4.86、6.25 个百分点，goodput 也分别低 20.12%、12.98%、17.33%。因此对这三种基线，节能伴随服务质量和有效吞吐的取舍。',
        '',
        '相对 DynamoLLM，PDBlend 在 1.75 rps 同时有更高达成率（高 10.42 个百分点）、更高 goodput（高 29.50%），以及更低八卡能耗（低 47.09%）和 J/good（低 52.97%）。相对 EcoServe 的同档能耗低 26.67%、J/good 低 21.78%，但上述质量差异不能省略。',
        '',
        f'最后到齐的 EcoServe 2.0 rps 已按实际结果核对：{eco2["completed_work_requests"]}/{eco2["n_expected"]} 个请求完整输出，{eco2["good_requests"]}/{eco2["n_expected"]} 达标（{eco2["slo_attainment_pct"]:.2f}%），八卡能耗 {eco2["energy_kj"]:.3f} kJ，goodput {eco2["goodput_measurement_rps"]:.5f} req/s，J/good {eco2["energy_per_good_request_j"]:.2f}。该点按实测值纳入，未从较低负载外推。',
        '',
        '2.0 rps 是 PDBlend 已确认低于 90% 的档位。Mixed 为 100%，DistServe 为 96.34%，DynamoLLM 为 75.61%。PDBlend 首测相对 Mixed 和 DistServe 分别少 31、25 个 good 请求；它的 goodput 从 1.75 rps 的 0.97302 降到首测 0.90810、确认 0.92259，J/good 从 629.71 升至 688.53、668.28。更高输入负载已伴随更差的有效服务，较低绝对能耗不能替代质量达标要求。',
        '',
        '以下以 PDBlend 首测为固定参照。能耗与 J/good 的降低以对应基线为分母，goodput 变化为 PDBlend / 基线 − 1。全部比率都在同一 C 机、同一档位和同一 trace 上计算。',
        '',
        '| 负载 (rps) | 相对基线 | SLO 差值 (百分点) | 八卡能耗降低 | goodput 变化 | J/good 降低 |',
        '|---:|---|---:|---:|---:|---:|']
    for row in paired:
        lines.append(f'| {row["rate_rps"]:g} | {NAMES[row["baseline"]]} | {row["quality_delta_pp"]:+.2f} | {row["energy_reduction_pct"]:.2f}% | {row["goodput_change_pct"]:+.2f}% | {row["Jgood_reduction_pct"]:.2f}% |')
    passing = [NAMES[system] for system in SYSTEMS if first_losses[system] is None]
    other_losses = [(NAMES[system], first_losses[system]) for system in SYSTEMS if system not in ('pdblend', 'dynamollm') and first_losses[system] is not None]
    lines += ['',
        'DynamoLLM 的首次已测失守点为 1.5 rps（105/122，86.07%），PDBlend 为 2.0 rps。' +
        ('、'.join(passing) + ' 在这八档均达到 90%；实验按 PDBlend 封顶结束，因此这些基线的更高负载边界未知。' if passing else '') +
        ''.join(f'{name} 的首次已测失守点为 {rate:g} rps。' for name, rate in other_losses),
        '',
        'C 的八卡拓扑记录有 56 个有向非自身连接，全部为 PHB。所有系统对比均在 C 机完成；C 与其他机器的不同 SLO scale 不能直接相除来解释为纯粹的 SLO 效应，主机拓扑及其他机器差异仍会混入。这里只使用本轮 C 新测证据，旧 B 参考的上下文完整性不参与 C 的失败计数。',
        '',
        '一个到达种子 701、一个抽样种子 20260907；同 trace 边界确认不等价于独立种子重复，不报告置信区间或统计显著性。结论限于 C 机、本模型和数据集、完整请求及此次 100 s 到达窗口；不把本次观察推广成其他硬件或负载下的优势保证。',
        '',
        f'完整数值见 [C-comparison.csv]({csv_path})；41 个核心复算、终态和镜像验收引用见 [C-assessment-evidence.json]({evidence_path})。完整非权重镜像状态见 [full-mirror-C-final-001/status.json]({mirror_path})。原阶段性报告及其 CSV、证据和输入快照均保留原字节。']
    md_path = OUT / "C-assessment.md"
    md_path.write_text("\n".join(lines) + "\n")
    evidence["assessment_markdown"] = ref(md_path)
    assert [ref(path) for path in prior] == prior_refs
    save(evidence_path, evidence)
    print(json.dumps(dict(passed=True, rows=41, first_losses=first_losses,
        eco2=evidence["final_eco2_explicitly_checked"],
        outputs=[ref(path) for path in (md_path, csv_path, evidence_path, snapshot_path)]), ensure_ascii=False))


if __name__ == "__main__":
    main()
