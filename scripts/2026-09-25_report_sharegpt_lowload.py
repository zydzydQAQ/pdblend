#!/usr/bin/env python3
"""Render the completed low-load experiment without modifying its inputs."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'results/2026-09-25/sharegpt-lowload-retest-v1'
LABELS = {'pd2100': 'PDblend 原版 2100 MHz', 'eco': 'EcoServe',
          'pd1500': 'PDblend 候选 1500 MHz'}


def main():
    report = json.loads((PACKAGE / 'report.json').read_text())
    records = report['records']
    assert len(records) == 9
    assert all(r['status'] in ('succeeded', 'failed', 'cancelled', 'blocked') for r in records)
    single = (PACKAGE / 'user-no-repeat-directive.json').exists()
    expected = 1 if single else 3
    if single:
        assert all(r['status'] == 'cancelled' for r in records if r['repeat'] > 1)
    rows, gates = [], {}
    for record in records:
        row = {'arm': record['arm'], 'repeat': record['repeat'], 'status': record['status']}
        row.update(record.get('metrics', {}))
        row['observationally_eligible'] = record.get('observationally_eligible', False)
        row['exclusions'] = '; '.join(record.get('observational_exclusions', []))
        ref = record.get('result')
        if ref:
            data = Path(ref['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == ref['sha256']
            raw = json.loads(data)
            row['output_tokens'] = raw['metrics'].get('output_tokens')
            row['joint_slo_requests'] = raw['metrics'].get('joint_slo_requests')
            row['result_path'] = ref['path']
            row['result_sha256'] = ref['sha256']
            row['measurement_evidence_valid'] = raw.get('measurement_evidence_valid')
            row['missing_gates'] = '; '.join(raw.get('missing_gates', []))
            a = raw.get('acceptance', {})
            gates[f"{record['arm']}-r{record['repeat']}"] = {
                'missing_gates': raw.get('missing_gates', []),
                'gate_failures': a.get('gate_failures', {}),
                'blocked_gates': a.get('blocked_gates', {}),
                'profile_missing_gates': a.get('profile_missing_gates', []),
                'formal_eligible': raw.get('formal_eligible'),
            }
        rows.append(row)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (PACKAGE / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    basis = '每种配置仅一次观测' if single else '每种配置三次重复观测'
    energy_label = '单次总能耗' if single else '平均总能耗 ± 样本标准差'
    latency_label = 'P99 TTFT' if single else '平均 P99 TTFT'
    lines = ['# ShareGPT ×0.25 三臂重测结果', '',
             f'这是同一评估 trace 的配置调优，{basis}。八卡服务及尾部能耗均计入；每次 262 个请求、150 秒服务窗口。', '']
    if single:
        lines += ['用户要求不重复测，原计划中后六次未执行任务已取消，仅保留三臂各一次。取消记录及原协议仍保留。', '']
    lines += [f'| 配置 | 完成记录 | 整体 SLO 通过 | {energy_label} | 能耗范围 | {latency_label} |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    comparable = {}
    for arm, label in LABELS.items():
        rr = [r for r in rows if r['arm'] == arm and r.get('energy_service_tail_j') is not None]
        energy = [r['energy_service_tail_j'] / 1000 for r in rr]
        if energy:
            spread = statistics.stdev(energy) if len(energy) > 1 else None
            ttft = statistics.mean(r['ttft_p99_s'] for r in rr)
            energy_text = f'{energy[0]:.3f} kJ' if single else (
                f'{statistics.mean(energy):.3f} ± {spread:.3f} kJ' if spread is not None else '重复不足')
            range_text = '—' if single else f'{min(energy):.3f}–{max(energy):.3f} kJ'
            lines.append(f'| {label} | {len(rr)}/{expected} | {sum(r["slo_pass"] is True for r in rr)}/{expected} | '
                         f'{energy_text} | {range_text} | {ttft:.3f} s |')
        else:
            lines.append(f'| {label} | 0/{expected} | 0/{expected} | 无记录 | 无记录 | 无记录 |')
        comparable[arm] = (statistics.mean(r['energy_service_tail_j'] for r in rr)
                           if len(rr) == expected and all(r['observationally_eligible'] for r in rr) else None)
    lines += ['', ('单次观测不能判断重复波动或稳定性。' if single else
                    '均值和标准差描述本次重复波动，不代表跨工作负载置信区间。') +
              '表中保留所有有能量记录的运行；比较要求两臂的预定运行都满足完整测量、请求完成及 SLO。', '']
    comparisons = {}
    candidate = comparable['pd1500']
    for other in ('pd2100', 'eco'):
        baseline = comparable[other]
        if candidate is not None and baseline is not None:
            savings = 100 * (1 - candidate / baseline)
            comparisons[other] = {'candidate_energy_reduction_percent': savings, 'observations_per_arm': expected}
            change = '降低' if savings >= 0 else '增加'
            lines.append(f'- 在{basis}下，1500 MHz 候选相对 {LABELS[other]} 的总能耗{change} {abs(savings):.2f}%。')
        else:
            lines.append(f'- 1500 MHz 候选与 {LABELS[other]} 未形成两臂预定运行均完整达标的比较，保留原始结果。')
    lines += ['', '## 原计划九个任务的最终状态', '',
              '| 配置 | 重复 | 状态 | 总能耗 kJ | 联合 SLO 请求数 | P99 TTFT s | P99 TPOT ms |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        if r.get('energy_service_tail_j') is None:
            lines.append(f'| {LABELS[r["arm"]]} | {r["repeat"]} | {r["status"]} | — | — | — | — |')
            continue
        lines.append(f'| {LABELS[r["arm"]]} | {r["repeat"]} | {r["status"]} | '
                     f'{r["energy_service_tail_j"]/1000:.3f} | {r.get("joint_slo_requests")}/262 | '
                     f'{r["ttft_p99_s"]:.3f} | {r["tpot_p99_s"]*1000:.3f} |')
    lines += ['', '## 验收边界', '',
              '本轮没有更改正式资格标记，也没有覆盖原矩阵结果。以下是原验收器记录的未通过门禁；其中被其他门禁阻断的检查并不等于已经验证通过。', '',
              '| 运行 | 未通过或被阻断的门禁 |', '| --- | --- |']
    for name, g in gates.items():
        lines.append(f'| {name} | {", ".join(g["missing_gates"]) or "无"} |')
    lines += ['', '具体失败原因见 `result-gates.json`，独立审查见 `result-independent-review.json`。原记录诊断、协议、脚本准备阶段溯源缺口见 `README.md`。', '',
              '1500 MHz 只作为此低负载点的候选配置；未更改全局默认参数，也未纳入工作区其他 Shield 源码改进。', '']
    (PACKAGE / 'RESULTS.md').write_text('\n'.join(lines))
    (PACKAGE / 'result-gates.json').write_text(json.dumps(gates, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'counts': dict(Counter(r['status'] for r in rows)), 'comparisons': comparisons,
                      'report': str(PACKAGE / 'RESULTS.md')}))


if __name__ == '__main__':
    main()
