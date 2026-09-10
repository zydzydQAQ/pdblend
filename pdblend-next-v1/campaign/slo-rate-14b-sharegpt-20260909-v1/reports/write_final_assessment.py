"""Render the final narrative from completed, separately audited artifacts (CPU only)."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/final-001'


def read(path):
    return json.loads(path.read_text())


def link(path, label=None):
    return f'[{label or path.name}]({path})'


def main():
    acceptance = read(OUT / 'completion-acceptance.json')
    assert acceptance['complete'] is True
    assert acceptance['full_context_mirror_status'] == 'verified'
    rows = list(csv.DictReader((OUT / 'observations.csv').open()))
    assert len(rows) == 93
    assert all(r['report_eligible'] == 'True' for r in rows)
    for node, count in [('A', 21), ('C', 41), ('B', 31)]:
        assert sum(r['measurement_host'] == node for r in rows) == count
    new_rows = [r for r in rows if r['measurement_host'] in ['A', 'C']]
    assert sum(float(r['n_expected']) for r in new_rows) == 5025
    assert sum(float(r['completed_work_requests']) for r in new_rows) == 5023
    energy_path = ROOT / 'reports/energy-close-final-001/energy-closeout.json'
    energy = read(energy_path)
    assert energy['new_formal_grid_complete'] is True
    assert energy['verified_new_primary_windows'] == 62
    for node, count in [('A', 21), ('C', 41)]:
        assert read(ROOT / f'staging/full-mirror-{node}-final-001/status.json')['complete'] is True
        replay = read(ROOT / f'reports/full-raw-replay-{node}-001/summary.json')
        assert replay['passed'] is True and replay['full_result_exact_equal_cells'] == count
    pairing = read(ROOT / 'reports/trace-pairing-final-001.json')
    assert pairing['passed'] is True and pairing['observed_cell_count'] == 93
    def row(node, rate, system='pdblend', repeat=1):
        found = [r for r in rows if r['measurement_host'] == node and
                 float(r['rate_rps']) == rate and r['system'] == system and int(r['repeat']) == repeat]
        assert len(found) == 1
        return found[0]
    def num(r, field):
        return float(r[field])
    lines = [
        '# 14B ShareGPT：三机 SLO 与均匀 rate 扫描结果', '',
        '本轮新增正式测量 **62／62 完成**：A 机 0.5× 共 21 次，C 机 2× 共 41 次；B 机已有 1× 的 31 次原样保留为参考，合计展示 93 次。两档均完成 PDBlend 边界确认，以及 Mixed、DistServe、DynamoLLM、EcoServe 从 0.25 rps 到对应终点的完整网格。正式测量没有工程失败或替代尝试，缺失点为零。', '',
        '新增网格共发出 5,025 次请求，其中 5,023 次完整输出、2 次达到经核实的请求超时；全部进入各自达标率分母。请求内容在系统和同 trace 确认之间配对复用，这不是 5,025 个独立随机样本。', '',
        '每档只使用一个到达种子 701、内容采样种子 20260907，到达窗口 100 秒，步长 0.25 rps。同 rate 的请求内容和原始到达 trace 配对，实际运行配置分别使用 A 的 TTFT < 2.5 s / TPOT < 0.075 s、B 的 < 5 s / < 0.15 s、C 的 < 10 s / < 0.30 s。完整输出且同时满足两项严格阈值才计为达标，分母包含全部请求。', '',
        'PDBlend 的边界如下：', '',
        '| 机器 / SLO | 首次失守前最后一个 ≥90% 的档位 | 首次 <90% 的档位 | 首测达标率 | 同 trace 确认 |',
        '|---|---:|---:|---:|---:|',
    ]
    for node, scale, goodrate, cap in [('A', .5, .75, 1), ('B', 1, 1.25, 1.5), ('C', 2, 1.75, 2)]:
        first, repeat = row(node, cap), row(node, cap, repeat=2)
        lines.append(f"| {node} / {scale:g}×{'（已有参考）' if node == 'B' else ''} | {goodrate:.2f} rps | {cap:.2f} rps | {num(first, 'slo_attainment'):.2%} | {num(repeat, 'slo_attainment'):.2%} |")
    lines += ['',
        '以上是本次离散网格中观测到的服务边界，不是连续负载轴上的精确容量。按预先声明，首次有效结果低于 90% 即固定终点；B 在确认时回升到 91.80% 仍保留 1.50 rps 终点。恰好 90% 继续递增。', '',
        'PDBlend 的边界失分主要来自 TTFT。A 边界两次分别有 13、12 个请求仅 TTFT 超标，全部 81 个请求均完整输出；C 边界两次分别有 30、29 个完整请求仅 TTFT 超标，另各有一个经独立核实的 120 秒请求超时。C 每次仍以全部 164 个请求计分，未把超时重新归类为工程故障。三机全部已测 PDBlend 完整请求中，没有 TPOT 超标。', '',
        '各机在 PDBlend 边界前最后一档的系统比较如下。所有差异都在同机、同 SLO、同 rate 下比较；边界确认单独保留，不与首测取最好值或求平均。', '',
        '| 机器 / rate | 系统 | 达标 / 全部 | 达标率 | 八卡能耗 (kJ) | goodput (req/s) | 每个达标请求 (J) |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    names = {'pdblend': 'PDBlend', 'mixed': 'Mixed', 'distserve': 'DistServe', 'dynamollm': 'DynamoLLM', 'ecoserve': 'EcoServe'}
    for node, rate in [('A', .75), ('B', 1.25), ('C', 1.75)]:
        for system, name in names.items():
            r = row(node, rate, system)
            lines.append(f"| {node} / {rate:.2f} | {name} | {num(r, 'good_requests'):.0f}/{num(r, 'n_expected'):.0f} | {num(r, 'slo_attainment'):.2%} | {num(r, 'energy_j')/1000:.3f} | {num(r, 'goodput_measurement_rps'):.5f} | {num(r, 'energy_per_good_request_j'):.2f} |")
    lines += ['', 'PDBlend 与同机 Mixed 的取舍：', '']
    for node, rate in [('A', .75), ('B', 1.25), ('C', 1.75)]:
        p, b = row(node, rate), row(node, rate, 'mixed')
        lines.append(f"- {node} 机 {rate:.2f} rps：八卡能耗低 {(1-num(p,'energy_j')/num(b,'energy_j')):.2%}，每个达标请求能耗低 {(1-num(p,'energy_per_good_request_j')/num(b,'energy_per_good_request_j')):.2%}；达标率低 {(num(b,'slo_attainment')-num(p,'slo_attainment'))*100:.2f} 个百分点，goodput 低 {(1-num(p,'goodput_measurement_rps')/num(b,'goodput_measurement_rps')):.2%}。")
    lines += ['',
        '节能结果需要与服务质量同时阅读。A 的 Mixed、DistServe、EcoServe 在已测全网格均 100% 达标，PDBlend 从 0.50 rps 起已有失分。基线按 PDBlend 的终点结束，未继续搜索各自容量，因此仍达标的基线容量上限未知。高负载下每个达标请求的能耗下降，也不能使低于 90% 的点成为合格运行点。', '',
        '图表横轴使用数值 rate（trace 生成时的目标 rps），不是等距类别编号；0.25、0.50、0.75、1.00、1.25、1.50、1.75、2.00 rps 的实际请求数分别为 24、48、59、81、102、122、144、164。图中 TTFT/TPOT 是完整请求的均值；达标率逐请求判断。goodput = 达标请求数 / 实测主窗口秒数，主窗口覆盖 100 秒到达、实际排空及测量控制尾段；J/good = 八卡主能耗 / 达标请求数。', '',
        '主测量能耗与部署、资格验证、失败准备操作分别记账。下面仅列互不重叠且有计量证据的窗口，不将与主窗口重叠的外层操作能耗再加一次。', '',
        '| 范围 | 正式测量主窗口小计 (kJ) | 已计量准备窗口小计 (kJ) |',
        '|---|---:|---:|',
    ]
    for node in ['A', 'C', 'B']:
        n = energy['nodes'][node]
        setup = n['setup']['measured_subtotal_j']
        lines.append(f"| {node}{'（旧参考，单列）' if node == 'B' else ''} | {n['primary_energy_subtotal_j']/1000:.3f} | {'未覆盖' if setup is None else f'{setup/1000:.3f}'} |")
    lines += ['',
        '准备阶段的编排错误、A 的旧控制器接口资格失败及修复证据均保留。相应系统的正式测量开始前，分别完成本机 14B 输出、原生排空、实际频率、空闲恢复和八卡功率资格验证；C 的 7B 资格未冒充 14B。准备和操作之间存在未计量间隙，因此完整准备能耗及整个实验总能耗保持未知，不以已知小计冒充总量。', '',
        '所有新增测量点均完成现场原始审计，并在 B 机逐点完整重放测量审计（请求、超时判定、实际配置、功率积分及清理证据），结果与现场审计一致；资格原始记录已镜像并核对哈希，未在 B 机重新运行 GPU 资格测试。A/C 的非权重原始证据镜像完整。模型分片已在机器资格阶段核对，每机约 29.54 GB 的部署权重二进制不重复打包。旧 B 参考的测量核心证据可核验，但一个历史背景声明缺失，详见原始哈希索引；不据此伪造历史完整性。', '',
        '解释范围：三机均为八张 L20，但 C 的卡间拓扑为 PHB，A/B 存在 PIX/NODE/SYS 链路；SLO 倍率与机器绑定。因此不能把跨机边界或能耗差异全部归因于 SLO 放宽。只有一个到达种子，同 trace 确认不是独立种子重复，不报告独立种子置信区间或统计显著性。', '',
        '交付文件：', '',
    ]
    artifacts = [
        (OUT/'observations.csv', '93 次逐次测量表（含旧 B 参考）'),
        (OUT/'boundaries.csv', '边界表'),
        (OUT/'raw-hash-index.csv', '原始证据与哈希索引'),
        (OUT/'completion-acceptance.json', '62 次正式网格完成验收'),
        (energy_path, '能耗去重与分项结账'),
        (ROOT/'reports/A-assessment.md', 'A 机详细分析'),
        (ROOT/'reports/C-assessment.md', 'C 机详细分析'),
        (ROOT/'reports/pdb-boundary-diagnostics.csv', 'PDBlend 请求失分原因'),
        (ROOT/'reports/trace-pairing-final-001.json', '最终 trace、请求内容和实际 SLO 配对验收'),
        (ROOT/'reports/full-raw-replay-A-001/summary.json', 'A 机 21 点完整测量审计重放'),
        (ROOT/'reports/full-raw-replay-C-001/summary.json', 'C 机 41 点完整测量审计重放'),
        (ROOT/'cpu-validation-final-001.json', '83 项实现与边界规则检查'),
        (ROOT/'protocol.json', '冻结的实验声明'),
        (ROOT/'README.md', '实现、执行及恢复说明'),
        (ROOT/'env/ENVIRONMENT.md', '机器、模型和资格证据说明'),
    ]
    for path, label in artifacts:
        assert path.exists(), path
        lines.append('- ' + link(path, label))
    for node, scale in [('A', '0.5'), ('B', '1'), ('C', '2')]:
        path = OUT/f'host-{node}-slo-{scale}-six-panel.png'
        assert path.exists(), path
        lines += ['', f'![{node} 机 SLO {scale}× 六项指标]({path})']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report': str(OUT/'REPORT.md'), 'sha256': hashlib.sha256((OUT/'REPORT.md').read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
