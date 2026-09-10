"""Read-only interpretation of already audited, completed rate grids."""
import csv
import io
import math
from decimal import Decimal

DATASETS = ('alpaca', 'sharegpt', 'longbench')
METRICS = ('energy_j', 'slo_attainment', 'ttft_avg_s', 'tpot_avg_s',
           'completed_work_throughput_rps', 'generated_token_throughput_tps', 'gpu_util')


def require(value, message):
    if not value:
        raise ValueError(message)


def number(row, field):
    value = float(row[field])
    require(math.isfinite(value), 'nonfinite metric: ' + field)
    return value


def build_analysis(result, summary_rows, models=('7b', '14b', '32b')):
    """Return Markdown and CSV; caller supplies a consistent verified snapshot."""
    expected = {(m, d) for m in models for d in DATASETS}
    groups = {(g['model'], g['dataset']): g for g in result['groups']
              if g['model'] in models}
    require(set(groups) == expected, 'missing or unexpected group')
    require(not [e for e in result['metric_audit_errors'] if e['model'] in models],
            'unresolved metric audit')
    indexed = {}
    for row in summary_rows:
        if row['model'] not in models:
            continue
        key = (row['model'], row['dataset'], row['system'], Decimal(str(row['rate_rps'])))
        require(key not in indexed, 'duplicate scientific coordinate')
        indexed[key] = row
    table = []
    for model in models:
        for dataset in DATASETS:
            g = groups[model, dataset]
            require(g['complete'] is True and g['decision']['phase'] == 'complete',
                    'unfinished group: ' + model + '-' + dataset)
            step = Decimal(str(g['rate_step_rps']))
            cap = Decimal(str(g['decision']['cap_rate_rps_decimal']))
            last = cap - step
            require(step > 0 and last >= step and cap % step == 0, 'invalid grid boundary')
            pdb = [o for o in result['observations'] if
                   (o['model'], o['dataset'], o['system']) == (model, dataset, 'pdblend')
                   and o.get('measurement_purpose', 'normal') == 'normal'
                   and Decimal(str(o['rate_rps'])) <= cap]
            expected_rates = {step * n for n in range(1, int(cap / step) + 1)}
            require({Decimal(str(o['rate_rps'])) for o in pdb} == expected_rates,
                    'PDBlend grid missing a lower point')
            for o in pdb:
                require(o['measurement_valid'] is True and o['work_complete'] is True,
                        'invalid or incomplete boundary observation')
                require(o['measurement_host'] == g['node'], 'foreign-host observation')
                count = o.get('n_requests', o.get('offered_requests', o['completed_work_requests']))
                require(count > 0 and count == o['completed_work_requests'], 'full-work count mismatch')
                require(math.isclose(o['slo_attainment'], o['good_requests'] / count,
                                     rel_tol=0, abs_tol=1e-12), 'SLO denominator mismatch')
                if Decimal(str(o['rate_rps'])) < cap:
                    require(o['slo_attainment'] >= .9, 'earlier single crossing')
            terminal = [o for o in pdb if Decimal(str(o['rate_rps'])) == cap]
            trigger_id = g['decision']['decision']['cap_trigger_cell_id']
            trigger = [o for o in terminal if o['cell_id'] == trigger_id]
            require(len(terminal) >= 2 and len(trigger) == 1 and trigger[0]['slo_attainment'] < .9,
                    'missing single crossing or confirmation')
            for rate in expected_rates:
                for system in ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve'):
                    row = indexed[model, dataset, system, rate]
                    require(row['measurement_host'] == g['node'], 'foreign-host aggregate')
                    for metric in METRICS:
                        number(row, metric + '_mean')
                        require(int(row[metric + '_n']) >= 1, 'missing measured metric')
            p, m = [indexed[model, dataset, system, last] for system in ('pdblend', 'mixed')]
            require(number(m, 'energy_j_mean') > 0 and
                    number(m, 'generated_token_throughput_tps_mean') > 0, 'invalid comparator')
            row = dict(model=model, dataset=dataset, measurement_host=g['node'],
                       step_rps=str(step), last_non_crossing_rps=str(last),
                       first_crossing_rps=str(cap), trigger_cell_id=trigger_id,
                       trigger_slo_percent=100 * trigger[0]['slo_attainment'],
                       boundary_single_slo_percent=';'.join(
                           f"{100 * o['slo_attainment']:.12g}" for o in
                           sorted(terminal, key=lambda o: (o.get('repeat', 0), o['cell_id']))),
                       energy_reduction_vs_mixed_percent=100 * (
                           1 - number(p, 'energy_j_mean') / number(m, 'energy_j_mean')),
                       token_throughput_change_vs_mixed_percent=100 * (
                           number(p, 'generated_token_throughput_tps_mean') /
                           number(m, 'generated_token_throughput_tps_mean') - 1))
            for system, values in (('pdblend', p), ('mixed', m)):
                for metric in METRICS:
                    row[system + '_' + metric + '_mean'] = number(values, metric + '_mean')
            table.append(row)
    text = ['# 固定步长 Rate Scale 实测结论', '',
            '以下比较取各组首次越界前的最后一个已测网格点；该点及更低网格的每次有效 PDBlend 测量均达到 90%。终点保留首次有效完整测量低于 90% 的坐标，确认复测高于 90% 也不撤销它。', '',
            '| 模型 | 数据集 | 越界前 rps | 终点 rps | 终点各次 SLO | PDBlend / Mixed SLO | 相对 Mixed 能耗降低 | token/s 变化 |',
            '|---|---|---:|---:|---|---|---:|---:|']
    for row in table:
        singles = ' / '.join(f'{float(x):.2f}%' for x in row['boundary_single_slo_percent'].split(';'))
        text.append(f"| {row['model']} | {row['dataset']} | {row['last_non_crossing_rps']} | "
                    f"{row['first_crossing_rps']} | {singles} | "
                    f"{100 * row['pdblend_slo_attainment_mean']:.2f}% / {100 * row['mixed_slo_attainment_mean']:.2f}% | "
                    f"{row['energy_reduction_vs_mixed_percent']:.2f}% | "
                    f"{row['token_throughput_change_vs_mixed_percent']:+.2f}% |")
    text += ['', '能耗降低 = 1 − PDBlend / Mixed；token/s 变化 = PDBlend / Mixed − 1。比较使用同物理主机、同 rate、同 trace 和相同 SLO 门槛的实测均值；各系统实际 SLO attainment 可不同。完整五系统六指标见各组曲线及 CSV，吞吐同时保留 request/s 和 token/s。', '',
             '这些结论针对本次固定 trace。正常新增点测一次，已有重复显示均值与实测范围；范围不是独立采样置信区间。最后通过的网格点与首次越界点之间没有插值测量。', '',
             '32B LongBench 使用 0.05 rps 步长；0.30 rps 首次测量低于 90%，即使两次均值超过 90% 仍停止。能耗包含完整测量及 drain / 转换尾部，准备与故障操作另列且按时间窗口去重。指标补采仅补缺失值，保留历史 SLO。']
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(table)
    return '\n'.join(text) + '\n', output.getvalue()
