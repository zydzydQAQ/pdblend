"""Verify actual new observations and report every declared failure or absence."""
import argparse
import bisect
from collections import Counter, defaultdict
import csv
import json
import importlib.util
import math
from pathlib import Path
import statistics
import time
import protocol as p
LOADED_SOURCE = Path(__file__).read_bytes()
RAW_METRICS_SOURCE = Path(__file__).with_name('raw_metrics.py')
LOADED_RAW_METRICS = RAW_METRICS_SOURCE.read_bytes()
_raw_spec = importlib.util.spec_from_file_location('_retained_raw_metrics', RAW_METRICS_SOURCE)
raw_metrics = importlib.util.module_from_spec(_raw_spec)
_raw_spec.loader.exec_module(raw_metrics)

METRICS = ('energy_j', 'energy_per_good_request_j', 'slo_attainment', 'gpu_util',
           'ttft_avg_s', 'tpot_avg_s', 'goodput_measurement_rps', 'completion_fraction')

def close(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)

def integrate(rows, start, end, columns):
    times = [float(r['t_s']) for r in rows]
    p.need(len(times) > 1 and times[0] <= start < end <= times[-1]
           and all(b > a for a, b in zip(times, times[1:])), 'power coverage/time order invalid')
    values = [[float(row[c]) for c in columns] for row in rows]
    p.need(all(math.isfinite(v) and v >= 0 for row in values for v in row), 'invalid physical sample')
    def at(t):
        i = min(max(bisect.bisect_right(times, t) - 1, 0), len(times) - 2)
        q = (t - times[i]) / (times[i+1] - times[i])
        return [a + q * (b-a) for a, b in zip(values[i], values[i+1])]
    samples = [(start, at(start))] + [(t, v) for t, v in zip(times, values) if start < t < end] + [(end, at(end))]
    return [sum((t1-t0)*(v0[g]+v1[g])/2 for (t0, v0), (t1, v1) in zip(samples, samples[1:]))
            for g in range(len(columns))]

def audit_raw(summary, directory, cell):
    trace = p.checked(cell['trace'])
    with (directory / 'bench.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    n = len(trace['requests'])
    p.need(len(rows) == n and [r['request_id'] for r in rows] == [str(i) for i in range(n)], 'request identity/denominator differs')
    good = complete = generated = 0
    for row, prescribed in zip(rows, trace['requests']):
        emitted = int(row['generated_tokens'])
        generated += emitted
        success = row['success'].lower() in ('1', 'true') and not row.get('error')
        work = (success and row['token_count_source'] == 'server_usage'
                and row['token_ids_verified'].lower() in ('1', 'true')
                and int(row['input_tokens']) == prescribed['prompt_len'] and emitted == prescribed['output_len'])
        complete += work
        # The original classifier uses strict TTFT/TPOT thresholds.
        good += bool(work and row['ttft_s'] and row['tpot_s']
                     and 0 <= float(row['ttft_s']) < cell['original_point']['slo_ttft_s']
                     and 0 <= float(row['tpot_s']) < cell['original_point']['slo_tpot_s'])
    expected_tokens = sum(r['output_len'] for r in trace['requests'])
    for field, value in dict(n_expected=n, completed_work_requests=complete, good_requests=good,
                              generated_tokens=generated, expected_generated_tokens=expected_tokens).items():
        p.need(summary[field] == value, 'recomputed request metric differs: ' + field)
    p.need(summary['work_complete'] == (complete == n and generated == expected_tokens), 'work completion differs')
    p.need(close(summary['slo_attainment'], good/n), 'recomputed SLO differs')
    start, end = summary['measurement_start_s'], summary['measurement_end_s']
    p.need(close(summary['measurement_duration_s'], end-start) and end-start >= 100, 'measurement denominator differs')
    p.need(close(summary['goodput_measurement_rps'], good/(end-start)), 'goodput denominator differs')
    with (directory / 'power.csv').open() as stream:
        powers = list(csv.DictReader(stream))
    energy = integrate(powers, start, end, [f'gpu{i}_w' for i in range(8)])
    p.need(close(sum(energy), summary['energy_j']), 'eight-GPU raw energy differs')
    util = [v / (end-start) / 100 for v in integrate(powers, start, end, [f'gpu{i}_util_pct' for i in range(8)])]
    p.need(all(close(x, y) for x, y in zip(util, summary['gpu_util_per_gpu']))
           and len(summary['gpu_util_per_gpu']) == 8 and close(statistics.mean(util), summary['gpu_util']),
           'eight-GPU utilization differs')
    return dict(raw_requests_recomputed=True, all_eight_gpu_energy_reintegrated=True,
                primary_energy_j=sum(energy), energy_per_gpu_j=energy, gpu_util_per_gpu=util)

def inspect(cell, checkpoint):
    point = {k: cell['original_point'][k] for k in p.PAIR_FIELDS}
    point.update(cell_id=cell['cell_id'], arm=cell['arm'], repeat=cell['repeat'], stage=cell['stage'],
                 original_cell_id=cell['original_cell_id'], measurement_valid=False,
                 status='unmeasured', work_complete=None, error=None)
    point.update({k: None for k in METRICS})
    if checkpoint is None:
        return point
    try:
        cp = p.read(checkpoint)
        p.need(cp['declaration'] == cell, 'new point declaration changed')
        p.need(cp['row']['cell_id'] == cell['cell_id'] and cp['row']['trace_sha256'] == cell['trace']['sha256'], 'executed workload differs')
        receipt = p.checked({'path': cp['receipt'], 'sha256': cp['receipt_sha256']})
        binding = p.checked({'path': cp['binding'], 'sha256': cp['binding_sha256']})
        p.need(binding['improvement']['arm'] == cell['arm'] and binding['improvement']['repeat'] == cell['repeat'], 'executed improvement arm differs')
        for path, digest in cp['artifacts'].items():
            p.need(p.sha(path) == digest, 'raw checkpoint artifact changed: ' + path)
        summary = receipt['summary']
        if cell['arm'] == 'dynamic':
            p.need(receipt.get('dynamic_inventory_verified') is True, 'dynamic inventory was not verified')
            for path, digest in receipt.get('dynamic_artifacts', {}).items():
                p.need(p.sha(path) == digest, 'dynamic transition raw artifact changed: '+path)
        directory = Path(cp['receipt']).parents[2] / 'cells' / cell['cell_id']
        p.need(p.read(directory / 'summary.json') == summary, 'summary and receipt disagree')
        p.need(receipt['measurement_valid'] is True and summary['measurement_valid'] is True
               and summary['fixed_window_valid'] is True and summary['gpu_count'] == 8
               and summary['power_source_verified'] is True and receipt['clock_restore_complete'] is True
               and receipt['child_stopped'] is True and not receipt['outer_cleanup_errors'], 'measurement/cleanup invalid')
        p.need(all(v.get('complete') is True for v in receipt['restoration'].values()), 'native cleanup incomplete')
        p.need(summary['trace_sha256'] == cell['trace']['sha256'], 'summary trace identity changed')
        proof = audit_raw(summary, directory, cell)
        additional = raw_metrics.audit_additional_metrics(summary, directory)
        proof['additional_metrics'] = additional
        diagnostics_path = directory / 'admission_diagnostics.json'
        if diagnostics_path.exists():
            diagnostics = p.read(diagnostics_path)
            p.need(diagnostics.get('actual_snapshot_only') is True
                   and diagnostics.get('hypothetical_beam_candidates_excluded') is True,
                   'admission diagnostic scope differs')
            timings = list(diagnostics['requests'].values())
            point.update(planning_requests_observed=len(timings),
                first_planning_wait_mean_s=statistics.mean(t['first_planning_wait_s'] for t in timings) if timings else None,
                planning_attempts=sum(t['attempts'] for t in timings),
                planning_total_s=sum(t['planning_total_s'] for t in timings),
                repeat_planning_s=sum(t['repeat_planning_s'] for t in timings),
                planner_reason_observations=dict(Counter({reason:sum(v['count'] for v in diagnostics['reasons']
                    if v['category']==reason) for reason in {v['category'] for v in diagnostics['reasons']}})),
                diagnostic=p.ref(diagnostics_path))
        outer_clocks = Path(cp['receipt']).parent / 'power' / 'clocks.csv'
        if outer_clocks.exists():
            with outer_clocks.open() as stream:
                clocks=list(csv.DictReader(stream))
            start,end=summary['measurement_start_s'],summary['measurement_end_s']
            actual=[v/(end-start) for v in integrate(clocks,start,end,[f'gpu{i}_sm_mhz' for i in range(8)])]
            point.update(actual_sm_mhz_per_gpu=actual,actual_sm_mhz_all8_mean=statistics.mean(actual))
        inventory_path=Path(cp['receipt']).parent/'inventory.final.json'
        if inventory_path.exists():
            inventory=p.read(inventory_path)
            start,end=summary['measurement_start_s'],summary['measurement_end_s']
            changes=[(start,2)]
            for event in inventory['events']:
                if event['kind']=='physical_commit' and start <= event['at_s'] <= end:
                    changes.append((event['at_s'],len(event['live_instances'])))
            changes.sort();changes.append((end,changes[-1][1]))
            point.update(service_instances_max=max(n for _,n in changes),
                service_instances_time_mean=sum((b-a)*n for (a,n),(b,_) in zip(changes,changes[1:]))/(end-start),
                dynamic_policy_commits=sum(e['kind']=='physical_commit' and e.get('scope')=='policy_transition' for e in inventory['events']),
                dynamic_cleanup_commits=sum(e['kind']=='physical_commit' and e.get('scope')=='measurement_cleanup' for e in inventory['events']))
        else:
            point.update(service_instances_max=2,service_instances_time_mean=2.)
        point.update({k: summary.get(k) for k in METRICS if k != 'completion_fraction'})
        point.update(additional['normalized_metrics'])
        point.update({k: summary[k] for k in ('completed_work_requests', 'generated_tokens',
            'expected_generated_tokens', 'n_expected', 'good_requests', 'measurement_duration_s', 'work_complete')})
        point.update(completion_fraction=summary['completed_work_requests']/summary['n_expected'],
            measurement_valid=True, status='measured_complete' if summary['work_complete'] else 'measured_incomplete',
            verification=proof, checkpoint=p.ref(checkpoint), receipt=p.ref(cp['receipt']),
            admission_planning=summary.get('admission_planning'), implementation_id=binding['improvement']['implementation'])
        events = []
        control = directory / 'control.jsonl'
        if control.exists():
            for line in control.open():
                row = json.loads(line)
                if row.get('kind') == 'request_timing' and row.get('queued_s') is not None and row.get('forward_started_s') is not None:
                    events.append(row['forward_started_s'] - row['queued_s'])
        point['pre_forward_wait_median_s'] = statistics.median(events) if events else None
    except (ValueError, KeyError, TypeError, OSError) as exc:
        point.update(status='observed_invalid', error=str(exc), measurement_valid=False)
    return point

def csv_write(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})

def figures(out, originals, points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    styles = {'pdblend': ('#999999', 'o'), 'mixed': ('#D99117', 's'), 'distserve': ('#278568', '^'),
              'dynamollm': ('#9865B3', 'D'), 'ecoserve': ('#687984', 'v'),
              'fixed2': ('#D84A40', 'P'), 'dynamic': ('#1769C2', 'o')}
    config = [('slo_attainment', 'Joint SLO attainment (%)', 100.),
              ('energy_j', 'Total eight-GPU energy (kJ)', .001),
              ('energy_per_good_request_j', 'Energy per SLO-qualified request (J/request)', 1.),
              ('gpu_util', 'Mean eight-GPU utilization (%)', 100.),
              ('ttft_avg_s', 'Mean TTFT (s)', 1.),
              ('tpot_avg_s', 'Mean TPOT (s)', 1.),
              ('goodput_measurement_rps', 'Goodput over measured energy window (req/s)', 1.),
              ('completion_fraction', 'Prescribed work completion (%)', 100.)]
    counts = {}
    with PdfPages(out / 'all-eight-metrics.pdf') as combined, PdfPages(out / 'slo-and-energy.pdf') as priority:
        for metric, ylabel, scale in config:
            count = 0
            fig, axes = plt.subplots(3, 3, figsize=(16, 11), squeeze=False)
            for i, model in enumerate(p.MODELS):
                for j, dataset in enumerate(p.DATASETS):
                    ax = axes[i][j]
                    for system in ('pdblend', *p.BASELINES):
                        rows = sorted((x for x in originals if x['model'] == model and x['dataset'] == dataset and x['system'] == system), key=lambda x: x['rate_rps'])
                        color, marker = styles[system]
                        ax.plot([x['rate_rps'] for x in rows], [x[metric]*scale for x in rows],
                                color=color, marker=marker, markersize=3, linewidth=1, alpha=.7)
                        incomplete = [x for x in rows if x.get('work_complete') is not True]
                        ax.scatter([x['rate_rps'] for x in incomplete], [x[metric]*scale for x in incomplete],
                                   marker='x', color='black', s=45, linewidths=1, zorder=6)
                    for arm in ('fixed2', 'dynamic'):
                        for repeat in (1, 2):
                            rows = sorted((x for x in points if x['model'] == model and x['dataset'] == dataset
                                           and x['arm'] == arm and x['repeat'] == repeat and x['measurement_valid']), key=lambda x:x['rate_rps'])
                            color, marker = styles[arm]
                            ys = [x[metric]*scale if isinstance(x.get(metric),(int,float))
                                  and math.isfinite(x[metric]) else float('nan') for x in rows]
                            ax.plot([x['rate_rps'] for x in rows], ys,
                                    color=color, marker=marker, linewidth=2, markersize=5,
                                    linestyle='-' if repeat == 1 else '--')
                            incomplete = [(x,y) for x,y in zip(rows,ys) if not x['work_complete'] and math.isfinite(y)]
                            ax.scatter([x['rate_rps'] for x,y in incomplete], [y for x,y in incomplete],
                                       marker='x', color='black', s=65, linewidths=1.5, zorder=7)
                            for x,y in zip(rows,ys):
                                if not math.isfinite(y):
                                    ax.annotate('undefined', (x['rate_rps'], .04),
                                                xycoords=('data','axes fraction'), rotation=90, fontsize=6, color=color)
                            count += len(rows)
                    failures = [x for x in points if x['model']==model and x['dataset']==dataset
                                and x['status'] in ('technical_failure','observed_invalid')]
                    for rate in sorted({x['rate_rps'] for x in failures}):
                        ax.annotate('technical failure', (rate,.02), xycoords=('data','axes fraction'),
                                    rotation=90, fontsize=6, color='black')
                    if metric == 'slo_attainment':
                        ax.axhline(90, color='black', linestyle=':', linewidth=.8)
                    if metric in ('slo_attainment','completion_fraction','gpu_util'):
                        ax.set_ylim(-2, 102)
                    ax.set_title(f'{model.upper()} / {dataset}')
                    ax.set_xlabel('Rate (req/s)')
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=.18)
            names = {'pdblend': 'Original PDBlend', 'fixed2': 'Improved fixed2', 'dynamic': 'Improved dynamic',
                     'mixed': 'Mixed', 'distserve': 'DistServe', 'dynamollm': 'DynamoLLM*', 'ecoserve': 'EcoServe'}
            handles = [Line2D([], [], color=color, marker=marker, label=names[name]) for name, (color, marker) in styles.items()]
            fig.legend(handles=handles, loc='upper center', ncol=7)
            fig.suptitle(ylabel + ' — each measured repeat retained', y=.96)
            fig.text(.5, .008, '*7B/32B DynamoLLM resident. Dashed: repeat 2. Black x: incomplete work. Baselines: snapshot-006.', ha='center')
            fig.tight_layout(rect=(0, .025, 1, .925))
            fig.savefig(out / (metric + '.png'), dpi=180)
            fig.savefig(out / (metric + '.pdf'))
            combined.savefig(fig)
            if metric in ('slo_attainment','energy_j'):
                priority.savefig(fig)
            counts[metric] = count
            plt.close(fig)
    return counts

def execution_statuses(roots):
    """Collapse read-only snapshots of one invocation, never separate attempts."""
    snapshots = defaultdict(list)
    paths = {path.resolve() for root in roots for path in root.glob('**/status.json')}
    for path in sorted(paths):
        try:
            value = p.read(path)
        except (OSError, ValueError):
            continue
        if value.get('stage') not in ('screen_fixed2', 'screen_dynamic', 'confirm_dynamic'):
            continue
        p.need(type(value.get('pid')) is int and type(value.get('started_s')) in (int, float),
               'execution snapshot lacks invocation identity: '+str(path))
        identity = (value['model'], value['stage'], value['pid'], value['started_s'])
        snapshots[identity].append((path, value))
    selected = []
    for identity, values in sorted(snapshots.items()):
        values.sort(key=lambda item: (item[1]['updated_s'], str(item[0])))
        attempted, completed, failures = set(), set(), {}
        for path, value in values:
            current_attempted = set(value.get('attempted', []))
            current_completed = set(value.get('completed', []))
            current_failures = {v['cell_id']:v['error'] for v in value.get('failed', [])}
            p.need(attempted <= current_attempted and completed <= current_completed
                   and all(current_failures.get(k) == v for k,v in failures.items()),
                   'execution snapshot erased earlier work or failure: '+str(path))
            attempted, completed, failures = current_attempted, current_completed, current_failures
        selected.append(values[-1])
    return selected

def generate(roots, out, make_figures=True):
    p.need(not out.exists(), 'new report snapshot required')
    cells = p.read(p.ROOT / 'work-declaration.json')['cells']
    originals = p.original_points()
    original_by_id = {v['cell_id']: v for v in originals}
    checkpoints = {}
    for root in roots:
        for cp in root.glob('**/results/checkpoints/slo-improve-*.json'):
            p.need(cp.stem not in checkpoints, 'duplicate physical observation for one declaration: ' + cp.stem)
            checkpoints[cp.stem] = cp
    progress = {}
    setup_costs = []
    selected_statuses = execution_statuses(roots)
    for status_path, value in selected_statuses:
        ordinary = value.get('ordinary', {})
        if ordinary.get('setup_energy_j') is not None:
            setup_costs.append(dict(model=value['model'], stage=value['stage'], status_path=str(status_path),
                setup_energy_j=ordinary['setup_energy_j'], passed=ordinary.get('passed') is True))
        failures = {v['cell_id']:v['error'] for v in value.get('failed', [])}
        for cid in value.get('attempted', []):
            p.need(cid not in progress, 'duplicate attempted declaration across physical invocations: '+cid)
            progress[cid] = dict(status=('technical_failure' if cid in failures else
                'running' if cid == value.get('current_cell') else 'checkpoint_missing'),
                error=failures.get(cid), execution_status=str(status_path))
    points = [inspect(c, checkpoints.get(c['cell_id'])) for c in cells]
    for point in points:
        if point['status'] == 'unmeasured' and point['cell_id'] in progress:
            point.update(progress[point['cell_id']])
    pairs = []
    for cell, point in zip(cells, points):
        for system, bid in cell['baseline_cell_ids'].items():
            pairs.append(dict(cell_id=cell['cell_id'], model=cell['model'], dataset=cell['dataset'],
                rate_rps=cell['original_point']['rate_rps'], arm=cell['arm'], repeat=cell['repeat'],
                status=point['status'], **p.verdict(point, original_by_id[bid])))
    out.mkdir(parents=True)
    (out / 'raw_metrics.py').write_bytes(LOADED_RAW_METRICS)
    p.write(out / 'execution-snapshots.json', [dict(source_path=str(path), value=value)
            for path, value in selected_statuses], exclusive=True)
    (out / 'generator.py').write_bytes(LOADED_SOURCE)
    (out / 'protocol.py').write_text(Path(p.__file__).read_text().replace("ROOT = Path(__file__).resolve().parent", "ROOT = Path("+repr(str(p.ROOT))+")"))
    csv_write(out / 'points.csv', points); csv_write(out / 'paired-baselines.csv', pairs)
    status = dict(schema=1, observed_s=time.time(), declared=len(cells),
        status_counts=dict(Counter(x['status'] for x in points)), verified=sum(x['measurement_valid'] for x in points),
        comparisons_total=len(pairs), comparisons_measured=sum(x['measurement_valid'] for x in pairs),
        comparisons_passed=sum(x['passed'] for x in pairs),
        original_results_unchanged=True, baseline_rerun=False,
        results=points, pairs=pairs, separately_measured_setup_costs=setup_costs,
        execution_failures=[dict(source_path=str(path), model=value['model'], stage=value['stage'],
            pid=value['pid'], attempted=len(value.get('attempted',[])), error=value.get('error'),
            phase=value['phase']) for path,value in selected_statuses if value.get('error')])
    p.write(out / 'results.json', status, exclusive=True)
    by_cell = defaultdict(list)
    for pair in pairs:
        by_cell[pair['cell_id']].append(pair)
    lines = ['# 三模型改进实测进度', '',
        f"已核验 {status['verified']}/{len(cells)} 个预声明新测量；其余缺失或无效测量没有标为通过。", '',
        '本报告仅汇入这一版本的实测结果。旧主实验与此前开发版本分别封存，不拼接不同版本的最佳点。', '',
        '| 模型 | 已核验测量 | 完整工作 | 能耗通过配对 | SLO通过配对 | 同时通过配对 |',
        '|---|---:|---:|---:|---:|---:|']
    for model in p.MODELS:
        ps = [x for x in pairs if x['model'] == model and x['measurement_valid']]
        observations = [x for x in points if x['model']==model and x['measurement_valid']]
        lines.append(f"| {model.upper()} | {len(observations)} | {sum(x['work_complete'] for x in observations)}/{len(observations)} | "
                     f"{sum(x['energy_pass'] for x in ps)}/{len(ps)} | {sum(x['slo_pass'] for x in ps)}/{len(ps)} | "
                     f"{sum(x['passed'] for x in ps)}/{len(ps)} |")
    lines += ['', '判定要求规定工作全部完成、总八卡能耗不高于 baseline、SLO ≥ min(90%, baseline SLO)。没有增加数值容差。', '',
        'baseline 使用 snapshot-006 原测量。seed=701 的重复不构成独立到达种子的统计显著性证明。动态转换与排空留在主测窗口，重叠外层能耗不相加；独立部署准备成本另列。', '',
        '下表逐次保留全部有效测量，包括低 SLO 和未完成请求。“通过数”是对四个 baseline 分别验收，不能解释为重复试验的成功率。', '',
        '| 模型 | 策略 | 数据集 | rate | 次数 | SLO | 八卡能耗 kJ | 完成率 | 通过数 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for point in points:
        if point['measurement_valid']:
            lines.append(f"| {point['model'].upper()} | {point['arm']} | {point['dataset']} | {point['rate_rps']:g} | "
                f"{point['repeat']} | {100*point['slo_attainment']:.2f}% | {point['energy_j']/1000:.3f} | "
                f"{100*point['completion_fraction']:.2f}% | {sum(x['passed'] for x in by_cell[point['cell_id']])}/4 |")
    if any(x['measurement_valid'] and not x['passed'] for x in pairs):
        lines += ['', '**当前仍有未通过点，证据不支持新版本在全部已测 rate 上同时满足四个 baseline 的能耗和 SLO 要求。**']
    lines += ['', '画像、排队、频率保护是本轮联合修改，不能仅凭联合版本结果把收益归于一个机制。实际频率、排队耗时、完整输出、实例数及转换记录保留在逐点表和原始结果中。', '',
        '八卡平均利用率只描述资源状态，不直接判为越高越好。图中黑色叉号表示未完成工作；没有删除这些点或截去极端延迟。', '',
        '未完成清单及每个失败原因见 points.csv；逐 baseline 判定见 paired-baselines.csv。尚未测量的动态实例点没有被当作通过，900 秒开发轨迹和校准不混入本表。']
    if status['execution_failures']:
        lines += ['', '另保留下列执行阶段失败；尚未发起测点的阶段失败不计作已测请求：', '']
        for item in status['execution_failures']:
            lines.append(f"- {item['model'].upper()} / {item['stage']}：已尝试 {item['attempted']} 个测点；{item['error']}")
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    if make_figures and status['verified']:
        status['plotted_new_observations_per_figure'] = figures(out, originals, points)
        p.write(out / 'figure-validation.json', dict(linear_actual_rate=True, smoothing=False,
            every_repeat_retained=True, expected_new_observations_per_figure=status['verified'],
            actual_new_observations=status['plotted_new_observations_per_figure'],
            count_passed=all(n==status['verified'] for n in status['plotted_new_observations_per_figure'].values())), exclusive=True)
    p.write(out / 'manifest.json', dict(declaration=p.ref(p.ROOT / 'work-declaration.json'),
        files={f.name: p.sha(f) for f in out.iterdir() if f.is_file()},
        original_snapshot=p.PINNED, code=p.ref(out / 'generator.py')), exclusive=True)
    return {k:v for k,v in status.items() if k not in ('results','pairs')}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', type=Path, nargs='+', default=[p.ROOT / x for x in ('A','B','C')])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    print(generate(args.roots, args.out, not args.no_figures))
