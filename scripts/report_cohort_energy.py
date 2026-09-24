#!/usr/bin/env python3
"""Rebuild a cohort-energy comparison from a frozen snapshot, without mutating runs.

Run with the project's plotting environment; --input-dir must already contain
compare-snapshot.csv, snapshot.json and historical-selected-points-180.csv.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path

from cohort_energy_failures import analyze_failures

BASELINES = ['mixed', 'ecoserve', 'distserve', 'dynamollm']
NAMES = dict(mixed='Mixed', ecoserve='EcoServe', distserve='DistServe',
             dynamollm='DynamoLLM', pd_previous='PDblend 上一完整轮', pd_current='PDblend 新轮')
MODELS = ['7B', '14B', '32B']
DATASETS = ['alpaca', 'sharegpt', 'longbench']


def number(value):
    if value is None or value == '':
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite input')
    return value


def close(a, b, *, abs_tol=1e-6):
    if a is None or b is None:
        return a is b
    # Epoch seconds must not receive a relative tolerance (~0.18 s at this epoch).
    return math.isclose(float(a), float(b), rel_tol=0., abs_tol=abs_tol)


def truth(value):
    return value in (True, 'true', 'True')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (list, dict))
                             else r.get(k) for k in fields})


def cell(row):
    return row['model'], row['dataset'], row['rate_scale']


def select_rows(root, raw):
    observed = [r for r in raw if r.get('receipt_path') and r.get('offered_requests')]
    selection_path = root / 'selection.json'
    if not selection_path.exists():
        times = {r['receipt_path']: Path(r['receipt_path']).stat().st_mtime for r in observed}
        groups = defaultdict(list)
        for r in observed:
            if r['system'] == 'pdblend':
                groups[r['revision']].append(r)
        complete = [rev for rev, rr in groups.items() if len({r['point_id'] for r in rr}) == 36]
        assert complete, 'No complete PDblend revision'
        newest = lambda rev: max(times[r['receipt_path']] for r in groups[rev])
        previous, current = max(complete, key=newest), max(groups, key=newest)
        selected, attempts = [], []
        for series in BASELINES + ['pd_previous', 'pd_current']:
            if series == 'pd_current' and previous == current:
                continue
            candidates = [r for r in observed if r['system'] == series] if series in BASELINES else groups[
                previous if series == 'pd_previous' else current]
            per_point = defaultdict(list)
            for r in candidates:
                per_point[r['point_id']].append(r)
            for point_id, rr in per_point.items():
                chosen = max(rr, key=lambda r: (times[r['receipt_path']], r['receipt_path']))
                selected.append(dict(series=series, point_id=point_id, receipt_path=chosen['receipt_path'],
                                     receipt_sha256=chosen['receipt_sha256'], revision=chosen['revision']))
                for r in rr:
                    attempts.append(dict(series=series, point_id=point_id, receipt_path=r['receipt_path'],
                                         receipt_mtime=times[r['receipt_path']], selected=r is chosen))
        json_write(selection_path, dict(rule='Latest receipt timestamp per baseline point; PDblend revisions remain separate; never select by performance.',
                   prior_revision=previous, current_revision=current, selected=selected, attempts=attempts,
                   source_rows=len(raw), observed_rows=len(observed), unmeasured_rows=len(raw)-len(observed)))
    selection = json.loads(selection_path.read_text())
    by_receipt = {r['receipt_path']: r for r in observed}
    return selection, [(s, by_receipt[s['receipt_path']]) for s in selection['selected']]


def analyze_point(selection, row):
    path = Path(row['receipt_path'])
    assert sha(path) == row['receipt_sha256'] == selection['receipt_sha256'], path
    receipt = json.loads(path.read_text())
    artifacts = receipt['artifacts']
    sources = {str(path): row['receipt_sha256']}

    def read(relative):
        source = path.parent / relative
        digest = sha(source)
        assert artifacts.get(relative) == digest, (source, 'Receipt artifact hash mismatch')
        sources[str(source)] = digest
        return json.loads(source.read_text())

    point = read('point.json')
    result = read('result.json')
    assert result == receipt['result'], path
    meter = read('run/comparison-metering.json')
    requests = read('run/comparison-requests.json')
    drain = read('run/native-drain.json')
    native = read('run/native-result.json')
    metrics = result['metrics']
    offered = int(metrics['offered_requests'])
    assert len(requests) == len({r['idx'] for r in requests}) == offered
    successful = sum(r['successful'] is True for r in requests)
    good = sum(r['joint_slo'] is True for r in requests)
    assert successful == int(metrics['successful_requests']) == int(row['successful_requests'])
    assert good == int(metrics['joint_slo_requests']) == int(row['joint_slo_requests'])
    assert offered == int(row['offered_requests'])
    for r in requests:
        expected = bool(r['successful'] and r['ttft_s'] <= metrics['slo_ttft_s'] and r['tpot_s'] <= metrics['slo_tpot_s'])
        assert expected == r['joint_slo'], (path, r['idx'])
    assert close(good / offered, metrics['joint_slo_rate'])
    assert close(good / offered, number(row['joint_slo_rate']))

    start, boundary, end = (meter[k] for k in ('service_start_s', 'service_end_s', 'tail_end_s'))
    assert close(start, metrics['service_started_s'])
    assert close(boundary-start, number(row['duration_s']))
    assert end >= boundary
    assert close(meter['service']['start_s'], start) and close(meter['service']['end_s'], boundary)
    assert close(meter['tail']['start_s'], boundary) and close(meter['tail']['end_s'], end)
    assert meter['gpu_count'] == len(set(meter['gpu_uuids'])) == 8
    assert meter['gpu_uuid_binding_verified'] is True
    horizon = number(metrics.get('request_elapsed_s'))
    assert horizon is None or start + horizon <= end + 1e-6, (path, 'Request exceeds metered horizon')
    for field in ('finished_s', 'request_finished_s', 'service_finished_s', 'requests_done_s'):
        stamp = number(native.get(field))
        assert stamp is None or stamp <= end + 1e-6, (path, field, 'Native completion exceeds metering')
    if drain.get('tail_end_s') is not None:
        assert close(drain['tail_end_s'], end)
    energy = {}
    for phase in ('service', 'tail'):
        item = meter[phase]
        assert close(item['duration_s'], item['end_s'] - item['start_s'])
        power = item['power']
        per_gpu = power['per_gpu']
        assert set(per_gpu) == set(meter['gpu_uuids'])
        value = number(meter[f'energy_{phase}_j'])
        assert close(value, number(power['energy_j']))
        assert close(value, number(row.get(f'energy_{phase}_j')))
        if value is not None:
            assert all(g['status'] == 'complete' and close(g['coverage_fraction'], 1.) for g in per_gpu.values())
            assert close(sum(g['integral'] for g in per_gpu.values()), value)
        energy[phase] = value
    total = None if None in energy.values() else sum(energy.values())
    assert close(total, number(meter['energy_service_tail_j']))
    assert close(total, number(row.get('energy_service_tail_j')))
    assert close(total, number(metrics.get('energy_service_tail_j')))
    if total is not None:
        assert meter['energy_comparable'] is True
        assert meter['power_source_verified'] is True and not meter['power_error_affects_window']

    failures = analyze_failures(path.parent, requests)
    assert sum(failures['counts'].values()) == offered - successful, path
    for source in failures['source_paths']:
        source = Path(source)
        digest = sha(source)
        relative = str(source.relative_to(path.parent))
        assert artifacts.get(relative) == digest, (source, 'Unbound failure evidence')
        sources[str(source)] = digest
    unresolved = int(metrics['unresolved_requests'])
    all_success = successful == offered and unresolved == 0
    ttft, tpot = number(metrics.get('ttft_p99_s')), number(metrics.get('tpot_p99_s'))
    slo_pass = bool(all_success and good/offered >= .9 and ttft is not None and tpot is not None
                    and ttft <= metrics['slo_ttft_s'] and tpot <= metrics['slo_tpot_s'])
    assert slo_pass == truth(row['slo_pass']) == metrics['slo_pass']
    cleanup = receipt.get('cleanup_passed') is True and drain.get('passed') is True
    identity = {k: row[k] for k in ('trace_sha256', 'measurement_protocol_version', 'model_hash', 'tokenizer_hash',
                                   'image_digest', 'runtime_source_sha256', 'measurement_source_sha256')}
    assert all(identity.values()), (path, 'Missing comparison identity')
    identity.update(seed=int(row['seed']), duration_s=number(row['duration_s']),
                    slo_ttft_s=metrics['slo_ttft_s'], slo_tpot_s=metrics['slo_tpot_s'],
                    gpu_uuids=sorted(meter['gpu_uuids']), offered_requests=offered, offered_rps=number(row['offered_rps']))
    out = dict(series=selection['series'], system=row['system'], model=row['model_id'].split('-')[1],
               dataset=row['dataset'], rate_scale=number(row['rate_scale']), offered_rps=number(row['offered_rps']),
               offered_requests=offered, successful_requests=successful, failed_requests=offered-successful,
               joint_slo_requests=good, slo_attainment_pct=100*good/offered, success_pct=100*successful/offered,
               all_requests_successful=all_success,
               request_completion_status='all_successful' if all_success else ('unresolved' if unresolved else 'requests_failed_or_cancelled'),
               unresolved_requests=unresolved, timeout_requests=failures['timeout_requests'],
               cancelled_requests=failures['cancelled_requests'], rejected_requests=failures['rejected_requests'],
               original_timeout_requests=metrics['timeout_requests'], failure_counts=failures['counts'],
               output_tokens=metrics.get('output_tokens'), good_output_tokens=metrics.get('good_output_tokens'),
               window_delivered_tokens=metrics.get('window_delivered_tokens'), tail_delivered_tokens=metrics.get('tail_delivered_tokens'),
               pending_at_window_end_requests=metrics.get('pending_at_window_end_requests'),
               window_successful_requests=metrics.get('window_successful_requests'),
               ttft_p99_s=ttft, tpot_p99_s=tpot, slo_ttft_s=metrics['slo_ttft_s'], slo_tpot_s=metrics['slo_tpot_s'],
               slo_pass=slo_pass, energy_rank_eligible=slo_pass and total is not None and cleanup,
               energy_measurement_complete=total is not None, cleanup_passed=cleanup,
               service_energy_kj=None if energy['service'] is None else energy['service']/1000,
               tail_energy_kj=None if energy['tail'] is None else energy['tail']/1000,
               total_energy_kj=None if total is None else total/1000,
               energy_scope='eight_gpu_boards_service_plus_tail', service_start_s=start, service_end_s=boundary,
               tail_end_s=end, tail_s=end-boundary, request_elapsed_s=horizon,
               service_power_coverage=meter['service']['power']['coverage_fraction'],
               tail_power_coverage=meter['tail']['power']['coverage_fraction'],
               service_max_power_gap_s=meter['service']['power']['max_gap_s'],
               tail_max_power_gap_s=meter['tail']['power']['max_gap_s'],
               missing_energy_reason=';'.join(f'{p}_energy_missing' for p in energy if energy[p] is None),
               revision=row['revision'], point_id=row['point_id'], receipt_path=str(path), receipt_sha256=row['receipt_sha256'],
               formal_eligible=truth(row.get('formal_eligible')), evidence_valid=truth(row.get('evidence_valid')),
               common_clock_evidence=row.get('common_clock_evidence'), comparison_identity=identity)
    details = [dict(series=out['series'], point_id=out['point_id'], revision=out['revision'],
                    receipt_path=str(path), **item) for item in failures['failure_details']]
    provenance = dict(point_id=out['point_id'], series=out['series'], revision=out['revision'],
                      receipt_path=str(path), source_hashes=sources, artifacts_bound_to_receipt=True,
                      failure_summary_semantics=failures.get('summary_semantics'),
                      observation_scope=result.get('observation_scope'), profile_qualified=result.get('profile_qualified'),
                      qualification_missing_gates=result.get('missing_gates'), topology=point.get('topology'))
    return out, details, provenance


def summarize(rows):
    n = sum(r['offered_requests'] for r in rows)
    complete = [r for r in rows if r['energy_measurement_complete']]
    counts = Counter()
    for r in rows:
        counts.update(r['failure_counts'])
    return dict(points=len(rows), offered_requests=n, successful_requests=sum(r['successful_requests'] for r in rows),
                success_pct=100*sum(r['successful_requests'] for r in rows)/n,
                attainment_pct=100*sum(r['joint_slo_requests'] for r in rows)/n,
                all_success_points=sum(r['all_requests_successful'] for r in rows),
                slo_pass_points=sum(r['slo_pass'] for r in rows),
                energy_complete_points=len(complete), energy_missing_points=len(rows)-len(complete),
                available_total_energy_kj=sum(r['total_energy_kj'] for r in complete),
                cancelled_requests=sum(r['cancelled_requests'] for r in rows),
                timeout_requests=sum(r['timeout_requests'] for r in rows), rejected_requests=sum(r['rejected_requests'] for r in rows),
                unresolved_requests=sum(r['unresolved_requests'] for r in rows), failure_counts=dict(counts))


def compare_pairs(points):
    by_cell = {(cell(r), r['series']): r for r in points}
    pairs, best = [], []
    for p in points:
        if p['system'] != 'pdblend':
            continue
        eligible = []
        for system in BASELINES:
            b = by_cell[cell(p), system]
            assert p['comparison_identity'] == b['comparison_identity'], (p['point_id'], b['point_id'])
            available = p['total_energy_kj'] is not None and b['total_energy_kj'] is not None
            ps, bs = p['service_energy_kj'], b['service_energy_kj']
            saving = 100*(1-p['total_energy_kj']/b['total_energy_kj']) if available else None
            service_saving = 100*(1-ps/bs) if ps is not None and bs is not None else None
            pairs.append(dict(pd_series=p['series'], pd_revision=p['revision'], model=p['model'], dataset=p['dataset'],
                              rate_scale=p['rate_scale'], offered_rps=p['offered_rps'], baseline=system,
                              total_energy_available=available, both_slo_pass=p['slo_pass'] and b['slo_pass'],
                              feasible_pair=p['energy_rank_eligible'] and b['energy_rank_eligible'],
                              pd_total_kj=p['total_energy_kj'], baseline_total_kj=b['total_energy_kj'],
                              saving_pct=saving, service_only_saving_pct=service_saving,
                              tail_reverses_saving=bool(saving is not None and service_saving is not None and saving*service_saving<0),
                              pd_receipt=p['receipt_path'], baseline_receipt=b['receipt_path']))
            if b['energy_rank_eligible']:
                eligible.append(b)
        winner = min(eligible, key=lambda b: (b['total_energy_kj'], b['system'])) if eligible else None
        best.append(dict(pd_series=p['series'], pd_revision=p['revision'], model=p['model'], dataset=p['dataset'],
                         rate_scale=p['rate_scale'], offered_rps=p['offered_rps'], pd_slo_pass=p['slo_pass'],
                         pd_energy_rank_eligible=p['energy_rank_eligible'], pd_total_kj=p['total_energy_kj'],
                         best_available_baseline=None if winner is None else winner['system'],
                         baseline_total_kj=None if winner is None else winner['total_energy_kj'],
                         pd_excess_pct=(100*(p['total_energy_kj']/winner['total_energy_kj']-1)
                                        if winner and p['energy_rank_eligible'] else None),
                         feasible_baseline_count=len(eligible),
                         baseline_energy_complete_count=sum(by_cell[cell(p), s]['energy_measurement_complete'] for s in BASELINES),
                         pd_receipt=p['receipt_path'], baseline_receipt=None if winner is None else winner['receipt_path']))
    aggregates = []
    for series in sorted({r['pd_series'] for r in pairs}):
        for baseline in BASELINES:
            for scope in ('all_available_observations', 'both_slo_feasible'):
                rr = [r for r in pairs if r['pd_series']==series and r['baseline']==baseline
                      and r['total_energy_available'] and (scope!='both_slo_feasible' or r['feasible_pair'])]
                pd_sum = sum(r['pd_total_kj'] for r in rr)
                b_sum = sum(r['baseline_total_kj'] for r in rr)
                aggregates.append(dict(pd_series=series, baseline=baseline, scope=scope, pairs=len(rr),
                                       pd_total_kj=pd_sum, baseline_total_kj=b_sum,
                                       saving_pct=100*(1-pd_sum/b_sum) if b_sum else None,
                                       pd_wins=sum(r['pd_total_kj']<r['baseline_total_kj'] for r in rr),
                                       pair_receipts=[(r['pd_receipt'],r['baseline_receipt']) for r in rr]))
    return pairs, best, aggregates


def historical_regression(root):
    rows = list(csv.DictReader((root/'historical-selected-points-180.csv').open()))
    assert len(rows) == len({r['point_id'] for r in rows}) == 180
    available = [r for r in rows if number(r.get('energy_service_j')) is not None and number(r.get('energy_tail_j')) is not None]
    assert len(available) == 157 and len(rows)-len(available) == 23
    by_id = {r['point_id']: r for r in rows}
    p = by_id['7b-pdblend-sharegpt-x1-seed701']; b = by_id['7b-ecoserve-sharegpt-x1-seed701']
    def total(r):
        return number(r['energy_service_j'])+number(r['energy_tail_j'])
    assert float(p['energy_service_j']) > float(b['energy_service_j']) and total(p) < total(b)
    assert close(total(p),303827.3425393093) and close(total(b),349438.79632221477)
    assert int(p['successful_requests']) == int(p['offered_requests']) == 1189
    assert int(b['successful_requests']) == 976
    return dict(points=180, complete_total_energy=157, missing_total_energy=23,
                pd_service_kj=float(p['energy_service_j'])/1000, eco_service_kj=float(b['energy_service_j'])/1000,
                pd_total_kj=total(p)/1000, eco_total_kj=total(b)/1000,
                reversal_confirmed=True, source_sha256=sha(root/'historical-selected-points-180.csv'))


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in r)+' |' for r in rows])


def fmt(value, digits=2):
    return 'NA' if value is None else f'{value:.{digits}f}'


def make_report(root, points, summary, best, aggregates):
    previous = [r for r in points if r['series']=='pd_previous']
    current = [r for r in points if r['series']=='pd_current']
    by = {r['point_id']: r for r in points if r['series'] != 'pd_current'}
    p,b = by['7b-pdblend-sharegpt-x1-seed701'],by['7b-ecoserve-sharegpt-x1-seed701']
    sections = ['# 完整请求周期能耗与 SLO 比较 · v2',
        f"冻结时间：{summary['snapshot']['captured_at']}；源表更新时间：{summary['snapshot']['source_modified_at']}。主能耗指标统一为八卡服务期＋尾部总能耗。",
        f"本版包含上一完整矩阵180点，以及新PDblend独立版本的{len(current)}点。新旧版本分别统计，不把新轮已完成点填进旧轮曲线。",
        '## 1. 修正后的含义',
        '150秒是请求到达窗口。总能耗从窗口起点持续到请求全部终结、控制器在途操作及窗口原生drain清理结束；成功、失败及取消前已发生的计算、八卡闲置功耗和窗口内冷唤醒均计入。服务开始前初始化、窗口间重置、会话结束后的卸载及CPU/主机能耗不在该指标内。',
        '“能耗完整”表示整个记录区间都有可用八卡采样，不表示全部请求成功。已取消请求没有执行的剩余计算无法补记能耗；这类点保留实测值并标记请求未全部成功。总能耗不分摊到单个请求。',
        '服务或尾部任一段缺失时总能耗为NA。沿用原始最大1秒插值间隔，未以平均功率、覆盖率或成功率填补。失败原因计数互斥；timeout、cancelled、rejected汇总是标签，可重叠，不能相加当失败总数。',
        'SLO attainment = 成功且TTFT、TPOT同时达标的请求数 / 全部到达请求数。迟于150秒完成的请求按真实延迟判定，不因跨窗口自动失败。失败和取消请求保留在分母。',
        '最低能耗观测排名要求：全部请求成功、无未决、joint attainment≥90%、TTFT/TPOT P99不超原门槛、总能耗完整且清理通过。资格标记仍沿用原记录，满足这些条件不自动获得论文复现或统计显著性资格。',
        '## 2. 能耗反转实例：7B / ShareGPT ×1',
        table(['系统','服务期 kJ','尾部 kJ','总能耗 kJ','成功/到达','SLO attainment'],[
            [NAMES[r['series']],fmt(r['service_energy_kj']),fmt(r['tail_energy_kj']),fmt(r['total_energy_kj']),
             f"{r['successful_requests']}/{r['offered_requests']}",fmt(r['slo_attainment_pct'])+'%'] for r in [p,b]]),
        f"EcoServe额外等待180秒后取消{b['cancelled_requests']}个请求；原表timeout={b['original_timeout_requests']}，结合原始取消收据修正后timeout标签={b['timeout_requests']}。PDblend总能耗比EcoServe低{100*(1-p['total_energy_kj']/b['total_energy_kj']):.2f}%，且全部请求成功；这个点不应再被描述为EcoServe节能。",
        '## 3. 逐系统覆盖与请求结果',
        table(['系列','已测点','总能耗完整','全部成功点','整点SLO通过','请求成功率','请求加权SLO'],[
            [NAMES[s],v['points'],v['energy_complete_points'],v['all_success_points'],v['slo_pass_points'],fmt(v['success_pct'])+'%',fmt(v['attainment_pct'])+'%']
            for s,v in summary['series'].items()]),
        '不同系列可用点数不同，不直接将各自可用总能耗相加后作跨系统排名。新轮仅在相同已测点上与baseline比较。',
        '## 4. 相同实验点上的总能耗比较',
        '正节能率表示PDblend更省。先列满足完整SLO约束的配对；原始观测配对包含服务失败点，只用于解释行为。各行配对集合不同。',
        table(['PD版本','baseline','比较集合','点数','PD总 kJ','baseline总 kJ','PD节能率'],[
            [NAMES[a['pd_series']],NAMES[a['baseline']], '双方SLO可行' if a['scope']=='both_slo_feasible' else '全部能耗完整观测',
             a['pairs'],fmt(a['pd_total_kj']) if a['pairs'] else 'NA',fmt(a['baseline_total_kj']) if a['pairs'] else 'NA',fmt(a['saving_pct'])+'%' if a['saving_pct'] is not None else 'NA']
            for a in sorted(aggregates,key=lambda a:(a['pd_series'],a['scope']!='both_slo_feasible',BASELINES.index(a['baseline'])))]),
        '## 5. 仍高于可用SLO合格baseline的点',
        '按完整周期能耗重新选择每个条件下的最低可用baseline。未测、缺能耗或未满足SLO的候选不参与排名；这不是所有配置的全局最优。',
        table(['PD版本','模型/数据集/倍率','PD总 kJ','最佳baseline','baseline总 kJ','PD高出'],[
            [NAMES[r['pd_series']],f"{r['model']} / {r['dataset']} / ×{r['rate_scale']:g}",fmt(r['pd_total_kj']),
             NAMES[r['best_available_baseline']],fmt(r['baseline_total_kj']),fmt(r['pd_excess_pct'])+'%']
            for r in sorted(best,key=lambda r:(r['pd_series'],-(r['pd_excess_pct'] or 0))) if r['pd_excess_pct'] is not None and r['pd_excess_pct']>0]),
        '优化优先级：先处理新轮仍复现的低负载ShareGPT劣势，修复Shield相同告警等级反复应用时重复扩容；再通过独立SLO验证开放低于4个M实例的候选。旧14B LongBench低rate的尾部扩容需在新版本确认；32B约几个百分点的差异需受控重复验证。该统计修正不修改控制器、不重跑GPU实验。',
        '## 6. rate比较图',
        '横轴是配置实际到达率（请求/秒）；每个子图使用相同trace和请求数量。能耗图红×表示存在请求失败/取消，空圈表示全部成功但整点SLO未通过，NA不连线。',
        '![八卡服务与尾部总能耗](energy-vs-rate.png)',
        '![SLO attainment](slo-attainment-vs-rate.png)',
        '![请求成功率](success-rate-vs-rate.png)',
        '## 7. 边界、验收和数据文件',
        '历史180点回归：157点完整总能耗、23点缺失（DistServe 5点、DynamoLLM 18点）。每点核对八卡积分之和、150秒服务边界与尾部首尾衔接、请求终结不超过计量结束、receipt绑定哈希、SLO分母及排名条件。',
        '现有数据是单seed短窗口观测。Dynamo每点重置控制器，150秒未覆盖300秒分片调整与1800秒实例伸缩周期；各系统历史外层超时规则也不同。修正统计不会将这些运行变成完整算法资格实验。',
        table(['文件','用途'],[
            ['points.csv / points.json','逐点服务、尾部和总能耗、成功/SLO/资格、失败归因和来源'],
            ['request-failures.csv','每个失败请求的原始错误、取消原因和证据索引'],
            ['paired-energy.csv / paired-summary.csv','完整周期配对比较与相同子集汇总'],
            ['best-available-slo-baseline.csv','按总能耗重新选出的可用SLO合格baseline'],
            ['missing-energy.csv','所有总能耗缺失点及服务/尾部覆盖率'],
            ['system-summary.csv / grouped-summary.csv / common-five-system-energy.csv','系列、模型/数据集/倍率以及五系统同一完整子集汇总'],
            ['snapshot.json / selection.json / provenance.json / validation.json','冻结时间、版本/重复点选择、证据哈希和验收记录'],
            ['comparison.pdf','三张矢量比较图；同时提供PNG/SVG'],
        ])]
    markdown='\n\n'.join(sections)+'\n'
    (root/'report.md').write_text(markdown)
    # A self-contained view (including embedded charts) is refreshed after plotting.
    render_html(root, markdown)


def render_html(root, markdown):
    import base64
    try:
        import markdown as md
        body = md.markdown(markdown, extensions=['tables'])
    except ImportError:
        # Render the intentionally small generated Markdown subset without a dependency.
        body_parts=[]; in_table=False
        for line in markdown.splitlines():
            if line.startswith('|'):
                cells=[x.strip() for x in line.strip('|').split('|')]
                if all(set(x)<=set('-: ') for x in cells):
                    continue
                if not in_table:
                    body_parts.append('<table>'); in_table=True
                body_parts.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
                continue
            if in_table:
                body_parts.append('</table>');in_table=False
            if line.startswith('!['):
                alt,src=line[2:].split('](',1);body_parts.append(f'<img alt="{html.escape(alt)}" src="{html.escape(src[:-1])}">')
            elif line.startswith('#'):
                n=len(line)-len(line.lstrip('#'));body_parts.append(f'<h{n}>{html.escape(line[n:].strip())}</h{n}>')
            elif line:
                body_parts.append('<p>'+html.escape(line)+'</p>')
        if in_table:body_parts.append('</table>')
        body='\n'.join(body_parts)
    for name in ('energy-vs-rate','slo-attainment-vs-rate','success-rate-vs-rate'):
        p=root/(name+'.png')
        if p.exists():
            body=body.replace('src="'+name+'.png"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
    (root/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整请求周期能耗比较 v2</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#203344}h1,h2{line-height:1.3}h2{margin-top:42px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 10px;border-bottom:1px solid #dce5eb;text-align:left}tr:first-child{background:#eef4f7;font-weight:600}img{width:100%;height:auto}p{max-width:1050px}</style><body>'+body+'</body></html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--render-html-only',action='store_true')
    args=parser.parse_args();root=args.input_dir.resolve()
    if args.render_html_only:
        render_html(root,(root/'report.md').read_text());return
    blob=(root/'compare-snapshot.csv').read_bytes();meta=json.loads((root/'snapshot.json').read_text())
    assert hashlib.sha256(blob).hexdigest()==meta['source_sha256']
    raw=list(csv.DictReader(io.StringIO(blob.decode())))
    selection, selected=select_rows(root,raw)
    points, failures, provenance=[],[],[]
    for i,(s,r) in enumerate(selected,1):
        point,details,prov=analyze_point(s,r)
        points.append(point);failures.extend(details);provenance.append(prov)
        if i % 12 == 0:
            print(f'Verified {i}/{len(selected)} points', flush=True)
    points.sort(key=lambda r:(MODELS.index(r['model']),DATASETS.index(r['dataset']),r['rate_scale'],r['series']))
    assert len(points)==len({(cell(r),r['series']) for r in points})
    for series in ('pd_previous','pd_current'):
        assert len({r['revision'] for r in points if r['series']==series})<=1
    by_cell={(cell(r),r['series']):r for r in points}
    for r in points:
        assert r['comparison_identity']==by_cell[cell(r),'mixed']['comparison_identity'],r['point_id']
    pairs,best,aggregates=compare_pairs(points)
    historical=historical_regression(root)
    eco=next(r for r in points if r['point_id']=='7b-ecoserve-sharegpt-x1-seed701')
    assert eco['cancelled_requests']==213 and eco['timeout_requests']>=213 and not eco['energy_rank_eligible']
    series_order=BASELINES+['pd_previous','pd_current']
    series_summary={s:summarize([r for r in points if r['series']==s]) for s in series_order if any(r['series']==s for r in points)}
    groups=[]
    for dimension in ('model','dataset','rate_scale'):
        for value in sorted({r[dimension] for r in points},key=str):
            for s in series_summary:
                rr=[r for r in points if r[dimension]==value and r['series']==s]
                if rr:groups.append(dict(dimension=dimension,value=value,series=s,**summarize(rr)))
    common=[]
    for pd_series in ('pd_previous','pd_current'):
        cells=[cell(r) for r in points if r['series']==pd_series]
        complete=[c for c in cells if all(by_cell[c,s]['energy_measurement_complete'] for s in BASELINES+[pd_series])]
        for s in BASELINES+[pd_series]:
            rr=[by_cell[c,s] for c in complete]
            if rr:common.append(dict(pd_series=pd_series,series=s,common_cells=[list(c) for c in complete],**summarize(rr)))
    summary=dict(snapshot=meta,selection=selection,series=series_summary,paired_summary=aggregates,
                 historical_regression=historical,common_five_systems=common,
                 scope='Single-observation, system-as-executed, total cohort energy; not formal qualification.')
    validation=dict(passed=True,selected_points=len(points),request_rows_verified=sum(r['offered_requests'] for r in points),
                    total_energy_available=sum(r['energy_measurement_complete'] for r in points),
                    total_energy_missing=sum(not r['energy_measurement_complete'] for r in points),
                    all_eight_gpu_sums_verified=True,service_tail_boundaries_contiguous=True,
                    request_horizons_within_metering=True,all_source_hashes_verified=True,
                    bound_native_completion_within_metering=True,power_source_and_error_state_verified=True,
                    matched_numeric_identities=True,slo_recomputed_all_offered=True,
                    pd_revisions_separate=True,failed_requests_excluded_from_feasible_rank=True,
                    no_missing_energy_imputed=True,eco_cohort_timeout_cancelled=213,
                    historical_regression=historical)
    for name,value in [('points.json',points),('summary.json',summary),('validation.json',validation),('provenance.json',provenance)]:
        json_write(root/name,value)
    for name,rr in [('points.csv',points),('request-failures.csv',failures),('paired-energy.csv',pairs),
                    ('paired-summary.csv',aggregates),('best-available-slo-baseline.csv',best),
                    ('missing-energy.csv',[r for r in points if not r['energy_measurement_complete']]),
                    ('system-summary.csv',[dict(series=s,**v) for s,v in series_summary.items()]),
                    ('grouped-summary.csv',groups),('common-five-system-energy.csv',common)]:
        csv_write(root/name,rr)
    make_report(root,points,summary,best,aggregates)
    print(json.dumps(dict(output=str(root),validation=validation,series=series_summary),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
