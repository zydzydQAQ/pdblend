"""Audited CSV and publication-style plots; missing work is never filled in."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import time

import contract as c
import metrics as m

ROOT = c.ROOT
HERE = Path(__file__).resolve().parent
DIAGNOSIS_REFERENCES = [dict(path=str(ROOT / 'A/uniform-rate-20260909-v1/diagnostics/root-idle-admission-001.json'),
    sha256='ff115e4f51bb8d44ae6c47bc18dc102b947ad2e9b09373d03235e6e68a546cdf')]
METRICS = ('energy_j', 'slo_attainment', 'ttft_avg_s', 'tpot_avg_s',
           'completed_work_throughput_rps', 'generated_token_throughput_tps', 'gpu_util')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def audit_observation(observation, cache, *, reused=False, summary_required=True):
    """Recompute every displayed point; cache only unchanged immutable evidence."""
    checkpoint = c.checked(observation['checkpoint'])
    receipt = checkpoint['receipt']
    if isinstance(receipt, str):
        receipt = dict(path=receipt, sha256=checkpoint['receipt_sha256'])
    row = checkpoint['row']
    directory = Path(receipt['path']).parents[2] / 'cells' / row['cell_id']
    paths = set(checkpoint['artifacts']) | {receipt['path'], row['trace'],
        str(directory / 'bench.csv'), str(directory / 'power.csv'), str(directory / 'summary.json')}
    fingerprint = []
    for path in sorted(paths):
        stat = Path(path).stat()
        fingerprint.append([path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    fingerprint.append(['auditor', c.sha(m.__file__), 2])
    fingerprint.append(['arrival_auditor', c.sha(ROOT / 'audit_cooperative_arrivals_v1.py')])
    cached = cache / (observation['checkpoint']['sha256'] + '.json')
    if cached.exists():
        saved = c.read(cached)
        if saved['fingerprint'] == fingerprint:
            raw = saved['raw_metrics']
        else:
            raw = None
    else:
        raw = None
    if raw is None:
        for path, digest in checkpoint['artifacts'].items():
            c.need(c.sha(path) == digest, 'artifact changed: ' + path)
        raw = m.audit_receipt(receipt['path'], row, receipt_reference=receipt)
        save(cached, dict(fingerprint=fingerprint, raw_metrics=raw))
    if summary_required:
        c.need(m.close(raw['slo_attainment'], observation['slo_attainment']), 'audited observation SLO differs from strict raw recomputation')
        c.need(raw['work_complete'] == observation['work_complete'], 'audited work completeness differs from raw')
    result = dict(observation)
    result.update(raw)
    summary = c.checked(raw['summary']) if raw.get('summary') else {}
    result.update(trace=row['trace'], measurement_start_s=summary.get('measurement_start_s'),
                  measurement_end_s=summary.get('measurement_end_s'))
    result.update(reused=reused, completed_work_throughput_rps=raw['request_throughput_rps'],
        generated_token_throughput_tps=raw['token_throughput_tps'],
        recorded_output_is_partial=not raw['generated_token_count_complete'])
    return normalized(result)


def historical(observation, cache):
    return audit_observation(observation, cache, reused=True)


def latest_states():
    result = {}
    for node, model in [('C', '7b'), ('A', '14b'), ('B', '32b')]:
        directory = ROOT / node
        candidates = []
        for path in directory.glob('uniform-*/**/status.json'):
            if 'pipeline' not in path.parent.name:
                continue
            try:
                state = c.read(path)
                if 'declaration' in state and 'observations' in state:
                    candidates.append((path.stat().st_mtime_ns, path, state))
            except (OSError, ValueError):
                continue
        if candidates:
            _, path, state = max(candidates, key=lambda x: x[0])
            result[model] = (path, state)
    return result


def pipeline_observation_refs(path, state):
    """Include finalized repeats from an active child before its whole rate ends."""
    refs = list(state.get('observations', []))
    if path is None:
        return refs
    path = Path(path)
    candidates = set(path.parent.glob('**/status.json'))
    for stage in state.get('stages', {}).values():
        argv = stage.get('argv', [])
        if '--out' in argv and any(str(x).endswith('/run_cells.py') for x in argv):
            out = Path(argv[argv.index('--out') + 1])
            if out.is_relative_to(path.parent.parent):
                candidates.add(out / 'status.json')
    for candidate in sorted(candidates):
        if candidate == path:
            continue
        try:
            refs.extend(c.read(candidate).get('observations', []))
        except (OSError, ValueError):
            continue
    return list({(ref['path'], ref['sha256']): ref for ref in refs}.values())


def normalized(value):
    value = dict(value)
    value.setdefault('completed_work_throughput_rps', value.get('request_throughput_rps'))
    value.setdefault('generated_token_throughput_tps', value.get('token_throughput_tps'))
    value.setdefault('token_throughput_is_exact', value.get('generated_token_count_complete'))
    return value


def cap_rate(decision):
    cap = decision.get('cap_rate_rps')
    if cap is None and decision.get('decision', {}).get('cap_observed'):
        cap = decision.get('rate_rps')
    return cap


def stop_trigger(observations, model, dataset, cap):
    candidates = [r for r in observations if r['model'] == model and r['dataset'] == dataset
        and r['system'] == 'pdblend' and r['rate_rps'] == cap and r.get('measurement_valid')
        and r.get('work_complete') and r['slo_attainment'] < .9]
    return min(candidates, key=lambda r: r.get('repeat', 1)) if candidates else None


def relevant_positions(group, decision):
    cap = cap_rate(decision)
    return [p for p in group['positions'] if cap is None or p['rate_rps'] <= cap]


def collect(declaration, out):
    base = c.load_declaration(declaration)
    states = latest_states()
    observations = {}
    errors = []
    for observation in base['reused_observations']:
        try:
            value = historical(observation, out / 'audit-cache')
            observations[value['cell_id']] = value
        except Exception as exc:
            errors.append(dict(cell_id=observation['cell_id'], model=observation['model'], dataset=observation['dataset'],
                error=repr(exc), missing_path=str(getattr(exc, 'filename', '') or ''), checkpoint=observation['checkpoint']))
    group_results = []
    def failure(reference, exc, **details):
        errors.append(dict(reference=reference, error=repr(exc),
            missing_path=str(getattr(exc, 'filename', '') or ''), **details))
    for model in c.MODELS:
        state_path, state = states.get(model, (None, {}))
        current = state.get('declaration', declaration)
        dynamic_refs = state.get('dynamic_reuse_observations', [])
        if state.get('plan'):
            try:
                dynamic_refs = c.checked(state['plan']).get('dynamic_reuse_observations', dynamic_refs)
            except Exception as exc:
                failure(state['plan'], exc, model=model)
        dynamic, fresh = [], []
        for references, destination, reused in ((dynamic_refs, dynamic, True),
                                                (pipeline_observation_refs(state_path, state), fresh, False)):
            for reference in references:
                try:
                    value = c.checked(reference)
                    audited = audit_observation(value, out / 'audit-cache', reused=reused)
                    destination.append(audited)
                    # An incomplete PDB attempt is an engineering/diagnostic
                    # outcome, never a point on the primary scaling curves.
                    # It remains in scheduling evaluation to expose diagnosis.
                    if audited['system'] != 'pdblend' or audited['work_complete'] is True:
                        observations[audited['cell_id']] = audited
                except Exception as exc:
                    failure(reference, exc, model=model)
        for dataset in c.DATASETS:
            try:
                group = c.resolve_group(current, model, dataset)
                reused = [v for v in dynamic if v['dataset'] == dataset]
                if reused:
                    group = c.apply_audited_reuse(group, reused)
                selected = c.select_group(group, [v for v in fresh if v['dataset'] == dataset])
                missing_audits = [t['cell_id'] for p in relevant_positions(group, selected) for ts in p['systems'].values()
                                  for t in ts if t['action'] == 'reuse' and t['cell_id'] not in observations]
            except Exception as exc:
                failure(current, exc, model=model, dataset=dataset)
                group = c.resolve_group(declaration, model, dataset)
                selected = dict(phase='awaiting_evidence', error=repr(exc))
                missing_audits = ['group_evidence_unavailable']
            group_results.append(dict(model=model, dataset=dataset, node=group['node'],
                rate_start_rps=float(c.step(model, dataset)), rate_step_rps=float(c.step(model, dataset)),
                declaration=current, decision=selected, missing_raw_metric_audits=missing_audits,
                rate_grid=[p['rate_rps'] for p in group['positions']], cap_rate_rps=cap_rate(selected),
                pipeline_state=dict(phase=state.get('phase'), complete=state.get('complete'),
                                   error=state.get('error'), updated_s=state.get('updated_s'),
                                   scope=state.get('scope'), pdb_complete=state.get('pdb_complete')),
                pdb_boundary_complete=selected['phase'] in ('baselines', 'complete')
                    and all(t['cell_id'] in observations for p in relevant_positions(group, selected)
                            for t in p['systems']['pdblend']),
                complete=selected['phase'] == 'complete' and not missing_audits))
    # The PDB-priority dispatcher deliberately omits baseline scheduling refs.
    # Keep already measured baseline outcomes visible without turning them into
    # scheduling reuse or using them to declare five-system completion.
    retained = list((ROOT / 'B/uniform-rate-20260909-v1').glob('predecessor-*/status.json'))
    retained += list((ROOT / 'B/ascending-resume-20260909-v2').glob('baseline-*-performance-001/status.json'))
    for status_path in retained:
        try:
            status = c.read(status_path)
            for reference in status.get('observed_checkpoints', []):
                checkpoint = c.checked(reference)
                row = checkpoint['row']
                if row['cell_id'] in observations or row['system'] == 'pdblend':
                    continue
                value = {key: row[key] for key in ('cell_id', 'model', 'dataset', 'system',
                    'rate_rps', 'repeat', 'seed', 'trace_sha256', 'content_pairing_sha256', 'slo_ttft_s', 'slo_tpot_s')}
                value.update(checkpoint=reference, measurement_host=row.get('measurement_host', row.get('node', 'B')),
                    report_only_pending_pairing=True)
                group = c.resolve_group(declaration, value['model'], value['dataset'])
                c.need(value['measurement_host'] == 'B' and any(c.identity(value) == c.identity(p['workload'])
                    for p in group['positions']), 'retained baseline is not an exact same-host grid point')
                audited = audit_observation(value, out / 'audit-cache', reused=True, summary_required=False)
                observations[audited['cell_id']] = audited
        except Exception as exc:
            failure(str(status_path), exc, model='32b', retained_baseline=True)
    rows = list(observations.values())
    for group in group_results:
        trigger = stop_trigger(rows, group['model'], group['dataset'], group['cap_rate_rps'])
        group['stop_trigger'] = ({key: trigger.get(key) for key in ('cell_id', 'repeat', 'slo_attainment', 'checkpoint')}
                                 if trigger else None)
    rows.sort(key=lambda r: (c.MODELS.index(r['model']), c.DATASETS.index(r['dataset']), r['rate_rps'],
                             c.SYSTEMS.index(r['system']), r.get('repeat', 1)))
    return dict(schema='uniform-rate-results-v1', updated_s=time.time(), declaration=declaration,
        complete=len(group_results) == 9 and all(g['complete'] for g in group_results), groups=group_results,
        observations=rows, metric_audit_errors=errors,
        semantics=dict(repeats='same seed701 trace; ranges are observed min/max, not independent-seed confidence intervals',
            stop='any valid complete PDB repeat with attainment < .90; equality continues',
            energy='all eight GPUs over the primary full measurement window; no overlapping setup energy added',
            slo='strict joint TTFT/TPOT; failures remain in denominator',
            throughput='completed requests and producer-recorded output tokens per full measurement duration'))


def write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export(result, out):
    observations = [normalized(v) for v in result['observations']]
    caps = {(g['model'], g['dataset']): cap_rate(g['decision']) for g in result['groups']}
    grids = {(g['model'], g['dataset']): set(g['rate_grid']) for g in result['groups']}
    selected = [v for v in observations if caps[(v['model'], v['dataset'])] is None
                or v['rate_rps'] <= caps[(v['model'], v['dataset'])]]
    selected = [v for v in selected if v['rate_rps'] in grids[(v['model'], v['dataset'])]]
    fields = ['model', 'dataset', 'measurement_host', 'system', 'rate_rps', 'repeat', *METRICS,
        'work_complete', 'completion_fraction', 'failed_requests', 'request_timeouts',
        'token_throughput_is_exact', 'recorded_output_is_partial', 'producer_slo_attainment', 'reused',
        'report_only_pending_pairing', 'cell_id']
    write_csv(out / 'measurements.csv', selected, fields)
    included = {v['cell_id'] for v in selected}
    evidence_rows, gpu_rows = [], []
    keys = ['model', 'dataset', 'measurement_host', 'system', 'rate_rps', 'repeat', 'cell_id',
            'measurement_valid', 'work_complete', 'strict_slo_recomputed',
            'measurement_start_s', 'measurement_end_s', 'measurement_duration_s',
            'recorded_output_is_partial', 'report_only_pending_pairing']
    ref_names = ('checkpoint', 'receipt', 'summary', 'raw_requests', 'raw_power', 'audit_reference', 'diagnosis_reference')
    for observation in observations:
        item = {key: observation.get(key) for key in keys}
        item.update(in_current_grid_report=observation.get('cell_id') in included,
                    trace_path=observation.get('trace'), trace_sha256=observation.get('trace_sha256'))
        for name in ref_names:
            reference = observation.get(name) or {}
            item[name + '_path'] = reference.get('path')
            item[name + '_sha256'] = reference.get('sha256')
        evidence_rows.append(item)
        energy = observation.get('energy_per_gpu_j', [])
        utilization = observation.get('gpu_util_per_gpu', [])
        duration = observation.get('measurement_duration_s')
        for index in range(8):
            joules = energy[index] if index < len(energy) else None
            fraction = utilization[index] if index < len(utilization) else None
            gpu_rows.append(dict(item, gpu_index=index, energy_j=joules, gpu_util=fraction,
                gpu_util_percent=fraction*100 if fraction is not None else None,
                average_power_w=joules/duration if joules is not None and duration else None))
    write_csv(out / 'raw-evidence-index.csv', evidence_rows,
        keys + ['in_current_grid_report', 'trace_path', 'trace_sha256'] +
        [name + suffix for name in ref_names for suffix in ('_path', '_sha256')])
    write_csv(out / 'gpu-details.csv', gpu_rows,
        keys + ['in_current_grid_report', 'gpu_index', 'energy_j', 'gpu_util', 'gpu_util_percent',
                'average_power_w', 'checkpoint_path', 'checkpoint_sha256', 'raw_power_path', 'raw_power_sha256'])
    buckets = defaultdict(list)
    for row in selected:
        buckets[(row['model'], row['dataset'], row['system'], row['rate_rps'])].append(row)
    aggregated = []
    for (model, dataset, system, rate), rows in sorted(buckets.items()):
        item = dict(model=model, dataset=dataset, system=system, rate_rps=rate, repeats=len(rows),
            incomplete_work_repeats=sum(r.get('work_complete') is not True for r in rows),
            partial_token_repeats=sum(r.get('token_throughput_is_exact') is not True for r in rows))
        for metric in METRICS:
            values = [r[metric] for r in rows if isinstance(r.get(metric), (int, float)) and math.isfinite(r[metric])]
            for suffix, value in [('mean', statistics.fmean(values) if values else None),
                                  ('min', min(values) if values else None), ('max', max(values) if values else None)]:
                item[metric + '_' + suffix] = value
        aggregated.append(item)
    write_csv(out / 'summary.csv', aggregated, ['model', 'dataset', 'system', 'rate_rps', 'repeats',
        'incomplete_work_repeats', 'partial_token_repeats'] +
        [metric + '_' + suffix for metric in METRICS for suffix in ('mean', 'min', 'max')])
    save(out / 'results.json', result)
    lines = ['# 固定步长 Rate Scale 进度', '', '状态：' + ('五系统全部完成' if result['complete'] else
        'PDB 范围已完成；五系统配对尚未完成' if result.get('scope_complete') and result.get('completion_scope') == 'pdblend'
        else '执行中，以下为已核验数据'), '',
        '| 模型 | 数据集 | 步长 rps | PDB 边界测量 | 五系统配对 | 已确认终点 rps |', '|---|---|---:|---|---|---:|']
    for group in result['groups']:
        boundary = '完成' if group.get('pdb_boundary_complete') else '进行中'
        paired = '完成' if group['complete'] else '待完成'
        cap = group.get('cap_rate_rps')
        lines.append(f"| {group['model']} | {group['dataset']} | {group['rate_step_rps']} | {boundary} | {paired} | {cap if cap is not None else '待测'} |")
    lines += ['', 'CSV 包含每次重复及均值/实测范围；范围不是独立种子置信区间。未完成请求保留在 SLO 分母中。',
              '能量、利用率和吞吐均使用完整主测量窗口。横轴为实际 rps，未测点不插值。',
              'PDB 边界测量完成与五系统配对完成分别统计。已确认首个低于 90% 的重复会立即截断横轴。',
              'raw-evidence-index.csv 保存每个已核验测量的原始文件和哈希；gpu-details.csv 保存逐 GPU 的能耗、利用率及平均功率。',
              'report_only_pending_pairing 标记已保留展示、尚未被当前调度明确绑定的历史基线，不用于判定配对完成。']
    if result['metric_audit_errors']:
        lines += ['', f"仍有 {len(result['metric_audit_errors'])} 个测量的原始证据缺失或尚未通过复算，未填零。"]
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    return aggregated


def checked_diagnoses(cache, references):
    results, errors = {}, []
    for reference in references:
        try:
            proof = c.checked(reference)
            evidence = proof['evidence']
            fingerprint = []
            for item in evidence:
                stat = Path(item['path']).stat()
                fingerprint.append([item['path'], item['sha256'], stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
            target = cache / ('diagnosis-' + reference['sha256'] + '.json')
            previous = c.read(target) if target.exists() else None
            if previous != fingerprint:
                for item in evidence:
                    c.need(c.sha(item['path']) == item['sha256'], 'diagnosis evidence changed: ' + item['path'])
                save(target, fingerprint)
            c.need(proof.get('slo_boundary_eligible') is False and proof.get('valid_performance_repeat') is False,
                   'engineering diagnosis must exclude the attempt from capacity and valid repeats')
            checkpoint = proof['measurement']['checkpoint']
            results[(proof['cell_id'], checkpoint['path'], checkpoint['sha256'])] = dict(
                classification=proof['classification'], diagnosis_reference=reference)
        except Exception as exc:
            errors.append(dict(reference=reference, error=repr(exc)))
    return results, errors


def collect_engineering_failures(cache, root=ROOT, primary_result=None):
    """Preserve failed attempts independently of the latest active pipeline."""
    entries = []
    visible, triggers = set(), set()
    if primary_result:
        groups = {(g['model'], g['dataset']): g for g in primary_result['groups']}
        for observed in primary_result['observations']:
            group = groups[(observed['model'], observed['dataset'])]
            cap = cap_rate(group['decision'])
            if observed['rate_rps'] in group['rate_grid'] and (cap is None or observed['rate_rps'] <= cap):
                reference = observed.get('checkpoint', {})
                visible.add((reference.get('path'), reference.get('sha256')))
        for group in groups.values():
            reference = (group.get('stop_trigger') or {}).get('checkpoint', {})
            if reference:
                triggers.add((reference['path'], reference['sha256']))
    diagnoses, diagnosis_errors = checked_diagnoses(cache, DIAGNOSIS_REFERENCES if root == ROOT else [])
    for node, model in (('A', '14b'), ('B', '32b'), ('C', '7b')):
        for path in sorted((root / node).glob('uniform-*/**/status.json')):
            try:
                state = c.read(path)
                if not state.get('finished_s') or not (state.get('error') or state.get('errors')
                        or state.get('failed') or state.get('passed') is False):
                    continue
                entry = dict(node='Anew20260909' if node == 'A' else node, model=model,
                    status=c.ref(path), schema=state.get('schema'), phase=state.get('phase'),
                    started_s=state.get('started_s'), finished_s=state['finished_s'],
                    error=state.get('error') or state.get('errors'), failed_cells=state.get('failed', []),
                    classification='needs_diagnosis', diagnosis_reference=None, checkpoints=[],
                    included_in_main_curves=False, used_as_capacity_boundary=False)
                diagnosis = state.get('diagnosis_reference')
                if diagnosis:
                    try:
                        c.checked(diagnosis)
                        entry.update(diagnosis_reference=diagnosis,
                            classification=state.get('failure_class', 'diagnosis_recorded'))
                    except Exception as exc:
                        entry['diagnosis_reference_error'] = repr(exc)
                for reference in state.get('observed_checkpoints', []):
                    key = (reference['path'], reference['sha256'])
                    item = dict(checkpoint=reference, raw_metrics_audited=False,
                        included_in_main_curves=key in visible, used_as_capacity_boundary=key in triggers)
                    try:
                        checkpoint = c.checked(reference)
                        row = checkpoint['row']
                        item.update({key: row.get(key) for key in ('cell_id', 'model', 'dataset', 'system', 'rate_rps', 'repeat')})
                        receipt = checkpoint['receipt']
                        if isinstance(receipt, str):
                            receipt = dict(path=receipt, sha256=checkpoint['receipt_sha256'])
                        item['receipt'] = receipt
                        directory = Path(receipt['path']).parents[2] / 'cells' / row['cell_id']
                        for name, filename in (('raw_requests', 'bench.csv'), ('raw_power', 'power.csv'), ('summary', 'summary.json')):
                            raw_path = str(directory / filename)
                            digest = checkpoint['artifacts'].get(raw_path)
                            item[name] = dict(path=raw_path, sha256=digest) if digest else None
                        proposed = dict(item, measurement_host=entry['node'])
                        audited = audit_observation(proposed, cache, summary_required=False)
                        item.update(raw_metrics_audited=True, metrics={key: audited.get(key) for key in
                            (*METRICS, 'n_expected', 'completed_work_requests', 'failed_requests', 'request_timeouts',
                             'work_complete', 'completion_fraction', 'generated_token_count_complete',
                             'measurement_duration_s', 'producer_slo_attainment')})
                    except Exception as exc:
                        item['raw_metric_audit_error'] = repr(exc)
                        item['missing_path'] = str(getattr(exc, 'filename', '') or '')
                    known = diagnoses.get((item.get('cell_id'), reference['path'], reference['sha256']))
                    if known:
                        item.update(known)
                    entry['checkpoints'].append(item)
                if entry['checkpoints'] and all(p.get('diagnosis_reference') for p in entry['checkpoints']):
                    classes = {p['classification'] for p in entry['checkpoints']}
                    entry['classification'] = classes.pop() if len(classes) == 1 else 'multiple_recorded_diagnoses'
                entry['included_in_main_curves'] = any(p['included_in_main_curves'] for p in entry['checkpoints'])
                entry['used_as_capacity_boundary'] = any(p['used_as_capacity_boundary'] for p in entry['checkpoints'])
                entries.append(entry)
            except (OSError, ValueError, KeyError) as exc:
                entries.append(dict(node=node, model=model, status=dict(path=str(path)),
                    classification='unreadable_failure_status', error=repr(exc), checkpoints=[],
                    included_in_main_curves=False, used_as_capacity_boundary=False))
    return dict(schema='uniform-engineering-failures-v1', updated_s=time.time(), entries=entries,
        diagnosis_evidence_errors=diagnosis_errors,
        semantics='Terminal failed attempts remain visible after a successor starts. Diagnosis is not inferred. '
                  'A failed supervisor may retain earlier independently valid checkpoints; inclusion flags describe '
                  'their separate primary use. This failure ledger never promotes a failed attempt into a capacity boundary.')


def export_engineering_failures(result, out):
    save(out / 'engineering-failures.json', result)
    rows = []
    for entry in result['entries']:
        for checkpoint in entry['checkpoints'] or [{}]:
            item = {key: entry.get(key) for key in ('node', 'model', 'classification', 'phase', 'started_s', 'finished_s', 'error')}
            item.update({key: checkpoint.get(key) for key in ('cell_id', 'dataset', 'system', 'rate_rps', 'repeat',
                'raw_metrics_audited', 'raw_metric_audit_error', 'missing_path')})
            item.update(checkpoint.get('metrics', {}))
            if checkpoint.get('classification'):
                item['classification'] = checkpoint['classification']
            item.update(included_in_main_curves=checkpoint.get('included_in_main_curves', False),
                        used_as_capacity_boundary=checkpoint.get('used_as_capacity_boundary', False))
            for name in ('status', 'diagnosis_reference', 'checkpoint', 'receipt', 'raw_requests', 'raw_power', 'summary'):
                reference = ((checkpoint.get(name) or entry.get(name)) if name == 'diagnosis_reference' else
                    (entry if name == 'status' else checkpoint).get(name)) or {}
                item[name + '_path'] = reference.get('path')
                item[name + '_sha256'] = reference.get('sha256')
            rows.append(item)
    fields = ['node', 'model', 'dataset', 'system', 'rate_rps', 'repeat', 'cell_id', 'phase',
        'classification', 'error', 'started_s', 'finished_s', 'raw_metrics_audited', 'raw_metric_audit_error', 'missing_path',
        'included_in_main_curves', 'used_as_capacity_boundary', 'n_expected', 'completed_work_requests',
        'failed_requests', 'request_timeouts', 'work_complete', 'completion_fraction', *METRICS]
    fields += [name + suffix for name in ('status', 'diagnosis_reference', 'checkpoint', 'receipt', 'raw_requests', 'raw_power', 'summary')
               for suffix in ('_path', '_sha256')]
    write_csv(out / 'engineering-failures.csv', rows, fields)


def plot(result, rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    from matplotlib.lines import Line2D
    panels = [('energy_j', 'Energy (kJ)', .001), ('slo_attainment', 'SLO attainment (%)', 100),
              ('ttft_avg_s', 'Avg TTFT (ms)', 1000), ('tpot_avg_s', 'Avg TPOT (ms)', 1000),
              ('generated_token_throughput_tps', 'Output throughput (tokens/s)', 1),
              ('gpu_util', 'GPU utilization (%)', 100)]
    colors = dict(zip(c.SYSTEMS, ['#c43c32', '#285f9e', '#3e8a61', '#9865ad', '#db9628']))
    for group in result['groups']:
        model, dataset = group['model'], group['dataset']
        cap = cap_rate(group['decision'])
        grid = [x for x in group['rate_grid'] if cap is None or x <= cap]
        figure, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
        label = 'complete' if group['complete'] else 'partial; boundary pending' if cap is None else 'partial; cap confirmed'
        figure.suptitle(f'{model.upper()} / {dataset} — {label}', fontsize=14)
        for ax, (metric, ylabel, scale) in zip(axes.flat, panels):
            for system in c.SYSTEMS:
                series = {r['rate_rps']: r for r in rows if (r['model'], r['dataset'], r['system']) == (model, dataset, system)}
                mean = [series[x].get(metric + '_mean') if x in series else None for x in grid]
                low = [series[x].get(metric + '_min') if x in series else None for x in grid]
                high = [series[x].get(metric + '_max') if x in series else None for x in grid]
                values = np.array([np.nan if v is None else v * scale for v in mean])
                lo = np.array([np.nan if v is None else v * scale for v in low])
                hi = np.array([np.nan if v is None else v * scale for v in high])
                ax.plot(grid, values, 'o-', color=colors[system], label=system, markersize=3.5, linewidth=1.3)
                ax.fill_between(grid, lo, hi, color=colors[system], alpha=.12)
                partial = [x for x in grid if x in series and series[x].get('incomplete_work_repeats')]
                if partial:
                    ax.scatter(partial, [series[x].get(metric + '_mean', float('nan')) * scale
                        if series[x].get(metric + '_mean') is not None else float('nan') for x in partial],
                        marker='x', s=34, color=colors[system], zorder=4)
            if metric == 'slo_attainment':
                ax.axhline(90, color='#555555', linestyle='--', linewidth=.9)
                ax.set_ylim(-2, 103)
                if cap is not None:
                    trigger = stop_trigger(result['observations'], model, dataset, cap)
                    if trigger:
                        ax.scatter([cap], [trigger['slo_attainment'] * 100], marker='*',
                                   s=115, color=colors['pdblend'], zorder=5)
            ax.set_xlabel('Offered rate (requests/s)')
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MultipleLocator(group['rate_step_rps']))
            ax.set_xlim(0, max(grid) + group['rate_step_rps'] * .25)
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if cap is not None and stop_trigger(result['observations'], model, dataset, cap):
            handles.append(Line2D([], [], color=colors['pdblend'], marker='*', linestyle='None', markersize=10))
            labels.append('First PDB repeat <90%')
        figure.legend(handles, labels, loc='outside lower center', ncol=6, frameon=False,
                      title='x: incomplete work; shaded range: observed repeats')
        for extension in ('png', 'pdf'):
            path = out / f'{model}-{dataset}.{extension}'
            temporary = path.with_suffix('.tmp.' + extension)
            figure.savefig(temporary, dpi=170, bbox_inches='tight')
            temporary.replace(path)
        plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declaration', type=Path, default=HERE / 'release-001/declaration.json')
    parser.add_argument('--out', type=Path, default=HERE / 'reports/current')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = collect(c.ref(args.declaration), args.out)
    aggregated = export(result, args.out)
    export_engineering_failures(collect_engineering_failures(args.out / 'audit-cache', primary_result=result), args.out)
    if args.plots:
        plot(result, aggregated, args.out)
    print(json.dumps(dict(complete=result['complete'], observations=len(result['observations']),
        metric_audit_errors=len(result['metric_audit_errors']), out=str(args.out))))
