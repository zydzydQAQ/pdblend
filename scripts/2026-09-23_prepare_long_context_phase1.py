#!/usr/bin/env python3
"""CPU-only cost accounting and a bounded 7B TP4 long-context phase-one plan."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREQUENCIES = (900, 1200, 1500, 1800, 2100, 2520)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefill_proxy(raw, frequency, context):
    """Interpolate only measured prefill lengths, for scheduling, never fitting."""
    rows = [r for r in raw['prefill'] if r['freq_mhz'] == frequency]
    if len({r['input_tokens'] for r in rows}) != len(rows):
        raise ValueError('ambiguous prefill anchors')
    if any(not math.isfinite(r['seconds']) or r['seconds'] <= 0 or r.get('concurrency', 1) != 1 for r in rows):
        raise ValueError('cost accounting requires positive single-request prefill observations')
    exact = next((r for r in rows if r['input_tokens'] == context), None)
    if exact is not None:
        return exact['seconds'], 'measured_single_request', [dict(input_tokens=context, seconds=exact['seconds'])]
    lower = [r for r in rows if r['input_tokens'] < context]
    upper = [r for r in rows if r['input_tokens'] > context]
    if not lower or not upper:
        raise ValueError('prefill cost estimate would extrapolate outside measured lengths')
    lo, hi = max(lower, key=lambda r: r['input_tokens']), min(upper, key=lambda r: r['input_tokens'])
    weight = (context-lo['input_tokens'])/(hi['input_tokens']-lo['input_tokens'])
    seconds = lo['seconds'] + weight*(hi['seconds']-lo['seconds'])
    return seconds, 'within_range_prefill_interpolation', [dict(input_tokens=r['input_tokens'], seconds=r['seconds']) for r in (lo, hi)]


def build_phase1(original, raw, *, original_path, original_sha256):
    identity = dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', tp=4, pp=1)
    if any(original.get(k) != v or raw.get(k) != v for k, v in identity.items()):
        raise ValueError('phase one is limited to the independent 7B TP4 PP1 profile')
    expected = {(f, c, b) for f in FREQUENCIES for c in (5120, 7168) for b in (1, 4, 8, 256)}
    actual = [(p['freq_mhz'], p['context_tokens'], p['batch']) for p in original['training']]
    if len(actual) != 48 or set(actual) != expected or original.get('fit_existing_holdout') is not False:
        raise ValueError('original plan must contain the immutable 48-point training matrix')
    if any(p.get('purpose') != 'training_extension' or p.get('repeats') != 3 or
           p.get('settle_s', 0) < 2 or p.get('measure_s', 0) < 5 for p in original['training']):
        raise ValueError('phase one cannot change existing training or measurement gates')
    phase = copy.deepcopy(original)
    selected = [copy.deepcopy(p) for p in original['training'] if p['batch'] <= 8]
    deferred = [dict(copy.deepcopy(p), scheduling_status='deferred', coverage_status='missing_profile',
                     reason='high_batch_long_context_prefill_cost_deferred_to_later_phase')
                for p in original['training'] if p['batch'] > 8]
    # Future holdout observations also cannot require the deferred training
    # region. Keep their original definitions, but do not schedule them here.
    holdout = [copy.deepcopy(p) for p in original.get('holdout', []) if p['batch'] <= 8]
    deferred_holdout = [dict(copy.deepcopy(p), scheduling_status='deferred', coverage_status='missing_profile',
                            reason='corresponding_high_batch_long_context_training_is_deferred')
                       for p in original.get('holdout', []) if p['batch'] > 8]
    phase.update(training=selected, holdout=holdout, deferred_training=deferred,
        deferred_holdout=deferred_holdout, phase='phase1_low_batch_long_context',
        original_full_plan_path=str(Path(original_path).resolve()), original_full_plan_sha256=original_sha256,
        original_training_point_count=48, selected_training_point_count=36, deferred_training_point_count=12,
        completion_scope='only_the_36_declared_phase1_training_points', full_training_matrix_complete=False,
        selected_training_minimum_window_seconds=sum(p['repeats']*(p['settle_s']+p['measure_s']) for p in selected),
        minimum_window_seconds=sum(p['repeats']*(p['settle_s']+p['measure_s']) for p in selected+holdout),
        holdout_execution='not_collected_in_this_phase; only_after_new_candidate_freeze',
        coverage_constraints=dict(long_context_batch_max=8, selected_prompt_context_tokens=[5120, 7168],
            actual_contexts_only=True, high_batch_long_context_status='missing_profile',
            excludes=[dict(batch_min=9, context_tokens_min_exclusive=4096, status='missing_profile')],
            forbid_low_batch_long_context_extrapolation_to_high_batch=True,
            forbid_rectangular_domain_union_with_short_context_high_batch=True,
            promotion_requires='new_shape_bounded_fit_and_fresh_independent_holdout'),
        formal_eligible=False)
    phase['domain_note'] = original.get('domain_note', '') + (
        ' Phase one collects only long-context B1/B4/B8. It provides no coverage for B>8 at contexts>4096. '
        'Combining short-context high-batch samples with these low-batch samples must not create a rectangular '
        'high-batch/long-context domain. Actual observed token contexts and new independent holdout remain required.')
    costs = []
    for p in original['training']:
        seconds, method, anchors = prefill_proxy(raw, p['freq_mhz'], p['context_tokens'])
        prefill_work = p['repeats']*p['batch']*seconds
        windows = p['repeats']*(p['settle_s']+p['measure_s'])
        costs.append(dict(freq_mhz=p['freq_mhz'], batch=p['batch'], context_tokens=p['context_tokens'],
            repeats=p['repeats'], single_request_prefill_seconds=seconds, prefill_estimate_method=method,
            prefill_anchors=anchors, serial_prefill_proxy_seconds=prefill_work,
            required_decode_window_seconds=windows, scheduling_proxy_seconds=prefill_work+windows,
            phase1_status='selected' if p['batch'] <= 8 else 'deferred',
            coverage_status='pending_measurement' if p['batch'] <= 8 else 'missing_profile'))

    def summarize(rows):
        return dict(points=len(rows), serial_prefill_proxy_seconds=sum(p['serial_prefill_proxy_seconds'] for p in rows),
            required_decode_window_seconds=sum(p['required_decode_window_seconds'] for p in rows),
            scheduling_proxy_seconds=sum(p['scheduling_proxy_seconds'] for p in rows))

    report = dict(schema=1, evidence_class='cpu_scheduling_estimate', actual_batch_runtime_measured=False,
        formal_eligible=False, system='pdblend', model_id=identity['model_id'], tp=4, pp=1,
        original_plan_sha256=original_sha256, training_source=original['training_source'],
        training_source_sha256=original['training_source_sha256'],
        formula='repeats * batch * measured_or_within_range_interpolated_single_request_prefill_seconds + repeats * (settle_s + measure_s)',
        limitations=['This serial-work proxy is neither a batch-runtime measurement nor a guaranteed lower/upper bound.',
            '5120 prefill uses same-frequency interpolation between measured 4096 and 7168; 7168 is measured directly.',
            'Concurrent chunked prefill, decode interference and admission can change actual batch startup time.',
            'Excludes model load, 16-token barrier, frequency changes, qualification, cancellation, cleanup, retries and queue waits.',
            'Never use this cost estimate as profile calibration or decode-domain evidence.'],
        rows=costs, full_plan=summarize(costs), phase1=summarize([p for p in costs if p['phase1_status']=='selected']),
        deferred=summarize([p for p in costs if p['phase1_status']=='deferred']),
        by_shape=[dict(batch=b, context_tokens=c, **summarize([p for p in costs if p['batch']==b and p['context_tokens']==c]))
                  for b in (1, 4, 8, 256) for c in (5120, 7168)])
    return phase, report


def write_immutable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError(f'output already exists with different bytes: {path}')
    if not path.exists():
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'results/2026-09-23/long-context-plan/Qwen2.5-7B-Instruct-tp4.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    original_bytes = args.plan.read_bytes()
    original = json.loads(original_bytes)
    source = Path(original['training_source'])
    if sha256(source) != original['training_source_sha256']:
        raise ValueError('immutable training raw checksum mismatch')
    phase, report = build_phase1(original, json.loads(source.read_text()), original_path=args.plan,
                                original_sha256=hashlib.sha256(original_bytes).hexdigest())
    write_immutable(args.out/'Qwen2.5-7B-Instruct-tp4-phase1.json', json.dumps(phase, indent=2)+'\n')
    write_immutable(args.out/'cpu-cost-report.json', json.dumps(report, indent=2)+'\n')
    # Preserve the complete original plan verbatim alongside its SHA binding.
    write_immutable(args.out/'original-full-plan.json', original_bytes.decode())
    keys = [k for k in report['rows'][0] if k != 'prefill_anchors']
    table = io.StringIO(); writer = csv.DictWriter(table, fieldnames=keys)
    writer.writeheader(); writer.writerows({k: row[k] for k in keys} for row in report['rows'])
    write_immutable(args.out/'cpu-cost-by-point.csv', table.getvalue())
    lines = ['# 7B TP4 long-context scheduling proxy', '',
        'Single-request prefill measurements estimate serial work; these are not measured batch runtimes or guaranteed bounds. '
        'Model loading, qualification, 16-token barriers, cleanup and other overhead are excluded.', '',
        '| Batch | Prompt context | Points | Prefill proxy (min) | Decode windows (min) | Combined proxy (min) | Phase 1 |',
        '|---:|---:|---:|---:|---:|---:|---|']
    for row in report['by_shape']:
        lines.append(f'| {row["batch"]} | {row["context_tokens"]} | {row["points"]} | '
            f'{row["serial_prefill_proxy_seconds"]/60:.2f} | {row["required_decode_window_seconds"]/60:.2f} | '
            f'{row["scheduling_proxy_seconds"]/60:.2f} | {"selected" if row["batch"]<=8 else "deferred / missing_profile"} |')
    lines += ['', f'Full 48-point proxy: {report["full_plan"]["scheduling_proxy_seconds"]/60:.2f} min. '
        f'Selected 36-point proxy: {report["phase1"]["scheduling_proxy_seconds"]/60:.2f} min. '
        f'Deferred 12-point proxy: {report["deferred"]["scheduling_proxy_seconds"]/60:.2f} min.', '',
        'The full original plan is unchanged. Phase-one completion covers only its 36 declared training points. '
        'B>8 with context>4096 remains missing_profile; low-batch long-context observations must not be combined '
        'with short-context high-batch data to claim an unmeasured rectangular domain.', '']
    write_immutable(args.out/'cpu-cost-report.md', '\n'.join(lines))
    print(json.dumps(dict(output=str(args.out.resolve()), selected_points=36, deferred_points=12,
        original_plan_unchanged=sha256(args.plan)==report['original_plan_sha256'],
        gpu_executed=False, live_queue_modified=False, scheduling_proxy_minutes={
            k:report[k]['scheduling_proxy_seconds']/60 for k in ('full_plan','phase1','deferred')}), indent=2))


if __name__ == '__main__':
    main()
