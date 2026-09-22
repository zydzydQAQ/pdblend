"""Profile evidence validation and experiment gates; no changes to planner models."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from .merge import sha256
from .profiler import DECODE_BATCHES, DECODE_CONTEXTS, DECODE_STEPS, PREFILL_INPUTS, MIXED_PROBES

FREQS = (900, 1200, 1500, 1800, 2100, 2520)
SEEDS = (701, 1701, 2701)


def relative_error(predicted, observed):
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in (predicted, observed)) or observed <= 0:
        raise ValueError('relative error requires finite values and positive observed value')
    return abs(predicted - observed) / observed


def quality_audit(raw, model, directory: Path):
    failures, points = [], []
    def fail(metric, **details):
        failures.append(dict(metric=metric, **details))
    env = raw.get('environment', {})
    for key in ('image_digest', 'source_hash', 'vllm', 'torch', 'cuda', 'timestamp_utc', 'gpu_uuids'):
        if not env.get(key):
            fail('environment', missing=key)
    if not isinstance(env.get('gpu_uuids'), list) or any(not x.startswith('GPU-') for x in env.get('gpu_uuids', [])):
        fail('environment', reason='invalid_gpu_uuids')
    if raw.get('schema') != 2:
        fail('schema', observed=raw.get('schema'))
    cfg = raw.get('config', {})
    for key, minimum in (('decode_repeats', 3), ('decode_settle_s', 2), ('decode_measure_s', 5)):
        if cfg.get(key, 0) < minimum:
            fail('config', field=key, minimum=minimum)
    if set(raw.get('freqs', [])) != set(FREQS) or set(model.freqs) != set(FREQS):
        fail('frequency_coverage', required=list(FREQS))
    cap = raw.get('kv_capacity_tokens', 0)
    if cap <= 0 or raw.get('kv_bytes_per_token', 0) <= 0:
        fail('kv', reason='missing_positive_capacity')
    expected = {
        'prefill': {(f, n) for f in FREQS for n in PREFILL_INPUTS},
        'decode': {(f, c, b) for f in FREQS for c in DECODE_CONTEXTS for b in DECODE_BATCHES
                   if not cap or b * (c + DECODE_STEPS) <= .9 * cap},
        'mixed': {(f, b, c) for f in (1500, 2100, 2520) for b, c in MIXED_PROBES},
        'transfer': {(n,) for n in (512, 2048, 7168)},
    }
    fields = {'prefill': ('freq_mhz', 'input_tokens'), 'decode': ('freq_mhz', 'context_tokens', 'batch'),
              'mixed': ('freq_mhz', 'batch', 'chunk_tokens'), 'transfer': ('input_tokens',)}
    for section, required in expected.items():
        keys = [tuple(x.get(k) for k in fields[section]) for x in raw.get(section, [])]
        missing = required - set(keys)
        if missing or len(keys) != len(set(keys)):
            fail(section + '_coverage', missing=sorted(missing), duplicate_count=len(keys) - len(set(keys)))
    required_static = {f'active_idle@{f}' for f in FREQS} | {'parked', 'off', 'active_idle_reset'}
    if not required_static <= set(raw.get('static', {})):
        fail('static_coverage', missing=sorted(required_static - set(raw.get('static', {}))))

    def check_file(row, point):
        name = row.get('samples_file')
        if not name or not (directory / name).is_file():
            fail('sample_evidence', point=point, file=name)
            return None
        path = directory / name
        if not row.get('samples_sha256') or sha256(path) != row['samples_sha256']:
            fail('sample_checksum', point=point, file=name)
        return json.loads(path.read_text())

    for shard in raw.get('shards', []):
        p = directory / shard['path']
        if not p.is_file() or sha256(p) != shard['sha256']:
            fail('shard_checksum', path=str(p))
    for d in raw.get('decode', []):
        point = {k: d[k] for k in fields['decode']}
        repeats = d.get('repeats', [])
        if len(repeats) < max(3, cfg.get('decode_repeats', 3)):
            fail('decode_repeats', point=point, actual=len(repeats))
        for rep in repeats:
            data = check_file(rep, point)
            if rep.get('steady_window_s', 0) < 5 or rep.get('end_s', 0) - rep.get('start_s', 0) < 5:
                fail('steady_window', point=point)
            if rep.get('min_steps', 0) < 8 or rep.get('power_samples', 0) < 2 or rep.get('frequency_samples', 0) < 1:
                fail('decode_evidence', point=point)
            if data is not None:
                if len(data.get('power', [])) != rep.get('power_samples') or len(data.get('frequency', [])) != rep.get('frequency_samples'):
                    fail('sample_counts', point=point)
        powers = d.get('power_repeats', [])
        if len(powers) != len(repeats) or len(d.get('step_repeats', [])) != len(repeats):
            fail('repeat_summary', point=point)

    # Recompute residuals from model predictions and measured points, not stored quality fields.
    metrics = {}
    for section, timing in (('prefill', 'prefill_time'), ('decode', 'decode_time')):
        for r in raw.get(section, []):
            f = r['freq_mhz']
            observed = r['seconds'] if section == 'prefill' else r['step_seconds']
            pred = model.prefill_seconds(r['input_tokens'], f) if section == 'prefill' else model.step_seconds(
                r['batch'], r.get('effective_context_tokens', r['context_tokens']), f)
            err = relative_error(pred, observed)
            point = dict(metric=f'{timing}@{f}', **{k: r[k] for k in fields[section]},
                         predicted=pred, observed=observed, relative_error=err)
            points.append(point)
            metrics.setdefault(point['metric'], []).append(err)
            if err > .10:
                fail(point['metric'], reason='max_relative_error', point=point, limit=.10)
    for f in FREQS:
        stable = []
        for d in raw.get('decode', []):
            if d['freq_mhz'] != f or d['batch'] < 4:
                continue
            values = d.get('power_repeats', [])
            cv = statistics.stdev(values) / statistics.mean(values) if len(values) >= 3 and min(values) > 0 else float('inf')
            if cv <= .10:
                err = relative_error(model.decode_power_w(d['batch'], f), d['power_w'])
                stable.append(d)
                metrics.setdefault(f'decode_power@{f}', []).append(err)
                points.append(dict(metric=f'decode_power@{f}', freq_mhz=f, batch=d['batch'],
                                   context_tokens=d['context_tokens'], relative_error=err,
                                   observed=d['power_w'], predicted=model.decode_power_w(d['batch'], f)))
        if len({d['batch'] for d in stable}) < 2:
            fail(f'decode_power@{f}', reason='insufficient_stable_batches')
    quality = {k: dict(mape=statistics.mean(v), max=max(v), samples=len(v)) for k, v in metrics.items()}
    for k, q in quality.items():
        if k.startswith('decode_power@') and (q['mape'] > .10 or q['max'] > .15):
            fail(k, reason='planner_power_error', value=q,
                 points=[p for p in points if p['metric'] == k and p['relative_error'] > .15])
    bases = {(r['freq_mhz'], r['batch'], r['context_tokens']): r for r in raw.get('decode', [])}
    prefills = {(r['freq_mhz'], r['input_tokens']): r for r in raw.get('prefill', [])}
    mixed_errors = []
    mixed_rows = []
    for row in raw.get('mixed', []):
        base = bases.get((row['freq_mhz'], row['batch'], row['context_tokens']))
        alone = prefills.get((row['freq_mhz'], row['chunk_tokens']))
        if not row.get('valid') or not base or not alone or not row.get('base_step_s') or row['base_step_s'] <= 0:
            fail('mixed_base_or_validity', point=row)
            continue
        if not math.isclose(row['base_step_s'], base['step_seconds'], rel_tol=1e-9):
            fail('mixed_base_mismatch', point=row)
        data = check_file(row, {k: row[k] for k in fields['mixed']})
        probes = row.get('probe_ttft_samples', [])
        if len(probes) < 3 or row.get('stable_tokens', 0) < 16:
            fail('mixed_stability', point=row)
        if data:
            start = row.get('stable_start_s', 0)
            backgrounds = data.get('background', [])
            if len(backgrounds) != row['batch'] or any(sum(t <= start for t in stream) < 17 for stream in backgrounds):
                fail('mixed_barrier_evidence', point={k: row[k] for k in fields['mixed']})
            if any(p['start_s'] < start for p in data.get('probes', [])):
                fail('mixed_probe_before_barrier', point=row)
        pred = base['step_seconds'] + alone['seconds']
        error = relative_error(pred, row['probe_ttft_s'])
        mixed_errors.append(error)
        mixed_rows.append(dict(freq_mhz=row['freq_mhz'], batch=row['batch'], chunk_tokens=row['chunk_tokens'],
                               predicted=pred, observed=row['probe_ttft_s'], relative_error=error,
                               probe_cv=statistics.stdev(probes) / statistics.mean(probes) if len(probes) >= 2 else None))
    mixed = dict(valid=len(mixed_errors), invalid=len(raw.get('mixed', [])) - len(mixed_errors), rows=mixed_rows,
                 missing_base=sum(not x.get('base_step_s') for x in raw.get('mixed', [])))
    if mixed_errors:
        mixed.update(median=statistics.median(mixed_errors), mape=statistics.mean(mixed_errors), max=max(mixed_errors))
        if mixed['median'] > .15:
            fail('mixed_additive', value=mixed['median'], limit=.15)
    else:
        fail('mixed_additive', reason='no_valid_points')
    return dict(passed=not failures, failures=failures, quality=quality, mixed=mixed, points=points,
                error_definition='abs(predicted-observed)/observed')


def m2_gate(rows):
    """Rows must be completed, provenance-checked results for exactly the required seeds."""
    reasons = []
    if len(rows) != 3 or {r.get('seed') for r in rows} != set(SEEDS):
        reasons.append('requires exactly seeds 701, 1701, 2701')
    for r in rows:
        checks = (r.get('complete') is True, r.get('joint_slo_rate', -1) >= .9,
                  r.get('ttft_p99', float('inf')) <= 5, r.get('tpot_p99', float('inf')) <= .15,
                  r.get('power_error') is not None and abs(r['power_error']) <= .05)
        if not all(checks):
            reasons.append(f'seed {r.get("seed")} failed completion/SLO/tail/power checks')
    return dict(passed=not reasons, reasons=reasons, min_m_instances=2 if not reasons else 4)
