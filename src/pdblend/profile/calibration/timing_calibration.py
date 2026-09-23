"""Independent B32 timing residual validation on an already resident TP4 engine.

The original candidate, power package, and old failed receipts stay immutable.
This overlay changes only 16 < batch < 64 and preserves the original coverage.
It is deliberately separate from PerfModel promotion until fresh validation.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import statistics
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from pdblend.profile.calibration.core import digest, evaluate_holdout
from pdblend.profile.calibration.decode_fit import predict as base_predict
from pdblend.profile.query.model import PerfModel
from pdblend.profile.calibration.power_calibration import FREQUENCIES, point_key, timing_component, write_immutable
from pdblend.profile.collection.wave import atomic_json
from pdblend.profile.collection.window_sampling import _background, _running, summarize_window

KIND = 'bounded_batch_knee_residual_v1'
PURPOSE = 'independent_decode_timing_overlay_holdout'




def fit_residual(rows, base_spec, *, degree=0):
    """Fit only original B32 observations, grouping repeats for later CV."""
    import numpy as np
    selected = [r for r in rows if r['batch'] == 32]
    if len({r['context_tokens'] for r in selected}) < 2 or degree not in (0, 1):
        raise ValueError('B32 residual needs at least two training context shapes')
    samples = [p for r in selected for p in r['repeats']]
    y = np.asarray([p['step_seconds'] for p in samples])
    c = np.asarray([p['effective_context_tokens'] for p in samples])
    if np.any(y <= 0) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(c)):
        raise ValueError('invalid timing training observations')
    p = np.asarray([base_predict(base_spec, 32, x) for x in c])
    x = np.stack([np.ones_like(c), c/4096], axis=1)[:, :degree+1]
    return np.linalg.lstsq(x/y[:, None], (y-p)/y, rcond=None)[0].tolist()






def source_hashes():
    from pdblend.source_inventory import implementation_hashes as inventory_hashes
    return inventory_hashes()


def resident_binding(profiler):
    environment = profiler.raw['environment']
    uuids = environment.get('gpu_uuids')
    if (not isinstance(uuids, list) or len(uuids) != 4 or len(set(uuids)) != 4 or
            not environment.get('source_hash') or not environment.get('image_digest')):
        raise ValueError('timing requires explicit four physical GPU UUIDs and source/image identity')
    evidence = profiler.raw.get('external_interference') or {}
    root = Path(profiler.out_dir).resolve()
    path = (root/evidence.get('samples_file', '')).resolve()
    if (not evidence.get('complete') or not path.is_relative_to(root) or not path.is_file() or
            digest(path) != evidence.get('samples_sha256')):
        raise ValueError('timing requires checksum-bound resident ProfileWave qualification')
    concurrent = profiler.raw.get('concurrency_environment') or {}
    return dict(environment={key: copy.deepcopy(environment.get(key)) for key in
        ('gpu_uuids', 'source_hash', 'image_digest', 'hardware_id', 'torch', 'cuda', 'vllm')},
        qualification_source_root=str(root), qualification_sha256=evidence['samples_sha256'],
        cohort_id=evidence.get('cohort_id'), measured_mode=evidence.get('measured_mode'),
        lease_id=concurrent.get('lease_id'),
        allocated_gpu_uuids=copy.deepcopy(concurrent.get('allocated_gpu_uuids')),
        clock_protocol=dict(frequencies=list(FREQUENCIES), settle_s=2, measurement_s=5, repeats=3))


def load_package(package):
    package = Path(package)
    manifest = json.loads((package/'manifest.json').read_text())
    for key, name in (('candidate_sha256', 'candidate.json'), ('plan_sha256', 'timing-plan.json')):
        if digest(package/name) != manifest[key]:
            raise ValueError('timing package checksum mismatch')
    if manifest['implementation_sha256'] != source_hashes():
        raise ValueError('timing implementation changed after package freeze')
    for info in manifest['inputs'].values():
        if digest(info['path']) != info['sha256']:
            raise ValueError('immutable timing input changed')
    base = PerfModel.load(manifest['inputs']['base_candidate']['path'])
    candidate = json.loads((package/'candidate.json').read_text())
    model = TimingOverlay(base, candidate)
    plan = json.loads((package/'timing-plan.json').read_text())
    expected = {(f, b, 1024) for f in FREQUENCIES for b in (24, 32, 48)}
    if (len(plan['points']) != 18 or
            {(p['freq_mhz'], p['batch'], p['context_tokens']) for p in plan['points']} != expected or
            candidate['base_candidate_sha256'] != manifest['inputs']['base_candidate']['sha256'] or
            candidate['training_raw_sha256'] != manifest['inputs']['training_raw']['sha256']):
        raise ValueError('timing panel/candidate binding mismatch')
    for p in plan['points']:
        if p['purpose'] != PURPOSE or p['repeats'] != 3 or p['settle_s'] < 2 or p['measure_s'] < 5:
            raise ValueError('timing panel changed sampling gates')
        if (p['context_tokens']+p['max_tokens'] > 8192 or
                p['batch']*(p['context_tokens']+p['max_tokens']) > .9*base.kv_capacity_tokens or
                any(not model.decode_supported(p['batch'], c, p['freq_mhz'])
                    for c in (p['context_tokens'], p['context_tokens']+p['max_tokens']-1))):
            raise ValueError('timing reservation exceeds original coverage/memory')
    return manifest, plan, model


def validate_repeat(root, rep, point, binding):
    root = Path(root)
    path = (root/rep['samples_file']).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != rep['samples_sha256']:
        raise ValueError('timing window checksum/path mismatch')
    data = json.loads(path.read_text())
    if (data['binding'] != binding or data['point'] != point or data['purpose'] != PURPOSE or
            data['repeat'] != rep['repeat'] or data['shared_decode_run'] != rep['shared_decode_run']):
        raise ValueError('timing window identity mismatch')
    measured = summarize_window(token_times=data['token_times_s'], context=point['context_tokens'],
        start_s=data['start_s'], end_s=data['end_s'], power=data['power'], frequency=data['frequency'],
        gpu_count=len(rep['measured_gpu_ids']), settle_s=data['start_s']-data['settle_start_s'],
        measurement_s=point['measure_s'])
    for key, actual in measured.items():
        if isinstance(actual, (int, float)) and not math.isclose(actual, rep[key], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('timing raw summary mismatch: '+key)
    if measured['observed_context_max'] >= point['context_tokens']+point['max_tokens']:
        raise ValueError('timing stream exceeded output reservation')
    return measured


def resume_windows(raw, root, points, binding):
    expected = {point_key(p): p for p in points}
    completed = set()
    def check(repeats, point):
        if len(repeats) > 3 or len({r['samples_sha256'] for r in repeats}) != len(repeats):
            raise ValueError('duplicate/extra timing windows')
        for i, rep in enumerate(repeats):
            if rep['repeat'] != i or (i and rep['start_s'] < repeats[i-1]['end_s']+2-1e-6):
                raise ValueError('timing repeats overlap or omit settle')
            validate_repeat(root, rep, point, binding)
    for row in raw.get('decode', []):
        key = point_key(row)
        if key not in expected or key in completed or len(row['repeats']) != 3:
            raise ValueError('unexpected/duplicate/incomplete timing point')
        check(row['repeats'], expected[key]); completed.add(key)
    for key, repeats in raw.get('pending', {}).items():
        if key not in expected or key in completed:
            raise ValueError('invalid pending timing point')
        check(repeats, expected[key])
    return completed


async def collect_point(profiler, client, gpus, point, *, model, binding, previous=(), on_window=None,
                        _clock=time.time, _sleep=asyncio.sleep, _background_factory=None):
    if (len(gpus) != 4 or len(set(gpus)) != 4 or
            point['batch']*(point['context_tokens']+point['max_tokens']) > .9*profiler.raw['kv_capacity_tokens']):
        raise ValueError('invalid timing GPU group/reservation')
    repeats = list(previous)
    for rep in repeats:
        validate_repeat(profiler.out_dir, rep, point, binding)
    if len(repeats) > 3:
        raise ValueError('extra previous timing windows')
    if len(repeats) < 3:
        run_id = uuid.uuid4().hex
        async with (_background_factory or _background)(profiler, client, point, 'timing-'+point_key(point)+'-'+run_id) as (live, tasks):
            if len(live) != point['batch']:
                raise ValueError('timing live batch differs from plan')
            for i in range(len(repeats), 3):
                settle_start = _clock(); await _sleep(point['settle_s']); _running(tasks)
                sampler = profiler.meter.sampler(gpus); sampler.start(); start = _clock()
                try:
                    await _sleep(point['measure_s']); end = _clock(); _running(tasks)
                finally:
                    sampler.stop()
                if sampler.error:
                    raise RuntimeError('timing sampler failed: '+str(sampler.error))
                data = dict(point=point, binding=binding, purpose=PURPOSE, shared_decode_run=run_id,
                    repeat=i, start_s=start, end_s=end, settle_start_s=settle_start,
                    token_times_s=[list(r.token_times_s) for r in live],
                    power=[r for r in sampler.samples if start <= r[0] <= end],
                    frequency=[r for r in sampler.frequency_samples if start <= r[0] <= end])
                path = profiler.out_dir/'samples'/f'timing-{point_key(point)}-{run_id}-r{i}.json'
                atomic_json(path, data)
                rep = summarize_window(token_times=data['token_times_s'], context=point['context_tokens'],
                    start_s=start, end_s=end, power=data['power'], frequency=data['frequency'],
                    gpu_count=4, settle_s=start-settle_start, measurement_s=point['measure_s'])
                rep.update(freq_mhz=point['freq_mhz'], measured_gpu_ids=list(gpus), repeat=i,
                    shared_decode_run=run_id, samples_file=str(path.relative_to(profiler.out_dir)), samples_sha256=digest(path))
                repeats.append(rep)
                if on_window: on_window(list(repeats))
                validate_repeat(profiler.out_dir, rep, point, binding)
                if any(not model.decode_supported(point['batch'], rep[k], point['freq_mhz'])
                       for k in ('observed_context_min', 'effective_context_tokens', 'observed_context_max')):
                    raise ValueError('timing actual window outside unchanged coverage')
    return dict(freq_mhz=point['freq_mhz'], batch=point['batch'], context_tokens=point['context_tokens'],
        repeats=repeats, independent_holdout=True, formal_eligible=False,
        sampling_method='three_distinct_timing_windows_shared_prefill',
        prefill_runs=len({r['shared_decode_run'] for r in repeats}))


def audit_fresh(raw, out, points, model, binding):
    complete = resume_windows(raw, out, points, binding)
    failures, values = [], []
    if complete != {point_key(p) for p in points}:
        failures.append(dict(metric='timing_matrix_coverage', points=len(complete), expected=len(points)))
    for row in raw.get('decode', []):
        timings = []
        for rep in row['repeats']:
            f, b, ctx = row['freq_mhz'], row['batch'], rep['effective_context_tokens']
            if any(not model.decode_supported(b, rep[k], f)
                   for k in ('observed_context_min', 'effective_context_tokens', 'observed_context_max')):
                raise ValueError('timing evidence outside unchanged coverage')
            predicted = model.step_seconds(b, ctx, f)
            error = abs(predicted/rep['step_seconds']-1)
            value = dict(freq_mhz=f, batch=b, repeat=rep['repeat'], context_tokens=ctx,
                observed=rep['step_seconds'], predicted=predicted, relative_error=error)
            values.append(value); timings.append(rep['step_seconds'])
            if not math.isfinite(error) or error > .10:
                failures.append(dict(value, metric='independent_timing_window', limit=.10))
            if round(rep['mean_freq_mhz']) != f:
                failures.append(dict(metric='timing_frequency_identity', freq_mhz=f, measured=rep['mean_freq_mhz']))
        if len(timings) == 3 and statistics.stdev(timings)/statistics.fmean(timings) > .10:
            failures.append(dict(metric='timing_repeat_noise', point=point_key(row), limit=.10))
    return dict(passed=not failures, failures=failures, points=values,
        timing_max=max((v['relative_error'] for v in values), default=None), formal_eligible=False,
        fit_performed=False, scope='18_fresh_points_54_windows_all_errors_at_most_10_percent')


def reuse_original(manifest, model):
    """Retain all old gates; replace precisely six affected B32 decode rows."""
    inputs = manifest['inputs']
    original = Path(inputs['original_raw']['path'])
    raw = json.loads(original.read_text())
    plan = json.loads(Path(inputs['original_manifest']['path']).read_text())['plan']
    wanted = {(f, 32, 1024) for f in FREQUENCIES}
    affected = [r for r in raw['decode'] if weight(r['batch'])]
    if {(r['freq_mhz'], r['batch'], r['context_tokens']) for r in affected} != wanted or len(affected) != 6:
        raise ValueError('original matrix changed: exactly six B32 points must be replaced')
    retained = [r for r in raw['decode'] if not weight(r['batch'])]
    if len(retained) != 42:
        raise ValueError('original unchanged matrix requires 42 retained points')
    for row in retained:
        for rep in row['repeats']:
            args = row['batch'], rep['effective_context_tokens'], row['freq_mhz']
            if model.step_seconds(*args) != model.base.step_seconds(*args):
                raise ValueError('retained timing prediction is not exactly unchanged')
    raw['decode'] = retained
    plan = copy.deepcopy(plan)
    plan['decode'] = [r for r in plan['decode'] if not weight(r['batch'])]
    audit = timing_component(evaluate_holdout(raw, model.base, original.parent, expected_plan=plan))
    audit.update(retained_decode_points=42, replaced_decode_points=6,
        original_failed_rows_preserved=True, original_completion_unchanged=True,
        mixed_gate_uses_measured_base_step_not_model_prediction=True,
        old_prefill_and_all_12_mixed_measurements_retained=True,
        original_decode_matrix_count=48, required_fresh_decode_points=18)
    return audit


async def _collect_existing(*, profiler, client, gpus, package, out):
    """Called inside the existing EngineClient and ProfileWave measurement."""
    package, out = Path(package), Path(out)
    manifest, plan, model = load_package(package)
    if (len(gpus) != 4 or
            (profiler.model_spec.model_hash, profiler.model_spec.tokenizer_hash) !=
            (manifest['model_hash'], manifest['tokenizer_hash']) or
            (profiler.raw.get('system'), profiler.raw.get('model_id'), profiler.raw.get('tp'), profiler.raw.get('pp')) !=
            ('pdblend', 'Qwen2.5-32B-Instruct', 4, 1)):
        raise ValueError('resident timing engine differs from frozen 32B TP4 package')
    for key in ('image_digest', 'vllm', 'torch', 'cuda', 'hardware_id'):
        if not manifest['training_environment'].get(key) or manifest['training_environment'][key] != profiler.raw['environment'].get(key):
            raise ValueError('timing engine environment mismatch: '+key)
    binding = dict(candidate_sha256=manifest['candidate_sha256'], plan_sha256=manifest['plan_sha256'],
        manifest_sha256=digest(package/'manifest.json'))
    runtime_binding = resident_binding(profiler)
    raw = dict(schema=1, purpose=PURPOSE, system='pdblend', model_id='Qwen2.5-32B-Instruct', tp=4, pp=1,
        model_hash=manifest['model_hash'], tokenizer_hash=manifest['tokenizer_hash'], binding=binding,
        environment=copy.deepcopy(profiler.raw['environment']), independent_holdout=True, resident_binding=runtime_binding,
        kv_capacity_tokens=profiler.raw['kv_capacity_tokens'], measured_gpu_ids=list(gpus),
        parallel_interference=copy.deepcopy(profiler.raw.get('parallel_interference')),
        external_interference=copy.deepcopy(profiler.raw.get('external_interference')),
        concurrency_environment=copy.deepcopy(profiler.raw.get('concurrency_environment')),
        parent_power_artifact_root=str(Path(profiler.out_dir).resolve()),
        parent_evidence_paths_are_relative_to='parent_power_artifact_root',
        decode=[], pending={}, formal_eligible=False, fit_performed=False)
    if (out/'raw.json').exists():
        raw = json.loads((out/'raw.json').read_text())
        if (raw['binding'] != binding or raw['measured_gpu_ids'] != list(gpus) or
                raw.get('resident_binding') != runtime_binding):
            raise ValueError('timing checkpoint candidate or lease differs')
    local = SimpleNamespace(out_dir=out, raw=raw, meter=profiler.meter)
    def checkpoint(): atomic_json(out/'raw.json', raw)
    complete = resume_windows(raw, out, plan['points'], binding)
    result = dict(status='running', complete=False, timing_passed=False, formal_eligible=False, energy_comparable=False,
        scope=PURPOSE, independent_holdout=True, fit_performed=False, binding=binding, started_s=time.time())
    try:
        previous = None
        for point in plan['points']:
            key = point_key(point)
            if key in complete: continue
            if previous != point['freq_mhz']:
                profiler._lock(point['freq_mhz'], gpus)
                previous = point['freq_mhz']; await asyncio.sleep(2)
            def on_window(repeats):
                raw['pending'][key] = repeats; checkpoint()
            row = await collect_point(local, client, gpus, point, model=model, binding=binding,
                previous=raw['pending'].get(key, ()), on_window=on_window)
            raw['decode'].append(row); raw['pending'].pop(key, None); complete.add(key); checkpoint()
            print(f'timing overlay f={point["freq_mhz"]} B={point["batch"]} windows=3', flush=True)
        checkpoint(); load_package(package)
        fresh = audit_fresh(raw, out, plan['points'], model, binding)
        retained = reuse_original(manifest, model)
        write_immutable(out/'fresh-timing-audit.json', fresh)
        write_immutable(out/'retained-timing-audit.json', retained)
        audit = dict(timing_passed=fresh['passed'] and retained['passed'], fresh=fresh, retained=retained,
            original_completion_unchanged=True, formal_eligible=False, power_not_evaluated=True,
            original_decode_gate_points=48, retained_original_decode_points=42,
            fresh_replacement_points=6, fresh_interpolation_points=12,
            candidate_sha256=manifest['candidate_sha256'], raw_sha256=digest(out/'raw.json'),
            original_inputs=manifest['inputs'], environment=raw['environment'])
        write_immutable(out/'timing-composite-audit.json', audit)
        result.update(status='completed', complete=True, timing_passed=audit['timing_passed'],
            composite_receipt_sha256=digest(out/'timing-composite-audit.json'), measured_decode_points=len(raw['decode']),
            old_timing_failure_retained=True, power_not_evaluated=True)
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        checkpoint(); result.update(finished_s=time.time(), raw_sha256=digest(out/'raw.json'))
        atomic_json(out/'completion.json', result)
    result = dict(result, receipt_sha256=digest(out/'completion.json'))
    return result


async def collect_existing(*, profiler, client, gpus, package, out):
    """Fail closed even when identity validation fails before sampling starts."""
    try:
        return await _collect_existing(profiler=profiler, client=client, gpus=gpus, package=package, out=out)
    except BaseException as exc:
        receipt = Path(out)/'completion.json'
        if not receipt.exists():
            atomic_json(receipt, dict(status='failed', complete=False, timing_passed=False,
                formal_eligible=False, energy_comparable=False, fit_performed=False, scope=PURPOSE,
                error=f'{type(exc).__name__}: {exc}', package=str(package), finished_s=time.time()))
        raise

from pdblend.profile.query.timing import weight, validate_candidate, TimingOverlay
