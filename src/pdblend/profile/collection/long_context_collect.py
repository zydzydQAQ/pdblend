"""Collect only an immutable plan's incremental PDBlend training points.

One engine remains resident across all frequencies and shapes.  The collector
never loads a fitted predictor, consumes the plan's holdout points, or promotes
its measurements to a formal profile.  GPU work starts only through the CLI.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from pdblend.profile.long_context_plan import FREQUENCIES
from pdblend.profile.collection.wave import atomic_json
from pdblend.profile.collection.window_sampling import _EarlyEnd, _background, _running, summarize_window


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_key(point):
    return f'{point["freq_mhz"]}-{point["context_tokens"]}-{point["batch"]}'


def validate_training_plan(plan, source_raw):
    if (plan.get('system') != 'pdblend' or source_raw.get('system') != 'pdblend' or
            plan.get('pp') != 1 or plan.get('tp') not in (1, 2, 4) or
            any(plan.get(k) != source_raw.get(k) for k in ('model_id', 'tp', 'pp'))):
        raise ValueError('incremental collector requires the same independent PDBlend PP1 identity')
    if plan.get('fit_existing_holdout') is not False:
        raise ValueError('training plan must explicitly forbid fitting existing holdout observations')
    points = plan.get('training')
    if not isinstance(points, list) or not points:
        raise ValueError('no declared training points')
    found = set()
    for point in points:
        if point.get('purpose') != 'training_extension':
            raise ValueError('collector cannot consume holdout or unclassified points')
        if any(type(point.get(k)) is not int for k in ('freq_mhz', 'batch', 'context_tokens', 'max_tokens', 'repeats')):
            raise ValueError('training point integer fields are invalid')
        if (point['freq_mhz'] not in FREQUENCIES or point['batch'] < 1 or
                not 4096 < point['context_tokens'] < 8192 or point['max_tokens'] < 25 or
                point['context_tokens'] + point['max_tokens'] > 8192 or point['repeats'] < 3):
            raise ValueError('illegal training frequency, shape, output reservation or repeat count')
        if any(not isinstance(point.get(k), (int, float)) or not math.isfinite(point[k]) or point[k] < floor
               for k, floor in (('settle_s', 2), ('measure_s', 5))):
            raise ValueError('training windows cannot shorten settle or measurement gates')
        key = point_key(point)
        if key in found:
            raise ValueError('duplicate training point')
        found.add(key)
    if {x['freq_mhz'] for x in points} != set(FREQUENCIES):
        raise ValueError('training extension must declare all six frequency tiers')
    return sorted(points, key=lambda p: (p['freq_mhz'], p['context_tokens'], p['batch']))


def point_capacity_error(point, measured_capacity):
    if not isinstance(measured_capacity, (int, float)) or not math.isfinite(measured_capacity) or measured_capacity <= 0:
        return 'missing_measured_kv_capacity'
    if point['batch'] * (point['context_tokens'] + point['max_tokens']) > .9 * measured_capacity:
        return 'full_output_reservation_exceeds_measured_kv_capacity'
    return None


def validate_repeat(root, row, *, expected_point=None, expected_plan_sha256=None):
    name = row.get('samples_file')
    path = (root / name).resolve() if isinstance(name, str) else root
    if (not name or not path.is_relative_to(root.resolve()) or not path.is_file() or
            digest(path) != row.get('samples_sha256')):
        raise ValueError('incomplete or corrupted training-window evidence')
    evidence = json.loads(path.read_text())
    if any(evidence['point'].get(k) != row.get(k) for k in ('batch', 'context_tokens', 'freq_mhz')):
        raise ValueError('training window belongs to another shape')
    if expected_point is not None and evidence['point'] != expected_point:
        raise ValueError('training window belongs to another declared point')
    if expected_plan_sha256 is not None and evidence.get('plan_sha256') != expected_plan_sha256:
        raise ValueError('training window belongs to another immutable plan')
    if evidence.get('repeat')!=row.get('repeat') or evidence.get('purpose')!=evidence['point']['purpose']:
        raise ValueError('window repeat/purpose differs from raw evidence')
    reconstructed = summarize_window(token_times=evidence['token_times_s'], context=evidence['prompt_context_tokens'],
        start_s=evidence['start_s'], end_s=evidence['end_s'], power=evidence['power'], frequency=evidence['frequency'],
        gpu_count=len(row['measured_gpu_ids']), settle_s=evidence['start_s']-evidence['settle_start_s'],
        measurement_s=evidence['point']['measure_s'])
    for key,value in reconstructed.items():
        matches=(math.isclose(row[key],value,rel_tol=1e-9,abs_tol=1e-9)
                 if isinstance(value,(int,float)) else row[key]==value)
        if not matches:
            raise ValueError('training window summary differs from immutable raw samples')


def validate_repeats(root,repeats,point,plan_sha,*,complete=False):
    if ((complete and len(repeats)!=point['repeats']) or len(repeats)>point['repeats'] or
        [r.get('repeat') for r in repeats]!=list(range(len(repeats))) or
        len({r.get('samples_file') for r in repeats})!=len(repeats) or
        len({r.get('samples_sha256') for r in repeats})!=len(repeats)):
        raise ValueError('measurement repeats must have unique ordered indexes and artifacts')
    for rep in repeats:validate_repeat(root,rep,expected_point=point,expected_plan_sha256=plan_sha)


def resume_points(raw, root, points):
    expected = {point_key(p): p for p in points}
    plan_sha = raw.get('measurement_plan_sha256', raw.get('training_plan_sha256'))
    complete = set()
    for row in raw.get('decode', []):
        key = point_key(row)
        if key not in expected or key in complete or len(row.get('repeats', [])) != expected[key]['repeats']:
            raise ValueError('unexpected, duplicate or partial completed training point')
        validate_repeats(root,row['repeats'],expected[key],plan_sha,complete=True)
        complete.add(key)
    for key, repeats in raw.get('decode_pending', {}).items():
        if key not in expected or key in complete or len(repeats) > expected[key]['repeats']:
            raise ValueError('invalid pending training-window checkpoint')
        validate_repeats(root,repeats,expected[key],plan_sha)
    return complete


async def collect_bounded_decode_point(profiler, client, gpus, point, *, purpose,
                                  previous=(), on_window=None,
                                  _clock=time.time, _sleep=asyncio.sleep, _background_factory=None):
    """Three bounded independent prefills; no predictor or extrapolation path."""
    if purpose not in ('training_extension', 'independent_holdout_repair') or point.get('purpose') != purpose:
        raise ValueError('bounded sample purpose must match its immutable point')
    if point_capacity_error(point, profiler.raw.get('kv_capacity_tokens')):
        raise ValueError('training point does not fit live measured KV capacity')
    if len(gpus) != profiler.tp or len(set(gpus)) != len(gpus):
        raise ValueError('training point requires exactly one physical TP group')
    factory = _background_factory or _background
    repeats = list(previous)
    plan_sha = profiler.raw.get('measurement_plan_sha256', profiler.raw.get('training_plan_sha256'))
    validate_repeats(profiler.out_dir,repeats,point,plan_sha)
    if len(repeats) > point['repeats']:
        raise ValueError('too many resumed measurement windows')
    for index in range(len(repeats), point['repeats']):
        prefix = 'longctx' if purpose == 'training_extension' else 'holdout-repair'
        tag = f'{prefix}-d-{point_key(point)}-r{index}'
        async with factory(profiler, client, point, tag) as (live, tasks):
            settle_start = _clock()
            await _sleep(point['settle_s'])
            _running(tasks)
            sampler = profiler.meter.sampler(gpus)
            sampler.start()
            start = _clock()
            try:
                await _sleep(point['measure_s'])
                end = _clock()
                _running(tasks)
            finally:
                sampler.stop()
            if sampler.error:
                raise RuntimeError(f'training sampler failed: {sampler.error}')
            evidence = dict(point=dict(point), repeat=index, prompt_context_tokens=point['context_tokens'],
                start_s=start, end_s=end, settle_start_s=settle_start,
                token_times_s=[list(r.token_times_s) for r in live],
                power=[x for x in sampler.samples if start <= x[0] <= end],
                frequency=[x for x in sampler.frequency_samples if start <= x[0] <= end],
                purpose=purpose, independent_holdout=purpose == 'independent_holdout_repair',
                profile_key=profiler.raw.get('profile_key'), plan_sha256=plan_sha)
            evidence['measured_kv_capacity_tokens'] = profiler.raw['kv_capacity_tokens']
            path = profiler.out_dir / 'samples' / f'{tag}.json'
            atomic_json(path, evidence)
            row = summarize_window(token_times=evidence['token_times_s'], context=point['context_tokens'],
                start_s=start, end_s=end, power=evidence['power'], frequency=evidence['frequency'],
                gpu_count=len(gpus), settle_s=start-settle_start, measurement_s=point['measure_s'])
            if len(live) != point['batch'] or row['observed_context_max'] >= point['context_tokens']+point['max_tokens']:
                raise ValueError('observed stream shape exceeded bounded training reservation')
            row.update(measured_gpu_ids=list(gpus), parallel_layout=profiler.parallel_layout,
                concurrency=point['batch'], freq_mhz=point['freq_mhz'], repeat=index,
                measured_kv_capacity_tokens=profiler.raw['kv_capacity_tokens'],
                samples_file=str(path.relative_to(profiler.out_dir)), samples_sha256=digest(path))
        repeats.append(row)
        if on_window is not None:
            on_window(list(repeats))
    powers = [r['power_w'] for r in repeats]
    return dict(batch=point['batch'], context_tokens=point['context_tokens'], freq_mhz=point['freq_mhz'],
        concurrency=point['batch'], max_tokens=point['max_tokens'],
        effective_context_tokens=statistics.median(r['effective_context_tokens'] for r in repeats),
        observed_context_min=min(r['observed_context_min'] for r in repeats),
        observed_context_max=max(r['observed_context_max'] for r in repeats),
        step_seconds=statistics.median(r['step_seconds'] for r in repeats), power_w=statistics.median(powers),
        step_repeats=[r['step_seconds'] for r in repeats], power_repeats=powers, repeats=repeats,
        power_repeat_cv=statistics.stdev(powers)/statistics.fmean(powers) if min(powers) > 0 else None,
        steady_window_s=min(r['steady_window_s'] for r in repeats), steps=min(r['steps'] for r in repeats),
        power_samples=sum(r['power_samples'] for r in repeats), frequency_samples=sum(r['frequency_samples'] for r in repeats),
        measured_gpu_ids=list(gpus), parallel_layout=profiler.parallel_layout,
        evidence_class=purpose, sampling_method='bounded_separate_prefill',
        measured_kv_capacity_tokens=profiler.raw['kv_capacity_tokens'],
        independent_holdout=purpose == 'independent_holdout_repair', formal_eligible=False)


async def collect_training_point(*args, **kwargs):
    return await collect_bounded_decode_point(*args, purpose='training_extension', **kwargs)


def run(*, plan_path: Path, training_raw_path: Path | None, model_path: str, gpus: list[int],
        base_port: int, out: Path, resident_followup=None):
    from pdblend.profile.collection.profiler import Profiler, _load_flock
    from pdblend.profile.collection.wave import ProfileWave
    from pdblend.engine.launcher import Fleet
    from pdblend.engine.client import EngineClient

    plan = json.loads(plan_path.read_text())
    source_path = training_raw_path or Path(plan['training_source'])
    if digest(source_path) != plan['training_source_sha256']:
        raise ValueError('immutable training source checksum mismatch')
    source_raw = json.loads(source_path.read_text())
    points = validate_training_plan(plan, source_raw)
    profiler = Profiler(model_path, gpus, tp=plan['tp'], pp=1, system='pdblend', out_dir=out,
                        hardware_id='8xL20-lease', base_port=base_port, kv_connector='P2pNcclConnector')
    if len(profiler.specs) != 1:
        raise ValueError('long-context collector requires one resident TP group')
    if ((profiler.model_spec.model_id, profiler.model_spec.model_hash, profiler.model_spec.tokenizer_hash) !=
            (plan['model_id'], source_raw['model_hash'], source_raw['tokenizer_hash'])):
        raise ValueError('live model/tokenizer identity differs from independent training source')
    binding = dict(plan_sha256=digest(plan_path), training_source_sha256=digest(source_path),
                   evidence_class='training_extension', independent_holdout=False)
    profiler.raw['config']['incremental_long_context'] = True
    profiler.raw['training_extension'] = binding
    profiler.raw['training_plan_sha256'] = binding['plan_sha256']
    if (out/'raw.json').is_file():
        profiler.resume()
        if profiler.raw.get('training_extension') != binding:
            raise ValueError('training checkpoint belongs to another immutable plan/source')
    complete = resume_points(profiler.raw, out, points)
    profiler.raw.setdefault('decode_pending', {})
    profiler.raw.setdefault('missing_training_points', {})
    atomic_json(out/'training-plan.json', plan)
    wave = ProfileWave.from_environment()
    if wave is None:
        raise ValueError('long-context training requires coordinated ProfileWave qualification')
    result = dict(status='running', complete=False, formal_eligible=False, energy_comparable=False,
        system='pdblend', model_id=plan['model_id'], tp=plan['tp'], pp=1, started_s=time.time(),
        evidence_class='training_evidence', independent_holdout=False, training_extension=binding,
        fit_performed=False, holdout_points_consumed=0, declared_training_points=len(points))
    try:
        with Fleet(profiler.specs, out/'logs') as fleet:
            with _load_flock():
                fleet.start_all()
            instance = fleet[profiler.specs[0].instance_id]
            profiler.raw['kv_capacity_tokens'] = profiler._kv_capacity(instance)
            if profiler.raw['kv_capacity_tokens'] <= 0:
                raise RuntimeError('engine did not report measured KV capacity')
            profiler.raw.setdefault('resident_engine_capacity_history', []).append(dict(
                started_s=time.time(), kv_capacity_tokens=profiler.raw['kv_capacity_tokens']))

            async def sample():
                await wave.qualify_external(profiler, fleet)
                if resident_followup is not None and hasattr(resident_followup,'after_qualification'):
                    resident_followup.after_qualification(profiler)
                async with wave.measurement():
                    async with EngineClient(instance.spec.instance_id, instance.spec.base_url) as client:
                        previous_frequency = None
                        for point in points:
                            key = point_key(point)
                            if key in complete:
                                continue
                            reason = point_capacity_error(point, profiler.raw['kv_capacity_tokens'])
                            if reason:
                                profiler.raw['missing_training_points'][key] = dict(
                                    status='missing_profile', reason=reason, point=point)
                                profiler._checkpoint()
                                continue
                            if previous_frequency != point['freq_mhz']:
                                profiler._lock(point['freq_mhz'], instance.spec.gpus)
                                previous_frequency = point['freq_mhz']
                                await asyncio.sleep(2)

                            def checkpoint(repeats):
                                qualifier=profiler.raw.get('active_long_training_qualification')
                                if qualifier:
                                    for rep in repeats:rep.setdefault('qualification_sha256',qualifier)
                                profiler.raw['decode_pending'][key] = repeats
                                profiler._checkpoint()

                            try:
                                row = await collect_training_point(profiler, client, instance.spec.gpus, point,
                                    previous=profiler.raw['decode_pending'].get(key, ()), on_window=checkpoint)
                            except _EarlyEnd as exc:
                                profiler.raw['missing_training_points'][key] = dict(status='missing_profile',
                                    reason='bounded_output_budget_exhausted', error=str(exc), point=point)
                                profiler._checkpoint()
                                continue
                            profiler.raw['decode'].append(row)
                            profiler.raw['decode_pending'].pop(key, None)
                            profiler.raw['missing_training_points'].pop(key, None)
                            complete.add(key)
                            profiler._checkpoint()
                            print(f'training TP{plan["tp"]} f={point["freq_mhz"]} B={point["batch"]} '
                                  f'context={point["context_tokens"]} observed={row["effective_context_tokens"]:.1f} '
                                  f'step={row["step_seconds"]:.6f}s', flush=True)
                        if resident_followup is not None and len(complete)==len(points):
                            profiler._checkpoint()
                            result['resident_followup']=await resident_followup(
                                profiler=profiler,client=client,gpus=instance.spec.gpus)
            asyncio.run(sample())
        if digest(plan_path) != binding['plan_sha256'] or digest(source_path) != binding['training_source_sha256']:
            raise RuntimeError('immutable plan or training source changed during collection')
        resume_points(profiler.raw, out, points)
        all_done = len(complete) == len(points)
        if resident_followup is not None:
            result['endpoint_training_complete']=all_done
            all_done=all_done and result.get('resident_followup',{}).get('complete') is True
            result['extended_calibration_passed']=result.get('resident_followup',{}).get('calibration_passed',False)
        result.update(status='passed' if all_done else 'inconclusive', complete=all_done,
                      completed_training_points=len(complete), missing_training_points=profiler.raw['missing_training_points'])
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        wave.write('error', dict(error=result['error']))
    finally:
        profiler._checkpoint()
        profiler.meter.reset_all()
        result.update(finished_s=time.time(), raw_sha256=digest(out/'raw.json'),
            missing_gates=['extended_candidate_fit', 'fresh_independent_holdout', 'full_profile_quality_audit',
                           'native_system_mechanisms', 'campaign_acceptance'])
        if resident_followup is not None and hasattr(resident_followup,'progress_metadata'):
            progress=resident_followup.progress_metadata();result.update(progress)
            if progress.get('candidate_fit_performed') is True:result['missing_gates'].remove('extended_candidate_fit')
            if progress.get('fresh_holdout_complete') is True:
                result['missing_gates'].remove('fresh_independent_holdout')
                if progress.get('fresh_holdout_calibration_passed') is not True:
                    result['missing_gates'].append('extended_candidate_calibration')
        atomic_json(out/'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--training-raw', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--base-port', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(plan_path=args.plan, training_raw_path=args.training_raw, model_path=args.model,
                 gpus=args.gpus, base_port=args.base_port, out=args.out)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
