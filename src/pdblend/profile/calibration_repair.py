"""Measure only four bounded 14B TP1 holdout repairs against an unchanged fit.

Original raw/samples remain read-only.  The combined audit uses explicit evidence
roots and retains superseded outside-domain observations as historical evidence.
No prefill grid, mixed grid, fitting, or full holdout rerun occurs.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
from pathlib import Path

from .calibration import _checkpoint_points, digest, evaluate_holdout
from .long_context_collect import collect_bounded_decode_point, point_capacity_error, point_key, resume_points
from .model import PerfModel
from .wave import atomic_json

REPAIR_FREQUENCIES = (1500, 1800, 2100, 2520)


def repair_points(plan, model):
    if (plan.get('system'), plan.get('model_id'), plan.get('tp'), plan.get('pp')) != (
            'pdblend', 'Qwen2.5-14B-Instruct', 1, 1):
        raise ValueError('this bounded repair is only for the declared 14B TP1 PDBlend holdout')
    if (model.system, Path(model.model).name, model.tp, model.pp) != (
            'pdblend', plan['model_id'], 1, 1):
        raise ValueError('repair and frozen candidate identities differ')
    if plan.get('fit_existing_holdout') is not False or plan.get('independent_holdout') is not True:
        raise ValueError('repair must remain an independent holdout without fitting')
    result = []
    for row in plan['points']:
        original, replacement = row['original'], row['replacement']
        if (original['batch'], original['context_tokens'], replacement['batch']) != (128, 256, 96):
            raise ValueError('repair may replace only the specified B128 point with B96')
        point = dict(freq_mhz=row['freq_mhz'], batch=96, context_tokens=256,
                     max_tokens=replacement['max_tokens'], repeats=row['repeats'],
                     settle_s=row['settle_s'], measure_s=row['measure_s'], purpose='independent_holdout_repair')
        if (type(point['max_tokens']) is not int or point['max_tokens'] < 25 or
                point['repeats'] != 3 or point['settle_s'] < 2 or point['measure_s'] < 5 or
                point['context_tokens']+point['max_tokens'] > 8192 or
                point_capacity_error(point, plan['kv_capacity_tokens'])):
            raise ValueError('invalid bounded repair reservation or window gates')
        if not model.decode_supported(96, 256+point['max_tokens']-1, point['freq_mhz']):
            raise ValueError('repair output reservation exceeds frozen candidate coverage')
        result.append(point)
    if len(result) != 4 or sorted(x['freq_mhz'] for x in result) != list(REPAIR_FREQUENCIES):
        raise ValueError('repair must contain exactly the four affected frequency points')
    return sorted(result, key=lambda x: x['freq_mhz'])


def compose_audit_view(original_raw, repair_raw, manifest, plan, model):
    """Build a view with explicit source roots; never modify either raw input."""
    points = repair_points(plan, model)
    expected = {(p['freq_mhz'], 96, 256) for p in points}
    actual = [(r['freq_mhz'], r['batch'], r['context_tokens']) for r in repair_raw.get('decode', [])]
    if len(actual) != 4 or set(actual) != expected:
        raise ValueError('repair is incomplete or contains unrequested measurements')
    remove = {(f, 128, 256) for f in REPAIR_FREQUENCIES}
    retained, superseded = [], []
    for row in original_raw['decode']:
        key = row['freq_mhz'], row['batch'], row['context_tokens']
        if key in remove:
            if all(model.decode_supported(row['batch'], r['effective_context_tokens'], row['freq_mhz'])
                   for r in row.get('repeats', [])):
                raise ValueError('refusing to replace an originally in-domain point')
            superseded.append(dict(reason='outside_coverage', error_class='measurement_domain_error',
                                   evidence_source='original', original_row=copy.deepcopy(row)))
        else:
            retained.append(dict(copy.deepcopy(row), evidence_source='original'))
    if len(superseded) != 4:
        raise ValueError('original archive does not contain exactly the four superseded domain errors')
    view = copy.deepcopy(original_raw)
    view['decode'] = retained + [dict(copy.deepcopy(r), evidence_source='repair') for r in repair_raw['decode']]
    for section in ('prefill', 'mixed'):
        view[section] = [dict(copy.deepcopy(r), evidence_source='original') for r in original_raw[section]]
    expected_plan = copy.deepcopy(manifest['plan'])
    for point in expected_plan['decode']:
        if (point['freq_mhz'], point['batch'], point['context_tokens']) in remove:
            point.update(batch=96, unseen_shape=True, supersedes_batch=128,
                         change_reason='bounded_sampling_domain_repair')
    return view, expected_plan, superseded


def run(*, candidate_dir, repair_plan, original_holdout, model_path, gpus, base_port, out):
    from .profiler import Profiler, _load_flock
    from .wave import ProfileWave
    from ..engine.launcher import Fleet
    from ..engine.client import EngineClient

    manifest = json.loads((candidate_dir/'manifest.json').read_text())
    candidate = candidate_dir/'candidate.json'
    plan = json.loads(repair_plan.read_text())
    if digest(candidate) != manifest['candidate_sha256'] or plan['candidate_sha256'] != manifest['candidate_sha256']:
        raise ValueError('repair candidate checksum differs from the original frozen fit')
    if plan['training_raw_sha256'] != manifest['training_raw_sha256']:
        raise ValueError('repair belongs to another training source')
    model = PerfModel.load(candidate)
    points = repair_points(plan, model)
    original_completion = json.loads((original_holdout/'completion.json').read_text())
    original_raw_path = original_holdout/'raw.json'
    if (original_completion.get('complete') is not True or
            original_completion.get('candidate_sha256') != manifest['candidate_sha256'] or
            digest(original_raw_path) != original_completion.get('raw_sha256')):
        raise ValueError('original holdout must be complete and bound to the same frozen candidate')
    original_raw = json.loads(original_raw_path.read_text())
    _checkpoint_points(original_raw, original_holdout)
    if (original_raw.get('system'), original_raw.get('model_id'), original_raw.get('tp'), original_raw.get('pp')) != (
            'pdblend', plan['model_id'], 1, 1):
        raise ValueError('original holdout identity differs from repair')
    if (original_raw.get('model_hash'), original_raw.get('tokenizer_hash')) != (manifest['model_hash'], manifest['tokenizer_hash']):
        raise ValueError('original holdout model/tokenizer differs from frozen candidate')
    binding = dict(candidate_sha256=manifest['candidate_sha256'], plan_sha256=digest(repair_plan),
                   original_raw_sha256=digest(original_raw_path), original_completion_sha256=digest(original_holdout/'completion.json'))
    profiler = Profiler(model_path, gpus, tp=1, pp=1, system='pdblend', out_dir=out,
                        hardware_id='8xL20-lease', base_port=base_port, kv_connector='P2pNcclConnector')
    if len(profiler.specs) != 1 or len(gpus) != 1:
        raise ValueError('sparse repair requires exactly one GPU and one resident engine')
    if (profiler.model_spec.model_hash, profiler.model_spec.tokenizer_hash) != (manifest['model_hash'], manifest['tokenizer_hash']):
        raise ValueError('live model/tokenizer differs from frozen fit')
    profiler.raw['config']['calibration_domain_repair'] = True
    profiler.raw['repair_binding'] = binding
    profiler.raw['measurement_plan_sha256'] = binding['plan_sha256']
    if (out/'raw.json').is_file():
        profiler.resume()
        if profiler.raw.get('repair_binding') != binding:
            raise ValueError('repair checkpoint belongs to another original archive or plan')
    completed = resume_points(profiler.raw, out, points)
    profiler.raw.setdefault('decode_pending', {})
    wave = ProfileWave.from_environment()
    if wave is None:
        raise ValueError('sparse repair requires coordinated profile qualification')
    result = dict(status='running', complete=False, calibration_passed=False, formal_eligible=False,
        energy_comparable=False, independent_holdout=True, fit_performed=False,
        measured_prefill_grid_points=0, measured_mixed_grid_points=0, requested_decode_points=4,
        model_id=plan['model_id'], system='pdblend', tp=1, pp=1, started_s=time.time(), **binding)
    atomic_json(out/'repair-plan.json', plan)
    try:
        with Fleet(profiler.specs, out/'logs') as fleet:
            with _load_flock():
                fleet.start_all()
            instance = fleet[profiler.specs[0].instance_id]
            profiler.raw['kv_capacity_tokens'] = profiler._kv_capacity(instance)
            if profiler.raw['kv_capacity_tokens'] <= 0:
                raise RuntimeError('engine did not report measured KV capacity')

            async def sample():
                await wave.qualify_external(profiler, fleet)
                async with wave.measurement():
                    async with EngineClient(instance.spec.instance_id, instance.spec.base_url) as client:
                        for point in points:
                            key = point_key(point)
                            if key in completed:
                                continue
                            reason = point_capacity_error(point, profiler.raw['kv_capacity_tokens'])
                            if reason:
                                raise RuntimeError('missing_profile: '+reason)
                            profiler._lock(point['freq_mhz'], instance.spec.gpus)
                            await asyncio.sleep(2)

                            def checkpoint(repeats):
                                profiler.raw['decode_pending'][key] = repeats
                                profiler._checkpoint()

                            row = await collect_bounded_decode_point(profiler, client, instance.spec.gpus, point,
                                purpose='independent_holdout_repair', previous=profiler.raw['decode_pending'].get(key, ()),
                                on_window=checkpoint)
                            row.update(requires_per_window_evaluation=True, candidate_sha256=manifest['candidate_sha256'])
                            profiler.raw['decode'].append(row)
                            profiler.raw['decode_pending'].pop(key, None)
                            completed.add(key)
                            profiler._checkpoint()
                            print(f'repair f={point["freq_mhz"]} B96 context256 '
                                  f'observed={row["effective_context_tokens"]:.1f} step={row["step_seconds"]:.6f}s', flush=True)
            asyncio.run(sample())
        if (digest(candidate) != binding['candidate_sha256'] or digest(repair_plan) != binding['plan_sha256'] or
                digest(original_raw_path) != binding['original_raw_sha256'] or
                digest(original_holdout/'completion.json') != binding['original_completion_sha256']):
            raise RuntimeError('immutable candidate, plan or original holdout changed during repair')
        resume_points(profiler.raw, out, points)
        view, expected_plan, superseded = compose_audit_view(original_raw, profiler.raw, manifest, plan, model)
        audit = evaluate_holdout(view, model, out, expected_plan=expected_plan,
                                 evidence_roots={'original': original_holdout, 'repair': out})
        atomic_json(out/'superseded-domain-evidence.json', dict(binding=binding, rows=superseded,
                                                               formal_eligible=False))
        atomic_json(out/'composite-audit-input.json', dict(binding=binding,
            evidence_roots={'original': str(original_holdout), 'repair': str(out)},
            effective_plan=expected_plan, view=view, formal_eligible=False))
        atomic_json(out/'holdout-audit.json', audit)
        result.update(status='passed', complete=True, calibration_passed=audit['passed'],
            calibration_status='passed' if audit['passed'] else 'failed', failures=audit['failures'],
            timing_max=audit['timing_max'], superseded_points=len(superseded),
            measured_decode_points=len(profiler.raw['decode']),
            effective_matrix_changed=True, original_matrix_preserved=True)
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        wave.write('error', dict(error=result['error']))
    finally:
        profiler._checkpoint()
        profiler.meter.reset_all()
        result.update(finished_s=time.time(), raw_sha256=digest(out/'raw.json'),
            missing_gates=['native_system_mechanisms', 'full_profile_provenance_revalidation', 'campaign_acceptance'])
        atomic_json(out/'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--repair-plan', type=Path, required=True)
    parser.add_argument('--original-holdout', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--base-port', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(candidate_dir=args.candidate_dir, repair_plan=args.repair_plan,
        original_holdout=args.original_holdout, model_path=args.model, gpus=args.gpus,
        base_port=args.base_port, out=args.out)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
