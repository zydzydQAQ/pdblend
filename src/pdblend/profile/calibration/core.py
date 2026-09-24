"""Incremental, frozen-model calibration against fresh GPU holdout windows.

Training artifacts are immutable inputs. This runner never fits on holdout
data, never relabels old measurements with new checksums, and does not grant
system correctness or formal benchmark eligibility.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from contextlib import AsyncExitStack
from pathlib import Path

import numpy as np

from pdblend.profile.calibration.decode_fit import fit_split_b1
from pdblend.profile.query.model import PerfModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_candidate(raw_path: Path, profile_path: Path, destination: Path) -> dict:
    raw = json.loads(raw_path.read_text())
    model = PerfModel.load(profile_path)
    if raw.get('system') != 'pdblend' or model.system != 'pdblend':
        raise ValueError('PDBlend fitting cannot be used as another system profile')
    if (raw['model_id'], raw['tp'], raw['pp']) != (Path(model.model).name, model.tp, model.pp):
        raise ValueError('training profile identity mismatch')
    before, after = [], []
    for f in model.freqs:
        rows = [r for r in raw['decode'] if r['freq_mhz'] == f]
        before.extend(abs(model.step_seconds(r['batch'], r['effective_context_tokens'], f)/r['step_seconds']-1) for r in rows)
        model.decode_overrides[f] = fit_split_b1(rows, raw['kv_capacity_tokens'])
        ps = [r for r in raw['prefill'] if r['freq_mhz'] == f]
        n = np.asarray([r['input_tokens'] for r in ps], float)
        y = np.asarray([r['seconds'] for r in ps], float)
        x = np.stack([np.ones_like(n), n / 1024, (n / 1024)**2], 1)
        a = np.linalg.lstsq(x / y[:, None], np.ones(len(y)), rcond=None)[0]
        model.prefill_time[f] = (float(a[0]), float(a[1]/1024), float(a[2]/1024**2))
        after.extend(abs(model.step_seconds(r['batch'], r['effective_context_tokens'], f)/r['step_seconds']-1) for r in rows)
    model.bounded_coverage = dict(prefill_tokens=[min(r['input_tokens'] for r in raw['prefill']),
                                                max(r['input_tokens'] for r in raw['prefill'])],
                                  frequency_interpolation=False, extrapolation=False)
    model.quality['calibration'] = dict(status='training_only', independent_holdout=False,
                                      training_raw_sha256=digest(raw_path), training_profile_sha256=digest(profile_path))
    destination.mkdir(parents=True, exist_ok=True)
    model.save(destination / 'candidate.json')
    plan = holdout_plan(raw)
    manifest = dict(schema=1, model_id=raw['model_id'], tp=raw['tp'], pp=raw['pp'],
                    system='pdblend', model_hash=raw['model_hash'], tokenizer_hash=raw['tokenizer_hash'],
                    training_raw=str(raw_path.resolve()), training_raw_sha256=digest(raw_path),
                    training_profile=str(profile_path.resolve()), training_profile_sha256=digest(profile_path),
                    candidate_sha256=digest(destination / 'candidate.json'),
                    decode_training_max_before=max(before), decode_training_max_after=max(after),
                    plan=plan, prepared_at_s=time.time(), formal_eligible=False,
                    training_environment=raw['environment'])
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def holdout_plan(raw: dict) -> dict:
    """48 legal decode shapes at most; two never-trained interpolation probes.

    Real capacity and measured shape coverage constrain the high-batch probes.
    Three short measurement windows remain mandatory at every selected point.
    An opt-in shared-prefill path labels their shared decode run explicitly.
    """
    decode = []
    for f in sorted(raw['freqs']):
        measured = {(r['batch'], r['context_tokens']) for r in raw['decode'] if r['freq_mhz'] == f}
        def maximum(ctx):
            return max(b for b, c in measured if c == ctx)
        shapes = [(1, 256), (1, 4096), (8, 1024), (32, 1024),
                  (maximum(256), 256), (maximum(4096), 4096), (2, 2048), (12, 2048)]
        for b, c in dict.fromkeys(shapes):
            if b*(c+64) <= raw['kv_capacity_tokens']*.9:
                decode.append(dict(freq_mhz=f, batch=b, context_tokens=c,
                                   unseen_shape=(b, c) not in measured))
    return dict(decode=decode,
                prefill=[dict(freq_mhz=f, input_tokens=n) for f in sorted(raw['freqs'])
                         for n in (128, 512, 2048, 3072, 7168)],
                mixed_freqs=[1500, 2100, 2520], repeats=3, settle_s=2, measure_s=5,
                minimum_decode_window_s=len(decode)*3*7)


def _evidence_file(root: Path, row: dict):
    name = row.get('samples_file')
    if not isinstance(name, str) or not name:
        return None
    path = (root / name).resolve()
    if (not path.is_relative_to(root.resolve()) or not path.is_file() or
            digest(path) != row.get('samples_sha256')):
        return None
    return path


def _checkpoint_points(raw: dict, root: Path) -> dict:
    """Only complete, checksum-bound rows may suppress resumed GPU work."""
    keys = dict(prefill=('freq_mhz', 'input_tokens'), decode=('freq_mhz', 'batch', 'context_tokens'))
    result = {}
    for section, fields in keys.items():
        found = set()
        for row in raw.get(section, []):
            key = tuple(row[k] for k in fields)
            if key in found:
                raise ValueError(f'duplicate holdout checkpoint point: {section}/{key}')
            records = [row] if section == 'prefill' else row.get('repeats', [])
            if (not records or (section == 'decode' and len(records) < 3) or
                    any(_evidence_file(root, record) is None for record in records)):
                raise ValueError(f'incomplete or corrupt holdout checkpoint: {section}/{key}')
            if section == 'decode' and any(r.get('steady_window_s', 0) < 5 or
                    r.get('min_steps', 0) < 8 or r.get('power_samples', 0) < 2 or
                    r.get('frequency_samples', 0) < 1 for r in records):
                raise ValueError(f'incomplete decode windows in checkpoint: {key}')
            found.add(key)
        result[section] = found
    return result


def evaluate_holdout(raw: dict, model: PerfModel, root: Path, *, expected_plan=None, evidence_roots=None) -> dict:
    rows, failures, power_points = [], [], []
    def source_root(row):
        source = row.get('evidence_source')
        if source is None:
            return root
        if source not in (evidence_roots or {}):
            raise ValueError(f'undeclared holdout evidence source: {source}')
        return Path(evidence_roots[source])
    for section, metric in (('prefill', 'prefill_time'), ('decode', 'decode_time')):
        for row in raw.get(section, []):
            f = row['freq_mhz']
            observed = row['seconds'] if section == 'prefill' else row['step_seconds']
            try:
                predicted = (model.prefill_seconds(row['input_tokens'], f) if section == 'prefill' else
                             model.step_seconds(row['batch'], row['effective_context_tokens'], f))
                error = abs(predicted/observed - 1)
            except (ValueError, ZeroDivisionError):
                predicted, error = None, None
            value = dict(metric=metric, freq_mhz=f, observed=observed, predicted=predicted,
                         relative_error=error, shape={k: row[k] for k in ('batch', 'context_tokens', 'input_tokens') if k in row})
            if row.get('evidence_source') is not None:value['evidence_source']=row['evidence_source']
            if section == 'decode' and not model.decode_supported(row['batch'], row['effective_context_tokens'], f):
                value.update(status='outside_coverage', error_class='measurement_domain_error',
                             effective_context_tokens=row['effective_context_tokens'])
            rows.append(value)
            if error is None or not math.isfinite(error) or error > .10:
                failures.append(value)
            if section == 'decode':
                if len(row.get('repeats', [])) < 3:
                    failures.append(dict(metric='decode_repeats', shape=value['shape']))
                for index, rep in enumerate(row.get('repeats', [])):
                    path = _evidence_file(source_root(row), rep)
                    if (path is None or
                            rep.get('steady_window_s', 0) < 5 or rep.get('min_steps', 0) < 8 or
                            rep.get('power_samples', 0) < 2 or rep.get('frequency_samples', 0) < 1):
                        failures.append(dict(metric='decode_raw_evidence', shape=value['shape']))
                    if row.get('requires_per_window_evaluation'):
                        try:
                            from pdblend.profile.collection.window_sampling import summarize_window
                            evidence = json.loads(path.read_text()) if path else {}
                            measured = summarize_window(token_times=evidence['token_times_s'],
                                context=row['context_tokens'], start_s=evidence['start_s'], end_s=evidence['end_s'],
                                power=evidence['power'], frequency=evidence['frequency'],
                                gpu_count=len(row['measured_gpu_ids']),
                                settle_s=evidence['start_s']-evidence['settle_start_s'])
                            for key in ('effective_context_tokens', 'observed_context_min', 'observed_context_max',
                                        'step_seconds', 'power_w', 'min_steps', 'power_samples', 'frequency_samples'):
                                if not math.isclose(measured[key], rep[key], rel_tol=1e-9, abs_tol=1e-9):
                                    raise ValueError(f'raw window differs from summary: {key}')
                            if not all(model.decode_supported(row['batch'], measured[k], f)
                                       for k in ('observed_context_min', 'observed_context_max')):
                                raise ValueError('window exceeds frozen context coverage')
                        except (ValueError, KeyError, TypeError, OSError) as exc:
                            failures.append(dict(metric='decode_window_reconstruction', shape=value['shape'],
                                                 repeat=index, error=str(exc)))
                    # Recompute from the frozen candidate, never trust the
                    # helper's stored prediction or a passing aggregate median.
                    try:
                        context = rep['effective_context_tokens']
                        prediction = model.step_seconds(row['batch'], context, f)
                        relative = abs(prediction / rep['step_seconds'] - 1)
                    except (ValueError, KeyError, ZeroDivisionError):
                        context, prediction, relative = rep.get('effective_context_tokens'), None, None
                    repeat_value = dict(metric='decode_time_repeat', freq_mhz=f, repeat=index,
                        shape=value['shape'], effective_context_tokens=context,
                        observed=rep.get('step_seconds'), predicted=prediction, relative_error=relative)
                    if row.get('evidence_source') is not None:repeat_value['evidence_source']=row['evidence_source']
                    if context is not None and not model.decode_supported(row['batch'], context, f):
                        repeat_value.update(status='outside_coverage', error_class='measurement_domain_error')
                    rows.append(repeat_value)
                    if relative is None or not math.isfinite(relative) or relative > .10:
                        failures.append(repeat_value)
                if row.get('prediction_failures'):
                    failures.append(dict(metric='shared_window_prediction_failure', shape=value['shape'],
                                         details=row['prediction_failures']))
            elif _evidence_file(source_root(row), row) is None:
                failures.append(dict(metric='prefill_raw_evidence', shape=value['shape']))
    if expected_plan is not None:
        for section, fields in (('prefill', ('freq_mhz', 'input_tokens')),
                                ('decode', ('freq_mhz', 'batch', 'context_tokens'))):
            wanted = {tuple(x[k] for k in fields) for x in expected_plan[section]}
            measured = [tuple(x[k] for k in fields) for x in raw.get(section, [])]
            if set(measured) != wanted or len(measured) != len(set(measured)):
                failures.append(dict(metric=section+'_matrix_coverage', missing=sorted(wanted-set(measured)),
                                     unexpected=sorted(set(measured)-wanted), duplicates=len(measured)-len(set(measured))))
    for f in model.freqs:
        errors = []
        for r in raw.get('decode', []):
            if r['freq_mhz'] != f or r['batch'] < 1:
                continue
            for index, rep in enumerate(r.get('repeats', [])):
                from pdblend.profile.query.power_table import PowerCoverageError
                point = dict(metric='decode_power_repeat', freq_mhz=f, repeat=index,
                             batch=r['batch'], context_tokens=rep.get('effective_context_tokens'),
                             observed=rep.get('power_w'), predicted=None, relative_error=None)
                if r.get('evidence_source') is not None:point['evidence_source']=r['evidence_source']
                try:
                    prediction = model.decode_power_w(r['batch'], f, ctx=rep.get('effective_context_tokens'))
                    if rep.get('power_w', 0) <= 0:
                        raise ValueError('invalid observed power')
                    error = abs(prediction/rep['power_w']-1)
                    point.update(predicted=prediction, relative_error=error)
                    errors.append(error)
                    if getattr(model, 'decode_power_overrides', {}) and (not math.isfinite(error) or error > .10):
                        failures.append(dict(point, limit=.10))
                except PowerCoverageError as exc:
                    point.update(status='outside_coverage', error_class='measurement_domain_error', error=str(exc))
                    failures.append(point)
                    errors.append(float('inf'))
                except (ValueError, TypeError, KeyError) as exc:
                    point.update(error_class='invalid_power_evidence', error=str(exc))
                    failures.append(point)
                    errors.append(float('inf'))
                power_points.append(point)
        if not errors or not all(math.isfinite(x) for x in errors) or statistics.fmean(errors) > .10 or max(errors) > .15:
            finite = errors and all(math.isfinite(x) for x in errors)
            failures.append(dict(metric='decode_power', freq_mhz=f, mape=statistics.fmean(errors) if finite else None,
                                 maximum=max(errors) if finite else None))
    mixed = []
    for r in raw['mixed']:
        file = source_root(r) / r.get('samples_file', '')
        if not r.get('valid') or not file.is_file() or digest(file) != r.get('samples_sha256'):
            failures.append(dict(metric='mixed_evidence', point=r))
        else:
            mixed.append(abs((r['base_step_s']+r['alone_prefill_s'])/r['probe_ttft_s']-1))
    if len(mixed) != 12 or statistics.median(mixed) > .15:
        failures.append(dict(metric='mixed_additive', samples=len(mixed), median=statistics.median(mixed) if mixed else None))
    return dict(passed=not failures, failures=failures, points=rows,
                power_points=power_points,
                invalid_sampling_domain=[r for r in failures if r.get('status') == 'outside_coverage'],
                timing_max=max((r['relative_error'] for r in rows if r['relative_error'] is not None), default=None),
                mixed_median=statistics.median(mixed) if mixed else None)


def run_holdout(*, candidate_dir: Path, model_path: str, gpus: list[int], base_port: int, out: Path,
                shared_prefill_windows: bool = False, training_raw_path: Path | None = None,
                prior_holdout: Path | None = None, prior_raw_sha256: str | None = None):
    from pdblend.profile.collection.profiler import Profiler, _load_flock
    from pdblend.profile.collection.wave import ProfileWave, atomic_json
    from pdblend.engine.launcher import Fleet
    from pdblend.engine.client import EngineClient
    from pdblend.bench.gates import random_prompt

    manifest = json.loads((candidate_dir/'manifest.json').read_text())
    candidate = candidate_dir/'candidate.json'
    if digest(candidate) != manifest['candidate_sha256']:
        raise ValueError('frozen candidate checksum mismatch')
    model = PerfModel.load(candidate)
    training_raw = None
    if shared_prefill_windows:
        training_input = training_raw_path or Path(manifest['training_raw'])
        if digest(training_input) != manifest['training_raw_sha256']:
            raise ValueError('frozen training raw checksum mismatch')
        training_raw = json.loads(training_input.read_text())
    profiler = Profiler(model_path, gpus, tp=model.tp, out_dir=out, base_port=base_port,
                        hardware_id='8xL20-lease', kv_connector='P2pNcclConnector')
    if len(profiler.specs) != 1:
        raise ValueError('one resident engine per holdout job is required')
    if (profiler.model_spec.model_hash, profiler.model_spec.tokenizer_hash) != (manifest['model_hash'], manifest['tokenizer_hash']):
        raise ValueError('holdout model/tokenizer differs from training')
    profiler.raw['config']['shared_prefill_windows'] = bool(shared_prefill_windows)
    profiler.raw['holdout_candidate_sha256'] = manifest['candidate_sha256']
    if (out/'raw.json').is_file():
        profiler.resume()
        if profiler.raw.get('holdout_candidate_sha256') != manifest['candidate_sha256']:
            raise ValueError('holdout checkpoint belongs to another frozen candidate')
    completed = _checkpoint_points(profiler.raw, out)
    prior_raw, prior_receipt = None, None
    if bool(prior_holdout) != bool(prior_raw_sha256):
        raise ValueError('prior holdout directory and frozen raw SHA256 are both required')
    if prior_holdout is not None:
        from pdblend.profile.calibration.holdout_inheritance import load_prior_holdout, merge_holdout_rows
        if Path(prior_holdout).resolve() == out.resolve():
            raise ValueError('prior holdout must be a separate read-only evidence directory')
        prior_raw, inherited, prior_receipt = load_prior_holdout(prior_holdout, prior_raw_sha256, manifest, profiler.raw)
        previous_binding = profiler.raw.get('prior_holdout_binding')
        if previous_binding is not None and previous_binding != prior_receipt['binding']:
            raise ValueError('new checkpoint references a different prior holdout')
        merge_holdout_rows(profiler.raw, prior_raw, prior_receipt, expected_plan=manifest['plan'])
        profiler.raw['prior_holdout_binding'] = prior_receipt['binding']
        completed = {section:completed[section] | inherited[section] for section in completed}
        atomic_json(out/'prior-holdout-manifest.json', prior_receipt)
    elif profiler.raw.get('prior_holdout_binding'):
        raise ValueError('resuming inherited evidence requires its declared prior directory and checksum')
    started = time.time()
    atomic_json(out/'frozen-fit.json', manifest)
    result = dict(status='running', complete=False, formal_eligible=False, energy_comparable=False,
                  system='pdblend', model_id=manifest['model_id'], tp=model.tp,
                  shared_prefill_windows=bool(shared_prefill_windows),
                  independent_holdout=True, candidate_sha256=manifest['candidate_sha256'], started_s=started)
    if prior_receipt is not None:
        result['prior_holdout'] = prior_receipt
    try:
        with Fleet(profiler.specs, out/'logs') as fleet:
            with _load_flock():
                fleet.start_all()
            inst = fleet[profiler.specs[0].instance_id]
            profiler.raw['kv_capacity_tokens'] = profiler._kv_capacity(inst)
            async def sample():
                wave = ProfileWave.from_environment()
                if wave is None:
                    raise ValueError('holdout requires coordinated parallel qualification')
                await wave.qualify_external(profiler, fleet)
                async with wave.measurement():
                    async with EngineClient(inst.spec.instance_id, inst.spec.base_url) as client:
                        for f in model.freqs:
                            profiler._lock(f, inst.spec.gpus)
                            await asyncio.sleep(2)
                            for point in manifest['plan']['prefill']:
                                if point['freq_mhz'] != f or (f, point['input_tokens']) in completed['prefill']:
                                    continue
                                n = point['input_tokens']; prompt = random_prompt(n, 9701+n)
                                warm = await client.complete(prompt, 1, f'holdout-p-warm-{f}-{n}', seed=9701)
                                if warm.error:
                                    raise RuntimeError(warm.error)
                                times, outcomes = [], []
                                with profiler.meter.measure(inst.spec.gpus) as measurement:
                                    deadline = time.time()+2
                                    while len(times) < 3 or time.time() < deadline:
                                        r = await client.complete(prompt, 1, f'holdout-p-{f}-{n}-{len(times)}', seed=9701)
                                        if r.error or r.ttft_s is None:
                                            raise RuntimeError(r.error or 'prefill missing token')
                                        times.append(r.ttft_s)
                                        outcomes.append(dict(submitted_s=r.submitted_s, first_token_s=r.first_token_s, finished_s=r.finished_s))
                                file = out/'samples'/f'holdout-p-{f}-{n}.json'
                                atomic_json(file, dict(outcomes=outcomes, measurement=measurement))
                                profiler.raw['prefill'].append(dict(freq_mhz=f, input_tokens=n, seconds=statistics.median(times),
                                    power_w=measurement['mean_power_w'], runs=len(times), samples_file=str(file.relative_to(out)), samples_sha256=digest(file)))
                                completed['prefill'].add((f, n))
                                profiler._checkpoint()
                            for point in manifest['plan']['decode']:
                                if point['freq_mhz'] != f:
                                    continue
                                b, c = point['batch'], point['context_tokens']
                                if (f, b, c) in completed['decode']:
                                    continue
                                tag = f'holdout-d-{f}-{c}-{b}'
                                shared = None
                                if shared_prefill_windows:
                                    from pdblend.profile.collection.window_sampling import measure_shared_prefill_windows
                                    shared = await measure_shared_prefill_windows(profiler, client, inst.spec.gpus,
                                        model=model, freq_mhz=f, batch=b, context=c, tag=tag, training_raw=training_raw)
                                if shared is None or shared['status'] == 'fallback_required':
                                    row = await profiler._decode_batch(client, inst.spec.gpus, b, c, 64, tag)
                                    if shared is not None:
                                        row['shared_prefill_attempt'] = shared
                                        row['sampling_method'] = 'legacy_separate_prefill_fallback'
                                else:
                                    row = shared['row']
                                    if row is None:
                                        raise RuntimeError(f'shared-prefill prediction failed before a complete point: {shared}')
                                    row['shared_prefill_attempt'] = {k: v for k, v in shared.items() if k != 'row'}
                                row.update(freq_mhz=f, unseen_shape=point['unseen_shape'])
                                profiler.raw['decode'].append(row)
                                completed['decode'].add((f, b, c))
                                profiler._checkpoint()
                                print(f'holdout TP{model.tp} f={f} B={b} context={c}: {row["step_seconds"]:.6f}s', flush=True)
                        if prior_receipt is None:
                            await profiler._mixed(client, inst.spec.gpus)
                        else:
                            mixed_reference, _ = merge_holdout_rows(profiler.raw, prior_raw, prior_receipt,
                                expected_plan=manifest['plan'], require_complete=True)
                            await profiler._mixed(client, inst.spec.gpus, reference_raw=mixed_reference)
                        profiler._checkpoint()
            asyncio.run(sample())
        if digest(candidate) != manifest['candidate_sha256']:
            raise RuntimeError('fit changed during holdout')
        audit_raw, evidence_roots = profiler.raw, None
        if prior_receipt is not None:
            # Revalidate all old bytes after GPU work; inherited rows are only
            # present in this explicitly derived view, never in new raw.json.
            checked_raw, _, checked = load_prior_holdout(prior_holdout, prior_raw_sha256, manifest, profiler.raw)
            if checked['binding'] != prior_receipt['binding']:
                raise RuntimeError('prior holdout binding changed during new measurement')
            audit_raw, evidence_roots = merge_holdout_rows(profiler.raw, checked_raw, prior_receipt,
                expected_plan=manifest['plan'], require_complete=True)
            atomic_json(out/'combined-holdout.json', audit_raw)
            result.update(combined_holdout_sha256=digest(out/'combined-holdout.json'),
                evidence_roots={key:str(path) for key,path in evidence_roots.items()},
                newly_measured_points={key:len(profiler.raw.get(key, [])) for key in ('prefill','decode')},
                inherited_points={key:len(prior_raw.get(key, [])) for key in ('prefill','decode')})
        audit = evaluate_holdout(audit_raw, model, out, expected_plan=manifest['plan'], evidence_roots=evidence_roots)
        atomic_json(out/'holdout-audit.json', audit)
        result.update(status='passed', complete=True,
                      calibration_status='passed' if audit['passed'] else 'failed',
                      calibration_passed=audit['passed'], failures=audit['failures'], timing_max=audit['timing_max'])
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        wave = ProfileWave.from_environment()
        if wave is not None:
            wave.write('error', dict(error=result['error']))
    finally:
        profiler._checkpoint()
        profiler.meter.reset_all()
        result.update(finished_s=time.time(), raw_sha256=digest(out/'raw.json'),
                      missing_gates=['native_system_mechanisms', 'full_profile_provenance_revalidation', 'campaign_acceptance'])
        # raw.json remains a measurement archive; independent status is bound
        # here to the pre-existing frozen model, not inferred from row splitting.
        atomic_json(out/'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--base-port', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shared-prefill-windows', action='store_true',
                        help='opt in to bounded consecutive decode windows; incompatible points use the existing sampler')
    parser.add_argument('--training-raw', type=Path,
                        help='read-only training raw path inside the container; original manifest checksum is required')
    parser.add_argument('--prior-holdout', type=Path, help='read-only prior partial holdout directory; source provenance stays separate')
    parser.add_argument('--prior-raw-sha256', help='SHA256 of the stopped prior raw checkpoint, required with --prior-holdout')
    args = parser.parse_args()
    result = run_holdout(candidate_dir=args.candidate_dir, model_path=args.model, gpus=args.gpus,
                         base_port=args.base_port, out=args.out, shared_prefill_windows=args.shared_prefill_windows,
                         training_raw_path=args.training_raw, prior_holdout=args.prior_holdout,
                         prior_raw_sha256=args.prior_raw_sha256)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
