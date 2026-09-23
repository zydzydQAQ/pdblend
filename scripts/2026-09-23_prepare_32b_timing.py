#!/usr/bin/env python3
"""CPU-only original-training B32 knee repair and resident holdout package."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import statistics
from pathlib import Path

import numpy as np

from pdblend.profile import timing_calibration as tc
from pdblend.profile.decode_fit import fit_split_b1, predict
from pdblend.profile.model import PerfModel
from pdblend.profile.power_calibration import digest, scheduling_proxy, write_immutable

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/2026-09-22/three-model/calibration-candidates/32b-tp4-5d5ae4a37aed7c851cf7'


def errors(values):
    supported = [r['relative_error'] for r in values if r['relative_error'] is not None]
    return dict(total=len(values), supported=len(supported), unsupported=len(values)-len(supported),
        mape=statistics.fmean(supported) if supported else None,
        maximum=max(supported) if supported else None, rows=values)


def cross_validation(rows, capacity, degree):
    """Refit the base too; no held shape leaks through frozen base coefficients."""
    values = []
    for f in tc.FREQUENCIES:
        group = [r for r in rows if r['freq_mhz'] == f]
        for held in (256, 1024, 4096):
            training = [r for r in group if not (r['batch'] == 32 and r['context_tokens'] == held)]
            test = [r for r in group if r['batch'] == 32 and r['context_tokens'] == held]
            local = [r for r in training if r['batch'] == 32]
            # Boundary folds are not interpolation evidence. Keep their counts.
            inside = min(r['context_tokens'] for r in local) < held < max(r['context_tokens'] for r in local)
            base = fit_split_b1(training, capacity)
            a = tc.fit_residual(training, base, degree=degree)
            for row in test:
                for i, rep in enumerate(row['repeats']):
                    c = rep['effective_context_tokens']
                    predicted = predict(base, 32, c)+a[0]+(a[1]*c/4096 if degree else 0) if inside else None
                    values.append(dict(freq_mhz=f, nominal_context_tokens=held, repeat=i,
                        effective_context_tokens=c, observed=rep['step_seconds'], predicted=predicted,
                        relative_error=abs(predicted/rep['step_seconds']-1) if predicted is not None else None,
                        status='supported_interpolation' if inside else 'unsupported_boundary_extrapolation'))
    return errors(values)


def training_comparison(raw, base, candidate):
    model = tc.TimingOverlay(base, candidate)
    before, after, local = [], [], []
    for row in raw['decode']:
        for i, rep in enumerate(row['repeats']):
            args = row['batch'], rep['effective_context_tokens'], row['freq_mhz']
            common = dict(batch=row['batch'], freq_mhz=row['freq_mhz'], context_tokens=row['context_tokens'],
                effective_context_tokens=rep['effective_context_tokens'], repeat=i, observed=rep['step_seconds'])
            before.append(dict(common, predicted=base.step_seconds(*args),
                relative_error=abs(base.step_seconds(*args)/rep['step_seconds']-1)))
            item = dict(common, predicted=model.step_seconds(*args),
                relative_error=abs(model.step_seconds(*args)/rep['step_seconds']-1))
            after.append(item)
            if row['batch'] == 32: local.append(item)
    return dict(original_training=errors(before), overlay_training=errors(after), B32_training=errors(local),
        constant_residual_shape_cv=cross_validation(raw['decode'], raw['kv_capacity_tokens'], 0),
        context_affine_residual_shape_cv=cross_validation(raw['decode'], raw['kv_capacity_tokens'], 1),
        batch_cv=dict(supported=0, unsupported=54, reason='Removing B32 leaves no measured kernel-knee anchor; never claim unseen B24/B48 validation.'),
        split_unit='whole frequency/batch/nominal-context group; all three repeats stay together',
        base_refitted_inside_each_cv_fold=True, holdout_used_for_fit_or_selection=False,
        selection='Constant residual is the lowest-dimensional correction; context-affine comparison is diagnostic only. Both require fresh independent validation.')


def reserve_points(raw, base, root):
    points = []
    for f in tc.FREQUENCIES:
        for batch in (24, 32, 48):
            brackets = (32,) if batch == 32 else ((16, 32) if batch == 24 else (32, 64))
            anchors = [r for r in raw['decode'] if r['freq_mhz'] == f and r['batch'] in brackets and r['context_tokens'] == 1024]
            fastest = min(p['step_seconds'] for r in anchors for p in r['repeats'])
            lead = max(max(json.loads((root/p['samples_file']).read_text())['start_token_counts'])
                for r in anchors for p in r['repeats'])
            tail = math.ceil(21/(fastest*.85))+lead+32
            p = dict(freq_mhz=f, batch=batch, context_tokens=1024, max_tokens=tail, repeats=3,
                settle_s=2., measure_s=5., purpose=tc.PURPOSE,
                reason='fresh_replacement_of_original_B32_gate' if batch == 32 else 'unseen_neighbor_batch_interpolation',
                reservation=dict(fastest_training_step_s=fastest, measured_start_count_lead=lead,
                    generation_speed_margin=.15, terminal_guard_tokens=32,
                    training_anchor_batches=list(brackets), full_reserved_context=[1024, 1024+tail-1]),
                sampling_method='three_distinct_windows_shared_prefill', automatic_prediction_failure_retry=False)
            if any(not base.decode_supported(batch, c, f) for c in (1024, 1024+tail-1)):
                raise ValueError('timing reservation escaped unchanged frozen coverage')
            if 1024+tail > 8192 or batch*(1024+tail) > raw['kv_capacity_tokens']*.9:
                raise ValueError('timing reservation escaped memory/model length')
            points.append(p)
    return points


def prepare(out):
    previous = json.loads((BASE/'manifest.json').read_text())
    raw_path = Path(previous['training_raw']); raw = json.loads(raw_path.read_text())
    if digest(raw_path) != previous['training_raw_sha256'] or digest(BASE/'candidate.json') != previous['candidate_sha256']:
        raise ValueError('original training candidate manifest mismatch')
    spec = importlib.util.spec_from_file_location('check_32b_original_training', ROOT/'scripts/2026-09-23_prepare_32b_power.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    checks = helper.check_training(raw, raw_path, 4)
    # Reconstruct timing itself from the raw count window, in addition to the
    # checksum, power, context, repeats and window checks above.
    for row in raw['decode']:
        for rep in row['repeats']:
            sample = json.loads((raw_path.parent/rep['samples_file']).read_text())
            observed = (sample['end_s']-sample['start_s'])/statistics.median(
                b-a for a, b in zip(sample['start_token_counts'], sample['end_token_counts']))
            if not math.isclose(observed, rep['step_seconds'], rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError('training timing differs from original raw token counts')
    base = PerfModel.load(BASE/'candidate.json')
    candidate = dict(schema=1, kind=tc.KIND, system='pdblend', model_id=raw['model_id'], tp=4, pp=1,
        model_hash=raw['model_hash'], tokenizer_hash=raw['tokenizer_hash'], batch_knots=[16,32,64],
        context_degree=0, residual_seconds={str(f):tc.fit_residual(
            [r for r in raw['decode'] if r['freq_mhz'] == f], base.decode_overrides[f])[0] for f in tc.FREQUENCIES},
        domains={str(f):copy.deepcopy(base.decode_overrides[f]['domain']) for f in tc.FREQUENCIES},
        base_candidate_sha256=digest(BASE/'candidate.json'), training_raw_sha256=digest(raw_path),
        fitting_objective='squared_relative_error_of_original_B32_training_repeats',
        training_only=True, independent_holdout=False, holdout_used_for_fit_or_selection=False, formal_eligible=False,
        unchanged_region='B <= 16 or B >= 64: exact original timing prediction; all original coverage bounds identical',
        changed_region='16 < B < 64 at all original supported contexts; context-independent residual',
        fresh_validation_required='B24/B32/B48 x six frequencies at original 1024 prompt; each of 54 windows <=10% error')
    model = tc.TimingOverlay(base, candidate)
    # Check a dense original-domain grid for positivity and batch monotonicity;
    # adding a descending triangular residual must not create a local reversal.
    monotone_checks = 0
    for f in tc.FREQUENCIES:
        for c in np.linspace(*base.decode_overrides[f]['domain']['context'], 33):
            values = [model.step_seconds(b, c, f) for b in np.linspace(16,64,193) if model.decode_supported(b,c,f)]
            if any(not math.isfinite(v) or v <= 0 for v in values) or any(b < a-1e-12 for a,b in zip(values,values[1:])):
                raise ValueError('timing overlay is not positive and batch-monotone')
            monotone_checks += len(values)
    report = training_comparison(raw, base, candidate)
    report.update(training_windows_checked=len(checks), raw_checksums=checks, monotone_grid_checks=monotone_checks,
        formal_eligible=False, training_raw=str(raw_path), training_raw_sha256=digest(raw_path),
        no_new_context_coefficient=True,
        context_cv_limit='Only the middle nominal context is an interpolation fold. Lower/upper boundary folds are unsupported, not zero-error validation.',
        original_parallel_qualification=raw.get('parallel_interference'),
        interference_conclusion='Original 2100MHz/B8 check does not establish 900/1200/1500MHz B32 interference. Stable original-training B32 residual proves model underfit; separate holdout drift remains unattributed.')
    write_immutable(out/'candidate.json', candidate)
    write_immutable(out/'training-comparison.json', report)
    points = reserve_points(raw, base, raw_path.parent)
    plan = dict(schema=1, purpose=tc.PURPOSE, system='pdblend', model_id=raw['model_id'], tp=4, pp=1,
        points=points, point_count=18, repeats=3, minimum_decode_window_seconds=378,
        timing_gate=dict(each_window_max_relative_error=.10, repeat_cv_max=.10),
        original_gate=dict(decode_points=48, retained_unchanged=42, fresh_replacement=6,
            additional_interpolation_points=12, prefill_gate_unchanged=True, mixed_gate_unchanged=True),
        candidate_sha256=digest(out/'candidate.json'), scheduling_proxy=scheduling_proxy(raw,points),
        formal_eligible=False, fit_performed=False, independent_holdout=True,
        qualification_frequency=2100, additional_qualification_frequencies=[],
        qualification_limit='Reuse this resident engine existing new 4+4 ProfileWave check only. No attribution of low-frequency drift without a dedicated paired B32 comparison.',
        lifecycle='After 32B power points, inside same EngineClient and ProfileWave measurement, before member done and Fleet.stop.',
        context_limitation='Residual is constant across original supported contexts and trained at all three contexts. Fresh validation covers actual contexts reached from 1024, not every context. No new long-context coverage is claimed.',
        error_action='Keep all windows and failed receipt; never refit on this panel or retry only numerical failures.')
    write_immutable(out/'timing-plan.json', plan)
    # Only now read old validation identity/reuse evidence. No old observations
    # entered fitting, family selection, or the training comparisons above.
    original = Path(json.loads((ROOT/'results/2026-09-23/tp4-holdout-audit/32b.json').read_text())['artifact'])
    completion = json.loads((original/'completion.json').read_text())
    if (not completion['complete'] or completion.get('independent_holdout') is not True or
            completion['candidate_sha256'] != candidate['base_candidate_sha256'] or
            completion['raw_sha256'] != digest(original/'raw.json')):
        raise ValueError('old independent timing receipt identity invalid')
    inputs = dict(training_raw=raw_path, base_candidate=BASE/'candidate.json', original_manifest=BASE/'manifest.json',
        original_raw=original/'raw.json', original_completion=original/'completion.json',
        preparation_script=Path(__file__).resolve(), training_comparison=out/'training-comparison.json')
    manifest = dict(schema=1, status='prepared_training_only_timing_overlay', system='pdblend', model_id=raw['model_id'],
        model_hash=raw['model_hash'], tokenizer_hash=raw['tokenizer_hash'], tp=4, pp=1,
        candidate_sha256=digest(out/'candidate.json'), plan_sha256=digest(out/'timing-plan.json'),
        inputs={k:dict(path=str(p.resolve()),sha256=digest(p)) for k,p in inputs.items()},
        implementation_sha256=tc.source_hashes(), training_environment=raw['environment'],
        formal_eligible=False, energy_comparable=False, original_completion_unchanged=True,
        power_candidate_unchanged=True, power_measurements_not_fitted_or_merged=True)
    write_immutable(out/'manifest.json', manifest)
    _,_,loaded = tc.load_package(out)
    retained = tc.reuse_original(manifest, loaded)
    write_immutable(out/'retained-timing-audit.json', retained)
    if not retained['passed']:
        raise ValueError('unchanged original timing gates must remain passing before minimal repair')
    cv = report['constant_residual_shape_cv']; local=report['B32_training']; proxy=plan['scheduling_proxy']
    lines = ['# 32B TP4 timing-only knee candidate', '',
        'Original model-owned training only; no holdout fitting, no GPU execution, no queue changes. The candidate is not promoted.', '',
        f'B32 training MAPE / max: {local["mape"]*100:.3f}% / {local["maximum"]*100:.3f}%. Grouped middle-context CV MAPE / max: {cv["mape"]*100:.3f}% / {cv["maximum"]*100:.3f}% ({cv["supported"]}/{cv["total"]} windows supported). The base was refitted without the held shape in every fold. Boundary folds and leave-B32-out remain explicitly unsupported.', '',
        'The correction is a constant per-frequency residual multiplied by a triangular batch weight with knots 16/32/64. It preserves the original timing exactly outside 16 < B < 64 and preserves all original coverage bounds. Training residuals across short/middle/long contexts support the constant approximation; this does not establish fresh validation at every context.', '',
        'The original 48-point decode gate becomes the same 42 unaffected original points plus 6 fresh B32 points. Twelve fresh B24/B48 interpolation points are additional checks. All original prefill and 12 mixed-additivity checks remain; mixed additivity uses measured base-step times, so it is unaffected by the candidate. Original failed completion and superseded B32 windows remain immutable diagnostic evidence.', '',
        f'18 fresh points / 54 windows: window floor 6.30 min; measured single-request-prefill × batch scheduling proxy {proxy["serial_prefill_work_proxy_seconds"]/60:.2f} min; combined proxy {proxy["combined_proxy_seconds"]/60:.2f} min. This is not a measured batch runtime or bound; clock changes, barrier/warmup and cleanup are extra. No model reload is needed when called inside the resident power cohort.', '',
        'Training itself underestimates stable B32 timing, so model underfit is real. The older holdout also drifts at low frequencies; the existing 2100 MHz/B8 paired check cannot attribute or exclude that drift. No new six-frequency interference campaign is added. Fresh timing must independently pass <=10% for every actual-context window.', '',
        'Integration: await pdblend.profile.timing_calibration.collect_existing(profiler=profiler, client=client, gpus=gpus, package=package, out=out). Call after the 24 power points while EngineClient and ProfileWave.measurement remain active. Timing has its own raw/checkpoint, candidate, receipt and failed/pass result; it never mutates power data or old timing receipts.', '']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(package=str(out), B32_train_max=local['maximum'], grouped_cv_max=cv['maximum'],
        retained_original_passed=retained['passed'], additional_runtime_proxy_minutes=proxy['combined_proxy_seconds']/60,
        formal_eligible=False)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    prepare(p.parse_args().out.resolve())
