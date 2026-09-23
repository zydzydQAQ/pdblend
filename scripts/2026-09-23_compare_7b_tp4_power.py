#!/usr/bin/env python3
"""Training-only power-fit comparison; never runs GPUs or edits active profiles.

All repeats of a nominal shape stay in the same cross-validation fold. Table
predictions interpolate observed effective contexts and never enlarge coverage
into a rectangular nominal-context domain. Outputs are proposals, not profiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FREQUENCIES = (900, 1200, 1500, 1800, 2100, 2520)
FAMILIES = ('legacy_batch_affine_refit', 'all_batch_affine', 'split_b1_context_affine',
            'bounded_table_linear_batch', 'bounded_table_log_batch')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    return (row['freq_mhz'], row['batch'], row['context_tokens'])


def errors(observed, predicted):
    pairs = [(float(y), float(p)) for y, p in zip(observed, predicted) if p is not None]
    e = [abs(p / y - 1) for y, p in pairs]
    return dict(total=len(observed), supported=len(e), unsupported=len(observed)-len(e),
                mape=float(np.mean(e)) if e else None, max_error=max(e) if e else None,
                power_training_gate=bool(e) and np.mean(e) <= .10 and max(e) <= .15)


def make_nodes(rows):
    return [dict(batch=r['batch'], nominal_context_tokens=r['context_tokens'],
                 context_min=min(p['effective_context_tokens'] for p in r['repeats']),
                 context_max=max(p['effective_context_tokens'] for p in r['repeats']),
                 power_w=float(np.mean([p['power_w'] for p in r['repeats']])))
            for r in rows]


def table_predict(nodes, batch, context, *, log_batch=False):
    """Flat only inside observed repeat context bands; linear between bands.

    B1 is a distinct execution regime. Missing B2/B3 and boundary gaps are
    unsupported, even if a broad min/max rectangle would include them.
    """
    batches = sorted({r['batch'] for r in nodes if (r['batch'] == 1) == (batch == 1)})

    def at(b):
        points = sorted((p for p in nodes if p['batch'] == b), key=lambda p: p['context_min'])
        if not points or context < points[0]['context_min'] or context > points[-1]['context_max']:
            return None
        for p in points:
            if p['context_min'] <= context <= p['context_max']:
                return p['power_w']
        for lo, hi in zip(points, points[1:]):
            if lo['context_max'] < context < hi['context_min']:
                w = (context-lo['context_max'])/(hi['context_min']-lo['context_max'])
                return lo['power_w'] + w*(hi['power_w']-lo['power_w'])
        return None

    if batch in batches:
        return at(batch)
    lower, upper = [b for b in batches if b < batch], [b for b in batches if b > batch]
    if not lower or not upper:
        return None
    lo, hi = max(lower), min(upper)
    lp, hp = at(lo), at(hi)
    if lp is None or hp is None:
        return None
    x = math.log2 if log_batch else float
    w = (x(batch)-x(lo))/(x(hi)-x(lo))
    return lp + w*(hp-lp)


def features(batch, context, family):
    if family == 'split_b1_context_affine':
        if batch == 1:
            return np.array([1., context/4096])
        return np.array([1., batch/256, context/4096, batch*context/(256*4096)])
    return np.array([1., batch/256])


def predictor(rows, family):
    nodes = make_nodes(rows)
    if family.startswith('bounded_table_'):
        return lambda b, c: table_predict(nodes, b, c, log_batch=family.endswith('log_batch'))
    models = {}
    regimes = (True, False) if family == 'split_b1_context_affine' else (None,)
    for regime in regimes:
        chosen = [r for r in rows if (family != 'legacy_batch_affine_refit' or r['batch'] >= 4)
                  and (regime is None or (r['batch'] == 1) == regime)]
        if not chosen:
            continue
        x = np.asarray([features(r['batch'], float(np.mean([p['effective_context_tokens'] for p in r['repeats']])), family)
                        for r in chosen])
        y = np.asarray([float(np.mean([p['power_w'] for p in r['repeats']])) for r in chosen])
        # Historical batch-affine was ordinary LS; the alternative contextual
        # model explicitly optimizes relative error without holdout observations.
        if family == 'split_b1_context_affine':
            x, y = x / y[:, None], np.ones_like(y)
        if len(chosen) >= x.shape[1]:
            models[regime] = np.linalg.lstsq(x, y, rcond=None)[0]

    def predict(batch, context):
        # Same conservative measured domain for every candidate family.
        if table_predict(nodes, batch, context) is None:
            return None
        key = (batch == 1) if family == 'split_b1_context_affine' else None
        if key not in models:
            return None
        value = float(features(batch, context, family) @ models[key])
        return value if math.isfinite(value) and value > 0 else None
    return predict


def split_group(rows, held_row, axis):
    """Frequency is modeled independently. Never split repeated windows."""
    def same(r):
        if axis == 'shape':
            return (r['batch'], r['context_tokens']) == (held_row['batch'], held_row['context_tokens'])
        if axis == 'batch':
            return r['batch'] == held_row['batch']
        if axis == 'nominal_context':
            return r['context_tokens'] == held_row['context_tokens']
        raise ValueError(axis)
    group = [r for r in rows if r['freq_mhz'] == held_row['freq_mhz']]
    return [r for r in group if not same(r)], [r for r in group if same(r)]


def compare(rows):
    report = {}
    for family in FAMILIES:
        values = {}
        for axis in ('training_resubstitution', 'shape', 'batch', 'nominal_context'):
            seen, predictions, observations, details = set(), [], [], []
            for held in rows:
                if identity(held) in seen:
                    continue
                if axis == 'training_resubstitution':
                    training = [r for r in rows if r['freq_mhz'] == held['freq_mhz']]
                    test = [held]
                else:
                    training, test = split_group(rows, held, axis)
                    assert not {identity(r) for r in training} & {identity(r) for r in test}
                predict = predictor(training, family)
                seen.update(identity(r) for r in test)
                for r in test:
                    for index, rep in enumerate(r['repeats']):
                        p = predict(r['batch'], rep['effective_context_tokens'])
                        observations.append(rep['power_w']); predictions.append(p)
                        details.append(dict(freq_mhz=r['freq_mhz'], batch=r['batch'],
                            nominal_context_tokens=r['context_tokens'], repeat=index,
                            effective_context_tokens=rep['effective_context_tokens'], observed_power_w=rep['power_w'],
                            predicted_power_w=p, relative_error=abs(p/rep['power_w']-1) if p is not None else None,
                            status='supported' if p is not None else 'outside_remaining_training_coverage'))
            values[axis] = dict(**errors(observations, predictions), rows=details)
        report[family] = values
    return report


def validate_training(raw, path):
    wanted = dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', tp=4, pp=1)
    if any(raw.get(k) != v for k, v in wanted.items()) or raw.get('holdout_independent') is not False:
        raise ValueError('requires independent PDBlend 7B TP4 PP1 training raw, never holdout')
    if tuple(sorted(raw['freqs'])) != FREQUENCIES:
        raise ValueError('missing one of the six training frequencies')
    if raw.get('config', {}).get('decode_settle_s', 0) < 2:
        raise ValueError('training settle protocol is insufficient')
    rows = raw['decode']
    expected = {(f, b, c) for f in FREQUENCIES for b in (1,4,8,16,32,64,128,256) for c in (256,1024,4096)}
    if len(rows) != len(expected) or {identity(r) for r in rows} != expected:
        raise ValueError('requires the full, unmodified 144-shape training matrix')
    checked = []
    for row in rows:
        if len(row['repeats']) < 3:
            raise ValueError('insufficient repeats')
        for rep in row['repeats']:
            sample_path = (path.parent / rep['samples_file']).resolve()
            if not sample_path.is_relative_to(path.parent.resolve()) or digest(sample_path) != rep['samples_sha256']:
                raise ValueError('invalid raw sample checksum/path')
            sample = json.loads(sample_path.read_text())
            if (rep['steady_window_s'] < 5 or rep['min_steps'] < 8 or rep['power_samples'] < 2 or
                    rep['frequency_samples'] < 1 or len(sample['power']) < 2 or not sample['frequency']):
                raise ValueError('sampling gates failed')
            measured_power = float(np.mean([sum(powers) for _, powers in sample['power']]))
            measured_context = row['context_tokens'] + float(np.mean(
                [(a+b)/2 for a,b in zip(sample['start_token_counts'], sample['end_token_counts'])]))
            if not math.isclose(measured_power, rep['power_w'], rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError('power does not match raw group samples')
            if not math.isclose(measured_context, rep['effective_context_tokens'], abs_tol=1e-6):
                raise ValueError('effective context does not match raw token counts')
            checked.append(dict(file=str(sample_path), sha256=rep['samples_sha256']))
    return checked


def fresh_plan(raw, candidate, candidate_path):
    """Training-derived strata, chosen before any new power validation exists."""
    points = []
    for freq in FREQUENCIES:
        # B1 launch regime, B64 measured trough, B256 upper context, unseen
        # intermediate batch. All targets are inside training table support.
        for batch, target, reason in ((1,1024,'single_sequence_execution_regime'),
                (64,2048,'measured_nonmonotonic_batch_trough'),
                (256,4096,'high_batch_high_context_power'),
                (192,2048,'unseen_batch_interpolation')):
            nodes = candidate['tables'][str(freq)]
            prediction = table_predict(nodes,batch,target)
            if prediction is None:
                raise ValueError('holdout target must be inside the frozen training table')
            # Estimate initial prompt only. Every actual window is evaluated
            # at its measured context; no nominal substitution or clipping.
            nearest = min((r for r in raw['decode'] if r['freq_mhz']==freq),
                key=lambda r: abs(math.log2(r['batch']/batch))*10000+abs(r['effective_context_tokens']-target))
            step = nearest['step_seconds']
            offset = math.ceil(4.5/step)
            prompt = max(128,target-offset)
            max_new = math.ceil(10/step)+64
            # Optional shared-prefill protocol uses exactly three distinct
            # 2+5 s windows. Move high-context B256 earlier so window three
            # remains inside measured coverage; never reuse one power window.
            same_batch = [r for r in raw['decode'] if r['freq_mhz']==freq and r['batch']==nearest['batch']]
            fast_step = min(r['step_seconds'] for r in same_batch)
            shared_prompt = max(128,target-math.ceil((18.5 if batch==256 else 4.5)/fast_step))
            shared_contexts = [shared_prompt+(4.5+7*i)/fast_step for i in range(3)]
            shared_max_new = math.ceil(30/fast_step)+64
            shared_eligible = (all(table_predict(nodes,batch,c) is not None for c in shared_contexts)
                and shared_prompt+shared_max_new<=8192
                and batch*(shared_prompt+shared_max_new)<=.9*raw['kv_capacity_tokens'])
            if not shared_eligible:
                raise ValueError('shared-prefill proposal does not meet frozen support/reservation')
            if prompt+max_new>8192 or batch*(prompt+max_new)>.9*raw['kv_capacity_tokens']:
                raise ValueError('power-only holdout exceeds physical reservation')
            points.append(dict(freq_mhz=freq,batch=batch,context_tokens=prompt,
                target_effective_context_tokens=target,initial_prompt_estimate_only=True,
                max_tokens=max_new,reason=reason,repeats=3,settle_s=2,measure_s=5,
                minimum_decode_steps=8,minimum_power_samples=2,minimum_frequency_samples=1,
                per_repeat_actual_context_required=True,training_table_prediction_w=prediction,
                shared_prefill_option=dict(context_tokens=shared_prompt,max_tokens=shared_max_new,
                    estimated_effective_contexts=shared_contexts,estimated_only=True,
                    distinct_windows=3,settle_seconds_per_window=2,measure_seconds_per_window=5,
                    actual_each_window_context_and_prediction_required=True,
                    fallback='separate_prefill_windows_if_trajectory_or_measurement_is_outside_frozen_support'),
                outside_coverage_action='reject_window_as_measurement_domain_error; do_not_clip_or_extrapolate'))
    def prefill_cost(point, prompt):
        rows=sorted((r for r in raw['prefill'] if r['freq_mhz']==point['freq_mhz']),key=lambda r:r['input_tokens'])
        if prompt<rows[0]['input_tokens'] or prompt>rows[-1]['input_tokens']:
            raise ValueError('prefill scheduling proxy outside measured lengths')
        return point['batch']*float(np.interp(prompt,[r['input_tokens'] for r in rows],[r['seconds'] for r in rows]))
    costs=dict(separate_prefill_serial_work_proxy_seconds=sum(3*prefill_cost(p,p['context_tokens']) for p in points),
        shared_prefill_serial_work_proxy_seconds=sum(prefill_cost(p,p['shared_prefill_option']['context_tokens']) for p in points),
        formula='batch * single-request prefill interpolation within measured lengths; times three for separate starts',
        actual_batch_runtime_measured=False,not_a_bound=True,excluded='model load, warmup barriers, clock changes, qualification, cleanup, retries')
    return dict(schema=1,purpose='fresh_independent_decode_power_holdout',system='pdblend',
        model_id=raw['model_id'],model_hash=raw['model_hash'],tokenizer_hash=raw['tokenizer_hash'],tp=4,pp=1,
        candidate=str(candidate_path.resolve()),candidate_sha256=digest(candidate_path),points=points,
        point_count=len(points),minimum_decode_window_seconds=len(points)*3*7,
        model_loads=1,gpu_count=4,enqueue=False,executable_collector_available=False,cpu_scheduling_proxy=costs,
        qualification='existing paired-layout 2100 MHz representative protocol if parallel; do not add six qualification sweeps',
        clock_frequencies=list(FREQUENCIES),power_unit='sum_of_the_four_assigned_GPUs_watts',
        gates=dict(mape_max=.10,max_error_max=.15,independent_holdout_each_window_max_error=.10),
        preserve_timing='Original timing coefficients, raw timestamps and passing timing receipts remain immutable and reusable for timing only.',
        composite_audit='Bind original timing receipt hashes plus this new candidate and fresh power-only receipt. Old failed power observations remain historical diagnostics; never relabel them independent for this candidate.',
        no_training_on_holdout=True,formal_eligible=False,
        timing_recollection_required=False,prefill_recollection_required=False,mixed_recollection_required=False,
        mixed_additive_audit='Re-evaluate the new power model against separately valid existing mixed raw/provenance. Missing or failed mixed evidence remains unresolved; this decode-only plan cannot waive that gate.',
        limitations=['Window budget excludes model load, prefill, warmup barrier, frequency changes, qualification and cleanup.',
            'Proposed shapes are an incremental power validation panel, not completion of every workload/long-context domain.',
            'Batch 2/3 and context outside measured effective-context support remain missing_profile.',
            'The stable core power API still lacks context; implementation and call-site review must precede collection/promotion.'])


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists() and path.read_text()!=text:
        raise ValueError(f'refusing to overwrite a different review artifact: {path}')
    path.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-raw',type=Path,required=True)
    parser.add_argument('--base-candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    raw=json.loads(args.training_raw.read_text());base=json.loads(args.base_candidate.read_text())
    checked=validate_training(raw,args.training_raw)
    if (base['system'],Path(base['model']).name,base['tp'],base['pp'])!=('pdblend',raw['model_id'],4,1):
        raise ValueError('base candidate identity mismatch')
    if base['quality']['calibration']['training_raw_sha256']!=digest(args.training_raw):
        raise ValueError('candidate is not bound to this immutable training raw')
    cv=compare(raw['decode'])
    # Select only within predeclared training-CV families: lower worst
    # leave-batch-out error breaks the otherwise comparable table candidates.
    choices=('bounded_table_linear_batch','bounded_table_log_batch')
    selected=min(choices,key=lambda k:cv[k]['batch']['max_error'])
    candidate=dict(schema=1,status='training_only_proposal',formal_eligible=False,
        system='pdblend',model_id=raw['model_id'],model_hash=raw['model_hash'],tokenizer_hash=raw['tokenizer_hash'],tp=4,pp=1,
        family=selected,override_scope='decode_power_only',power_unit='sum_of_assigned_GPU_watts',
        training_raw=str(args.training_raw.resolve()),training_raw_sha256=digest(args.training_raw),
        base_candidate=str(args.base_candidate.resolve()),base_candidate_sha256=digest(args.base_candidate),
        script_sha256=digest(__file__),timing_unchanged=True,independent_power_holdout_required=True,
        tables={str(f):make_nodes([r for r in raw['decode'] if r['freq_mhz']==f]) for f in FREQUENCIES},
        domain=dict(frequencies=list(FREQUENCIES),frequency_interpolation=False,extrapolation=False,
            split_single_sequence=True,unsupported_batches=[2,3],nominal_context_is_not_coverage=True,
            context_rule='Per-batch observed effective-context bands; linear only between bands. Between batches >=4 requires context supported on both adjacent measured batches.',
            long_context_ge_5120='missing_profile'),
        selection='Training-only grouped CV; lower leave-whole-batch-out maximum error among two bounded table families.',
        promotion='Fresh independent power holdout, core context-aware API integration and composite provenance/mixed audit required.')
    candidate_path=args.out/'power-candidate-proposal.json';write(candidate_path,candidate)
    report=dict(schema=1,formal_eligible=False,training_only=True,holdout_used_for_fit_or_selection=False,
        training_raw_sha256=digest(args.training_raw),base_candidate_sha256=digest(args.base_candidate),
        training_windows_checked=len(checked),raw_sample_bindings=checked,
        exact_frozen_candidate_training=errors([p['power_w'] for r in raw['decode'] for p in r['repeats']],
            [sum(a*x for a,x in zip(base['decode_power'][str(r['freq_mhz'])],(1,r['batch'])))
             for r in raw['decode'] for p in r['repeats']]),
        error_definition='abs(predicted / observed - 1)',selected_family=selected,comparisons=cv,
        boundary_note='Unsupported CV boundaries are coverage limitations, never zero error. Shape/context folds can only test the middle nominal-context slice; batch folds only interior batches where adjacent contexts overlap.',
        leakage_policy='All three repeated windows of every (frequency, batch, nominal context) are assigned to the same held-out group.',
        training_resubstitution_is_validation=False)
    write(args.out/'training-power-comparison.json',report)
    plan=fresh_plan(raw,candidate,candidate_path);write(args.out/'fresh-power-only-holdout-plan.json',plan)
    lines=['# 7B TP4 training-only power model comparison','',
        'No holdout data were read by this script. All 432 training windows were verified against their declared raw checksums, actual GPU-group power and effective token contexts. Outputs are review proposals; no core profile or queue was changed.','',
        '| Family | Training MAPE / max | Leave-shape-out MAPE / max | Leave-batch-out MAPE / max |','|---|---:|---:|---:|']
    for family in FAMILIES:
        vals=[]
        for axis in ('training_resubstitution','shape','batch'):
            v=cv[family][axis];vals.append(f'{100*v["mape"]:.2f}% / {100*v["max_error"]:.2f}% ({v["supported"]}/{v["total"]})')
        lines.append('| '+family+' | '+' | '.join(vals)+' |')
    lines+=['',f'Recommended provisional family: `{selected}`. This retains the measured B1 regime and context-dependent high-batch power; it also preserves the B64 power trough instead of imposing batch monotonicity.',
        '',f'Exact frozen candidate on all training repeats: MAPE={100*report["exact_frozen_candidate_training"]["mape"]:.2f}%, maximum={100*report["exact_frozen_candidate_training"]["max_error"]:.2f}%. The `legacy_batch_affine_refit` family is re-fitted independently within each training fold using per-shape mean powers; it is not the exact original coefficient file.',
        '',report['boundary_note'],'',
        'The table keeps each nominal shape’s observed effective-context repeat band at its measured mean power and interpolates only between bands. Training residuals assess repeat noise, not independent accuracy. B2/B3 are intentionally unsupported; crossing B1 to B4 would assume an unmeasured execution regime.',
        '',f'Fresh power-only panel: {plan["point_count"]} shapes (six frequencies × B1, B64, B256 and unseen B192), three independent 2+5 s windows each, {plan["minimum_decode_window_seconds"]/60:.1f} minutes of measurement/settle windows. One resident TP4 engine; prefill/load/qualification overhead is additional. This is a prepared plan, not a runnable or queued GPU task.',
        '',f'CPU serial-prefill work proxy plus the same 8.4 min windows: separate starts {(plan["cpu_scheduling_proxy"]["separate_prefill_serial_work_proxy_seconds"]+504)/60:.1f} min; one shared prefill per point {(plan["cpu_scheduling_proxy"]["shared_prefill_serial_work_proxy_seconds"]+504)/60:.1f} min. These are scheduling proxies, not measured batch runtimes or guaranteed bounds. The optional shared path keeps three separate 2+5 s windows and evaluates each actual context; it requires a future collector implementation and must reject out-of-coverage windows.',
        '',plan['preserve_timing'],plan['composite_audit'],plan['mixed_additive_audit'],
        '', 'The existing `decode_power_w(batch, frequency)` API cannot represent this proposal: a context argument and all energy/planner/mixed-audit callers require a separate reviewed change. No stable API was changed here. Passing this incremental panel alone does not establish full-system eligibility.','']
    (args.out/'review.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(args.out),selected_family=selected,checked_windows=len(checked),
        fresh_power_holdout_points=len(plan['points']),minimum_window_seconds=plan['minimum_decode_window_seconds'])))


if __name__=='__main__':
    main()
