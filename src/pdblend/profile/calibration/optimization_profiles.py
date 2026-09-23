"""Independent, opt-in low-batch and common-window Mixed power components.

The fitting matrix and holdout are separate. A failed/missing holdout never
widens an existing profile; these components cannot qualify a formal campaign.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

from pdblend.profile.query.power_table import PowerCoverageError

KIND = 'pdblend_optimization_power_components_v1'
SERVING_ENTRYPOINT = 'pdblend_runtime.serve'
IDENTITY = ('system', 'model_id', 'model_hash', 'tokenizer_hash', 'tp', 'pp')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(point):
    return '/'.join(str(point[k]) for k in ('family', 'freq_mhz', 'batch', 'context_tokens',
                                            'chunk_tokens', 'prefill_rate_rps'))


def family_key(point):
    return '/'.join(str(point[k]) for k in ('family', 'freq_mhz', 'batch',
                                            'chunk_tokens', 'prefill_rate_rps'))


def make_plan(identity, *, frequencies=(1500,), contexts=(512, 2048),
              holdout_context=1280, mixed_batches=(8,), chunks=(512,), rates=(1.0,)):
    if (identity.get('system') != 'pdblend' or identity.get('pp') != 1 or
            type(identity.get('tp')) is not int or identity['tp'] < 1):
        raise ValueError('optimization measurements require PDBlend PP1 identity')
    if (len(contexts) != 2 or not 1 <= contexts[0] < holdout_context < contexts[1] <= 4096 or
            any(type(f) is not int or f not in (900,1200,1500,1800,2100,2520) for f in frequencies) or
            not frequencies or len(set(frequencies)) != len(frequencies) or
            any(type(b) is not int or not 1 <= b <= 32 for b in mixed_batches) or
            any(type(c) is not int or not 1 <= c <= 2048 for c in chunks) or
            any(not math.isfinite(r) or not .5 <= r <= 4 for r in rates)):
        raise ValueError('invalid bounded optimization matrix')
    plan = dict(kind=KIND, **{k:identity[k] for k in IDENTITY}, training=[], holdout=[],
        serving_entrypoint=SERVING_ENTRYPOINT,native_control=True,native_timing_crosscheck_passed=False,
        formal_eligible=False, energy_comparable=False, holdout_used_for_fit=False,
        batch_interpolation_qualified=False, pure_decode_power_scope='continuous_decode_only',
        mixed_power_scope='common_wall_clock_window_with_decode_and_periodic_prefill')
    for phase, ctxs in (('training', contexts), ('holdout', (holdout_context,))):
        for frequency in frequencies:
            for context in ctxs:
                shapes = [('decode', b, 0, 0.0) for b in (2, 3)]
                shapes += [('mixed', b, c, float(r)) for b in mixed_batches for c in chunks for r in rates]
                for family, batch, chunk, rate in shapes:
                    plan[phase].append(dict(family=family, freq_mhz=frequency, batch=batch,
                        context_tokens=context, chunk_tokens=chunk, prefill_rate_rps=rate,
                        repeats=3, settle_s=2.0, measure_s=5.0, phase=phase))
    return plan


def integrate_power(samples, start, end, gpu_count):
    """Trapezoidal group energy on the same declared wall-clock interval."""
    points = [(float(t), float(sum(values))) for t, values in samples if start <= t <= end]
    if (not math.isfinite(start) or not math.isfinite(end) or end <= start or len(points) < 2 or
            any(len(values) != gpu_count or any(not math.isfinite(v) or v <= 0 for v in values)
                for t, values in samples if start <= t <= end) or
            any(not math.isfinite(t) for t, _ in points) or
            any(a[0] >= b[0] for a,b in zip(points,points[1:]))):
        raise ValueError('invalid common-window power evidence')
    if points[0][0] - start > .5 or end - points[-1][0] > .5:
        raise ValueError('common-window power endpoint coverage missing')
    points = [(start,points[0][1]), *points, (end,points[-1][1])]
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(points,points[1:]))


def native_batch_observation(evidence):
    """Integrate real scheduler batch residency, clipped to the power window.

    Scheduler timestamps mark scheduling decisions, not CUDA kernel duration.
    The previous decision carries forward until the next; no RPC is issued
    inside the power interval. Pure low-batch windows need 99% exact residency.
    """
    trace=evidence['native_schedule'];before,after=trace['before'],trace['after']
    start,end=evidence['start_s'],evidence['end_s'];point=evidence['point']
    background=set(trace['background_request_ids'])
    if len(background) != point['batch'] or trace['before_received_s'] > start or trace['after_requested_s'] < end:
        raise ValueError('native batch snapshot overlapped power window or background shape differs')
    cursor=before.get('next_seq')
    if (type(cursor) is not int or type(after.get('next_seq')) is not int or after['next_seq'] < cursor or
            before.get('gap') is not False or after.get('gap') is not False):
        raise ValueError('native batch event cursor overflow/gap')
    rows=after.get('events',[])
    if [r.get('seq') for r in rows] != list(range(cursor+1,after['next_seq']+1)):
        raise ValueError('native batch event sequence incomplete')
    prefix=before.get('events',[])
    if (not prefix or prefix[-1].get('seq') != cursor or
            any(type(r.get('seq')) is not int for r in prefix) or
            any(a['seq'] >= b['seq'] for a,b in zip(prefix,prefix[1:]))):
        raise ValueError('native batch initial cursor has no ordered schedule history')
    events=[r for r in [*prefix,*rows] if r.get('kind')=='schedule']
    if any(not isinstance(r.get('timestamp'),(int,float)) or not math.isfinite(r['timestamp']) for r in events):
        raise ValueError('native scheduler timestamp missing')
    if any(a['timestamp'] > b['timestamp'] for a,b in zip(events,events[1:])):
        raise ValueError('native scheduler clock moved backwards')
    anchors=[r for r in events if r['timestamp'] <= start]
    if not anchors or start-anchors[-1]['timestamp'] > .5:
        raise ValueError('native schedule does not cover common-window beginning')
    selected=[anchors[-1],*(r for r in events if start < r['timestamp'] < end)]
    if end-selected[-1]['timestamp'] > .5:
        raise ValueError('native schedule does not cover common-window ending')
    if len({r.get('generation') for r in selected}) != 1:
        raise ValueError('native scheduler generation changed inside measurement')
    histogram={};exact=0.;count_histogram={}
    for index,row in enumerate(selected):
        queue=row.get('schedule_queue')
        if (not isinstance(queue,list) or not queue or any(not isinstance(r,str) for r in queue) or
                len(set(queue)) != len(queue) or type(row.get('prefill_mode')) is not bool):
            raise ValueError('native schedule batch fields invalid')
        left=max(start,row['timestamp'])
        right=min(end,selected[index+1]['timestamp'] if index+1<len(selected) else end)
        duration=max(0.,right-left)
        actual=len(set(queue)&background);foreign=len(set(queue)-background)
        name=f'{actual}/{len(queue)}/{int(row["prefill_mode"])}'
        record=histogram.setdefault(name,dict(background_decode_batch=actual,scheduled_batch=len(queue),
            prefill_mode=row['prefill_mode'],seconds=0.,schedule_decisions=0))
        record['seconds']+=duration;record['schedule_decisions']+=1
        count_histogram[str(len(queue))]=count_histogram.get(str(len(queue)),0)+1
        if actual==point['batch'] and foreign==0 and row['prefill_mode'] is False:
            exact+=duration
    covered=sum(r['seconds'] for r in histogram.values())
    if not math.isclose(covered,end-start,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('native schedule clipped coverage incomplete')
    for row in histogram.values():row['time_fraction']=row['seconds']/covered
    exact_fraction=exact/covered
    passed=point['family']!='decode' or exact_fraction>=.99
    return dict(passed=passed,exact_decode_batch_fraction=exact_fraction,minimum_exact_fraction=.99,
        histogram=[histogram[k] for k in sorted(histogram)],schedule_count_histogram=count_histogram,
        window_s=covered,start_seq=cursor,end_seq=after['next_seq'],
        semantics='scheduler_decision_residency_not_cuda_kernel_time',
        batch_interpolation_qualified=False)


def observations(raw, root, plan):
    root = Path(root).resolve()
    if (plan.get('serving_entrypoint') != SERVING_ENTRYPOINT or
            raw.get('binding',{}).get('serving_entrypoint') != SERVING_ENTRYPOINT):
        raise ValueError('optimization serving variant differs from native measurement plan')
    seen = set()
    result = {}
    for phase in ('training','holdout'):
        expected = {key(p):p for p in plan[phase]}
        if set(raw[phase]) != set(expected):
            raise ValueError('incomplete independent optimization matrix')
        result[phase] = []
        for name, point in expected.items():
            row = raw[phase][name]
            if row['point'] != point or len(row['repeats']) != 3:
                raise ValueError('optimization needs three exact repeats per point')
            values = []
            for index, repeat in enumerate(row['repeats']):
                path = (root/repeat['samples_file']).resolve()
                if (not path.is_relative_to(root) or digest(path) != repeat['samples_sha256'] or
                        repeat['samples_sha256'] in seen):
                    raise ValueError('changed or reused optimization sample')
                seen.add(repeat['samples_sha256'])
                data = json.loads(path.read_text())
                if data['point'] != point or data['repeat'] != index:
                    raise ValueError('optimization sample shape/phase differs')
                if data.get('serving_entrypoint') != SERVING_ENTRYPOINT:
                    raise ValueError('optimization window has another serving variant')
                native=native_batch_observation(data)
                if data.get('native_batch_observation') != native or not native['passed']:
                    raise ValueError('native actual batch distribution does not qualify offered low batch')
                if data.get('plan_sha256') != raw['binding']['plan_sha256']:
                    raise ValueError('optimization sample belongs to another plan')
                from pdblend.profile.collection.long_context_followup import qualification_receipt
                from pdblend.profile.calibration.short_version import verify_epoch
                qualification_receipt(root, repeat['qualification'])
                saved = repeat['qualification']
                if (data['epoch_binding']['qualification_sha256'] != saved['samples_sha256'] or
                        any(data['epoch_binding'][k] != saved[k] for k in ('epoch_id','layout_sha256'))):
                    raise ValueError('optimization epoch binding changed')
                verify_epoch(data,root/saved['samples_file'])
                summary = data['summary']
                from pdblend.profile.collection.window_sampling import summarize_window
                derived = summarize_window(token_times=data['token_times_s'],context=point['context_tokens'],
                    start_s=data['start_s'],end_s=data['end_s'],power=data['power'],frequency=data['frequency'],
                    gpu_count=data['gpu_count'],settle_s=data['start_s']-data['settle_start_s'],
                    measurement_s=point['measure_s'])
                if any(summary[k] != v for k,v in derived.items() if k != 'power_w'):
                    raise ValueError('optimization raw timing/context summary changed')
                if summary['raw_window_mean_power_w'] != derived['power_w']:
                    raise ValueError('optimization original power summary changed')
                if point['family'] == 'mixed':
                    probes = data['probes']
                    if (len(probes) != round(point['measure_s']*point['prefill_rate_rps']) or
                            any(not data['start_s'] <= p['submitted_s'] <= p['first_token_s'] <= p['finished_s'] <= data['end_s']
                                for p in probes) or
                            any(abs(p['submitted_s']-(data['start_s']+i/point['prefill_rate_rps'])) > .1
                                for i,p in enumerate(probes))):
                        raise ValueError('optimization measured Mixed cadence/window differs')
                if (summary['steady_window_s'] < 5 or summary['settle_s'] < 2 or
                        summary['min_steps'] < 8 or
                        abs(summary['mean_freq_mhz']/point['freq_mhz']-1) > .05):
                    raise ValueError('optimization sampling duration/clock/progress gate failed')
                measured = integrate_power(data['power'],data['start_s'],data['end_s'],data['gpu_count'])
                if not math.isclose(summary['energy_j'], measured, rel_tol=1e-12):
                    raise ValueError('optimization common-window energy changed')
                if not math.isclose(summary['power_w'], measured/(data['end_s']-data['start_s']),rel_tol=1e-12):
                    raise ValueError('optimization power and energy scopes differ')
                values.append(summary)
            result[phase].append(dict(point=point, repeats=values))
    return result


def fit_component(plan, training, base_sha256):
    if {key(r['point']) for r in training} != {key(p) for p in plan['training']}:
        raise ValueError('exact training matrix required, holdout may not be fitted')
    nodes = {}
    for row in training:
        values = row['repeats']
        nodes.setdefault(family_key(row['point']), []).append(dict(
            context_min=min(r['effective_context_tokens'] for r in values),
            context_max=max(r['effective_context_tokens'] for r in values),
            power_w=statistics.median(r['power_w'] for r in values),
            measured_nominal_context=row['point']['context_tokens']))
    for points in nodes.values():
        points.sort(key=lambda p:p['context_min'])
        if len(points) != 2 or points[0]['context_max'] >= points[1]['context_min']:
            raise ValueError('training bands must be disjoint and ordered')
    return dict(kind=KIND, **{k:plan[k] for k in IDENTITY}, nodes=nodes,
        serving_entrypoint=plan['serving_entrypoint'],native_control=True,native_timing_crosscheck_passed=False,
        base_profile_sha256=base_sha256, holdout_used=False, batch_interpolation_qualified=False,
        formal_eligible=False, energy_comparable=False,
        mixed_power_scope=plan['mixed_power_scope'], decode_power_scope=plan['pure_decode_power_scope'])


def predict(candidate, point, context):
    points = candidate['nodes'].get(family_key(point))
    if not points or not math.isfinite(context) or not points[0]['context_min'] <= context <= points[-1]['context_max']:
        raise PowerCoverageError('missing_profile: exact optimization shape/rate/context unavailable')
    for node in points:
        if node['context_min'] <= context <= node['context_max']:
            return node['power_w']
    for left,right in zip(points,points[1:]):
        if left['context_max'] < context < right['context_min']:
            ratio=(context-left['context_max'])/(right['context_min']-left['context_max'])
            return left['power_w']+ratio*(right['power_w']-left['power_w'])
    raise PowerCoverageError('missing_profile: optimization gap')


def audit_component(candidate, plan, heldout):
    if {key(r['point']) for r in heldout} != {key(p) for p in plan['holdout']}:
        raise ValueError('independent exact holdout matrix required')
    errors, failures, residuals = [], [], {}
    for row in heldout:
        powers = [r['power_w'] for r in row['repeats']]
        if len(powers) != 3 or min(powers) <= 0:
            raise ValueError('three positive independent holdout windows required')
        cv = statistics.stdev(powers)/statistics.mean(powers)
        local = []
        for index, measured in enumerate(row['repeats']):
            try:
                value = predict(candidate,row['point'],measured['effective_context_tokens'])
                error = abs(value/measured['power_w']-1)
            except PowerCoverageError:
                value, error = None, None
            record = dict(point=key(row['point']),repeat=index,relative_error=error,
                          predicted_power_w=value,observed_power_w=measured['power_w'],
                          residual_w=None if value is None else measured['power_w']-value)
            errors.append(record); local.append(error)
            if error is not None:
                residuals.setdefault(family_key(row['point']),[]).append(record)
            if error is None or not math.isfinite(error) or error > .15:
                failures.append(record)
        if cv > .10 or any(x is None for x in local) or statistics.mean(x for x in local if x is not None) > .10:
            failures.append(dict(point=key(row['point']),metric='power_repeat_cv_or_mean_error',cv=cv))
    bounds = {name:dict(mean_relative_error=statistics.mean(r['relative_error'] for r in rows),
                       max_relative_error=max(r['relative_error'] for r in rows),
                       max_abs_residual_w=max(abs(r['residual_w']) for r in rows)) for name,rows in residuals.items()}
    return dict(passed=not failures, independent_holdout=True, errors=errors,failures=failures,residuals=bounds,
        actual_batch_distribution_required=True,minimum_exact_low_batch_fraction=.99,
        power_max_limit=.15,power_mean_limit=.10,power_repeat_cv_limit=.10,
        formal_eligible=False,energy_comparable=False,
        missing_gates=['native_timing','full_profile','formal_workload_energy'],
        mixed_energy_is_pure_decode=False)


def missing_domain_ledger(model, queries):
    """Rank actual missed queries by demand and decision sensitivity."""
    missing = []
    for query in queries:
        point = deepcopy(query)
        try:
            if point['family'] == 'mixed':
                ok = hasattr(model,'mixed_power_supported') and model.mixed_power_supported(
                    point['batch'],point['context_tokens'],point['freq_mhz'],
                    chunk_tokens=point['chunk_tokens'],prefill_rate_rps=point['prefill_rate_rps'])
            else:
                ok = model.decode_power_supported(point['batch'],point['context_tokens'],point['freq_mhz'])
        except (ValueError, TypeError):
            ok = False
        if not ok:
            point.update(reason='missing_measured_power_domain', priority=
                float(point.get('query_count',1))*abs(float(point.get('decision_sensitivity_j',1))))
            missing.append(point)
    return sorted(missing,key=lambda row:row['priority'],reverse=True)


def merge_components(roots, out):
    """Publish a reference-only union of disjoint qualified panels; never refit."""
    from pdblend.profile.calibration.power_calibration import write_immutable
    panels, occupied, identity = [], set(), None
    for root in map(lambda p:Path(p).resolve(),roots):
        completion = json.loads((root/'completion.json').read_text())
        if not completion.get('complete') or not completion.get('components_passed'):
            raise ValueError('cannot merge unqualified optimization component')
        for name in ('raw','candidate','audit'):
            if digest(root/(name+'.json')) != completion.get(name+'_sha256'):
                raise ValueError('optimization component changed before union')
        raw = json.loads((root/'raw.json').read_text())
        plan = json.loads((root/'plan.json').read_text())
        if digest(root/'plan.json') != raw['binding']['plan_sha256']:
            raise ValueError('optimization component plan changed')
        candidate = json.loads((root/'candidate.json').read_text())
        measured = observations(raw,root,plan)
        audit = audit_component(candidate,plan,measured['holdout'])
        if (candidate != fit_component(plan,measured['training'],candidate['base_profile_sha256']) or
                audit != json.loads((root/'audit.json').read_text()) or not audit['passed']):
            raise ValueError('optimization independent audit does not reproduce')
        current = {k:candidate[k] for k in (*IDENTITY,'base_profile_sha256','serving_entrypoint')}
        if identity is not None and identity != current:
            raise ValueError('cannot union different base/model/tokenizer/topology')
        identity = current
        if occupied.intersection(candidate['nodes']):
            raise ValueError('optimization union domains overlap; no implicit precedence')
        occupied.update(candidate['nodes'])
        panels.append(dict(path=str(root),completion_sha256=digest(root/'completion.json')))
    if len(panels) < 2:
        raise ValueError('a union requires two or more independent panels')
    value = dict(kind='pdblend_optimization_power_union_v1',components=panels,**identity,
                 refit_performed=False,formal_eligible=False,energy_comparable=False)
    write_immutable(Path(out),value)
    return value
