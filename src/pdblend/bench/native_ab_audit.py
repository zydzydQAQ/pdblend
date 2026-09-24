"""Independent, read-only acceptance of the bounded native PDBlend A/B smoke.

The two arms use one implementation and one resident fleet: A holds its initial
plan; B permits periodic control.  This audit never qualifies formal ranking or
total energy savings.  Failed or absent evidence is retained as inconclusive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from pdblend.online.native_control import validate_state


SCHEMA = 'pdblend.native-ab-audit/v1'
MODELS = {'Qwen2.5-7B-Instruct': 1, 'Qwen2.5-14B-Instruct': 1,
          'Qwen2.5-32B-Instruct': 2}
SHAPES = {512, 1024, 2048}
CHANGE_THRESHOLD = .05


def _number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _read(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _quantiles(values):
    ordered = sorted(values)
    return {f'p{round(q*100)}': ordered[max(0, math.ceil(len(ordered)*q)-1)] if ordered else None
            for q in (.50, .95, .99)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _binding(value):
    _require(isinstance(value, dict), 'missing arm bindings')
    _require(value.get('model_id') in MODELS and value.get('tp') == MODELS[value['model_id']]
             and value.get('pp') == 1, 'unregistered model/TP/PP')
    _require(value.get('seed') == 701 and _number(value.get('duration_s'), positive=True),
             'seed701 and an explicit positive observation duration are required')
    for key in ('source_sha256', 'trace_sha256'):
        _require(_sha(value.get(key)), 'missing immutable ' + key)
    _require(isinstance(value.get('image_digest'), str) and
             _sha(value['image_digest'].removeprefix('sha256:')), 'missing immutable image digest')
    inputs = value.get('inputs', {})
    for key in ('profile', 'trace', 'tuning_trace', 'model_verification'):
        _require(isinstance(inputs.get(key), dict) and _sha(inputs[key].get('sha256')),
                 'missing input identity: ' + key)
    _require(inputs['trace']['sha256'] == value['trace_sha256'], 'trace identities disagree')
    return {key: value[key] for key in ('model_id', 'tp', 'pp', 'seed', 'duration_s',
                                       'source_sha256', 'image_digest', 'trace_sha256')} | {
        'inputs': {key: inputs[key]['sha256'] for key in
                   ('profile', 'trace', 'tuning_trace', 'model_verification')}} | {
        key: value[key] for key in ('initial_plan', 'policy') if key in value}


def _trace(root, binding):
    reference = binding['inputs']['trace']
    candidates = [root/'trace.json']
    if reference.get('path'):
        candidates.append(root/Path(reference['path']))
    path = next((p for p in candidates if p.is_file()), None)
    _require(path is not None, 'frozen evaluation trace is unavailable')
    payload = path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == reference['sha256'], 'trace checksum mismatch')
    value = json.loads(payload)
    rows = value if isinstance(value, list) else value.get('requests')
    _require(isinstance(rows, list) and rows, 'empty evaluation trace')
    expected = {}
    for row in rows:
        _require(isinstance(row, dict) and type(row.get('idx')) is int and row['idx'] not in expected,
                 'invalid or duplicate trace request ID')
        prompt = row.get('prompt')
        _require(isinstance(prompt, list) and len(prompt) in SHAPES and
                 all(type(token) is int and token >= 0 for token in prompt), 'unexpected input shape/token payload')
        _require(row.get('max_tokens') == 128 and _number(row.get('arrival_s')) and
                 row['arrival_s'] < binding['duration_s'], 'trace output workload or arrival is invalid')
        expected[row['idx']] = row
    _require({len(row['prompt']) for row in rows} == SHAPES, 'trace must cover all three approved input shapes')
    return expected


def _cleanup(value, *, tp, expected_instances=None, generations=None):
    _require(isinstance(value, dict) and value, 'missing native cleanup inventory')
    if expected_instances is not None:
        _require(set(value) == expected_instances, 'native cleanup omits or adds fleet instances')
    observed = {}
    for iid, receipt in value.items():
        generation = receipt.get('generation')
        _require(type(generation) is int and generation >= 0, 'native generation absent')
        if generations is not None:
            _require(generations.get(iid) == generation, 'native generation changed between arms')
        validate_state(receipt, generation=generation, tp=tp, pp=1, drained=True)
        _require(receipt.get('acknowledged') is True and receipt.get('drained') is True,
                 'native drain ACK is absent')
        _require(_number(receipt.get('native_at_s'), positive=True), 'native receipt lacks observation time')
        for rank in receipt['ranks']:
            _require(_number(rank.get('at_s'), positive=True), 'native rank receipt lacks observation time')
        observed[iid] = generation
    return observed


def _functional(root, *, tp):
    value = _read(root/'functional'/'completion.json')
    checks = value.get('checks', {})
    _require(value.get('functional_passed') is True and not value.get('error') and
             all(checks.get(key) is True for key in ('native_cancel_and_kv_cleanup',
                 'reservation_released', 'admission_reopened', 'successful_reuse')),
             'functional cancellation/reuse qualification failed')
    receipts = value.get('cancel_receipts')
    _require(isinstance(receipts, dict) and receipts, 'native cancellation receipts are absent')
    request_ids = set()
    for iid, receipt in receipts.items():
        request_id = receipt.get('request_id')
        _require(isinstance(request_id, str) and request_id and
                 receipt.get('acknowledged') is True and receipt.get('cancelled') is True and
                 receipt.get('instance_id') == iid, 'native cancellation ACK identity mismatch')
        generation = receipt.get('generation')
        _require(type(generation) is int and generation >= 0, 'cancel generation missing')
        validate_state(receipt.get('native_state'), generation=generation, tp=tp, pp=1,
                       request_id=request_id)
        request_ids.add(request_id)
    _require(len(request_ids) == 1, 'cancellation ranks disagree on request identity')
    reuse = value.get('reuse', {})
    times = reuse.get('token_times_s', [])
    _require(reuse.get('completion_tokens') in (128, 512) and isinstance(times, list) and times and
             all(_number(t, positive=True) for t in times) and times == sorted(times),
             'post-cancellation successful output evidence missing')
    return {'passed': True, 'cancelled_request_id': next(iter(request_ids)),
            'native_instances': sorted(receipts), 'reuse_tokens': reuse['completion_tokens']}


def _interference(root, binding):
    path = root/'qualification'/'comparison.json'
    comparison = _read(path)
    arm_modes = {_read(root/arm/'summary.json').get('comparison_mode') for arm in ('A', 'B')}
    _require(len(arm_modes) == 1, 'A/B concurrency modes disagree')
    mode = next(iter(arm_modes)) or comparison.get('comparison_mode', binding.get('comparison_mode', 'parallel'))
    if mode in ('serial', 'serial_after_interference'):
        receipt = _read(root/'qualification'/'serial-execution.json')
        intervals = receipt.get('intervals', {})
        _require(receipt.get('exclusive_member') is True and receipt.get('model_id') == binding['model_id']
                 and receipt.get('cohort_id') == binding.get('cohort_id'),
                 'serial fallback has no exclusive cohort membership receipt')
        for arm in ('A', 'B'):
            interval = intervals.get(arm, {})
            _require(_number(interval.get('start_s'), positive=True) and
                     _number(interval.get('end_s'), positive=True) and
                     interval['end_s']-interval['start_s'] >= binding['duration_s'],
                     'serial receipt does not cover complete ' + arm + ' window')
        _require(intervals['A']['end_s'] <= intervals['B']['start_s'], 'serial A/B windows overlap')
        return dict(passed=True, comparison_mode=mode, requires_serial_retest=False,
                    original_parallel_passed=comparison.get('passed') if mode == 'serial_after_interference' else None)
    measurements = [_read(root/'qualification'/f'{name}.json') for name in ('solo', 'parallel')]
    medians, powers = [], []
    for row in measurements:
        samples = row.get('latency_s')
        _require(isinstance(samples, list) and len(samples) >= 3 and all(_number(x, positive=True) for x in samples),
                 'paired interference probe has too few valid latency samples')
        medians.append(statistics.median(samples))
        _require(_number(row.get('mean_power_w'), positive=True), 'paired group power missing')
        powers.append(row['mean_power_w'])
        groups = row.get('gpu_uuids', row.get('group_gpu_uuids', row.get('group_gpus', row.get('gpus'))))
        _require(groups == measurements[0].get('gpu_uuids', measurements[0].get('group_gpu_uuids',
                 measurements[0].get('group_gpus', measurements[0].get('gpus')))) and isinstance(groups, list) and groups,
                 'paired probes used different or unspecified GPU groups')
        if all(isinstance(g, str) for g in groups):
            _require(groups == binding['gpu_uuids'], 'interference probes are not bound to the leased GPU UUIDs')
        else:
            _require(all(type(g) is int for g in groups) and len(groups) == len(binding['gpu_uuids'])
                     and len(set(groups)) == len(groups), 'invalid local group to leased UUID mapping')
    latency = abs(medians[1]-medians[0])/medians[0]
    power = abs(powers[1]-powers[0])/powers[0]
    passed = latency <= .05 and power <= .05 and comparison.get('members_unchanged', True) is True
    return dict(passed=passed, comparison_mode='parallel', latency_relative_deviation=latency,
                group_power_relative_deviation=power, requires_serial_retest=not passed,
                affected_model=binding['model_id'] if not passed else None)


def _arm(root, name, expected, expected_binding, *, gpu_count, tp):
    directory = root/name
    summary = _read(directory/'summary.json')
    binding = summary.get('bindings', summary.get('conditions'))
    _require(_binding(binding) == expected_binding, name + ' identity/trace differs from root')
    _require(summary.get('arm') == name and summary.get('formal_eligible') is False and
             summary.get('energy_comparable') is False, 'arm must identify its development-only scope')
    pids = summary.get('shared_lifecycle_pids')
    _require(isinstance(pids, (dict, list)) and pids, 'resident engine lifecycle identity absent')
    _require(summary.get('resident_processes_unchanged', True) is True, 'resident process changed during arm')
    if summary.get('comparison_mode', '').startswith('parallel'):
        _require(summary.get('cohort_members_unchanged') is True, 'parallel cohort changed during arm')
    rows = _rows(directory/'outcomes.jsonl')
    _require(len(rows) == len(expected) and {r.get('idx') for r in rows} == set(expected),
             name + ' requests missing/duplicated/foreign')
    ttfts, tpots, good, succeeded, tokens, good_tokens = [], [], 0, 0, 0, 0
    for row in rows:
        request = expected[row['idx']]
        _require(row.get('sampling_seed') == 701 and row.get('max_tokens') == 128 and
                 row.get('input_tokens') == len(request['prompt']) and
                 row.get('arrival_s') == request['arrival_s'], name + ' workload/seed changed')
        if row.get('error'):
            continue
        _require(row.get('completion_tokens') == 128 and row.get('path') in ('M', 'PD', 'P_ONLY'),
                 name + ' incomplete successful output or absent route')
        start, first, end = (row.get(k) for k in ('submitted_s', 'first_token_s', 'finished_s'))
        _require(all(_number(t, positive=True) for t in (start, first, end)) and start <= first <= end,
                 name + ' invalid request timing')
        ttft, tpot = first-start, (end-first)/127
        ttfts.append(ttft); tpots.append(tpot)
        succeeded += 1; tokens += 128
        if ttft <= 5. and tpot <= .15:
            good += 1; good_tokens += 128
    duration = expected_binding['duration_s']
    _require(_number(summary.get('window_s'), positive=True) and
             math.isclose(summary['window_s'], duration, abs_tol=1e-6), 'arm duration changed')
    metrics = dict(offered=len(expected), succeeded=succeeded, success_rate=succeeded/len(expected),
                   joint_slo_requests=good, joint_slo_rate=good/len(expected), output_tokens=tokens,
                   joint_output_tokens=good_tokens, ttft_s=_quantiles(ttfts), tpot_s=_quantiles(tpots),
                   throughput_request_s=succeeded/duration, throughput_token_s=tokens/duration,
                   goodput_request_s=good/duration, goodput_token_s=good_tokens/duration)
    metrics['slo_passed'] = (metrics['success_rate'] >= .9 and metrics['joint_slo_rate'] >= .9 and
                            metrics['ttft_s']['p99'] is not None and metrics['ttft_s']['p99'] <= 5. and
                            metrics['tpot_s']['p99'] is not None and metrics['tpot_s']['p99'] <= .15)
    native = _read(directory/'native-cleanup.json')
    generations = _cleanup(native, tp=tp)
    _require(len(generations)*tp == gpu_count, 'native cleanup does not cover the entire leased fleet')
    routes = _rows(directory/'routes.jsonl')
    successful = {f"r{row['idx']}": row for row in rows if not row.get('error')}
    earliest = min(row['submitted_s'] for row in rows)
    service_routes = [r for r in routes if r.get('request_id') in successful
                      and r.get('submitted_s', -1) >= earliest]
    _require(len(service_routes) == len(successful) and
             len({r['request_id'] for r in service_routes}) == len(successful),
             'successful service routing receipts missing/duplicated')
    for route in service_routes:
        _require(route.get('tp') == tp and route.get('pp') == 1 and
                 route.get('prefill_instance') in generations and route.get('decode_instance') in generations and
                 route.get('generation') == generations[route['prefill_instance']] == generations[route['decode_instance']],
                 'route rank/topology/generation differs from native fleet')
        _require(route.get('path') == successful[route['request_id']]['path'] and
                 route.get('terminal_state') == 'completed', 'successful outcome disagrees with its native route')
    log = _rows(directory/'controller.jsonl')
    forecasts = [r for r in log if r.get('kind') == 'forecast']
    plans = [r for r in log if r.get('kind') == 'plan']
    plan_identity_errors = []
    wanted = expected_binding.get('initial_plan', {})
    for row in plans:
        identity = row.get('plan_identity')
        if not isinstance(identity, dict):
            plan_identity_errors.append('executed Plan identity was not recorded')
        elif any(identity.get(key) != wanted.get(key) for key in
                 ('tp', 'pp', 'pool_id', 'generation', 'profile_key')):
            plan_identity_errors.append('executed Plan lost its TP/profile/generation identity')
    planning_path = directory/'planning-times.jsonl'
    planning = _rows(planning_path) if planning_path.is_file() else []
    for row in planning:
        _require(row.get('origin') == 'periodic_controller' and _number(row.get('duration_s')) and
                 _number(row.get('t'), positive=True), 'invalid measured periodic planner duration')
    if name == 'B':
        _require(len(forecasts) >= 2 and len(planning) >= 2, 'periodic controller/planner did not execute')
        _require(all(_number(r.get('t'), positive=True) and r.get('decision_reason') for r in forecasts),
                 'controller decisions lack time/reason')
    else:
        _require(not forecasts and not planning, 'fixed-plan control unexpectedly performed periodic planning')
    # Initialization is not an online action; count successful physical phases
    # only after the first observed periodic decision.
    decision_start = min((r['t'] for r in forecasts), default=math.inf)
    phases = [r for r in log if r.get('kind') == 'transition_phase' and r.get('status') == 'passed'
              and r.get('started_s', -1) >= decision_start and r.get('operation') in
              ('native_drain', 'native_resume', 'clock_set', 'clock_reset', 'park', 'unpark', 'stop', 'start')]
    automatic_pd = [r for r in service_routes if r.get('path') == 'PD' and r.get('terminal_state') == 'completed']
    pd_plans = [r for r in plans if r.get('t', 0) >= decision_start and
                r.get('counts', {}).get('P', 0) > 0 and r.get('counts', {}).get('D', 0) > 0 and
                not r.get('query_results', {}).get('mechanism_forced_roles')]
    meter = _read(directory/'metering.json')
    _require(meter.get('error') is None, 'metering failed')
    return dict(metrics=metrics, shared_lifecycle_pids=pids, generations=generations,
                plan_identity_errors=sorted(set(plan_identity_errors)),
                controller=dict(periodic_decisions=len(forecasts), periodic_planner_calls=len(planning),
                    planning_seconds=_quantiles([r['duration_s'] for r in planning]),
                    actual_actions=len(phases), automatic_pd_requests=len(automatic_pd),
                    automatic_pd_plans=len(pd_plans)), _routes=service_routes)


def _pd_golden(root):
    value = _read(root/'pd-golden.json')
    _require(value.get('status') == 'passed' and value.get('complete') is True,
             'PD golden receipt did not pass')
    rows = value.get('rows')
    _require(isinstance(rows, list) and rows and {row.get('input_tokens') for row in rows} >= {512, 2048},
             'PD golden lacks approved input lengths')
    for row in rows:
        reference = row.get('reference', [])
        _require(isinstance(reference, list) and len(reference) >= 2, 'repeatable PD golden reference missing')
        expected = reference[0].get('token_ids')
        _require(isinstance(expected, list) and expected and all(type(x) is int for x in expected),
                 'PD golden exact output IDs missing')
        values = reference + [row.get('client_pd', {}), row.get('proxy_PD', {}).get('completion', {})]
        for observed in values:
            _require(observed.get('token_ids') == expected and observed.get('completion_tokens') == len(expected)
                     and observed.get('prompt_tokens') == row['input_tokens'] and
                     observed.get('stream_done') is True and observed.get('usage_received') is True and
                     not observed.get('error'), 'PD golden output/usage mismatch')
        route = row.get('proxy_PD', {}).get('record', {})
        _require(route.get('path') == 'PD' and route.get('prefill_instance') != route.get('decode_instance'),
                 'PD golden never used disaggregated instances')
    return dict(passed=True, manual_mechanism_only=True, input_lengths=sorted(row['input_tokens'] for row in rows))


def classify(a, b, *, triggered):
    """Descriptive same-trace comparison, not statistical or energy evidence."""
    if not a['slo_passed'] or not b['slo_passed']:
        return '退化' if a['slo_passed'] and not b['slo_passed'] else 'inconclusive'
    if not triggered:
        return '未触发'
    changes = [(b['goodput_token_s']-a['goodput_token_s'])/max(a['goodput_token_s'], 1e-12)]
    for key in ('ttft_s', 'tpot_s'):
        changes.append((a[key]['p95']-b[key]['p95'])/max(a[key]['p95'], 1e-12))
    if min(changes) < -CHANGE_THRESHOLD:
        return '退化'
    return '改善' if max(changes) > CHANGE_THRESHOLD else '无改善'


def audit(root: Path) -> dict:
    root = Path(root)
    result = dict(schema=SCHEMA, status='inconclusive', classification='inconclusive',
                  formal_eligible=False, energy_comparable=False, scope='fixed_initial_vs_periodic_same_source',
                  change_threshold=CHANGE_THRESHOLD, errors=[], arms={}, requires_serial_retest=False)
    try:
        preflight = _read(root/'preflight.json')
        binding = preflight.get('bindings', preflight)
        # Runtime fields may be outside the immutable input binding.
        binding = dict(binding)
        for key in ('gpu_uuids', 'cohort_id', 'comparison_mode', 'seed', 'duration_s', 'trace_sha256'):
            if key not in binding and key in preflight:
                binding[key] = preflight[key]
        identity = _binding(binding)
        result['identity'] = identity
        gpus = binding.get('gpu_uuids')
        if isinstance(gpus, str):
            gpus = [value.strip() for value in gpus.split(',') if value.strip()]
            binding['gpu_uuids'] = gpus
        _require(isinstance(gpus, list) and len(gpus) in (2, 4, 8) and len(set(gpus)) == len(gpus)
                 and all(isinstance(g, str) and g.startswith('GPU-') for g in gpus), 'invalid GPU UUID lease')
        expected = _trace(root, binding)
        result['functional'] = _functional(root, tp=binding['tp'])
        result['interference'] = _interference(root, binding)
        for arm in ('A', 'B'):
            result['arms'][arm] = _arm(root, arm, expected, identity, gpu_count=len(gpus), tp=binding['tp'])
        a, b = result['arms']['A'], result['arms']['B']
        _require(not a['plan_identity_errors'] and not b['plan_identity_errors'],
                 'executed Plan identity is incomplete or inconsistent; latency retained, strategy comparison inconclusive')
        _require(a['shared_lifecycle_pids'] == b['shared_lifecycle_pids'], 'A/B engines were reloaded or changed')
        _require(a['generations'] == b['generations'], 'A/B native generations differ')
        if len(gpus) == 8:
            _require(binding['model_id'] == 'Qwen2.5-7B-Instruct', 'this eight-GPU smoke is scoped to 7B')
            result['pd_golden'] = _pd_golden(root)
            _require(b['controller']['automatic_pd_requests'] > 0 and b['controller']['automatic_pd_plans'] > 0,
                     'eight-GPU run lacks real automatic P/D routing and a periodic P/D plan')
        if not result['interference']['passed']:
            result['requires_serial_retest'] = True
            result['affected_model'] = binding['model_id']
            result['errors'].append('parallel interference exceeds 5%; only this model needs serial remeasurement')
        else:
            triggered = b['controller']['actual_actions'] > 0 or b['controller']['automatic_pd_plans'] > 0
            result['classification'] = classify(a['metrics'], b['metrics'], triggered=triggered)
            result['status'] = ('passed' if a['metrics']['slo_passed'] and b['metrics']['slo_passed'] else 'failed')
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        for arm in result['arms'].values():
            arm.pop('_routes', None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args(argv)
    result = audit(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
