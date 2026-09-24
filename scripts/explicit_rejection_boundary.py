"""Host-only classification of fully recorded, explicit admission rejections.

No GPU imports, execution changes, raw outcome scans, or original-evidence writes.
The immutable original boundary remains unchanged. Callers must additionally bind
this sidecar's source_head to their exact active manifest (or verified ancestry).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'pdblend-explicit-admission-boundary/v1'
BOUNDARY_SCHEMA = 'single_observation_slo_boundary/v1'
CLASSIFICATION = 'observed_slo_failure_explicit_admission_rejections'
MAX_SMALL_BYTES = 4 * 1024 * 1024
CLASSIFIED_COLUMNS = (
    'classified_boundary_status', 'classified_boundary_role', 'classified_boundary_passed_lower',
    'classified_boundary_failed_upper', 'classified_boundary_policy_sha256',
    'classified_boundary_source_head_sha256', 'classified_boundary_sidecar_path',
    'classified_boundary_sidecar_sha256', 'classified_boundary_failure_reason',
    'classified_boundary_canonical_requests_sha256', 'classified_boundary_statistical_scope',
    'classified_boundary_energy_ranking_changed',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    require(path.stat().st_size <= MAX_SMALL_BYTES, 'artifact exceeds small-read limit')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_bound(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'invalid exact reference')
    require(Path(ref['path']).is_absolute(), 'reference path must be absolute')
    require(binding(ref['path']) == ref, 'artifact SHA mismatch')
    return json.loads(Path(ref['path']).read_text())


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def equal_number(left, right):
    return number(left) and number(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)


def artifact(receipt_ref, receipt, name, load):
    checksum = receipt.get('artifacts', {}).get(name)
    require(isinstance(checksum, str) and len(checksum) == 64, 'missing receipt artifact: ' + name)
    ref = dict(path=str(Path(receipt_ref['path']).parent / name), sha256=checksum)
    return ref, load(ref)


def explicit_rejection(error):
    if not isinstance(error, str):
        return False
    prefix, separator, payload = error.partition(':')
    if prefix.strip() != '503' or not separator:
        return False
    try:
        value = json.loads(payload)
    except (ValueError, TypeError):
        return False
    return value == {'error': 'no instance accepting requests'}


def validate_window(endpoint, policy_ref, policy, dataset, *, rejected, load):
    point, receipt = load(endpoint['point']), load(endpoint['receipt'])
    require(point.get('system') == 'pdblend', 'only explicit PD admission rejections are supported')
    for key in ('run_id', 'model_id', 'revision', 'seed', 'duration_s'):
        require(point.get(key) == policy.get(key) and type(point.get(key)) is not bool,
                'point differs from frozen policy: ' + key)
    require(point.get('dataset') == dataset and point.get('boundary_policy') == policy_ref,
            'point dataset/policy mismatch')
    require(number(point.get('scale')) and point['scale'] > 0 and point['scale'] == endpoint['scale'],
            'endpoint scale mismatch')
    require(receipt.get('point_sha256') == digest(point) and receipt.get('point') == point['name'],
            'receipt point digest/name mismatch')
    require(digest(point['engine_identity']) == policy['engine_signature'] == receipt.get('engine_signature'),
            'execution engine mismatch')
    template = load(policy['datasets'][dataset]['template_point'])
    for key in ('model_id', 'system', 'revision', 'seed', 'duration_s', 'slo', 'engine_identity',
                'source_manifest', 'topology', 'observation_scope', 'qualification_mode', 'result_policy'):
        require(point.get(key) == template.get(key), 'frozen template mismatch: ' + key)
    for key in ('system_config', 'profiles', 'source_manifest', 'qualifications'):
        require(point['inputs'].get(key) == template['inputs'].get(key), 'frozen input mismatch: ' + key)
    require(point['inputs']['trace'] == point['trace'], 'point trace binding mismatch')
    require(equal_number(point['rate_rps'], point['scale'] * policy['datasets'][dataset]['base_rate_rps']),
            'point rate mismatch')
    require(receipt.get('cleanup_passed') is True and not receipt.get('error')
            and receipt.get('recorded_window_complete') is True, 'execution/drain is incomplete')
    point_ref, recorded_point = artifact(endpoint['receipt'], receipt, 'point.json', load)
    require(recorded_point == point and point_ref == endpoint['point'], 'receipt artifact point mismatch')
    result_ref, result = artifact(endpoint['receipt'], receipt, 'result.json', load)
    require(result == receipt.get('result'), 'embedded result mismatch')
    drain_ref, drain = artifact(endpoint['receipt'], receipt, 'drain.json', load)
    require(drain.get('passed') is True, 'all-rank drain did not pass')
    request_ref, requests = artifact(endpoint['receipt'], receipt, 'run/comparison-requests.json', load)
    require(isinstance(requests, list) and requests, 'canonical requests missing')
    require(all(type(r.get('idx')) is int for r in requests)
            and sorted(r['idx'] for r in requests) == list(range(len(requests))), 'request IDs incomplete/duplicated')
    metrics = result['metrics']
    start = metrics.get('service_start_s', metrics.get('service_started_s'))
    end = metrics.get('service_end_s', metrics.get('service_finished_s'))
    require(number(start) and start > 0 and number(end) and equal_number(end - start, 150)
            and equal_number(metrics.get('duration_s'), 150), 'incomplete service clock')
    require(number(metrics.get('request_elapsed_s')) and metrics['request_elapsed_s'] > 0,
            'missing cohort termination timing')
    require(number(metrics.get('tail_end_s')) and metrics['tail_end_s'] >= end
            and equal_number(drain.get('tail_end_s'), metrics['tail_end_s']), 'drain/tail clock mismatch')
    for key in ('timeout_requests', 'unresolved_requests'):
        require(type(metrics.get(key)) is int and metrics[key] == 0, 'timeout/unresolved requests')
    require(metrics.get('token_timing_complete') is True, 'canonical token event accounting incomplete')
    slo = point['slo']
    require(all(number(slo.get(k)) and slo[k] > 0 for k in ('ttft_s', 'tpot_s')), 'invalid SLO limits')
    successes, failures, good = [], [], 0
    for row in requests:
        require(type(row.get('successful')) is bool and row.get('timeout') is False,
                'unknown request terminal classification')
        require(row.get('token_events_complete') is True, 'incomplete token events')
        if row['successful']:
            require(not row.get('error') and row.get('timing_complete') is True
                    and all(number(row.get(k)) and row[k] >= 0 for k in ('ttft_s', 'tpot_s')),
                    'successful request lacks valid timing')
            passes = row['ttft_s'] <= slo['ttft_s'] and row['tpot_s'] <= slo['tpot_s']
            require(row.get('joint_slo') is passes, 'joint SLO classification mismatch')
            successes.append(row)
            good += passes
        else:
            require(rejected and explicit_rejection(row.get('error')), 'failure is not an explicit supported rejection')
            require(row.get('timing_complete') is False and row.get('ttft_s') is None
                    and row.get('tpot_s') is None and row.get('completion_tokens') == 0
                    and row.get('native_reported_completion_tokens') in (None, 0)
                    and row.get('joint_slo') is False and row.get('window_good') is False
                    and row.get('pending_at_window_end') is False
                    and row.get('completed_in_window') is False and row.get('terminal_after_window') is False,
                    'rejection timing/token flags are inconsistent')
            failures.append(row)
    require(bool(failures) is rejected, 'endpoint does not match requested pass/rejection classification')
    counts = dict(offered_requests=len(requests), successful_requests=len(successes),
                  failed_requests=len(failures), joint_slo_requests=good,
                  invalid_timing_requests=len(failures), ttft_samples=len(successes), tpot_samples=len(successes))
    for key, count in counts.items():
        require(type(metrics.get(key)) is int and metrics[key] == count, 'canonical count mismatch: ' + key)
    require(equal_number(metrics.get('success_rate'), len(successes) / len(requests))
            and equal_number(metrics.get('joint_slo_rate'), good / len(requests)), 'rate denominator mismatch')
    for name in ('ttft', 'tpot'):
        values = sorted(row[name + '_s'] for row in successes)
        for percentile in (50, 90, 95, 99):
            value = values[math.ceil(len(values) * percentile / 100) - 1] if values else None
            actual = metrics.get(f'{name}_p{percentile}_s')
            require(actual is None if value is None else equal_number(actual, value), 'latency percentile mismatch')
    passes = (not failures and good / len(requests) >= .9 and all(
        number(metrics.get(k + '_p99_s')) and metrics[k + '_p99_s'] <= slo[k + '_s'] for k in ('ttft', 'tpot')))
    require(metrics.get('slo_pass') is passes and passes is not rejected, 'SLO verdict mismatch')
    return dict(endpoint=endpoint, canonical_requests=request_ref, result=result_ref, drain=drain_ref,
                point_sha256=digest(point), trace=point['trace'], source_manifest=point['source_manifest'],
                counts=counts, failed_request_indices=[r['idx'] for r in failures],
                failure_error_counts={'503: no instance accepting requests': len(failures)} if failures else {},
                service_start_s=start, service_end_s=end, tail_end_s=metrics['tail_end_s'],
                request_elapsed_s=metrics['request_elapsed_s'], slo=slo,
                slo_pass=passes, latency_population='successful_requests_only',
                terminal_evidence='hash-bound canonical explicit response errors; complete result and passed all-rank drain',
                per_rejection_terminal_timestamps='not_present_in_small_canonical; not invented')


def classification_evidence(source_head, dataset, *, load):
    manifest = load(source_head)
    require(manifest.get('schema') == BOUNDARY_SCHEMA and manifest.get('mode') == 'manifest',
            'source_head must be an immutable manifest, not a pointer')
    policy_ref = manifest['policy']; policy = load(policy_ref)
    require(policy.get('schema') == BOUNDARY_SCHEMA and policy.get('mode') == 'policy', 'unsupported policy')
    for key in ('run_id', 'model_id', 'revision', 'engine_signature'):
        require(manifest.get(key) == policy.get(key), 'manifest policy identity mismatch')
    require(policy.get('duration_s') == 150 and type(policy.get('duration_s')) is not bool
            and type(policy.get('seed')) is int and policy['seed'] == 701, 'unsupported measurement protocol')
    require(dataset in policy['datasets'], 'dataset outside policy')
    state = manifest['boundaries'][dataset]
    require(state.get('status') == 'incomplete_observation' and state.get('failed_upper') is None
            and state.get('next_rate_scale') is None, 'source boundary is not stopped on incomplete observation')
    observations = [row for row in manifest['observations'] if row['dataset'] == dataset]
    require(len({row['scale'] for row in observations}) == len(observations), 'duplicate observation scale')
    candidates = [row for row in observations if row.get('verdict') == 'incomplete']
    require(len(candidates) == 1, 'one exact incomplete candidate is required')
    upper = candidates[0]
    require(upper.get('reason') == 'incomplete_or_inconsistent_request_timing', 'execution obstruction cannot be reclassified')
    lower = dict(point=state['passed_lower_point'], receipt=state['passed_lower_receipt'], scale=state['passed_lower'])
    require(any(row.get('verdict') == 'pass' and all(row.get(k) == lower[k] for k in lower)
                for row in observations), 'lower endpoint absent from observation ledger')
    require(number(lower['scale']) and number(upper['scale']) and upper['scale'] > lower['scale']
            and equal_number(upper['scale'], lower['scale'] * policy['growth_factor']), 'upper is not next predeclared growth step')
    require(max(row['scale'] for row in observations) == upper['scale'], 'a later observation already exists')
    upper = {k: upper[k] for k in ('point', 'receipt', 'scale')}
    # Authorize both generated points by the existing decision chain, without opening corpus/trace raw data.
    for endpoint in (lower, upper):
        entries = [e for e in manifest['points'] if e.get('receipt') == endpoint['receipt']]
        require(len(entries) == 1, 'endpoint is absent/duplicated in generated points')
        entry = entries[0]; point = load(entry['point']); observed_point = load(endpoint['point'])
        require(point == observed_point and entry['decision'] in manifest['decisions'], 'generated point differs from observation')
        decision = load(entry['decision'])
        require(decision.get('policy') == policy_ref and decision.get('point') == entry['point']
                and decision.get('dataset') == dataset and decision.get('scale') == endpoint['scale'],
                'endpoint was not authorized by exact policy decision')
    summaries = {
        'lower': validate_window(lower, policy_ref, policy, dataset, rejected=False, load=load),
        'upper': validate_window(upper, policy_ref, policy, dataset, rejected=True, load=load),
    }
    return dict(policy=policy_ref, source_head=source_head, dataset=dataset, lower=lower, upper=upper,
                run_id=policy['run_id'], model_id=policy['model_id'], revision=policy['revision'],
                engine_signature=policy['engine_signature'], classification=CLASSIFICATION,
                summary=summaries, original_boundary=state,
                statistical_scope='single_observation_no_significance_or_exact_capacity',
                original_manifest_modified=False, canonical_metrics_modified=False,
                used_for_energy_ranking=False, repeat_measurement_authorized=False)


def validate_explicit_rejection_sidecar(ref, *, load_bound=None):
    load = load_bound or read_bound
    sidecar = load(ref)
    require(sidecar.get('schema') == SCHEMA and sidecar.get('mode') == 'analysis_classification', 'unsupported sidecar schema')
    expected = classification_evidence(sidecar['source_head'], sidecar['dataset'], load=load)
    require(sidecar == dict(schema=SCHEMA, mode='analysis_classification', **expected), 'sidecar evidence or scope mismatch')
    return dict(expected, sidecar_ref=ref)


def build_sidecar(source_head, dataset, out_path, *, load_bound=None):
    load = load_bound or read_bound
    value = dict(schema=SCHEMA, mode='analysis_classification',
                 **classification_evidence(source_head, dataset, load=load))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if out_path.exists():
        require(out_path.read_text() == encoded, 'immutable sidecar already exists with different content')
    else:
        with out_path.open('x') as stream:
            stream.write(encoded)
    ref = binding(out_path)
    validate_explicit_rejection_sidecar(ref, load_bound=load_bound)
    return ref


def _cell(value):
    return json.loads(value) if isinstance(value, str) and value else value


def annotate_classified_boundary_rows(rows, sidecar_refs, *, load_bound=None):
    """Append analysis-only columns; never change any existing measurement/rank cell.

    PD endpoints bind exact receipts. Baseline members additionally bind their
    immutable endpoint-selection document and its explicit sidecar authorization.
    Intermediate points, historical revisions, and unpaired attempts stay blank.
    """
    load = load_bound or read_bound
    result = [dict(row) for row in rows]
    annotations = [dict.fromkeys(CLASSIFIED_COLUMNS, '') for _ in rows]
    seen_scopes = set()
    for ref in sidecar_refs:
        classified = validate_explicit_rejection_sidecar(ref, load_bound=load)
        scope = (classified['policy']['sha256'], classified['dataset'])
        require(scope not in seen_scopes, 'duplicate classified boundary scope')
        seen_scopes.add(scope)
        for side, role in [('lower', 'passed_lower'), ('upper', 'failed_upper')]:
            endpoint = classified[side]
            point = load(endpoint['point'])
            matches = [i for i,row in enumerate(rows) if row.get('receipt_sha256') == endpoint['receipt']['sha256']]
            require(len(matches) == 1, 'classified PD endpoint receipt is missing/duplicated in publication')
            i = matches[0]; row = rows[i]
            require(row.get('system') == 'pdblend' and row.get('point_id') == point['name']
                    and row.get('point_sha256') == digest(point) and row.get('receipt_path') == endpoint['receipt']['path']
                    and row.get('run_id') == classified['run_id'] and row.get('revision') == classified['revision']
                    and row.get('model_id') == classified['model_id'] and row.get('dataset') == classified['dataset']
                    and row.get('trace_sha256') == point['trace']['sha256'], 'CSV endpoint identity differs')
            for index, candidate in enumerate(rows):
                if index == i:
                    pass
                elif candidate.get('system') in {'mixed', 'distserve', 'ecoserve', 'dynamollm'}:
                    pd_ref = _cell(candidate.get('pd_boundary_point'))
                    if not isinstance(pd_ref, dict) or pd_ref.get('sha256') != endpoint['point']['sha256']:
                        continue
                    require(load(pd_ref) == point, 'baseline endpoint point reference differs')
                    selection_ref = _cell(candidate.get('boundary_selection'))
                    require(isinstance(selection_ref, dict) and selection_ref, 'baseline endpoint lacks selection')
                    selection = load(selection_ref)
                    require(ref in selection.get('classification_sidecars', []), 'baseline selection lacks sidecar authorization')
                    require(candidate.get('model_id') == classified['model_id']
                            and candidate.get('dataset') == classified['dataset']
                            and candidate.get('trace_sha256') == point['trace']['sha256'], 'baseline endpoint trace/model differs')
                else:
                    continue
                update = dict(zip(CLASSIFIED_COLUMNS, (
                    'bracketed_single_observation', role, str(classified['lower']['scale']),
                    str(classified['upper']['scale']), classified['policy']['sha256'],
                    classified['source_head']['sha256'], ref['path'], ref['sha256'], CLASSIFICATION,
                    classified['summary']['upper']['canonical_requests']['sha256'],
                    classified['statistical_scope'], 'False',
                )))
                require(not any(annotations[index].values()) or annotations[index] == update,
                        'conflicting classified boundary annotation')
                annotations[index] = update
    for index,row in enumerate(result):
        for key,value in annotations[index].items():
            require(key not in row or row[key] == value, 'existing classification column differs: ' + key)
            row[key] = value
    return result


def annotate_classified_boundary_csv(path, sidecar_refs, *, load_bound=None):
    """Only annotate a caller-owned staging CSV, before its guarded publication."""
    path = Path(path).resolve()
    require(path.name != 'compare.csv', 'formal CSV must only be replaced by its sole writer')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    annotated = annotate_classified_boundary_rows(rows, sidecar_refs, load_bound=load_bound)
    fields += [key for key in CLASSIFIED_COLUMNS if key not in fields]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(annotated)
    return dict(rows=len(rows), classified_rows=sum(bool(r['classified_boundary_role']) for r in annotated),
                canonical_cells_changed=0, ranking_cells_changed=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-head', type=Path, required=True)
    parser.add_argument('--source-head-sha256', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_sidecar(dict(path=str(args.source_head.resolve()), sha256=args.source_head_sha256),
                                  args.dataset, args.out), sort_keys=True))
