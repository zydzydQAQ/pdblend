"""Prepare, group and export the fixed 180-point native comparison.

No historical metric is promoted to a new measurement. Missing qualifications
remain explicit rows in the single plotting CSV.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
import math
import time
from pathlib import Path

from .campaign import DATASETS, SYSTEMS
from .client import load_split, poisson_trace
from .first_batch import deployment
from .resident_session import digest, engine_signature, file_sha as _hash_file_bytes, write_new

MODELS = tuple('Qwen2.5-' + s + '-Instruct' for s in ('7B', '14B', '32B'))
SCALES = (.5, .25, .75, 1.)
SYSTEM_ORDER = ('mixed', 'distserve', 'ecoserve', 'dynamollm', 'pdblend')
PROTOCOL = 'native-comparison-150s/v1'
_WATCH_DIGEST_CACHE = None


def file_sha(path):
    if _WATCH_DIGEST_CACHE is None:
        return _hash_file_bytes(path)
    return _WATCH_DIGEST_CACHE.read(path, _hash_file_bytes)


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def load_bound(ref):
    if file_sha(ref['path']) != ref['sha256']:
        raise ValueError('binding checksum differs: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text())


def point_order(point):
    return (MODELS.index(point['model_id']), SCALES.index(point['scale']),
            SYSTEM_ORDER.index(point['system']), list(DATASETS).index(point['dataset']))


def group_points(points):
    groups = {}
    for point in sorted(points, key=point_order):
        identity = point.get('engine_identity')
        if not identity or point.get('blockers'):
            continue
        signature = engine_signature(identity)
        # Cross-system reuse requires an explicit qualified reset artifact.
        cross = point.get('cross_system_reset')
        reuse_scope = point['system']
        if cross:
            receipt = load_bound(cross)
            if (receipt.get('passed') is not True or receipt.get('engine_signature') != signature
                    or point['system'] not in receipt.get('systems', [])):
                raise ValueError('cross-system reset evidence differs')
            reuse_scope = cross['sha256']
        key = digest(dict(engine=signature, reset_scope=reuse_scope))
        group = groups.setdefault(key, dict(session_id='resident-' + key[:20],
            model_id=point['model_id'], engine_signature=signature,
            engine_identity=identity, points=[], gpu_count=8, exclusive=True, reserve_host=True))
        group['points'].append(point)
    return sorted(groups.values(), key=lambda g: (
        MODELS.index(g['model_id']), min(SYSTEM_ORDER.index(p['system']) for p in g['points']),
        g['engine_signature']))


def prepare(first_spec, out, corpus_root, *, prepared_inputs=None):
    first_spec, out, corpus_root = Path(first_spec), Path(out), Path(corpus_root)
    original = json.loads(first_spec.read_text())
    prepared_inputs = prepared_inputs or {}
    parents = {(p['model_id'], p['dataset']): p for p in original['points'] if p.get('trace')}
    points, traces = [], []
    for model in MODELS:
        size = model.split('-')[1].lower()
        corpus = corpus_root / f'2026-09-22-{size}-v1'
        manifest = json.loads((corpus / 'manifest.json').read_text())
        if manifest.get('model_name') != model or manifest.get('complete') is not True:
            raise ValueError('model-owned prepared corpus incomplete')
        for dataset, slo in DATASETS.items():
            dataset_path = corpus / (dataset + '.json')
            corpus_sha = file_sha(dataset_path)
            if corpus_sha != manifest['dataset_sha256'][dataset]:
                raise ValueError('corpus bytes changed')
            parent = parents.get((model, dataset))
            for scale in SCALES:
                ref, rate = None, None
                if parent:
                    frozen = load_bound(parent['trace'])
                    if (frozen['model_id'] != model or frozen['dataset'] != dataset
                            or frozen['slo'] != slo or frozen['seed'] != 701
                            or frozen['corpus_sha256'] != corpus_sha):
                        raise ValueError('frozen workload/corpus identity differs')
                    rate = parent['rate_rps'] * scale / parent['scale']
                    requests = ([r for r in frozen['requests'] if r['arrival_s'] < 150.]
                                if scale == .5 else [asdict(r) for r in poisson_trace(
                                    load_split(corpus, dataset, 'evaluation'), rate, 150., 701, dataset)])
                    trace = dict(frozen, duration_s=150., rate_rps=rate, requests=requests,
                                 parent_trace=parent['trace'], measurement_protocol_version=PROTOCOL)
                    trace_path = out / 'traces' / f'{size}-{dataset}-x{scale:g}-seed701.json'
                    write_new(trace_path, trace)
                    ref = dict(binding(trace_path), requests=len(requests))
                    traces.append(ref)
                for system in SYSTEMS:
                    name = f'{size}-{system}-{dataset}-x{scale:g}-seed701'
                    item = prepared_inputs.get(name, {})
                    blockers = ([] if ref else ['missing_same_model_independent_tuning_anchor'])
                    if not item:
                        blockers += ['missing_bound_system_configuration', 'missing_native_qualification']
                        if system not in ('mixed',):
                            blockers.append('missing_accepted_independent_profile_and_coverage')
                        if system in ('distserve', 'pdblend'):
                            blockers.append('missing_tuning_only_offline_deployment_choice')
                        if system == 'dynamollm':
                            blockers.append('missing_original_period_weight_retention_qualification')
                    else:
                        blockers.extend(item.get('blockers', []))
                    point = dict(name=name, model_id=model, dataset=dataset, system=system,
                        scale=scale, rate_rps=rate, seed=701, duration_s=150., slo=slo,
                        single_seed=True, trace=ref, topology=deployment(system, model),
                        revision=item.get('revision', 'initial'), runner='independent_native_dispatch',
                        measurement_protocol_version=PROTOCOL, **{k: v for k, v in item.items()
                            if k not in ('blockers', 'revision')})
                    point.update(blockers=sorted(set(blockers)), formal_eligible=False,
                                 status='blocked' if blockers else 'prepared')
                    points.append(point)
    groups = group_points(points)
    campaign = dict(schema='resident-comparison-campaign/v1', campaign_id=out.name,
                    parent_spec=binding(first_spec), measurement_protocol_version=PROTOCOL,
                    duration_s=150., seed=701, scales=SCALES, points=sorted(points, key=point_order),
                    groups=groups, traces=traces,
                    summary=dict(points=len(points), trace_sets=len(traces),
                                 prepared_points=sum(not p['blockers'] for p in points),
                                 resident_sessions=len(groups), pure_service_s=150 * len(points)))
    if len(points) != 180 or len({p['name'] for p in points}) != 180:
        raise AssertionError('comparison must contain exactly 180 unique points')
    write_new(out / 'campaign.json', campaign)
    return campaign


BASE_FIELDS = ['campaign_id', 'point_id', 'model_id', 'dataset', 'system', 'rate_scale',
               'offered_rps', 'seed', 'duration_s', 'revision', 'attempt_id', 'status',
               'failure_reason', 'formal_eligible', 'evidence_valid', 'baseline_frozen',
               'slo_pass', 'rank_eligible', 'energy_rank', 'best_feasible_baseline',
               'pdblend_saving_vs_best_feasible_baseline', 'comparison_status',
               'session_id', 'window_index', 'resident_reused', 'engine_signature',
               'trace_sha256', 'measurement_protocol_version', 'slo_ttft_s', 'slo_tpot_s',
               'receipt_path', 'receipt_sha256']
METRIC_FIELDS = ['offered_requests', 'successful_requests', 'failed_requests', 'output_tokens',
                 'good_output_tokens', 'joint_slo_requests', 'success_rate', 'joint_slo_rate',
                 'goodput_request_s', 'goodput_token_s', 'cohort_goodput_request_s',
                 'cohort_goodput_token_s', 'throughput_request_s', 'throughput_token_s',
                 'energy_service_j', 'service_mean_power_w', 'energy_tail_j', 'tail_s',
                 'gpu_util_mean_pct', 'gpu_util_coverage_fraction', 'util_max_gap_s']
METRIC_FIELDS += [f'{m}_{p}_s' for m in ('ttft', 'tpot') for p in ('p50', 'p90', 'p95', 'p99', 'mean', 'max')]
METRIC_FIELDS += [f'{m}_samples' for m in ('ttft', 'tpot')]
METRIC_FIELDS += [f'gpu{i}_{m}' for i in range(8) for m in ('uuid', 'util_mean_pct', 'util_peak_pct')]


def rank_rows(rows):
    """Preserve strict qualification and add a separate as-executed view."""
    from .comparison_clock_evidence import qualified
    from .comparison_ranking import rank_rows_core
    from .comparison_observed import add_observed_comparisons
    baseline_systems = tuple(s for s in SYSTEMS if s != 'pdblend')
    rank_rows_core(rows, baseline_systems=baseline_systems, clock_predicate=qualified)
    return add_observed_comparisons(rows, baseline_systems=baseline_systems)


def failed_session_points(path, report):
    """Bind a startup failure to point revisions, not just a reusable engine ID."""
    if 'planned_points' in report:
        points = report['planned_points']
        if not isinstance(points, dict) or not all(isinstance(k, str) and isinstance(v, str)
                and len(v) == 64 for k, v in points.items()):
            raise ValueError('failed session planned point identities are invalid')
        return points
    # Immutable older sessions did not include planned_points. Recover only
    # from the actual lease invocation and the content-addressed job identity.
    manifest = path.parent/'lease-manifest.json'
    if not manifest.is_file():
        return {}
    lease = json.loads(manifest.read_text())
    payload = lease.get('payload', {}); argv = payload.get('argv', [])
    if argv.count('--group') != 1:
        return {}
    group_path = Path(argv[argv.index('--group')+1])
    group = json.loads(group_path.read_text())
    if (lease.get('immutable') is not True
            or lease.get('job_id') != 'comparison-'+group['model_id'].split('-')[1].lower()+'-'+digest(group)[:16]
            or group.get('session_id') != report['session_id']
            or group.get('engine_signature') != report['engine_signature']
            or payload.get('session_id') != report['session_id']):
        raise ValueError('failed session original group/lease identity differs: '+str(path))
    return {p['name']:digest(p) for p in group['points']}


def export(campaign_path, output, *, session_roots=(), qualification_evidence_ref=None, analysis_policy=None):
    from .comparison_clock_evidence import annotate as clock_annotation
    from .comparison_recorded import POLICY as recorded_policy
    if analysis_policy not in (None, recorded_policy):
        raise ValueError('unknown comparison analysis policy: ' + str(analysis_policy))
    campaign_path = Path(campaign_path)
    campaign = json.loads(campaign_path.read_text())
    receipts, seen, failed_sessions = {}, set(), {}
    for root in session_roots:
        for path in sorted(Path(root).glob('**/completion.json')):
            report = json.loads(path.read_text())
            if report.get('schema') == 'resident-group-session/v1' and not report.get('complete'):
                failed_sessions.setdefault(report['session_id'], []).append(
                    (path.resolve(), report, failed_session_points(path, report)))
        for path in sorted(Path(root).glob('**/windows/*/receipt.json')):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            row = json.loads(path.read_text())
            receipts.setdefault(row['point'], []).append((path, row))

    gpu_util_fields = ('coverage_fraction', 'status', 'samples', 'missing_samples',
                       'max_gap_s', 'max_missing_gap_s')

    def base_row(p):
        return dict(campaign_id=campaign['campaign_id'], point_id=p['name'], point_sha256=digest(p), model_id=p['model_id'],
            dataset=p['dataset'], system=p['system'], rate_scale=p['scale'], offered_rps=p['rate_rps'],
            seed=p['seed'], duration_s=p['duration_s'], revision=p['revision'], status=p['status'],
            failure_reason=';'.join(p['blockers']), formal_eligible=False, evidence_valid=False,
            baseline_frozen=False, slo_pass='', trace_sha256=(p.get('trace') or {}).get('sha256', ''),
            measurement_protocol_version=p.get('measurement_protocol_version', PROTOCOL),
            metering_execution=p.get('metering_execution', 'in_process'),
            slo_ttft_s=p['slo']['ttft_s'], slo_tpot_s=p['slo']['tpot_s'],
            session_cost_scope='session_attempt_shared_do_not_sum_rows', session_cost_scope_id='',
            session_cost_evidence_status='unavailable', session_completion_path='', session_completion_sha256='',
            session_complete=None, session_status='', session_engine_loads=None, session_engine_load_cycles=None,
            session_initial_engine_loads=None, session_engine_loads_source='',
            session_engine_load_s=None, session_load_lock_wait_s=None, session_total_s=None,
            session_cleanup_s=None, session_windows_attempted=None, session_windows_measured=None,
            session_windows_skipped=None, window_reset_s=None, window_warmup_s=None,
            window_warmup_included_in_reset=True,
            pdblend_window_engine_loads=None, pdblend_window_cumulative_engine_loads=None,
            pdblend_reset_engine_loads=None, pdblend_reset_cumulative_engine_loads=None,
            **clock_annotation(p, {}),
            **{f'gpu{i}_util_{field}': None for i in range(8) for field in gpu_util_fields})

    def gpu_utilization(path, receipt, row):
        # The receipt artifact loop has already checked these exact bytes.
        # Older windows without this artifact retain unknown coverage/counts.
        name = 'run/comparison-metering.json'
        if name not in receipt['artifacts']:
            return {}
        meter = json.loads((path.parent/name).read_text())
        uuids = meter.get('gpu_uuids')
        if (not isinstance(uuids, list) or len(uuids) != 8 or len(set(uuids)) != 8
                or any(not isinstance(u, str) or not u.startswith('GPU-') for u in uuids)):
            raise ValueError('per-GPU utilization lacks physical UUID order: ' + str(path))
        if row.get('gpu_uuids') is not None and row['gpu_uuids'] != uuids:
            raise ValueError('per-GPU utilization UUID order differs from window: ' + str(path))
        per_gpu = meter.get('service', {}).get('utilization', {}).get('per_gpu', {})
        if not isinstance(per_gpu, dict):
            raise ValueError('per-GPU utilization must be a UUID mapping: ' + str(path))
        values = {}
        for index, uuid in enumerate(uuids):
            recorded_uuid = row.get(f'gpu{index}_uuid')
            if recorded_uuid not in (None, '', uuid):
                raise ValueError('per-GPU utilization UUID differs from plotted GPU index: ' + str(path))
            value = per_gpu.get(uuid, {})
            if not isinstance(value, dict):
                raise ValueError('per-GPU utilization row is not an object: ' + str(path))
            for field in gpu_util_fields:
                item = value.get(field)
                if item is not None:
                    if field == 'status':
                        valid = item in ('complete', 'missing')
                    elif field in ('samples', 'missing_samples'):
                        valid = type(item) is int and item >= 0
                    else:
                        valid = type(item) in (int, float) and math.isfinite(item) and item >= 0
                        if field == 'coverage_fraction':
                            valid = valid and item <= 1
                    if not valid:
                        raise ValueError('invalid per-GPU utilization ' + field + ': ' + str(path))
                values[f'gpu{index}_util_{field}'] = item
        return values

    completion_cache = {}

    def nonnegative(value, field, *, integer=False):
        if value is None:
            return None
        if (type(value) not in ((int,) if integer else (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError('invalid session/window cost: ' + field)
        return value

    def costs(path, receipt):
        root = path.parent.parent.parent
        completion = root/'completion.json'
        value = dict(session_cost_scope_id=str(root), session_cost_evidence_status='pending_completion')
        if 'reset.json' in receipt['artifacts']:
            reset = json.loads((path.parent/'reset.json').read_text())
            value.update(window_reset_s=nonnegative(reset.get('reset_s'), 'reset_s'),
                         window_warmup_s=nonnegative(reset.get('warmup_s'), 'warmup_s'))
            inventory = reset.get('pdblend_inventory_reset')
            if inventory is not None:
                added = nonnegative(inventory.get('engine_loads'), 'pdblend_reset.engine_loads', integer=True)
                cumulative = nonnegative(inventory.get('cumulative_engine_loads'),
                                         'pdblend_reset.cumulative_engine_loads', integer=True)
                before = nonnegative(inventory.get('before', {}).get('engine_load_accounting', {}).get('engine_loads'),
                                     'pdblend_reset.before.engine_loads', integer=True)
                if (added is not None and cumulative is not None
                        and (added > cumulative or before is not None and before+added != cumulative)):
                    raise ValueError('PDblend reset engine-load accounting differs: ' + str(path))
                value.update(pdblend_reset_engine_loads=added, pdblend_reset_cumulative_engine_loads=cumulative)
        if 'drain.json' in receipt['artifacts']:
            drain = json.loads((path.parent/'drain.json').read_text())
            if 'window_engine_loads' in drain or 'engine_load_accounting' in drain:
                added = nonnegative(drain.get('window_engine_loads'), 'pdblend_window.engine_loads', integer=True)
                cumulative = nonnegative(drain.get('engine_load_accounting', {}).get('engine_loads'),
                                         'pdblend_window.cumulative_engine_loads', integer=True)
                before = value.get('pdblend_reset_cumulative_engine_loads')
                if (added is not None and cumulative is not None
                        and (added > cumulative or before is not None and before+added != cumulative)):
                    raise ValueError('PDblend window engine-load accounting differs: ' + str(path))
                value.update(pdblend_window_engine_loads=added, pdblend_window_cumulative_engine_loads=cumulative)
        if root not in completion_cache:
            completion_cache[root] = (json.loads(completion.read_text()), file_sha(completion)) if completion.is_file() else None
        item = completion_cache[root]
        if item is None:
            return value
        report, checksum = item
        if (receipt.get('session_id') is not None and report.get('session_id') != receipt['session_id']
                or receipt.get('engine_signature') is not None
                and report.get('engine_signature') != receipt['engine_signature']):
            raise ValueError('session completion identity differs from window: ' + str(path))
        windows = report.get('windows', [])
        if not any(row.get('point') == receipt['point'] and row.get('sha256') == file_sha(path) for row in windows):
            raise ValueError('session completion does not bind window receipt: ' + str(path))
        start, end = report.get('started_s'), report.get('finished_s')
        elapsed = None
        if start is not None and end is not None:
            nonnegative(start, 'session.started_s'); nonnegative(end, 'session.finished_s')
            elapsed = nonnegative(end-start, 'session_total_s')
        startup = report.get('startup', {})
        cleanup = report.get('cleanup', {})
        initial_loads = nonnegative(startup.get('engine_loads'), 'startup.engine_loads', integer=True)
        cumulative_counts = [value[k] for k in ('pdblend_reset_cumulative_engine_loads',
                            'pdblend_window_cumulative_engine_loads') if value.get(k) is not None]
        if 'engine_loads' in cleanup:
            actual_loads = nonnegative(cleanup['engine_loads'], 'cleanup.engine_loads', integer=True)
            if actual_loads is None or any(actual_loads < n for n in [initial_loads, *cumulative_counts] if n is not None):
                raise ValueError('session cleanup engine-load count regressed: ' + str(path))
            load_source = 'cleanup_cumulative'
        elif cumulative_counts:
            actual_loads, load_source = None, 'missing_cleanup_cumulative'
        else:
            actual_loads, load_source = initial_loads, 'startup_only'
        value.update(session_cost_evidence_status='bound_completion', session_completion_path=str(completion),
            session_completion_sha256=checksum, session_complete=report.get('complete'), session_status=report.get('status', ''),
            session_engine_loads=actual_loads, session_initial_engine_loads=initial_loads,
            session_engine_loads_source=load_source,
            session_engine_load_cycles=nonnegative(startup.get('engine_load_cycles'), 'engine_load_cycles', integer=True),
            session_engine_load_s=nonnegative(startup.get('engine_load_s'), 'engine_load_s'),
            session_load_lock_wait_s=nonnegative(startup.get('load_lock_wait_s'), 'load_lock_wait_s'),
            session_total_s=elapsed, session_cleanup_s=nonnegative(report.get('cleanup_s'), 'cleanup_s'),
            session_windows_attempted=len(windows), session_windows_measured=sum(r.get('evidence_valid') is True for r in windows),
            session_windows_skipped=len(report.get('skipped', [])))
        return value

    def unqualified_observation(path, receipt, point, result):
        scope = 'pdblend_profile_unqualified_evaluation/v1'
        if (result.get('observation_scope') != scope or result.get('measurement_evidence_valid') is not True
                or receipt.get('cleanup_passed') is not True):
            return None
        def need(value, reason):
            if not value:
                raise ValueError('PD observation receipt differs: ' + reason + ': ' + str(path))
        need(point.get('system') == 'pdblend' and point.get('observation_scope') == scope
             and point.get('duration_s') == 150.
             and result.get('metrics', {}).get('duration_s') == 150., 'system or measured duration')
        need(receipt.get('cleanup_passed') is True and receipt.get('evidence_valid') is False
             and receipt.get('baseline_frozen') is False and result.get('evidence_valid') is False
             and result.get('formal_eligible') is False and result.get('profile_qualified') is False,
             'unqualified observation must not claim formal or baseline qualification')
        relative = 'run/observation-acceptance.json'
        artifacts = receipt['artifacts']
        need(relative in artifacts, 'hash-bound observation acceptance is missing')
        audit_path = (path.parent / relative).resolve()
        audit = load_bound(dict(path=str(audit_path), sha256=artifacts[relative]))
        need(audit == result.get('observation_acceptance'), 'embedded observation acceptance')
        need(audit.get('schema') == 'pdblend-observation-acceptance/v1' and audit.get('scope') == scope
             and audit.get('point_sha256') == digest(point)
             and audit.get('metrics_sha256') == digest(result['metrics']), 'schema, point or metric binding')
        need(audit.get('measurement_evidence_valid') is True and audit.get('profile_qualified') is False
             and audit.get('formal_eligible') is False and audit.get('evidence_valid') is False
             and audit.get('missing_gates') == [] and audit.get('gate_failures') == {}, 'measurement verdict')
        gaps = audit.get('profile_missing_gates')
        need(isinstance(gaps, list) and gaps and all(isinstance(v, str) and v for v in gaps),
             'explicit profile qualification gaps are missing')
        raw_names = ('trace', 'outcomes', 'power', 'native_result', 'canonical_requests', 'metering',
                     'startup_qualification', 'reset', 'drain', 'controller', 'routes', 'native_cleanup',
                     'transition_measurements', 'frequencies')
        required = {'raw.' + name for name in raw_names} | {
            'binding.' + name for name in ('trace', 'native_result', 'startup_qualification', 'reset', 'metering', 'drain')
        } | {'pdblend.observation_inputs', 'pdblend.inventory', 'pdblend.full_physical_inventory',
             'pdblend.actual_window', 'pdblend.startup', 'pdblend.reset', 'pdblend.inventory_restoration',
             'pdblend.controller_actions', 'pdblend.physical_clocks', 'pdblend.request_routes',
             'pdblend.published_route_roles', 'pdblend.native_release_and_off', 'pdblend.canonical_metrics',
             'metering.raw_eight_gpu_window'}
        if point.get('metering_execution') == 'isolated_process':
            required.add('metering.isolated_process_method')
        checked = audit.get('checked_gates')
        need(isinstance(checked, list) and len(checked) == len(set(checked)) and required <= set(checked),
             'required measurement gates were not checked')
        refs = audit.get('raw_refs')
        need(isinstance(refs, dict) and set(raw_names) <= set(refs)
             and digest(refs) == audit.get('evidence_sha256'), 'raw evidence digest')
        names = {'native_result': 'native-result.json', 'canonical_requests': 'comparison-requests.json',
                 'metering': 'comparison-metering.json', 'drain': 'native-drain.json',
                 'controller': 'controller.jsonl', 'routes': 'routes.jsonl', 'native_cleanup': 'native-cleanup.json',
                 'transition_measurements': 'transition-measurements.json', 'frequencies': 'freq.jsonl',
                 'power': 'power.json'}
        for name, reference in refs.items():
            need(isinstance(reference, dict) and isinstance(reference.get('path'), str)
                 and Path(reference['path']).is_absolute(), 'raw reference is not absolute: ' + name)
            target = Path(reference['path']).resolve()
            if name in ('trace', 'startup_qualification'):
                need(file_sha(target) == reference.get('sha256'), 'raw evidence checksum: ' + name)
            # Window-local bytes were already verified by the artifact loop.
            # Reconstruct their exact reference without a second large raw read.
            if name == 'trace':
                need(reference == point.get('trace'), 'evaluation trace differs')
            elif name == 'startup_qualification':
                need(reference == result.get('qualification'), 'startup qualification differs')
            else:
                need(target.is_relative_to(path.parent.resolve()), 'raw artifact belongs to another window: ' + name)
                local = str(target.relative_to(path.parent.resolve()))
                need(artifacts.get(local) == reference['sha256'], 'raw artifact is not bound by window: ' + name)
                if name in names:
                    need(local == 'run/' + names[name], 'raw artifact name differs: ' + name)
                elif name == 'reset':
                    need(local == 'reset.json', 'reset artifact differs')
                elif name == 'outcomes':
                    need(local in ('run/outcomes.json', 'run/outcomes.jsonl', 'run/outcomes.jsonl.gz'),
                         'outcome artifact differs')
        return dict(observation_scope=scope, measurement_evidence_valid=True, profile_qualified=False,
                    profile_missing_gates=gaps, observation_acceptance_path=str(audit_path),
                    observation_acceptance_sha256=artifacts[relative])

    rows, recorded_results = [], {}
    for current in campaign['points']:
        matched_current = False
        for path, receipt in receipts.get(current['name'], []):
            artifacts = receipt.get('artifacts')
            if not isinstance(artifacts, dict) or not artifacts:
                raise ValueError('window receipt lacks bound artifacts: ' + str(path))
            for name, expected in artifacts.items():
                target = path.parent / name
                if not target.resolve().is_relative_to(path.parent.resolve()) or file_sha(target) != expected:
                    raise ValueError('window artifact changed: ' + str(target))
            result = receipt.get('result')
            if result is not None:
                if 'result.json' not in artifacts or json.loads((path.parent/'result.json').read_text()) != result:
                    raise ValueError('embedded window result differs from bound result.json: ' + str(path))
            elif receipt.get('evidence_valid') is True:
                raise ValueError('valid window receipt lacks its result: ' + str(path))
            else:
                result = json.loads((path.parent/'result.json').read_text()) if 'result.json' in artifacts else {}
            if not isinstance(result, dict):
                raise ValueError('window result is not an object: ' + str(path))
            point = None
            if 'point.json' in artifacts:
                point = json.loads((path.parent/'point.json').read_text())
                if digest(point) != receipt.get('point_sha256') or point.get('name') != current['name']:
                    raise ValueError('bound historical point differs from receipt: ' + str(path))
            if receipt.get('point_sha256') == digest(current):
                point = current
                matched_current = True
            if point is None:
                # SHA alone cannot recover revision or trace. Preserve the
                # existence of old evidence without borrowing today's identity.
                row = base_row(current)
                row.update(point_sha256=receipt.get('point_sha256', ''), revision='', trace_sha256='',
                    status='unresolved_historical_receipt', failure_reason='historical_point_spec_unbound',
                    receipt_path=str(path), receipt_sha256=file_sha(path), attempt_id=str(path.parent.parent.parent))
                row.update(costs(path, receipt))
                rows.append(row)
                continue
            if point.get('engine_identity') and receipt.get('engine_signature') != engine_signature(point['engine_identity']):
                raise ValueError('window engine identity differs from frozen point: ' + str(path))
            row = base_row(point)
            protected = ('campaign_id', 'point_id', 'point_sha256', 'rate_scale',
                         'model_id', 'dataset', 'system', 'revision', 'trace_sha256', 'duration_s',
                         'offered_rps', 'seed', 'measurement_protocol_version', 'metering_execution',
                         'slo_ttft_s', 'slo_tpot_s')
            for section in ('identity', 'metrics'):
                values = result.get(section, {})
                if any(key in values and values[key] != row[key] for key in protected):
                    raise ValueError('window result overrides frozen point identity: ' + str(path))
                row.update(values)
            row.update({k: receipt.get(k, '') for k in ('session_id', 'window_index', 'resident_reused',
                                                      'engine_signature')})
            evidence_valid = (receipt.get('evidence_valid') is True and result.get('evidence_valid') is True
                              and receipt.get('cleanup_passed') is True)
            row.update(status='measured' if evidence_valid else 'invalid_measurement', evidence_valid=evidence_valid,
                baseline_frozen=point['system'] != 'pdblend' and evidence_valid and receipt.get('baseline_frozen') is True,
                formal_eligible=result.get('formal_eligible') is True and evidence_valid,
                failure_reason=receipt.get('error', ''), receipt_path=str(path.resolve()),
                receipt_sha256=file_sha(path), attempt_id=str(path.parent.parent.parent))
            # The opt-in interpretation retains strict failures as diagnostics;
            # receipt, result, point and artifact byte bindings above still apply.
            observation = (unqualified_observation(path, receipt, point, result)
                           if analysis_policy is None else None)
            if observation is not None:
                row.update(observation, status='measured_unqualified',
                           failure_reason='profile_unqualified: ' + ';'.join(observation['profile_missing_gates']))
            row.update(costs(path, receipt))
            row.update(gpu_utilization(path, receipt, row))
            row.update(clock_annotation(point, result, receipt=receipt, receipt_path=path))
            recorded_results[str(path)] = (result, receipt)
            rows.append(row)
        if not matched_current:
            row = base_row(current)
            for group in campaign.get('groups', []):
                if not any(p['name'] == current['name'] for p in group['points']):
                    continue
                historical = failed_sessions.get(group['session_id'], [])
                failures = [(path,report) for path,report,points in historical
                            if points.get(current['name']) == digest(current)]
                if historical:
                    row['prior_session_failures'] = [dict(path=str(path),sha256=file_sha(path),
                        point_sha256=points.get(current['name']),error=report.get('error'),
                        matches_current_revision=points.get(current['name'])==digest(current))
                        for path,report,points in historical]
                if failures and not current['blockers']:
                    path, report = failures[-1]
                    if report.get('engine_signature') != group['engine_signature']:
                        raise ValueError('failed session identity differs: ' + str(path))
                    row.update(status='blocked', failure_reason='resident_session_failed: '+report.get('error', 'cleanup failure'),
                               session_id=group['session_id'], session_completion_path=str(path),
                               session_completion_sha256=file_sha(path))
            rows.append(row)
    rank_rows(rows)
    if analysis_policy == recorded_policy:
        from .comparison_recorded import analyze
        analyze(rows, recorded_results)
    if qualification_evidence_ref is not None:
        from .comparison_qualification_evidence import load_qualification_evidence,annotate
        qualification=load_qualification_evidence(qualification_evidence_ref,load_bound=load_bound,file_sha=file_sha)
        rows=annotate(rows,qualification)
    fields = BASE_FIELDS + METRIC_FIELDS
    fields += sorted(set().union(*(r.keys() for r in rows)) - set(fields))
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
                          for k, v in r.items()} for r in rows)
    os.replace(tmp, path)
    summary = dict(rows=len(rows), measured=sum(r['evidence_valid'] is True for r in rows),
                   rank_eligible=sum(r['rank_eligible'] is True for r in rows))
    if analysis_policy == recorded_policy:
        summary.update(analysis_policy=analysis_policy, qualification_valid=summary['measured'],
                       measured=sum(r['measurement_usable'] is True for r in rows))
    unqualified_count = sum(r['status'] == 'measured_unqualified' for r in rows)
    if unqualified_count:
        summary['measured_unqualified'] = unqualified_count
    return summary


def _campaign_ancestry(path):
    """Retain ancestor execution jobs after adopting a runtime overlay."""
    watched, visited = set(), set()
    while path is not None:
        path = Path(path).resolve()
        if path in visited:
            raise ValueError('cyclic comparison campaign ancestry')
        visited.add(path); watched.add(str(path))
        campaign = json.loads(path.read_text())
        watched.update(str(Path(p).resolve()) for p in campaign.get('execution_campaigns', []))
        ref = campaign.get('parent_campaign')
        if ref:
            load_bound(ref)
        path = ref['path'] if ref else None
    return watched


def _attempt_output(attempt, relative):
    path = Path(relative)
    target = (attempt/path).resolve()
    if path.is_absolute() or '..' in path.parts or not target.is_relative_to(attempt):
        raise ValueError('comparison output escapes queue attempt')
    return target


def _frozen_watch_overlay(job, lease, current):
    """Validate the producer's completed freeze, without waiting for its run.

    A campaign written before freeze is intentionally never read. Completion
    may describe a later measurement failure; that does not unfreeze evidence.
    """
    payload = job['payload']; attempt = Path(lease['attempt_dir']).resolve()
    path = _attempt_output(attempt, payload['overlay_campaign_output'])
    freeze_path = path.parent/'freeze.json'
    if not freeze_path.exists():
        return None
    try:
        freeze = json.loads(freeze_path.read_text())
    except json.JSONDecodeError:
        if job['status'] in ('queued', 'running'):
            return None  # write_new uses exclusive creation, not rename.
        raise
    manifest = json.loads((attempt/'manifest.json').read_text())
    if (manifest.get('immutable') is not True or manifest.get('job_id') != job['job_id']
            or manifest.get('lease_id') != lease['lease_id']
            or any(manifest.get('payload', {}).get(k) != payload.get(k) for k in (
                'comparison_campaign', 'combined_plan', 'overlay_campaign_output', 'session_output',
                'source_sha256', 'source_snapshot'))):
        raise ValueError('overlay queue attempt does not bind declared payload')
    plan_ref = payload['combined_plan']; plan = load_bound(plan_ref)
    parent_ref = binding(current)
    if (plan.get('parent_campaign') != parent_ref or freeze.get('parent_campaign') != parent_ref
            or str(Path(payload['comparison_campaign']).resolve()) != str(current)
            or freeze.get('schema') != plan.get('schema')
            or freeze.get('evaluation_has_started') is not False):
        raise ValueError('overlay freeze differs from queue plan/parent boundary')
    source = load_bound(plan['source_manifest'])
    if (source.get('source_sha256') != payload.get('source_sha256')
            or Path(plan['source_manifest']['path']).resolve()
                != Path(payload['source_snapshot']).resolve()/'manifest.json'):
        raise ValueError('overlay source differs from queue plan')
    if (Path(freeze['campaign']['path']).resolve() != path
            or Path(freeze['group']['path']).resolve() != path.parent/'group.json'
            or Path(freeze['anchor']['path']).resolve() != path.parent/'anchor/completion.json'):
        raise ValueError('overlay freeze artifact paths differ from declared output')
    campaign, group, anchor = (load_bound(freeze[k]) for k in ('campaign', 'group', 'anchor'))
    outcome = freeze.get('anchor_outcome')
    if (campaign.get('combined_plan') != plan_ref or campaign.get('parent_campaign') != parent_ref
            or campaign.get('execution_source_manifest') != plan.get('source_manifest')
            or campaign.get('anchor_outcome') != outcome or outcome not in ('confirmed', 'slo_exhausted')
            or anchor.get('hardware_executed') is not True or anchor.get('cleanup_errors')
            or (outcome == 'confirmed' and (anchor.get('complete') is not True or anchor.get('status') != 'passed'))
            or (outcome == 'slo_exhausted' and (anchor.get('complete') is not False or anchor.get('status') != 'failed'))):
        raise ValueError('overlay campaign/anchor differs from completed freeze')
    parent = json.loads(current.read_text())
    before = {p['name']: p for p in parent['points']}
    after = {p['name']: p for p in campaign['points']}
    if (len(before) != len(parent['points']) or len(after) != len(campaign['points']) or before.keys() != after.keys()
            or any(campaign.get(k) != parent.get(k) for k in ('schema', 'seed', 'duration_s', 'scales',
                                                             'measurement_protocol_version'))):
        raise ValueError('overlay changed fixed comparison inventory/protocol')
    for name, old in before.items():
        new = after[name]
        if old == new:
            continue
        # This discovery protocol fills absent workloads. Existing frozen
        # points cannot become another revision or a duplicate placeholder.
        mutable = {'trace', 'rate_rps', 'blockers', 'status'}
        if old.get('trace') or {k:v for k,v in old.items() if k not in mutable} != {
                k:v for k,v in new.items() if k not in mutable}:
            raise ValueError('overlay changed an inherited comparison point')
    if (group not in campaign.get('groups', []) or len(group['points']) != freeze.get('point_count')
            or len({p['name'] for p in group['points']}) != len(group['points'])
            or any(p != after.get(p['name']) for p in group['points'])):
        raise ValueError('overlay frozen group differs from campaign points')
    for inherited in freeze.get('inherited_points', []):
        old = before.get(inherited['name'])
        if (old is None or inherited.get('sha256') != digest(old)
                or inherited.get('trace') != old.get('trace') or after[inherited['name']] != old):
            raise ValueError('overlay inherited point binding differs')
    evidence = [freeze_path]
    completion_path = path.parent/'completion.json'
    if completion_path.exists():
        try:
            report = json.loads(completion_path.read_text())
        except json.JSONDecodeError:
            if job['status'] not in ('queued', 'running'):
                raise
            report = None
        if report is not None and (report.get('freeze') != binding(freeze_path) or report.get('campaign') != freeze['campaign']
                or report.get('parent_campaign') != parent_ref or report.get('anchor_outcome') != outcome):
            raise ValueError('overlay completion differs from completed freeze')
        if report is not None:
            evidence.append(completion_path)
    session = _attempt_output(attempt, payload['session_output'])
    return dict(campaign=path, session=session, evidence=evidence)


def comparison_watch_inputs(campaign_path, queue, *, session_roots=()):
    """Read-only queue discovery; all attempts stay visible, never best-picked."""
    current = Path(campaign_path).resolve()
    jobs = list(queue['jobs'].values()) if isinstance(queue['jobs'], dict) else queue['jobs']
    leases = list(queue.get('leases', {}).values()) if isinstance(queue.get('leases', {}), dict) else queue['leases']
    roots = {Path(p).resolve() for p in session_roots}
    evidence = []
    adopted = set()
    while True:
        watched = _campaign_ancestry(current)
        relevant = [j for j in jobs if j.get('payload', {}).get('comparison_campaign')
                    and str(Path(j['payload']['comparison_campaign']).resolve()) in watched]
        candidates = []
        for job in relevant:
            payload = job['payload']
            for lease in leases:
                if lease.get('job_id') != job['job_id']:
                    continue
                attempt = Path(lease['attempt_dir']).resolve()
                if not payload.get('overlay_campaign_output'):
                    roots.add(_attempt_output(attempt, payload.get('session_output', 'session')))
                elif str(Path(payload['comparison_campaign']).resolve()) == str(current):
                    item = _frozen_watch_overlay(job, lease, current)
                    if item:
                        candidates.append(item)
        if len(candidates) > 1:
            raise ValueError('ambiguous frozen campaign overlays; cannot select an attempt')
        if not candidates:
            break
        item = candidates[0]
        if item['campaign'] in adopted:
            raise ValueError('cyclic frozen campaign overlays')
        adopted.add(item['campaign'])
        current = item['campaign']; roots.add(item['session']); evidence.extend(item['evidence'])
    roots = sorted(p for p in roots if not any(p != other and p.is_relative_to(other) for other in roots))
    return dict(campaign=current, session_roots=roots, evidence_paths=evidence,
                watched_campaigns=watched, relevant_jobs=[j['job_id'] for j in relevant],
                terminal=bool(relevant and all(j['status'] not in ('queued', 'running') for j in relevant)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--first-spec', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--corpus-root', type=Path, required=True)
    p.add_argument('--inputs', type=Path)
    p = sub.add_parser('export'); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results/compare.csv'))
    p.add_argument('--sessions', type=Path, action='append', default=[])
    p.add_argument('--watch', action='store_true')
    p.add_argument('--queue', type=Path)
    p.add_argument('--qualification-evidence',type=Path,
                   help='Immutable historical component-attempt evidence for unmeasured blocked rows')
    p.add_argument('--analysis-policy', choices=['all_recorded_windows/v1'],
                   help='Interpret bound recorded windows as usable data while preserving strict audit diagnostics')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.first_spec, args.out, args.corpus_root,
                         prepared_inputs=json.loads(args.inputs.read_text()) if args.inputs else {})
        print(json.dumps(result['summary']))
    else:
        global _WATCH_DIGEST_CACHE
        if args.watch:
            from .comparison_hash_cache import UnchangedFileHashes
            _WATCH_DIGEST_CACHE = UnchangedFileHashes()
        qualification_ref=None;qualification_paths=[]
        if args.qualification_evidence:
            from .comparison_qualification_evidence import load_qualification_evidence
            qualification_ref=binding(args.qualification_evidence)
            qualification=load_qualification_evidence(qualification_ref,load_bound=load_bound,file_sha=file_sha)
            qualification_paths=[Path(ref['path']) for ref in qualification.refs]
        previous = None
        while True:
            terminal = not args.watch
            campaign, roots, evidence = args.campaign, args.sessions, []
            if args.queue:
                queue = json.loads(args.queue.read_text())
                inputs = comparison_watch_inputs(args.campaign, queue, session_roots=args.sessions)
                campaign, roots = inputs['campaign'], inputs['session_roots']
                evidence, terminal = inputs['evidence_paths'], inputs['terminal']
            # New immutable receipts trigger export. The process-local cache
            # avoids re-reading unchanged historical raw files each window;
            # completion still verifies all bytes again. Idle polling is metadata only.
            paths = sorted({Path(campaign), *evidence, *qualification_paths, *(p for root in roots for pattern in (
                '**/windows/*/receipt.json', '**/session/completion.json')
                for p in Path(root).glob(pattern)), *(Path(root)/'completion.json' for root in roots
                                                    if (Path(root)/'completion.json').is_file())})
            current = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
            if current != previous or terminal:
                if terminal and _WATCH_DIGEST_CACHE is not None:
                    _WATCH_DIGEST_CACHE.clear()
                print(json.dumps(dict(export(campaign, args.out, session_roots=roots,
                                             qualification_evidence_ref=qualification_ref,
                                             analysis_policy=args.analysis_policy),
                                      campaign=str(campaign))), flush=True)
                previous = current
            if terminal or not args.watch:
                break
            time.sleep(15)


if __name__ == '__main__':
    main()
