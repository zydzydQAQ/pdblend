"""One CSV row per execution attempt or explicitly named service window.

Absent metrics stay empty. Historical CSVs remain historical; collecting a
summary never grants formal eligibility. No predictor or policy is imported.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

IDENTITY = ('model_id', 'system', 'topology_id', 'tp_mode', 'tp', 'pp', 'tp_layout',
            'dataset', 'rate_scale', 'offered_rps', 'seed', 'duration_s',
            'slo_ttft_s', 'slo_tpot_s', 'trace_sha256', 'profile_sha256',
            'source_sha256', 'image_digest', 'model_hash', 'tokenizer_hash',
            'energy_protocol', 'measurement_source_sha256', 'runtime_source_sha256')
ENERGY_PHASES = ('service', 'cold_start', 'wake', 'park', 'off', 'drain',
                 'weight_transfer', 'kv_transfer', 'rollback', 'tail', 'total')
NATIVE_METRICS = ('throughput_request_s', 'throughput_token_s', 'goodput_request_s', 'goodput_token_s',
                  'planner_p50_s', 'planner_p95_s', 'planner_p99_s', 'periodic_decisions',
                  'periodic_planner_calls', 'control_actions', 'pd_requests')
METRICS = ('offered_requests', 'successful_requests', 'output_tokens', 'good_output_tokens',
           'success_rate', 'joint_slo_rate',
           *[f'{metric}_p{p}_s' for metric in ('ttft', 'tpot') for p in (50, 90, 95, 99)],
           *[f'energy_{phase}_j' for phase in ENERGY_PHASES], 'energy_recorded_j', 'energy_scope', 'j_per_token', 'j_per_good_token',
           'instance_count', 'reconfigurations', *NATIVE_METRICS)
FIELDS = ('run_id', 'attempt_id', 'campaign_id', 'purpose', 'record_kind',
          *IDENTITY, 'status', 'formal_eligible', 'single_seed', 'evidence_status',
          'queue_status', 'arm', 'comparison_mode', 'baseline_definition', 'efficacy_classification',
          'audit_status', 'latency_comparable', 'functional_passed', 'execution_complete',
          'requires_serial_retest', 'failure_reason', *METRICS, 'artifact_path', 'artifact_sha256',
          'manifest_path', 'audit_path', 'audit_sha256', 'retention_manifest')
PAIR_FIELDS = ('model_id', 'dataset', 'trace_sha256', 'offered_rps', 'seed', 'duration_s',
               'slo_ttft_s', 'slo_tpot_s', 'energy_protocol', 'image_digest',
               'measurement_source_sha256', 'runtime_source_sha256')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def scalar(value):
    if value is None or isinstance(value, (dict, list, tuple)):
        return ''
    if isinstance(value, float) and not math.isfinite(value):
        return ''
    return value


def _take(row, source, names):
    for target, aliases in names.items():
        if row.get(target, '') != '':
            continue
        for alias in aliases:
            value = scalar(source.get(alias))
            if value != '':
                row[target] = value
                break


ALIASES = {name: (name,) for name in (*IDENTITY, *METRICS) if name not in NATIVE_METRICS}
ALIASES.update(model_id=('model_id', 'model'), system=('system', 'policy'),
    rate_scale=('rate_scale', 'scale'), offered_rps=('offered_rps', 'rate_rps', 'mean_rps', 'rate'),
    duration_s=('duration_s', 'window_s', 'duration'), offered_requests=('offered_requests', 'offered'),
    successful_requests=('successful_requests', 'correct', 'succeeded'),
    good_output_tokens=('good_output_tokens', 'joint_output_tokens'),
    j_per_good_token=('j_per_good_token', 'j_per_goodput_token'),
    image_digest=('image_digest', 'image_id'),
    energy_service_j=('energy_service_j', 'window_energy_j'), energy_recorded_j=('energy_recorded_j', 'energy_j'))
for metric in ('ttft', 'tpot'):
    for p in (50, 90, 95, 99):
        ALIASES[f'{metric}_p{p}_s'] = (f'{metric}_p{p}_s', f'{metric}_p{p}')


def normalize(data, *, path, root, context=None, historical=False, record_kind='experiment_attempt'):
    path, root = Path(path), Path(root)
    context = context or {}
    row = dict.fromkeys(FIELDS, '')
    relative = os.path.relpath(path, root)
    row.update(run_id=hashlib.sha256(relative.encode()).hexdigest()[:24],
        attempt_id=context.get('attempt_id', path.parent.name),
        campaign_id=context.get('campaign_id', path.parent.parent.name),
        record_kind=record_kind, artifact_path=str(path.resolve()), artifact_sha256=sha(path),
        manifest_path=context.get('manifest_path', ''),
        purpose='historical' if historical else context.get('purpose', 'development'),
        status=scalar(data.get('status')) or context.get('status', 'recorded'),
        failure_reason=scalar(data.get('error')) or scalar(data.get('failure_reason')),
        formal_eligible=False, evidence_status='historical' if historical else 'current')
    # Recorded metrics outrank declared configuration, and scalar aliases are
    # explicit. No missing model, profile, seed or clock identity is guessed.
    provenance = data.get('provenance', {})
    sources = [data.get('metrics', {}), data.get('slo', {}), data,
               data.get('trace_meta', {}), data.get('trace', {}), context]
    if isinstance(provenance, dict):
        sources[3:3] = [provenance, provenance.get('model_identity', {})]
    for source in sources:
        if isinstance(source, dict):
            _take(row, source, ALIASES)
    if isinstance(data.get('policy'), dict):
        row['system'] = data['policy'].get('name', row['system'])
    if (isinstance(row['model_id'], str) and row['model_id'].startswith('/')
            and context.get('model_id') == Path(row['model_id']).name):
        row['model_id'] = context['model_id']
    if isinstance(data.get('requests'), int) and row['offered_requests'] == '':
        row['offered_requests'] = data['requests']
    if isinstance(data.get('metrics'), dict) and 'passed' in data['metrics'] and 'status' not in data:
        row['status'] = 'passed' if data['metrics']['passed'] else 'failed'
    row['single_seed'] = row['seed'] in (701, '701')
    # Only an explicit completed campaign acceptance receipt may elevate a row.
    row['formal_eligible'] = (not historical and data.get('formal_eligible') is True
        and data.get('energy_comparable') is True and data.get('complete') is True
        and data.get('status') == 'passed' and data.get('scope') == 'formal_campaign_acceptance'
        and all(row.get(field, '') != '' for field in PAIR_FIELDS))
    if row['energy_scope'] == '':
        if row['energy_total_j'] != '': row['energy_scope'] = 'explicit_total'
        elif row['energy_recorded_j'] != '': row['energy_scope'] = 'recorded_interval_not_proven_full_lifecycle'
    energy = row['energy_total_j'] if row['energy_total_j'] != '' else row['energy_recorded_j']
    tokens, good = row['output_tokens'], row['good_output_tokens']
    if isinstance(energy, (int, float)):
        for key, count in [('j_per_token', tokens), ('j_per_good_token', good)]:
            if row[key] == '' and isinstance(count, (int, float)) and count > 0:
                row[key] = energy/count
    return row


def write_csv(path, rows, fields=FIELDS):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def comparison_key(row):
    if str(row.get('formal_eligible')).lower() != 'true' or row.get('evidence_status') != 'current':
        raise ValueError('only qualified current evidence can enter a formal comparison')
    if any(row.get(k, '') == '' for k in PAIR_FIELDS):
        raise ValueError('comparison identity is incomplete')
    numbers = {'offered_rps','seed','duration_s','slo_ttft_s','slo_tpot_s'}
    def value(key):
        if key not in numbers: return str(row[key])
        try: number=Decimal(str(row[key]))
        except InvalidOperation as exc: raise ValueError('invalid numeric comparison identity') from exc
        if not number.is_finite() or number < 0:
            raise ValueError('invalid numeric comparison identity')
        return str(number.normalize())
    return tuple(value(k) for k in PAIR_FIELDS)


def _native_ab_row(data, *, path, root, attempt, context, preflight, audit):
    """Keep native A/B execution, qualification, and efficacy distinct.

    No outcomes are rewritten or reclassified here. Audited arm metrics are
    consumed only for their bound final A/B directories, never for retained
    rejected parallel windows or another source/trace attempt.
    """
    path = Path(path)
    bindings = data.get('bindings') or preflight.get('bindings', {})
    local = dict(context, status=data.get('status', context.get('status', '') if path.parent == attempt else 'recorded'),
                 purpose='development_attempt')
    local.update({key: bindings[key] for key in IDENTITY if scalar(bindings.get(key)) != ''})
    local.update(system='pdblend', tp_mode='fixed_tp', seed=bindings.get('seed', ''),
                 profile_sha256=bindings.get('inputs', {}).get('profile', {}).get('sha256', ''))
    relative = path.relative_to(attempt)
    rejected_parallel = 'parallel-unqualified' in relative.parts
    arm = path.parent.name if path.parent.name in ('A', 'B') else ''
    functional = relative.parts[0] == 'functional'
    kind = 'native_ab_arm' if arm else 'functional_qualification' if functional else 'attempt_aggregate'
    row = normalize(data, path=path, root=root, context=local, record_kind=kind)
    row.update(queue_status=context.get('status', ''), formal_eligible=False,
               arm=arm, latency_comparable=False,
               baseline_definition='fixed_initial_vs_periodic_same_source' if not functional else '',
               comparison_mode=scalar(data.get('comparison_mode')),
               functional_passed=scalar(data.get('functional_passed')),
               execution_complete=scalar(data.get('execution_complete', data.get('complete'))))
    # A functional probe has a different request set. It must not acquire the
    # evaluation trace/duration merely because its fleet is shared with A/B.
    if functional:
        row.update(purpose='development_functional', dataset='synthetic_functional',
                   trace_sha256='', duration_s='', offered_rps='', rate_scale='')
        start, end = data.get('started_s'), data.get('finished_s')
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            row['duration_s'] = end-start
    elif arm:
        row.update(purpose='development_ab', dataset='synthetic_native_ab',
                   duration_s=bindings.get('duration_s', ''), slo_ttft_s=5., slo_tpot_s=.15,
                   execution_complete=True)
        _take(row, data, {key: (key,) for key in NATIVE_METRICS})
    else:
        row.update(purpose='development_attempt', dataset='', duration_s='',
                   slo_ttft_s='', slo_tpot_s='', offered_rps='', rate_scale='')
    gpu_uuids = bindings.get('gpu_uuids', [])
    if isinstance(gpu_uuids, str):
        gpu_uuids = [v for v in gpu_uuids.split(',') if v]
    tp, pp = bindings.get('tp'), bindings.get('pp')
    if (isinstance(gpu_uuids, list) and gpu_uuids and type(tp) is int and tp > 0
            and type(pp) is int and pp > 0 and len(gpu_uuids) % (tp*pp) == 0):
        layout = [dict(tp=tp, pp=pp, gpu_uuids=gpu_uuids[i:i+tp*pp])
                  for i in range(0, len(gpu_uuids), tp*pp)]
        encoded = json.dumps(layout, sort_keys=True, separators=(',', ':'))
        row.update(instance_count=len(layout), tp_layout=encoded,
                   topology_id=hashlib.sha256(encoded.encode()).hexdigest()[:24])
    valid_audit = (isinstance(audit, dict) and audit.get('schema') == 'pdblend.native-ab-audit/v1'
                   and all(audit.get('identity', {}).get(key) == bindings.get(key)
                       for key in ('model_id', 'tp', 'pp', 'seed', 'source_sha256', 'trace_sha256', 'image_digest'))
                   and audit.get('identity', {}).get('inputs', {}).get('profile') == local['profile_sha256'])
    if valid_audit and not rejected_parallel:
        audit_path = Path(context.get('_native_audit_path', attempt/'audit.json'))
        row.update(audit_path=str(audit_path.resolve()), audit_sha256=sha(audit_path),
                   audit_status=scalar(audit.get('status')),
                   requires_serial_retest=scalar(audit.get('requires_serial_retest')))
        if functional:
            row['audit_status'] = ('passed' if audit.get('functional', {}).get('passed') is True else '')
            row['requires_serial_retest'] = ''
        if not functional:
            row['efficacy_classification'] = scalar(audit.get('classification'))
            row['comparison_mode'] = row['comparison_mode'] or scalar(audit.get('interference', {}).get('comparison_mode'))
        if arm:
            proof = audit.get('arms', {}).get(arm, {})
            metrics = proof.get('metrics', {})
            # Audit recomputes these from exact outcomes; replace summary
            # values only when the audit belongs to this source and trace.
            for target, source in (('offered_requests', 'offered'), ('successful_requests', 'succeeded'),
                    ('output_tokens', 'output_tokens'), ('good_output_tokens', 'joint_output_tokens'),
                    ('success_rate', 'success_rate'), ('joint_slo_rate', 'joint_slo_rate'),
                    ('throughput_request_s', 'throughput_request_s'), ('throughput_token_s', 'throughput_token_s'),
                    ('goodput_request_s', 'goodput_request_s'), ('goodput_token_s', 'goodput_token_s')):
                if source in metrics:
                    row[target] = scalar(metrics[source])
            for metric in ('ttft', 'tpot'):
                for percentile in (50, 95, 99):
                    row[f'{metric}_p{percentile}_s'] = scalar(metrics.get(metric+'_s', {}).get(f'p{percentile}'))
            control = proof.get('controller', {})
            for percentile in (50, 95, 99):
                row[f'planner_p{percentile}_s'] = scalar(control.get('planning_seconds', {}).get(f'p{percentile}'))
            for target, source in (('periodic_decisions', 'periodic_decisions'),
                    ('periodic_planner_calls', 'periodic_planner_calls'), ('control_actions', 'actual_actions'),
                    ('pd_requests', 'automatic_pd_requests')):
                row[target] = scalar(control.get(source))
            row['latency_comparable'] = (bool(proof) and audit.get('interference', {}).get('passed') is True
                                         and not audit.get('errors') and audit.get('status') in ('passed', 'failed'))
        if not functional and audit.get('errors') and not row['failure_reason']:
            row['failure_reason'] = json.dumps(audit['errors'], ensure_ascii=False, separators=(',', ':'))
    if rejected_parallel:
        row.update(evidence_status='parallel_unqualified', latency_comparable=False,
                   audit_status='parallel_interference_failed', requires_serial_retest=True,
                   comparison_mode='parallel_unqualified', efficacy_classification='inconclusive')
    return row


def collect(root):
    root = Path(root).resolve()
    rows, consumed, errors = [], set(), []
    queues = sorted(root.glob('*/three-model/queue.json'))
    for queue_path in queues:
        queue = read_json(queue_path)
        for lease in queue.get('leases', {}).values():
            attempt = Path(lease.get('attempt_dir', ''))
            if not attempt.is_dir():
                continue
            job = queue['jobs'].get(lease['job_id'], {})
            payload = job.get('payload', {})
            context = dict(payload, attempt_id=attempt.name, campaign_id=queue_path.parent.name,
                status=job.get('status', ''), manifest_path=str(attempt/'manifest.json'), purpose='development')
            native_ab = lease['job_id'].startswith(('pdblend-quick-ab-', 'pdblend-quick-pd8-ab-'))
            preflight, native_audit = {}, {}
            if native_ab:
                audit_path = attempt/'audit-plan-identity.json'
                if not audit_path.is_file():
                    audit_path = attempt/'audit.json'
                context['_native_audit_path'] = str(audit_path)
                for target, destination in ((attempt/'preflight.json', preflight), (audit_path, native_audit)):
                    if target.is_file():
                        try:
                            destination.update(read_json(target))
                        except (ValueError, TypeError, OSError) as exc:
                            errors.append(dict(path=str(target), error=str(exc)))
            if 'profile' in lease['job_id'] or 'resident-domain' in lease['job_id']:
                context['purpose'] = 'profiling'
            elif 'rate-anchor' in lease['job_id']:
                context['purpose'] = 'calibration'
            proofs = sorted(set(attempt.rglob('completion.json')) | set(attempt.rglob('summary.json')) |
                            set(attempt.rglob('completion.json.gz')) | set(attempt.rglob('summary.json.gz')))
            # completion and summary in the same directory describe one run.
            # Prefer the completion receipt; never count a retry as the same run.
            by_directory = {}
            for path in proofs:
                if path.parent not in by_directory or path.name.startswith('completion'):
                    by_directory[path.parent] = path
            proofs = sorted(by_directory.values())
            if native_ab and not any(p.parent == attempt for p in proofs) and (attempt/'manifest.json').is_file():
                proofs.append(attempt/'manifest.json')
            # A rate-anchor aggregate contains no additional experiment; its
            # model/source identity is inherited by the individual windows.
            aggregates = set()
            for path in proofs:
                if path.parent.name == 'anchor' and any(p.parent.parent == path.parent for p in proofs):
                    try:
                        aggregate = read_json(path)
                        for key in IDENTITY:
                            if scalar(aggregate.get(key)) != '':
                                context[key] = aggregate[key]
                        aggregates.add(path)
                    except (ValueError, OSError) as exc:
                        errors.append(dict(path=str(path), error=str(exc)))
            for path in proofs:
                if path.is_symlink() or path in aggregates:
                    continue
                try:
                    data = read_json(path)
                    if not isinstance(data, dict):
                        continue
                    if native_ab:
                        row = _native_ab_row(data, path=path, root=root, attempt=attempt, context=context,
                                             preflight=preflight, audit=native_audit)
                    else:
                        row = normalize(data, path=path, root=root, context=context,
                            record_kind='service_window' if path.parent.parent.name == 'anchor' else 'experiment_attempt')
                    rows.append(row); consumed.add(path.resolve())
                except (ValueError, OSError) as exc:
                    errors.append(dict(path=str(path), error=str(exc)))
            if not proofs:
                manifest = attempt/'manifest.json'
                if manifest.is_file():
                    rows.append(normalize({}, path=manifest, root=root, context=context))
    # Historical point summaries retain their own evidence identity. Never
    # traverse symbolic aliases or source snapshots as additional experiments.
    for path in sorted(set(root.rglob('summary.json')) | set(root.rglob('summary.json.gz'))):
        if path.resolve() in consumed or path.is_symlink() or 'queue-attempts' in path.parts:
            continue
        try:
            data = read_json(path)
            if not isinstance(data, dict) or not any(k in data for k in ('policy', 'slo', 'window_energy_j')):
                continue
            defaults = {}
            for parent in path.parents:
                if parent == root.parent:
                    break
                spec = parent/'spec.json'
                if spec.is_file():
                    defaults = read_json(spec).get('defaults', {}); break
            rows.append(normalize(data, path=path, root=root, context=defaults, historical=True))
            consumed.add(path.resolve())
        except (ValueError, OSError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    # Persist historical rows after their raw evidence has intentionally gone.
    existing = root/'runs.csv'
    if existing.exists():
        with existing.open(newline='') as stream:
            previous = list(csv.DictReader(stream))
        known = {r['run_id'] for r in rows}
        pruned = {r['run_id']:r for r in previous if r.get('evidence_status') == 'raw_pruned'}
        for row in rows:
            if row['run_id'] in pruned:
                row.update(evidence_status='raw_pruned', formal_eligible=False, purpose='historical',
                           retention_manifest=pruned[row['run_id']].get('retention_manifest',''))
        rows.extend(r for r in previous if r.get('evidence_status') == 'raw_pruned' and r['run_id'] not in known)
    return sorted(rows, key=lambda r:(str(r['campaign_id']), str(r['attempt_id']), r['run_id'])), errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    rows, errors = collect(args.root)
    dest = args.out or args.root/'runs.csv'
    write_csv(dest, rows)
    print(json.dumps(dict(path=str(dest.resolve()), rows=len(rows), partial_or_invalid_artifacts=errors)))


if __name__ == '__main__':
    main()
