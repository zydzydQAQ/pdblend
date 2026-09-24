"""Read-only terminal point inventory. This module never schedules retries.

Failed collectors may leave useful complete windows before the failing point.
Inventory each raw file under its immutable plan/owner/source and preserve its
actual audit status; a failed whole job is not a reason to repeat its prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .native_timing_audit import need, finite
from .native_timing_plan import binding, digest
from .native_timing_replay import Resolver, mounts, _audited_window
from .native_timing_replay_v2 import _plan, _identities
from .native_timing_capacity import audit_unsupported_capacity
from .native_timing_single_pass import COLLECTION_SCHEMA, INPUT_SCHEMA, is_single_pass


def inventory(attempt, queue, *, path_map=()):
    """Bind terminal evidence and audit each point without qualifying a profile."""
    attempt = Path(attempt).resolve()
    manifest = json.loads((attempt/'manifest.json').read_text())
    execution = json.loads((attempt/'execution.json').read_text())
    queue_bytes = Path(queue).read_bytes()
    queue_value = json.loads(queue_bytes)
    queue_ref = dict(path=str(Path(queue).resolve()), sha256=hashlib.sha256(queue_bytes).hexdigest())
    job = queue_value['jobs'].get(manifest.get('job_id'), {})
    payload = manifest.get('payload', {})
    need(manifest.get('immutable') is True and job.get('job_id') == manifest.get('job_id')
         and job.get('attempts') == manifest.get('attempt') and job.get('payload') == payload
         and job.get('status') in ('succeeded', 'failed') and job.get('lease_id') is None,
         'partial inventory requires the actual terminal immutable queue attempt')
    need(execution.get('status') == ('passed' if job['status'] == 'succeeded' else 'failed')
         and finite(execution.get('finished_s')) and type(execution.get('returncode')) is int
         and payload.get('system') == 'pdblend' and payload.get('gpu_count') == 8
         and payload.get('exclusive') is True and payload.get('reserve_host') is True
         and payload.get('scope') == 'native_cuda_timing_component_only'
         and 'pdblend.profile.collection.native_timing_collect' in payload.get('argv', []),
         'partial timing worker status or exclusive invocation differs')
    uuids = manifest.get('gpu_uuids', [])
    need(len(uuids) == len(set(uuids)) == 8
         and all(isinstance(u, str) and u.startswith('GPU-') for u in uuids),
         'partial timing physical owner inventory differs')
    if job['status'] == 'succeeded':
        need(execution.get('complete') is True and execution['returncode'] == 0 and not execution.get('error'),
             'terminal worker success differs')
    else:
        need(execution.get('complete') is False and execution['returncode'] != 0,
             'terminal worker failure differs')
    resolver = Resolver([*path_map, *mounts(execution.get('argv', []))])
    root = attempt/'native-timing'
    completion_ref = binding(root/'completion.json')
    report = resolver.read(completion_ref)
    original_sha = execution.get('receipt_sha256', {}).get('native-timing/completion.json')
    lease_id = manifest.get('lease_id')
    lease = queue_value.get('leases', {}).get(lease_id) if lease_id else None
    if lease_id:
        need(isinstance(lease, dict) and lease.get('lease_id') == lease_id
             and lease.get('job_id') == manifest['job_id'] and lease.get('attempt') == manifest['attempt']
             and lease.get('status') == job['status'] and lease.get('gpu_uuids') == uuids
             and resolver.path(lease.get('attempt_dir', '')) == attempt,
             'partial terminal lease identity/status differs')
    if original_sha is not None:
        need(original_sha == completion_ref['sha256'], 'worker terminal timing completion checksum differs')
    else:
        # The worker records required-receipt hashes only after exit zero.
        # This new snapshot preserves that absence; it does not backfill it.
        need(job['status'] == 'failed' and execution['returncode'] == 2
             and execution.get('error') == 'RuntimeError: process exited 2'
             and report.get('system') == 'pdblend' and report.get('status') == 'failed'
             and report.get('complete') is False and report.get('hardware_executed') is True
             and bool(report.get('error')) and isinstance(lease, dict),
             'unbound completion requires a terminal collector saved-failure exit and its original lease')
        start = execution.get('started_s'); end = execution['finished_s']
        cleanup = report.get('physical_cleanup', {})
        observations = cleanup.get('observations', [])
        need(finite(start) and finite(lease.get('claimed_at')) and lease['claimed_at'] <= start <= end
             and not report.get('cleanup_errors') and cleanup.get('passed') is True and not cleanup.get('error')
             and finite(cleanup.get('started_s')) and finite(cleanup.get('finished_s'))
             and start <= cleanup['started_s'] <= cleanup['finished_s'] <= end and observations,
             'unbound completion needs ordered terminal physical cleanup evidence')
        for observation in observations:
            devices = observation.get('devices', [])
            need(finite(observation.get('at_s')) and cleanup['started_s'] <= observation['at_s'] <= cleanup['finished_s']
                 and len(devices) == 8 and [d.get('gpu') for d in devices] == list(range(8))
                 and [d.get('gpu_uuid') for d in devices] == uuids
                 and all(isinstance(d.get('compute_pids'), list) for d in devices),
                 'unbound completion physical cleanup inventory differs')
        need(all(a['at_s'] <= b['at_s'] for a, b in zip(observations, observations[1:]))
             and all(not d['compute_pids'] for d in observations[-1]['devices']),
             'unbound completion physical fleet was not empty')
    inputs = resolver.read(payload['input_manifest']); plan = _plan(inputs, resolver)
    need(is_single_pass(plan) and inputs.get('schema') == INPUT_SCHEMA and report.get('schema') == COLLECTION_SCHEMA
         and report.get('point_plan') == inputs['point_plan']
         and report.get('capacity_policy') == plan['capacity_policy'],
         'partial inventory is restricted to bound single-pass development timing')
    source = resolver.read(inputs['source_manifest'])
    need(digest(source['files']) == source['source_sha256'] == inputs['source_sha256'] == payload['source_sha256']
         and inputs['image_digest'] == payload['image_digest'], 'partial source or image identity differs')
    source_root = resolver.path(inputs['source_manifest']['path']).parent
    for name, checksum in source['files'].items():
        path = (source_root/name).resolve()
        need(path.is_relative_to(source_root) and binding(path)['sha256'] == checksum,
             'partial frozen source bytes differ: '+name)
    ids = tuple('pd-timing-'+str(i) for i in range(8))
    _, identities = _identities(report, inputs, plan, manifest, resolver, ids)
    expected = {f'{digest(point)[:20]}-0.json': (point, ids[index % 8])
                for index, point in enumerate(plan['points'])}
    raw_refs = report.get('raw_bindings', [])
    owners = report.get('window_owners', [])
    need([r.get('raw') for r in owners] == raw_refs,
         'partial completed raw ownership inventory differs')
    completed = {}
    for owner in owners:
        ref = owner['raw']; path = resolver.path(ref['path'])
        need(path.parent == root/'samples' and path.name in expected and path.name not in completed
             and owner.get('instance_id') == expected[path.name][1],
             'partial raw point path or owner differs')
        completed[path.name] = ref
    observed = set((root/'samples').glob('*.json'))
    need({p.name for p in observed} <= set(expected), 'partial sample has an unplanned point or repeat')
    rows = []
    for name, (point, owner) in expected.items():
        path = root/'samples'/name
        row = dict(point=point, instance_id=owner, status='missing', raw=None,
                   bound_in_original_completion=name in completed, skip_remeasurement=False)
        if not path.exists():
            if name in completed:
                row.update(status='invalid', error='original completion refers to a missing raw file')
            rows.append(row); continue
        row['raw'] = binding(path)
        try:
            if name in completed:
                need(row['raw']['sha256'] == completed[name]['sha256'], 'original raw checksum differs')
            raw = resolver.read(row['raw'])
            need(digest(raw.get('point')) == digest(dict(point, repeat=0)),
                 'raw point differs from its immutable filename and owner')
            if raw.get('status') == 'unsupported_capacity':
                audit_unsupported_capacity(raw, plan=plan, identity=identities[owner])
                need(raw['observed_s'] <= execution['finished_s'], 'capacity observation follows terminal worker')
                row.update(status='unsupported_capacity', skip_remeasurement=True)
            elif raw.get('status') == 'measured':
                events = _audited_window(raw, identities[owner])
                need(raw['drain']['response_at_s'] <= execution['finished_s'], 'timing observation follows terminal worker')
                row.update(status='measured', audited_cuda_events=len(events), skip_remeasurement=True)
            else:
                need(raw.get('status') == 'failed' and raw.get('schema') == 'pdblend-native-timing-window-v1'
                     and raw.get('system') == 'pdblend', 'unknown raw failure identity or status')
                capability = raw.get('capability')
                if capability is not None:
                    need(all(capability.get(k) == v for k, v in identities[owner].items()),
                         'failed raw owner/model/source identity differs')
                row.update(status='failed', error=raw.get('error'),
                           partial_requests=len(raw.get('client_requests', [])),
                           raw_owner_identity_verified=capability is not None)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            row.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    counts = {status: sum(row['status'] == status for row in rows)
              for status in ('measured', 'unsupported_capacity', 'failed', 'invalid', 'missing')}
    return dict(schema='pdblend-native-timing-partial-inventory/v1', captured_s=time.time(),
        capture_scope='new_post_terminal_read_only_inventory', hardware_executed=False,
        original_job_status=job['status'], original_error=report.get('error'),
        job_id=job['job_id'], queue_job_sha256=digest(job),
        queue_snapshot=queue_ref, queue_job=job, terminal_lease=lease,
        terminal_lease_sha256=digest(lease) if lease is not None else None,
        original_worker_completion_binding=dict(present=original_sha is not None, sha256=original_sha),
        completion_binding_scope=('original_worker_receipt' if original_sha is not None
                                  else 'new_post_terminal_snapshot_not_original_worker_binding'),
        attempt_manifest=binding(attempt/'manifest.json'), worker_execution=binding(attempt/'execution.json'),
        completion=completion_ref, input_manifest=payload['input_manifest'],
        point_plan=inputs['point_plan'], source_manifest=inputs['source_manifest'],
        counts=counts, points=rows, full_job_retry_allowed=False, automatic_retry_allowed=False,
        missing_points_need_new_immutable_selective_plan=True,
        failed_or_invalid_points_need_explicit_cause_review=True,
        formal_eligible=False, component_qualified=False, full_profile_qualified=False)


def capture_inventory(attempt, queue, out):
    """Save a new exact queue snapshot and inventory; never amend the attempt."""
    out = Path(out).resolve()
    need(not out.exists(), 'new immutable partial inventory directory required')
    data = Path(queue).read_bytes()
    out.mkdir(parents=True)
    snapshot = out/'terminal-queue.json'
    with snapshot.open('xb') as stream: stream.write(data)
    result = inventory(attempt, snapshot)
    result['observed_queue_source'] = dict(path=str(Path(queue).resolve()),
                                         sha256=hashlib.sha256(data).hexdigest())
    with (out/'inventory.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    return binding(out/'inventory.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='New directory for an immutable terminal snapshot and inventory')
    args = parser.parse_args()
    result = capture_inventory(args.attempt, args.queue, args.out) if args.out else inventory(args.attempt, args.queue)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
