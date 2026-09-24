"""Explicit timing-stage evidence, including a failed later supplement.

This new kind preserves the actual job outcome. It cannot rescue timing,
snapshot, source, interference or final-cleanup failure. The legacy replayers
still require whole-job success. Raw replay is shared with the resident bridge.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import time

from .native_timing_audit import need, finite
from .native_timing_plan import binding, digest
from .native_timing_replay import Resolver, mounts
from .native_runtime_collect import write_new
from .native_layout_stage import _replay_resident_timing

STAGE_SCHEMA = 'pdblend-native-independent-timing-stage/v1'
EVIDENCE_SCHEMA = 'pdblend-native-terminal-timing-stage-evidence/v1'
REPLAY_SCHEMA = 'pdblend-native-terminal-timing-stage-replay/v1'
SUPPLEMENTS = {
    'power_pilot': ('resident_power_pilot', 'power-pilot', 'power_pilot_plan'),
    'request_cycles': ('resident_request_cycles', 'request-cycles', 'request_cycle_plan'),
    'layout_energy': ('resident_layout_energy', 'layout-energy', 'layout_energy_plan'),
}


def _order(inputs):
    need(inputs.get('schema') == 'pdblend-native-timing-inputs/v2'
         and inputs.get('timing_first') is True, 'independent timing requires explicit v2 timing_first')
    order = (['runtime'] if inputs.get('collect_runtime') else []) + ['timing']
    order += [name for name, (_, _, key) in SUPPLEMENTS.items() if inputs.get(key)]
    need(inputs.get('phase_order') == order, 'independent timing frozen phase order differs')
    return order


def _events(report, *, running_snapshot=False):
    events = report.get('phase_events', [])
    names = [r.get('phase') for r in events]
    order = ['startup', *report['phase_order']]
    order.insert(order.index('timing') + 1, 'timing_snapshot')
    need(events and names == order[:len(names)] and len(names) == len(set(names)),
         'timing phase history is not a unique ordered prefix')
    previous = None
    for index, event in enumerate(events):
        start = event.get('started_s')
        need(finite(start) and (previous is None or previous <= start), 'timing phase timestamps overlap')
        if running_snapshot and index == len(events)-1:
            need(event.get('phase') == 'timing_snapshot' and event.get('status') == 'running'
                 and 'finished_s' not in event and not event.get('error'), 'snapshot is not the actual running phase')
        else:
            end = event.get('finished_s')
            need(finite(end) and start <= end and event.get('status') in ('passed', 'failed'),
                 'timing phase completion timestamp/status absent')
            need((event['status'] == 'failed') == bool(event.get('error')), 'timing phase failure/error differs')
            need(event['status'] == 'passed' or index == len(events)-1, 'collection continued after a failed phase')
            previous = end
    return events


def capture_timing_stage(report, *, input_manifest_ref, attempt_manifest_ref, specs, fleet, out):
    """Freeze the actual live final-drain boundary, before any supplement."""
    out = Path(out)
    need(not out.exists() and not report.get('physical_cleanup'), 'timing stage already exists or fleet released')
    inputs = Resolver().read(input_manifest_ref)
    need(report.get('phase_order') == _order(inputs), 'timing report phase order differs from inputs')
    _events(report, running_snapshot=True)
    need(report.get('active_phase') == 'timing_snapshot' and not report.get('resident_timing_stage'),
         'timing stage can only be captured once at timing_snapshot')
    observed = []
    for spec in specs:
        instance = fleet[spec.instance_id]
        need(instance.alive() and instance.process is not None and asdict(instance.spec) == asdict(spec),
             'timing stage lost its actual native engine/spec epoch')
        observed.append(dict(instance_id=spec.instance_id, spec=asdict(spec), process_alive=instance.alive(),
            process_pid=instance.process.pid, checked_s=time.time(),
            starts=[dict(e) for e in instance.events if e.get('kind') == 'start']))
    saved = write_new(out.with_name(out.stem+'-report.json'), deepcopy(report))
    root = Path(report['timing_component']['path']).resolve().parent
    reference = write_new(out, dict(schema=STAGE_SCHEMA, captured_s=time.time(), report=saved,
        input_manifest=input_manifest_ref, attempt_manifest=attempt_manifest_ref, timing_root=str(root),
        live_instances=observed, interference=[binding(p) for p in sorted((root/'interference').glob('*.json'))],
        resident_stage_only=True, physical_cleanup_verified=False, queue_terminal_verified=False,
        worker_execution_complete=False, formal_eligible=False, full_profile_qualified=False))
    replay_timing_stage(reference)
    return reference


def replay_timing_stage(reference, *, path_map=()):
    """Strict raw/holdout replay without pretending live engines were released."""
    resolver = Resolver(path_map)
    stage = resolver.read(reference); report = resolver.read(stage['report'])
    inputs = resolver.read(stage['input_manifest'])
    need(report.get('phase_order') == _order(inputs), 'timing stage phase order differs')
    events = _events(report, running_snapshot=True)
    need(report.get('active_phase') == 'timing_snapshot' and report.get('status') == 'passed'
         and report.get('complete') is True and not report.get('failed_phase'), 'timing stage did not finish sampling')
    timing = events[-2]; snapshot = events[-1]
    need(timing['phase'] == 'timing' and timing['status'] == 'passed'
         and report.get('timing_completed_s') == timing['finished_s']
         and timing['finished_s'] <= snapshot['started_s'] <= stage['captured_s'],
         'timing completed boundary differs from actual phases')
    result = _replay_resident_timing(reference, path_map=path_map, stage_schema=STAGE_SCHEMA, layout_only=False)
    need(timing['started_s'] <= result['collection_first_window_s']
         and result['timing_last_window_s'] <= timing['finished_s'], 'raw timing windows escape their declared phase')
    need(max(row['received_s'] for row in report['final_drains']) <= timing['finished_s'],
         'timing phase ended before its final native drain')
    result.update(schema='pdblend-native-independent-timing-stage-replay/v1', timing_completed_s=timing['finished_s'])
    return result


def _terminal_attempt(manifest, execution, job, lease):
    payload = manifest.get('payload', {})
    need(manifest.get('immutable') is True and manifest.get('job_id') == job.get('job_id')
         and manifest.get('attempt') == job.get('attempts') and payload == job.get('payload')
         and job.get('status') in ('succeeded', 'failed') and job.get('lease_id') is None,
         'timing stage needs the actual terminal immutable queue attempt')
    need(lease.get('lease_id') == manifest.get('lease_id') and lease.get('job_id') == manifest['job_id']
         and lease.get('status') == job['status'] and lease.get('gpu_uuids') == manifest.get('gpu_uuids'),
         'timing stage terminal lease identity/status differs')
    need(payload.get('system') == 'pdblend' and payload.get('scope') == 'native_cuda_timing_component_only'
         and payload.get('gpu_count') == 8 and payload.get('exclusive') is True and payload.get('reserve_host') is True
         and 'pdblend.profile.collection.native_timing_collect' in payload.get('argv', [])
         and '--timing-first' in payload.get('argv', []),
         'timing stage requires the original exclusive full-fleet invocation')
    need(finite(execution.get('started_s')) and finite(execution.get('finished_s'))
         and execution['started_s'] <= execution['finished_s'], 'worker lifetime timestamps absent')
    if job['status'] == 'succeeded':
        need(execution.get('status') == 'passed' and execution.get('complete') is True
             and execution.get('returncode') == 0 and not execution.get('error'), 'terminal worker success differs')
    else:
        # The current collector returns2 after saving its failed completion.
        # Worker failures before/after that boundary are not attributable here.
        need(execution.get('status') == 'failed' and execution.get('complete') is False
             and execution.get('returncode') == 2 and execution.get('error') == 'RuntimeError: process exited 2',
             'failed worker was not the collector saved-failure exit')


def _later_failure(report, inputs, root, resolver, stage):
    phase = report.get('failed_phase')
    need(phase in SUPPLEMENTS, 'failure was not a later independent supplement')
    field, directory, key = SUPPLEMENTS[phase]
    need(report.get('active_phase') == phase and inputs.get(key), 'failed supplement was not scheduled')
    ref = report.get(field)
    need(isinstance(ref, dict) and resolver.path(ref.get('path', '')) == root/directory/'completion.json',
         'failed supplement lacks its actual bound completion')
    receipt = resolver.read(ref); event = report['phase_events'][-1]
    need(event['phase'] == phase and event['status'] == 'failed' and event['error'] == report.get('error')
         and event['started_s'] >= stage['captured_s'], 'supplement failure does not follow the frozen stage')
    need(receipt.get('operational_failure') is True or receipt.get('error') or receipt.get('cleanup_errors'),
         'supplement receipt does not contain the claimed failure')
    plan_ref = inputs[key]
    if phase == 'power_pilot':
        need(receipt.get('schema') == 'pdblend-native-power-pilot/v1'
             and receipt.get('plan') == resolver.read(plan_ref), 'failed power supplement plan differs')
        start = receipt.get('lease', {}).get('checked_s')
        observations = receipt.get('restoration', {}).get('instances', [])
        stamps = [row.get('capability', {}).get('state', {}).get('response_at_s') for row in observations]
        need(stamps and all(finite(t) for t in stamps), 'power failure has no restored native observation boundary')
        end = max(stamps)
    else:
        expected = ('pdblend-native-request-cycle-revision/v1' if phase == 'request_cycles'
                    else 'pdblend-native-layout-revision/v1')
        need(receipt.get('schema') == expected and receipt.get('plan_sha256') == digest(resolver.read(plan_ref)),
             'failed supplement schema/plan binding differs')
        start, end = receipt.get('started_s'), receipt.get('finished_s')
    need(finite(start) and finite(end) and event['started_s'] <= start <= end <= event['finished_s'],
         'supplement raw receipt was not produced in its later phase')
    return dict(phase=phase, completion=ref, receipt_status=receipt.get('status'),
                error=receipt.get('error'), safe_restore_passed=receipt.get('safe_restore_passed'))


def _replay_terminal(evidence, resolver):
    from .native_timing_replay_v2 import _cleanup
    from pdblend.bench.comparison_acceptance import _equal
    manifest = resolver.read(evidence['attempt_manifest']); execution = resolver.read(evidence['worker_execution'])
    job = evidence['queue_job']; lease = evidence['queue_lease']
    need(digest(job) == evidence['queue_job_sha256'] and digest(lease) == evidence['queue_lease_sha256'],
         'captured terminal queue evidence differs')
    _terminal_attempt(manifest, execution, job, lease)
    resolver = Resolver([*[(str(a), str(b)) for a, b in resolver.mappings],
                         *evidence.get('path_map', []), *mounts(execution['argv'])])
    root = resolver.path(evidence['attempt_manifest']['path']).parent/'native-timing'
    need(resolver.path(evidence['completion']['path']) == root/'completion.json', 'terminal completion path differs')
    report = resolver.read(evidence['completion']); stage_ref = report.get('resident_timing_stage')
    need(stage_ref == evidence['timing_stage'], 'terminal report changed the original timing stage binding')
    need(isinstance(stage_ref, dict) and resolver.path(stage_ref.get('path', '')) == root/'timing-stage.json',
         'terminal timing stage is not the original snapshot path')
    stage = resolver.read(stage_ref); saved = resolver.read(stage['report'])
    need(resolver.path(stage['report']['path']) == root/'timing-stage-report.json',
         'terminal timing stage report snapshot path differs')
    need(stage['attempt_manifest'] == evidence['attempt_manifest']
         and stage['input_manifest'] == manifest['payload']['input_manifest'], 'stage belongs to another attempt/input')
    inputs = resolver.read(stage['input_manifest'])
    need(report.get('phase_order') == saved.get('phase_order') == _order(inputs), 'terminal phase plan differs')
    events = _events(report); before = _events(saved, running_snapshot=True); index = len(before)-1
    need(len(events) > index and events[:index] == before[:index]
         and {k: events[index].get(k) for k in before[-1]} == before[-1] | {'status': 'passed'}
         and events[index]['finished_s'] >= stage['captured_s'] and not events[index].get('error'),
         'timing snapshot itself failed or phase history changed')
    need(execution['started_s'] <= events[0]['started_s'] and events[-1]['finished_s'] <= execution['finished_s'],
         'phase history escapes worker lifetime')
    need(report.get('schema') == 'pdblend-native-timing-collection/v2' and report.get('system') == 'pdblend'
         and report.get('hardware_executed') is True and not report.get('cleanup_errors'),
         'terminal collector or cleanup failed')
    # The saved report is immutable. Supplement receipt additions are allowed;
    # all fields proving timing/identity/ownership stay byte-value equivalent.
    protected = ('model_id', 'tp', 'pp', 'frequency_domain_ref', 'frequency_domain', 'frequency_domain_sha256',
        'point_plan', 'capacity_policy', 'capabilities', 'actual_launch', 'raw_bindings',
        'window_owners', 'interference_peer_states', 'measurement_qualification', 'window_partition',
        'timing_component', 'component_qualified', 'measured_windows', 'unsupported_windows', 'timing_completed_s')
    need(all(_equal(report.get(k), saved.get(k)) for k in protected), 'terminal report rewrote timing evidence')
    if job['status'] == 'succeeded':
        need(report.get('status') == 'passed' and report.get('complete') is True
             and not report.get('error') and not report.get('failed_phase') and report.get('active_phase') == 'completed'
             and all(e['status'] == 'passed' for e in events)
             and len(events) == len(report['phase_order'])+2, 'successful terminal collector/phase inventory differs')
        need(execution.get('receipt_sha256', {}).get('native-timing/completion.json') == evidence['completion']['sha256'],
             'successful worker did not bind its completion')
        failure = None
    else:
        need(report.get('status') == 'failed' and report.get('complete') is False and bool(report.get('error')),
             'failed job was relabeled as a successful collector')
        failure = _later_failure(report, inputs, root, resolver, stage)
        worker_sha = execution.get('receipt_sha256', {}).get('native-timing/completion.json')
        need(worker_sha in (None, evidence['completion']['sha256']), 'failed worker completion hash conflicts')
    replay = replay_timing_stage(stage_ref, path_map=[(str(a), str(b)) for a, b in resolver.mappings])
    ids = tuple(row['instance_id'] for row in stage['live_instances'])
    need(report['physical_cleanup'].get('started_s', -1) >= events[-1]['finished_s'],
         'physical cleanup predates later phase completion')
    _cleanup(report, manifest['gpu_uuids'], ids, replay['timing_last_window_s'], execution)
    # Supplement may re-drain the same fleet; its final native timestamps must
    # remain after the original stage boundary. PID restart history is a prefix.
    for row in stage['live_instances']:
        starts = report['actual_engine_starts'][row['instance_id']]
        need(starts[:len(row['starts'])] == row['starts'], 'final engine history lost the timing-stage PID lineage')
    replay.update(schema=REPLAY_SCHEMA, resident_stage_only=False, physical_cleanup_verified=True,
        queue_terminal_verified=True, worker_execution_complete=execution['complete'],
        parent_job_status=job['status'], parent_execution_status=execution['status'],
        parent_job_succeeded=job['status'] == 'succeeded', later_independent_failure=failure,
        timing_component_reusable=replay['component_qualified'], whole_job_result_modified=False,
        auxiliary_power_qualifies_power_component=False, full_profile_qualified=False, formal_eligible=False)
    return replay


def capture_terminal_evidence(attempt, queue, out, *, path_map=()):
    """New post-terminal bindings; never attribute absent hashes to the worker."""
    attempt, out = Path(attempt).resolve(), Path(out).resolve()
    need(not out.exists(), 'refusing to overwrite terminal timing evidence')
    manifest = json.loads((attempt/'manifest.json').read_text())
    execution = json.loads((attempt/'execution.json').read_text())
    state = json.loads(Path(queue).read_text()); job = state['jobs'].get(manifest['job_id'], {})
    original_lease = state.get('leases', {}).get(manifest.get('lease_id'), {})
    # Never copy queue lease tokens or unrelated private process metadata.
    lease = {key: original_lease.get(key) for key in ('lease_id', 'job_id', 'status', 'gpu_uuids')}
    _terminal_attempt(manifest, execution, job, lease)
    resolver = Resolver([*path_map, *mounts(execution['argv'])])
    completion = binding(attempt/'native-timing/completion.json'); report = resolver.read(completion)
    evidence = dict(schema=EVIDENCE_SCHEMA, created_s=time.time(),
        binding_scope='new_post_terminal_snapshot_not_original_worker_binding',
        attempt_manifest=binding(attempt/'manifest.json'), worker_execution=binding(attempt/'execution.json'),
        completion=completion, timing_stage=report.get('resident_timing_stage'),
        queue_job=job, queue_job_sha256=digest(job), queue_lease=lease, queue_lease_sha256=digest(lease),
        path_map=[list(row) for row in path_map], formal_eligible=False, full_profile_qualified=False)
    _replay_terminal(evidence, resolver)
    return write_new(out, evidence)


def replay_terminal_evidence(reference, *, path_map=()):
    resolver = Resolver(path_map); evidence = resolver.read(reference)
    need(evidence.get('schema') == EVIDENCE_SCHEMA and evidence.get('formal_eligible') is False
         and evidence.get('binding_scope') == 'new_post_terminal_snapshot_not_original_worker_binding',
         'explicit terminal timing-stage evidence kind required')
    result = _replay_terminal(evidence, resolver)
    result['evidence'] = binding(resolver.path(reference['path']))
    return result
