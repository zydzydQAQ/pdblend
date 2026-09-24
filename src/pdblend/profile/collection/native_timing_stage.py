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
OPERATOR_STOP_SCHEMA = 'pdblend-native-timing-operator-stop/v1'
SUPPLEMENTS = {
    'power_pilot': ('resident_power_pilot', 'power-pilot', 'power_pilot_plan'),
    'request_cycles': ('resident_request_cycles', 'request-cycles', 'request_cycle_plan'),
    'layout_energy': ('resident_layout_energy', 'layout-energy', 'layout_energy_plan'),
}


def capture_operator_stop_request(attempt, queue, out, *, reason='user_requested_profile_replan', path_map=()):
    """Record an authorized request before signaling; this function sends no signal.

    The completed timing snapshot is replayed before this receipt is written.
    A receipt alone proves neither cancellation nor safe physical cleanup.
    """
    attempt = Path(attempt).resolve()
    need(reason == 'user_requested_profile_replan', 'explicit user-requested profile replan reason required')
    need(not (attempt/'execution.json').exists() and not (attempt/'native-timing/completion.json').exists(),
         'operator stop cannot be requested after terminal evidence exists')
    manifest_ref = binding(attempt/'manifest.json'); resolver = Resolver(path_map)
    manifest = resolver.read(manifest_ref)
    state = json.loads(Path(queue).read_text()); job = state['jobs'].get(manifest['job_id'], {})
    lease = state.get('leases', {}).get(manifest.get('lease_id'), {})
    need(manifest.get('immutable') is True and job.get('status') == 'running'
         and job.get('attempts') == manifest.get('attempt') and job.get('payload') == manifest.get('payload')
         and job.get('lease_id') == manifest.get('lease_id') == lease.get('lease_id')
         and lease.get('job_id') == manifest['job_id'] and lease.get('status') == 'active'
         and lease.get('gpu_uuids') == manifest.get('gpu_uuids'),
         'operator stop requires the actual running immutable attempt and active lease')
    stage_ref = binding(attempt/'native-timing/timing-stage.json'); stage = resolver.read(stage_ref)
    need(stage.get('attempt_manifest') == manifest_ref
         and stage.get('input_manifest') == manifest['payload']['input_manifest'],
         'operator stop timing snapshot belongs to another attempt/input')
    inputs = resolver.read(stage['input_manifest']); source_ref = inputs['source_manifest']
    source = resolver.read(source_ref)
    need(source.get('source_sha256') == inputs.get('source_sha256') == manifest['payload'].get('source_sha256'),
         'operator stop source identity differs')
    need(any(phase in SUPPLEMENTS for phase in _order(inputs)), 'operator stop has no later optional supplement')
    replay_timing_stage(stage_ref, path_map=path_map)
    current = json.loads(Path(queue).read_text())['jobs'].get(manifest['job_id'], {})
    need(current.get('status') == 'running' and current.get('lease_id') == manifest['lease_id']
         and not (attempt/'execution.json').exists() and not (attempt/'native-timing/completion.json').exists(),
         'operator stop attempt became terminal during timing replay')
    now = time.time()
    need(now >= stage['captured_s'], 'operator stop predates completed timing snapshot')
    return write_new(Path(out), dict(schema=OPERATOR_STOP_SCHEMA, requested_s=now, reason=reason,
        action='SIGINT_after_timing_snapshot', signal_sent_by_this_function=False,
        job_id=manifest['job_id'], attempt=manifest['attempt'], lease_id=manifest['lease_id'],
        attempt_manifest=manifest_ref, input_manifest=stage['input_manifest'], source_manifest=source_ref,
        source_sha256=source['source_sha256'], timing_stage=stage_ref,
        observed_job_status='running', observed_lease_status='active',
        timing_component_only=True, whole_job_success_claimed=False, formal_eligible=False))


def _operator_stop_failure(reference, report, inputs, root, resolver, stage, evidence, manifest, execution):
    request = resolver.read(reference)
    need(request.get('schema') == OPERATOR_STOP_SCHEMA
         and request.get('reason') == 'user_requested_profile_replan'
         and request.get('action') == 'SIGINT_after_timing_snapshot'
         and request.get('signal_sent_by_this_function') is False
         and request.get('timing_component_only') is True
         and request.get('whole_job_success_claimed') is False and request.get('formal_eligible') is False,
         'explicit operator timing-only stop receipt required')
    need(request.get('attempt_manifest') == evidence['attempt_manifest']
         and request.get('input_manifest') == stage['input_manifest']
         and request.get('source_manifest') == inputs['source_manifest']
         and request.get('source_sha256') == inputs['source_sha256'] == manifest['payload']['source_sha256']
         and request.get('timing_stage') == evidence['timing_stage']
         and request.get('job_id') == manifest['job_id'] and request.get('attempt') == manifest['attempt']
         and request.get('lease_id') == manifest['lease_id']
         and request.get('observed_job_status') == 'running' and request.get('observed_lease_status') == 'active',
         'operator stop attempt/input/source/stage binding differs')
    phase = report.get('failed_phase'); event = report['phase_events'][-1]
    snapshot_stop = (phase == 'timing_snapshot' and report.get('error') == 'KeyboardInterrupt()'
                     and not report.get('resident_timing_stage') and evidence.get('operator_stop_delivery'))
    need((snapshot_stop or (phase in SUPPLEMENTS and inputs.get(SUPPLEMENTS[phase][2])))
         and report.get('active_phase') == phase and event['phase'] == phase
         and event['status'] == 'failed' and event.get('error') == report.get('error')
         and (snapshot_stop or event['started_s'] >= stage['captured_s']),
         'operator stop did not occur in a later optional phase or verified snapshot-write race')
    requested = request.get('requested_s')
    need(finite(requested) and execution['started_s'] <= stage['captured_s'] <= requested
         <= event['finished_s'] <= execution['finished_s'], 'operator stop request is retrospective or predates timing')
    if snapshot_stop:
        delivery_ref = evidence['operator_stop_delivery']; delivery = resolver.read(delivery_ref)
        name = manifest['payload'].get('container_name')
        need(isinstance(name, str) and bool(name)
             and delivery.get('request') == reference and delivery.get('stage') == evidence['timing_stage']
             and delivery.get('status') == 'signal_sent' and delivery.get('signal') == 'SIGINT'
             and delivery.get('signal_sent') is True and delivery.get('returncode') == 0
             and delivery.get('whole_job_success_claimed') is False
             and delivery.get('command') == ['docker', 'kill', '--signal=SIGINT', name]
             and delivery.get('stdout', '').strip() == name and delivery.get('stderr') == '',
             'snapshot-write stop lacks the actual bound SIGINT delivery to its container')
        sent, acknowledged = delivery.get('requested_s'), delivery.get('completed_s')
        need(finite(sent) and finite(acknowledged) and requested <= sent <= event['finished_s']
             and sent <= acknowledged <= execution['finished_s'], 'snapshot-write SIGINT delivery timing differs')
        return dict(phase=phase, kind='operator_requested_stop_after_snapshot_write',
            operator_stop_request=reference, operator_stop_delivery=delivery_ref,
            reason=request['reason'], cancellation_error='KeyboardInterrupt()',
            snapshot_assignment_interrupted=True, whole_job_success_claimed=False)
    # Direct asyncio cancellation may occur before a supplement creates its
    # receipt. If the supplement catches cancellation, require its independently
    # validated receipt and the known wrapper failure, rather than any error.
    supplement = None
    if report.get('error') != 'CancelledError()':
        supplement = _later_failure(report, inputs, root, resolver, stage)
        receipt = resolver.read(supplement['completion'])
        need(phase == 'layout_energy'
             and report['error'] == "ValueError('layout-energy supplement failed its collection or safe restoration boundary')",
             'operator stop lacks an observed asyncio cancellation')
        if receipt.get('error') != 'CancelledError()':
            # The existing layout collector catches cancellation first at its
            # raw-window boundary, then emits two operational-failure wrappers.
            # Follow only that bound failure chain; never rescue arbitrary errors.
            child_phase = next((name for name in ('training', 'holdout') if receipt.get('error')
                == f"ValueError('layout {name} operationally incomplete')"), None)
            need(child_phase is not None, 'operator stop layout wrapper is not attributable')
            child_ref = receipt.get(child_phase)
            need(isinstance(child_ref, dict) and resolver.path(child_ref.get('path', ''))
                 == root/'layout-energy'/child_phase/'completion.json', 'operator stop child receipt path differs')
            child = resolver.read(child_ref)
            need(child.get('schema') == 'pdblend-native-layout-energy-collection/v1'
                 and child.get('phase') == child_phase and child.get('plan_sha256') == receipt['plan_sha256']
                 and child.get('collection_complete') is False and child.get('operational_failure') is True
                 and finite(child.get('started_s')) and finite(child.get('finished_s'))
                 and receipt['started_s'] <= child['started_s'] <= child['finished_s'] <= receipt['finished_s']
                 and requested <= child['finished_s'], 'operator stop child collection boundary differs')
            if child.get('error') != 'CancelledError()':
                windows = child.get('windows', [])
                need(windows and 'native layout raw collection invalid:' in child.get('error', ''),
                     'operator stop child lacks an interrupted raw window')
                raw_ref = windows[-1].get('raw')
                need(isinstance(raw_ref, dict) and resolver.path(raw_ref.get('path', '')).parent
                     == root/'layout-energy'/child_phase/'windows', 'operator stop raw window path differs')
                raw = resolver.read(raw_ref); plan = resolver.read(inputs['layout_energy_plan'])
                need(raw.get('schema') == 'pdblend-native-request-cycle-window/v1'
                     and raw.get('system') == 'pdblend' and raw.get('hardware_executed') is True
                     and raw.get('status') == 'failed' and raw.get('error') == 'CancelledError()'
                     and raw.get('plan_sha256') == receipt['plan_sha256']
                     and raw.get('point') in plan.get('points', [])
                     and raw['point'].get('purpose') == child_phase and windows[-1].get('audit', {}).get('passed') is False,
                     'operator stop bound raw window lacks the actual cancellation')
                supplement['cancelled_raw_window'] = raw_ref
            supplement['cancelled_collection'] = child_ref
    return dict(phase=phase, kind='operator_requested_stop_observed_cancellation',
        operator_stop_request=reference, reason=request['reason'],
        cancellation_error='CancelledError()', supplement=supplement,
        whole_job_success_claimed=False)


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
    snapshot_stop = (not stage_ref and report.get('failed_phase') == 'timing_snapshot'
        and report.get('error') == 'KeyboardInterrupt()' and evidence.get('operator_stop_request')
        and evidence.get('operator_stop_delivery'))
    if snapshot_stop:
        stage_ref = resolver.read(evidence['operator_stop_request']).get('timing_stage')
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
    snapshot_status = 'failed' if snapshot_stop else 'passed'
    need(len(events) > index and events[:index] == before[:index]
         and {k: events[index].get(k) for k in before[-1]} == before[-1] | {'status': snapshot_status}
         and events[index]['finished_s'] >= stage['captured_s']
         and ((snapshot_stop and len(events) == index+1 and events[index].get('error') == 'KeyboardInterrupt()')
              or (not snapshot_stop and not events[index].get('error'))),
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
        need(not evidence.get('operator_stop_request'), 'operator stop cannot relabel whole-job success')
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
        if evidence.get('operator_stop_request'):
            failure = _operator_stop_failure(evidence['operator_stop_request'], report, inputs, root,
                resolver, stage, evidence, manifest, execution)
        else:
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


def capture_terminal_evidence(attempt, queue, out, *, path_map=(), operator_stop_ref=None, operator_delivery_ref=None):
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
    if operator_stop_ref is not None:
        evidence['operator_stop_request'] = operator_stop_ref
    if operator_delivery_ref is not None:
        need(operator_stop_ref is not None, 'operator signal delivery requires its prior stop request')
        evidence['operator_stop_delivery'] = operator_delivery_ref
        if evidence['timing_stage'] is None:
            evidence['timing_stage'] = resolver.read(operator_stop_ref).get('timing_stage')
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
