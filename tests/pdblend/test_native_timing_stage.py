"""No GPU: completed native raw timing survives only an attributable later failure."""
from copy import deepcopy
import json
from pathlib import Path
from dataclasses import asdict
import shutil

import pytest

from pdblend.profile.collection import native_timing_stage as stage
from pdblend.profile.collection.native_timing_plan import binding, digest
from test_native_timing_replay_v2 import collected_v2, collected, plans
from test_native_timing_replay import put
from test_native_layout_stage import resident
from test_comparison_acceptance import state


@pytest.fixture
def staged(collected_v2, monkeypatch):
    return _stage(collected_v2, monkeypatch)


def _stage(x, monkeypatch):
    report, specs, fleet = resident(x)
    end = max(r['received_s'] for r in report['final_drains']) + .1
    inputs = json.loads(Path(x.inputs_ref['path']).read_text())
    inputs.update(timing_first=True, phase_order=['timing', 'request_cycles'],
                  request_cycle_plan=put(x.tmp/'cycle-plan.json', {'immutable_training_design': True}))
    x.inputs_ref = put(Path(x.inputs_ref['path']), inputs)
    manifest = json.loads((x.attempt/'manifest.json').read_text())
    manifest.update(lease_id='stage-lease')
    manifest['payload']['input_manifest'] = x.inputs_ref
    manifest['payload']['argv'].append('--timing-first')
    manifest_ref = put(x.attempt/'manifest.json', manifest)
    report.update(phase_order=inputs['phase_order'], active_phase='timing_snapshot', timing_completed_s=end,
        phase_events=[dict(phase='startup', started_s=800., finished_s=950., status='passed'),
                      dict(phase='timing', started_s=960., finished_s=end, status='passed'),
                      dict(phase='timing_snapshot', started_s=end+.1, status='running')])
    monkeypatch.setattr(stage.time, 'time', lambda: end+.2)
    stage_ref = stage.capture_timing_stage(report, input_manifest_ref=x.inputs_ref,
        attempt_manifest_ref=manifest_ref, specs=specs, fleet=fleet, out=x.root/'timing-stage.json')
    final = deepcopy(report)
    final.update(resident_timing_stage=stage_ref, active_phase='request_cycles', failed_phase='request_cycles',
        status='failed', complete=False, error="ValueError('request-cycle supplement failed')", hardware_executed=True,
        actual_engine_starts=x.v2_complete['actual_engine_starts'], engine_loads=4)
    final['phase_events'][-1].update(finished_s=end+.3, status='passed')
    final['phase_events'].append(dict(phase='request_cycles', started_s=end+.4, finished_s=end+2.,
                                     status='failed', error=final['error']))
    final['resident_request_cycles'] = put(x.root/'request-cycles/completion.json',
        dict(schema='pdblend-native-request-cycle-revision/v1', plan_sha256=digest({'immutable_training_design': True}),
             started_s=end+.5, finished_s=end+1.9, status='failed', error='observed frequency differs',
             operational_failure=True, safe_restore_passed=True, formal_eligible=False))
    final['physical_cleanup'] = deepcopy(x.v2_complete['physical_cleanup'])
    final['physical_cleanup'].update(started_s=end+3., finished_s=end+3.2)
    final['physical_cleanup']['observations'][0]['at_s'] = end+3.1
    complete_ref = put(x.root/'completion.json', final)
    execution = dict(status='failed', complete=False, returncode=2, error='RuntimeError: process exited 2',
        started_s=700., finished_s=end+4., argv=['docker', 'run', '-v', str(x.attempt)+':/output:rw'],
        receipt_sha256={})  # A real nonzero worker does not run receipt().
    put(x.attempt/'execution.json', execution)
    queue = dict(jobs={'timing-job': dict(job_id='timing-job', attempts=1, payload=manifest['payload'],
        status='failed', lease_id=None)}, leases={'stage-lease': dict(lease_id='stage-lease', job_id='timing-job',
        status='failed', gpu_uuids=manifest['gpu_uuids'], token='DO-NOT-COPY')})
    put(x.queue, queue)
    x.stage_ref = stage_ref; x.final = final; x.execution = execution; x.queue_value = queue; x.end = end
    return x


def test_terminal_failed_supplement_reuses_exact_timing_without_relabeling_parent(staged):
    x = staged
    ref = stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'terminal.json')
    result = stage.replay_terminal_evidence(ref)
    assert result['supported_fit'] == x.v2_fitted
    assert result['component_qualified'] and result['timing_component_reusable']
    assert result['parent_job_status'] == 'failed' and not result['parent_job_succeeded']
    assert not result['worker_execution_complete'] and not result['formal_eligible']
    assert result['physical_cleanup_verified'] and result['queue_terminal_verified']
    assert result['later_independent_failure']['phase'] == 'request_cycles'
    assert result['later_independent_failure']['safe_restore_passed'] is True
    assert 'DO-NOT-COPY' not in Path(ref['path']).read_text()
    assert json.loads((x.attempt/'execution.json').read_text()) == x.execution
    assert json.loads((x.root/'completion.json').read_text()) == x.final
    with pytest.raises(ValueError, match='overwrite'):
        stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'terminal.json')
    # The original whole-job gate continues to reject this same attempt.
    from pdblend.profile.collection.native_timing_replay_v2 import capture_evidence
    with pytest.raises(ValueError, match='completed immutable'):
        capture_evidence(x.attempt, x.queue, x.tmp/'legacy.json')


def _operator_request(x, monkeypatch):
    queue = deepcopy(x.queue_value)
    queue['jobs']['timing-job'].update(status='running', lease_id='stage-lease')
    queue['leases']['stage-lease']['status'] = 'active'
    put(x.queue, queue)
    (x.attempt/'execution.json').unlink(); (x.root/'completion.json').unlink()
    monkeypatch.setattr(stage.time, 'time', lambda: x.end+.6)
    try:
        return stage.capture_operator_stop_request(x.attempt, x.queue, x.tmp/'operator-stop.json')
    finally:
        put(x.attempt/'execution.json', x.execution); put(x.root/'completion.json', x.final)
        put(x.queue, x.queue_value)


def test_operator_stop_preserves_completed_timing_without_supplement_receipt(staged, monkeypatch):
    x = staged; request = _operator_request(x, monkeypatch)
    final = deepcopy(x.final); final.pop('resident_request_cycles')
    final['error'] = final['phase_events'][-1]['error'] = 'CancelledError()'
    put(x.root/'completion.json', final)
    ref = stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'stopped.json', operator_stop_ref=request)
    result = stage.replay_terminal_evidence(ref)
    assert result['timing_component_reusable'] and result['supported_fit'] == x.v2_fitted
    assert result['parent_job_status'] == 'failed' and not result['parent_job_succeeded']
    assert result['physical_cleanup_verified'] and result['queue_terminal_verified']
    assert result['later_independent_failure']['kind'] == 'operator_requested_stop_observed_cancellation'
    assert not result['whole_job_result_modified'] and not result['full_profile_qualified']
    assert not result['formal_eligible'] and not result['auxiliary_power_qualifies_power_component']
    assert 'DO-NOT-COPY' not in Path(request['path']).read_text()
    assert json.loads((x.attempt/'execution.json').read_text()) == x.execution
    with pytest.raises(ValueError, match='actual bound completion'):
        stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'no-request.json')


def test_operator_stop_rejects_rebound_wrong_identity_late_request_or_unsafe_cleanup(staged, monkeypatch):
    x = staged; request = _operator_request(x, monkeypatch)
    original = json.loads(Path(request['path']).read_text())
    final = deepcopy(x.final); final.pop('resident_request_cycles')
    final['error'] = final['phase_events'][-1]['error'] = 'CancelledError()'
    for index, mutation in enumerate(('source','input','stage','attempt','lease','reason','retrospective',
                                      'before_stage','wrong_error','wrong_phase','busy_cleanup','killed')):
        value = deepcopy(original); changed = deepcopy(final); execution = deepcopy(x.execution)
        if mutation in ('source','input','stage','attempt'):
            key = {'source':'source_manifest','input':'input_manifest','stage':'timing_stage','attempt':'attempt_manifest'}[mutation]
            value[key]['sha256'] = '0'*64
        elif mutation == 'lease': value['lease_id'] = 'another-lease'
        elif mutation == 'reason': value['reason'] = 'generic_failure_rescue'
        elif mutation == 'retrospective': value['requested_s'] = x.end+2.1
        elif mutation == 'before_stage': value['requested_s'] = x.end
        elif mutation == 'wrong_error': changed['error'] = changed['phase_events'][-1]['error'] = "RuntimeError('other failure')"
        elif mutation == 'wrong_phase': changed['failed_phase'] = 'timing'
        elif mutation == 'busy_cleanup': changed['physical_cleanup']['observations'][0]['devices'][0]['compute_pids'] = [55]
        else: execution.update(returncode=-9, error='RuntimeError: process exited -9')
        changed_ref = put(x.tmp/f'operator-mutation-{index}.json', value)
        put(x.root/'completion.json', changed); put(x.attempt/'execution.json', execution)
        with pytest.raises(ValueError):
            stage.capture_terminal_evidence(x.attempt,x.queue,x.tmp/f'stop-rejected-{index}.json',operator_stop_ref=changed_ref)
        assert not (x.tmp/f'stop-rejected-{index}.json').exists()


def test_operator_request_is_preterminal_and_replays_original_raw(staged, monkeypatch):
    x = staged
    with pytest.raises(ValueError, match='after terminal'):
        stage.capture_operator_stop_request(x.attempt, x.queue, x.tmp/'too-late.json')
    sample = next((x.root/'samples').glob('*.json')); sample.write_bytes(sample.read_bytes()+b' ')
    with pytest.raises(ValueError, match='checksum|bytes'):
        _operator_request(x, monkeypatch)
    assert not (x.tmp/'operator-stop.json').exists()


def test_operator_stop_snapshot_assignment_race_requires_actual_sigint_delivery(staged, monkeypatch):
    x = staged
    manifest=json.loads((x.attempt/'manifest.json').read_text())
    manifest['payload']['container_name']='timing-job'
    manifest_ref=put(x.attempt/'manifest.json',manifest)
    document=json.loads(Path(x.stage_ref['path']).read_text());document['attempt_manifest']=manifest_ref
    x.stage_ref=put(Path(x.stage_ref['path']),document);x.final['resident_timing_stage']=x.stage_ref
    x.queue_value['jobs']['timing-job']['payload']=manifest['payload']
    request=_operator_request(x,monkeypatch)
    final=deepcopy(x.final);final.pop('resident_timing_stage');final.pop('resident_request_cycles')
    final.update(active_phase='timing_snapshot',failed_phase='timing_snapshot',error='KeyboardInterrupt()')
    final['phase_events']=final['phase_events'][:3]
    final['phase_events'][-1].update(status='failed',finished_s=x.end+.8,error='KeyboardInterrupt()')
    put(x.root/'completion.json',final)
    delivery=dict(request=request,stage=x.stage_ref,status='signal_sent',signal='SIGINT',signal_sent=True,
        returncode=0,whole_job_success_claimed=False,command=['docker','kill','--signal=SIGINT','timing-job'],
        stdout='timing-job\n',stderr='',requested_s=x.end+.65,completed_s=x.end+.7)
    delivery_ref=put(x.tmp/'signal-delivery.json',delivery)
    ref=stage.capture_terminal_evidence(x.attempt,x.queue,x.tmp/'snapshot-stopped.json',
        operator_stop_ref=request,operator_delivery_ref=delivery_ref)
    result=stage.replay_terminal_evidence(ref)
    assert result['timing_component_reusable'] and not result['parent_job_succeeded']
    assert result['later_independent_failure']['snapshot_assignment_interrupted']
    assert result['physical_cleanup_verified'] and not result['whole_job_result_modified']
    assert json.loads((x.root/'completion.json').read_text())==final
    with pytest.raises(ValueError):
        stage.capture_terminal_evidence(x.attempt,x.queue,x.tmp/'missing-delivery.json',operator_stop_ref=request)
    for index, mutation in enumerate(('request','stage','container','signal','not_sent','returncode','late','ordinary_error')):
        changed=deepcopy(delivery);changed_final=deepcopy(final)
        if mutation in ('request','stage'):changed[mutation]['sha256']='0'*64
        elif mutation=='container':changed['command'][-1]='different-container'
        elif mutation=='signal':changed['signal']='SIGTERM'
        elif mutation=='not_sent':changed['signal_sent']=False
        elif mutation=='returncode':changed['returncode']=1
        elif mutation=='late':changed['requested_s']=x.end+.9
        else:changed_final['error']=changed_final['phase_events'][-1]['error']="ValueError('snapshot audit failed')"
        changed_ref=put(x.tmp/f'wrong-delivery-{index}.json',changed);put(x.root/'completion.json',changed_final)
        with pytest.raises(ValueError):
            stage.capture_terminal_evidence(x.attempt,x.queue,x.tmp/f'race-rejected-{index}.json',
                operator_stop_ref=request,operator_delivery_ref=changed_ref)


@pytest.mark.parametrize('layer', ['revision', 'collection', 'raw_window'])
def test_operator_stop_follows_layout_cancellation_wrappers(staged, monkeypatch, layer):
    x = staged; plan = {'points': [{'purpose': 'training', 'frequency_mhz': 1500}]}
    inputs = json.loads(Path(x.inputs_ref['path']).read_text())
    inputs.pop('request_cycle_plan'); inputs.update(phase_order=['timing','layout_energy'],
        layout_energy_plan=put(x.tmp/'layout-plan.json', plan))
    x.inputs_ref = put(Path(x.inputs_ref['path']), inputs)
    manifest = json.loads((x.attempt/'manifest.json').read_text()); manifest['payload']['input_manifest'] = x.inputs_ref
    manifest_ref = put(x.attempt/'manifest.json', manifest)
    document = json.loads(Path(x.stage_ref['path']).read_text()); saved = json.loads(Path(document['report']['path']).read_text())
    saved['phase_order'] = inputs['phase_order']
    document.update(input_manifest=x.inputs_ref, attempt_manifest=manifest_ref,
        report=put(Path(document['report']['path']), saved))
    x.stage_ref = put(Path(x.stage_ref['path']), document)
    x.final.update(phase_order=inputs['phase_order'], resident_timing_stage=x.stage_ref,
        active_phase='layout_energy', failed_phase='layout_energy')
    x.final.pop('resident_request_cycles'); x.final['phase_events'][-1]['phase'] = 'layout_energy'
    x.final['error'] = x.final['phase_events'][-1]['error'] = (
        "ValueError('layout-energy supplement failed its collection or safe restoration boundary')")
    revision = dict(schema='pdblend-native-layout-revision/v1',plan_sha256=digest(plan),
        started_s=x.end+.5,finished_s=x.end+1.9,status='failed',error='CancelledError()',
        operational_failure=True,safe_restore_passed=True)
    if layer != 'revision':
        child = dict(schema='pdblend-native-layout-energy-collection/v1',phase='training',
            plan_sha256=digest(plan),started_s=x.end+.6,finished_s=x.end+1.8,
            collection_complete=False,operational_failure=True,error='CancelledError()')
        if layer == 'raw_window':
            raw = dict(schema='pdblend-native-request-cycle-window/v1',system='pdblend',
                hardware_executed=True,status='failed',error='CancelledError()',plan_sha256=digest(plan),point=plan['points'][0])
            child['windows'] = [dict(raw=put(x.root/'layout-energy/training/windows/00.json',raw),audit={'passed':False})]
            child['error'] = "ValueError('native layout raw collection invalid: incomplete')"
        revision.update(error="ValueError('layout training operationally incomplete')",
            training=put(x.root/'layout-energy/training/completion.json',child))
    x.final['resident_layout_energy'] = put(x.root/'layout-energy/completion.json', revision)
    x.queue_value['jobs']['timing-job']['payload'] = manifest['payload']
    request = _operator_request(x,monkeypatch)
    ref = stage.capture_terminal_evidence(x.attempt,x.queue,x.tmp/'layout-stop.json',operator_stop_ref=request)
    result = stage.replay_terminal_evidence(ref)
    assert result['timing_component_reusable'] and not result['parent_job_succeeded']
    assert result['later_independent_failure']['kind'] == 'operator_requested_stop_observed_cancellation'
    if layer == 'raw_window':
        raw['error'] = "RuntimeError('unrelated measurement failure')"
        child['windows'][0]['raw'] = put(x.root/'layout-energy/training/windows/00.json',raw)
        revision['training'] = put(x.root/'layout-energy/training/completion.json',child)
        x.final['resident_layout_energy'] = put(x.root/'layout-energy/completion.json',revision)
        put(x.root/'completion.json',x.final)
        with pytest.raises(ValueError,match='actual cancellation'):
            stage.capture_terminal_evidence(x.attempt,x.queue,x.tmp/'not-cancellation.json',operator_stop_ref=request)



def test_terminal_cannot_add_or_change_frequency_identity_after_frozen_stage(staged):
    x=staged
    for index,key in enumerate(('model_id','tp','pp','frequency_domain_ref','frequency_domain','frequency_domain_sha256')):
        changed=deepcopy(x.final);changed[key]='changed-after-timing'
        put(x.root/'completion.json',changed)
        with pytest.raises(ValueError,match='rewrote timing evidence'):
            stage.capture_terminal_evidence(x.attempt,x.queue,x.tmp/f'domain-drift-{index}.json')
    put(x.root/'completion.json',x.final)


def test_successful_terminal_stage_still_requires_worker_binding(staged):
    x = staged; final = deepcopy(x.final); execution = deepcopy(x.execution); queue = deepcopy(x.queue_value)
    final.update(status='passed', complete=True, active_phase='completed')
    final.pop('failed_phase'); final.pop('error')
    final['phase_events'][-1].update(status='passed'); final['phase_events'][-1].pop('error')
    ref = put(x.root/'completion.json', final)
    execution.update(status='passed', complete=True, returncode=0, error=None,
                     receipt_sha256={'native-timing/completion.json': ref['sha256']})
    put(x.attempt/'execution.json', execution)
    queue['jobs']['timing-job']['status'] = queue['leases']['stage-lease']['status'] = 'succeeded'
    put(x.queue, queue)
    result = stage.replay_terminal_evidence(stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'passed.json'))
    assert result['parent_job_succeeded'] and result['worker_execution_complete']
    assert result['later_independent_failure'] is None
    execution['receipt_sha256'] = {}; put(x.attempt/'execution.json', execution)
    with pytest.raises(ValueError, match='worker did not bind'):
        stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'unbound.json')


def test_rehashed_claims_cannot_rescue_wrong_phase_parent_or_cleanup(staged):
    x = staged
    for index, mutation in enumerate(('early_failure', 'snapshot_failed', 'phase_overlap', 'changed_prefix',
        'unbound_failure', 'wrong_plan', 'failure_predates_stage', 'gpu_busy', 'cleanup_early', 'start_lineage',
        'tampered_timing', 'running_job', 'active_lease', 'worker_killed', 'false_success')):
        final = deepcopy(x.final); execution = deepcopy(x.execution); queue = deepcopy(x.queue_value)
        rpath = Path(final['resident_request_cycles']['path']); rbytes = rpath.read_bytes()
        if mutation == 'early_failure': final['failed_phase'] = 'timing'
        elif mutation == 'snapshot_failed':
            final['phase_events'][2].update(status='failed', error='snapshot raw invalid')
        elif mutation == 'phase_overlap': final['phase_events'][-1]['started_s'] = x.end
        elif mutation == 'changed_prefix': final['phase_events'][0]['finished_s'] -= 1
        elif mutation == 'unbound_failure': final.pop('resident_request_cycles')
        elif mutation in ('wrong_plan', 'failure_predates_stage'):
            value = json.loads(rbytes)
            if mutation == 'wrong_plan': value['plan_sha256'] = '0'*64
            else: value['started_s'] = x.end-1
            final['resident_request_cycles'] = put(rpath, value)
        elif mutation == 'gpu_busy': final['physical_cleanup']['observations'][-1]['devices'][0]['compute_pids'] = [55]
        elif mutation == 'cleanup_early': final['physical_cleanup']['started_s'] = x.end
        elif mutation == 'start_lineage': final['actual_engine_starts']['pd-timing-0'][0]['pid'] += 1
        elif mutation == 'tampered_timing': final['measured_windows'] -= 1
        elif mutation == 'running_job': queue['jobs']['timing-job']['status'] = 'running'
        elif mutation == 'active_lease': queue['leases']['stage-lease']['status'] = 'active'
        elif mutation == 'worker_killed': execution.update(returncode=-9, error='RuntimeError: process exited -9')
        else: final.update(status='passed', complete=True)
        put(x.root/'completion.json', final); put(x.attempt/'execution.json', execution); put(x.queue, queue)
        try:
            with pytest.raises((ValueError, KeyError)):
                stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/f'rejected-{index}.json')
            assert not (x.tmp/f'rejected-{index}.json').exists()
        finally:
            rpath.write_bytes(rbytes)
    put(x.root/'completion.json', x.final); put(x.attempt/'execution.json', x.execution); put(x.queue, x.queue_value)


def test_raw_source_and_interference_still_replay_independently(staged):
    x = staged
    ref = stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'terminal.json')
    for path in [x.tmp/'source/test.py', next((x.root/'samples').glob('*.json')),
                 next((x.root/'interference').glob('*.json'))]:
        saved = path.read_bytes(); path.write_bytes(saved+b' ')
        try:
            with pytest.raises(ValueError, match='checksum|bytes'):
                stage.replay_terminal_evidence(ref)
        finally: path.write_bytes(saved)


def test_power_and_layout_failures_bind_their_real_plan_semantics(staged):
    x = staged
    for phase in ('power_pilot', 'layout_energy'):
        field, directory, key = stage.SUPPLEMENTS[phase]
        original_inputs = Path(x.inputs_ref['path']).read_bytes()
        original_manifest = (x.attempt/'manifest.json').read_bytes()
        old_stage = json.loads(Path(x.stage_ref['path']).read_text())
        old_saved = json.loads(Path(old_stage['report']['path']).read_text())
        inputs = json.loads(original_inputs); plan = {'immutable_training_design': True}
        inputs.pop('request_cycle_plan'); inputs[key] = put(x.tmp/(phase+'-plan.json'), plan)
        inputs['phase_order'] = ['timing', phase]
        input_ref = put(Path(x.inputs_ref['path']), inputs)
        manifest = json.loads(original_manifest); manifest['payload']['input_manifest'] = input_ref
        manifest_ref = put(x.attempt/'manifest.json', manifest)
        saved = deepcopy(old_saved); saved['phase_order'] = inputs['phase_order']
        stage_document = deepcopy(old_stage)
        stage_document.update(input_manifest=input_ref, attempt_manifest=manifest_ref,
            report=put(Path(old_stage['report']['path']), saved))
        stage_ref = put(Path(x.stage_ref['path']), stage_document)
        final = deepcopy(x.final)
        final.update(phase_order=inputs['phase_order'], resident_timing_stage=stage_ref,
                     failed_phase=phase, active_phase=phase)
        final['phase_events'][-1]['phase'] = phase
        final.pop('resident_request_cycles')
        receipt = dict(status='failed', error='actual supplement qualification failure', operational_failure=True)
        if phase == 'power_pilot':
            receipt.update(schema='pdblend-native-power-pilot/v1', plan=plan,
                lease={'checked_s': x.end+.5}, restoration={'instances': [
                    {'capability': {'state': {'response_at_s': x.end+1.9}}}]})
        else:
            receipt.update(schema='pdblend-native-layout-revision/v1', plan_sha256=digest(plan),
                           started_s=x.end+.5, finished_s=x.end+1.9)
        final[field] = put(x.root/directory/'completion.json', receipt)
        put(x.root/'completion.json', final)
        queue = deepcopy(x.queue_value); queue['jobs']['timing-job']['payload'] = manifest['payload']; put(x.queue, queue)
        result = stage.replay_terminal_evidence(stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/(phase+'.json')))
        assert result['later_independent_failure']['phase'] == phase and result['timing_component_reusable']
        # Restore before checking the other explicitly named supplement.
        Path(x.inputs_ref['path']).write_bytes(original_inputs); (x.attempt/'manifest.json').write_bytes(original_manifest)
        put(Path(old_stage['report']['path']), old_saved); put(Path(x.stage_ref['path']), old_stage)


def test_model_owned_tp1_eight_instance_stage_uses_the_same_strict_replay(collected, plans, monkeypatch):
    from pdblend.profile.collection.native_timing_capacity import (
        capacity_decision, unsupported_capacity_record, partition_windows, fit_measured_partition)
    from pdblend_runtime.probe import NativeSpec
    x = collected; plan = plans['plans']['Qwen2.5-7B-Instruct']; ids = tuple(x.caps)
    inputs = json.loads(Path(x.inputs_ref['path']).read_text())
    inputs.update(schema='pdblend-native-timing-inputs/v2', point_plan=put(x.tmp/'v2-plan.json', plan))
    x.inputs_ref = put(Path(x.inputs_ref['path']), inputs)
    peers = []
    for f in (1500, 2520):
        start = min(json.loads(p.read_text())['measurement_started_s']
                    for p in (x.root/'interference').glob(f'{f}-*-isolated.json')) - 4.
        peers.append(dict(frequency_mhz=f, drains=[dict(instance_id=iid, received_s=start,
            drain=dict(state(1, start, 5), acknowledged=True, drained=True), state=state(1, start, 5)) for iid in ids],
            clocks=[dict(instance_id=iid, clock=dict(acknowledged=True, success=True,
                requested_frequency_mhz=f, at_s=start+.1, gpus=[dict(gpu_uuid=u, frequency_mhz=f)
                for u in x.caps[iid]['gpu_uuids']])) for iid in ids],
            observations=[dict(at_s=start+.2, frequencies_mhz=[f]*8)],
            settle_started_s=start+.3, observed_s=start+2.4))
    qualification_ref = binding(x.root/'measurement-qualification.json')
    qualification = json.loads(Path(qualification_ref['path']).read_text())
    shutil.rmtree(x.root/'samples')
    refs = []; owners = []; windows = []; now = 3000.
    for index, point in enumerate(plan['points']):
        owner = index % 8; iid = ids[owner]
        for repeat in range(3):
            p = dict(point, repeat=repeat); name = f'{digest(point)[:20]}-{repeat}'
            idle = dict(state(1, now, 5), total_kv_tokens=16384, free_kv_tokens=16384)
            args = dict(identity=x.caps[iid], capability=dict(x.caps[iid], supported=True, state=deepcopy(idle)),
                capability_received_s=now, drain=dict(idle, acknowledged=True, drained=True),
                drain_received_s=now, observed_s=now+.1)
            if capacity_decision(plan, p, **args)['supported']:
                value = x.raw_factory(p, owner, now, name); now += 12.
            else:
                value = unsupported_capacity_record(plan, p, **args); now += 1.
            ref = put(x.root/'samples'/(name+'.json'), value)
            refs.append(ref); owners.append(dict(instance_id=iid, raw=ref)); windows.append(dict(instance_id=iid, raw=value))
    partition = partition_windows(plan, iter(windows), identities=x.caps)
    fitted = fit_measured_partition(partition, identity=dict(system='pdblend', **x.identity), raw_bindings=refs,
        measurement_qualification=dict(qualification, receipt=qualification_ref), limits=plan['holdout_limits'])
    specs = [NativeSpec(iid, (i,), 20000+i*4, '/models/'+plan['model_id'], max_num_seqs=32, generation=5,
        extra_args=('--enforce-eager', '--worker-cls',
            'pdblend.profile.collection.native_timing_worker.PDNativeTimingWorker')) for i, iid in enumerate(ids)]
    complete = dict(schema='pdblend-native-timing-collection/v2', system='pdblend', status='passed', complete=True,
        hardware_executed=True, cleanup_errors=[], point_plan=inputs['point_plan'], capacity_policy=plan['capacity_policy'],
        capabilities={iid: dict(cap, supported=True, state=state(1, 900., 5)) for iid, cap in x.caps.items()},
        raw_bindings=refs, window_owners=owners, interference_peer_states=peers,
        timing_component=put(x.root/'timing-component.json', fitted), component_qualified=fitted['component_qualified'],
        measurement_qualification=qualification_ref, window_partition=put(x.root/'window-partition.json', partition),
        measured_windows=len(partition['measured']), unsupported_windows=len(partition['unsupported']),
        actual_launch=[dict(spec=json.loads(json.dumps(asdict(s))), argv=s.command(), environment=dict(
            CUDA_VISIBLE_DEVICES=str(i), VLLM_USE_V1='1', NCCL_CUMEM_ENABLE='0', NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='0'))
            for i, s in enumerate(specs)],
        final_drains=[dict(instance_id=iid, received_s=now+1., drain=dict(state(1, now+1., 5), acknowledged=True,
            drained=True), state=state(1, now+1., 5)) for iid in ids],
        actual_engine_starts={iid: [dict(instance=iid, kind='start', pid=1000+i, t_s=900.)] for i, iid in enumerate(ids)},
        engine_loads=8, physical_cleanup=dict(passed=True, started_s=now+2., finished_s=now+2.2,
            observations=[dict(at_s=now+2.1, devices=[dict(gpu=i, gpu_uuid=f'GPU-{i}', compute_pids=[]) for i in range(8)])]))
    x.v2_complete = complete; x.v2_fitted = fitted
    x = _stage(x, monkeypatch)
    # _stage's fixture default is TP2; use the real TP1 inventory count here.
    x.final['engine_loads'] = 8; put(x.root/'completion.json', x.final)
    ref = stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'tp1-terminal.json')
    result = stage.replay_terminal_evidence(ref)
    assert result['identity']['tp'] == 1 and result['resident_boundary']['live_instances'] == 8
    assert result['replayed_interference_windows'] == 96 and result['supported_fit'] == fitted
    assert result['timing_component_reusable'] and not result['formal_eligible']
