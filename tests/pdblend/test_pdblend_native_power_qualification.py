"""An observed wrong clock is invalid data for fitting, but not engine failure."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from pdblend.profile.collection.native_power_audit import audit_power_window, PowerFrequencyQualificationError
from test_pdblend_native_power import raw_fixture, UUIDS


def wrong_clock():
    raw = deepcopy(raw_fixture('prefill'))
    raw['spec']['model'] = '/models/Qwen2.5-7B-Instruct'
    raw['capability'].update(model_hash='modelhash', tokenizer_hash='tokenizerhash',
        engine_revision='vllm-test', image_digest='sha256:test', source_revision='sourcehash')
    raw['before'] = deepcopy(raw['drain'])
    raw['before']['native_at_s'] = 99.
    raw['clock'] = dict(ack=dict(acknowledged=True, success=True, requested_frequency_mhz=1500,
        gpus=[dict(gpu_uuid=UUIDS[0])]), observations=[dict(at_s=99.5, frequencies_mhz=[1500])])
    raw['measurement_start']['ranks'] = [dict(rank=0, tp=1, pp=1, acknowledged=True,
        system='pdblend', scope='runner')]
    raw['power']['frequency_samples'][40][1][0] = 1455
    return raw


def test_only_complete_frequency_value_mismatch_raises_typed_gap_and_stays_invalid():
    raw = wrong_clock()
    with pytest.raises(PowerFrequencyQualificationError) as caught:
        audit_power_window(raw)
    value = caught.value.audit
    assert not value['passed'] and not value['power_component_qualified']
    assert not value['formal_eligible'] and not value['full_profile_qualified']
    assert value['non_frequency_checks_passed'] and value['frequency_data_complete']
    assert value['frequency_evidence']['mismatch_count'] == 1
    assert value['frequency_evidence']['mismatches'][0]['observed_mhz'] == 1455
    assert value['public_eight_board_metering']['energy_comparable']


@pytest.mark.parametrize('damage', ['missing_frequency', 'frequency_gap', 'frequency_nan', 'frequency_order',
    'frequency_width', 'foreign_gpu_uuid', 'sampler_error_after_window', 'power_gap', 'power_source',
    'request_error', 'protocol_partial', 'native_rank', 'wrong_model', 'cleanup', 'raw_error', 'drain_generation',
    'clock_ack', 'clock_success', 'clock_setpoint', 'clock_uuid', 'clock_missing_rank', 'clock_never_reached',
    'before_generation', 'before_rank_generation', 'measurement_arm_rank', 'source_missing', 'source_changed'])
def test_other_failure_cannot_hide_behind_simultaneous_wrong_frequency(damage):
    raw = wrong_clock()
    power = raw['power']
    if damage == 'missing_frequency':
        power.pop('frequency_samples')
    elif damage == 'frequency_gap':
        power['frequency_samples'] = power['frequency_samples'][:30] + power['frequency_samples'][50:]
    elif damage == 'frequency_nan':
        power['frequency_samples'][40][1][7] = float('nan')
    elif damage == 'frequency_order':
        power['frequency_samples'][41] = power['frequency_samples'][40]
    elif damage == 'frequency_width':
        power['frequency_samples'][40][1].pop()
    elif damage == 'foreign_gpu_uuid':
        power['gpu_uuids'] = ['GPU-other'] + list(UUIDS[1:])
    elif damage == 'sampler_error_after_window':
        power.update(error='sampler failure', error_at_s=110.)
    elif damage == 'power_gap':
        for key in ('samples', 'power_metadata'):
            power[key] = power[key][:30] + power[key][50:]
    elif damage == 'power_source':
        power['power_source']['field_id'] = 185
    elif damage == 'request_error':
        raw['client_requests'][0]['error'] = 'HTTPError'
    elif damage == 'protocol_partial':
        raw['client_requests'][0]['completion_tokens'] = 0
    elif damage == 'native_rank':
        raw['sample']['ranks'] = []
    elif damage == 'wrong_model':
        raw['capability']['model_id'] = 'different-model'
    elif damage == 'cleanup':
        raw['cleanup_errors'].append('stop failed')
    elif damage == 'raw_error':
        raw['error'] = 'hidden operational error'
    elif damage == 'drain_generation':
        raw['drain']['generation'] += 1
    elif damage == 'clock_ack': raw['clock']['ack']['acknowledged'] = False
    elif damage == 'clock_success': raw['clock']['ack']['success'] = False
    elif damage == 'clock_setpoint': raw['clock']['ack']['requested_frequency_mhz'] = 2520
    elif damage == 'clock_uuid': raw['clock']['ack']['gpus'][0]['gpu_uuid'] = 'GPU-foreign'
    elif damage == 'clock_missing_rank': raw['clock']['ack']['gpus'] = []
    elif damage == 'clock_never_reached': raw['clock']['observations'][0]['frequencies_mhz'] = [1455]
    elif damage == 'before_generation': raw['before']['generation'] += 1
    elif damage == 'before_rank_generation': raw['before']['ranks'][0]['generation'] += 1
    elif damage == 'measurement_arm_rank': raw['measurement_start']['ranks'] = []
    elif damage == 'source_missing': raw['capability'].pop('source_revision')
    expected = deepcopy(raw['capability'])
    if damage == 'source_changed': expected['source_revision'] = 'different-source'
    with pytest.raises((ValueError, KeyError, RuntimeError)) as caught:
        audit_power_window(raw, expected_capability=expected)
    assert not isinstance(caught.value, PowerFrequencyQualificationError)


def test_frequency_tolerance_is_unchanged_and_invalid_window_is_not_relabelled_at_observed_clock():
    raw = wrong_clock()
    raw['power']['frequency_samples'][40][1][0] = 1470
    assert audit_power_window(raw)['passed']
    raw['power']['frequency_samples'][40][1][0] = 1469
    with pytest.raises(PowerFrequencyQualificationError) as caught:
        audit_power_window(raw)
    assert caught.value.audit['frequency_mhz'] == 1500
    assert caught.value.audit['frequency_evidence']['tolerance_mhz'] == 30


@pytest.mark.parametrize('damage', [None, 'missing_clock_rank', 'wrong_clock_rank_uuid',
    'missing_arm_rank', 'stale_initial_rank', 'stale_final_rank'])
def test_tp2_continuation_requires_both_physical_ranks(damage):
    raw = wrong_clock()
    raw['spec'].update(tp=2, gpus=[0, 1], model='/models/Qwen2.5-32B-Instruct')
    raw['capability'].update(tp=2, model_id='Qwen2.5-32B-Instruct', gpu_uuids=UUIDS[:2])
    for name in ('before', 'drain'):
        raw[name]['tp'] = 2
        raw[name]['ranks'].append(dict(deepcopy(raw[name]['ranks'][0]), rank=1))
    for row in raw['sample']['ranks'][0]['samples']: row['tp'] = 2
    other = deepcopy(raw['sample']['ranks'][0]['samples'])
    for row in other: row['rank'] = 1
    raw['sample']['ranks'].append(dict(rank=1, samples=other))
    raw['measurement_start']['ranks'] = [dict(rank=i, tp=2, pp=1, acknowledged=True,
        system='pdblend', scope='runner') for i in range(2)]
    raw['measurement_stop']['ranks'].append(dict(rank=1, acknowledged=True))
    raw['clock']['ack']['gpus'].append(dict(gpu_uuid=UUIDS[1]))
    raw['clock']['observations'][0]['frequencies_mhz'].append(1500)
    for _, clocks in raw['power']['frequency_samples']: clocks[1] = 1500
    if damage == 'missing_clock_rank': raw['clock']['ack']['gpus'].pop()
    elif damage == 'wrong_clock_rank_uuid': raw['clock']['ack']['gpus'][1]['gpu_uuid'] = UUIDS[0]
    elif damage == 'missing_arm_rank': raw['measurement_start']['ranks'].pop()
    elif damage == 'stale_initial_rank': raw['before']['ranks'][1]['generation'] += 1
    elif damage == 'stale_final_rank': raw['drain']['ranks'][1]['generation'] += 1
    with pytest.raises((ValueError, RuntimeError)) as caught:
        audit_power_window(raw, expected_capability=raw['capability'])
    assert isinstance(caught.value, PowerFrequencyQualificationError) is (damage is None)


@pytest.mark.parametrize('fault', [None, 'sampler', 'cleanup', 'lease', 'request', 'frequency_data', 'lookalike_error'])
def test_pilot_can_finish_other_windows_but_timing_requires_independent_safe_restore(tmp_path, monkeypatch, fault):
    from pdblend.profile.collection import native_power_collect as C
    from pdblend_runtime.probe import NativeSpec
    spec = NativeSpec('pd-timing-0', (0,), 20000, '/models/Qwen2.5-7B-Instruct', max_num_seqs=32,
                      extra_args=('--worker-cls', C.WORKER))
    specs = [replace(spec, instance_id=f'pd-timing-{i}', gpus=(i,)) for i in range(8)]
    calls = []
    inventory_calls = []
    class Runner:
        def __init__(self, *args, **kwargs):
            self.capabilities = {}
        async def stop_measurement(self, s):
            calls.append(('stop', s.instance_id))
            return dict(ranks=[dict(rank=0, acknowledged=True)])
        async def capability(self, s):
            return deepcopy(wrong_clock()['capability'])
        async def drain(self, s):
            calls.append(('drain', s.instance_id))
            return dict(acknowledged=True, drained=True)
        async def clock(self, s, f):
            if fault == 'cleanup':
                raise RuntimeError('restore clock failed')
            return dict(requested_frequency_mhz=f)
        async def state(self, s):
            return dict(all_queue=[])
    def inventory(*args):
        inventory_calls.append(1)
        if len(inventory_calls) > 1 and fault in ('sampler', 'lease'):
            raise ValueError('independent restoration inventory or sampler failed')
        return dict(gpu_uuids=UUIDS, gpu_uuid_binding_verified=True)
    async def window(runner, s, point, path):
        raw = wrong_clock() if point['repeat'] == 0 else deepcopy(raw_fixture('prefill'))
        if fault == 'request':
            raw['client_requests'][0]['error'] = 'HTTPError'
        if fault == 'frequency_data':
            raw['power']['frequency_samples'] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
        return raw
    monkeypatch.setattr(C, 'NativeRuntimeCollector', Runner)
    monkeypatch.setattr(C, 'validate_inventory', inventory)
    monkeypatch.setattr(C, 'validate_plan', lambda p: p)
    monkeypatch.setattr(C, 'power_window', window)
    if fault == 'lookalike_error':
        def audit(raw, **kwargs):
            raise ValueError('observed target physical board frequency differs from requested clock')
        monkeypatch.setattr(C, 'audit_power_window', audit)
    plan = dict(model_id='Qwen2.5-7B-Instruct', tp=1, points=[dict(repeats=3)], remaining_gates=['holdout'])
    fleet = {s.instance_id: SimpleNamespace(alive=lambda: True) for s in specs}
    result = asyncio.run(C.collect_power_pilot(specs, fleet, None, None, tmp_path/'pilot', gpu_uuids=UUIDS, plan=plan))
    fatal_window = fault in ('request', 'frequency_data', 'lookalike_error')
    safe = fault not in ('sampler', 'lease', 'cleanup')
    assert result['safe_restore_passed'] is safe
    assert result['operational_failure'] is (fault is not None)
    assert result['ready_for_timing'] is (fault is None)
    assert not result['complete'] and not result['power_component_qualified']
    assert not result['full_profile_qualified'] and not result['formal_eligible']
    assert len(result['windows']) == (1 if fatal_window else 3)
    assert len(result['measurement_qualification_gaps']) == (0 if fatal_window else 1)
    assert len([r for r in calls if r[0] == 'stop']) == 8
    assert len(inventory_calls) == 2
    if fault is None:
        assert result['collection_complete']
        assert result['status'] == 'partial_measurement_qualification'
        assert result['windows'][0]['audit']['passed'] is False
        assert all(r['audit']['passed'] is True for r in result['windows'][1:])
    disk = json.loads((tmp_path/'pilot/completion.json').read_text())
    assert disk == json.loads(json.dumps(result))
