"""CPU fault injection for independent timing and real phase bookkeeping."""
import asyncio
from contextlib import nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from pdblend.profile.collection import native_timing_collect as collector


@pytest.mark.parametrize('fault', [None, 'flag', 'order', 'missing_order', 'old_plan'])
def test_frozen_phase_order_cannot_be_changed_at_launch(fault):
    args = SimpleNamespace(collect_runtime=True, power_pilot_plan='pilot', request_cycle_plan='cycle',
                           layout_energy_plan=None, timing_first=True)
    plan = {'schema': 'pdblend-native-timing-plan/v2'}
    expected = dict(timing_first=True, phase_order=['runtime', 'timing', 'power_pilot', 'request_cycles'])
    if fault == 'flag': expected['timing_first'] = False
    if fault == 'order': expected['phase_order'] = ['runtime', 'power_pilot', 'request_cycles', 'timing']
    if fault == 'missing_order': expected.pop('phase_order')
    if fault == 'old_plan': plan['schema'] = 'pdblend-native-timing-plan-v1'
    if fault:
        with pytest.raises(ValueError): collector.validate_phase_order(args, plan, expected)
    else:
        assert collector.validate_phase_order(args, plan, expected) == expected['phase_order']


def test_legacy_launch_retains_original_order_without_new_inputs():
    args = SimpleNamespace(collect_runtime=True, power_pilot_plan='pilot', request_cycle_plan='cycle')
    assert collector.validate_phase_order(args, {'schema': 'pdblend-native-timing-plan-v1'}, {}) == [
        'runtime', 'power_pilot', 'request_cycles', 'timing']


@pytest.mark.parametrize('first,fault', [(True, None), (True, 'runtime'), (True, 'timing'),
    (True, 'timing_snapshot'), (True, 'power_pilot'), (True, 'request_cycles'),
    (True, 'cleanup'), (False, 'power_pilot')])
def test_later_failure_preserves_completed_timing_without_claiming_job_success(tmp_path, monkeypatch, first, fault):
    from pdblend.engine import launcher
    from pdblend.bench import metering
    from pdblend_runtime import cleanup, probe
    from pdblend_baselines import resident_campaign
    from pdblend.profile.collection import native_runtime_collect, native_power_collect, native_timing_capacity

    out = tmp_path/'native-timing'; out.mkdir()
    point_plan = tmp_path/'points.json'; point_plan.write_text('{}')
    inputs = tmp_path/'inputs.json'; inputs.write_text('{}')
    (tmp_path/'manifest.json').write_text('{}')
    pilot = tmp_path/'pilot.json'; pilot.write_text('{}')
    args = SimpleNamespace(out=out, gpus=list(range(8)), point_plan=point_plan, input_manifest=inputs,
        collect_runtime=True, power_pilot_plan=pilot, request_cycle_plan='cycle',
        layout_energy_plan=None, timing_first=first)
    plan = dict(schema='pdblend-native-timing-plan/v2', capacity_policy={}, points=[],
                holdout_limits={}, required_remaining=['energy'])
    specs = [probe.NativeSpec(f'pd-timing-{i}', (i,), 17000+i, 'Qwen2.5-7B-Instruct') for i in range(8)]
    calls = []
    frequency = [2520]

    class Instance:
        def __init__(self, spec): self.spec, self.events = spec, []
        def start(self):
            self.events.append(dict(kind='start', instance=self.spec.instance_id, pid=123))
        def wait_ready(self, **kwargs): pass

    class Fleet:
        def __init__(self, got, logs): self.instances = {s.instance_id: Instance(s) for s in got}
        def __getitem__(self, name): return self.instances[name]

    class Sampler:
        def start(self): pass

    class Meter:
        def __init__(self, *args, **kwargs): pass
        def sampler(self, **kwargs): return Sampler()
        def current_freq(self, gpu): return frequency[0]

    async def caps(got):
        keys = ('model_id', 'model_hash', 'tokenizer_hash', 'engine_revision', 'source_revision', 'image_digest')
        return {s.instance_id: dict({k: k for k in keys}, tp=1, pp=1, gpu_uuids=[f'GPU-{i}'])
                for i, s in enumerate(got)}

    async def drain(got): return ['fresh all-rank drain']
    async def warmup(*args): return []
    async def no_sleep(seconds): pass
    async def clock(session, url, route, payload):
        assert route == '/baseline/clock'
        frequency[0] = payload['frequency_mhz']
        return dict(acknowledged=True, success=True, requested_frequency_mhz=frequency[0])

    async def runtime(got, fleet, meter, sampler, path, **kwargs):
        calls.append('runtime'); path.mkdir()
        result = dict(ready_for_timing=fault != 'runtime')
        (path/'completion.json').write_text(json.dumps(result)); return result

    async def window(spec, meter, point, path, before_measure=None):
        calls.append('timing')
        if fault == 'timing': raise RuntimeError('timing raw failed')
        return dict(start_s=10., end_s=20., power_samples=[(11., [100.])])

    async def power(got, fleet, meter, sampler, path, **kwargs):
        calls.append('power_pilot'); path.mkdir()
        result = dict(ready_for_timing=fault != 'power_pilot')
        (path/'completion.json').write_text(json.dumps(result)); return result

    async def cycles(args, specs, fleet, meter, sampler, report):
        calls.append('request_cycles')
        path = out/'request-cycles'; path.mkdir()
        (path/'completion.json').write_text(json.dumps(dict(operational_failure=fault == 'request_cycles')))
        if fault == 'request_cycles': raise RuntimeError('cycle raw failed after safe restore')

    def snapshot(report, **kwargs):
        calls.append('timing_snapshot')
        assert report['complete'] is True and report['timing_completed_s'] > 0
        assert report['phase_events'][-1]['phase'] == 'timing_snapshot'
        assert report['phase_events'][-2]['status'] == 'passed'
        if fault == 'timing_snapshot': raise RuntimeError('snapshot replay failed')
        kwargs['out'].write_text('{}'); return collector.binding(kwargs['out'])

    def clean(*args):
        calls.append('cleanup'); return ['actual cleanup failure'] if fault == 'cleanup' else []

    async def empty(*args):
        return dict(passed=fault != 'cleanup', gpu_count=8)

    stage = ModuleType('pdblend.profile.collection.native_timing_stage')
    stage.capture_timing_stage = snapshot
    monkeypatch.setitem(sys.modules, stage.__name__, stage)
    monkeypatch.setenv('PDBLEND_GPU_UUIDS', ','.join(f'GPU-{i}' for i in range(8)))
    monkeypatch.setattr(collector, 'resident_specs', lambda *a: specs)
    monkeypatch.setattr(collector, 'adopt_warmup_generations', lambda a, *rest: a)
    monkeypatch.setattr(launcher, 'Fleet', Fleet)
    monkeypatch.setattr(metering, 'Gpus', Meter)
    monkeypatch.setattr(resident_campaign, 'model_load_lock', nullcontext)
    monkeypatch.setattr(resident_campaign, 'verify_endpoints', caps)
    monkeypatch.setattr(resident_campaign, 'drain_endpoints', drain)
    monkeypatch.setattr(resident_campaign, 'warmup_endpoints', warmup)
    monkeypatch.setattr(collector.asyncio, 'sleep', no_sleep)
    monkeypatch.setattr(probe, 'call', clock)
    monkeypatch.setattr(native_runtime_collect, 'collect_runtime', runtime)
    monkeypatch.setattr(native_power_collect, 'collect_power_pilot', power)
    monkeypatch.setattr(collector, 'collect_cycle_supplement', cycles)
    monkeypatch.setattr(collector, 'window', window)
    monkeypatch.setattr(collector, 'audit_window', lambda *a, **kw: [dict(latency_ms=1.)])
    monkeypatch.setattr(native_timing_capacity, 'partition_windows', lambda *a, **kw: dict(measured=[], unsupported=[]))
    monkeypatch.setattr(native_timing_capacity, 'fit_measured_partition', lambda *a, **kw: dict(component_qualified=True))
    monkeypatch.setattr(cleanup, 'cleanup_owned', clean)
    monkeypatch.setattr(collector, 'verify_compute_empty', empty)
    result = asyncio.run(collector.collect(args, plan))
    assert calls[-1] == 'cleanup' and result['engine_loads'] == 8
    assert result['complete'] is (fault is None)
    assert json.loads((out/'completion.json').read_text()) == json.loads(json.dumps(result))
    if first and fault in (None, 'power_pilot', 'request_cycles', 'cleanup'):
        assert result['resident_timing_stage'] and result['timing_component']
        assert calls.index('timing') < calls.index('timing_snapshot') < calls.index('power_pilot')
    elif fault == 'timing_snapshot':
        assert 'resident_timing_stage' not in result and 'power_pilot' not in calls
    elif fault == 'runtime' or not first:
        assert 'timing' not in calls and 'timing_component' not in result
    if fault and fault != 'cleanup':
        assert result['failed_phase'] == fault
        event = result['phase_events'][-1]
        assert event['status'] == 'failed' and event['finished_s'] >= event['started_s']
    if fault == 'request_cycles':
        assert result['resident_request_cycles'] == collector.binding(out/'request-cycles/completion.json')
