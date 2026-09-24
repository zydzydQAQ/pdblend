"""Explicit observation metering isolation; native controllers remain untouched."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from pdblend.bench import comparison_baseline_observation as obs
from pdblend.bench import comparison_runtime as runtime
from pdblend.bench.resident_session import digest
from test_comparison_baseline_observation import prepared


def isolated_group(point, identity):
    point['metering_execution'] = identity['metering_execution'] = 'isolated_process'
    return dict(engine_identity=identity, points=[point])


@pytest.mark.parametrize('system', ['distserve', 'dynamollm'])
def test_explicit_method_requires_uniform_observation_identity(tmp_path, system):
    point, identity, _ = prepared(tmp_path, system=system)
    group = isolated_group(point, identity)
    assert obs.isolated_observation_metering_requested(group)
    assert runtime.NativeResidentAdapter.isolated_metering_requested(group)
    for change in ('identity', 'point', 'scope', 'policy', 'mixed_system'):
        changed = deepcopy(group)
        if change == 'identity': changed['engine_identity'].pop('metering_execution')
        elif change == 'point': changed['points'][0].pop('metering_execution')
        elif change == 'scope': changed['points'][0].pop('observation_scope')
        elif change == 'policy': changed['points'][0].pop('result_policy')
        else: changed['points'].append(dict(point, system='ecoserve'))
        with pytest.raises(ValueError): obs.isolated_observation_metering_requested(changed)


class Meter:
    def __init__(self, *args):
        self.calls = []; self.active = False; self.stopped = False
    def start(self): self.calls.append('start'); return self
    def begin_window(self):
        assert not self.active; self.active = True; self.calls.append('begin')
    def end_window(self):
        assert self.active; self.active = False; self.calls.append('end')
    def snapshot(self):
        assert not self.active; self.calls.append('snapshot'); return {'samples': []}
    def stop(self, **kwargs):
        assert not self.active; self.calls.append('stop'); self.stopped = True
    def method_receipt(self):
        return dict(window_guard_active=self.active, child_alive=not self.stopped,
                    child_exitcode=0 if self.stopped else None, calls=list(self.calls))


def test_dynamo_starts_isolated_child_before_native_loading_and_records_it(tmp_path, monkeypatch):
    from pdblend.bench import isolated_comparison_meter, comparison_meter_preflight
    adapter = obs.make_dynamo_adapter(tmp_path, base_port=18000)
    point, identity, _ = prepared(tmp_path/'inputs', system='dynamollm')
    adapter.group = isolated_group(point, identity)
    monkeypatch.setattr(isolated_comparison_meter, 'IsolatedComparisonMeter', Meter)
    checked = []
    def preflight(snapshot, method, actual):
        checked.append((snapshot, method, actual)); return {'passed': True}
    monkeypatch.setattr(comparison_meter_preflight, 'qualify_startup_snapshot', preflight)
    async def sleep(_): pass
    monkeypatch.setattr(obs.asyncio, 'sleep', sleep)
    monitor = asyncio.run(adapter._start_monitor(['GPU-'+str(i) for i in range(8)]))
    assert adapter.monitor is monitor and monitor.calls == ['start', 'snapshot']
    assert checked and (tmp_path/'metering-method-startup.json').is_file()
    assert (tmp_path/'metering-startup-preflight.json').is_file()


@pytest.mark.parametrize('system', ['distserve', 'dynamollm'])
def test_native_work_and_drain_are_guarded_but_snapshot_is_outside(tmp_path, monkeypatch, system):
    point, identity, _ = prepared(tmp_path, system=system)
    group = isolated_group(point, identity)
    adapter = runtime.make_resident_adapter(group, tmp_path/'session', base_port=18000)
    adapter.identity = identity; adapter.isolated_metering = True
    adapter.monitor = Meter(); adapter.specs = []
    adapter.reset_receipt = adapter.observation_reset = {'passed': True}
    adapter.qualification = {'startup': 'bound'}
    session_dir = tmp_path/'session'; session_dir.mkdir(exist_ok=True)
    obs.write_new(session_dir/'qualification.json', adapter.qualification)
    raw = {'system': system, 'resident_boundaries': {'after': {'drained': True}}}
    adapter.dynamo_session = SimpleNamespace(observation_reuse={
        'original_native_result_sha256': digest(raw), 'reuse_permitted': True})
    adapter._point = lambda p: None
    async def native(*args, **kwargs):
        assert adapter.monitor.active
        adapter.monitor.calls.append('native')
        return raw
    async def drain(specs):
        assert adapter.monitor.active
        adapter.monitor.calls.append('drain')
        return [{'drained': True}]
    def finalize(*args, **kwargs):
        assert not adapter.monitor.active
        assert kwargs['metering_method']['window_guard_active'] is False
        adapter.monitor.calls.append('finalize')
        return {'metrics': {'duration_s': 150}}
    async def sleep(_): pass
    monkeypatch.setattr(obs, 'execute_native_observation', native)
    monkeypatch.setattr(obs, 'finalize_observation', finalize)
    monkeypatch.setattr(runtime, 'drain_endpoints', drain)
    monkeypatch.setattr(obs.asyncio, 'sleep', sleep)
    for i in range(2):
        out = tmp_path/f'run-{i}'; out.mkdir()
        asyncio.run(adapter.execute(point, out))
    expected = ['begin', 'native'] + (['drain'] if system == 'distserve' else []) + ['end', 'snapshot', 'finalize']
    assert adapter.monitor.calls == expected * 2


def test_dynamo_failure_closes_guard_and_close_reaps_meter(tmp_path, monkeypatch):
    adapter = obs.make_dynamo_adapter(tmp_path, base_port=18000)
    adapter.isolated_metering = True; adapter.monitor = Meter()
    adapter._point = lambda p: None
    async def fail(*args, **kwargs): raise RuntimeError('native failure')
    monkeypatch.setattr(obs, 'execute_native_observation', fail)
    adapter.identity = {}; adapter.dynamo_session = None
    point = {'metering_execution': 'isolated_process'}
    with pytest.raises(RuntimeError, match='native failure'):
        asyncio.run(adapter.execute(point, tmp_path))
    assert not adapter.monitor.active
    assert (tmp_path/'metering-method-failure.json').is_file()
    receipt = asyncio.run(adapter.close())
    assert receipt['passed'] and adapter.monitor.stopped
    assert (tmp_path/'metering-method-cleanup.json').is_file()
    assert asyncio.run(adapter.close()) is receipt
