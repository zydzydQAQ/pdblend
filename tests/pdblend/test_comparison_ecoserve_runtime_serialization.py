import asyncio
import json
from types import SimpleNamespace

import pytest

from pdblend.bench import comparison_runtime as runtime
from pdblend.bench import comparison_ecoserve_acceptance as acceptance
from pdblend_baselines.ecoserve import run_native


@pytest.mark.parametrize('isolated', [False, True])
def test_ecoserve_audits_frozen_native_result_with_real_integer_gpu_keys(tmp_path, monkeypatch, isolated):
    """The native transport uses integer GPU IDs; JSON evidence uses strings."""
    raw = dict(started_s=100., gpu_uuids={0: 'GPU-a', 1: 'GPU-b'})
    trace = dict(requests=[])
    trace_path = tmp_path / 'trace.json'
    runtime.write_new(trace_path, trace)
    point = dict(name='eco', system='ecoserve', qualification_mode='ecoserve_native_bootstrap',
                 model_id='Qwen2.5-7B-Instruct',
                 trace=runtime.binding(trace_path), slo=dict(ttft_s=15., tpot_s=.2))
    out = tmp_path / 'window' / 'run'
    out.mkdir(parents=True)
    adapter = runtime.NativeResidentAdapter.__new__(runtime.NativeResidentAdapter)
    adapter.out = tmp_path
    adapter.specs = [SimpleNamespace(instance_id='e0', base_url='http://native')]
    adapter.ecoserve_inputs = {'eco': dict(config={})}
    adapter.identity = dict(fleet_gpu_uuids=['GPU-a', 'GPU-b'], model_hash='model',
                            tokenizer_hash='tokenizer', image_digest='image',
                            runtime_source_sha256='runtime', measurement_source_sha256='measure')
    adapter.qualification = dict(passed=True)
    adapter.reset_receipt = dict(passed=True)
    runtime.write_new(tmp_path / 'qualification.json', adapter.qualification)
    adapter.isolated_metering = isolated
    adapter.metering_cleanup_errors = []
    operations = []
    if isolated:
        point['metering_execution'] = 'isolated_process'
        class Meter:
            active = False
            def begin_window(self): self.active=True;operations.append('begin')
            def end_window(self): self.active=False;operations.append('end')
            def snapshot(self):
                assert not self.active
                operations.append('snapshot')
                return []
            def method_receipt(self): return dict(window_guard_active=self.active, status='running')
        adapter.monitor = Meter()
    else:
        adapter.monitor = SimpleNamespace(snapshot=lambda: [])
    monkeypatch.setenv('PDBLEND_SOURCE_SHA256', 'source')

    async def execute(*args, **kwargs):
        if isolated: assert adapter.monitor.active
        operations.append('execute')
        return raw

    async def drain(*args, **kwargs):
        if isolated: assert adapter.monitor.active
        operations.append('drain')
        return []

    async def sleep(*args, **kwargs):
        pass

    monkeypatch.setattr(run_native, 'execute', execute)
    monkeypatch.setattr(runtime, 'drain_endpoints', drain)
    monkeypatch.setattr(runtime.asyncio, 'sleep', sleep)
    monkeypatch.setattr(runtime.time, 'time', lambda: 250.5)
    monkeypatch.setattr(runtime, 'read_native_measurement', lambda *args: (100., [], None))
    monkeypatch.setattr(runtime, 'canonical_outcomes', lambda *args, **kwargs: [])
    monkeypatch.setattr(runtime, 'reduce_comparison', lambda *args, **kwargs: dict(request_metrics=[]))
    monkeypatch.setattr(runtime, 'summarize_comparison', lambda *args, **kwargs: dict(
        util_coverage_fraction=1., service=dict(utilization=dict(per_gpu={
            u: dict(mean_pct=10., peak_pct=20.) for u in adapter.identity['fleet_gpu_uuids']}))))
    monkeypatch.setattr(runtime, 'write_power_archive', lambda path, _: path.write_text('[]'))
    observed = []

    def audit(point, identity, startup, reset, supplied, metrics, metering, drained, refs):
        frozen = runtime.load_bound(refs['native_result'])
        assert supplied == frozen
        assert supplied['gpu_uuids'] == {'0': 'GPU-a', '1': 'GPU-b'}
        if isolated:
            assert runtime.load_bound(refs['metering_method']) == dict(window_guard_active=False,status='running')
        observed.append(supplied)
        # Serialization normalization cannot promote independently invalid energy.
        return dict(evidence_valid=False, formal_eligible=False, missing_gates=['power_gap'])

    monkeypatch.setattr(acceptance, 'audit_ecoserve_window', audit)
    result = asyncio.run(adapter.execute(point, out))
    assert len(observed) == 1
    assert raw['gpu_uuids'] == {0: 'GPU-a', 1: 'GPU-b'}
    assert result['formal_eligible'] is False and result['missing_gates'] == ['power_gap']
    assert json.loads((out / 'acceptance.json').read_text())['evidence_valid'] is False
    if isolated: assert operations == ['begin','execute','drain','end','snapshot']


@pytest.mark.parametrize('identity_mode,systems,point_modes,expected', [
    (None,['mixed'],[None],False), (None,['ecoserve'],[None],False),
    ('isolated_process',['ecoserve'],['isolated_process'],True),
    ('isolated_process',['distserve'],['isolated_process'],True),
    ('isolated_process',['ecoserve','distserve'],['isolated_process','isolated_process'],None),
    ('isolated_process',['mixed'],['isolated_process'],None),
    ('isolated_process',['ecoserve','ecoserve'],['isolated_process',None],None),
    (None,['ecoserve'],['isolated_process'],None),
    ('isolated_process',['ecoserve'],[None],None)])
def test_isolated_meter_requires_explicit_matching_single_system_identity(identity_mode,systems,point_modes,expected):
    group=dict(engine_identity=dict(metering_execution=identity_mode),points=[
        dict(system=system,metering_execution=mode,qualification_mode=system+'_native_bootstrap')
        for system,mode in zip(systems,point_modes)])
    if expected is None:
        with pytest.raises(ValueError,match='matching explicit qualified single-system'):
            runtime.NativeResidentAdapter.isolated_metering_requested(group)
    else:
        assert runtime.NativeResidentAdapter.isolated_metering_requested(group) is expected


def test_isolated_distserve_requires_its_own_bootstrap_qualification():
    group=dict(engine_identity=dict(metering_execution='isolated_process'),points=[dict(
        system='distserve',metering_execution='isolated_process',qualification_mode='ecoserve_native_bootstrap')])
    with pytest.raises(ValueError,match='qualified single-system'):
        runtime.NativeResidentAdapter.isolated_metering_requested(group)


@pytest.mark.parametrize('stop_fails', [False, True])
def test_native_failure_closes_guard_before_stopping_child_and_preserves_error(tmp_path, stop_fails):
    adapter=runtime.NativeResidentAdapter(tmp_path,base_port=17000)
    adapter.isolated_metering=True
    operations=[]
    class Meter:
        active=False
        stopped=False
        def begin_window(self): self.active=True;operations.append('begin')
        def end_window(self): self.active=False;operations.append('end')
        def method_receipt(self): return dict(window_guard_active=self.active,status='stopped' if self.stopped else 'running')
        def stop(self,**kwargs):
            assert not self.active
            operations.append('stop');self.stopped=True
            if stop_fails: raise RuntimeError('child failure')
    adapter.monitor=Meter()
    async def fail(*args):
        assert adapter.monitor.active
        raise ValueError('native request failed')
    adapter._execute_window=fail
    with pytest.raises(ValueError,match='native request failed'):
        asyncio.run(adapter.execute(dict(system='ecoserve',metering_execution='isolated_process'),tmp_path/'run'))
    assert operations == ['begin','end','stop']
    assert (tmp_path/'run/metering-method-failure.json').is_file()
    assert bool(adapter.metering_cleanup_errors) is stop_fails
