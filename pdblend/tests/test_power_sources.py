"""Power provenance regressions with a fake NVML module; no GPU access."""
import sys
from types import SimpleNamespace

import pytest

from ecopadg.measure import backends
from ecopadg.measure.backends import BackendError, PynvmlBackend, INSTANT_POWER_SOURCE_ID
from ecopadg.measure.power import PowerSampler


def field(**changes):
    values=dict(fieldId=186,scopeId=0,valueType=1,nvmlReturn=0,timestamp=100_000_000,
                latencyUsec=12,value=SimpleNamespace(uiVal=123456))
    values.update(changes)
    return SimpleNamespace(**values)


def nvml(monkeypatch, value=None, init_error=False):
    calls=[]
    def init():
        if init_error: raise RuntimeError('unavailable NVML')
    def fields(handle, requests):
        calls.append(('fields',handle,requests))
        return [value or field()]
    def average(handle):
        calls.append(('average',handle))
        return 200000
    module=SimpleNamespace(nvmlInit=init,nvmlDeviceGetHandleByIndex=lambda gpu:gpu,
        nvmlDeviceGetFieldValues=fields,nvmlDeviceGetPowerUsage=average)
    monkeypatch.setitem(sys.modules,'pynvml',module)
    monkeypatch.setattr(backends,'_smi_query',lambda *a: calls.append(('smi',a)) or '88.5')
    return module,calls


def test_explicit_instant_uses_uint_milliwatts_and_records_exact_source(monkeypatch):
    _,calls=nvml(monkeypatch)
    backend=PynvmlBackend(power_mode='instant',clock=lambda:100.)
    reading=backend.power_reading(3)
    assert reading['watts']==pytest.approx(123.456)
    assert reading['source_id']==INSTANT_POWER_SOURCE_ID
    assert reading['nvml_timestamp_us']==100_000_000
    assert reading['nvml_latency_us']==12
    assert backend.power_source['field_id']==186
    assert backend.power_source['unit']=='W'
    assert calls==[('fields',3,[(186,0)])]


class UndefinedUnion:
    @property
    def uiVal(self):
        raise AssertionError('undefined value union was accessed')


@pytest.mark.parametrize('changes',[dict(nvmlReturn=3),dict(valueType=0),dict(fieldId=185),dict(scopeId=1)])
def test_invalid_field_header_never_reads_union_or_falls_back(monkeypatch,changes):
    _,calls=nvml(monkeypatch,field(value=UndefinedUnion(),**changes))
    backend=PynvmlBackend(power_mode='instant',clock=lambda:100.)
    with pytest.raises(BackendError,match='invalid field ID'):
        backend.power_w(0)
    assert len(calls)==1 and calls[0][0]=='fields'


@pytest.mark.parametrize('changes',[
    dict(timestamp=0),dict(timestamp=99_000_000),dict(timestamp=100_100_000),
    dict(timestamp=100_000_000.),dict(latencyUsec=-1),dict(latencyUsec=None),
    dict(value=SimpleNamespace(uiVal=-1)),dict(value=SimpleNamespace(uiVal=2**32-1)),
    dict(value=SimpleNamespace(uiVal=42.5))])
def test_invalid_instant_value_or_timestamp_never_falls_back(monkeypatch,changes):
    _,calls=nvml(monkeypatch,field(**changes))
    backend=PynvmlBackend(power_mode='instant',clock=lambda:100.)
    with pytest.raises(BackendError,match='no average fallback'):
        backend.power_w(0)
    assert len(calls)==1 and calls[0][0]=='fields'


def test_timestamp_can_repeat_but_cannot_regress_per_gpu(monkeypatch):
    value=field()
    nvml(monkeypatch,value)
    backend=PynvmlBackend(power_mode='instant',clock=lambda:100.)
    backend.power_w(0); backend.power_w(0)
    value.timestamp-=1
    backend.power_w(1)  # Separate GPU timestamp history.
    with pytest.raises(BackendError,match='regressed'):
        backend.power_w(0)


def test_unavailable_instant_fails_without_subprocess_fallback(monkeypatch):
    _,calls=nvml(monkeypatch,init_error=True)
    with pytest.raises(BackendError,match='no average fallback'):
        PynvmlBackend(power_mode='instant')
    assert calls==[]


def test_default_average_retains_existing_api_and_explicit_provenance(monkeypatch):
    module,calls=nvml(monkeypatch)
    backend=PynvmlBackend(clock=lambda:100.)
    assert backend.power_mode=='average'
    assert backend.power_reading(0)['watts']==200
    assert calls==[('average',0)]
    module.nvmlDeviceGetPowerUsage=lambda h: (_ for _ in ()).throw(RuntimeError('unsupported'))
    reading=backend.power_reading(0)
    assert reading['watts']==88.5 and reading['mode']=='average'
    assert reading['source_id']=='nvidia-smi:power.draw'
    assert reading['nvml_timestamp_us'] is None


def test_sampler_preserves_power_rows_and_matching_per_gpu_metadata(monkeypatch):
    nvml(monkeypatch)
    backend=PynvmlBackend(power_mode='instant',clock=lambda:100.)
    sampler=PowerSampler(range(8),backend=backend,clock=lambda:100.)
    row=sampler._read()
    assert row==(100.,[123.456]*8)
    assert sampler.power_source['source_id']==INSTANT_POWER_SOURCE_ID
    assert sampler.power_metadata[0]['gpus']==list(range(8))
    assert sampler.power_metadata[0]['t_s']==row[0]
    assert sampler.power_metadata[0]['nvml_timestamp_us']==[100_000_000]*8
    assert sampler.power_metadata[0]['field_id']==[186]*8


def test_sampler_failure_is_visible_and_does_not_emit_a_fallback_row(monkeypatch):
    _,calls=nvml(monkeypatch,field(nvmlReturn=3))
    backend=PynvmlBackend(power_mode='instant',clock=lambda:100.)
    sampler=PowerSampler(range(8),backend=backend,clock=lambda:100.)
    sampler._loop()
    assert 'no average fallback' in sampler.error
    assert sampler.samples==[] and sampler.power_metadata==[]
    assert len(calls)==1


def test_sampler_rejects_a_mode_change(monkeypatch):
    nvml(monkeypatch)
    backend=PynvmlBackend(power_mode='instant',clock=lambda:100.)
    sampler=PowerSampler(range(8),backend=backend,clock=lambda:100.)
    backend._power_mode='average'
    with pytest.raises(ValueError,match='mode changed'):
        sampler._read()
    assert sampler.power_metadata==[]
