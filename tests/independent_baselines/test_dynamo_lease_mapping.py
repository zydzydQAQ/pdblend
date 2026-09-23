"""Dynamo's CUDA-local / UUID-NVML contract, without any actual GPU access."""
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from pdblend_baselines.dynamollm.deployment import SubprocessLifecycle
from pdblend_baselines.dynamollm.telemetry import GroupTelemetry


@pytest.mark.asyncio
async def test_real_lifecycle_passes_local_ordinals_to_subprocess_and_keeps_uuid_lease(tmp_path, monkeypatch):
    monkeypatch.setenv('PDBLEND_GPU_UUIDS', 'GPU-host7,GPU-host3,GPU-host5')
    captured = {}
    async def create(*command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=123456, returncode=None)
    class Transport:
        instances = {}
        async def state(self, iid):
            return dict(evidence_complete=True, transport_healthy=True, generation=2)
    monkeypatch.setattr('asyncio.create_subprocess_exec', create)
    lifecycle = SubprocessLifecycle(dict(base_port=19000, model_id='Qwen2.5-7B-Instruct',
        model_path='/models/Qwen2.5-7B-Instruct', legal_tp=[1, 2, 4], node_gpus=[0, 1, 2]),
        Transport(), lambda *args, **kwargs: None, tmp_path)
    try:
        ready = await lifecycle.start(dict(id='own', gpus=[2, 0], tp=2, port=19032, generation=2),
                                      dummy=True, golden={'tp': 2, 'token_ids': [1]})
        assert ready['ready']
        assert captured['env']['CUDA_VISIBLE_DEVICES'] == '2,0'
        assert captured['env']['PDBLEND_GPU_UUIDS'] == 'GPU-host7,GPU-host3,GPU-host5'
        assert captured['env']['DYNAMO_GENERATION'] == '2'
        assert captured['env']['DYNAMO_DUMMY'] == '1'
        assert captured['start_new_session'] is True
        with pytest.raises(ValueError, match='outside physical UUID lease'):
            lifecycle.environment(dict(id='foreign', gpus=[3]))
    finally:
        # No real child was created; close the actual lifecycle's owned log.
        for stream in lifecycle.logs.values():
            stream.close()


@pytest.mark.asyncio
async def test_telemetry_samples_physical_uuid_subset_and_resets_only_changed_owned_clocks(monkeypatch):
    monkeypatch.setenv('PDBLEND_GPU_UUIDS', 'GPU-host7,GPU-host3,GPU-host5')
    calls, rows = [], []
    nv = ModuleType('pynvml')
    nv.NVML_SUCCESS, nv.NVML_CLOCK_SM = 0, 1
    nv.nvmlInit = lambda: calls.append(('init',))
    nv.nvmlShutdown = lambda: calls.append(('shutdown',))
    nv.nvmlDeviceGetHandleByUUID = lambda uuid: uuid
    nv.nvmlDeviceGetUUID = lambda handle: handle
    def no_index(_): raise AssertionError('NVML must never use CUDA-local ordinal as physical index')
    nv.nvmlDeviceGetHandleByIndex = no_index
    nv.nvmlDeviceSetGpuLockedClocks = lambda handle, low, high: calls.append(('clock', handle, low, high))
    nv.nvmlDeviceResetGpuLockedClocks = lambda handle: calls.append(('reset', handle))
    nv.nvmlDeviceGetClockInfo = lambda handle, domain: 1200
    def field(handle, fields):
        calls.append(('sample', handle))
        assert fields == [(186, 0)]
        return [SimpleNamespace(nvmlReturn=0, valueType=1, fieldId=186, scopeId=0,
            timestamp=int(time.time()*1e6), value=SimpleNamespace(uiVal=100000))]
    nv.nvmlDeviceGetFieldValues = field
    monkeypatch.setitem(sys.modules, 'pynvml', nv)
    def journal(kind, **record):
        rows.append(dict(kind=kind, **record))
        if sum(row['kind'] == 'dynamo_power' for row in rows) == 2:
            meter.stop_event.set()
    meter = GroupTelemetry([2, 0], journal)
    assert meter.uuids == {2: 'GPU-host5', 0: 'GPU-host7'}
    meter.clock([2], 1200)
    with pytest.raises(ValueError, match='outside owned GPU lease'):
        meter.clock([1], 1200)
    meter.start()
    await meter.task
    await meter.close()
    assert [row[1] for row in calls if row[0] == 'sample'] == ['GPU-host5', 'GPU-host7']
    assert [row for row in calls if row[0] == 'reset'] == [('reset', 'GPU-host5')]
    assert {(row['gpu'], row['gpu_uuid']) for row in rows if row['kind'] == 'dynamo_power'} == {
        (2, 'GPU-host5'), (0, 'GPU-host7')}
    assert not any('GPU-host3' in row for row in calls)
