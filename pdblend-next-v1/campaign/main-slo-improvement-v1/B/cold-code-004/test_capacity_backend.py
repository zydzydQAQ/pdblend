import asyncio
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity_backend import PinnedDockerBackend
from capacity_backend import TransitionMeter
from capacity_executor import Inventory


class Response:
    status = 200
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    async def json(self):
        return {'id':'actual'}


class Session:
    def __init__(self):
        self.calls = []
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response()
    get = post


def test_request_uses_actual_header_identity_for_later_cancellation(tmp_path):
    session = Session()
    backend = object.__new__(PinnedDockerBackend)
    backend.controller = SimpleNamespace(session=session)
    backend.inventory = Inventory(tmp_path/'inventory.json', [], {})
    asyncio.run(backend.request(dict(id='new',url='http://127.0.0.1:1234'), '/v1/completions',
        dict(prompt=[1],max_tokens=1,request_id='controlled-id'), limit=time.time()+10))
    assert session.calls[0][1]['headers'] == {'X-Request-Id':'controlled-id'}


def test_expired_native_operation_is_rejected_before_dispatch(tmp_path):
    session = Session()
    backend = object.__new__(PinnedDockerBackend)
    backend.controller = SimpleNamespace(session=session)
    backend.inventory = Inventory(tmp_path/'inventory.json', [], {})
    with pytest.raises(ValueError, match='deadline'):
        asyncio.run(backend.request(dict(id='new',url='http://127.0.0.1:1234'), '/runtime', limit=time.time()-1))
    assert not session.calls


def test_meter_integrates_exact_window_without_default_energy_padding(tmp_path, monkeypatch):
    import types
    import capacity_backend
    power = types.ModuleType('ecopadg.measure.power')
    power.trapezoid_energy = lambda rows: 80.
    metrics = types.ModuleType('ecopadg.metrics')
    observed = {}
    def clip(rows, start, end, pad_s=1.):
        observed.update(start=start,end=end,pad_s=pad_s)
        if pad_s:
            raise ValueError('default extra padding was not sampled')
        return rows
    metrics.clip_power_window = clip
    measurement = types.ModuleType('ecopadg.serving.measurement')
    measurement.save_raw = lambda *a,**k:None
    measurement.power_evidence = lambda *a:dict(power_source_verified=True)
    for name,module in [('ecopadg.measure.power',power),('ecopadg.metrics',metrics),
                        ('ecopadg.serving.measurement',measurement)]:
        monkeypatch.setitem(sys.modules,name,module)
    now = time.time()
    sampler = SimpleNamespace(samples=[(now-1.,[10.]*8),(now+1.,[10.]*8)],
        utilization_samples=[],power_source={},power_metadata=[],error=None,stop=lambda:None)
    meter = TransitionMeter(tmp_path/'meter')
    meter.sampler = sampler
    meter.started_s = now-.5
    meter.memory = [dict(gpu=g,used_bytes=100,at_s=now) for g in range(8)]
    result = asyncio.run(meter.finish())
    assert observed['pad_s'] == 0. and result['measurement_valid'] is True
    assert result['energy_j'] == 80.
