import asyncio
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity_backend import PinnedDockerBackend
from capacity_backend import TransitionMeter, cancel_transfer_rows
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
        utilization_samples=[],frequency_samples=[(now,[1500]*8)],power_source={},power_metadata=[],error=None,stop=lambda:None)
    meter = TransitionMeter(tmp_path/'meter')
    meter.sampler = sampler
    meter.started_s = now-.5
    meter.memory = [dict(gpu=g,used_bytes=100,at_s=now) for g in range(8)]
    result = asyncio.run(meter.finish())
    assert observed['pad_s'] == 0. and result['measurement_valid'] is True
    assert result['energy_j'] == 80.
    assert result['clock_samples_observed']==1
    assert str(tmp_path/'meter/clocks.json') in result['artifacts']


def test_actual_unlabeled_collective_cancel_rows_are_validated():
    row = dict(allocations={},buffered_tensors=0,buffered_gpu_bytes=0,inflight_receives=0,
               inflight_sends=0,send_failed=0,listener_alive=True,send_counters_observed=True,
               send_healthy=True,send_started=0,send_completed=0)
    cancel_transfer_rows([dict(row),dict(row)],2)
    with pytest.raises(ValueError):
        cancel_transfer_rows([row],2)
    with pytest.raises(ValueError):
        cancel_transfer_rows([row,dict(row,inflight_sends=1)],2)


def test_owned_clock_bootstrap_and_release_require_fresh_unpublished_lease_bound_inventory(tmp_path, monkeypatch):
    import capacity_backend
    from capacity_executor import sha
    initial=[dict(id='old0',gpus=[0,1]),dict(id='old1',gpus=[2,3])]
    inventory=Inventory(tmp_path/'inventory.json',initial,{})
    extra=dict(id='extra',gpus=[4,5])
    inventory.value['transition_inflight']=True
    inventory.intent(extra,'start','real-transaction')
    checks=[]
    monkeypatch.setattr(capacity_backend,'check_lease',lambda **kw:checks.append(kw))
    backend=object.__new__(PinnedDockerBackend);backend.inventory=inventory
    backend.controller=SimpleNamespace(config={'capacity_inventory_path':str(inventory.path)},
        backend=SimpleNamespace(instances={i['id']:i for i in initial}))
    async def spare(gpus):return [dict(gpu=g,at_s=time.time(),process_pids=[]) for g in gpus]
    backend.assert_spare=spare
    proof=asyncio.run(backend.clock_ownership_proof(extra,'bootstrap'))
    assert checks and proof['schema']=='capacity-clock-bootstrap-v1'
    assert proof['inventory_sha256']==sha(inventory.path)
    assert inventory.value['events'][-1]['kind']=='clock_bootstrap_proof'
    inventory.intent(extra,'stop','cleanup-transaction')
    proof=asyncio.run(backend.clock_ownership_proof(extra,'release'))
    assert proof['transaction']=='cleanup-transaction' and proof['inventory_sha256']==sha(inventory.path)
    backend.controller.backend.instances['extra']=extra
    with pytest.raises(ValueError,match='published'):
        asyncio.run(backend.clock_ownership_proof(extra,'release'))
    backend.controller.backend.instances.pop('extra')
    inventory.value['transition_inflight']=False
    with pytest.raises(ValueError,match='intent'):
        asyncio.run(backend.clock_ownership_proof(extra,'release'))
    inventory.value['transition_inflight']=True
    with pytest.raises(ValueError,match='intent'):
        asyncio.run(backend.clock_ownership_proof(initial[0],'release'))
    async def stale(gpus):return [dict(gpu=g,at_s=time.time()-2.,process_pids=[]) for g in gpus]
    backend.assert_spare=stale
    with pytest.raises(ValueError,match='fresh'):
        asyncio.run(backend.clock_ownership_proof(extra,'release'))
