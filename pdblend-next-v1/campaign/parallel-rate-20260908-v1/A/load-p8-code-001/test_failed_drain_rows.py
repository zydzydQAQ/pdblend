"""A failed physical drain keeps original client rows without claiming validity."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
HOST = HERE.parents[1] / 'hosts/14b-capacity-p8'
sys.path[:0] = [str(HERE), str(HOST / 'src'), str(HOST), '/root/workspace/pdblend/.runtime-deps']
s = importlib.util.spec_from_file_location('p8_rows_producer', HERE / 'capacity_load_calibrate.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def test_completed_client_rows_survive_failed_physical_drain(tmp_path, monkeypatch):
    from benchmarks.scripts import bench_vllm as bench
    import capacity_backend
    rows = [dict(idx=0, request_id='0', success=1, generated_tokens=1)]
    trace_path = tmp_path / 'trace.json'
    trace_path.write_text(json.dumps(dict(duration_s=.001, n_requests=1, requests=[], prompts=[])))
    out = tmp_path / 'phase'

    class Meter:
        def __init__(self, path): self.path = path
        async def start(self): self.path.mkdir(parents=True); return self
        async def finish(self):
            p = self.path / 'synthetic-cpu-measurement.json'
            p.write_text(json.dumps(dict(cpu_only=True, measurement_valid=True)))
            return dict(receipt=m.ref(p), measurement_valid=True)

    async def run_trace(*args, **kwargs):
        return [dict(planned_arrival_s=time.time())], .001
    async def native_idle(*args): pass
    async def failed_drain(*args): return dict(drain_complete=False, error='historical capacity task failed')
    monkeypatch.setattr(capacity_backend, 'TransitionMeter', Meter)
    monkeypatch.setattr(bench, 'run_trace', run_trace)
    monkeypatch.setattr(bench, 'bench_rows', lambda *args: rows)
    monkeypatch.setattr(m, 'native_idle', native_idle)
    controller = SimpleNamespace(config=dict(slo_ttft_s=1, slo_tpot_s=.1),
        backend=SimpleNamespace(instances={'initial':dict(id='initial', gpus=[6])}),
        finish_measurement=failed_drain)
    spec = dict(demand_domain_sha256='x'*64, original_binding={}, capacity_binding={}, config={},
        host_release=str(HOST), api_base='http://cpu-only.invalid', served_model='cpu-only',
        mode='qualification900', deadline_s=None)
    with pytest.raises(ValueError, match='native cleanup incomplete'):
        asyncio.run(m.measure_phase(controller, None, m.ref(trace_path), out, spec))
    assert json.loads((out / 'requests.json').read_text()) == rows
    result = json.loads((out / 'result.json').read_text())
    assert result['complete'] is False and 'error' in result
    assert 'n_expected' not in result and 'work_complete' not in result
    assert str(out / 'requests.json') in result['artifacts']
