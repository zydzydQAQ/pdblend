import json
from types import SimpleNamespace

import pytest

from pdblend.bench.client import Request
from pdblend.bench.tp_fixed_pd_qualification import MODES, audit_window, prepare
from tests.pdblend.test_tp_runtime import profiled_model


def prepared(tmp_path):
    path = tmp_path / 'tp2.json'
    profiled_model(2, 'Qwen2.5-32B-Instruct').save(path)
    trace = [Request(i, (i // 2) * 8, [123] * (512, 1024, 2048)[i % 3], 16) for i in range(6)]
    args = SimpleNamespace(model='Qwen2.5-32B-Instruct', tp=2, gpus=[0, 1, 2, 3],
                           profile=path, base_port=18000, out=tmp_path)
    runtimes, receipt = prepare(args, {(2, 1): path}, trace)
    return runtimes, receipt, trace


def test_cpu_windows_share_native_specs_and_label_forced_pd(tmp_path, monkeypatch):
    monkeypatch.setattr('pdblend.bench.tp_fixed_pd_qualification.Gpus', lambda *a: pytest.fail('GPU access'))
    runtimes, receipt, _ = prepared(tmp_path)
    assert set(runtimes) == set(MODES)
    assert all(runtime.specs == runtimes['fixed_tp'].specs for runtime in runtimes.values())
    assert [list(s.gpus) for s in runtimes['fixed_tp'].specs] == [[0, 1], [2, 3]]
    assert receipt['controllers']['mechanism_pd']['initial_plan']['counts'] == {'P': 1, 'D': 1}
    assert not receipt['controllers']['mechanism_pd']['policy_decision']
    assert receipt['controllers']['offline_tp']['policy_m_floor'] == 4
    assert not receipt['hardware_executed']


@pytest.mark.parametrize('path_kind', ['PD', 'M'])
def test_pd_window_rejects_mixed_route_receipts(tmp_path, path_kind):
    runtimes, _, trace = prepared(tmp_path)
    runtime = runtimes['mechanism_pd']
    p, d = runtime.specs
    outcomes, routes = [], []
    for request in trace:
        outcomes.append(dict(idx=request.idx, error=None, completion_tokens=16, sampling_seed=701,
                             first_token_s=1, finished_s=2, prefill=p.instance_id, decode=d.instance_id, path=path_kind))
        routes.append(dict(request_id=f'r{request.idx}', prefill_instance=p.instance_id, decode_instance=d.instance_id,
                           path=path_kind, tp=d.tp, pp=d.pp, generation=d.generation, pool_id=d.pool_id, profile_key=d.profile_key))
    for name, rows in [('outcomes.jsonl', outcomes), ('routes.jsonl', routes)]:
        (tmp_path / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (tmp_path / 'metering.json').write_text('{"error":null,"power_samples":2}')
    (tmp_path / 'power.jsonl').write_text('[1,[100,100,100,100]]\n[2,[100,100,100,100]]\n')
    errors, _ = audit_window(tmp_path, trace, runtime, {}, 'mechanism_pd')
    assert bool(errors) == (path_kind != 'PD')
