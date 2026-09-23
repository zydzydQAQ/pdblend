import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from pdblend.bench.client import Request
from pdblend.bench.tp_qualification import audit_gpu_result, canonical_sha, cpu_prepare, read_trace
from pdblend.control.topology import ResidentPool, Topology
from tests.pdblend.test_tp_runtime import profiled_model, profiles


def prepared(tmp_path):
    args = SimpleNamespace(model='Qwen2.5-7B-Instruct', gpus=[0, 1, 2], base_port=18000, out=tmp_path)
    trace = [Request(i, 0, [123] * 512, 128) for i in range(96)]
    runtime, receipt = cpu_prepare(args, profiles(tmp_path, (1, 2)),
                                   (ResidentPool('small', Topology(1), 1),
                                    ResidentPool('large', Topology(2), 1)), trace)
    return runtime, receipt, trace


def test_preflight_uses_three_gpus_and_never_emits_hardware_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr('pdblend.bench.run.Gpus', lambda *a: pytest.fail('GPU access during CPU preflight'))
    runtime, receipt, trace = prepared(tmp_path)
    assert [s.gpus for s in runtime.specs] == [(0,), (1, 2)]
    assert not receipt['hardware_executed']
    assert all(set(row['admitted_by_tp']) == {1, 2} for row in receipt['admission_envelope'])
    assert not (tmp_path / 'completion.json').exists()
    assert not (tmp_path / 'outcomes.jsonl').exists()


def test_32b_resident_tp2_tp4_uses_exactly_six_gpus(tmp_path):
    paths = {}
    for tp in (2, 4):
        path = tmp_path / f'tp{tp}.json'
        profiled_model(tp, 'Qwen2.5-32B-Instruct').save(path)
        paths[tp, 1] = path
    args = SimpleNamespace(model='Qwen2.5-32B-Instruct', gpus=list(range(6)), base_port=18000, out=tmp_path)
    trace = [Request(i, 0, [123] * 512, 128) for i in range(96)]
    runtime, receipt = cpu_prepare(args, paths, (ResidentPool('small', Topology(2), 1),
                                               ResidentPool('large', Topology(4), 1)), trace)
    assert [s.gpus for s in runtime.specs] == [(0, 1), (2, 3, 4, 5)]
    assert all(set(row['admitted_by_tp']) == {2, 4} for row in receipt['admission_envelope'])
    assert not receipt['hardware_executed']


def write_unit_receipts(out, trace, runtime, *, one_pool=False, bad_generation=False):
    # Test doubles exercise the auditor only; these are not GPU qualifications.
    outcomes, routes, owners = [], [], []
    for i, request in enumerate(trace):
        spec = runtime.specs[0 if one_pool else i % 2]
        identity = dict(tp=spec.tp, pp=1, pool_id=spec.pool_id, generation=spec.generation)
        route = dict(request_id=f'r{i}', path='M', prefill_instance=spec.instance_id,
                     decode_instance=spec.instance_id, profile_key=spec.profile_key, **identity)
        owner = dict(route, profile_keys=[spec.profile_key], model_id=spec.model,
                     input_tokens=request.input_tokens, max_tokens=request.max_tokens)
        if bad_generation:
            owner['generation'] += 1
        routes.append(route)
        owners.append(owner)
        outcomes.append(dict(idx=i, error=None, completion_tokens=128, sampling_seed=701,
                             first_token_s=1, finished_s=2, path='M', prefill=spec.instance_id,
                             decode=spec.instance_id))
    for name, rows in [('outcomes.jsonl', outcomes), ('routes.jsonl', routes), ('resident-routes.jsonl', owners)]:
        (out / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (out / 'metering.json').write_text(json.dumps(dict(error=None, power_samples=2)))
    (out / 'power.jsonl').write_text('[1,[100,100,100]]\n[2,[100,100,100]]\n')
    return dict(startup_s={s.instance_id: 1 for s in runtime.specs}, quarantined_instances=[])


@pytest.mark.parametrize('one_pool,bad_generation', [(True, False), (False, True), (False, False)])
def test_gpu_audit_requires_both_pools_and_pinned_generation(tmp_path, one_pool, bad_generation):
    runtime, _, trace = prepared(tmp_path)
    summary = write_unit_receipts(tmp_path, trace, runtime, one_pool=one_pool, bad_generation=bad_generation)
    errors, _ = audit_gpu_result(tmp_path, trace, runtime, summary)
    assert bool(errors) == (one_pool or bad_generation)


def test_trace_checksum_detects_mutation(tmp_path):
    rows = [asdict(Request(0, 0, [123] * 512, 128))]
    trace = tmp_path / 'trace.json'
    value = dict(seed=701, requests=rows, trace_sha256=canonical_sha(rows))
    trace.write_text(json.dumps(value))
    assert read_trace(trace)[0][0].input_tokens == 512
    value['requests'][0]['prompt'][0] = 124
    trace.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='checksum'):
        read_trace(trace)
