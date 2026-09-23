import asyncio
import hashlib
import json
import sys

import pytest

from pdblend_baselines.dynamollm.run_v1 import collect_functional_profile_stage, _stop_profile_collectors


def test_profile_stage_is_opt_in():
    assert asyncio.run(collect_functional_profile_stage({}, __import__('pathlib').Path('/tmp'))) is None


def test_profile_stage_requires_two_resident_instances(tmp_path):
    with pytest.raises(ValueError, match='exactly two'):
        asyncio.run(collect_functional_profile_stage({
            'functional_profile_stage': {'enabled': True},
            'instances': [],
        }, tmp_path))


def test_profile_stage_rejects_nonfunctional_mode(tmp_path):
    with pytest.raises(ValueError, match='only allowed'):
        asyncio.run(collect_functional_profile_stage({
            'mode': 'primitive',
            'functional_profile_stage': {'enabled': True},
            'instances': [{}, {}],
        }, tmp_path))


def test_profile_stage_has_two_disjoint_frequency_groups():
    # Keep the split explicit: 9 cells per resident (3 inputs x 3 freqs).
    stage = {'enabled': True, 'frequency_groups': [[900, 1200, 1500], [1800, 2100, 2520]]}
    assert set(sum(stage['frequency_groups'], [])) == {900, 1200, 1500, 1800, 2100, 2520}
    assert len(stage['frequency_groups'][0]) * 3 == 9
    assert len(stage['frequency_groups'][1]) * 3 == 9


def test_collector_cleanup_kills_process_after_bounded_term_timeout():
    class Stuck:
        returncode = None
        def __init__(self): self.terminated = self.killed = False
        def terminate(self): self.terminated = True
        def kill(self): self.killed = True; self.returncode = -9
        async def wait(self):
            if not self.killed:
                await asyncio.sleep(1)
            return self.returncode
    process = Stuck()
    asyncio.run(_stop_profile_collectors([process], terminate_timeout_s=.001, kill_timeout_s=.1))
    assert process.terminated and process.killed


def stage_config(tmp_path):
    def write(path, value):
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    predictor = tmp_path/'predictor'; predictor.mkdir()
    predictor_sha = write(predictor/'manifest.json', {'model_id':'test-model'})
    trace = tmp_path/'trace.json'
    trace_sha = write(trace, dict(seed=701, requests=[dict(prompt=[1]*128, max_tokens=16)]))
    receipt = tmp_path/'prediction.json'
    receipt_sha = write(receipt, dict(model_id='test-model', seed=701, trace_sha256=trace_sha,
        predictor_manifest_sha256=predictor_sha,
        predictions=[dict(request_index=0, input_tokens=128, trace_max_tokens=16, predicted_output=74)]))
    return dict(mode='functional', model_id='test-model', model_path='/unloaded-model', trace=str(trace),
        dynamo_predictor_dir=str(predictor),
        instances=[dict(tp=1, gpus=[index], port=19000+index*2, url='http://unused', id=str(index))
                   for index in range(2)],
        functional_profile_stage=dict(enabled=True, outputs=[74], prediction_receipt=str(receipt),
            prediction_receipt_sha256=receipt_sha, trace_sha256=trace_sha, collector_timeout_s=.05))


@pytest.mark.parametrize('mutation', ['drop_request', 'index', 'input', 'output_count'])
def test_stage_rejects_rehashed_prediction_for_other_request(tmp_path, mutation):
    from pathlib import Path
    config = stage_config(tmp_path)
    receipt = Path(config['functional_profile_stage']['prediction_receipt'])
    value = json.loads(receipt.read_text())
    if mutation == 'drop_request': value['predictions'] = []
    else:
        field = {'index':'request_index', 'input':'input_tokens', 'output_count':'trace_max_tokens'}[mutation]
        value['predictions'][0][field] += 1
    receipt.write_text(json.dumps(value))
    config['functional_profile_stage']['prediction_receipt_sha256'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='every trace request'):
        asyncio.run(collect_functional_profile_stage(config, tmp_path/'out'))


def test_stage_deadline_reaps_real_collectors(tmp_path, monkeypatch):
    config = stage_config(tmp_path)
    processes = []
    original = asyncio.create_subprocess_exec
    async def sleeper(*unused):
        process = await original(sys.executable, '-c', 'import time; time.sleep(30)')
        processes.append(process)
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', sleeper)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(collect_functional_profile_stage(config, tmp_path/'out'))
    assert len(processes) == 2 and all(process.returncode is not None for process in processes)
