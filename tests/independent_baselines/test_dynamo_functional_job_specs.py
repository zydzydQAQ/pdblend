"""Exercise the real spec builder using minimal, independent input artifacts."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def builder(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / 'scripts/2026-09-23_prepare_dynamo_functional_jobs.py'
    spec = importlib.util.spec_from_file_location('functional_job_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    predictors = {}
    for key, (model, tp, _) in module.MODELS.items():
        prepared = tmp_path / 'results/2026-09-23/functional-prepared' / (key + '-run1')
        prepared.mkdir(parents=True)
        (prepared / 'config.json').write_text(json.dumps({'instances': [{'id': 'a'}, {'id': 'b'}]}))
        (prepared / 'profiles.json').write_text(json.dumps({'points': []}))
        predictor = tmp_path / 'predictors' / key
        predictor.mkdir(parents=True)
        (predictor / 'manifest.json').write_text('{}')
        predictors[key] = predictor
        trace = tmp_path / (key + '-trace.json')
        trace.write_text(json.dumps({'seed': 701, 'requests': [
            {'prompt': [1, 2, 3], 'max_tokens': 16, 'arrival_s': 0}]}))
        prediction = tmp_path / 'results/2026-09-23/independent-baseline-coverage-v7' / (key + '-predictor-trace-predictions.json')
        prediction.parent.mkdir(exist_ok=True)
        prediction.write_text(json.dumps({'model_id': model, 'seed': 701,
            'checkpoint': str(predictor), 'trace': str(trace), 'predictions': [
                {'request_index': 0, 'input_tokens': 3, 'trace_max_tokens': 16, 'predicted_output': 74}]}))
    monkeypatch.setattr(module, 'PREDICTORS', predictors)
    monkeypatch.setattr(module, 'VERIFY', tmp_path / 'model-verification.json')
    return module


@pytest.mark.parametrize('key,tp,count', [('7b', 1, 2), ('14b', 1, 2), ('32b', 2, 4)])
def test_real_builder_preserves_lease_and_predictor_contract(builder, tmp_path, key, tp, count):
    job = builder.make_spec(key, tmp_path / 'specs', tmp_path / 'source', 'source-hash', ['power-a'])
    payload = job['payload']
    assert payload['gpu_count'] == count and payload['tp'] == tp
    assert payload['depends_on'] == ['power-a']
    assert payload['argv'][payload['argv'].index('--gpus') + 1] == 'all'
    assert 'PDBLEND_GPU_UUIDS={lease_gpu_uuids}' in payload['argv']
    assert 'CUDA_VISIBLE_DEVICES={lease_local_indices}' in payload['argv']
    assert 'PDBLEND_LEASE_PORT={lease_port}' in payload['argv']
    assert payload['required_receipts'] == ['dynamo/completion.json']
    mounts = {target: (host, mode) for host, target, mode in payload['mounts']}
    assert mounts['/prediction-receipt.json'][1] == 'ro'
    assert mounts['/spec/config.json'][1] == 'ro'
    config = json.loads(Path(payload['config_path']).read_text())
    assert config['functional_profile_stage']['outputs'] == [74]
    assert config['node_gpus'] == list(range(count))
    assert [i['gpus'] for i in config['instances']] == [list(range(tp)), list(range(tp, count))]
    assert payload['formal_eligible'] is False


def test_real_builder_rejects_predictor_receipt_from_different_trace(builder, tmp_path):
    path = tmp_path / 'results/2026-09-23/independent-baseline-coverage-v7/7b-predictor-trace-predictions.json'
    value = json.loads(path.read_text())
    value['predictions'][0]['input_tokens'] = 4
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='bind every trace request'):
        builder.make_spec('7b', tmp_path / 'specs', tmp_path / 'source', 'source-hash', ['power-a'])
