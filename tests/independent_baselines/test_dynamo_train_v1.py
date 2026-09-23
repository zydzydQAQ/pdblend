import json
from pathlib import Path

import pytest

from pdblend_baselines.dynamollm import train_v1
from pdblend_baselines.dynamollm.deployment import sha


def arguments(tmp_path):
    return ['--model', '/models/Qwen2.5-14B-Instruct', '--corpus-root', '/corpus/14b',
            '--encoder', '/models/bert-base-uncased', '--out', str(tmp_path/'checkpoint'),
            '--report', str(tmp_path/'training.json'), '--device', 'cuda']


def test_training_completion_binds_new_manifest_without_qualifying_predictor(tmp_path, monkeypatch):
    monkeypatch.setattr(train_v1, 'prepare', lambda *args: dict(model_identity={'model':'Qwen2.5-14B-Instruct'}))
    def fake_train(corpora, encoder, output, **kwargs):
        assert kwargs['seed'] == 701 and kwargs['device'] == 'cuda'
        assert all('/corpus/14b/' in str(path) for path in corpora)
        output.mkdir()
        (output/'manifest.json').write_text(json.dumps(dict(schema=1, model_identity={'model':'Qwen2.5-14B-Instruct'})))
        kwargs['report_path'].write_text(json.dumps(dict(schema=1, manifest_sha256='old')))
        return {'schema':1}
    monkeypatch.setattr(train_v1, 'train', fake_train)
    train_v1.main(arguments(tmp_path))
    completion = json.loads((tmp_path/'completion.json').read_text())
    report = json.loads((tmp_path/'training.json').read_text())
    manifest = json.loads((tmp_path/'checkpoint/manifest.json').read_text())
    assert completion['status'] == 'passed' and completion['complete'] is True
    assert not completion['predictor_qualified'] and not completion['heldout_passed']
    assert not completion['formal_eligible'] and report['schema'] == 1
    assert report['manifest_sha256'] == completion['checkpoint_manifest_sha256'] == sha(tmp_path/'checkpoint/manifest.json')
    assert manifest['output_work'] == train_v1.OUTPUT_WORK and manifest['ignore_eos'] is True


def test_failed_prepare_writes_queue_failure_receipt_without_training(tmp_path, monkeypatch):
    def fail(*args): raise ValueError('wrong model corpus')
    monkeypatch.setattr(train_v1, 'prepare', fail)
    with pytest.raises(ValueError, match='wrong model'):
        train_v1.main(arguments(tmp_path))
    completion = json.loads((tmp_path/'completion.json').read_text())
    assert completion['status'] == 'failed' and not completion['complete']
    assert not (tmp_path/'checkpoint').exists()
