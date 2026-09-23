import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip('fastapi')
from pdblend_runtime import serve


def setup_identity(tmp_path, monkeypatch):
    model = tmp_path/'Qwen2.5-7B-Instruct'
    model.mkdir()
    records = []
    for name, kind in [('model.safetensors', 'weight'), ('tokenizer.json', 'tokenizer')]:
        payload = (name+'-verified').encode()
        (model/name).write_bytes(payload)
        records.append(dict(path=name, kind=kind, bytes=len(payload),
                            sha256=hashlib.sha256(payload).hexdigest()))
    receipt = tmp_path/'verified.json'
    receipt.write_text(json.dumps(dict(all_pass=True, models={'7b': dict(
        model_id=model.name, model_path=str(model), verified=True, files=records)})))
    monkeypatch.setenv('PDBLEND_MODEL_VERIFICATION_RECEIPT', str(receipt))
    monkeypatch.setattr(serve, 'instance_uuids', lambda: ['GPU-cpu-test'])
    return SimpleNamespace(model=str(model)), receipt, records


def test_relative_receipt_paths_are_resolved_against_model(tmp_path, monkeypatch):
    args, receipt, records = setup_identity(tmp_path, monkeypatch)
    result = serve.serving_identity(args)
    assert result['model_id'] == 'Qwen2.5-7B-Instruct'
    assert result['model_hash'] == hashlib.sha256(json.dumps([
        (r['path'], r['bytes'], r['sha256']) for r in records if r['kind']=='weight'
    ], separators=(',', ':'), sort_keys=True).encode()).hexdigest()
    assert result['verification_receipt_sha256'] == hashlib.sha256(receipt.read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['missing', 'size', 'escape'])
def test_model_mount_must_match_receipt(tmp_path, monkeypatch, change):
    args, receipt, _ = setup_identity(tmp_path, monkeypatch)
    from pathlib import Path
    weight = Path(args.model)/'model.safetensors'
    if change=='missing':
        weight.unlink()
    elif change=='size':
        weight.write_bytes(b'changed')
    else:
        data=json.loads(receipt.read_text())
        data['models']['7b']['files'][0]['path']='../model.safetensors'
        (tmp_path/'model.safetensors').write_bytes(weight.read_bytes())
        receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='model files differ'):
        serve.serving_identity(args)
