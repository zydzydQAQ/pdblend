"""Keep physical recovery identical while requiring this model's actual proof."""
import ast
import copy
import json
import sys
import time
from pathlib import Path

import pytest
import restore
import release_gate as gate


def model_release(model='14b'):
    module = gate.verifier()
    path = gate.CAMPAIGN / ('A14B-main-completion-proof-v1/host-main-proof.json' if model == '14b'
                            else 'C7B-main-proof-prepared-v1/host-main-proof.json')
    proof = module.read(path)
    return dict(schema=1, kind=module.KIND, model=model, protocol_id=module.PROTOCOL,
                deadline_s=module.DEADLINE, models={model: proof},
                proof_refs={model: dict(path=str(path), sha256=module.AUTHORIZED[model][0],
                                       canonical_sha256=module.AUTHORIZED[model][1])},
                baseline_systems=list(module.v1.BASELINES), main_records=150,
                baseline_main_records=120, pdblend_main_records=30,
                coordinator_deep_verification=True, global_release=False, created_s=time.time())


def test_actual_A_model_proof_accepted_without_B_or_C(tmp_path):
    path = tmp_path / 'CPU-contract-only.json'
    path.write_text(json.dumps(model_release()))
    result = gate.verify_release(path, restore.sha(path))
    assert result['model'] == '14b' and result['main_records'] == 150
    assert result['global_release'] is False and result['released'] is True


@pytest.mark.parametrize('change', ['other_model', 'global450', 'missing_eco', 'wrong_sha', 'deadline'])
def test_wrong_scope_or_evidence_rejected(tmp_path, change):
    value = model_release('7b' if change == 'other_model' else '14b')
    if change == 'global450':
        value.update(kind='global-main-release-per-cell', main_records=450, global_release=True)
    if change == 'missing_eco':
        value['models']['14b']['records'] = [r for r in value['models']['14b']['records']
                                             if r['row']['system'] != 'ecoserve']
    if change == 'deadline':
        value['deadline_s'] += 1
    path = tmp_path / 'CPU-invalid.json'
    path.write_text(json.dumps(value))
    with pytest.raises((RuntimeError, ValueError)):
        gate.verify_release(path, 'a' * 64 if change == 'wrong_sha' else restore.sha(path))


def test_physical_path_is_exact_original_after_metadata_key_normalization():
    current = Path(restore.__file__).read_text()
    normalized = current.replace('model_main_release', 'global_main_release').replace(
        'real_model_release', 'real_global_release')
    original = (gate.CAMPAIGN / 'A14B-resident-restore-v2/restore.py').read_text()
    assert normalized == original


def test_original_process_scan_and_local_module_isolation():
    marker = object()
    previous = sys.modules.get('partition')
    sys.modules['partition'] = marker
    try:
        module = gate.verifier()
        assert sys.modules['partition'] is marker
        assert module.v1.process_scan.__code__.co_filename == str(
            gate.CAMPAIGN / 'main-first-barrier-v1/barrier.py')
    finally:
        if previous is None:
            sys.modules.pop('partition', None)
        else:
            sys.modules['partition'] = previous
