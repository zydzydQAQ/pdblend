"""Actual P6/P8 files with source, policy and validator-isolation counterexamples."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest

A = Path(__file__).resolve().parent
R = A.parent
sys.path.insert(0, str(R))
from capacity_calibration_compatibility_v3 import verify, sha


def ref(p): return dict(path=str(p), sha256=sha(p))


@pytest.fixture
def actual(tmp_path):
    proof = json.loads((R / 'common/P6-numerical-P8-recalibration-bootstrap-v1.json').read_text())
    verifier = ref(R / 'capacity_calibration_compatibility_v3.py')
    proof['verifier'] = verifier
    proof['files'][verifier['path']] = verifier['sha256']
    cap = json.loads(Path(proof['measured_capacity_binding']['path']).read_text())
    cap['physical_operation_timeout_s'] = 120
    path = tmp_path / 'proof.json'
    def freeze():
        path.write_text(json.dumps(proof))
        reference = ref(path)
        cap['controller_calibration_compatibility'] = reference
        cap['files'][reference['path']] = reference['sha256']
        return reference
    return proof, cap, freeze


def test_actual_source_bound_validation_survives_poisoned_parent_module(actual, monkeypatch):
    proof, cap, freeze = actual
    bad = types.ModuleType('capacity_certificate')
    def reject(*args): raise AssertionError('ambient validator must never be used')
    bad.validate = reject
    monkeypatch.setitem(sys.modules, 'capacity_certificate', bad)
    assert verify(freeze(), proof['actual_controller_manifest'], cap)['authorized']


@pytest.mark.parametrize('change', ['timeout', 'identity', 'policy', 'profile', 'certificate',
    'relabel', 'actual_host', 'patch', 'verifier', 'missing_cap_pin', 'unearned_new_transitions'])
def test_explicit_compatibility_rejects_scope_or_source_changes(actual, change):
    proof, cap, freeze = actual
    actual_host = proof['actual_controller_manifest']
    if change == 'timeout': cap['physical_operation_timeout_s'] = 121
    elif change == 'identity': cap['identity']['source_sha256'] = '0' * 64
    elif change == 'policy': cap['policy'] = dict(min_off_s=0)
    elif change == 'profile': cap['calibrated_source_semantics']['profile']['sha256'] = '0' * 64
    elif change == 'certificate': cap['calibration']['sha256'] = '0' * 64
    elif change == 'relabel': proof['original_measurements_not_relabelled'] = False
    elif change == 'actual_host': actual_host = proof['P7_controller_manifest']
    elif change == 'patch': proof['approved_patch_sources']['capacity_executor.py']['sha256'] = '0' * 64
    elif change == 'verifier': proof['verifier']['sha256'] = '0' * 64
    elif change == 'unearned_new_transitions':
        proof['certificate_scope'] = 'original_P6_layout_savings_with_actual_P8_transition_groups'
        proof['new_transition_qualification_complete'] = False
    reference = freeze()
    if change == 'missing_cap_pin': cap['files'].pop(reference['path'])
    with pytest.raises((ValueError, KeyError)):
        verify(reference, actual_host, cap)
