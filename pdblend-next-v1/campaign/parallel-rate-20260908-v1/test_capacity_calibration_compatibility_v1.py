"""P7 may reuse numerical P6 evidence only with its distinct source retained."""
import copy
import json
from pathlib import Path
import pytest
import capacity_calibration_compatibility_v1 as c

ROOT=Path(__file__).resolve().parent
PROOF=ROOT/'common/P6-calibration-P7-controller-compatibility-v1.json'
REF=dict(path=str(PROOF),sha256=c.sha(PROOF))
P=c.checked(REF)
CAP=c.checked(P['measured_capacity_binding'])


def test_real_frozen_delta_and_original_certificate_validate():
    before=copy.deepcopy(CAP)
    proof=c.verify(REF,P['actual_controller_manifest'],CAP)
    assert CAP==before
    assert proof['measured_controller_manifest']!=proof['actual_controller_manifest']
    assert proof['claims_P7_was_measured_for_original_certificate'] is False
    assert proof['requires_new_actual_autonomous_underload_gate']
    assert proof['requires_new_actual_candidate900']


def test_other_actual_controller_cannot_use_the_compatibility():
    with pytest.raises(ValueError,match='actual controller'):
        c.verify(REF,P['measured_controller_manifest'],CAP)


def test_original_calibration_identity_cannot_be_relabelled():
    cap=copy.deepcopy(CAP);cap['identity']['source_sha256']='0'*64
    with pytest.raises(ValueError,match='source identity'):
        c.verify(REF,P['actual_controller_manifest'],cap)


def test_measured_controller_semantics_cannot_be_relabelled_as_P7():
    cap=copy.deepcopy(CAP);cap['calibrated_source_semantics']['candidate_manifest']=P['actual_controller_manifest']
    with pytest.raises(ValueError,match='source identity'):
        c.verify(REF,P['actual_controller_manifest'],cap)


def test_different_certificate_cannot_be_substituted():
    cap=copy.deepcopy(CAP);cap['calibration']=P['clock_age_cpu_reproduction']
    with pytest.raises(ValueError,match='different numerical certificate'):
        c.verify(REF,P['actual_controller_manifest'],cap)


@pytest.mark.parametrize('key',['requires_new_actual_autonomous_underload_gate','requires_new_actual_candidate900'])
def test_reuse_does_not_waive_new_actual_control_qualification(tmp_path,key):
    altered=copy.deepcopy(P);altered[key]=False
    path=tmp_path/'altered.json';path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='separate P7 qualification'):
        c.verify(dict(path=str(path),sha256=c.sha(path)),P['actual_controller_manifest'],CAP)


def test_mutated_proof_is_rejected_by_frozen_reference(tmp_path):
    path=tmp_path/'changed.json';path.write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):
        c.verify(dict(path=str(path),sha256=REF['sha256']),P['actual_controller_manifest'],CAP)
