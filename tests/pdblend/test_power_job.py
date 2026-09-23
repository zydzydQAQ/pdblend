import json
import pytest
from pdblend.profile.power_job import queue_receipt,require_paired_cohort
from pdblend.profile.power_calibration import digest


def test_worker_receipt_does_not_turn_failed_timing_into_pass(tmp_path):
    (tmp_path/'composite-audit.json').write_text('{}')
    result=dict(status='completed',complete=True,power_passed=True,reused_timing_passed=False,
        calibration_components_passed=False,composite_audit_sha256=digest(tmp_path/'composite-audit.json'))
    completion=tmp_path/'completion.json';completion.write_text(json.dumps(result));original=completion.read_bytes()
    receipt=queue_receipt(result,tmp_path)
    assert receipt['status']=='passed' and receipt['complete']
    assert receipt['power_status']=='passed' and receipt['reused_timing_status']=='failed'
    assert receipt['composite_status']==receipt['calibration_status']=='failed'
    assert receipt['formal_eligible'] is False and completion.read_bytes()==original
    result['calibration_components_passed']=True;completion.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='inconsistent'):queue_receipt(result,tmp_path)


def test_additional_timing_does_not_replace_original_component_result(tmp_path):
    (tmp_path/'composite-audit.json').write_text('{}')
    overlay=dict(complete=True,timing_passed=True,receipt_sha256='separate')
    result=dict(status='completed',complete=True,power_passed=True,reused_timing_passed=False,
        calibration_components_passed=False,composite_audit_sha256=digest(tmp_path/'composite-audit.json'),
        timing_overlay_requested=True,timing_overlay=overlay)
    (tmp_path/'completion.json').write_text(json.dumps(result))
    receipt=queue_receipt(result,tmp_path)
    assert receipt['complete'] and receipt['timing_overlay']==overlay
    assert receipt['calibration_status']==receipt['reused_timing_status']=='failed'
    assert receipt['formal_eligible'] is False


def test_failed_measurement_stays_failed(tmp_path):
    result=dict(status='failed',complete=False,error='sampling failed')
    (tmp_path/'completion.json').write_text(json.dumps(result))
    receipt=queue_receipt(result,tmp_path)
    assert not receipt['complete'] and receipt['status']=='failed'


def test_solo_and_quad_cannot_be_mislabelled_as_power_pair(tmp_path,monkeypatch):
    monkeypatch.setenv('PDBLEND_PROFILE_WAVE',str(tmp_path));monkeypatch.setenv('PDBLEND_PROFILE_MEMBER','7b-power')
    wave=dict(cohort_id='power-4-4',coordinator=True,purpose='independent_power_holdouts_4_plus_4',members=['7b-power','32b-power'])
    (tmp_path/'wave.json').write_text(json.dumps(wave));assert require_paired_cohort()==wave
    for members in [['7b-power'],['7b-power','a','b','c']]:
        (tmp_path/'wave.json').write_text(json.dumps(dict(wave,members=members)))
        with pytest.raises(ValueError,match='cannot claim'):require_paired_cohort()
