import json

import pytest

from pdblend.profile.local_power import SCOPE
from pdblend.profile.local_power_job import queue_receipt
from pdblend.profile.power_calibration import digest


def artifacts(root, *, power=True, repaired=True):
    audit = dict(validation_scope=SCOPE, full_profile_qualified=False,
        concurrency_qualified=True, power=dict(passed=power), timing=dict(passed=False),
        repaired_timing=dict(passed=repaired, complete=True, original_failed_rows_preserved=True),
        calibration_components_passed=power and repaired)
    (root/'composite-audit.json').write_text(json.dumps(audit))
    result = dict(status='completed', complete=True, scope=SCOPE,
        full_profile_qualified=False, concurrency_qualified=True,
        power_passed=power, reused_timing_passed=False, repaired_timing_passed=repaired,
        calibration_components_passed=power and repaired,
        composite_audit_sha256=digest(root/'composite-audit.json'))
    (root/'completion.json').write_text(json.dumps(result))
    return result


@pytest.mark.parametrize('power,repaired', [(True,True), (False,True), (True,False)])
def test_measurement_completion_does_not_erase_failed_qualification(tmp_path, power, repaired):
    result = artifacts(tmp_path, power=power, repaired=repaired)
    before = (tmp_path/'completion.json').read_bytes()
    receipt = queue_receipt(result, tmp_path)
    assert receipt['status'] == 'passed' and receipt['complete']
    assert receipt['original_timing_passed'] is False
    assert receipt['calibration_components_passed'] is (power and repaired)
    assert receipt['full_profile_qualified'] is False
    assert (tmp_path/'completion.json').read_bytes() == before


def test_truncated_sampling_cannot_release_as_success(tmp_path):
    result = dict(status='failed', complete=False, error='lost stream')
    (tmp_path/'completion.json').write_text(json.dumps(result))
    receipt = queue_receipt(result, tmp_path)
    assert receipt['status'] == 'failed' and receipt['complete'] is False


@pytest.mark.parametrize('alter', ['hash', 'power', 'scope', 'concurrency'])
def test_corrupt_or_contradictory_receipts_rejected(tmp_path, alter):
    result = artifacts(tmp_path)
    if alter == 'hash':
        (tmp_path/'composite-audit.json').write_text('{}')
    else:
        if alter == 'power': result['power_passed'] = False
        if alter == 'scope': result['scope'] = 'full_profile'
        if alter == 'concurrency': result['concurrency_qualified'] = False
        (tmp_path/'completion.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):
        queue_receipt(result, tmp_path)
