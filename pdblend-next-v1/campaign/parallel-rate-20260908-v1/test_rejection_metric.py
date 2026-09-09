"""Verify both historical boolean flags and actual native rejection codes."""
import json
import pytest
from raw_metrics_v2 import admission_rejected


@pytest.mark.parametrize('marker', [None, '', False, 'False', '0', 'null'])
def test_absent_rejection(marker):
    assert not admission_rejected({'admission_rejection': marker})


@pytest.mark.parametrize('marker', [True, 'true', '1'])
def test_legacy_boolean_rejection(marker):
    assert admission_rejected({'admission_rejection': marker})


def row(code='admission_queue_full'):
    return dict(admission_rejection=code, http_status='429', success='0',
        error='RuntimeError: HTTP 429: ' + json.dumps({'error': {'type': 'admission_rejection',
                                                              'code': 'admission_queue_full'}}))


def test_actual_named_rejection_is_counted():
    assert admission_rejected(row())


@pytest.mark.parametrize('change', [dict(success='1'), dict(http_status='200'),
    dict(admission_rejection='different_code'), dict(error='unstructured failure')])
def test_named_marker_requires_matching_failed_http_evidence(change):
    with pytest.raises(ValueError):
        admission_rejected(dict(row(), **change))
