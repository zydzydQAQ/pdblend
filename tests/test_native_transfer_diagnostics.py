import json
from pathlib import Path

import pytest

from pdblend.profile.collection.native_transfer_diagnostics import decompose, proposed_plan


def payload():
    row = json.loads((Path(__file__).parent / 'fixtures/native_runtime/2026-09-24_first_512_request.json').read_text())
    return row['request']['result']


def parts(result):
    return decompose(result, input_tokens=512, prefill_instance=result['prefill']['instance_id'],
                     decode_instance=result['combined']['instance_id'])


def test_real_first_512_client_record_decomposes_but_cannot_recover_physical_transport():
    result = payload()
    row = parts(result)
    assert row['reconstructed_overhead_s'] == pytest.approx(row['overhead_s'], abs=1e-12)
    assert row['pd_second_output_s'] == pytest.approx(row['pd_prefill_client_s'] + row['client_bridge_s'] + row['decode_submit_to_next_output_s'])
    assert row['physical_transport_s'] is None
    assert not row['physical_transport_identifiable'] and not row['formal_eligible']


def test_negative_difference_is_valid_signed_data_and_is_never_truncated_or_absolutized():
    result = payload()
    result['ordinary'][0]['submitted_s'] -= 1.
    row = parts(result)
    assert row['overhead_s'] < 0
    assert row['reconstructed_overhead_s'] < 0
    assert row['decode_submit_to_next_output_s'] > 0
    assert row['physical_transport_s'] is None


def test_second_ordinary_reference_is_diagnostic_not_selected_to_improve_the_fit():
    result = payload()
    before = parts(result)
    result['ordinary'][1]['submitted_s'] -= 10.
    after = parts(result)
    assert after['overhead_s'] == before['overhead_s']
    assert after['ordinary_reference_difference_s'] == pytest.approx(before['ordinary_reference_difference_s'] + 10)


def test_partial_or_out_of_order_token_data_remains_rejected():
    result = payload()
    result['combined']['decode_submitted_s'] = result['combined']['decode_first_token_s'] + 1
    with pytest.raises(ValueError):
        parts(result)


def test_new_plan_preserves_limits_and_requires_new_frozen_holdout_without_promoting_proxy():
    plan = proposed_plan()
    assert plan['holdout_limits'] == dict(mean_relative_error=.1, p95_relative_error=.2, max_relative_error=.25)
    assert (plan['training_repeats'], plan['holdout_repeats'], plan['settle_s'], plan['measure_s']) == (3, 1, 2., 5.)
    assert not plan['old_failed_holdout_reused_for_fit']
    assert not plan['transfer_seconds_replacement_authorized']
    assert not plan['component_qualified'] and not plan['formal_eligible']
