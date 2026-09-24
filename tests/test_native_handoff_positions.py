from copy import deepcopy

import pytest

from pdblend.profile.collection.native_handoff_positions import observe_endpoint_positions,require_planner_handoff_prediction
from tests.test_native_runtime_transfer_contract import full_request


def test_real_native_position_partition_preserves_signed_contrast_and_stays_unqualified():
    row,specs=full_request();result=observe_endpoint_positions(row,specs=specs)
    assert result['first_output_latency_s']+result['first_to_second_gap_s']==pytest.approx(result['second_output_latency_s'])
    assert result['first_to_second_gap_s']==pytest.approx(result['client_bridge_s']+result['decode_submit_to_next_output_s'])
    assert result['physical_copy_time_s'] is None and not result['proxy_client_ttft_observed']
    assert not result['predictor_qualified'] and not result['planner_transfer_seconds_compatible']
    with pytest.raises(ValueError,match='cannot be transfer_seconds'):require_planner_handoff_prediction(result)


def test_negative_contrast_is_retained_and_future_or_missing_arrivals_are_rejected():
    row,specs=full_request()
    # Synthetic reference timing change; the PD observation is kept identical.
    # This shows why an ordinary/PD signed contrast is not a physical duration.
    for ordinary in row['result']['ordinary']:
        ordinary['submitted_s']-=1.
    for state in row['native_epoch']['before'].values():state['native_at_s']-=2.
    for admission in row['native_epoch']['admission'].values():admission['state']['native_at_s']-=2.
    observed=observe_endpoint_positions(row,specs=specs)
    assert observed['signed_second_output_contrast_s']<0
    assert observed['first_output_latency_s']>=0 and observed['first_to_second_gap_s']>=0
    bad=deepcopy(row);bad['result']['combined']['decode_submitted_s']=bad['result']['combined']['decode_first_token_s']+1
    with pytest.raises(ValueError):observe_endpoint_positions(bad,specs=specs)
    bad=deepcopy(row);bad['result']['combined'].pop('decode_first_token_s')
    with pytest.raises(ValueError):observe_endpoint_positions(bad,specs=specs)


def test_new_predeclared_seeds_are_explicit_and_cannot_relabel_an_existing_request():
    row,specs=full_request()
    with pytest.raises(ValueError,match='seed differs'):
        observe_endpoint_positions(row,specs=specs,training_seed=9911,holdout_seed=9912)
    row['seed']=9911  # Synthetic fresh request; no old bound artifact is changed.
    assert observe_endpoint_positions(row,specs=specs,training_seed=9911,holdout_seed=9912)['predictor_qualified'] is False
