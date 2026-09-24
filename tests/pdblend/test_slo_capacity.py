"""Synthetic unit fixtures exercise the capacity ledger; no experiments run."""
from dataclasses import replace
import math

import pytest

from pdblend.bench.slo_capacity import (
    CapacityConfig, CapacityTrial, capacity_state, freeze_evaluation_grid, trial_verdict,
)


CONFIG = CapacityConfig('qwen7b/pdblend/sharegpt', 1., .15)


def trial(rate=1., repeat=0, *, split='tuning', **metrics):
    values = dict(offered_requests=100, successful_requests=100, failed_requests=0,
                  joint_slo_requests=100, unresolved_requests=0, ttft_samples=100,
                  tpot_samples=100, measurement_usable=True, slo_ttft_s=1.,
                  slo_tpot_s=.15, ttft_p99_s=.8, tpot_p99_s=.12)
    values.update(metrics)
    return CapacityTrial(CONFIG.series_id, split, rate, str(repeat),
                         f'synthetic-{split}-{rate}-{repeat}', values)


def repeats(rate, **metrics):
    return [trial(rate, repeat, **metrics) for repeat in range(3)]


def bracket(lower=1., upper=1.04):
    return repeats(lower) + repeats(upper, ttft_p99_s=1.1)


def test_every_slo_criterion_is_applied_at_inclusive_threshold():
    record = trial(joint_slo_requests=90, ttft_p99_s=1., tpot_p99_s=.15)
    assert trial_verdict(record, CONFIG) == {'verdict': 'pass', 'reasons': []}
    assert trial_verdict(trial(joint_slo_requests=89), CONFIG)['reasons'] == [
        'joint_slo_below_90_percent']
    assert trial_verdict(trial(ttft_p99_s=1.001), CONFIG)['reasons'] == ['ttft_p99_exceeds_slo']
    assert trial_verdict(trial(tpot_p99_s=.151), CONFIG)['reasons'] == ['tpot_p99_exceeds_slo']
    failed = trial(successful_requests=99, failed_requests=1, joint_slo_requests=99,
                   ttft_samples=99, tpot_samples=99)
    assert trial_verdict(failed, CONFIG)['reasons'] == ['success_rate_below_100_percent']


def test_completed_all_failed_trial_supplies_failure_without_inventing_latency():
    record = trial(successful_requests=0, failed_requests=100, joint_slo_requests=0,
                   ttft_samples=0, tpot_samples=0, ttft_p99_s=None, tpot_p99_s=None,
                   invalid_timing_requests=100)
    assert trial_verdict(record, CONFIG)['verdict'] == 'fail'
    state = capacity_state([replace(record, repeat_id=str(i), evidence_id=f'failed-{i}')
                            for i in range(3)], CONFIG)
    assert state['failed_upper'] == 1.
    assert not state['converged']


@pytest.mark.parametrize('metrics,reason', [
    ({'offered_requests': 99, 'successful_requests': 99, 'joint_slo_requests': 99,
      'ttft_samples': 99, 'tpot_samples': 99}, 'insufficient_request_samples'),
    ({'ttft_samples': 99}, 'incomplete_latency_samples'),
    ({'tpot_p99_s': None}, 'missing:tpot_p99_s'),
    ({'measurement_usable': False}, 'measurement_unusable'),
    ({'successful_requests': 99, 'failed_requests': 1, 'joint_slo_requests': 99,
      'ttft_samples': 99, 'tpot_samples': 99, 'unresolved_requests': 1}, 'unresolved_requests'),
    ({'slo_ttft_s': None}, 'missing:slo_ttft_s'),
])
def test_missing_or_incomplete_measurement_is_never_a_bound(metrics, reason):
    records = repeats(1., **metrics)
    verdict = trial_verdict(records[0], CONFIG)
    assert verdict['verdict'] == 'incomplete'
    assert reason in verdict['reasons']
    state = capacity_state(records, CONFIG)
    assert state['status'] == 'incomplete'
    assert state['passed_lower'] is state['failed_upper'] is state['capacity_interval'] is None
    assert state['action'] == 'repeat_rate' and state['next_rate_scale'] == 1.


def test_empty_and_missing_bounds_search_by_doubling_or_halving():
    assert capacity_state([], replace(CONFIG, initial_rate=.25))['next_rate_scale'] == .25
    state = capacity_state(repeats(1.), CONFIG)
    assert state['status'] == 'missing_upper_bound' and state['next_rate_scale'] == 2.
    assert state['passed_lower'] == 1. and state['capacity_interval'] is None
    state = capacity_state(repeats(2., ttft_p99_s=2.), CONFIG)
    assert state['status'] == 'missing_lower_bound' and state['next_rate_scale'] == 1.


def test_binary_search_uses_strongest_observed_bounds():
    records = repeats(.5) + repeats(1.) + repeats(2., ttft_p99_s=2.) + repeats(4., ttft_p99_s=2.)
    state = capacity_state(records, CONFIG)
    assert state['status'] == 'bracketing'
    assert (state['passed_lower'], state['failed_upper'], state['next_rate_scale']) == (1., 2., 1.5)
    assert state['capacity_interval'] is None


def test_at_least_three_repeats_required_and_all_must_agree():
    state = capacity_state(repeats(1.)[:2], CONFIG)
    assert state['status'] == 'incomplete' and state['passed_lower'] is None
    records = repeats(1.)[:2] + [trial(1., 2, ttft_p99_s=2.)]
    state = capacity_state(records, CONFIG)
    assert state['status'] == 'unstable_repeats' and state['next_rate_scale'] is None
    assert state['passed_lower'] is state['failed_upper'] is None
    assert capacity_state(repeats(1.), replace(CONFIG, required_repeats=4))['status'] == 'incomplete'


@pytest.mark.parametrize('incomplete_attempt', [False, True])
def test_known_repeat_conflict_is_not_hidden_by_missing_or_incomplete_repeats(incomplete_attempt):
    records = [trial(1., 0), trial(1., 1, ttft_p99_s=2.)]
    if incomplete_attempt:
        records.append(trial(1., 2, measurement_usable=False))
    state = capacity_state(records, CONFIG)
    assert state['status'] == 'unstable_repeats'
    assert state['action'] == 'investigate'
    assert state['next_rate_scale'] is state['capacity_interval'] is None
    assert state['passed_lower'] is state['failed_upper'] is None
    assert state['rates'][0]['complete_repeats'] == 2
    with pytest.raises(ValueError, match='unstable_repeats'):
        freeze_evaluation_grid(records, CONFIG)


def test_nonmonotonic_evidence_blocks_capacity_claim_and_grid():
    records = repeats(1., ttft_p99_s=2.) + repeats(2.)
    state = capacity_state(records, CONFIG)
    assert state['status'] == 'nonmonotonic' and state['action'] == 'investigate'
    assert state['capacity_interval'] is state['next_rate_scale'] is None
    with pytest.raises(ValueError, match='nonmonotonic'):
        freeze_evaluation_grid(records, CONFIG)


@pytest.mark.parametrize('upper,converged', [(1.04, True), (1.05, True), (1.050001, False)])
def test_five_percent_is_relative_to_passing_lower_bound(upper, converged):
    state = capacity_state(bracket(1., upper), CONFIG)
    assert state['converged'] is converged
    assert state['capacity_interval'] == ({'passed_lower': 1., 'failed_upper': upper}
                                         if converged else None)


def test_any_incomplete_or_unstable_rate_prevents_cherry_picking_tight_bracket():
    state = capacity_state(bracket() + repeats(.5)[:2], CONFIG)
    assert state['status'] == 'incomplete' and not state['converged']
    state = capacity_state(bracket() + repeats(.5)[:2] + [trial(.5, 2, ttft_p99_s=2.)], CONFIG)
    assert state['status'] == 'unstable_repeats' and not state['converged']


def test_grid_is_frozen_only_from_tuning_with_bound_receipts_and_slo():
    records = bracket()
    frozen = freeze_evaluation_grid(records, CONFIG, multipliers=(.5, 1., 1.1))
    assert frozen['selection_split'] == 'tuning' and frozen['target_split'] == 'evaluation'
    assert frozen['evaluation_used_for_selection'] is False
    assert frozen['rate_scales'] == [.5, 1., 1.04, 1.1]
    assert frozen['selection_evidence_ids'] == sorted(record.evidence_id for record in records)
    assert frozen['slo'] == {'ttft_s': 1., 'tpot_s': .15, 'success_rate': 1., 'joint_slo_rate': .9}
    assert frozen['required_repeats'] == 3 and frozen['min_requests_per_trial'] == 100
    calibrated = [replace(record, split='calibration') for record in records]
    assert freeze_evaluation_grid(calibrated, CONFIG)['selection_split'] == 'calibration'


def test_evaluation_and_different_series_or_slo_are_rejected():
    with pytest.raises(ValueError, match='never evaluation'):
        trial(split='evaluation')
    with pytest.raises(ValueError, match='series'):
        capacity_state([replace(trial(), series_id='different-system')], CONFIG)
    with pytest.raises(ValueError, match='same frozen SLO'):
        capacity_state([trial(slo_tpot_s=.2)], CONFIG)
    with pytest.raises(ValueError, match='one calibration or tuning'):
        capacity_state([trial(), trial(2., split='calibration')], CONFIG)


def test_relabeling_same_receipt_or_reusing_repeat_does_not_supply_independence():
    record = trial()
    with pytest.raises(ValueError, match='duplicate'):
        capacity_state([record, replace(record, repeat_id='1')], CONFIG)
    with pytest.raises(ValueError, match='duplicate'):
        capacity_state([record, replace(record, evidence_id='another-receipt')], CONFIG)


@pytest.mark.parametrize('changes', [
    {'required_repeats': 2}, {'required_repeats': True}, {'min_requests_per_trial': 99},
    {'relative_tolerance': .051}, {'relative_tolerance': 0}, {'slo_ttft_s': math.nan},
    {'slo_tpot_s': math.inf}, {'initial_rate': -1}, {'series_id': ''},
])
def test_config_parameters_are_explicit_and_validated(changes):
    with pytest.raises(ValueError):
        replace(CONFIG, **changes)


@pytest.mark.parametrize('metrics', [
    {'successful_requests': True}, {'offered_requests': -1}, {'joint_slo_requests': 101},
    {'failed_requests': 1}, {'measurement_usable': 'yes'}, {'ttft_p99_s': math.nan},
    {'tpot_p99_s': math.inf}, {'ttft_p99_s': -1}, {'invalid_timing_requests': True},
])
def test_bad_measurement_values_are_rejected(metrics):
    with pytest.raises(ValueError):
        trial_verdict(trial(**metrics), CONFIG)


@pytest.mark.parametrize('multipliers', [(), (1., 1.), (0.,), (math.inf,), (True,)])
def test_invalid_evaluation_grid_parameters_are_rejected(multipliers):
    with pytest.raises(ValueError):
        freeze_evaluation_grid(bracket(), CONFIG, multipliers=multipliers)


def test_grid_cannot_be_selected_without_converged_failure_bound():
    with pytest.raises(ValueError, match='missing_upper_bound'):
        freeze_evaluation_grid(repeats(1.), CONFIG)


def test_float_overflow_search_has_no_fabricated_next_rate():
    state = capacity_state(repeats(1e308), CONFIG)
    assert state['status'] == 'numeric_search_limit'
    assert state['next_rate_scale'] is None and state['capacity_interval'] is None


def test_float_resolution_cannot_repeat_a_bound_forever():
    records = bracket(1., math.nextafter(1., 2.))
    state = capacity_state(records, replace(CONFIG, relative_tolerance=1e-20))
    assert state['status'] == 'numeric_search_limit'
    assert state['next_rate_scale'] is None and not state['converged']
