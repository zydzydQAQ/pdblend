from copy import deepcopy

import pytest

from pdblend.bench.cohort_dominance import BASELINE_SYSTEMS, analyze_points, compare_point


def point(system, *, energy=100., good=100, revision=None, repeat_id=None):
    identity = dict(trace_sha256='trace-701', seed=701, duration_s=150.,
                    slo_ttft_s=1., slo_tpot_s=.1, offered_requests=100, offered_rps=2.,
                    measurement_protocol_version='native-comparison-150s/v1',
                    model_hash='weights', tokenizer_hash='tokens', image_digest='image',
                    runtime_source_sha256='runtime', measurement_source_sha256='meter', gpu_uuids=['gpu1', 'gpu0'])
    row = dict(series='pd_current' if system == 'pdblend' else system, system=system,
               model='7B', dataset='alpaca', rate_scale=.25, offered_rps=2.,
               offered_requests=100, successful_requests=100, failed_requests=0,
               unresolved_requests=0, pending_at_window_end_requests=10,
               joint_slo_requests=good, all_requests_successful=True,
               slo_attainment_pct=good, ttft_p99_s=.5, tpot_p99_s=.05,
               slo_ttft_s=1., slo_tpot_s=.1, slo_pass=True,
               service_energy_kj=energy-10 if energy is not None else None,
               tail_energy_kj=10., total_energy_kj=energy,
               energy_measurement_complete=energy is not None,
               evidence_valid=False, formal_eligible=False, measurement_evidence_valid=True,
               revision=revision or system+'-revision',
               point_id=system+'-point', receipt_path=f'{system}/{revision}/{repeat_id}/receipt.json',
               comparison_identity=identity)
    if repeat_id is not None:
        row['repeat_id'] = repeat_id
    return row


def baseline_rows(**kwargs):
    return {system: point(system, **kwargs) for system in BASELINE_SYSTEMS}


def test_failed_baseline_still_sets_energy_target_even_when_pd_beats_feasible_baselines():
    baselines = baseline_rows(energy=200.)
    cheap = baselines['ecoserve'] = point('ecoserve', energy=90., good=20)
    cheap.update(successful_requests=30, failed_requests=70, all_requests_successful=False, slo_pass=False)
    result = compare_point(point('pdblend', energy=100.), baselines)
    assert result['min_available_baseline_energy_kj'] == 90.
    assert result['status'] == 'failed' and result['energy_goal_met'] is False
    assert result['baselines']['ecoserve']['baseline_all_requests_successful'] is False


def test_complete_slo_pass_is_insufficient_when_joint_good_is_below_a_baseline():
    candidate = point('pdblend', energy=80., good=99)
    result = compare_point(candidate, baseline_rows())
    assert result['absolute_slo_pass'] and result['all_requests_successful']
    assert result['max_baseline_joint_slo_requests'] == 100
    assert result['joint_good_goal_met'] is False and result['status'] == 'failed'


@pytest.mark.parametrize('change', [dict(successful_requests=99, failed_requests=1, all_requests_successful=False),
                                    dict(unresolved_requests=1), dict(ttft_p99_s=1.01),
                                    dict(tpot_p99_s=.101), dict(joint_slo_requests=89)])
def test_pd_must_succeed_finish_and_meet_both_p99s_and_absolute_attainment(change):
    candidate = point('pdblend', energy=80.)
    candidate.update(change)
    assert compare_point(candidate, baseline_rows(good=80))['status'] == 'failed'


def test_window_end_pending_requests_are_allowed_after_the_cohort_finishes():
    candidate = point('pdblend', energy=80.)
    candidate.pop('measurement_evidence_valid')
    assert candidate['pending_at_window_end_requests'] == 10
    result = compare_point(candidate, baseline_rows())
    assert result['numerical_goal_met'] is True and result['observed_goal_met'] is None
    assert result['measurement_qualification'] == 'unknown'
    assert result['comparison_measurement_qualification'] == 'unknown'
    assert result['formal_eligible'] is False


@pytest.mark.parametrize('missing_kind', ['baseline', 'energy', 'tail', 'trace', 'seed'])
def test_missing_or_unpaired_baseline_cannot_produce_a_win(missing_kind):
    baselines = baseline_rows()
    if missing_kind == 'baseline':
        baselines.pop('dynamollm')
    elif missing_kind == 'energy':
        baselines['dynamollm']['total_energy_kj'] = None
    elif missing_kind == 'tail':
        baselines['dynamollm']['tail_energy_kj'] = None
    else:
        baselines['dynamollm']['comparison_identity'][{'trace': 'trace_sha256', 'seed': 'seed'}[missing_kind]] = 'different'
    result = compare_point(point('pdblend', energy=80.), baselines)
    assert result['status'] == 'incomplete' and result['observed_goal_met'] is None
    assert result['missing']


def test_equal_energy_is_not_a_strict_win():
    result = compare_point(point('pdblend'), baseline_rows())
    assert result['energy_goal_met'] is False and result['status'] == 'failed'


def test_missing_baseline_does_not_hide_a_known_energy_loss():
    baselines = baseline_rows()
    baselines['dynamollm']['total_energy_kj'] = None
    result = compare_point(point('pdblend', energy=110.), baselines)
    assert result['status'] == 'incomplete' and result['energy_goal_met'] is False
    assert result['saving_vs_min_available_baseline_pct'] == pytest.approx(-10.)
    assert 'candidate_not_strictly_below_every_baseline_energy' in result['reasons']


def test_row_counts_must_agree_with_bound_comparison_identity():
    candidate = point('pdblend', energy=80.)
    candidate['comparison_identity']['offered_requests'] = 99
    result = compare_point(candidate, baseline_rows())
    assert result['status'] == 'incomplete'


def repeated_snapshot(n, *, margin=2.):
    rows = []
    for repeat in range(n):
        rows.extend(baseline_rows(repeat_id=repeat).values())
        rows.append(point('pdblend', energy=100.-margin, repeat_id=repeat))
    return rows


def test_small_margin_requires_three_distinct_fully_paired_repeats():
    for n in (1, 2):
        result = analyze_points(repeated_snapshot(n))
        assert result['cases'][0]['status'] == 'small_margin_needs_3_paired_repeats'
    result = analyze_points(repeated_snapshot(3))
    assert result['cases'][0]['status'] == 'stable_observed_win'
    assert result['cases'][0]['independent_paired_repeats']


def test_reused_baseline_receipts_do_not_count_as_three_paired_repeats():
    rows = repeated_snapshot(3)
    for row in rows:
        if row['system'] == 'mixed':
            row['receipt_path'] = 'same-mixed-receipt'
    case = analyze_points(rows)['cases'][0]
    assert case['status'] == 'small_margin_needs_3_paired_repeats'


def test_old_and_new_revisions_cannot_be_pooled_to_claim_repeat_support():
    rows = repeated_snapshot(3)
    for row in rows:
        if row['system'] == 'pdblend' and row['repeat_id'] == 0:
            row.update(series='pd_previous', revision='old-pd')
    cases = analyze_points(rows)['cases']
    assert len(cases) == 2
    assert {r['status'] for r in cases} == {'small_margin_needs_3_paired_repeats'}


def test_one_losing_repeat_prevents_a_stable_win():
    rows = repeated_snapshot(3)
    candidate = rows[-1]
    candidate.update(total_energy_kj=101., service_energy_kj=91.)
    assert analyze_points(rows)['cases'][0]['status'] == 'failed'


def test_ambiguous_baseline_revision_is_not_selected_by_energy_or_recency():
    rows = [point('pdblend', energy=80.), *baseline_rows().values()]
    another = deepcopy(rows[1])
    another.update(revision='other', receipt_path='another-receipt', total_energy_kj=150., service_energy_kj=140.)
    rows.append(another)
    assert analyze_points(rows)['cases'][0]['status'] == 'incomplete'


def test_explicit_measurement_and_formal_qualification_are_reported_independently():
    baselines, candidate = baseline_rows(), point('pdblend', energy=80.)
    for row in [candidate, *baselines.values()]:
        row['measurement_qualified'] = True
    result = compare_point(candidate, baselines)
    assert result['energy_complete'] and result['comparison_measurement_qualification'] == 'pass'
    assert result['comparison_formal_eligible'] is False
