from copy import deepcopy

import pytest

from pdblend.bench.comparison_campaign import SYSTEMS, rank_rows
from pdblend.bench.comparison_observed import RANK_FIELDS, SCOPE, add_observed_comparisons
from test_resident_comparison import ranking_fixture


BASELINES = tuple(s for s in SYSTEMS if s != 'pdblend')


def candidate(rows):
    return next(r for r in rows if r['system'] == 'pdblend')


@pytest.mark.parametrize('scenario', ['full', 'partial', 'slo_failure', 'revisions',
                                     'duplicate_baseline', 'duplicate_candidate', 'identity'])
def test_clock_qualified_observed_and_strict_results_have_exact_parity(scenario):
    rows = ranking_fixture()
    for row in rows:row['energy_tail_j'] = 1.
    if scenario == 'partial':rows = [r for r in rows if r['system'] in ('mixed', 'pdblend')]
    elif scenario == 'slo_failure':rows[0]['slo_pass'] = False
    elif scenario == 'revisions':
        rows.append(dict(deepcopy(candidate(rows)), revision='v2', energy_service_j=5., energy_tail_j=50.))
    elif scenario == 'duplicate_baseline':rows.append(deepcopy(rows[0]))
    elif scenario == 'duplicate_candidate':rows.append(deepcopy(candidate(rows)))
    elif scenario == 'identity':candidate(rows)['model_hash'] = 'another'
    rank_rows(rows)
    for row in rows:
        assert row['observed_comparison_scope'] == SCOPE
        assert {key:row['observed_'+key] for key in RANK_FIELDS} == {key:row[key] for key in RANK_FIELDS}


def test_unknown_clock_allows_descriptive_comparison_without_touching_strict_or_raw_fields():
    rows = ranking_fixture();rows[0]['common_clock_evidence'] = 'unknown'
    rows[0].update(ttft_samples=33, ttft_low_sample=True, tpot_low_sample=True)
    for row in rows:row['energy_tail_j'] = 0.
    before = deepcopy(rows);rank_rows(rows);pd = candidate(rows)
    assert not pd['rank_eligible'] and pd['comparison_status'] == 'clock_evidence_unqualified'
    assert pd['observed_rank_eligible'] and pd['observed_energy_rank'] == 5
    assert pd['observed_best_feasible_baseline'] == 'mixed'
    assert pd['observed_pdblend_saving_vs_best_feasible_baseline'] == pytest.approx(-.4)
    assert pd['observed_available_baseline_systems'] == sorted(BASELINES)
    for previous, row in zip(before, rows):
        assert {key:row[key] for key in previous} == previous
    strict = deepcopy(rows)
    add_observed_comparisons(rows, baseline_systems=BASELINES)
    for previous, row in zip(strict, rows):
        assert {k:v for k,v in row.items() if not k.startswith('observed_')} == {
            k:v for k,v in previous.items() if not k.startswith('observed_')}


@pytest.mark.parametrize('changes', [dict(evidence_valid=False), dict(formal_eligible=False),
    dict(evidence_valid=False, formal_eligible=False, status='invalid_measurement',
         common_clock_evidence='fail'), dict(baseline_frozen=False)])
def test_invalid_or_unfrozen_eco_is_not_upgraded_into_observed_rank(changes):
    rows = ranking_fixture();eco = next(r for r in rows if r['system'] == 'ecoserve')
    eco.update(changes, energy_service_j=1.);rank_rows(rows);pd = candidate(rows)
    assert not eco['observed_rank_eligible'] and eco['observed_available_baseline_energy_rank'] is None
    assert not pd['observed_rank_eligible']
    assert 'ecoserve' not in pd['observed_available_baseline_systems']
    assert pd['observed_best_available_feasible_baseline'] == 'mixed'
    for key, value in changes.items():assert eco[key] == value


def test_distinct_revisions_keep_same_frozen_baselines_and_tail_reversal():
    rows = ranking_fixture();rows[0]['common_clock_evidence'] = 'unknown'
    for row in rows:row['energy_tail_j'] = 1.
    first = candidate(rows);first.update(revision='v1', energy_service_j=9., energy_tail_j=0.)
    second = dict(deepcopy(first), revision='v2', energy_service_j=8., energy_tail_j=10.)
    rows.append(second);rank_rows(rows)
    assert first['observed_energy_rank'] == second['observed_energy_rank'] == 1
    assert first['observed_comparison_baseline_receipts'] == second['observed_comparison_baseline_receipts']
    assert len(second['observed_comparison_baseline_receipts']) == 4
    assert first['observed_tail_reverses_saving'] is False
    assert second['observed_tail_reverses_saving'] is True
    assert second['observed_pdblend_saving_with_tail_vs_best_feasible_baseline'] == pytest.approx(1-18/11)
    assert rows[0]['observed_energy_rank'] == ''
    assert rows[0]['observed_energy_rank_by_revision'] == {'v1':2, 'v2':2}
    assert all(not r['rank_eligible'] for r in rows)


@pytest.mark.parametrize('duplicate_system', ['mixed', 'pdblend'])
def test_repeated_valid_attempt_never_picks_the_lower_energy(duplicate_system):
    rows = ranking_fixture();rows[0]['common_clock_evidence'] = 'unknown'
    duplicate = deepcopy(next(r for r in rows if r['system'] == duplicate_system))
    duplicate['energy_service_j'] = .1;rows.append(duplicate);rank_rows(rows)
    pd = candidate(rows)
    assert not pd['observed_rank_eligible']
    assert pd['observed_comparison_status'] == 'ambiguous_attempts'
    assert pd['observed_pdblend_saving_vs_best_available_baseline'] is None


@pytest.mark.parametrize('mismatch', ['trace_sha256', 'runtime_source_sha256', 'measurement_source_sha256',
    'model_hash', 'tokenizer_hash', 'image_digest', 'gpu_uuids', 'slo_ttft_s'])
def test_partial_observed_comparison_requires_exact_bound_identity(mismatch):
    rows = [r for r in ranking_fixture() if r['system'] in ('mixed','pdblend')]
    rows[0]['common_clock_evidence'] = 'unknown'
    rank_rows(rows);pd = candidate(rows)
    assert pd['observed_comparison_status'] == 'incomplete'
    assert not pd['observed_rank_eligible']
    assert pd['observed_available_comparison_status'] == 'partial_single_observation'
    assert pd['observed_available_baseline_systems'] == ['mixed']
    assert pd['observed_pdblend_outperformed_by_available_baseline'] is True
    assert pd['observed_available_baseline_receipts']['mixed']['sha256'] == rows[0]['receipt_sha256']
    rows[0][mismatch] = ['GPU-other'] if mismatch == 'gpu_uuids' else 'other'
    rank_rows(rows)
    assert pd['observed_available_baseline_count'] == 0
    assert pd['observed_pdblend_saving_vs_best_available_baseline'] is None


def test_slo_failure_is_observed_but_not_a_feasible_energy_winner():
    rows = ranking_fixture();rows[0].update(slo_pass=False, energy_service_j=.01,
                                           common_clock_evidence='unknown')
    rank_rows(rows)
    assert not rows[0]['observed_rank_eligible']
    assert candidate(rows)['observed_best_feasible_baseline'] != 'mixed'
    assert rows[0]['baseline_frozen'] and rows[0]['slo_pass'] is False


def test_export_emits_observed_scope_without_changing_historical_values_or_clock(tmp_path):
    from test_resident_comparison import export_point, export_receipt, export_csv
    point = export_point();receipt = export_receipt(tmp_path/'session', point)
    before = receipt.read_bytes()
    summary, rows = export_csv(tmp_path, point, [tmp_path/'session'])
    assert summary['measured'] == 1 and len(rows) == 1
    row = rows[0]
    assert row['observed_comparison_scope'] == SCOPE
    assert row['common_clock_evidence'] == 'unknown'
    assert row['energy_service_j'] == '10.0' and row['evidence_valid'] == 'True'
    assert row['observed_rank_eligible'] == row['rank_eligible'] == 'False'
    assert row['observed_energy_rank'] == row['energy_rank'] == ''
    assert receipt.read_bytes() == before
