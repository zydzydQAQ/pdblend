from copy import deepcopy

from pdblend.bench.comparison_campaign import rank_rows
from test_resident_comparison import ranking_fixture


def pair():
    rows=ranking_fixture()
    return [r for r in rows if r['system'] in ('mixed','pdblend')]


def test_existing_baseline_can_disprove_advantage_without_complete_ranking():
    rows=rank_rows(pair());mixed,pd=rows
    assert not pd['rank_eligible'] and pd['energy_rank']==''
    assert pd['comparison_status']=='incomplete'
    assert pd['available_comparison_status']=='partial_single_observation'
    assert pd['available_baseline_count']==1
    assert pd['available_baseline_systems']==['mixed']
    assert pd['available_baseline_receipts']['mixed']['path']==mixed['receipt_path']
    assert pd['available_candidate_energy_rank']==2
    assert pd['pdblend_outperformed_by_available_baseline'] is True
    assert pd['pdblend_saving_vs_best_available_baseline']<0


def test_slo_failure_is_not_a_feasible_interim_energy_winner():
    rows=pair();rows[0]['slo_pass']=False
    rank_rows(rows)
    assert rows[-1]['best_available_feasible_baseline']==''
    assert rows[-1]['pdblend_outperformed_by_available_baseline'] is None
    assert rows[0]['available_baseline_energy_rank'] is None


def test_identity_mismatch_and_unfrozen_baseline_are_excluded():
    for change in ({'runtime_source_sha256':'other'},{'baseline_frozen':False}):
        rows=pair();rows[0].update(change);rank_rows(rows)
        assert rows[-1]['available_baseline_count']==0
        assert rows[-1]['pdblend_saving_vs_best_available_baseline'] is None


def test_interim_ranking_never_best_picks_duplicate_observations():
    rows=pair();rows.append(deepcopy(rows[0]));rank_rows(rows)
    assert rows[-2]['available_comparison_status']=='ambiguous_baseline_attempts'
    assert rows[-2]['pdblend_saving_vs_best_available_baseline'] is None
    rows=pair();rows.append(deepcopy(rows[-1]));rank_rows(rows)
    assert rows[-1]['available_comparison_status']=='ambiguous_candidate_revision'
    assert rows[-1]['available_candidate_energy_rank'] is None


def test_revisions_remain_separate_and_tail_can_reverse_service_saving():
    rows=pair();rows[0]['energy_tail_j']=0
    other=deepcopy(rows[-1]);other.update(revision='v2',energy_service_j=5.,energy_tail_j=10.)
    rows.append(other);rank_rows(rows)
    assert rows[1]['pdblend_outperformed_by_available_baseline'] is True
    assert other['pdblend_outperformed_by_available_baseline'] is False
    assert other['pdblend_saving_vs_best_available_baseline']==.5
    assert other['pdblend_saving_with_tail_vs_best_available_baseline']==-.5
    assert other['available_tail_reverses_saving'] is True
    assert other['revision']=='v2'


def test_unknown_clock_does_not_create_an_interim_winner():
    rows=pair();rows[0]['common_clock_evidence']='unknown'
    rank_rows(rows)
    assert rows[-1]['clock_unqualified_baseline_systems']==['mixed']
    assert rows[-1]['available_baseline_count']==0
    assert rows[-1]['available_comparison_status']=='clock_evidence_unqualified'
    assert rows[-1]['pdblend_saving_vs_best_available_baseline'] is None
    assert rows[0]['baseline_frozen'] and rows[0]['evidence_valid']


def test_partial_clock_qualified_subset_keeps_excluded_baselines_explicit():
    rows=[r for r in ranking_fixture() if r['system'] in ('mixed','ecoserve','pdblend')]
    rows[0]['common_clock_evidence']='unknown'
    rank_rows(rows)
    candidate=next(r for r in rows if r['system']=='pdblend')
    assert candidate['available_baseline_systems']==['ecoserve']
    assert candidate['clock_unqualified_baseline_systems']==['mixed']
    assert candidate['available_comparison_status']=='partial_single_observation'
    assert candidate['best_available_feasible_baseline']=='ecoserve'
