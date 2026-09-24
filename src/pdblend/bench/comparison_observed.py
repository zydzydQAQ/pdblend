"""Namespace the same rank arithmetic for systems as actually executed.

This is descriptive, single-observation evidence, not proof of clock-control
conformance or causal mechanism superiority. It does not reclassify invalid
observations or alter frozen measurements and strict comparison fields.
"""
from copy import deepcopy

from .comparison_ranking import rank_rows_core


SCOPE = 'system_as_executed_single_observation/v1'
RANK_FIELDS = (
    'rank_eligible', 'energy_rank', 'comparison_status', 'best_feasible_baseline',
    'pdblend_saving_vs_best_feasible_baseline',
    'pdblend_saving_with_tail_vs_best_feasible_baseline', 'tail_reverses_saving',
    'candidate_revision', 'energy_rank_by_revision', 'comparison_baseline_receipts',
    'baseline_energy_rank', 'available_comparison_status', 'available_baseline_count',
    'available_baseline_systems', 'available_baseline_receipts',
    'available_baseline_energy_rank', 'available_candidate_energy_rank',
    'best_available_feasible_baseline', 'pdblend_saving_vs_best_available_baseline',
    'pdblend_saving_with_tail_vs_best_available_baseline',
    'available_tail_reverses_saving', 'pdblend_outperformed_by_available_baseline',
)


def add_observed_comparisons(rows, *, baseline_systems):
    """Add only ``observed_*`` columns; keep original evidence/clock gates intact.

    Use the shared identity, evidence, SLO, revision, duplicate-attempt and tail
    rules. The only difference is that unknown physical clocks do not suppress
    descriptive comparisons of already valid observations. Clock values are
    never changed to pass, even in the isolated work copy.
    """
    work = deepcopy(rows)
    rank_rows_core(work, baseline_systems=baseline_systems, clock_predicate=lambda row: True)
    for original, result in zip(rows, work):
        original['observed_comparison_scope'] = SCOPE
        original.update({'observed_'+name: result[name] for name in RANK_FIELDS})
    return rows
