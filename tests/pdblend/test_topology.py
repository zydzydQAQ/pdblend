from dataclasses import replace

import pytest

from pdblend.control.topology import TopologyPlanner, candidate_topologies, require_profiles
from pdblend.control.planner import SLO
from pdblend.model_registry import ModelRegistry
from synthetic import fc, synthetic_model


def test_tp_only_candidates_do_not_project_illegal_pp_memory_layouts():
    registry = ModelRegistry()
    assert [(x.tp, x.pp) for x in candidate_topologies(registry.get('7b'))] == [(1, 1), (2, 1), (4, 1)]
    assert [(x.tp, x.pp) for x in candidate_topologies(registry.get('14b'))] == [(1, 1), (2, 1), (4, 1)]
    assert [(x.tp, x.pp) for x in candidate_topologies(registry.get('32b'))] == [(2, 1), (4, 1)]
    assert [(x.tp, x.pp) for x in candidate_topologies(registry.get('32b'), gpu_budget=4)] == [(2, 1)]
    with pytest.raises(ValueError, match='PP1'):
        candidate_topologies(registry.get('7b'), include_pp=True)


def test_resident_m_pool_allows_single_tp_budget_without_pd_pair():
    registry = ModelRegistry()
    assert [(x.tp, x.pp) for x in candidate_topologies(
        registry.get('7b'), gpu_budget=3, require_pd_pair=False)] == [(1, 1), (2, 1)]
    assert [(x.tp, x.pp) for x in candidate_topologies(
        registry.get('32b'), gpu_budget=6, require_pd_pair=False)] == [(2, 1), (4, 1)]
    # The default remains the conservative P/D-pair space.
    assert [(x.tp, x.pp) for x in candidate_topologies(
        registry.get('32b'), gpu_budget=6)] == [(2, 1)]


def test_offline_search_returns_real_role_counts_and_standby_cost():
    spec = ModelRegistry().get('7b')
    profiles = {(tp, 1): replace(synthetic_model(), tp=tp) for tp in (1, 2, 4)}
    planner = TopologyPlanner(spec, profiles)
    plan = planner.search(fc(.5), SLO(5, .15))
    assert plan.detail['replicas'] == sum(plan.counts.values())
    assert plan.detail['gpu_count'] <= 8
    assert plan.detail['candidate_count'] == 3
    assert plan.detail['standby_power_w'] >= 0
    assert not plan.detail['formal_eligible']
    fixed = planner.search(fc(.5), SLO(5, .15), mode='fixed_tp', fixed_topology=(2, 1))
    assert (fixed.tp, fixed.pp) == (2, 1)
    with pytest.raises(ValueError, match='explicit fixed_topology'):
        planner.search(fc(.5), SLO(5, .15), mode='fixed_tp')


def test_search_rejects_cross_system_and_cross_topology_profiles():
    spec = ModelRegistry().get('7b')
    with pytest.raises(ValueError, match="another system"):
        TopologyPlanner(spec, {(1, 1): replace(synthetic_model(), system='dynamollm')})
    with pytest.raises(ValueError, match='lookup key'):
        TopologyPlanner(spec, {(2, 1): synthetic_model()})
    with pytest.raises(ValueError, match='identity'):
        TopologyPlanner(spec, {(1, 1): replace(synthetic_model(), profile_key={
            'system': 'pdblend', 'model_id': 'Qwen2.5-14B-Instruct', 'tp': 1, 'pp': 1})})


def test_missing_profiles_match_only_feasible_pdblend_tp_space():
    spec = ModelRegistry().get('32b')
    profiles = {(tp, 1): replace(synthetic_model(), tp=tp) for tp in (2, 4)}
    require_profiles(spec, profiles)
    with pytest.raises(ValueError, match='missing profile'):
        require_profiles(spec, {(2, 1): profiles[2, 1]})
