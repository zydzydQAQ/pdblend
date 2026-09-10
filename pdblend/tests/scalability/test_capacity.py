import copy

import pytest

from ecopadg.scalability import capacity as c
from ecopadg.scalability import protocol as p


def advance(state, threshold=1.0, **overrides):
    action = c.next_action(state)
    obs = dict(action, measurement_valid=True, capacity_pass=action["rate_rps"] <= threshold)
    obs.pop("action")
    obs.update(overrides)
    return c.record_observation(state, obs)


def pilot_complete(**kwargs):
    state = c.new_search(system="pdblend", dataset="sharegpt", n_gpus=3, **kwargs)
    for _ in range(100):
        if state["status"] != "pilot":
            return state
        state = advance(state)
    pytest.fail("pilot did not converge")


def test_doubling_bisection_and_independent_five_seed_confirmation():
    state = pilot_complete()
    lo, hi = state["lower_rate_rps"], state["upper_rate_rps"]
    assert lo <= 1 < hi and (hi - lo) / lo <= .05
    assert [o["rate_rps"] for o in state["observations"][:4]] == [.25, .5, 1, 2]
    assert all(o["seed"] == 9701 for o in state["observations"])
    for _ in range(10):
        state = advance(state)
    assert state["status"] == "complete" and state["confirmed"] is True
    assert {o["seed"] for o in state["observations"] if o["stage"] == "capacity"} == set(p.FORMAL_SEEDS)
    assert state["lower_rate_rps"] == lo and state["upper_rate_rps"] == hi
    assert c.next_action(state)["action"] == "complete"


def test_search_shrinks_initial_failure_until_positive_lower_bound():
    state = pilot_complete(initial_rate_rps=4)
    assert [o["rate_rps"] for o in state["observations"][:3]] == [4, 2, 1]
    assert (state["upper_rate_rps"] - state["lower_rate_rps"]) / state["lower_rate_rps"] <= .05


def test_engineering_failure_blocks_without_changing_capacity_or_losing_energy():
    state = c.new_search(system="mixed", dataset="sharegpt", n_gpus=4)
    original = copy.deepcopy(state)
    failed = c.record_observation(state, dict(measurement_valid=False, capacity_pass=False,
                                             energy_j=125, error="power sampling failed"))
    assert state == original
    assert failed["status"] == "blocked"
    assert failed["lower_rate_rps"] is failed["upper_rate_rps"] is None
    assert failed["observations"][0]["energy_j"] == 125
    with pytest.raises(ValueError, match="terminal"):
        c.record_observation(failed, dict(measurement_valid=True, capacity_pass=True))


def test_failed_formal_low_endpoint_refines_downward_without_deleting_failure():
    state = pilot_complete()
    pilot_lo = state["lower_rate_rps"]
    for _ in range(10):
        action = c.next_action(state)
        state = advance(state, threshold=.75 if action["seed"] == p.FORMAL_SEEDS[0] else 1.)
    action = c.next_action(state)
    assert state["status"] == "confirm" and action["boundary"] == "refine"
    assert action["seed"] == p.FORMAL_SEEDS[0] and action["rate_rps"] == pilot_lo / 2
    original = copy.deepcopy(state["observations"])
    for _ in range(100):
        if state["status"] != "confirm":
            break
        action = c.next_action(state)
        state = advance(state, threshold=.75 if action["seed"] == p.FORMAL_SEEDS[0] else 1.)
    assert state["status"] == "complete" and state["confirmed"]
    assert state["observations"][:len(original)] == original
    bracket = state["formal_seed_brackets"][str(p.FORMAL_SEEDS[0])]
    assert bracket["lower_rate_rps"] <= .75 < bracket["upper_rate_rps"]
    assert bracket["relative_width"] <= .05
    assert any(not o["capacity_pass"] and o["rate_rps"] == pilot_lo and o["seed"] == p.FORMAL_SEEDS[0]
               for o in state["observations"] if o["stage"] == "capacity")


def test_all_failed_formal_endpoints_keep_halving_same_seed():
    state = pilot_complete()
    for _ in range(10):
        state = advance(state, threshold=0.)
    rates = []
    for _ in range(3):
        action = c.next_action(state)
        rates.append(action["rate_rps"])
        assert action["seed"] == p.FORMAL_SEEDS[0] and action["boundary"] == "refine"
        state = advance(state, threshold=0.)
    assert rates == [state["lower_rate_rps"] / divisor for divisor in (2, 4, 8)]
    assert state["status"] == "confirm" and not state["confirmed"]
    assert len([o for o in state["observations"] if o["stage"] == "capacity"]) == 13


def test_passing_upper_endpoint_expands_upward_and_refines_each_seed():
    state = pilot_complete()
    pilot_hi = state["upper_rate_rps"]
    for _ in range(10):
        state = advance(state, threshold=2.)
    action = c.next_action(state)
    assert action["rate_rps"] == pilot_hi * 2 and action["boundary"] == "refine"
    for _ in range(200):
        if state["status"] != "confirm":
            break
        state = advance(state, threshold=2.)
    assert state["confirmed"]
    assert all(b["lower_rate_rps"] <= 2 < b["upper_rate_rps"] and b["relative_width"] <= .05
               for b in state["formal_seed_brackets"].values())
    assert all(any(o["boundary"] == "upper" and o["capacity_pass"] and o["seed"] == seed
                   for o in state["observations"] if o["stage"] == "capacity") for seed in p.FORMAL_SEEDS)


def test_nonmonotonic_formal_seed_is_inconclusive_instead_of_selective_retry():
    state = pilot_complete()
    for _ in range(10):
        action = c.next_action(state)
        if action["seed"] == p.FORMAL_SEEDS[0]:
            state = advance(state, capacity_pass=action["boundary"] == "upper")
        else:
            state = advance(state)
    assert state["status"] == "confirmation_failed" and not state["confirmed"]
    assert state["formal_seed_brackets"][str(p.FORMAL_SEEDS[0])]["status"] == "inconclusive"
    assert len([o for o in state["observations"] if o["stage"] == "capacity"]) == 10
    with pytest.raises(ValueError, match="terminal"):
        c.record_observation(state, dict(measurement_valid=True, capacity_pass=True))


def test_engineering_failure_during_refinement_stops_with_full_history():
    state = pilot_complete()
    for _ in range(10):
        state = advance(state, threshold=.5)
    previous = copy.deepcopy(state["observations"])
    state = advance(state, measurement_valid=False, error="transport failure", energy_j=900.)
    assert state["status"] == "blocked"
    assert state["observations"][:-1] == previous
    assert state["observations"][-1]["energy_j"] == 900.


def test_observation_cannot_be_misattributed_to_another_seed_or_rate():
    state = c.new_search(system="pdblend", dataset="sharegpt", n_gpus=3)
    for wrong in [dict(seed=701), dict(rate_rps=99), dict(system="mixed")]:
        with pytest.raises(ValueError, match="next action"):
            c.record_observation(state, dict(measurement_valid=True, capacity_pass=True, **wrong))


def pilot_rows(dataset):
    return [dict(system=system, dataset=dataset, n_gpus=n, stage="pilot", seed=9701,
                 capacity_rps=n * (1 + index), measurement_valid=True, pilot_interval_complete=True)
            for index, system in enumerate(p.SYSTEMS) for n in (3, 4)]


def test_weak_load_uses_minimum_of_all_systems_and_own_dataset():
    rows = pilot_rows("sharegpt")
    assert c.weak_load_q(rows, dataset="sharegpt") == pytest.approx(.7)
    with pytest.raises(ValueError, match="same dataset"):
        c.weak_load_q(rows, dataset="longbench")
    with pytest.raises(ValueError, match="same dataset"):
        c.weak_load_q(rows[:-1], dataset="sharegpt")
    with pytest.raises(ValueError, match="duplicate"):
        c.weak_load_q(rows + rows[:1], dataset="sharegpt")
    rows[0]["seed"] = 701
    with pytest.raises(ValueError, match="pilot"):
        c.weak_load_q(rows, dataset="sharegpt")


def test_longbench_n3_stops_at_pilot_without_creating_illegal_formal_cells():
    state = c.new_search(system="pdblend", dataset="longbench", n_gpus=3)
    for _ in range(100):
        if state["status"] != "pilot":
            break
        state = advance(state)
    assert state["status"] == "complete" and state["confirmed"] is False
    row = c.pilot_capacity_row(state)
    assert row["stage"] == "pilot" and row["n_gpus"] == 3
    assert all(o["stage"] == "pilot" for o in state["observations"])
