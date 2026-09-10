import pytest

from ecopadg.scalability.statistics import backlog_stability, paired_ratio_interval, capacity_interval


def test_backlog_stable_and_material_growth_are_distinguished():
    steady = backlog_stability([dict(t_s=t, pending=5) for t in range(1000, 1301)], 1300)
    rising = backlog_stability([dict(t_s=t, pending=.05 * (t - 1000)) for t in range(1000, 1301)], 1300)
    assert steady["valid"] and steady["upper_rps"] < 1e-10
    assert rising["valid"] and rising["upper_rps"] == pytest.approx(.05)


def test_backlog_requires_complete_recent_arrival_window():
    assert not backlog_stability([dict(t_s=t, pending=0) for t in range(1100, 1301)], 1300)["valid"]
    assert not backlog_stability([dict(t_s=t, pending=0) for t in range(1000, 1301, 10)], 1300)["valid"]


def test_paired_bootstrap_preserves_per_seed_relationship():
    result = paired_ratio_interval([2, 4, 6, 8, 10], [1, 2, 3, 4, 5], scale=2)
    assert result["estimate"] == 1
    assert result["ci95_low"] == result["ci95_high"] == 1
    assert paired_ratio_interval([2], [1])["ci95_low"] is None
    with pytest.raises(ValueError):
        paired_ratio_interval([1, 2], [1])


def test_capacity_interval_preserves_invalid_and_inconsistent_measurements():
    def row(rate, passed, valid=True):
        return dict(rate_rps=rate, capacity_pass=passed, measurement_valid=valid, stage="capacity")
    bracket = capacity_interval([row(10, True), row(10.4, False), row(20, True, False)])
    assert bracket["bracket_complete"] and bracket["capacity_lower_rps"] == 10
    assert bracket["invalid_runs"] == 1
    assert capacity_interval([row(10, True)])["upper_censored"]
    assert capacity_interval([row(10, True), row(9, False)])["inconsistent"]
    assert capacity_interval([row(10, True), row(10, False)])["inconsistent"]
