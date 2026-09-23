import pytest

from pdblend.profile.parallel import evaluate_interference, partition_frequencies, qualified


def test_partition_frequencies_is_deterministic_and_disjoint():
    assert partition_frequencies((900, 1200, 1500, 1800, 2100), 2) == (
        (900, 1500, 2100), (1200, 1800))


def test_interference_requires_both_timing_and_power_within_limit():
    result = evaluate_interference(
        {"step_seconds": 1.0, "power_w": 100.0},
        {"step_seconds": 1.04, "power_w": 95.1})
    assert result["passed"]
    assert not evaluate_interference(
        {"step_seconds": 1.0, "power_w": 100.0},
        {"step_seconds": 1.06, "power_w": 100.0})["passed"]


def test_qualified_requires_complete_validation():
    assert qualified({"complete": True, "isolated": {"step_seconds": 1},
                      "parallel": {"step_seconds": 1},
                      "validation": {"passed": True}})
    assert not qualified({"complete": False, "validation": {"passed": True}})


def test_qualified_accepts_profile_wave_receipt_shape():
    assert qualified({"complete": True, "isolated": [{"instances": []}],
                      "parallel": [{"instances": []}],
                      "comparisons": [{"passed": True}],
                      "overlapping_windows": True})
