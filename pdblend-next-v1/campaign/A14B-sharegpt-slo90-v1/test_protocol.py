import copy
from decimal import Decimal, localcontext

import pytest

import protocol as p


def summary(good=9, offered=10, **overrides):
    value = dict(measurement_valid=True, fixed_window_valid=True,
                 post_measurement_cleanup=dict(cleanup_complete=True),
                 offered_requests=offered, good_requests=good,
                 slo_attainment=good / offered)
    value.update(overrides)
    return value


def test_exact_ninety_continues_and_first_lower_stops_one_scale():
    old = p.new_state()
    result = p.record_pdb(old, 0.5, "0.4", summary(), technical_valid=True)
    assert old == p.new_state()
    assert p.next_rate(result, 0.5) == "0.6"
    result = p.record_pdb(result, 0.5, "0.6", summary(8), technical_valid=True)
    assert p.next_rate(result, 0.5) is None
    assert result["scales"]["0.5"]["endpoint_rate"] == "0.6"
    assert p.required_baseline_rates(result, 0.5) == ["0.4", "0.6"]
    assert p.required_baseline_rates(result, 2) == []
    assert p.next_rate(result, 2) == "0.4"
    assert not p.pdb_complete(result)
    with pytest.raises(ValueError, match="endpoint"):
        p.record_pdb(result, 0.5, "0.8", summary(10), technical_valid=True)


@pytest.mark.parametrize("overrides", [
    {"measurement_valid": False}, {"fixed_window_valid": False},
    {"runtime_error": "HTTP503 upstream"}, {"http_503_engineering_count": 1},
    {"post_measurement_cleanup": {"cleanup_complete": False}},
    {"incomplete_drain": True}, {"sampling_error": "lost sensor"},
])
def test_technical_failures_never_advance_or_define_endpoint(overrides):
    result = p.record_pdb(p.new_state(), 2, "0.4", summary(0, **overrides), technical_valid=True)
    track = result["scales"]["2"]
    assert track["status"] == "blocked" and track["next_index"] == 0
    assert track["endpoint_rate"] is None and not track["observations"]
    assert len(track["technical_attempts"]) == 1
    assert p.next_rate(result, 2) is None
    assert not p.required_baseline_rates(result, 2)


def test_raw_technical_proof_is_required_and_failed_attempt_is_retained():
    blocked = p.record_pdb(p.new_state(), 2, "0.4", summary(0))
    assert blocked["scales"]["2"]["status"] == "blocked"
    repaired = p.record_pdb(blocked, 2, "0.4", summary(10), technical_valid=True,
                            evidence={"receipt_sha256": "a" * 64})
    assert p.next_rate(repaired, 2) == "0.6"
    assert len(repaired["scales"]["2"]["technical_attempts"]) == 1


def test_valid_timeouts_and_rejections_count_as_slo_failures():
    result = p.record_pdb(p.new_state(), 0.5, "0.4",
                          summary(6, request_timeouts=2, admission_rejections=2, work_complete=False),
                          technical_valid=True)
    assert result["scales"]["0.5"]["status"] == "stopped"
    assert p.required_baseline_rates(result, 0.5) == ["0.4"]


def test_raw_audited_explicit_admission_503_is_not_blanket_engineering_failure():
    result = p.record_pdb(p.new_state(), 0.5, "0.4",
                          summary(6, http_503_count=4, admission_rejections=4, work_complete=False),
                          technical_valid=True)
    assert result["scales"]["0.5"]["status"] == "stopped"


def test_invalid_or_inconsistent_denominators_are_technical_failures():
    for overrides in ({"offered_requests": 0}, {"good_requests": 11},
                      {"good_requests": True}, {"slo_attainment": 0.899}):
        result = p.record_pdb(p.new_state(), 0.5, "0.4", summary(**overrides), technical_valid=True)
        assert result["scales"]["0.5"]["status"] == "blocked"
    state = p.new_state()
    with pytest.raises(ValueError, match="declared order"):
        p.record_pdb(state, 2, "0.6", summary(), technical_valid=True)


def test_rate_extension_is_exact_and_not_limited_to_twelve_rates():
    assert [p.rate_at(i) for i in range(15)] == [*p.INITIAL_RATES, "9", "13.5", "20.25"]
    with localcontext() as context:
        context.prec = 6
        text = p.rate_at(111)
    with localcontext() as context:
        context.prec = 300
        assert Decimal(text) == Decimal(6) * Decimal("1.5") ** 100
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            p.rate_at(bad)


def test_both_scales_have_independent_pdb_endpoints_and_baseline_prefixes():
    state = p.record_pdb(p.new_state(), 0.5, "0.4", summary(0), technical_valid=True)
    for index in range(3):
        state = p.record_pdb(state, 2, p.rate_at(index), summary(10), technical_valid=True)
    state = p.record_pdb(state, 2, "1", summary(8), technical_valid=True)
    assert p.pdb_complete(state)
    assert p.required_baseline_rates(state, 0.5) == ["0.4"]
    assert p.required_baseline_rates(state, 2) == ["0.4", "0.6", "0.8", "1"]


def test_declaration_slo_and_idle_host_scope():
    declaration = p.declaration()
    assert p.effective_slo(0.5) == dict(ttft_s=2.5, tpot_s=0.075, attainment_target=0.9)
    assert p.effective_slo(2) == dict(ttft_s=10.0, tpot_s=0.3, attainment_target=0.9)
    assert declaration["no_interrupt_active_campaigns"] is True
    assert declaration["whole_campaign_single_host"] is True
    assert declaration["execution_host"] is None
    assert declaration["scale_one_required"] is False
    for bad in (1, "nan", 0, -2, True):
        with pytest.raises(ValueError):
            p.scale_key(bad)
