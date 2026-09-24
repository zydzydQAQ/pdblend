import pytest

from pdblend.online.router import RequestRecord
from pdblend.online.shield import Pressure, Shield
from pdblend.planner.pool import Plan, SLO


def record(name='r', *, output=100, tokens=20, last=99.7, path='PD'):
    return RequestRecord(name, path, 'p', 'd', 100, output, 98., first_token_s=99.,
                         tokens_so_far=tokens, last_token_s=last)


def shield(**kwargs):
    return Shield(SLO(15., .1), mode='budget_aware', **kwargs)


def test_one_transient_gap_does_not_keep_expanding_an_amortised_long_decode():
    row = record()
    current = shield()
    pressure = current.observe([row], 100.)
    assert pressure.decode_stalled == 1 and pressure.decode_stalled_fraction == 1.
    assert not pressure.decode and pressure.decode_budget_risks == 0
    for now in (100., 100.1, 100.2):
        current.update(current.observe([row], now), now)
    assert current.level == 0
    assert Shield(SLO(15., .1)).observe([row], 100.).decode  # explicit legacy control


def test_two_token_carry_reply_is_protected_before_the_first_decode_token():
    p = shield().observe([record(output=2, tokens=1, last=99.)], 99.09)
    assert p.decode and p.decode_short_output_risks == 1
    assert p.decode_paths == ('PD',)


def test_multi_sequence_stall_requires_configured_count_and_fraction():
    rows = [record('a'), record('b')]
    rows += [record(str(i), last=99.99, path='M') for i in range(6)]
    p = shield().observe(rows, 100.)
    assert p.decode and p.decode_stalled_fraction == .25
    assert p.decode_paths == ('PD',)
    assert not shield(stalled_fraction=.5).observe(rows, 100.).decode


def test_sustained_stall_and_unrecoverable_output_budget_remain_protected():
    stalled = shield().observe([record(last=98.9)], 100.1)
    assert stalled.decode and stalled.decode_sustained_stalls == 1
    budget = shield(sustained_gap_s=5).observe([record(output=10, tokens=8, last=99.7)], 100.)
    assert budget.decode and budget.decode_budget_risks == 1


def test_completed_tpot_violation_and_old_prefill_stuck_are_still_visible():
    done = record(output=3, tokens=3, last=99.3, path='M')
    done.finished_s, done.completion_tokens = 99.3, 3
    old = RequestRecord('old', 'PD', 'p', 'd', 100, 10, 0.)
    p = shield().observe([done, old], 100.)
    assert p.decode and p.tpot_p90 == pytest.approx(.15)
    assert p.prefill and p.stuck == 1
    assert p.prefill_paths == ('PD',) and p.decode_paths == ('M',)
    s = shield()
    s.update(p, 100.)
    assert s.events[-1]['pressure']['criteria']['sustained_gap_s'] == 1.


def test_one_token_response_has_no_decode_stall_budget():
    p = shield().observe([record(output=1, tokens=1, last=99.)], 101.)
    assert not p.decode and p.decode_active == p.decode_stalled == 0


def test_delayed_terminal_confirmation_is_not_a_completed_tpot_violation():
    row = record(output=2, tokens=2, last=99.05)
    waiting_terminal = shield().observe([row], 100.)
    assert not waiting_terminal.decode and waiting_terminal.decode_active == 0
    row.finished_s, row.completion_tokens = 99.5, 2
    pressure = shield().observe([row], 100.)
    assert pressure.tpot_p90 == pytest.approx(.05)
    assert not pressure.decode
    row.last_token_s = None
    assert shield().observe([row], 100.).decode  # conservative missing-token-time fallback


@pytest.mark.parametrize('kwargs', [dict(mode='unknown'), dict(sustained_gap_s=0),
    dict(stalled_fraction=0), dict(stalled_min_requests=1)])
def test_invalid_modes_and_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Shield(SLO(15., .1), **kwargs)


def deployment(counts=None):
    return Plan(counts or {'M': 4, 'off': 4}, 1500, 1200, 900, 0, 800., .8, .03,
                tp=2, pp=1, pool_id='mixed-tp2', generation=7, profile_key='bound-profile')


def strong_m_decode():
    return Pressure(decode=True, decode_paths=('M',), tpot_p90=.12, mode='budget_aware')


def test_capacity_is_added_once_per_real_escalation_not_per_apply_or_hold():
    s, pressure, base = shield(), strong_m_decode(), deployment()
    assert s.update(pressure, 100.) == 1
    plan = s.apply(base, pressure, 2520)
    assert plan.counts['M'] == 4
    assert s.update(pressure, 102.) == 2
    plan = s.apply(plan, pressure, 2520)
    assert plan.counts['M'] == s.capacity_target_active == s.floor_active == 5
    for _ in range(10):
        plan = s.apply(plan, Pressure(), 2520)
        assert plan.counts['M'] == 5
    # A planner candidate with fewer instances respects the same absolute target.
    plan = s.apply(deployment({'M': 2, 'off': 6}), Pressure(), 2520)
    assert plan.counts['M'] == 5
    assert s.update(pressure, 104.) == 3
    plan = s.apply(plan, pressure, 2520)
    assert plan.counts['M'] == 6
    assert s.apply(plan, pressure, 2520).counts['M'] == 6
    assert (plan.tp, plan.pp, plan.pool_id, plan.generation, plan.profile_key) == (
        base.tp, base.pp, base.pool_id, base.generation, base.profile_key)
    assert base.counts == {'M': 4, 'off': 4}


def test_pending_escalations_are_consumed_once_and_never_exceed_slots():
    s, pressure = shield(), strong_m_decode()
    for now in (100., 102., 104.):
        s.update(pressure, now)
    plan = s.apply(deployment({'M': 7, 'off': 1}), pressure, 2520)
    assert plan.counts['M'] == s.capacity_target_active == 8
    assert s.apply(plan, pressure, 2520).counts['M'] == 8
    assert all(v >= 0 for v in plan.counts.values()) and sum(plan.counts.values()) == 8


def test_planner_capacity_growth_is_not_a_new_shield_capacity_event():
    s, pressure = shield(), strong_m_decode()
    s.update(pressure, 100.)
    s.apply(deployment(), pressure, 2520)
    s.update(pressure, 102.)
    s.apply(deployment(), pressure, 2520)
    bigger = s.apply(deployment({'M': 7, 'off': 1}), pressure, 2520)
    assert bigger.counts['M'] == 7
    assert s.capacity_target_active == s.floor_active == 5
    # A later genuine pressure escalation may add capacity to that larger pool.
    s.update(pressure, 104.)
    assert s.apply(bigger, pressure, 2520).counts['M'] == 8


def test_collective_short_gaps_get_one_bounded_clock_probe_without_capacity():
    s, base = shield(cooldown_s=5., protect_s=60.), deployment()
    pressure = s.observe([record('a', path='M'), record('b', path='M')], 100.)
    assert pressure.decode and s._collective_only(pressure)
    assert s.update(pressure, 100.) == 1
    plan = s.apply(base, pressure, 2520)
    assert plan.f_M == 2520 and plan.counts['M'] == 4
    for now in (102., 104.):
        assert s.update(pressure, now) == 1
        plan = s.apply(plan, pressure, 2520)
        assert plan.counts['M'] == 4
    assert s.floor_active == 0 and not s.protection_active(104.)
    assert s.update(pressure, 105.) == 0
    for now in (107., 109., 120.):
        assert s.update(pressure, now) == 0
    released = s.apply(base, pressure, 2520)
    assert released.f_M == 900 and released.counts == base.counts
    assert not any(e.get('capacity_escalation') for e in s.events)


def test_collective_short_gap_does_not_fail_or_restore_a_capacity_probe():
    s = shield(cooldown_s=5., floor_active=2, peak_active=3, release_s=99.)
    pressure = s.observe([record('a'), record('b')], 100.)
    s.update(pressure, 100.)
    assert s.floor_active == 2 and s.probe_windows == 1.
    s.apply(deployment({'M': 2, 'off': 6}), pressure, 2520)
    s.update(pressure, 105.)
    s.update(pressure, 110.)
    assert s.floor_active == 1
    assert not any(e.get('probe') == 'failed' for e in s.events)


def test_true_pressure_can_escalate_after_a_collective_clock_probe():
    s = shield()
    short = s.observe([record('a', path='M'), record('b', path='M')], 100.)
    s.update(short, 100.)
    plan = s.apply(deployment(), short, 2520)
    real = strong_m_decode()
    assert s.update(real, 102.) == 2
    assert s.apply(plan, real, 2520).counts['M'] == 5


@pytest.mark.parametrize('pressure,pool', [
    (Pressure(prefill=True, prefill_paths=('M',)), 'M'),
    (Pressure(decode=True, decode_paths=('M',), tpot_p90=.12), 'M'),
    (Pressure(prefill=True, prefill_paths=('PD',)), 'P'),
    (Pressure(decode=True, decode_paths=('PD',), tpot_p90=.12), 'D'),
])
def test_pressure_path_targets_its_own_pool_and_clock(pressure, pool):
    s, base = shield(), deployment({'M': 2, 'P': 1, 'D': 1, 'off': 4})
    s.update(pressure, 100.)
    first = s.apply(base, pressure, 2520)
    s.update(pressure, 102.)
    plan = s.apply(first, pressure, 2520)
    for role in ('M', 'P', 'D'):
        assert plan.counts[role] == base.counts[role] + int(role == pool)
        assert getattr(plan, 'f_' + role) == (2520 if role == pool else getattr(base, 'f_' + role))
    assert s.apply(plan, Pressure(), 2520).counts == plan.counts


def test_released_floor_is_not_resurrected_by_the_previous_capacity_target():
    s, pressure = shield(cooldown_s=5.), strong_m_decode()
    s.update(pressure, 100.)
    s.apply(deployment({'M': 2, 'off': 6}), pressure, 2520)
    s.update(pressure, 102.)
    assert s.apply(deployment({'M': 2, 'off': 6}), pressure, 2520).counts['M'] == 3
    calm = Pressure()
    assert s.update(calm, 107.) == 1
    assert s.update(calm, 112.) == 0
    assert s.capacity_target_active == 0
    lean = deployment({'M': 1, 'off': 7})
    assert s.apply(lean, calm, 2520).counts['M'] == 3
    s.update(calm, 117.)
    assert s.apply(lean, calm, 2520).counts['M'] == 2
    assert s.update(pressure, 118.) == 1
    assert s.floor_active == 3 and s.probe_windows == 2.
    s.update(calm, 123.)
    s.update(calm, 133.)
    released = s.apply(lean, calm, 2520)
    assert released.counts['M'] == 2 and released.f_M == lean.f_M
    assert released.detail['shield_level'] == 0


def test_legacy_shield_keeps_its_original_escalation_behavior():
    s = Shield(SLO(15., .1), level=2)
    first = s.apply(deployment(), Pressure(decode=True), 2520)
    second = s.apply(first, Pressure(), 2520)
    assert first.counts['M'] == 5 and second.counts['M'] == 6
    assert (first.f_M, first.f_P, first.f_D) == (2520, 2520, 2520)
