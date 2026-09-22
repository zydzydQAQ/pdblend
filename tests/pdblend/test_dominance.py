"""Safety and compatibility regressions for the experimental policy."""
from dataclasses import replace

from pdblend.control.controller import Controller
from pdblend.control.planner import Plan, PlannerConfig, PoolPlanner, SLO
from pdblend.control.shield import Pressure, Shield
from pdblend.proxy.router import Router
from synthetic import fc, synthetic_model
from test_controller import FakeFleet, FakeGpus


def controller():
    fleet = FakeFleet(8)
    p = PoolPlanner(synthetic_model(), PlannerConfig(8, SLO(5, .15), min_m_instances=4))
    c = Controller(fleet, Router(fleet.instances), FakeGpus(), p,
                   Shield(p.cfg.slo, protect_s=60), dynamic_m_floor=True,
                   base_m_floor=4, transition_cooldown_s=45, down_plan_votes=2)
    c.plan_now = p.plan(fc(.2))
    c._m_floor_last_change_s = 1000
    return c


def test_floor_steps_and_emergency_restore():
    c = controller()
    calm = Pressure()
    demand = fc(.2)
    for now in (1010, 1020):
        c._update_strategy(demand, calm, 0, now, True)
        assert c._m_floor == 4
    c._update_strategy(demand, calm, 0, 1030, True)
    assert c._m_floor == 3
    for now in (1040, 1050, 1060):
        c._update_strategy(demand, calm, 0, now, True)
    assert c._m_floor == 2
    changed, urgent = c._update_strategy(demand, Pressure(decode=True), 1, 1061, False)
    assert changed and urgent and c._m_floor == c.planner.cfg.min_m_instances == 4
    assert not c._update_strategy(demand, calm, 0, 1062, False)[0]
    assert c._m_floor == 4


def test_no_floor_release_on_repeated_shield_ticks():
    c = controller()
    for now in range(1010, 1090):
        c._update_strategy(fc(.2), Pressure(), 0, now, False)
    assert c._m_floor == 4


def test_small_pool_is_checked_at_its_actual_clock_with_load_reserve():
    c = controller()
    c.planner.cfg.min_m_instances = 2
    demand = fc(10)
    for p in c.planner.candidates(demand):
        if p.counts.get('M', 0) < 4:
            q = c.planner.mixed_pressure(replace(demand, rate_rps=12.5), p.counts['M'], p.f_M)
            assert q['pressure'] <= .55


def test_protection_cooldown_and_quiet_windows_block_shrink():
    c = controller()
    c.plan_now = Plan({'M': 4, 'off': 4}, 2520, 2520, 2520, 0, 100, 1, .03)
    c._last_plan_change_s = 1000
    candidate = replace(c.plan_now, f_M=1800)
    c._quiet_windows = 2
    assert c._gate_plan_change(candidate, 1050, shield_protected=True)[1] == 'shield_protection'
    assert c._gate_plan_change(candidate, 1040)[1] == 'transition_cooldown'
    c._quiet_windows = 0
    assert c._gate_plan_change(candidate, 1070)[1] == 'slo_quiet_confirmation'
    c._quiet_windows = 2
    assert c._gate_plan_change(candidate, 1080)[1] == 'downshift_confirmation'
    assert c._gate_plan_change(candidate, 1090)[1] == 'confirmed_downshift'


def test_legacy_router_ignores_pressure_updates():
    r = Router(['m', 'p', 'd'], pd_threshold_tokens=0)
    r.set_roles(dict(m='M', p='P', d='D'))
    assert not r.set_pressure_state(m_pressure=2, decode_risk=True, now=100)
    assert r.choose(10)[0] == 'PD'


def test_pressure_requires_two_observed_windows_and_commits_capacity_split():
    c = controller()
    demand = fc(.05, in_mean=1500, inputs=[512, 2048] * 30)
    c.plan_now = c.planner.plan(demand)
    c._update_strategy(demand, Pressure(decode=True), 0, 1010, True)
    assert not c.router.pressure_state()['pd_active']
    c._update_strategy(demand, Pressure(decode=True), 0, 1011, False)
    assert not c.router.pressure_state()['pd_active']
    c._update_strategy(demand, Pressure(decode=True), 0, 1020, True)
    assert c.router.pressure_state()['pd_active']
    p = c.planner.plan(demand, c.plan_now)
    assert p.counts['P'] and p.counts['D'] and p.tau in (0, 1024)
