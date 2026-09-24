"""Safety and overload paths must keep the deployment identity used in Plan.key."""
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from pdblend.online.controller import Controller
from pdblend.online.shield import Pressure, Shield
from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner, SLO
from synthetic import fc, synthetic_model


def identity(plan):
    return plan.tp, plan.pp, plan.pool_id, plan.generation, plan.profile_key


def bound_plan():
    return Plan({'M': 2, 'L1': 2}, 2520, 2520, 2520, 0, 500., 1., .1,
                tp=2, pp=1, pool_id='tp2', generation=7,
                profile_key='{"system":"pdblend","tp":2}')


@pytest.mark.parametrize('level,floor', [(1, 0), (0, 3), (2, 0)])
def test_shield_escalation_and_floor_wake_keep_all_plan_identity(level, floor):
    source = bound_plan()
    safe = Shield(SLO(5, .15), level=level, floor_active=floor).apply(source, Pressure(decode=True), 2520)
    assert identity(safe) == identity(source)
    assert source.counts == {'M': 2, 'L1': 2}
    assert safe.counts['M'] >= source.counts['M']


def test_identity_only_shield_copy_does_not_trigger_an_extra_hold_interval():
    current = bound_plan()
    candidate = Shield(SLO(5, .15), level=1).apply(current, Pressure(decode=True), 2520)
    controller = NS(plan_now=current, _last_plan_change_s=99., min_plan_hold_s=30.)
    chosen, reason = Controller._gate_plan_change(controller, candidate, 100.)
    assert chosen is candidate and reason == 'unchanged'
    # A missing TP tag previously made an identical physical plan look changed.
    _, broken_reason = Controller._gate_plan_change(controller, replace(candidate, tp=1, profile_key=''), 100.)
    assert broken_reason == 'minimum_hold'


def test_overload_fallback_keeps_model_topology_and_current_pool_generation(monkeypatch):
    model = replace(synthetic_model(), tp=2, pp=1, profile_key={'system': 'pdblend', 'tp': 2})
    planner = PoolPlanner(model, PlannerConfig(4, SLO(5, .15)))
    monkeypatch.setattr(planner, 'candidates', lambda demand: [])
    monkeypatch.setattr(planner, '_mixed_pool', lambda *args, **kwargs: dict(power_w=500, ttft_s=1, tpot_s=.1))
    result = planner.plan(fc(.1), bound_plan())
    assert identity(result) == (2, 1, 'tp2', 7, json.dumps(model.profile_key, sort_keys=True, separators=(',', ':')))
    assert result.counts == {'M': 4} and result.detail['fallback']


@pytest.mark.parametrize('with_initial', [False, True])
def test_cold_start_plan_uses_bound_native_topology(with_initial):
    model = NS(tp=2, pp=1, profile_key={'system': 'pdblend', 'tp': 2})
    spec = NS(pool_id='tp2', generation=7)
    ctl = NS(initial_plan=bound_plan() if with_initial else None, max_freq=2520,
             fleet=NS(instances={'i0': NS(spec=spec), 'i1': NS(spec=spec)}), planner=NS(model=model))
    result = Controller._fail_open_plan(ctl)
    assert identity(result)[:4] == (2, 1, 'tp2', 7)
    assert result.profile_key
    assert result.counts == {'M': 2} and result.detail['cold_start']
