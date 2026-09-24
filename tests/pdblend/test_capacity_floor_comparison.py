"""Qualified-domain replay and CPU-only restore regressions; no GPU claims."""
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from pdblend.bench.comparison_campaign import binding
from pdblend.bench.comparison_pdblend_acceptance import _controller, _plan_capacity_floor
from pdblend.bench.pdblend_runtime_options import (
    capacity_floor_selection, comparison_capacity_floors, comparison_options, preflight_comparison_options, runtime_receipt,
)
from pdblend.planner.capacity import capacity_floor_decision, forecast_from_snapshot, forecast_snapshot
from pdblend.planner.forecast import InFlightWork
from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner, QualifiedCapacityFloor, SLO
from pdblend.planner.transitions import identity
from synthetic import fc, synthetic_model
from test_comparison_pdblend_acceptance import fixture, raw
from test_controller import make_controller
from test_optimization_acceptance import campaign as acceptance_campaign


def planner(slots=8):
    model = synthetic_model()
    floor = QualifiedCapacityFloor('synthetic', 1, 1, 1, (0, 1), (1, 4096), (1, 256),
        'synthetic-controlled-acceptance#sha256='+'a'*64, qualified=True,
        profile_key='{}', accepted_slo=(1., .1))
    return PoolPlanner(model, PlannerConfig(slots, SLO(1., .1), min_m_instances=4,
        capacity_floors=(floor,), capacity_floor_reserve_canonical=True,
        preserve_overload_capacity=True))


def selected(pl):
    return dict(profile_key='{}', frequencies=list(pl.model.freqs), capacity_floor=dict(
        schema='pdblend-capacity-floor-selection/v1', reserve_canonical=True, canonical_floor=4,
        qualified_m_frequency_mhz=max(pl.model.freqs),
        model=dict(model=pl.model.model, tp=pl.model.tp, pp=pl.model.pp, profile_key=pl.model.profile_key),
        slo=dict(ttft_s=pl.cfg.slo.ttft_s, tpot_s=pl.cfg.slo.tpot_s),
        floors=[asdict(q) for q in pl.cfg.capacity_floors]))


def logged(plan):
    return dict(counts=plan.counts, f_P=plan.f_P, f_D=plan.f_D, f_M=plan.f_M,
                tau=plan.tau, query_results=plan.detail)


def low_plan(pl, demand=None):
    value = pl.evaluate({'M': 1, 'L1': pl.cfg.slots-1}, 2520, 2520, 2520, 0,
                        demand or fc(.05), strict=False)
    assert value is not None
    return value


def test_plan_binds_complete_forecast_and_replays_qualified_floor():
    pl = planner()
    demand = replace(fc(.05), outputs=(32, 256), length_pairs=((512, 32), (512, 256)),
        backlog=(InFlightWork('old', 512, 128, 0, 4096, 'M', 'tp1'),), inflight=1)
    plan = low_plan(pl, demand)
    snapshot = plan.detail['capacity_floor']['forecast']
    assert forecast_from_snapshot(snapshot) == demand
    assert snapshot['length_pairs'] == [[512, 32], [512, 256]]
    assert snapshot['backlog'][0]['request_id'] == 'old'
    assert _plan_capacity_floor(logged(plan), selected(pl), 8) == 1


@pytest.mark.parametrize('demand,reason', [
    (fc(2.), 'rate_outside_accepted_domain'),
    (fc(.05, inputs=[4097]), 'input_outside_accepted_domain'),
    (replace(fc(.05), length_pairs=((512, 257),)), 'output_outside_accepted_domain'),
    (replace(fc(.05), backlog=(InFlightWork('old', 512, 257, branch='M'),)), 'backlog_outside_accepted_domain'),
])
def test_domain_exit_restores_canonical_m_without_changing_pd_or_threshold(demand, reason):
    pl = planner()
    plan = Plan({'P': 2, 'D': 2, 'M': 1, 'L1': 3}, 2100, 1500, 1500, 1024, 10., .1, .01)
    restored = pl.enforce_capacity_floor(plan, demand)
    assert restored.counts == {'P': 2, 'D': 2, 'M': 4}
    assert (restored.f_P, restored.f_D, restored.tau) == (2100, 1500, 1024)
    assert restored.f_M == 2520
    assert reason in restored.detail['capacity_floor']['rejected_floors'].values()
    assert _plan_capacity_floor(logged(restored), selected(pl), 8) == 4
    forged = logged(restored); forged['counts'] = plan.counts
    assert forged['counts']['M'] < _plan_capacity_floor(logged(restored), selected(pl), 8)


def test_comparison_reserve_rejects_pd_layout_that_cannot_restore_m():
    pl = planner()
    assert pl.evaluate({'P': 4, 'D': 2, 'M': 1, 'L1': 1}, 2520, 2520, 2520, 1024, fc(.05)) is None
    with pytest.raises(ValueError, match='restoration slots'):
        pl.enforce_capacity_floor(Plan({'P': 4, 'D': 2, 'M': 1, 'L1': 1},
            2520, 2520, 2520, 1024, 1, 1, .1), fc(2.))
    pl4 = planner(4)
    assert pl4.evaluate({'P': 1, 'D': 1, 'M': 1, 'L1': 1}, 2520, 2520, 2520, 1024, fc(.05)) is None
    assert pl4.enforce_capacity_floor(low_plan(pl4), fc(2.)).counts['M'] == 4


@pytest.mark.parametrize('tp,minimum', [(1, 6), (2, 5), (4, 3)])
def test_comparison_rejects_any_floor_above_canonical_restore_capacity(tmp_path, monkeypatch, tp, minimum):
    artifact = tmp_path/'floor.json'
    artifact.write_text(json.dumps(dict(kind='pdblend_capacity_floor_v1')))
    floor = planner().cfg.capacity_floors[0]
    monkeypatch.setattr('pdblend.planner.capacity.load_capacity_floors',
                        lambda *args, **kw: (floor, replace(floor, min_m_instances=minimum)))
    with pytest.raises(ValueError, match='only relax the canonical M reserve'):
        comparison_capacity_floors(artifact, model=replace(synthetic_model(), tp=tp))


@pytest.mark.parametrize('fault', ['missing_snapshot', 'short_snapshot', 'fake_match', 'profile', 'stricter_slo', 'evidence'])
def test_audit_rejects_missing_or_substituted_floor_evidence(fault):
    pl = planner(); plan = logged(low_plan(pl)); selection = selected(pl)
    if fault == 'missing_snapshot': plan['query_results'].pop('capacity_floor')
    elif fault == 'short_snapshot': plan['query_results']['capacity_floor']['forecast'].pop('backlog')
    elif fault == 'fake_match': plan['query_results']['capacity_floor']['effective_floor'] = 0
    elif fault == 'profile': selection['capacity_floor']['model']['profile_key'] = {'model_id': 'other'}
    elif fault == 'stricter_slo': selection['capacity_floor']['slo']['tpot_s'] = .05
    else: selection['capacity_floor']['floors'][0]['evidence'] = 'different-acceptance'
    with pytest.raises(ValueError):
        _plan_capacity_floor(plan, selection, 8)


def test_no_artifact_keeps_the_original_canonical_floor():
    plan = logged(low_plan(planner()))
    assert _plan_capacity_floor(plan, {'profile_key': '{}'}, 8) == 4


def test_comparison_lower_m_cannot_inherit_unmeasured_low_clock_qualification():
    pl = planner()
    assert pl.evaluate({'M': 1, 'L1': 7}, 2520, 2520, 900, 0, fc(.05), strict=False) is None
    old = replace(low_plan(pl), f_M=900)
    with pytest.raises(ValueError, match='frequency'):
        _plan_capacity_floor(logged(old), selected(pl), 8)
    restored = pl.enforce_capacity_floor(old, fc(.05))
    assert restored.counts['M'] == 4 and restored.f_M == 2520
    assert restored.detail['capacity_floor_restoration']['reason'] == 'unqualified_low_m_frequency'


def test_complete_controller_actions_accept_lower_m_only_with_qualified_replay(tmp_path, monkeypatch):
    from pdblend.bench import comparison_pdblend_acceptance as module
    pl = planner()
    args = fixture(tmp_path, monkeypatch, parked=True, mixed_instances=1, profile_key='{}')
    selection = module._inputs(None, None)
    selection['capacity_floor'] = selected(pl)['capacity_floor']
    selection['choice']['plan']['f_M'] = 2520
    events = raw(args, 'controller')
    plan = next(row for row in events if row['kind'] == 'plan')
    plan['f_M'] = 2520
    plan['query_results'] = low_plan(pl).detail
    instances = {row['instance_id']: row for row in args['engine_identity']['instances']}
    actual = _controller(events, args['native_result'], instances, args['engine_identity'],
        args['reset'], selection, raw(args, 'transition_measurements'))
    assert actual[0][0]['counts']['M'] == 1
    selection.pop('capacity_floor')
    with pytest.raises(ValueError, match='canonical PD mixed reserve'):
        _controller(events, args['native_result'], instances, args['engine_identity'],
            args['reset'], selection, raw(args, 'transition_measurements'))


def test_receipt_counts_actual_low_m_use_and_canonical_fallbacks():
    pl = planner(); low = low_plan(pl); restored = pl.enforce_capacity_floor(low, fc(2.))
    all_m = pl.enforce_capacity_floor(replace(low, counts={'M': 8}), fc(.05))
    ctl = SimpleNamespace(_log=[dict(kind='plan', **logged(p)) for p in (low, restored, all_m)], shield=None)
    value = runtime_receipt(requested={}, enabled={}, controller=ctl, router=SimpleNamespace(), profile_key={})
    assert value['capacity_floor'] == dict(recorded_plan_decisions=3, matched_plans=2, relaxed_plans=1,
        canonical_fallback_plans=1, restoration_plans=1, rejection_reasons={'rate_outside_accepted_domain': 1})


@pytest.mark.parametrize('frequency', [2520, 900, None])
def test_direct_artifact_and_original_acceptance_are_independently_bound(tmp_path, frequency):
    model = replace(synthetic_model(), model='Qwen2.5-7B-Instruct')
    def floor_runs(arm, name, dataset, repeat, conditions, summary, execution, uncertainty):
        plan = {'counts': {'M': 1, 'L1': 7}}
        if frequency is not None:
            plan['f_M'] = frequency
        summary.update(fixed_plan=plan, profile_key=model.profile_key,
            trace=dict(input_min=1, input_max=512, output_min=20, output_max=20))
    manifest = acceptance_campaign(tmp_path, mutate_run=floor_runs)
    artifact = tmp_path/'floor.json'
    manifest_ref = binding(manifest)
    artifact.write_text(json.dumps(dict(kind='pdblend_capacity_floor_v1', identity=identity(model),
        stage='stage2-candidate-ranking', acceptance_manifest=manifest_ref,
        floors=[dict(min_m_instances=1, rate_range=[10, 10], input_range=[1, 512], output_range=[20, 20],
                     slo=dict(ttft_s=1, tpot_s=.1))])))
    options = comparison_options({'pdblend_runtime': {'capacity_floor_path': binding(artifact)}}, tmp_path/'config.json')
    if frequency != 2520:
        with pytest.raises(ValueError, match='explicit maximum-frequency controlled acceptance'):
            capacity_floor_selection(options, model, dict(ttft_s=1, tpot_s=.1), 8)
        return
    value = capacity_floor_selection(options, model, dict(ttft_s=1, tpot_s=.1), 8)
    assert value['artifact'] == binding(artifact) and value['acceptance_manifest'] == manifest_ref
    assert value['reserve_canonical']
    original = manifest.read_text(); manifest.write_text(original+' ')
    with pytest.raises(ValueError, match='checksum mismatch'):
        capacity_floor_selection(options, model, dict(ttft_s=1, tpot_s=.1), 8)
    manifest.write_text(original)
    artifact.write_text(artifact.read_text()+' ')
    with pytest.raises(ValueError, match='artifact changed'):
        capacity_floor_selection(options, model, dict(ttft_s=1, tpot_s=.1), 8)


def test_comparison_refuses_mutable_topology_index_but_generic_selection_remains(tmp_path):
    from pdblend.planner.capacity import select_artifact
    index = tmp_path/'index.json'
    index.write_text(json.dumps(dict(kind='pdblend_optimization_artifact_set_v1', profiles={'tp1-pp1':'floor.json'})))
    assert select_artifact(index, synthetic_model()) == tmp_path/'floor.json'
    with pytest.raises(ValueError, match='direct floor file'):
        comparison_options({'pdblend_runtime': {'capacity_floor_path': binding(index)}}, tmp_path/'config.json')


@pytest.mark.parametrize('switch', ['preserve_overload_capacity', 'safety_recovery', 'capacity_floor_reserve_canonical'])
def test_floor_comparison_cannot_disable_safe_restore(tmp_path, switch):
    artifact = tmp_path/'floor.json'; artifact.write_text('{}')
    with pytest.raises(ValueError, match='canonical M reserve'):
        comparison_options({'pdblend_runtime': {'capacity_floor_path': binding(artifact), switch:False}}, tmp_path/'config.json')


def test_initial_bound_plan_must_reserve_restore_space(tmp_path, monkeypatch):
    artifact = tmp_path/'floor.json'; artifact.write_text('{}')
    options = comparison_options({'pdblend_runtime': {'capacity_floor_path': binding(artifact)}}, tmp_path/'config.json')
    monkeypatch.setattr('pdblend.bench.pdblend_runtime_options.comparison_capacity_floors',
                        lambda *args, **kw: (planner().cfg.capacity_floors, 2520))
    with pytest.raises(ValueError, match='restoration slots'):
        preflight_comparison_options(options, synthetic_model(),
            Plan({'P':5, 'D':1, 'M':1, 'L1':1},2520,2520,2520,1024,1,1,.1), [None]*8)


@pytest.mark.parametrize('forecast,slo', [(fc(2.), dict(ttft_s=1.,tpot_s=.1)),
                                       (fc(.05), dict(ttft_s=1.,tpot_s=.05))])
def test_initial_low_m_rejects_outside_domain_or_stricter_slo_before_launch(tmp_path, monkeypatch, forecast, slo):
    artifact = tmp_path/'floor.json'; artifact.write_text('{}')
    options = comparison_options({'pdblend_runtime': {'capacity_floor_path': binding(artifact)}}, tmp_path/'config.json')
    pl = planner()
    monkeypatch.setattr('pdblend.bench.pdblend_runtime_options.comparison_capacity_floors',
                        lambda *args, **kw: (pl.cfg.capacity_floors, 2520))
    with pytest.raises(ValueError, match='outside qualified workload/SLO domain'):
        preflight_comparison_options(options, pl.model, low_plan(pl), [None]*8, forecast=forecast, slo=slo)
    with pytest.raises(ValueError, match='planning forecast'):
        preflight_comparison_options(options, pl.model, low_plan(pl), [None]*8)
    preflight_comparison_options(options, pl.model, low_plan(pl), [None]*8,
                                 forecast=fc(.05), slo=dict(ttft_s=1.,tpot_s=.1))
    with pytest.raises(ValueError, match='frequency outside its controlled floor acceptance'):
        preflight_comparison_options(options, pl.model, replace(low_plan(pl), f_M=900), [None]*8,
                                     forecast=fc(.05), slo=dict(ttft_s=1.,tpot_s=.1))


def test_controller_rechecks_old_plan_and_restores_during_hold_before_period(monkeypatch):
    from pdblend.online import controller as module
    ctl, _, _ = make_controller(n=8, shield=False)
    ctl.planner = planner(); ctl.safety_recovery = True
    ctl.initial_plan = Plan({'P':2, 'D':1, 'M':1, 'L1':4},2520,1500,2520,1024,1,1,.1)
    ctl.period_s, ctl.tick_s, ctl.min_plan_hold_s = 60., 1., 600.
    ctl._informed = lambda *args: True
    # A normal candidate could reduce P. Domain exit must restore the current
    # layout first, even if the ordinary planner would otherwise be runnable.
    def forbidden_replan(*args): raise AssertionError('domain restoration invoked ordinary planner')
    ctl.planner.plan = forbidden_replan
    ctl.hold_initial = True
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda:clock.now))
    ctl.forecaster = SimpleNamespace(set_backlog=lambda work:None,
        forecast=lambda now:fc(.05 if now == 100 else 2.))
    executed = []
    async def execute(plan):
        executed.append(plan); ctl.plan_now = plan; ctl._last_plan_change_s = clock.now
    ctl.execute = execute
    async def run():
        stop = asyncio.Event()
        async def tick(delay):
            clock.now += delay
            if clock.now >= 101: stop.set()
        async def inline(function, *args): return function(*args)
        monkeypatch.setattr(module, 'asyncio', SimpleNamespace(sleep=tick, to_thread=inline))
        await ctl.run(stop)
    asyncio.run(run())
    assert [p.counts['M'] for p in executed] == [1, 4]
    assert (executed[1].counts['P'], executed[1].counts['D'], executed[1].tau) == (2, 1, 1024)
    assert executed[1].f_M == 2520
    assert next(r for r in ctl._log if r['kind'] == 'forecast')['decision_reason'] == 'capacity_floor_restore'


def test_shield_cannot_consume_reserved_m_space_or_reduce_existing_pd(monkeypatch):
    from pdblend.online import controller as module
    from pdblend.online.shield import Pressure, Shield
    ctl, _, _ = make_controller(n=8, shield=False)
    ctl.planner = planner(); ctl.safety_recovery = True
    initial = Plan({'P':1, 'D':1, 'M':1, 'L1':5},2520,2520,2520,1024,1,1,.1)
    ctl.initial_plan = initial
    ctl.period_s, ctl.tick_s, ctl.min_plan_hold_s = 60., 1., 600.
    ctl._informed = lambda *args:False
    ctl.hold_initial = True
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda:clock.now))
    ctl.forecaster = SimpleNamespace(set_backlog=lambda work:None, forecast=lambda now:fc(.05))
    shield = Shield(SLO(1., .1))
    shield.observe = lambda *args:Pressure(prefill=True)
    def escalate(*args): shield.level = 8; return 8
    shield.update = escalate
    ctl.shield = shield
    executed = []
    async def execute(plan):
        executed.append(plan); ctl.plan_now = plan; ctl._last_plan_change_s = clock.now
    ctl.execute = execute
    async def run():
        stop = asyncio.Event()
        async def tick(delay): clock.now += delay; stop.set()
        async def inline(function, *args): return function(*args)
        monkeypatch.setattr(module, 'asyncio', SimpleNamespace(sleep=tick, to_thread=inline))
        await ctl.run(stop)
    asyncio.run(run())
    assert executed[1].counts == {'P':3, 'D':1, 'M':4}
    assert executed[1].tau == initial.tau and executed[1].f_M == 2520
    assert _plan_capacity_floor(logged(executed[1]), selected(ctl.planner), 8) == 1
    assert executed[1].detail['capacity_floor_restoration']['reason'] == 'shield_capacity_reserve'
