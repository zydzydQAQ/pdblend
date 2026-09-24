import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.bench.comparison_campaign import binding
from pdblend.bench.comparison_runtime import pdblend_window_resources
from pdblend.bench.independent_dispatch import Resources, execute, validate
from pdblend.bench.pdblend_runtime_options import CONTROL_OPTIONS, DEFAULTS, comparison_options, control_options, scoped_control_options, runtime_receipt
from test_comparison_pdblend_observation import observation_point
from test_comparison_runtime_inputs import change_choice


def configure(point, options):
    path = Path(point['inputs']['system_config']['path'])
    value = json.loads(path.read_text())
    value['pdblend_runtime'] = options
    path.write_text(json.dumps(value))
    point['inputs']['system_config'] = binding(path)


def dispatch(point, specs, tmp_path):
    loaded, plan = pdblend_window_resources(point, specs)
    seen = {}
    async def runner(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return {'executed': True}
    resources = Resources(specs, pd_model=loaded.model, pd_plan=plan)
    result = asyncio.run(execute(point, point['inputs'], resources, tmp_path / 'run',
                                 runner_overrides={'pdblend': runner}))
    assert result['executed']
    return seen, plan


def test_candidate_controls_are_explicit_and_allow_component_isolation(tmp_path):
    point, specs = observation_point(tmp_path)
    options = dict(shield_mode='legacy', slo_routing=False, preserve_overload_capacity=False,
                   safety_recovery=False)
    configure(point, options)
    seen, plan = dispatch(point, specs, tmp_path)
    assert seen['kwargs']['initial_plan'] is plan
    assert seen['kwargs']['runtime_requested'] == options
    assert seen['kwargs']['pdblend_runtime'] == dict({k: DEFAULTS[k] for k in CONTROL_OPTIONS}, **options)
    assert not seen['kwargs']['joint_resident']
    assert all(seen['kwargs'][key] is None for key in
               ('incremental_energy_path', 'capacity_floor_path', 'transition_catalog_path'))


@pytest.mark.parametrize('options, message', [
    ({'slo_routing': 'false'}, 'boolean'),
    ({'joint_resident': True}, 'homogeneous comparison'),
    ({'capacity_floor_path': 'latest.json'}, 'SHA256 binding'),
    ({'unknown_feature': True}, 'unknown'),
    ({'shield_mode': 'none'}, 'shield_mode'),
    ({'shield_sustained_gap_s': 0}, 'positive'),
    ({'shield_stalled_fraction': 2}, 'in \\(0,1\\]'),
    ({'shield_stalled_min_requests': True}, 'integer'),
    ({'slo_routing_handoff_floor_s': -1}, 'nonnegative'),
    ({'slo_routing_safety': float('nan')}, 'finite'),
])
def test_invalid_or_unsupported_config_is_rejected_before_runner(tmp_path, options, message):
    point, specs = observation_point(tmp_path)
    configure(point, options)
    with pytest.raises(ValueError, match=message):
        validate(point, point['inputs'])


def test_bound_optimization_artifacts_are_forwarded_through_existing_guards(tmp_path, monkeypatch):
    point, specs = observation_point(tmp_path)
    refs = {}
    for key in ('capacity_floor_path', 'transition_catalog_path'):
        path = tmp_path / (key + '.json')
        path.write_text('{}')
        refs[key] = binding(path)
    configure(point, refs)
    calls = []
    def floors(path, *, model):
        calls.append(('capacity', path))
        return (), max(model.freqs)
    monkeypatch.setattr('pdblend.bench.pdblend_runtime_options.comparison_capacity_floors', floors)
    monkeypatch.setattr('pdblend.planner.transitions.TransitionCatalog.load',
        lambda path, *, model, qualified_only: calls.append(('transition', path, qualified_only)))
    seen, _ = dispatch(point, specs, tmp_path)
    for key, ref in refs.items():
        assert seen['kwargs'][key] == Path(ref['path'])
    assert seen['kwargs']['runtime_artifact_bindings'] == refs
    assert any(call[0] == 'capacity' for call in calls)
    assert any(call[0] == 'transition' and call[2] is True for call in calls)


def test_unqualified_or_changed_artifacts_cannot_reach_execution(tmp_path):
    point, specs = observation_point(tmp_path)
    path = tmp_path / 'capacity.json'
    path.write_text('{}')
    configure(point, {'capacity_floor_path': binding(path)})
    with pytest.raises(ValueError, match='identity mismatch'):
        pdblend_window_resources(point, specs)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        validate(point, point['inputs'])


def test_comparison_cannot_request_unqualified_transition_fallback(tmp_path):
    path = tmp_path / 'transition.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='qualified_only'):
        comparison_options({'pdblend_runtime': dict(transition_catalog_path=binding(path),
            transition_qualified_only=False)}, tmp_path / 'config.json')


def test_fixed_all_m_uses_bound_initial_plan_and_rejects_pd_choice(tmp_path):
    point, specs = observation_point(tmp_path)
    configure(point, {'experiment_mode': 'freeze_initial_all_m'})
    seen, plan = dispatch(point, specs, tmp_path)
    assert seen['kwargs']['fixed_plan'] is seen['kwargs']['initial_plan'] is plan
    change_choice(point, lambda choice: choice['plan'].update(counts={'P': 1, 'D': 1}, tau=1024))
    with pytest.raises(ValueError, match='bound all-M'):
        pdblend_window_resources(point, specs)


def test_native_selection_identity_is_allowed_but_loader_gates_remain_closed(tmp_path):
    point, specs = observation_point(tmp_path)
    path = Path(point['inputs']['profiles'][0]['path'])
    path.write_text(json.dumps(dict(kind='pdblend_native_profile_selection_v1', identity=dict(
        system='pdblend', model_id=point['model_id'], tp=1, pp=1))))
    point['inputs']['profiles'] = [binding(path)]
    cfg = Path(point['inputs']['system_config']['path'])
    value = json.loads(cfg.read_text())
    value['profile'] = binding(path)
    cfg.write_text(json.dumps(value))
    point['inputs']['system_config'] = binding(cfg)
    change_choice(point, lambda choice: choice.update(profile_sha256=binding(path)['sha256']))
    assert validate(point, point['inputs'])['own_profile_count'] == 1
    with pytest.raises(ValueError, match='native profile blocked'):
        pdblend_window_resources(point, specs)


def test_baseline_settings_stay_legacy_and_cannot_accept_pd_overrides():
    baseline = SimpleNamespace(name='mixed')
    settings = control_options(baseline)
    assert settings['shield_mode'] == 'legacy' and not settings['slo_routing']
    assert not settings['safety_recovery'] and not settings['preserve_overload_capacity']
    with pytest.raises(ValueError, match='independent baseline'):
        control_options(baseline, {'slo_routing': True})


@pytest.mark.parametrize('legacy', [False, True])
def test_controller_construction_applies_candidate_or_isolated_controls(tmp_path, legacy):
    from pdblend.bench.run import _make_controller
    from pdblend.bench.client import Request
    from pdblend.control.policies import get_policy
    from pdblend.online.router import Router
    from pdblend.planner.pool import SLO
    point, specs = observation_point(tmp_path)
    loaded, plan = pdblend_window_resources(point, specs)
    fleet = SimpleNamespace(instances={str(i): SimpleNamespace(spec=SimpleNamespace(
        max_num_seqs=32, max_model_len=8192)) for i in range(2)})
    router = Router(list(fleet.instances))
    options = dict(shield_mode='legacy', slo_routing=False, preserve_overload_capacity=False,
                   safety_recovery=False) if legacy else dict(shield_sustained_gap_s=.7,
                       shield_stalled_fraction=.4, shield_stalled_min_requests=3,
                       slo_routing_safety=.9, slo_routing_handoff_floor_s=.12)
    ctl = _make_controller(fleet, router, None, loaded.model, get_policy('pdblend'), SLO(1., .1),
        [Request(0, 0., [1, 2], 16), Request(1, 1., [1, 2], 16)], tmp_path, 10.,
        initial_plan=plan, pdblend_runtime=options)
    assert ctl.planner.cfg.preserve_overload_capacity is (not legacy)
    assert ctl.safety_recovery is (not legacy)
    assert ctl.shield.mode == ('legacy' if legacy else 'budget_aware')
    assert router.slo_routing_summary()['enabled'] is (not legacy)
    if not legacy:
        assert (ctl.shield.sustained_gap_s, ctl.shield.stalled_fraction, ctl.shield.stalled_min_requests) == (.7, .4, 3)
        assert router.slo_routing_summary()['config']['max_num_seqs'] == 32
        assert router.slo_routing_summary()['config']['max_model_len'] == 8192
        assert router.slo_routing_summary()['config']['safety'] == .9
        assert router.slo_routing_summary()['config']['handoff_floor_s'] == .12


def test_nondefault_thresholds_reach_dispatch_and_remain_explicitly_requested(tmp_path):
    point, specs = observation_point(tmp_path)
    options = dict(shield_sustained_gap_s=.7, shield_stalled_fraction=.4,
        shield_stalled_min_requests=3, slo_routing_safety=.9, slo_routing_handoff_floor_s=.12)
    configure(point, options)
    seen, _ = dispatch(point, specs, tmp_path)
    assert seen['kwargs']['runtime_requested'] == options
    assert all(seen['kwargs']['pdblend_runtime'][k] == v for k, v in options.items())


def test_runtime_receipt_distinguishes_enabled_and_actual_trigger():
    ctl = SimpleNamespace(_log=[{'kind': 'plan', 'fallback': True, 'query_results': {
        'fallback_reason': 'capacity_insufficient'}}, {'kind': 'forecast',
        'decision_reason': 'shield_override'}])
    router = SimpleNamespace(slo_routing_summary=lambda: {'decisions': {'spillover_m': 3}})
    result = runtime_receipt(requested={'slo_routing': True}, enabled={'slo_routing': True},
        controller=ctl, router=router, profile_key={'native_selection_sha256': 'bound'})
    assert result['trigger_counts'] == {'plans': 1, 'shield_overrides': 1,
                                       'safety_recoveries': 0, 'fallback_plans': 1,
                                       'fallback_candidates': 0, 'capacity_preserving_fallback_candidates': 0}
    assert result['fallback_reasons'] == {'capacity_insufficient': 1}
    assert result['slo_routing']['decisions']['spillover_m'] == 3
    assert not result['profile_qualification_promoted']


def test_resident_scope_cannot_claim_a_bypassed_slo_selector_is_enabled():
    policy = SimpleNamespace(name='pdblend')
    implicit = scoped_control_options(policy, resident_pools=True)
    assert not implicit['slo_routing'] and implicit['safety_recovery']
    assert not scoped_control_options(policy, {'slo_routing': False}, resident_pools=True)['slo_routing']
    with pytest.raises(ValueError, match='not integrated'):
        scoped_control_options(policy, {'slo_routing': True}, resident_pools=True)
    router = SimpleNamespace(pools={}, slo_routing_summary=lambda: {'enabled': False})
    receipt = runtime_receipt(requested={}, enabled=implicit, controller=SimpleNamespace(),
                             router=router, profile_key={})
    assert not receipt['enabled']['slo_routing']
    assert receipt['slo_routing']['scope'] == 'unsupported_resident_pool_selector'
