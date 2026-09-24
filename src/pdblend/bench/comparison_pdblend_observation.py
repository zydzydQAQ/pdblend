"""Explicit original-policy evaluation with unqualified, unchanged own profiles.

This scope never bypasses the formal loader. Raw measurement acceptance and
profile qualification are separate; it does not establish optimality.
"""
from __future__ import annotations
import json
from types import SimpleNamespace
from .comparison_pdblend_acceptance import (
    RAW_REFS, _read_raw, _boundary, _inventory_reset, _controller, _frequencies, _audit_physical_clocks,
    _routes, _routing_roles, _drain)
from .comparison_acceptance import _bound, _need, _equal
from .comparison_native_acceptance import (native_topology, audit_native_startup,
    audit_native_reset, audit_native_metrics, audit_native_meter)
from .resident_session import digest

SCOPE = 'pdblend_profile_unqualified_evaluation/v1'
SCHEMA = 'pdblend-observation-acceptance/v1'
PROFILE_GAPS = ('legacy_profile_not_formally_qualified',
                'native_profile_holdout_and_query_coverage_not_established')
REQUIRED_MEASUREMENT_GATES = frozenset(
    {'raw.'+name for name in RAW_REFS}
    | {'binding.'+name for name in ('trace','native_result','startup_qualification','reset','metering','drain')}
    | {'pdblend.observation_inputs','pdblend.inventory','pdblend.full_physical_inventory',
       'pdblend.actual_window','pdblend.startup','pdblend.reset','pdblend.inventory_restoration',
       'pdblend.controller_actions','pdblend.physical_clocks','pdblend.request_routes',
       'pdblend.published_route_roles','pdblend.native_release_and_off','pdblend.canonical_metrics',
       'metering.raw_eight_gpu_window'})


def observation_requested(point):
    return point.get('observation_scope') == SCOPE


def validate_observation_inputs(point, inputs):
    from .independent_dispatch import validate
    _need(point.get('system') == 'pdblend' and observation_requested(point)
          and point.get('qualification_mode') == SCOPE and point.get('duration_s') == 150,
          'explicit PD original-policy observation scope is missing')
    checked = validate(point, inputs)
    config = checked['config']
    _need(config.get('observation_scope') == SCOPE and config.get('profile_usage') == 'development',
          'observation configuration must explicitly select development profile usage')
    _need(inputs.get('trace') == point.get('trace'), 'evaluation trace binding differs')
    choice = _bound(inputs['offline_choice']); prior = _bound(inputs['planning_trace'])
    _need(choice.get('planning_trace_sha256') == inputs['planning_trace']['sha256'],
          'offline choice is not bound to its independent planning trace')
    _need(inputs['planning_trace']['sha256'] != inputs['trace']['sha256']
          and prior.get('dataset', point['dataset']) == point['dataset'],
          'independent planning trace split/dataset differs')
    selected = config.get('profile')
    _need(isinstance(selected, dict), 'observation requires an explicit original profile binding')
    _bound(selected)
    _need(selected in inputs.get('profiles', []) and choice.get('profile_sha256') == selected['sha256'],
          'offline choice selected profile binding differs')
    checked.update(formal_eligible=False, profile_qualified=False,
                   observation_scope=SCOPE, profile_missing_gates=list(PROFILE_GAPS))
    return checked


def observation_selection(point, instances):
    from .comparison_runtime import pdblend_window_resources
    from .pdblend_runtime_options import capacity_floor_selection
    checked = validate_observation_inputs(point, point['inputs'])
    specs = [SimpleNamespace(tp=r['tp'], pp=r['pp'], generation=0) for r in instances.values()]
    loaded, _ = pdblend_window_resources(point, specs)
    return dict(choice=_bound(point['inputs']['offline_choice']),
        experiment_mode=checked['pdblend_runtime']['values']['experiment_mode'],
        profile_key=json.dumps(loaded.profile_key, sort_keys=True, separators=(',', ':')),
        frequencies=list(loaded.model.freqs), calibration=loaded.manifest_fields(),
        qualification_bindings=checked['qualification_bindings'],
        capacity_floor=capacity_floor_selection(checked['pdblend_runtime'], loaded.model, point['slo'], len(instances)))


def valid_observation_result(point, result):
    """Continuation contract; caller also verifies final drain and file hashes."""
    audit = result.get('observation_acceptance', {})
    return (observation_requested(point) and point.get('system') == 'pdblend'
        and point.get('qualification_mode') == SCOPE
        and result.get('observation_scope') == SCOPE
        and result.get('measurement_evidence_valid') is True
        and result.get('evidence_valid') is False and result.get('formal_eligible') is False
        and result.get('profile_qualified') is False
        and audit.get('schema') == SCHEMA and audit.get('scope') == SCOPE
        and audit.get('point_sha256') == digest(point)
        and audit.get('metrics_sha256') == digest(result.get('metrics', {}))
        and audit.get('measurement_evidence_valid') is True
        and audit.get('evidence_valid') is False and audit.get('formal_eligible') is False
        and audit.get('profile_qualified') is False
        and audit.get('missing_gates') == [] and audit.get('gate_failures') == {}
        and REQUIRED_MEASUREMENT_GATES <= set(audit.get('checked_gates', []))
        and isinstance(audit.get('profile_missing_gates'), list) and bool(audit['profile_missing_gates'])
        and isinstance(audit.get('raw_refs'), dict) and set(RAW_REFS) <= set(audit['raw_refs'])
        and audit.get('evidence_sha256') == digest(audit['raw_refs']))


def audit_observation_window(point, engine_identity, startup_qualification, reset, native_result,
                         canonical_metrics, metering, drain, raw_refs):
    failures, checked, data, blocked = {}, [], {}, {}
    def gate(name, fn):
        try: value = fn()
        except (ValueError, TypeError, KeyError, OSError, IndexError, AttributeError, OverflowError, RuntimeError) as exc:
            failures[name] = str(exc); return None
        checked.append(name); return value
    for name in RAW_REFS + (('frequency_readings',) if 'frequency_readings' in raw_refs else ()):
        value = gate('raw.'+name, lambda name=name:_read_raw(raw_refs.get(name), name))
        if value is not None: data[name] = value
    for name, value in (('native_result',native_result), ('startup_qualification',startup_qualification),
                         ('reset',reset), ('metering',metering), ('drain',drain)):
        gate('binding.'+name, lambda name=name,value=value:_need(_equal(data[name],value), 'supplied receipt differs from bound raw file'))
    gate('binding.trace', lambda:_need(raw_refs.get('trace') == point.get('trace'), 'point trace binding differs'))
    instances = gate('pdblend.inventory', lambda:native_topology(point, engine_identity))
    if instances is not None:
        gate('pdblend.full_physical_inventory', lambda:_need([u for r in instances.values() for u in r['gpu_uuids']]
              == engine_identity['fleet_gpu_uuids'], 'PD planning must account for every metered GPU'))
    selected = gate('pdblend.observation_inputs', lambda:observation_selection(point, instances))
    origin = gate('pdblend.actual_window', lambda:_boundary(native_result, data['outcomes'], metering))
    gate('pdblend.startup', lambda:audit_native_startup(point, engine_identity, startup_qualification, instances, raw_refs))
    gate('pdblend.reset', lambda:audit_native_reset(reset, instances, origin))
    gate('pdblend.inventory_restoration', lambda:_inventory_reset(reset, instances, engine_identity, startup_qualification, origin))
    control = gate('pdblend.controller_actions', lambda:_controller(data['controller'], native_result, instances,
              engine_identity, reset, selected, data['transition_measurements']))
    _audit_physical_clocks(gate, blocked, data, control, instances, engine_identity, origin)
    gate('pdblend.request_routes', lambda:_routes(data['routes'], data['outcomes'], data['trace'], instances, reset, native_result))
    gate('pdblend.published_route_roles', lambda:_routing_roles(data['routes'], data['controller'], instances))
    gate('pdblend.native_release_and_off', lambda:_drain(native_result, drain, data['native_cleanup'], instances,
         engine_identity, reset, metering, [r for r in data['controller'] if r.get('kind') == 'transition_phase'],
         _bound(startup_qualification['source_manifest'])['source_sha256']))
    reduced = gate('pdblend.canonical_metrics', lambda:audit_native_metrics(point, data['trace'], data['outcomes'], None,
                   origin, data['canonical_requests'], canonical_metrics))
    gate('metering.raw_eight_gpu_window', lambda:audit_native_meter(engine_identity, data['power'], metering, origin))
    valid = not failures and not blocked
    return dict(schema=SCHEMA, scope=SCOPE, point_sha256=digest(point), metrics_sha256=digest(canonical_metrics),
        evidence_valid=False, formal_eligible=False, measurement_evidence_valid=valid, profile_qualified=False, slo_pass=reduced['slo_pass'] if reduced else False,
        missing_gates=list(failures)+list(blocked), gate_failures=failures, blocked_gates=blocked, checked_gates=checked,
        capacity_floor_selection=(selected or {}).get('capacity_floor'),
        optimality_established=False, per_request_kv_transaction_audited=False,
        inherited_artifact_flags_unchanged=True, evidence_sha256=digest(raw_refs), raw_refs=raw_refs,
        profile_missing_gates=list(PROFILE_GAPS))
