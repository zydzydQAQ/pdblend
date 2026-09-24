"""Explicit, hash-bound options for PD comparison windows and ablations."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ARTIFACTS = ('incremental_energy_path', 'transition_catalog_path', 'capacity_floor_path')
DEFAULTS = dict(joint_resident=False, transition_qualified_only=True,
    shield_mode='budget_aware', slo_routing=True, preserve_overload_capacity=True,
    safety_recovery=True, experiment_mode='adaptive', shield_sustained_gap_s=1.0,
    shield_stalled_fraction=0.25, shield_stalled_min_requests=2,
    slo_routing_safety=0.85, slo_routing_handoff_floor_s=0.0,
    capacity_floor_reserve_canonical=False, safety_max_freq=None,
    startup_safety=False, deadline_safety=False)
BOOLS = ('joint_resident', 'transition_qualified_only', 'slo_routing',
         'preserve_overload_capacity', 'safety_recovery', 'capacity_floor_reserve_canonical',
         'startup_safety', 'deadline_safety')
CONTROL_OPTIONS = ('shield_mode', 'slo_routing', 'preserve_overload_capacity',
                   'safety_recovery', 'experiment_mode', 'shield_sustained_gap_s',
                   'shield_stalled_fraction', 'shield_stalled_min_requests',
                   'slo_routing_safety', 'slo_routing_handoff_floor_s', 'capacity_floor_reserve_canonical',
                   'safety_max_freq', 'startup_safety', 'deadline_safety')


def _validate_thresholds(values):
    ceiling = values['safety_max_freq']
    if ceiling is not None and (type(ceiling) is not int or ceiling <= 0):
        raise ValueError('safety_max_freq must be a positive integer or null')
    for name in ('shield_sustained_gap_s', 'shield_stalled_fraction',
                 'slo_routing_safety', 'slo_routing_handoff_floor_s'):
        value = values[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(name + ' must be a finite number')
    if values['shield_sustained_gap_s'] <= 0:
        raise ValueError('shield_sustained_gap_s must be positive')
    if any(not 0 < values[k] <= 1 for k in ('shield_stalled_fraction', 'slo_routing_safety')):
        raise ValueError('shield_stalled_fraction and slo_routing_safety must lie in (0,1]')
    if values['slo_routing_handoff_floor_s'] < 0:
        raise ValueError('slo_routing_handoff_floor_s must be nonnegative')
    if type(values['shield_stalled_min_requests']) is not int or values['shield_stalled_min_requests'] < 2:
        raise ValueError('shield_stalled_min_requests must be an integer >=2')


def control_options(policy, requested=None):
    requested = {} if requested is None else requested
    if not isinstance(requested, dict) or set(requested) - set(CONTROL_OPTIONS):
        raise ValueError('unknown PDblend control options')
    if not policy.name.startswith('pdblend'):
        if requested:
            raise ValueError('PDblend control options cannot change an independent baseline')
        return dict({k: DEFAULTS[k] for k in CONTROL_OPTIONS}, shield_mode='legacy', slo_routing=False, preserve_overload_capacity=False,
                    safety_recovery=False, experiment_mode='adaptive')
    values = {k: requested.get(k, DEFAULTS[k]) for k in CONTROL_OPTIONS}
    if values['shield_mode'] not in ('legacy', 'budget_aware'):
        raise ValueError('shield_mode must be legacy or budget_aware')
    if values['experiment_mode'] not in ('adaptive', 'freeze_initial_all_m'):
        raise ValueError('unknown PDblend experiment_mode')
    if any(type(values[k]) is not bool for k in ('slo_routing', 'preserve_overload_capacity',
                                               'safety_recovery', 'capacity_floor_reserve_canonical',
                                               'startup_safety', 'deadline_safety')):
        raise ValueError('PDblend control switches must be booleans')
    _validate_thresholds(values)
    return values


def scoped_control_options(policy, requested=None, *, resident_pools=False):
    values = control_options(policy, requested)
    if resident_pools:
        if requested is not None and requested.get('slo_routing') is True:
            raise ValueError('SLO routing is not integrated with the resident-pool selector')
        values['slo_routing'] = False
    return values


def comparison_options(config, config_path, *, point=None):
    """Resolve explicit artifact bindings; never choose a latest profile/file."""
    requested = config.get('pdblend_runtime', {})
    if not isinstance(requested, dict) or set(requested) - set(DEFAULTS) - set(ARTIFACTS) - {'capacity_workload_binding'}:
        raise ValueError('unknown PDblend runtime options')
    values = dict(DEFAULTS, **requested)
    for name in BOOLS:
        if type(values[name]) is not bool:
            raise ValueError(name + ' must be an explicit boolean')
    if values['shield_mode'] not in ('legacy', 'budget_aware'):
        raise ValueError('shield_mode must be legacy or budget_aware')
    if values['experiment_mode'] not in ('adaptive', 'freeze_initial_all_m'):
        raise ValueError('unknown PDblend experiment_mode')
    _validate_thresholds(values)
    bindings = {}
    for name in ARTIFACTS:
        ref = requested.get(name)
        if ref is None:
            values[name] = None
            continue
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
            raise ValueError(name + ' requires an explicit path/SHA256 binding')
        path = (Path(config_path).parent / ref['path']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError(name + ' checksum mismatch')
        bindings[name] = dict(path=str(path), sha256=ref['sha256'])
        values[name] = path
    if values['transition_qualified_only'] is False and values['transition_catalog_path'] is not None:
        raise ValueError('comparison transition catalogs require qualified_only=true')
    # This comparison adapter owns one homogeneous fleet and one offline Plan.
    # General run_point supports resident pools; silently enabling them here
    # would leave their profile, placement and lifecycle evidence unbound.
    if values['joint_resident'] or values['incremental_energy_path'] is not None:
        raise ValueError('homogeneous comparison does not support joint_resident or incremental energy routing; '
                         'explicit qualified resident-pool comparison support is required')
    if values['capacity_floor_path'] is not None:
        artifact = json.loads(values['capacity_floor_path'].read_text())
        if artifact.get('kind') == 'pdblend_optimization_artifact_set_v1':
            raise ValueError('comparison capacity_floor_path must bind the direct floor file, not a topology index')
        if (not values['preserve_overload_capacity'] or not values['safety_recovery']
                or requested.get('capacity_floor_reserve_canonical') is False):
            raise ValueError('comparison capacity floors require overload recovery, safety recovery and canonical M reserve')
        values['capacity_floor_reserve_canonical'] = True
    context = {}
    workload = requested.get('capacity_workload_binding')
    if workload is not None:
        if values['capacity_floor_path'] is None or point is None:
            raise ValueError('capacity workload binding requires a floor and actual point identity')
        if (workload.get('model_id') != point.get('model_id')
                or workload.get('dataset') != point.get('dataset')
                or workload.get('nominal_rate_rps') != point.get('rate_rps')):
            raise ValueError('capacity corpus family differs from actual comparison point')
        context = capacity_workload_context(values['capacity_floor_path'], workload, values)
    return dict(values=values, requested=requested, artifact_bindings=bindings,
                capacity_floor_context=context, capacity_workload_binding=workload)


def capacity_workload_context(path, workload, runtime_options):
    """Verify independently selected corpus bytes before resolving v2 context."""
    from .comparison_acceptance import _bound
    from .resident_session import digest
    from .capacity_floor_v2 import runtime_context
    if not isinstance(workload, dict) or set(workload) != {'model_id','dataset','corpus_manifest','corpus_dataset','nominal_rate_rps'}:
        raise ValueError('capacity workload requires explicit model/dataset/corpus bindings')
    manifest, dataset = _bound(workload['corpus_manifest']), _bound(workload['corpus_dataset'])
    if (manifest.get('complete') is not True or manifest.get('model_name') != workload['model_id']
            or dataset.get('model_name') != workload['model_id'] or dataset.get('dataset') != workload['dataset']
            or manifest.get('dataset_sha256',{}).get(workload['dataset']) != workload['corpus_dataset']['sha256']):
        raise ValueError('capacity workload corpus identity/checksum differs')
    family = {key:value for key,value in workload.items() if key!='nominal_rate_rps'}
    return runtime_context(path, runtime_options=runtime_options, workload_family_sha256=digest(family),
                           nominal_rate_rps=workload['nominal_rate_rps'])


def preflight_comparison_options(options, model, plan, specs, *, forecast=None, slo=None):
    """Replay existing qualification guards before a benchmark launches engines."""
    values = options['values']
    from pdblend.planner.capacity import select_artifact
    from pdblend.planner.transitions import TransitionCatalog
    if values['capacity_floor_path'] is not None:
        floors, frequency = comparison_capacity_floors(values['capacity_floor_path'], model=model)
        canonical = min(4, len(specs))
        if any(floor.version == 2 for floor in floors):
            ceiling = values.get('safety_max_freq') or max(model.freqs)
            if (plan.counts != {'M':canonical,'L1':len(specs)-canonical} or plan.tau != 0
                    or any(getattr(plan,'f_'+role) != ceiling for role in ('P','D','M'))):
                raise ValueError('v2 offline choice must bind canonical M4 startup before adaptive low-M selection')
        if (plan.counts.get('P', 0) + plan.counts.get('D', 0) > len(specs) - canonical):
            raise ValueError('bound capacity-floor plan lacks canonical M restoration slots')
        if plan.counts.get('M', 0) < canonical:
            if frequency is not None and plan.f_M != frequency:
                raise ValueError('bound lower-M plan uses a frequency outside its controlled floor acceptance')
            if forecast is None or slo is None:
                raise ValueError('bound lower-M plan requires its independent planning forecast and SLO')
            from pdblend.planner.pool import SLO
            minimum = min((q.min_m_instances for q in floors if q.matches(model, forecast, SLO(**slo),
                context=options.get('capacity_floor_context', {}), n_m=plan.counts.get('M',0),
                frequency_mhz=plan.f_M)),
                          default=canonical)
            if plan.counts.get('M', 0) < minimum:
                raise ValueError('bound capacity-floor plan lies outside qualified workload/SLO domain')
    if values['transition_catalog_path'] is not None:
        TransitionCatalog.load(select_artifact(values['transition_catalog_path'], model),
                               model=model, qualified_only=True)
    if values['experiment_mode'] == 'freeze_initial_all_m':
        if (plan.counts.get('M') != len(specs) or any(v for k, v in plan.counts.items() if k != 'M')
                or plan.tau != 0 or plan.f_M != max(model.freqs)):
            raise ValueError('freeze_initial_all_m requires a bound all-M, maximum-frequency, tau=0 offline choice')


def comparison_capacity_floors(path, *, model):
    """Comparison relaxes M only at an explicitly measured maximum clock.

    Generic floor artifacts retain their historical workload-only meaning.
    Comparison must not extend an M=1 acceptance at one clock to lower clocks.
    """
    from pdblend.planner.capacity import load_capacity_floors
    path = Path(path)
    artifact = json.loads(path.read_text())
    floors = load_capacity_floors(path, model=model)
    if artifact.get('kind') not in ('pdblend_capacity_floor_v1','pdblend_capacity_floor_v2'):
        raise ValueError('comparison requires a direct capacity-floor artifact')
    canonical = min(4, 8 // (model.tp * model.pp))
    if any(floor.min_m_instances > canonical for floor in floors):
        raise ValueError('comparison capacity floor may only relax the canonical M reserve')
    if artifact['kind'] == 'pdblend_capacity_floor_v2':
        if any(floor.frequency_mhz not in model.freqs for floor in floors):
            raise ValueError('v2 capacity frequency lacks exact profile coverage')
        return floors, None  # Each v2 floor binds its own measured frequency.
    ref = artifact['acceptance_manifest']
    manifest_path = (path.parent / ref['path']).resolve()
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('capacity-floor original acceptance manifest checksum mismatch')
    manifest, frequency = json.loads(raw), max(model.freqs)
    for floor in floors:
        pairs = [p['candidate'] for p in manifest['pairs']
                 if p['candidate']['conditions']['model_id'] == floor.model_id
                 and (p['candidate']['conditions']['tp'], p['candidate']['conditions']['pp']) == (floor.tp, floor.pp)
                 and p['candidate']['conditions']['slo'] == dict(ttft_s=floor.accepted_slo[0], tpot_s=floor.accepted_slo[1])
                 and floor.rate_range[0] <= p['candidate']['conditions']['rate_rps'] <= floor.rate_range[1]]
        if not pairs:
            raise ValueError('capacity-floor comparison lacks frequency-bound trials')
        for pair in pairs:
            ref = pair['summary']
            payload = (manifest_path.parent / ref['path']).read_bytes()
            if hashlib.sha256(payload).hexdigest() != ref['sha256']:
                raise ValueError('capacity-floor frequency evidence checksum mismatch')
            if json.loads(payload).get('fixed_plan', {}).get('f_M') != frequency:
                raise ValueError('capacity-floor comparison requires explicit maximum-frequency controlled acceptance')
    return floors, frequency


def capacity_floor_selection(options, model, slo, slots):
    """Bind a direct qualified floor and its original acceptance for audit replay."""
    from dataclasses import asdict
    path = options['values'].get('capacity_floor_path')
    if path is None:
        return None
    ref = options['artifact_bindings']['capacity_floor_path']
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('capacity-floor artifact changed after configuration binding')
    artifact = json.loads(raw)
    if artifact.get('kind') not in ('pdblend_capacity_floor_v1','pdblend_capacity_floor_v2'):
        raise ValueError('comparison requires a direct capacity-floor artifact')
    floors, frequency = comparison_capacity_floors(path, model=model)
    if artifact['kind'] == 'pdblend_capacity_floor_v2':
        return dict(schema='pdblend-capacity-floor-selection/v2', artifact=ref,
            tuning_manifest=artifact['tuning_manifest'], floors=[asdict(floor) for floor in floors],
            model=dict(model=model.model,tp=model.tp,pp=model.pp,profile_key=model.profile_key),
            slo=dict(ttft_s=slo['ttft_s'],tpot_s=slo['tpot_s']),canonical_floor=min(4,slots),
            context=options.get('capacity_floor_context', {}),
            startup_plan=dict(counts={'M':min(4,slots),'L1':max(0,slots-4)},
                f_P=options['values'].get('safety_max_freq') or max(model.freqs),
                f_D=options['values'].get('safety_max_freq') or max(model.freqs),
                f_M=options['values'].get('safety_max_freq') or max(model.freqs),tau=0),
            recovery_frequency_mhz=options['values'].get('safety_max_freq') or max(model.freqs),
            reserve_canonical=options['values']['capacity_floor_reserve_canonical'])
    acceptance = artifact['acceptance_manifest']
    manifest_path = (Path(path).parent / acceptance['path']).resolve()
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != acceptance['sha256']:
        raise ValueError('capacity-floor original acceptance manifest changed')
    return dict(schema='pdblend-capacity-floor-selection/v1', artifact=ref,
        acceptance_manifest=dict(path=str(manifest_path), sha256=acceptance['sha256']),
        floors=[asdict(floor) for floor in floors],
        model=dict(model=model.model, tp=model.tp, pp=model.pp, profile_key=model.profile_key),
        slo=dict(ttft_s=slo['ttft_s'], tpot_s=slo['tpot_s']), canonical_floor=min(4, slots),
        qualified_m_frequency_mhz=frequency,
        reserve_canonical=options['values']['capacity_floor_reserve_canonical'])


def runtime_receipt(*, requested, enabled, controller, router, profile_key, artifact_bindings=None):
    """Separate requested/constructed mechanisms from actual observed triggers."""
    controllers = list(getattr(controller, 'controllers', {}).values()) or [controller]
    events = [row for ctl in controllers for row in getattr(ctl, '_log', ())]
    plans = [row for row in events if row.get('kind') == 'plan']
    forecasts = [row for row in events if row.get('kind') == 'forecast']
    fallback = [row for row in plans if row.get('fallback')]
    fallback_candidates = [row for row in forecasts if row.get('candidate_fallback') is True]
    candidate_reasons = {}
    for row in fallback_candidates:
        reason = row.get('candidate_fallback_reason') or 'planner_no_feasible_candidate'
        candidate_reasons[reason] = candidate_reasons.get(reason, 0) + 1
    reasons = {}
    for row in fallback:
        detail = row.get('query_results', {})
        reason = detail.get('fallback_reason', detail.get('reason', 'planner_no_feasible_candidate'))
        reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    routing = router.slo_routing_summary() if hasattr(router, 'slo_routing_summary') else {}
    routing['scope'] = ('unsupported_resident_pool_selector' if hasattr(router, 'pools')
                        else 'homogeneous_router')
    shields = [ctl.shield for ctl in controllers if getattr(ctl, 'shield', None) is not None]
    shield_config = [dict(mode=s.mode, threshold=s.threshold, sustained_gap_s=s.sustained_gap_s,
        stalled_fraction=s.stalled_fraction, stalled_min_requests=s.stalled_min_requests)
        for s in shields]
    floor_decisions = [row.get('query_results', {}).get('capacity_floor') for row in plans]
    floor_decisions = [row for row in floor_decisions if isinstance(row, dict)]
    floor_reasons = {}
    for row in floor_decisions:
        for reason in set(row.get('rejected_floors', {}).values()):
            floor_reasons[reason] = floor_reasons.get(reason, 0) + 1
    return dict(schema='pdblend-runtime-options/v1', requested=requested, enabled=enabled,
        artifact_bindings=artifact_bindings or {}, profile_key=profile_key,
        trigger_counts=dict(plans=len(plans), shield_overrides=sum(
            row.get('decision_reason') == 'shield_override' for row in forecasts),
            safety_recoveries=sum(row.get('decision_reason') == 'safety_recovery' for row in forecasts),
            fallback_plans=len(fallback), fallback_candidates=len(fallback_candidates),
            capacity_preserving_fallback_candidates=sum(row.get('candidate_fallback_reason') in
                {'capacity_preserving_full_clock', 'retain_uncertified_capacity'} for row in fallback_candidates)),
        fallback_reasons=reasons, fallback_candidate_reasons=candidate_reasons, slo_routing=routing,
        shield_config=shield_config,
        capacity_floor=dict(recorded_plan_decisions=len(floor_decisions),
            matched_plans=sum(bool(row.get('selected_floor_ids')) for row in floor_decisions),
            relaxed_plans=sum(bool(row.get('query_results', {}).get('capacity_floor', {}).get('selected_floor_ids'))
                and row.get('counts', {}).get('M', 0) < row['query_results']['capacity_floor']['canonical_floor']
                for row in plans),
            canonical_fallback_plans=sum(bool(row.get('fallback_reason')) for row in floor_decisions),
            restoration_plans=sum(bool(row.get('query_results', {}).get('capacity_floor_restoration')) for row in plans),
            rejection_reasons=floor_reasons),
        profile_qualification_promoted=False)
