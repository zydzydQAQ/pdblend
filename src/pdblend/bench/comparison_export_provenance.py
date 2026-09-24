"""Small provenance annotations for append-only comparison exports."""
from __future__ import annotations

import math

from .resident_session import digest


def energy_supplement_columns(point, inventory_gaps=None):
    """Keep raw metadata while classifying only a bound original gap attempt."""
    original = point.get('energy_supplement_of', {})
    reason = 'not_declared'
    matched = False
    if original:
        gap = inventory_gaps.get(point['name']) if inventory_gaps is not None else None
        if inventory_gaps is None:
            reason = 'unverified_missing_frozen_inventory'
        elif gap is None or point['system'] == 'pdblend':
            reason = 'not_a_frozen_inventory_gap'
        elif original != gap['receipt']:
            reason = 'original_receipt_reference_mismatch'
        else:
            frozen = gap['point']
            identity = ('name', 'model_id', 'dataset', 'system', 'scale', 'seed',
                        'duration_s', 'slo', 'trace')
            numeric = [point.get(k) for k in ('scale', 'seed', 'duration_s')]
            numeric += [point.get('slo', {}).get(k) for k in ('ttft_s', 'tpot_s')]
            matched = (all(type(v) in (int, float) and
                           (type(v) is int or math.isfinite(v)) for v in numeric)
                       and all(point.get(k) == frozen.get(k) for k in identity))
            reason = ('frozen_inventory_gap_supplement' if matched else
                      'original_workload_identity_mismatch')
    return dict(original_energy_supplement_of=original,
                energy_supplement_of=original if matched else {},
                is_energy_supplement=matched, energy_supplement_classification=reason)


def point_columns(point, *, inventory_gaps=None):
    inputs = point.get('inputs', {})
    profiles = inputs.get('profiles', []) or [inputs[k] for k in ('eco_profile_csv', 'eco_profile_manifest') if k in inputs]
    return dict(run_id=point.get('run_id', ''),
        experiment_scope=point.get('experiment_scope', point.get('observation_scope', '')),
        profile_bindings=profiles, profile_sha256=[r['sha256'] for r in profiles],
        profile_provenance=point.get('profile_provenance', {}),
        profile_composition=point.get('profile_composition', {}),
        optimization_version=point.get('optimization_version', {}),
        experiment_phase=point.get('experiment_phase', ''),
        requested_comparison_scope=point.get('comparison_scope', ''),
        **energy_supplement_columns(point, inventory_gaps),
        boundary_selection=point.get('boundary_selection', {}),
        pd_boundary_point=point.get('pd_boundary_point', {}),
        system_config_sha256=inputs.get('system_config', {}).get('sha256', ''),
        execution_source_manifest=point.get('source_manifest', inputs.get('source_manifest', {})),
        planning_trace_sha256=inputs.get('planning_trace', {}).get('sha256', ''),
        offline_choice_sha256=inputs.get('offline_choice', {}).get('sha256', ''),
        boundary_scope='', boundary_policy_sha256='', boundary_manifest_sha256='',
        boundary_status='', boundary_passed_lower=None, boundary_failed_upper=None,
        boundary_role='', boundary_stop_reason='', boundary_decision_refs=[])


def load_baseline_inventory(ref, *, load_bound):
    """Read the inventory's receipt/point bindings, without replaying large raw."""
    manifest = load_bound(ref)
    if manifest.get('schema') != 'pdblend-baseline-energy-gap-inventory/v1':
        raise ValueError('unsupported frozen baseline inventory schema')
    selected, gaps = {}, {}
    for kind, row in ([('frozen', r) for r in manifest['frozen_baselines']]
                      + [('gap', r) for r in manifest.get('gaps', [])]):
        point, receipt = load_bound(row['point']), load_bound(row['receipt'])
        if (point.get('name') != row.get('point_id') or point.get('system') == 'pdblend'
                or receipt.get('point') != row['point_id']
                or receipt.get('point_sha256') != row.get('point_sha256')
                or digest(point) != row.get('point_sha256')
                or receipt.get('artifacts', {}).get('point.json') != row['point']['sha256']):
            raise ValueError('frozen baseline inventory point/receipt binding differs')
        name, sha = row['point_id'], row['receipt']['sha256']
        if kind == 'frozen':
            if name in selected and selected[name] != sha:
                raise ValueError('frozen baseline inventory selects multiple receipts')
            selected[name] = sha
        else:
            if name in gaps or name in selected:
                raise ValueError('frozen baseline inventory has duplicate or overlapping gap')
            gaps[name] = dict(point=point, receipt=row['receipt'])
    return dict(frozen_baselines=selected, gaps=gaps)


def load_frozen_baseline_inventory(ref, *, load_bound):
    """Compatibility interface for callers requiring only fixed selections."""
    return load_baseline_inventory(ref, load_bound=load_bound)['frozen_baselines']


def annotate_boundary(row, manifest, manifest_ref):
    """Attach a verified boundary to the exact model/revision observation."""
    if row['model_id'] != manifest['model_id']:
        return
    if row['system'] == 'pdblend' and row['revision'] != manifest['revision']:
        return
    if row.get('run_id') and row['run_id'] != manifest['run_id']:
        return
    state = manifest.get('boundaries', {}).get(row['dataset'])
    if state is None:
        return
    lower, upper = state.get('passed_lower'), state.get('failed_upper')
    scale = row['rate_scale']
    role = ('L' if lower is not None and math.isclose(scale, lower, rel_tol=1e-12) else
            'U' if upper is not None and math.isclose(scale, upper, rel_tol=1e-12) else '')
    row.update(boundary_scope=manifest['schema'], boundary_policy_sha256=manifest['policy']['sha256'],
        boundary_manifest_sha256=manifest_ref['sha256'], boundary_status=state.get('status', ''),
        boundary_passed_lower=lower, boundary_failed_upper=upper, boundary_role=role,
        boundary_stop_reason=state.get('stop_reason', manifest.get('stop_reason', '')),
        boundary_decision_refs=manifest.get('decisions', []))
    # Historical baseline observations are linked to the run's boundary without
    # relabelling their original execution as part of the new run.
    row['boundary_run_id'] = manifest['run_id']
