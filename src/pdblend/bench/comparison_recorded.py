"""Explicit analysis of recorded windows, independent of qualification verdicts.

The exporter verifies the original receipt/result/artifact bindings before this
module is called. No request, energy, clock or audit evidence is recomputed here.
"""
from copy import deepcopy
import json
import math

from .comparison_observed import RANK_FIELDS

POLICY = 'all_recorded_windows/v1'
SCOPE = 'available_recorded_system_variants_single_observation'


def finite(value, *, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def count(value):
    return type(value) is int and value >= 0


def usable_metrics(metrics):
    """Recognize an actual service window, including windows with failed requests.

    Missing latency observations when no corresponding samples exist remain
    missing; they are neither fabricated nor a reason to discard all failures.
    """
    if not isinstance(metrics, dict) or metrics.get('duration_s') != 150.:
        return False, 'missing_recorded_150s_metrics'
    start, end = metrics.get('service_start_s'), metrics.get('service_end_s')
    if not (finite(start) and finite(end) and math.isclose(end-start, 150., rel_tol=0, abs_tol=1e-5)):
        return False, 'missing_recorded_150s_service_interval'
    if not finite(metrics.get('energy_service_j')):
        return False, 'missing_recorded_service_energy'
    if not count(metrics.get('offered_requests')) or metrics['offered_requests'] == 0:
        return False, 'missing_recorded_request_denominator'
    if any(not count(metrics.get(key)) for key in ('successful_requests', 'failed_requests')):
        return False, 'missing_recorded_request_outcomes'
    for name in ('ttft', 'tpot'):
        value, samples = metrics.get(name+'_p99_s'), metrics.get(name+'_samples')
        if not finite(value) and not (value is None and samples == 0):
            return False, 'missing_recorded_'+name+'_statistics'
    return True, ''


def recorded_request_metrics(metrics):
    """Recognize recorded request data even when the energy sampler had gaps."""
    if not isinstance(metrics, dict):
        return False, 'missing_recorded_150s_metrics'
    # This is a shape check, not an energy estimate. The original metric stays
    # untouched; only the existing request/timestamp validation is reused.
    checked = dict(metrics, energy_service_j=0.)
    return usable_metrics(checked)


def slo_pass(row):
    offered, successful, joint = (row.get(k) for k in (
        'offered_requests', 'successful_requests', 'joint_slo_requests'))
    return bool(count(offered) and offered > 0 and count(successful) and successful == offered
        and row.get('failed_requests') == 0 and row.get('unresolved_requests', 0) == 0
        and count(joint) and .9 <= joint/offered <= 1.
        and all(finite(row.get(name+'_p99_s')) and finite(row.get('slo_'+name+'_s'))
                and row[name+'_p99_s'] <= row['slo_'+name+'_s'] for name in ('ttft', 'tpot')))


def _identity(row):
    # Source revisions and target-clock policy are deliberately not identities
    # for this descriptive, system-as-executed comparison.
    names = ('model_id', 'dataset', 'rate_scale', 'seed', 'duration_s', 'trace_sha256',
             'measurement_protocol_version', 'model_hash', 'tokenizer_hash', 'gpu_uuids',
             'image_digest', 'slo_ttft_s', 'slo_tpot_s')
    if any(row.get(k) in (None, '', []) for k in names):
        return None
    numeric = {'rate_scale', 'seed', 'duration_s', 'slo_ttft_s', 'slo_tpot_s'}
    if any(type(row[k]) not in (int, float)
           or (type(row[k]) is float and not math.isfinite(row[k])) for k in numeric):
        return None
    if row.get('analysis_trace_identity_sha256'):
        # Separate proven event identities from raw file-hash identities. A
        # coincidentally identical unannotated hash must not join this group.
        row = dict(row, trace_sha256={'exact_boundary_events_v1': row['analysis_trace_identity_sha256']})
    # JSON distinguishes 150 from 150.0 even though both denote the same
    # duration. Python numeric equality/hash preserves exact mathematical
    # equality across int/float without rounding distinct floats or large
    # integer seeds. Reject booleans above rather than letting True equal 1.
    return tuple(('number', row[k]) if k in numeric else
                 json.dumps(sorted(row[k]) if k == 'gpu_uuids' else row[k], sort_keys=True)
                 for k in names)


def _total(row):
    service, tail = row.get('energy_service_j'), row.get('energy_tail_j')
    if not finite(service) or not finite(tail):
        return None
    total = service+tail
    reported = row.get('energy_service_tail_j')
    if reported is not None and (not finite(reported) or not math.isclose(total, reported)):
        return None
    return total


def _reference(row):
    return dict(system=row['system'], revision=row['revision'],
                path=row['receipt_path'], sha256=row['receipt_sha256'])


def _variant(row):
    return json.dumps([row['system'], row['revision']], separators=(',', ':'))


def _rank(row, feasible):
    return 1+sum(other['energy_service_j'] < row['energy_service_j'] for other in feasible)


def analyze(rows, records, *, frozen_baselines=None):
    """Annotate in place after strict export/ranking; ``records`` are bound results.

    Every PD revision receives its own comparison with the available baseline
    variants. Duplicate attempts of the same system/revision are excluded as
    ambiguous, regardless of whether selecting one would improve the result.
    """
    groups = {}
    for row in rows:
        for name in RANK_FIELDS:
            row['strict_'+name] = deepcopy(row.get(name))
        row.update(analysis_policy=POLICY, comparison_scope=SCOPE,
            analysis_statistical_scope='single_observation_no_significance_or_causal_claim',
            original_status=row['status'], original_failure_reason=row.get('failure_reason', ''),
            qualification_evidence_valid=row.get('evidence_valid'),
            qualification_formal_eligible=row.get('formal_eligible'),
            qualification_baseline_frozen=row.get('baseline_frozen'),
            analysis_baseline_frozen=False,
            measurement_usable=False, measurement_unusable_reason='no_bound_recorded_window',
            analysis_energy_usable=False, energy_unusable_reason='no_bound_recorded_window',
            analysis_slo_pass=False, rank_eligible=False, energy_rank='',
            comparison_status='unmeasured', best_feasible_baseline='',
            pdblend_saving_vs_best_feasible_baseline=None,
            pdblend_saving_with_tail_vs_best_feasible_baseline=None, tail_reverses_saving=None,
            candidate_revision=row.get('revision', '') if row['system'] == 'pdblend' else '',
            energy_rank_by_revision={}, comparison_baseline_receipts={}, baseline_energy_rank='',
            comparison_participant_receipts=[], comparison_systems=[], comparison_system_count=0,
            comparison_participant_receipts_by_revision={},
            comparison_variant_count=0, comparison_complete_five_systems=False,
            comparison_excluded_duplicate_receipts=[], best_feasible_baseline_revision='',
            best_feasible_baseline_receipt=None, available_comparison_status='unmeasured',
            available_baseline_count=0, available_baseline_systems=[], available_baseline_receipts={},
            available_baseline_energy_rank='', available_candidate_energy_rank='',
            best_available_feasible_baseline='', pdblend_saving_vs_best_available_baseline=None,
            pdblend_saving_with_tail_vs_best_available_baseline=None, available_tail_reverses_saving=None,
            pdblend_outperformed_by_available_baseline=None)
        record = records.get(row.get('receipt_path'))
        if record is None:
            continue
        result, receipt = record
        for name in ('evidence_valid', 'formal_eligible', 'measurement_evidence_valid', 'profile_qualified'):
            row['original_result_'+name] = result.get(name)
        row['original_receipt_evidence_valid'] = receipt.get('evidence_valid')
        row['original_receipt_cleanup_passed'] = receipt.get('cleanup_passed')
        audit = result.get('observation_acceptance', result.get('acceptance', {}))
        row['qualification_missing_gates'] = deepcopy(result.get('missing_gates', audit.get('missing_gates', [])))
        row['qualification_gate_failures'] = deepcopy(audit.get('gate_failures', {}))
        usable, reason = recorded_request_metrics(result.get('metrics'))
        energy_usable, energy_reason = usable_metrics(result.get('metrics'))
        row.update(measurement_usable=usable, measurement_unusable_reason=reason)
        if not usable:
            row.update(status='failed', comparison_status='recorded_metrics_incomplete')
            continue
        row.update(status='measured', analysis_slo_pass=slo_pass(row), comparison_status='identity_missing',
                   analysis_energy_usable=energy_usable, energy_unusable_reason=energy_reason,
                   analysis_baseline_frozen=row['system'] != 'pdblend')
        key = _identity(row)
        if key is not None:
            groups.setdefault(key, []).append(row)
    if frozen_baselines is not None:
        select_frozen_baselines(rows, frozen_baselines)
    for values in groups.values():
        variants = {}
        for row in values:
            if row['system'] != 'pdblend' and not row.get('frozen_baseline_selected', True):
                row['comparison_status'] = 'historical_baseline_not_selected'
                continue
            if not isinstance(row.get('revision'), str) or not row['revision']:
                row['comparison_status'] = 'missing_revision'
            else:
                variants.setdefault((row['system'], row['revision']), []).append(row)
        duplicates = [r for attempts in variants.values() if len(attempts) != 1 for r in attempts]
        unique = [attempts[0] for attempts in variants.values() if len(attempts) == 1]
        for row in duplicates:
            row['comparison_status'] = 'ambiguous_attempts'
        baselines = [r for r in unique if r['system'] != 'pdblend'
                     and r.get('frozen_baseline_selected', True)]
        candidates = [r for r in unique if r['system'] == 'pdblend']
        baseline_feasible = [r for r in baselines if r['analysis_slo_pass'] and r['analysis_energy_usable']]
        refs = {_variant(r): _reference(r) for r in baselines}
        for row in unique:
            row['comparison_excluded_duplicate_receipts'] = [_reference(r) for r in duplicates]
            row['comparison_baseline_receipts'] = deepcopy(refs)
        for row in baseline_feasible:
            row['baseline_energy_rank'] = _rank(row, baseline_feasible)
        cohorts = [([*baselines, candidate], candidate) for candidate in candidates]
        if not candidates:
            cohorts = [(baselines, None)]
        for cohort, candidate in cohorts:
            feasible = [r for r in cohort if r['analysis_slo_pass'] and r['analysis_energy_usable']]
            systems = sorted({r['system'] for r in cohort})
            revision = candidate['revision'] if candidate is not None else '__baselines_only__'
            for row in cohort:
                row.update(comparison_status='single_observation' if len(systems) > 1 else 'single_system_observation',
                    comparison_systems=systems, comparison_system_count=len(systems),
                    comparison_variant_count=len(cohort), comparison_complete_five_systems=len(systems) == 5)
                row['comparison_participant_receipts_by_revision'][revision] = [_reference(r) for r in cohort]
                if candidate is None or row is candidate or len(candidates) == 1:
                    row['comparison_participant_receipts'] = [_reference(r) for r in cohort]
                if row['analysis_slo_pass'] and row['analysis_energy_usable']:
                    row['rank_eligible'] = True
                    row['energy_rank_by_revision'][revision] = _rank(row, feasible)
                    if row is candidate or len(candidates) <= 1:
                        row['energy_rank'] = _rank(row, feasible)
            if (candidate is not None and candidate['analysis_slo_pass']
                    and candidate['analysis_energy_usable'] and baseline_feasible):
                best = min(baseline_feasible, key=lambda r: (r['energy_service_j'], _variant(r)))
                if best['energy_service_j'] > 0:
                    saving = 1-candidate['energy_service_j']/best['energy_service_j']
                    candidate.update(best_feasible_baseline=best['system'], best_feasible_baseline_revision=best['revision'],
                        best_feasible_baseline_receipt=_reference(best), pdblend_saving_vs_best_feasible_baseline=saving)
                    total, baseline_total = _total(candidate), _total(best)
                    if total is not None and baseline_total is not None and baseline_total > 0:
                        candidate.update(pdblend_saving_with_tail_vs_best_feasible_baseline=1-total/baseline_total,
                            tail_reverses_saving=saving > 0 and total > baseline_total)
        if len(candidates) > 1:
            for row in baselines:
                row['comparison_status'] = 'multiple_candidate_revisions'
                # A scalar list cannot describe several independent rankings.
                # The mapping above retains each candidate's exact cohort.
                row['comparison_participant_receipts'] = []
        # Older available_* fields keep their original meanings under strict_*
        # and expose the same actual-data subset in this explicit analysis mode.
        for row in unique:
            row.update(available_comparison_status=row['comparison_status'],
                available_baseline_count=len(baselines), available_baseline_systems=sorted({r['system'] for r in baselines}),
                available_baseline_receipts=deepcopy(refs), available_baseline_energy_rank=row['baseline_energy_rank'],
                available_candidate_energy_rank=row['energy_rank'] if row['system'] == 'pdblend' else '',
                best_available_feasible_baseline=row['best_feasible_baseline'],
                pdblend_saving_vs_best_available_baseline=row['pdblend_saving_vs_best_feasible_baseline'],
                pdblend_saving_with_tail_vs_best_available_baseline=row['pdblend_saving_with_tail_vs_best_feasible_baseline'],
                available_tail_reverses_saving=row['tail_reverses_saving'],
                pdblend_outperformed_by_available_baseline=(row['pdblend_saving_vs_best_feasible_baseline'] < 0
                    if row['pdblend_saving_vs_best_feasible_baseline'] is not None else None))
    return rows


def select_frozen_baselines(rows, frozen_baselines):
    """Select fixed receipt references, then first complete observations for gaps.

    ``frozen_baselines`` maps point names to original receipt SHA256 values.
    An empty map opts into first-complete selection for a new extension series.
    Neither SLO success nor the size of the energy value influences selection.
    """
    groups = {}
    for row in rows:
        if row['system'] == 'pdblend':
            continue
        row.update(frozen_baseline_selected=False,
                   baseline_selection_rule='first_recorded_complete_service_energy',
                   frozen_baseline_receipt=None)
        key = _identity(row)
        if key is not None and row.get('measurement_usable'):
            groups.setdefault((row['system'], key), []).append(row)
    for values in groups.values():
        fixed = {frozen_baselines[r['point_id']] for r in values if r['point_id'] in frozen_baselines}
        if len(fixed) > 1:
            raise ValueError('conflicting frozen baseline references for one comparison identity')
        if fixed:
            matches = [r for r in values if r['receipt_sha256'] in fixed]
            if len(matches) != 1:
                raise ValueError('frozen baseline receipt missing or duplicated in exported observations')
            selected = matches[0]
            if not selected['analysis_energy_usable']:
                raise ValueError('frozen comparison baseline does not have recorded service energy')
            rule = 'explicit_frozen_receipt'
        else:
            complete = [r for r in values if r['analysis_energy_usable']]
            if not complete:
                continue
            selected = min(complete, key=lambda r: (r['service_start_s'], r['receipt_sha256']))
            rule = 'first_recorded_complete_service_energy'
        for row in values:
            row.update(frozen_baseline_selected=row is selected,
                       baseline_selection_rule=rule, frozen_baseline_receipt=_reference(selected))
