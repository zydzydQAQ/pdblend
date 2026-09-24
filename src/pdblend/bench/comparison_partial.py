"""Explicit interim comparisons against the frozen baselines already observed.

These columns never replace the complete five-system ranking or claim global
optimality. One matching baseline can already disprove an observed advantage.
"""
import json
import math
from .comparison_clock_evidence import qualified as clock_qualified


def add_available_comparisons(rows, identity_fields, baseline_systems, *, clock_predicate=clock_qualified):
    clock_qualified = clock_predicate
    def positive(value):
        return type(value) in (int,float) and math.isfinite(value) and value>0

    def identity(row):
        if any(row.get(k) in (None,'',[]) for k in identity_fields):return None
        return tuple(json.dumps(row[k],sort_keys=True) for k in identity_fields)

    def total(row):
        service,tail=row.get('energy_service_j'),row.get('energy_tail_j')
        if not positive(service) or type(tail) not in (int,float) or not math.isfinite(tail) or tail<0:return None
        actual=service+tail; reported=row.get('energy_service_tail_j')
        return actual if reported is None or positive(reported) and math.isclose(reported,actual) else None

    groups={}
    for row in rows:
        row.update(available_comparison_status='no_matching_frozen_baseline',
            available_baseline_count=0,available_baseline_systems=[],available_baseline_receipts={},
            clock_unqualified_baseline_systems=[],
            available_baseline_energy_rank=None,available_candidate_energy_rank=None,
            best_available_feasible_baseline='',pdblend_saving_vs_best_available_baseline=None,
            pdblend_saving_with_tail_vs_best_available_baseline=None,
            available_tail_reverses_saving=None,pdblend_outperformed_by_available_baseline=None)
        key=tuple(row.get(k) for k in ('model_id','dataset','rate_scale','seed','duration_s',
                                      'trace_sha256','measurement_protocol_version'))
        groups.setdefault(key,[]).append(row)
    for values in groups.values():
        valid=[r for r in values if r.get('evidence_valid') is True and r.get('formal_eligible') is True]
        if any(sum(r['system']==s for r in valid)>1 for s in baseline_systems):
            for row in values:row['available_comparison_status']='ambiguous_baseline_attempts'
            continue
        baselines=[r for r in valid if r['system'] in baseline_systems and r.get('baseline_frozen') is True]
        for row in valid:
            key=identity(row)
            if key is None:
                row['available_comparison_status']='incomplete_identity';continue
            candidate=row['system']=='pdblend'
            if candidate and (not row.get('revision') or sum(r['system']=='pdblend'
                    and r.get('revision')==row['revision'] for r in valid)!=1):
                row['available_comparison_status']='ambiguous_candidate_revision';continue
            if not candidate and row not in baselines:continue
            matching=[r for r in baselines if identity(r)==key]
            row['clock_unqualified_baseline_systems']=sorted(
                r['system'] for r in matching if not clock_qualified(r))
            matched=[r for r in matching if clock_qualified(r)]
            row.update(available_baseline_count=len(matched),available_baseline_systems=sorted(r['system'] for r in matched),
                available_baseline_receipts={r['system']:dict(path=r.get('receipt_path',''),sha256=r.get('receipt_sha256',''))
                                            for r in matched})
            if not clock_qualified(row) or (matching and not matched):
                row['available_comparison_status']='clock_evidence_unqualified'
                continue
            if not matched:continue
            row['available_comparison_status']=('partial_single_observation' if len(matched)<len(baseline_systems)
                                                else 'all_baselines_single_observation')
            feasible=sorted([r for r in matched if r.get('slo_pass') is True and positive(r.get('energy_service_j'))],
                            key=lambda r:r['energy_service_j'])
            if row.get('slo_pass') is not True or not positive(row.get('energy_service_j')):continue
            rank=1+sum(r['energy_service_j']<row['energy_service_j'] for r in feasible)
            row['available_candidate_energy_rank' if candidate else 'available_baseline_energy_rank']=rank
            if not candidate or not feasible:continue
            best=feasible[0];saving=1-row['energy_service_j']/best['energy_service_j']
            row.update(best_available_feasible_baseline=best['system'],pdblend_saving_vs_best_available_baseline=saving,
                       pdblend_outperformed_by_available_baseline=saving<0)
            full,baseline_full=total(row),total(best)
            if full is not None and baseline_full is not None:
                row.update(pdblend_saving_with_tail_vs_best_available_baseline=1-full/baseline_full,
                           available_tail_reverses_saving=saving>0 and full>baseline_full)
    return rows
