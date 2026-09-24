"""Shared single-observation rank arithmetic; clock scope is caller-explicit."""
import json
import math


def rank_rows_core(rows, *, baseline_systems, clock_predicate):
    """Compare each PDblend revision against the same four frozen baselines.

    A second valid attempt of one revision is ambiguous, never an opportunity
    to select its minimum energy. Baseline rows occur once in the CSV; their
    per-candidate ranks are explicit because one scalar cannot represent
    several different five-system comparisons. Service energy is the primary
    measure; tail reversal is a separate diagnostic against that same baseline.
    """
    clock_qualified = clock_predicate

    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    def total_energy(row):
        service, tail = row.get('energy_service_j'), row.get('energy_tail_j')
        if not positive(service) or type(tail) not in (int, float) or not math.isfinite(tail) or tail < 0:
            return None
        total = service + tail
        reported = row.get('energy_service_tail_j')
        if reported is not None and (not positive(reported) or not math.isclose(reported, total)):
            return None
        return total

    identity_fields = ('image_digest', 'runtime_source_sha256', 'measurement_source_sha256',
                       'model_hash', 'tokenizer_hash', 'gpu_uuids', 'slo_ttft_s', 'slo_tpot_s')
    groups = {}
    for row in rows:
        row.update(rank_eligible=False, energy_rank='', comparison_status='incomplete',
                   best_feasible_baseline='', pdblend_saving_vs_best_feasible_baseline=None,
                   pdblend_saving_with_tail_vs_best_feasible_baseline=None, tail_reverses_saving=None,
                   candidate_revision=row.get('revision', '') if row['system'] == 'pdblend' else '',
                   energy_rank_by_revision={}, comparison_baseline_receipts={}, baseline_energy_rank='')
        key = tuple(row.get(k) for k in ('model_id', 'dataset', 'rate_scale', 'seed', 'duration_s',
                    'trace_sha256', 'measurement_protocol_version'))
        groups.setdefault(key, []).append(row)
    for values in groups.values():
        valid = [r for r in values if r.get('evidence_valid') is True and r.get('formal_eligible') is True]
        counts = {s: sum(r['system'] == s for r in valid) for s in baseline_systems}
        if any(n != 1 for n in counts.values()):
            for r in values:
                r['comparison_status'] = 'ambiguous_attempts' if any(n > 1 for n in counts.values()) else 'incomplete'
            continue
        baselines = [r for r in valid if r['system'] in baseline_systems]
        if any(r.get('baseline_frozen') is not True for r in baselines):
            for r in values:
                r['comparison_status'] = 'baseline_not_frozen'
            continue
        candidates = {}
        for r in valid:
            if r['system'] == 'pdblend':
                if not isinstance(r.get('revision'), str) or not r['revision']:
                    r['comparison_status'] = 'missing_revision'
                else:
                    candidates.setdefault(r['revision'], []).append(r)
        compared = []
        for revision, attempts in candidates.items():
            if len(attempts) != 1:
                for r in attempts:
                    r['comparison_status'] = 'ambiguous_attempts'
                continue
            candidate = attempts[0]
            cohort = [*baselines, candidate]
            # Public runtime, hardware and measurement identities must agree
            # within each revision, not across unrelated optimization attempts.
            if any(any(r.get(field) in (None, '', []) for r in cohort)
                   or len({json.dumps(r[field], sort_keys=True) for r in cohort}) != 1
                   for field in identity_fields):
                candidate['comparison_status'] = 'identity_mismatch'
                continue
            if not all(clock_qualified(r) for r in cohort):
                candidate['comparison_status'] = 'clock_evidence_unqualified'
                continue
            feasible = sorted((r for r in cohort if r.get('slo_pass') is True
                               and positive(r.get('energy_service_j'))), key=lambda r: r['energy_service_j'])
            feasible_baselines = [r for r in feasible if r['system'] != 'pdblend']
            candidate['comparison_status'] = 'single_observation'
            candidate['comparison_baseline_receipts'] = {
                r['system']: dict(path=r.get('receipt_path', ''), sha256=r.get('receipt_sha256', ''))
                for r in baselines}
            compared.append(revision)
            for r in feasible:
                rank = 1 + sum(x['energy_service_j'] < r['energy_service_j'] for x in feasible)
                r['rank_eligible'] = True
                r['energy_rank_by_revision'][revision] = rank
                if r is candidate:
                    r['energy_rank'] = rank
            if candidate in feasible and feasible_baselines:
                best = feasible_baselines[0]
                saving = 1 - candidate['energy_service_j'] / best['energy_service_j']
                candidate.update(best_feasible_baseline=best['system'],
                    pdblend_saving_vs_best_feasible_baseline=saving)
                total, baseline_total = total_energy(candidate), total_energy(best)
                if total is not None and baseline_total is not None:
                    candidate.update(pdblend_saving_with_tail_vs_best_feasible_baseline=1-total/baseline_total,
                                     tail_reverses_saving=saving > 0 and total > baseline_total)
        for row in baselines:
            if compared:
                row['comparison_status'] = 'single_observation' if len(compared) == 1 else 'multiple_candidate_revisions'
                ranks = row['energy_rank_by_revision']
                row['energy_rank'] = next(iter(ranks.values())) if len(compared) == 1 and ranks else ''
                if row['rank_eligible']:
                    row['baseline_energy_rank'] = 1 + sum(
                        b.get('slo_pass') is True and positive(b.get('energy_service_j'))
                        and b['energy_service_j'] < row['energy_service_j'] for b in baselines)
            elif candidates:
                row['comparison_status'] = ('ambiguous_attempts' if any(len(a) > 1 for a in candidates.values())
                                            else 'clock_evidence_unqualified' if any(
                                                r.get('comparison_status') == 'clock_evidence_unqualified'
                                                for attempts in candidates.values() for r in attempts)
                                            else 'identity_mismatch')
    from .comparison_partial import add_available_comparisons
    return add_available_comparisons(rows, identity_fields, baseline_systems,
                                     clock_predicate=clock_predicate)

