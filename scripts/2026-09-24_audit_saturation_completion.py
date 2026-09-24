#!/usr/bin/env python3
"""Account for exact frozen PD windows and baseline endpoints without new gates.

This is a CPU-only completion report. SLO failures and incomplete energy remain
observations; an unmeasured queue job is never counted as a measurement.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
from pathlib import Path

from pdblend.bench.comparison_campaign import binding, load_bound, failed_session_points
from pdblend.bench.resident_session import digest
from pdblend.bench.single_observation_slo_boundary import read_extension_manifest

DATASETS = ('alpaca', 'sharegpt', 'longbench')
BASELINES = ('mixed', 'distserve', 'ecoserve', 'dynamollm')


def read(path):
    return json.loads(Path(path).read_text())


def truth(value):
    return value is True or value == 'True'


def observed(point, rows):
    """Use exact point identity; retain all attempts without choosing a winner."""
    matching = [r for r in rows if r.get('point_id') == point['name']
                and r.get('point_sha256') == digest(point)]
    receipts, failures = {}, []
    for row in matching:
        if row.get('receipt_sha256'):
            ref = dict(path=row['receipt_path'], sha256=row['receipt_sha256'])
            receipt = load_bound(ref)
            if receipt.get('point_sha256') != digest(point):
                raise ValueError('receipt differs from expected frozen point')
            receipts[ref['sha256']] = dict(receipt=ref,
                recorded_window_complete=bool(receipt.get('recorded_window_complete')),
                slo_pass=truth(row.get('analysis_slo_pass', row.get('slo_pass'))),
                service_energy_present=row.get('energy_service_j') not in ('', None),
                tail_energy_present=row.get('energy_tail_j') not in ('', None),
                energy_usable=truth(row.get('analysis_energy_usable')))
        elif row.get('session_completion_sha256'):
            ref = dict(path=row['session_completion_path'], sha256=row['session_completion_sha256'])
            report = load_bound(ref)
            identities = failed_session_points(Path(ref['path']), report)
            if identities.get(point['name']) != digest(point):
                raise ValueError('session failure is not bound to the expected point')
            if not report.get('error') and report.get('complete') is not False:
                continue
            failures.append(dict(evidence=ref, reason=row.get('failure_reason') or report.get('error')))
    completed = any(r['recorded_window_complete'] for r in receipts.values())
    status = 'observed' if completed else 'obstructed' if receipts or failures else 'pending'
    return dict(point=point['name'], point_sha256=digest(point), revision=point.get('revision'),
                status=status, observations=list(receipts.values()), obstructions=failures)


def reused_observation(ref, rows):
    receipt = load_bound(ref)
    matches = [r for r in rows if r.get('receipt_sha256') == ref['sha256']]
    if not matches:
        return dict(status='pending', expected_receipt=ref, reason='frozen receipt absent from CSV')
    if any(r.get('point_sha256') != receipt.get('point_sha256') for r in matches):
        raise ValueError('reused receipt point identity differs from CSV')
    return dict(status='observed', receipt=ref, selection='predeclared_frozen_reuse')


def counts(records):
    return {status: sum(r['status'] == status for r in records)
            for status in ('observed', 'bracketed', 'obstructed', 'pending')}


def group_failure(point, queue, run_id):
    """A new failed session may be hidden by an older CSV receipt of the same point."""
    evidence = []
    for lease in queue.get('leases', {}).values():
        job = queue.get('jobs', {}).get(lease['job_id'], {})
        if job.get('payload', {}).get('run_id') != run_id or lease.get('status') == 'active':
            continue
        path = Path(lease['attempt_dir']) / 'session/completion.json'
        if not path.is_file():
            continue
        report = read(path)
        if (report.get('schema') == 'resident-group-session/v1' and report.get('complete') is False
                and failed_session_points(path, report).get(point['name']) == digest(point)):
            evidence.append(dict(evidence=binding(path), reason=report.get('error') or 'resident session incomplete'))
    return evidence


def audit(package, csv_path, *, gap_review=None):
    package = Path(package)
    campaign = read(package / 'campaign.json')
    state = read(package / 'orchestration/status.json')
    rows = list(csv.DictReader(Path(csv_path).open()))
    run_id = campaign['run_id']
    points = [p for p in campaign['points'] if p['system'] == 'pdblend'
              and p.get('run_id') == run_id and p.get('comparison_scope') == 'original_matrix_full_group']
    slots = {(p['model_id'], p['dataset'], p['scale']) for p in points}
    models = {p['model_id'] for p in points}
    expected_slots = {(m, d, scale) for m in models for d in DATASETS for scale in (.25, .5, .75, 1.)}
    if (len(points) != 36 or len(models) != 3 or slots != expected_slots
            or len({p['revision'] for p in points}) != 1):
        raise ValueError('completion requires exactly 36 original points from one frozen PD revision')
    originals = [observed(p, rows) for p in points]
    by_model = {model: [r for p, r in zip(points, originals) if p['model_id'] == model]
                for model in sorted({p['model_id'] for p in points})}
    boundaries, endpoints, baseline_points = [], [], []
    for model in by_model:
        selection_path = package / 'orchestration' / ('boundary-selection-' + model.split('-')[1].lower() + '.json')
        active_selection = state.get('boundary_selections', {}).get(model)
        if active_selection:
            selection_path = Path(active_selection['path'])
            selection = load_bound(active_selection)
        else:
            selection = read(selection_path) if selection_path.is_file() else None
        if selection and (selection['run_id'] != run_id or selection['model_id'] != model):
            raise ValueError('boundary selection belongs to another round')
        baseline_path = state.get('baseline_campaigns', {}).get(model)
        baseline = read(baseline_path) if baseline_path else None
        if baseline and baseline['run_id'] != run_id:
            raise ValueError('baseline campaign belongs to another round')
        registry, trace_proofs = [], []
        if baseline:
            from pdblend.bench.comparison_trace_equivalence import read_registry, read_equivalence, validate_reuse
            registry_ref = baseline.get('historical_baseline_registry')
            registry = read_registry(registry_ref, load=load_bound) if registry_ref else []
            trace_proofs = [(read_equivalence(ref, baseline, registry, load=load_bound), ref)
                            for ref in baseline.get('trace_equivalence_refs', [])]
            baseline_points.extend(p for g in baseline['groups'] for p in g['points'])
        manifest = (read_extension_manifest(selection['manifest'])['manifest']
                    if selection and selection.get('manifest') else None)
        active_policies = campaign.get('active_extension_policy_refs', campaign.get('extension_policy_refs', []))
        if manifest and (manifest['run_id'] != run_id or manifest['model_id'] != model
                         or manifest['revision'] != points[0]['revision']
                         or manifest['policy'] not in active_policies):
            raise ValueError('boundary manifest belongs to another round')
        if selection and (selection.get('classification_sidecars') or any(
                any(key in row for key in ('classification', 'classification_sidecars', 'raw_boundary_state'))
                for row in selection.get('datasets', {}).values())):
            helper_path = Path(__file__).with_name('2026-09-25_apply_explicit_rejection_boundaries.py')
            helper_spec = importlib.util.spec_from_file_location('classified_boundary_selection', helper_path)
            helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
            helper.validate_selection(selection, campaign,
                authorized_refs=state.get('explicit_rejection_sidecars', []))
        for dataset in DATASETS:
            selected = selection['datasets'][dataset] if selection else {}
            boundary = dict(model_id=model, dataset=dataset, status='pending')
            source_state = manifest.get('boundaries', {}).get(dataset, {}) if manifest else {}
            if selected.get('classification_sidecars'):
                source_state = selected  # Independently revalidated above; native manifest stays unchanged.
                boundary.update(classification=selected['classification'],
                    classification_sidecars=selected['classification_sidecars'],
                    native_boundary_status=selected['raw_boundary_state']['status'])
            if selected.get('lower') and selected.get('upper'):
                if source_state.get('status') != 'bracketed':
                    raise ValueError('endpoint selection lacks the new round boundary evidence')
                boundary.update(status='bracketed', unique_boundary=not source_state.get('nonmonotonic'),
                                evidence=selection['manifest'])
                for side in ('lower', 'upper'):
                    ref = selected[side]
                    expected_ref = source_state['passed_lower_point' if side == 'lower' else 'failed_upper_point']
                    if ref != expected_ref:
                        raise ValueError('selected endpoint differs from verified boundary manifest')
                    pd = load_bound(ref)
                    if (pd.get('run_id') != run_id or pd['system'] != 'pdblend' or pd['dataset'] != dataset
                            or pd['model_id'] != model or pd['revision'] != points[0]['revision']):
                        raise ValueError('endpoint is not a PD point in the new round')
                    for system in BASELINES:
                        reuse = [r for r in (baseline or {}).get('predeclared_baseline_reuse', [])
                                 if r['dataset'] == dataset and r['side'] == side and r['system'] == system]
                        scheduled = [p for group in (baseline or {}).get('groups', [])
                                     for p in group['points'] if p['system'] == system
                                     and p['name'] == pd['name'].replace('-pdblend-', '-' + system + '-', 1)
                                     and p['trace'] == pd['trace']]
                        if len(reuse) > 1 or len(scheduled) > 1 or (reuse and scheduled):
                            raise ValueError('ambiguous baseline endpoint selection')
                        if reuse:
                            validate_reuse(reuse[0], ref, baseline, registry, trace_proofs, load=load_bound)
                        outcome = reused_observation(reuse[0]['receipt'], rows) if reuse else (
                            observed(scheduled[0], rows) if scheduled else dict(status='pending'))
                        endpoints.append(dict(model_id=model, dataset=dataset, side=side, system=system,
                                              pd_point=ref, **outcome))
            elif manifest and source_state.get('status') in (
                    'incomplete_observation', 'conflicting_observations', 'numeric_search_limit'):
                boundary.update(status='obstructed', reason=source_state['status'],
                                evidence=selection['manifest'])
            elif selection and any(r['status'] == 'obstructed' for r in by_model[model]):
                boundary.update(status='obstructed', reason='bound resident session execution failure',
                                evidence=binding(selection_path))
            boundaries.append(boundary)
    gaps = load_bound(gap_review) if gap_review else None
    gap_accounted, gap_outcomes = False, []
    if gaps is not None:
        gap_rows = gaps.get('rows', [])
        inventory = load_bound(campaign.get('authorized_baseline_energy_gaps', campaign['baseline_energy_gaps']))
        expected = {g['point_id']: g for g in inventory['gaps']}
        if (gaps.get('schema') != 'authorized-baseline-completion/v1'
                or gaps.get('campaign') != binding(package / 'campaign.json')
                or gaps.get('authorized_count') != 20 or len(gap_rows) != 20
                or {g['point_id'] for g in gap_rows} != set(expected)):
            raise ValueError('baseline gap report does not cover the original 20 identities')
        path = Path(__file__).with_name('2026-09-24_prepare_saturation_round.py')
        spec = importlib.util.spec_from_file_location('completion_identity_verifier', path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        queue_path = Path(__file__).resolve().parents[1] / 'results/2026-09-22/three-model/queue.json'
        queue = read(queue_path) if queue_path.is_file() else {}
        for entry in gap_rows:
            original = expected[entry['point_id']]
            if entry['original_receipt'] != original['receipt']:
                raise ValueError('baseline gap report changed the original receipt')
            if entry['status'] == 'completed':
                if not builder.complete_energy_receipt(entry['completion_receipt'], load_bound(original['point'])):
                    raise ValueError('claimed complete baseline lacks its complete energy observation')
                publication = reused_observation(entry['completion_receipt'], rows)
                if publication['status'] != 'observed':
                    gap_outcomes.append(dict(point_id=entry['point_id'], status='pending',
                        complete_energy=True, receipt=entry['completion_receipt'],
                        reason='complete baseline receipt absent from CSV'))
                    continue
                gap_outcomes.append(dict(point_id=entry['point_id'], status='observed',
                                         complete_energy=True, receipt=entry['completion_receipt']))
            elif entry['status'] not in ('pending', 'external_inflight'):
                raise ValueError('unsupported baseline gap completion status')
            else:
                original_point = load_bound(original['point'])
                old_shas = {r['sha256'] for r in original.get('historical_receipts', [])}
                old_shas.add(original['receipt']['sha256'])
                fresh_rows = [r for r in rows if r.get('receipt_sha256') not in old_shas]
                candidates = [p for p in baseline_points
                              if builder.baseline_identity(p) == builder.baseline_identity(original_point)]
                attempts = [observed(p, fresh_rows) for p in candidates]
                failures = [f for p in candidates for f in group_failure(p, queue, run_id)]
                accounted = failures or any(a['status'] != 'pending' for a in attempts)
                gap_outcomes.append(dict(point_id=entry['point_id'], complete_energy=False,
                    status='obstructed' if accounted else 'pending', attempts=attempts, obstructions=failures,
                    reason='new attempt did not supply complete requested energy' if accounted else 'no new attempt evidence'))
        actual = sum(g['status'] == 'completed' for g in gap_rows)
        if gaps.get('completed_count') != actual or gaps.get('all_complete') != (actual == 20):
            raise ValueError('baseline gap report counters differ from its evidence')
        gap_accounted = all(row['status'] != 'pending' for row in gap_outcomes)
    result = dict(schema='saturation-round-completion-audit/v1', run_id=run_id,
                  campaign=binding(package / 'campaign.json'), csv=binding(csv_path),
                  pd_originals=originals, boundaries=boundaries, baseline_endpoints=endpoints,
                  counts=dict(pd_originals=counts(originals), boundaries=counts(boundaries),
                              baseline_endpoints=counts(endpoints), baseline_service_gaps=counts(gap_outcomes)),
                  baseline_service_gap_review=gap_review, baseline_service_gaps=gaps,
                  baseline_gap_outcomes=gap_outcomes,
                  queue_terminal_status_used_as_measurement=False,
                  slo_failure_disqualifies_observation=False)
    # The gap review has its own exact-identity completion contract. The caller
    # must inspect it; this report does not reinterpret a prepared task as done.
    result['scope_pending'] = any(r['status'] == 'pending'
                                 for r in originals + boundaries + endpoints + gap_outcomes)
    result['all_scope_accounted'] = not result['scope_pending'] and gap_accounted
    result['has_obstructions'] = any(r['status'] == 'obstructed'
                                     for r in originals + boundaries + endpoints + gap_outcomes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--baseline-gap-review', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.package, args.csv,
                   gap_review=binding(args.baseline_gap_review) if args.baseline_gap_review else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(counts=result['counts'], scope_pending=result['scope_pending'], out=str(args.out))))


if __name__ == '__main__':
    main()
