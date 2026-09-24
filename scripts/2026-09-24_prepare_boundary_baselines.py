#!/usr/bin/env python3
"""Prepare each model's four frozen baseline groups after its PD boundary.

The observation endpoints are selected before baseline outcomes are known.
No historical result, baseline policy, profile or controller is modified.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path

from pdblend.bench.comparison_campaign import binding, group_points, load_bound
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest, write_new

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ('mixed', 'distserve', 'ecoserve', 'dynamollm')
ISOLATED = ROOT / 'results/2026-09-24/resident-comparison-isolated-sampling-v1'


def same_workload(old, point, compatibility=(), trace_proofs=()):
    from pdblend.bench.comparison_trace_equivalence import permits
    same = (old['model_id'] == point['model_id'] and old['dataset'] == point['dataset']
        and old['rate_scale'] == point['scale'] and old['seed'] == point['seed']
        and old['duration_s'] == point['duration_s']
        and (old['trace'] == point['trace'] or permits(trace_proofs, old['trace'], point['trace']))
        and old['slo'] == point['slo']
        and all(old['engine_identity'][k] == point['engine_identity'][k]
            for k in ('image_digest', 'model_hash', 'tokenizer_hash', 'fleet_gpu_uuids',
                      'runtime_source_sha256')))
    if not same:
        return False
    old_meter = old['engine_identity']['measurement_source_sha256']
    new_meter = point['engine_identity']['measurement_source_sha256']
    if old_meter == new_meter:
        return True
    for review in compatibility:
        sources = review['sources']
        for left, right in (sources, sources[::-1]):
            if (left['source_manifest'] == old.get('source_manifest')
                    and right['source_manifest'] == point['source_manifest']
                    and left['measurement_source_sha256'] == old_meter
                    and right['measurement_source_sha256'] == new_meter):
                return True
    # A telemetry-only method revision must not silently cause an already
    # frozen baseline workload to run again. Bind the exact source review.
    raise ValueError('matching frozen baseline needs explicit measurement compatibility, not a rerun')


def active_standard_points(campaign):
    """Resolve one declared scheduling variant while retaining all history."""
    points=campaign['points'];selected=campaign.get('active_standard_point_sha256s')
    if selected is None:
        if len({p['name'] for p in points})!=len(points):
            raise ValueError('multiple historical variants require an explicit active point map')
        return deepcopy(points)
    variants={(p['name'],digest(p)):p for p in points}
    try:return [deepcopy(variants[(name,sha)]) for name,sha in selected.items()]
    except KeyError as exc:raise ValueError('active standard point binding differs') from exc


def append_publication_variants(points, additions):
    variants={(p['name'],digest(p)):p for p in points}
    for point in additions:variants.setdefault((point['name'],digest(point)),point)
    return list(variants.values())


def prepare(campaign_path, selection_path, out, *, root_path=None, historical_baseline_registry=None, include_original_gaps=True):
    root=Path(root_path).resolve() if root_path else ROOT
    campaign_ref, selection_ref = binding(campaign_path), binding(selection_path)
    campaign, selection = load_bound(campaign_ref), load_bound(selection_ref)
    if selection.get('classification_sidecars') or any(
            any(key in row for key in ('classification', 'classification_sidecars', 'raw_boundary_state'))
            for row in selection.get('datasets', {}).values()):
        import importlib.util
        helper_path = Path(__file__).with_name('2026-09-25_apply_explicit_rejection_boundaries.py')
        helper_spec = importlib.util.spec_from_file_location('classified_boundary_selection', helper_path)
        helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
        helper.validate_selection(selection, campaign)
    inventory = load_bound(campaign['baseline_energy_gaps'])
    from pdblend.bench.measurement_compatibility import load_compatibility
    compatibility = [load_compatibility(ref) for ref in campaign.get('measurement_compatibility', [])]
    model, run_id = selection['model_id'], campaign['run_id']
    size = model.split('-')[1].lower()
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    from pdblend.bench.comparison_trace_equivalence import prepare_reuse, read_equivalence
    registry_ref = historical_baseline_registry or campaign.get('historical_baseline_registry')
    registry, proof_refs, trace_proofs = [], [], []
    if registry_ref:
        endpoints_to_check = [boundary[side] for boundary in selection['datasets'].values()
                              for side in ('lower', 'upper') if boundary.get(side)]
        registry, proof_refs = prepare_reuse(campaign, endpoints_to_check, registry_ref,
                                             out/'trace-equivalence', load=load_bound)
        trace_proofs = [read_equivalence(ref, campaign, registry, load=load_bound) for ref in proof_refs]
    frozen_rows = list(inventory['frozen_baselines'])
    for record in registry:
        prior = record['point_spec']
        frozen_rows.append(dict(prior, point_id=prior['name'], rate_scale=prior['scale'],
                                point=record['point'], receipt=record['receipt']))
    points = active_standard_points(campaign)
    gap_ids = {row['point_id'] for row in inventory['gaps'] if row['model_id'] == model}
    todo = {p['name']: p for p in points if p['name'] in gap_ids} if include_original_gaps else {}
    reused, endpoints = [], []
    isolated_source = Path(load_bound(load_bound(binding(root / 'results/2026-09-24/resident-comparison-isolated-sampling-v1/campaign.json'))['execution_inputs'])['source'])
    # Explicit baseline sources are frozen before any boundary measurement.
    baseline_sources = campaign.get('baseline_execution_sources', {})
    for dataset, boundary in selection['datasets'].items():
        for side in ('lower', 'upper'):
            ref = boundary.get(side)
            if not ref:
                continue
            pd = load_bound(ref)
            if (pd['system'] != 'pdblend' or pd['dataset'] != dataset or pd['model_id'] != model
                    or pd.get('run_id') != run_id):
                raise ValueError('boundary endpoint does not belong to the frozen PD revision')
            endpoints.append(dict(dataset=dataset, side=side, scale=pd['scale'], point=ref, trace=pd['trace']))
            for system in SYSTEMS:
                template = next(p for p in points if p['system'] == system
                    and p['model_id'] == model and p['dataset'] == dataset and p['scale'] == 1.)
                frozen = [r for r in frozen_rows
                          if r['system'] == system and same_workload(r, pd, compatibility, trace_proofs)]
                frozen = list({r['receipt']['sha256']:r for r in frozen}.values())
                if len(frozen) > 1:
                    raise ValueError('ambiguous predeclared frozen baseline')
                if frozen:
                    load_bound(frozen[0]['receipt'])
                    reused.append(dict(dataset=dataset, side=side, scale=pd['scale'], system=system,
                        receipt=frozen[0]['receipt'], reason='matching_predeclared_frozen_result',
                        trace_equivalence_refs=[proof_ref for proof, proof_ref in zip(trace_proofs, proof_refs)
                            if proof['historical_receipt']==frozen[0]['receipt'] and proof['target_point']==ref]))
                    continue
                name = pd['name'].replace('-pdblend-', '-' + system + '-', 1)
                if name in todo:
                    if todo[name]['trace'] != pd['trace']:
                        raise ValueError('energy supplement and boundary use different traces')
                    todo[name].setdefault('boundary_sides', []).append(side)
                    continue
                point = deepcopy(template)
                # A new workload is an endpoint measurement, not an energy
                # retry of the scale-one template. Real gaps were handled by
                # the existing-name branch above and keep their original link.
                point.pop('energy_supplement_of', None)
                # Base points use original short IDs; adaptive points carry the
                # exact same suffix and trace identity for every system.
                point.update(name=name, scale=pd['scale'], rate_rps=pd['rate_rps'], trace=pd['trace'],
                    run_id=run_id, status='prepared', blockers=[], result_policy='all_recorded_windows/v1',
                    comparison_scope='boundary_full_group', boundary_sides=[side],
                    boundary_selection=selection_ref, pd_boundary_point=ref,
                    comparison_selection='predeclared_endpoint_before_baseline_measurement')
                source_ref = baseline_sources.get(size, {}).get(system)
                if source_ref is None:
                    source_ref = binding(isolated_source / 'manifest.json') if system in (
                        'distserve', 'dynamollm') else template['source_manifest']
                load_bound(source_ref)
                point.update(source_manifest=source_ref,
                             revision=load_bound(source_ref)['source_sha256'])
                if system in ('distserve', 'dynamollm'):
                    point['metering_execution'] = 'isolated_process'
                    point['engine_identity']['metering_execution'] = 'isolated_process'
                if 'inputs' in point:
                    point['inputs'].update(trace=pd['trace'], source_manifest=source_ref)
                    config = load_bound(point['inputs']['system_config'])
                    if system == 'dynamollm':
                        config['trace'] = pd['trace']['path']
                    config_path = out / 'configs' / (name + '.json')
                    write_new(config_path, config)
                    point['inputs']['system_config'] = binding(config_path)
                    if system == 'distserve':
                        choice = load_bound(point['inputs']['offline_choice'])
                        if choice.get('selection') != 'predeclared_fixed_native_topology_no_offline_search':
                            raise ValueError('boundary requires the unchanged predeclared DistServe deployment')
                        choice.update(trace=pd['trace'], rate_rps=pd['rate_rps'], slo=pd['slo'],
                                      boundary_selection=selection_ref)
                        choice_path = out / 'choices' / (name + '.json')
                        write_new(choice_path, choice)
                        point['inputs']['offline_choice'] = binding(choice_path)
                todo[name] = point
    groups = group_points(list(todo.values()))
    if len(groups) > 4:
        raise ValueError('baseline points do not fit the four predeclared compatible groups')
    for group in groups:
        system = group['points'][0]['system']
        source_refs = {p['source_manifest']['sha256'] for p in group['points']}
        if len(source_refs) != 1:
            raise ValueError('one baseline resident group cannot mix source packages')
        group['session_id'] = 'boundary-' + digest(dict(run_id=run_id, model=model,
            system=system, points=group['points']))[:20]
    new = dict(campaign, campaign_id=out.name, parent_campaign=campaign_ref,
        points=append_publication_variants(campaign['points'],todo.values()), groups=groups,
        execution_campaigns=sorted(set(campaign.get('execution_campaigns', []) + [str(Path(campaign_path).resolve())])),
        boundary_selection=selection_ref, predeclared_baseline_reuse=reused,
        original_energy_gaps_included=include_original_gaps)
    if selection.get('classification_sidecars'):
        new['explicit_rejection_sidecars'] = list(campaign.get('explicit_rejection_sidecars', []))
        for ref in selection['classification_sidecars']:
            if ref not in new['explicit_rejection_sidecars']:
                new['explicit_rejection_sidecars'].append(ref)
    if registry_ref:
        new['historical_baseline_registry'] = registry_ref
        new['trace_equivalence_refs'] = list(campaign.get('trace_equivalence_refs', [])) + proof_refs
    write_new(out / 'campaign.json', new)
    write_new(out / 'comparison-selection.json', dict(schema='boundary-baseline-selection/v1',
        model_id=model, run_id=run_id, selection=selection_ref, endpoints=endpoints,
        reused=reused, new_points=[p['name'] for p in todo.values()],
        baseline_results_used_for_selection=False, original_evidence_changed=False))
    execution = load_bound(campaign['execution_inputs'])
    jobs = []
    for system in SYSTEMS:
        for group in groups:
            if group['points'][0]['system'] != system:
                continue
            path = out / 'groups' / (group['session_id'] + '.json'); write_new(path, group)
            source = Path(group['points'][0]['source_manifest']['path']).parent
            job = resident_job(group, path, root=root, source=source, image=execution['image_digest'],
                verification=execution['model_verification']['path'], campaign=out / 'campaign.json',
                priority=2500-len(jobs))
            external = []
            if system == 'dynamollm':
                config = load_bound(group['points'][0]['inputs']['system_config'])
                history = json.loads(Path(config['dynamo_weekly_history']).read_text())
                history_path = Path(history['source_path']).resolve()
                external = [dict(path=str(history_path), expected_sha256=history['source_sha256'],
                    bytes=history_path.stat().st_size, mode='ro', rehashed_during_preparation=False,
                    execution_verifier='pdblend_baselines.dynamollm.validation.verified_history')]
                at = job['payload']['argv'].index(execution['image_digest'])
                job['payload']['argv'][at:at] = ['-v', f'{history_path}:{history_path}:ro']
            job['payload'].update(system=system, model_id=model, run_id=run_id,
                result_policy='all_recorded_windows/v1', external_readonly_inputs=external,
                # The single-owner driver submits each group after the prior
                # lease is terminal. Queue dependencies do not include blocked.
                after_terminal=[],
                baseline_comparison_selection=binding(out / 'comparison-selection.json'))
            jobs.append(job)
    write_new(out / 'jobs.json', jobs)
    write_new(out / 'prepared.json', dict(schema='boundary-baselines-prepared/v1',
        campaign=binding(out / 'campaign.json'), jobs=binding(out / 'jobs.json'),
        selection=selection_ref, parent_campaign=campaign_ref))
    return new, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--historical-baseline-registry', type=Path)
    args = parser.parse_args()
    campaign, jobs = prepare(args.campaign, args.selection, args.out, root_path=args.root,
        historical_baseline_registry=binding(args.historical_baseline_registry) if args.historical_baseline_registry else None)
    print(json.dumps(dict(campaign=str(args.out.resolve() / 'campaign.json'), groups=len(jobs),
                          windows=sum(len(g['points']) for g in campaign['groups']))))


if __name__ == '__main__':
    main()
