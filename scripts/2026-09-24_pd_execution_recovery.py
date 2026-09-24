#!/usr/bin/env python3
"""Explicit same-source recovery of one pre-service PD reset failure.

Imported by the single-owner driver; this module never runs on import.
Preparation preserves the old campaign/selection and all completed receipts.
"""
from copy import deepcopy
from pathlib import Path

from pdblend.bench import comparison_campaign as cc
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest, write_new
from pdblend.bench.comparison_extension_recovery import retired_refs

SCHEMA = 'saturation-execution-recovery/v1'


def _dynamic_recovery_module():
    import importlib.util
    path = Path(__file__).with_name('2026-09-25_prepare_dynamic_observation_recovery.py')
    spec = importlib.util.spec_from_file_location('dynamic_observation_recovery', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_plan(runner, ref):
    plan = cc.load_bound(ref)
    if plan.get('recovery_kind') == 'failed_dynamic_observation_resume/v1':
        return _dynamic_recovery_module().verify_plan(runner, ref)
    if plan.get('schema') != SCHEMA or plan['run_id'] != runner.package.name:
        raise ValueError('recovery authority belongs to another round')
    campaign = cc.load_bound(plan['recovery_campaign'])
    prior = cc.load_bound(plan['prior_baseline_campaign'])
    selected = cc.load_bound(plan['prior_selection'])
    job = cc.load_bound(plan['pd_recovery_job'])
    if (campaign['run_id'] != plan['run_id'] or prior['run_id'] != plan['run_id']
            or selected['run_id'] != plan['run_id'] or selected['model_id'] != plan['model_id']
            or job['payload']['run_id'] != plan['run_id'] or job['payload']['model_id'] != plan['model_id']
            or job['payload'].get('system') != 'pdblend'
            or campaign.get('parent_campaign') != plan['prior_baseline_campaign']
            or not retired_refs(campaign, load_bound=cc.load_bound)
            or plan.get('selection_generation', 0) < 2):
        raise ValueError('recovery inputs differ from the explicit campaign authority')
    from pdblend.bench.measurement_compatibility import load_compatibility
    for compatibility_ref in plan.get('additional_measurement_compatibility', []):
        load_compatibility(compatibility_ref)
    return plan, job


def receipts_for_job(queue, job_id):
    refs = []
    for lease in queue['leases'].values():
        if lease['job_id'] != job_id:
            continue
        for path in sorted((Path(lease['attempt_dir']) / 'session/windows').glob('*/receipt.json')):
            ref = cc.binding(path)
            if ref not in refs:
                refs.append(ref)
    return refs


def pending_original_gaps(prior, completed_refs, complete_energy_receipt):
    """Only a matching complete receipt may remove a point, including SLO fails."""
    points = {p['name']: p for group in prior['groups'] for p in group['points']}
    completed = {}
    for ref in completed_refs:
        cc.load_bound(ref)
        point = cc.load_bound(cc.binding(Path(ref['path']).parent / 'point.json'))
        expected = points.get(point['name'])
        if expected is not None and complete_energy_receipt(ref, expected):
            # A second complete receipt is never chosen by energy or SLO.
            if point['name'] in completed and completed[point['name']] != ref:
                raise ValueError('multiple complete gap receipts require explicit first-frozen selection')
            completed[point['name']] = ref
    return [p for name, p in points.items() if name not in completed], completed


def merged_groups(endpoints, pending):
    variants = {}
    for point in [p for g in endpoints['groups'] for p in g['points']] + list(pending):
        if point['name'] in variants and variants[point['name']] != point:
            endpoint = variants[point['name']]
            fields = ('model_id', 'system', 'dataset', 'scale', 'rate_rps', 'seed', 'duration_s',
                      'trace', 'slo', 'source_manifest', 'engine_identity')
            if any(point.get(k) != endpoint.get(k) for k in fields):
                raise ValueError('endpoint and pending original gap have conflicting workload identities')
            point = deepcopy(point)
            point['boundary_sides'] = list(endpoint.get('boundary_sides', []))
        variants[point['name']] = point
    groups = cc.group_points(list(variants.values()))
    if len(groups) > 4:
        raise ValueError('recovery requires more than the four compatible baseline inventories')
    for group in groups:
        sources = {p['source_manifest']['sha256'] for p in group['points']}
        if len(sources) != 1:
            raise ValueError('recovery cannot combine frozen baseline sources')
        group['session_id'] = 'recovered-boundary-' + digest(dict(
            run_id=endpoints['run_id'], points=group['points']))[:20]
    return groups


def prepare_combined(runner, plan, selection, endpoint_dir, out, queue):
    """Freeze endpoint-only preparation, then merge only still-unmeasured gaps."""
    prior = cc.load_bound(plan['prior_baseline_campaign'])
    receipts = []
    for job_id in plan['finish_before_job_ids']:
        receipts += receipts_for_job(queue, job_id)
    verifier = runner.load_module(runner.root_path / 'scripts/2026-09-24_prepare_saturation_round.py')
    pending, complete = pending_original_gaps(prior, receipts, verifier.complete_energy_receipt)
    # The finish barrier must actually account for its declared points. An
    # execution failure needs another explicit recovery, not an automatic rerun.
    finished_names = set()
    for job_id in plan['finish_before_job_ids']:
        argv = queue['jobs'][job_id]['payload']['argv']
        group = runner.read_json(argv[argv.index('--group') + 1])
        finished_names.update(p['name'] for p in group['points'])
    if not finished_names <= set(complete):
        raise ValueError('finish-before baseline job lacks complete bound measurements')
    # Append completed original gaps to a derived immutable inventory. This
    # permits exact endpoint reuse without changing any pre-existing selection,
    # receipt, authorized gap contract, source or metric.
    inventory_dir = endpoint_dir.with_name(endpoint_dir.name + '-inventory')
    inventory_campaign = inventory_dir / 'campaign.json'
    if inventory_campaign.is_file():
        prepared_inventory = cc.load_bound(cc.binding(inventory_campaign))
        if (prepared_inventory.get('completed_baseline_gap_receipts') != complete
                or prepared_inventory.get('parent_campaign') != cc.binding(runner.campaign)
                or any(ref not in prepared_inventory.get('measurement_compatibility', [])
                       for ref in plan.get('additional_measurement_compatibility', []))):
            raise ValueError('completed gap inventory projection changed')
    else:
        parent = cc.load_bound(cc.binding(runner.campaign))
        inventory = deepcopy(cc.load_bound(parent['baseline_energy_gaps']))
        inventory['parent_inventory'] = parent['baseline_energy_gaps']
        inventory['authorized_inventory'] = parent['baseline_energy_gaps']
        inventory['completed_gap_receipts'] = complete
        inventory['prior_summary_fields_retained_as_historical'] = True
        existing = {row['receipt']['sha256'] for row in inventory['frozen_baselines']}
        authorized = {row['point_id']: row for row in inventory['gaps']}
        for name, ref in complete.items():
            point_ref = cc.binding(Path(ref['path']).parent / 'point.json')
            point = cc.load_bound(point_ref)
            if ref['sha256'] not in existing:
                original = authorized[name]
                receipt = cc.load_bound(ref)
                row = deepcopy(original)
                row.update(point_id=name, original_point_id=name, point=point_ref,
                    point_sha256=receipt['point_sha256'], receipt=ref,
                    engine_identity=point['engine_identity'], source_manifest=point['source_manifest'],
                    revision=point['revision'], inputs=point.get('inputs'), trace=point['trace'],
                    result=receipt['result'], energy_service_j=receipt['result']['metrics']['energy_service_j'],
                    energy_tail_j=receipt['result']['metrics']['energy_tail_j'],
                    frozen_completion_of=original['receipt'], selection_rule='first_explicit_complete_gap_receipt')
                inventory['frozen_baselines'].append(row)
                existing.add(ref['sha256'])
        inventory['gaps'] = [row for row in inventory['gaps'] if row['point_id'] not in complete]
        write_new(inventory_dir / 'inventory.json', inventory)
        from pdblend.bench.comparison_export_provenance import load_baseline_inventory
        load_baseline_inventory(cc.binding(inventory_dir / 'inventory.json'), load_bound=cc.load_bound)
        projected = deepcopy(parent)
        from pdblend.bench.measurement_compatibility import load_compatibility
        for compatibility_ref in plan.get('additional_measurement_compatibility', []):
            load_compatibility(compatibility_ref)
            if compatibility_ref not in projected.setdefault('measurement_compatibility', []):
                projected['measurement_compatibility'].append(compatibility_ref)
        projected.update(campaign_id=inventory_dir.name, parent_campaign=cc.binding(runner.campaign),
            baseline_energy_gaps=cc.binding(inventory_dir / 'inventory.json'),
            completed_baseline_gap_receipts=complete)
        projected['baseline_comparison_policy']['frozen_baselines'] = projected['baseline_energy_gaps']
        write_new(inventory_campaign, projected)
    if (endpoint_dir / 'prepared.json').is_file():
        marker = runner.read_json(endpoint_dir / 'prepared.json')
        if marker['selection'] != cc.binding(selection) or marker['parent_campaign'] != cc.binding(inventory_campaign):
            raise ValueError('endpoint-only preparation changed selection or inventory')
        endpoints, endpoint_jobs = cc.load_bound(marker['campaign']), cc.load_bound(marker['jobs'])
    else:
        endpoints, endpoint_jobs = runner.baselines.prepare(inventory_campaign, selection, endpoint_dir,
            root_path=runner.root_path, historical_baseline_registry=runner.baseline_registry_ref,
            include_original_gaps=False)
    groups = merged_groups(endpoints, pending)
    out.mkdir(parents=True, exist_ok=False)
    new = dict(endpoints, campaign_id=out.name, parent_campaign=cc.binding(endpoint_dir / 'campaign.json'),
        groups=groups, points=runner.baselines.append_publication_variants(
            endpoints['points'], [point for group in groups for point in group['points']]),
        execution_campaigns=sorted(set(endpoints.get('execution_campaigns', []) +
            [plan['prior_baseline_campaign']['path'], str(endpoint_dir / 'campaign.json')])),
        completed_baseline_gap_receipts=complete, execution_recovery_plan=runner.recovery_plan_ref)
    write_new(out / 'campaign.json', new)
    # Original gap jobs supply existing external-history mount metadata; the
    # endpoint jobs supply the same frozen configuration for newly built points.
    templates = runner.read_json(Path(plan['prior_baseline_campaign']['path']).parent / 'jobs.json') + endpoint_jobs
    execution = cc.load_bound(new['execution_inputs'])
    jobs = []
    order = {'mixed': 0, 'distserve': 1, 'ecoserve': 2, 'dynamollm': 3}
    for group in sorted(groups, key=lambda g: order[g['points'][0]['system']]):
        system = group['points'][0]['system']
        template = next(j for j in templates if j['payload']['system'] == system)
        path = out / 'groups' / (group['session_id'] + '.json'); write_new(path, group)
        source = Path(group['points'][0]['source_manifest']['path']).parent
        job = resident_job(group, path, root=runner.root_path, source=source,
            image=execution['image_digest'], verification=execution['model_verification']['path'],
            campaign=out / 'campaign.json', priority=2900-len(jobs))
        external = deepcopy(template['payload'].get('external_readonly_inputs', []))
        for item in external:
            at = job['payload']['argv'].index(execution['image_digest'])
            job['payload']['argv'][at:at] = ['-v', f"{item['path']}:{item['path']}:ro"]
        job['payload'].update(system=system, model_id=plan['model_id'], run_id=plan['run_id'],
            result_policy='all_recorded_windows/v1', external_readonly_inputs=external,
            after_terminal=[], baseline_comparison_selection=cc.binding(endpoint_dir / 'comparison-selection.json'))
        jobs.append(job)
    write_new(out / 'jobs.json', jobs)
    write_new(out / 'prepared.json', dict(schema='boundary-baselines-prepared/v1',
        campaign=cc.binding(out / 'campaign.json'), jobs=cc.binding(out / 'jobs.json'),
        selection=cc.binding(selection), parent_campaign=cc.binding(endpoint_dir / 'campaign.json'),
        execution_recovery_plan=runner.recovery_plan_ref))
    return new, jobs


def unused_preparation_path(base):
    out, n = base, 0
    while ((out.exists() and not (out / 'prepared.json').is_file())
           or (not out.exists() and out.with_name(out.name + '-inventory').exists()
               and not (out.with_name(out.name + '-inventory') / 'campaign.json').is_file())):
        n += 1
        out = base.with_name(base.name + f'-partial-recovery-{n:04d}')
    return out


def recover_model(runner, ref):
    if cc.load_bound(ref).get('recovery_kind') == 'failed_dynamic_observation_resume/v1':
        return _dynamic_recovery_module().recover_model(runner, ref)
    plan, job = verify_plan(runner, ref)
    model = plan['model_id']
    progress = runner.state.setdefault('execution_recovery_state', {}).setdefault(ref['sha256'], {})
    if progress.get('complete'):
        if model not in runner.state['completed_models']:
            raise ValueError('recovery checkpoint lost completed model')
        return
    # Small injected interfaces keep the preparation independently testable.
    runner.root_path = Path(__file__).resolve().parents[1]
    import importlib.util, json
    def load_module(path):
        spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-', '_'), path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
    runner.load_module = load_module
    runner.read_json = lambda path: json.loads(Path(path).read_text())
    runner.recovery_plan_ref = ref
    if progress.get('combined_prepared'):
        marker = cc.load_bound(progress['combined_prepared'])
        runner.campaign = Path(marker['campaign']['path'])
    else:
        runner.campaign = Path(plan['recovery_campaign']['path'])
    if plan['finish_before_job_ids']:
        queue = runner.wait(plan['finish_before_job_ids'])
    else:
        queue = runner.read_json(runner.root_path / 'results/2026-09-22/three-model/queue.json')
    runner.state.update(model=model, phase='pdblend_execution_recovery')
    runner.emit('execution_recovery_started', plan=ref)
    if not progress.get('pd_manifest'):
        manifest = runner.pd(job)
        if not manifest:
            raise ValueError('explicit PD recovery failed without a boundary manifest')
        # Store even an obstructed head; never silently execute a second retry.
        progress['pd_manifest'] = manifest
        runner.emit('execution_recovery_pd_observed', manifest=manifest)
    selected = runner.selection(model, progress['pd_manifest'], generation=plan['selection_generation'],
                                supersedes=plan['prior_selection'])
    runner.state.setdefault('boundary_selections', {})[model] = cc.binding(selected)
    base = runner.package / ('baselines-' + model.split('-')[1].lower() + '-selection-v' + str(plan['selection_generation']))
    if progress.get('combined_prepared'):
        marker = cc.load_bound(progress['combined_prepared'])
        if marker.get('execution_recovery_plan') != ref or marker['selection'] != cc.binding(selected):
            raise ValueError('recovery prepared checkpoint changed identity')
        prepared, jobs = cc.load_bound(marker['campaign']), cc.load_bound(marker['jobs'])
        out = Path(marker['campaign']['path']).parent
    else:
        endpoint_dir = unused_preparation_path(base.with_name(base.name + '-endpoints-only'))
        out = unused_preparation_path(base)
        if (out / 'prepared.json').is_file():
            marker = runner.read_json(out / 'prepared.json')
            if marker.get('execution_recovery_plan') != ref or marker['selection'] != cc.binding(selected):
                raise ValueError('existing recovery preparation differs')
            prepared, jobs = cc.load_bound(marker['campaign']), cc.load_bound(marker['jobs'])
        else:
            prepared, jobs = prepare_combined(runner, plan, selected, endpoint_dir, out, queue)
        progress['combined_prepared'] = cc.binding(out / 'prepared.json')
    runner.campaign = out / 'campaign.json'
    runner.state['baseline_campaigns'][model] = str(runner.campaign)
    runner.state['phase'] = 'recovered_baseline_supplements_and_endpoints'
    runner.emit('recovered_baselines_prepared', plan=ref, prepared=progress['combined_prepared'],
                windows=sum(len(g['points']) for g in prepared['groups']), groups=len(jobs))
    for baseline in jobs:
        runner.run_baseline(baseline)
    runner.publish(force=True)
    progress['complete'] = True
    if model not in runner.state['completed_models']:
        runner.state['completed_models'].append(model)
    runner.emit('recovered_model_complete', model_id=model, plan=ref)
