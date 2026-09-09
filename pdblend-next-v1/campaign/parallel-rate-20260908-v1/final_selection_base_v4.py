"""Explicit whole-model selection, preserving the physically executed identities."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
REUSE_SHA = '76376fcc054dd23ae4716c8339c033e5b5ce16481572e63cd42b58fa1f07cb0d'
EXPECTED_SERIES = {'7b': 'p4', '14b': 'p8', '32b': 'p4'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(value, message):
    if not value:
        raise ValueError(message)


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'changed selection reference ' + reference['path'])
    return read(reference['path'])


def validate(selection_path, selection_sha, draft=False):
    need(sha(selection_path) == selection_sha, 'selection SHA differs')
    selection = read(selection_path)
    need(selection['schema'] == 'whole-model-final-series-selection-v1', 'unknown selection schema')
    need(selection['no_per_point_version_selection'] is True, 'best-per-point selection forbidden')
    need(set(selection['models']) == set(EXPECTED_SERIES), 'all three models must be selected explicitly')
    need(selection['reuse_declaration']['sha256'] == REUSE_SHA, 'wrong frozen capacity-off reuse declaration')
    reuse = checked(selection['reuse_declaration'])
    need(reuse['actual_source_hashes_never_relabelled'] is True
         and reuse['physical_source_hashes_all_equal'] is False
         and reuse['capacity_policy_equal_across_models'] is False,
         'actual source/policy distinctions must remain visible')
    for path, digest in reuse['files'].items():
        need(sha(path) == digest, 'changed reuse evidence ' + path)
    reuse_models = {entry['model']: entry for entry in reuse['models']}
    for model, spec in selection['models'].items():
        need(spec['series'] == EXPECTED_SERIES[model], 'different model series needs new reviewed collector')
        need(spec['selection_mode'] == 'whole_declared_model_series', 'per-point selection forbidden')
        need(not any(k in spec for k in ('best_points', 'include_cell_ids', 'exclude_cell_ids')),
             'scientific points cannot be selected by outcome')
        enabled = model == '14b'
        need(spec['capacity_integration_v1'] is enabled, 'selected capacity policy differs')
        expected = reuse_models[model]['candidate_manifest' if enabled else 'original_manifest']
        need(spec['actual_manifest'] == expected, 'actual source relabelled or mixed')
        host_manifest = checked(expected)
        host = Path(expected['path']).parent
        for path, digest in host_manifest['files'].items():
            need(sha(host / path) == digest, 'actual serving source changed: ' + str(host / path))
        declaration = spec.get('formal_declaration')
        if declaration is not None:
            checked(declaration)
        if enabled and not draft:
            need(declaration is not None and spec.get('formal_release') is not None,
                 'A actual final declaration/release has not been frozen')
            release = checked(spec['formal_release'])
            need(release['declaration'] == declaration and release['host_manifest'] == expected
                 and release['dynamic_qualified'] is True, 'A formal source/qualification differs')
    for reference in selection['baseline_declarations']:
        checked(reference)
    for reference in selection.get('baseline_continuations', []):
        checked(reference)
    for reference in selection.get('scope_declarations', []):
        checked(reference)
    if not draft:
        need(selection['approved'] is True, 'CPU draft cannot produce a final selected report')
    return selection


def select_stage(selection, model, series, order):
    if model not in selection['models'] or series != selection['models'][model]['series']:
        return False
    spec = selection['models'][model]
    # Capacity qualification experiments do not belong to A's final rate queue.
    if model == '14b':
        need(spec.get('formal_declaration') is not None, 'A final rate declaration missing')
        declared = checked(spec['formal_declaration'])['cells']
        by_id = {row['cell_id']: row for row in declared}
        overlap = [row for row in order if row['cell_id'] in by_id]
        if not overlap:
            return False
        need(len(overlap) == len(order), 'mixed formal and development stage')
        need(all(row == by_id[row['cell_id']] for row in order), 'A formal declaration changed')
    return True


def preserve_identity(point, binding, config, checkpoint, selection):
    spec = selection['models'][point['model']]
    need(point['host_manifest_sha256'] == spec['actual_manifest']['sha256'], 'selected point executed another source')
    need((config.get('capacity_integration_v1') is True) == spec['capacity_integration_v1'],
         'actual capacity flag differs from declared model policy')
    need(point.get('profile_sha256') and point.get('policy_sha256') and point.get('controller_source_sha256'),
         'actual source/profile/policy fingerprint missing')
    point.update(actual_controller_source_sha256=point['controller_source_sha256'],
                 capacity_integration_v1=spec['capacity_integration_v1'],
                 configured_instance_count=len(config['instances']),
                 configured_gpu_count=len({gpu for instance in config['instances'] for gpu in instance['gpus']}),
                 all_eight_gpu_energy=True, original_checkpoint=str(checkpoint),
                 reuse_basis=selection['reuse_declaration'] if point['model'] in ('7b', '32b') else None,
                 selected_model_series=spec['series'], selection_scope='whole_declared_model_series')
    return point


def annotate_scope(points, origins):
    """Honor explicit runner boundary records without deleting an observed negative."""
    for point in points:
        if point['measurement_valid']:
            continue
        entries = origins[point['cell_id']]
        explicit_skips = [point['cell_id'] in (entry['status'].get('skipped_saturated') or []) for entry in entries]
        if any(explicit_skips):
            point.update(scope_status='not_required_above_first_loss', required_execution=False)
    return points


def annotate_grid(grid, selection, originals, points, protocol):
    for reference in selection.get('scope_declarations', []):
        document = checked(reference)
        need(document['schema']=='terminal-first-loss-original-scope-v1','unknown exclusion authority')
        status=checked(document['terminal_status'])
        need(status['complete'] and status['phase']=='complete' and not status['node_lease_held']
             and not status['failed'],'boundary queue not successfully terminal')
        declaration=checked(document['declaration'])
        declared={row['cell_id']:row for row in declaration['cells']}
        order=checked(document['executed_order'])
        by_order={row['cell_id']:row for row in order}
        skipped={item['cell_id'] for item in status['skipped']
                 if item['reason']=='above first complete SLO<0.90'}
        excluded=set()
        original_by_pair={protocol.pair_identity(row):row['cell_id'] for row in originals if row['system']=='pdblend'}
        for cell_id in skipped:
            need(cell_id in by_order,'skip absent from actual executed order')
            row=by_order[cell_id]['source_row']
            need(row['system']=='pdblend','baseline separately above-boundary skipped')
            complete_losses=[point for point in points if point['cell_id'] in status['completed']
                and point['model']==row['model'] and point['dataset']==row['dataset']
                and point['measurement_valid'] and point['work_complete']
                and point['slo_attainment']<.9 and point['rate_rps']==status['first_complete_breach'][row['dataset']]]
            need(complete_losses and row['rate_rps']>complete_losses[0]['rate_rps'],
                 'excluded rate lacks an independently verified complete loss checkpoint')
            metadata=dict(row,n_expected=row['n_requests'])
            if cell_id in declared:
                need(protocol.pair_identity(metadata)==protocol.pair_identity(dict(declared[cell_id],n_expected=declared[cell_id]['n_requests'])),
                     'executed order and parent five-system declaration differ')
            original_id=original_by_pair.get(protocol.pair_identity(metadata))
            need(cell_id in declared or original_id is not None,'excluded point has no original or five-system authority')
            if original_id is not None:excluded.add(original_id)
        exclusions=sorted(excluded)
        need(exclusions==sorted(document['excluded_original_cell_ids']),'scope does not match exact original trace mapping')
        by_id = {row['original_cell_id']: row for row in grid}
        for cell_id in exclusions:
            need(cell_id in by_id, 'scope excludes unknown original point ' + cell_id)
            row = by_id[cell_id]
            need(row['verified_executions'] == 0, 'scope cannot erase an observed point')
            row.update(status='excluded_above_first_complete_loss', required_executions=0,
                       repeat_requirement_complete=True, scope_reference=reference,
                       scope_statuses=['not_required_above_first_loss'])
    return grid
