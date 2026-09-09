"""Whole-model selection with pinned profiles and explicit same-release extensions."""
import final_selection_base_v4 as prior
import final_a_qualification_v1 as a_qualification
from final_selection_base_v4 import checked, need, sha, read


def stage_rows(reference):
    value=checked(reference)
    if isinstance(value,list):return value
    if 'cell' in value:return [value['cell']]
    return value['cells']


def frozen_stages(selection):
    result=[]
    for model,spec in selection['models'].items():
        for stage in spec.get('required_stage_orders',[]):
            rows=stage_rows(stage['order']);status=checked(stage['terminal_status'])
            need(status['phase'] in ('complete','stopped_at_boundary') and status.get('finished_s')
                 and status.get('node_lease_held') is False and not status.get('failed'),
                 'required stage authority is not terminal and cleaned')
            need(all(row['model']==model for row in rows),'required stage changed model')
            result.append((model,stage,rows,status))
    return result


def a_declarations(selection):
    spec=selection['models']['14b']
    return ([spec['formal_declaration']] if spec.get('formal_declaration') else []) + spec.get('additional_formal_declarations',[])


def declared_a_cells(selection):
    result={}
    for reference in a_declarations(selection):
        declaration=checked(reference)
        need(declaration.get('model')=='14b' and declaration.get('authorized') is True,
             'A extension is not an authorized model declaration')
        for row in declaration['cells']:
            need(row['model']=='14b' and row['cell_id'].startswith('parallel-rate-p8-'),
                 'A extension contains another model or source series')
            need(row['cell_id'] not in result,'overlapping formal declarations')
            result[row['cell_id']]=(row,reference)
    return result


def validate(selection_path, selection_sha, draft=False):
    selection=prior.validate(selection_path,selection_sha,draft=draft)
    for model,spec in selection['models'].items():
        if not draft:need(spec.get('profile') is not None,'final model profile has not been pinned')
        if not draft and model!='14b':need(spec.get('required_stage_orders'),'entire required C/B queues must be declared')
        if spec.get('profile'):checked(spec['profile'])
    frozen_stages(selection)
    declared_a_cells(selection)
    spec=selection['models']['14b']
    if spec.get('formal_release'):
        release=checked(spec['formal_release'])
        if spec.get('profile'):need(release['profile']==spec['profile'],'A release profile differs')
        for path,digest in release['files'].items():need(sha(path)==digest,'A release dependency changed: '+path)
        checked(release['binding']);checked(release['qualification900'])
        a_qualification.validate(prior,spec,release)
    return selection


def select_stage(selection,model,series,order):
    if model!='14b':return prior.select_stage(selection,model,series,order)
    if series!=selection['models']['14b']['series']:return False
    declared=declared_a_cells(selection)
    overlap=[row for row in order if row['cell_id'] in declared]
    if not overlap:return False
    need(len(overlap)==len(order),'mixed formal and development stage')
    need(all(row==declared[row['cell_id']][0] for row in order),'A formal execution declaration changed')
    return True


def preserve_identity(point,binding,config,checkpoint,selection):
    prior.preserve_identity(point,binding,config,checkpoint,selection)
    spec=selection['models'][point['model']]
    need(point['profile_sha256']==spec['profile']['sha256'],'selected point changed the frozen model profile')
    if point['model']=='14b':
        import source_identity_v4
        release=checked(spec['formal_release'])
        expected=source_identity_v4.identity(checked(release['binding']),point['dataset'])
        need(point['version_id']==expected['version_id'],'A extension changed final source/profile/policy')
    return point


def add_pending_declarations(selection,declarations,origins,sources):
    """A declared extension without a running stage remains an explicit gap."""
    for cid,(row,reference) in declared_a_cells(selection).items():
        sources[reference['path']]=reference['sha256']
        if cid in declarations:
            need(declarations[cid]==row,'formal pending row conflicts with executed declaration')
            continue
        declarations[cid]=row
        origins[cid].append(dict(path=reference['path'],cell=row,
            status=dict(model='14b',phase='awaiting_formal_execution',complete=False)))
    for model,stage,rows,status in frozen_stages(selection):
        for ref in (stage['order'],stage['terminal_status']):sources[ref['path']]=ref['sha256']
        for row in rows:
            cid=row['cell_id']
            if cid in declarations:
                need(declarations[cid]==row,'required frozen stage conflicts with scanned row')
            else:
                declarations[cid]=row
                origins[cid].append(dict(path=stage['order']['path'],cell=row,status=status))


def annotate_scope(points,origins,selection,protocol):
    prior.annotate_scope(points,origins)
    by_id={point['cell_id']:point for point in points}
    for mapping in selection.get('superseded_executions',[]):
        old,new=by_id[mapping['old_cell_id']],by_id[mapping['replacement_cell_id']]
        need(old['status']=='unmeasured' and not old['measurement_valid'],
             'supersession cannot erase an attempted observation')
        need(protocol.pair_identity(old)==protocol.pair_identity(new) and old['repeat']==new['repeat'],
             'superseding execution is another workload or repeat')
        need(all(old['cell_id'] not in (entry['status'].get('attempted',[])+entry['status'].get('completed',[]))
                 for entry in origins[old['cell_id']]),'superseded execution was attempted')
        old.update(scope_status='superseded_declaration',required_execution=False,
                   replacement_cell_id=new['cell_id'])
        if not new['measurement_valid']:
            new.update(scope_status='required_replacement_pending',required_execution=True)
    return points


def annotate_grid(grid,selection,originals,points,protocol):
    prior.annotate_grid(grid,selection,originals,points,protocol)
    reference=selection.get('a_terminal_scope')
    if reference is None:return grid
    status=checked(reference)
    need(status['model']=='14b' and status['complete'] and status['phase']=='complete'
         and not status['failed'] and status['node_lease_held'] is False,'A exclusion authority not terminal')
    skipped=set(status['skipped_saturated'])
    for row in grid:
        actual=[point for point in points if point['original_cell_id']==row['original_cell_id']]
        if row['model']!='14b' or row['verified_executions'] or not actual:continue
        if not all(point['cell_id'] in skipped for point in actual):continue
        losses=[point for point in points if point['cell_id'] in status['completed']
                and point['model']=='14b' and point['dataset']==row['dataset']
                and point['measurement_valid'] and point['work_complete'] and point['slo_attainment']<.9
                and point['rate_rps']==status['saturated_at'][row['dataset']]]
        need(losses and row['rate_rps']>losses[0]['rate_rps'],'A exclusion lacks complete first-loss evidence')
        row.update(status='excluded_above_first_complete_loss',required_executions=0,
                   repeat_requirement_complete=True,scope_reference=reference,
                   scope_statuses=['not_required_above_first_loss'])
    return grid
