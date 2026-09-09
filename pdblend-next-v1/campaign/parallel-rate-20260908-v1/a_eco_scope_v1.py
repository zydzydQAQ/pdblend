"""Apply A's actual first-loss scope while retaining all 31 declared rows."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'A/eco-drain31-execution-v1/manifest.json'
PACKAGE_SHA='df24027b0e4806e898d16caa2d90021c027389eeefbea4b3fd35e7825f5f4c2c'
CONTRACT=PACKAGE.parent/'contract.py'
CONTRACT_SHA='e23f738d8eaa74407d1109a22693753945a8f8e209c778cf4fa4f943bb72a4f4'


def contract(p):
    package=p.checked(dict(path=str(PACKAGE),sha256=PACKAGE_SHA))
    p.need(p.sha(CONTRACT)==CONTRACT_SHA and package['files'][str(CONTRACT)]==CONTRACT_SHA,
           'A Eco scope contract differs from reviewed execution source')
    for path,digest in package['files'].items():
        p.need(p.sha(path)==digest,'A Eco execution package changed')
    spec=importlib.util.spec_from_file_location('selected_A_eco_scope_contract',CONTRACT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,package


def apply(p,selection,overlay):
    reference=selection.get('a_eco_execution')
    p.need(reference is not None,'A Eco needs its actual scoped/fresh-qualified execution release')
    release=p.checked(reference)
    p.need(release['schema']=='A-Eco-scoped-qualified-execution-release-v1'
           and release['model']=='14b' and release['source_package']==dict(path=str(PACKAGE),sha256=PACKAGE_SHA),
           'A Eco execution release uses another source contract')
    sources=overlay['sources'];sources[reference['path']]=reference['sha256']
    for path,digest in release['files'].items():
        p.need(p.sha(path)==digest,'A Eco execution dependency changed');sources[path]=digest
    module,package=contract(p);sources.update(package['files']);sources[str(PACKAGE)]=PACKAGE_SHA
    scope_ref=release['declaration'];scope=p.checked(scope_ref)
    p.need(scope['schema']=='A-Eco-final-PDB-execution-scope-v1' and scope['approved'] is True,
           'A Eco scope is not an approved final execution scope')
    frozen_binding=p.checked(release['binding']);qualified_binding=p.checked(release['qualification_binding'])
    p.need(release['host_release']==str(module.HOST)
           and release['host_manifest']==dict(path=str(module.HOST/'manifest.json'),sha256=module.HOST_SHA)
           and frozen_binding['host_release']==qualified_binding['host_release']==str(module.HOST)
           and release['logical_declaration']==scope['logical_declaration'],
           'A appended and original Eco rates must use the same frozen logical group and host source')
    p.need(scope['final_pdb_release']==selection['models']['14b']['formal_release'],
           'A Eco scope compares another PDB source group')
    for ref in (scope['final_pdb_release'],scope['final_pdb_terminal_audit']):
        p.need(qualified_binding['files'].get(ref['path'])==ref['sha256'],
               'A actual fresh qualification must freeze the same final PDB predecessor')
    logical=p.checked(scope['logical_declaration'])
    terminal=p.checked(scope['final_pdb_terminal_audit'])
    p.need(terminal['schema']=='A-final-pdb-terminal-for-baseline-restore-v1'
           and terminal['release']==scope['final_pdb_release'], 'A Eco terminal provenance differs')
    observations=[]
    for stage in terminal['stages']:
        for cid in stage['expected_cell_ids']:
            path=Path(stage['checkpoint_root'])/(cid+'.json');cp=p.read(path)
            receipt_ref=module.cp_reference(cp,'receipt');receipt=p.checked(receipt_ref)
            row=cp['row'];summary=receipt['summary']
            p.need(row['cell_id']==cid and summary['work_complete'] is True
                   and summary['failed_requests']==summary['request_timeouts']==0,
                   'A Eco cannot derive a first-loss boundary from incomplete PDB work')
            cp_ref=dict(path=str(path),sha256=p.sha(path))
            observations.append(dict(row=row,checkpoint=cp_ref,slo_attainment=summary['slo_attainment']))
            sources[str(path)]=cp_ref['sha256'];sources[receipt_ref['path']]=receipt_ref['sha256']
    p.need(len({x['row']['cell_id'] for x in observations})==len(observations),
           'A terminal scope repeats a measured PDB point')
    appends={ref['path']:p.checked(ref) for ref in scope.get('append_declarations',[])}
    rows,limits=module.select_rows(logical,scope,observations,appends)
    excluded={row['cell_id']:row for row in scope['excluded_cells']}
    for cid in {row['cell_id'] for row in logical['cells']}:
        item=overlay['fresh'][cid]
        p.need(item['declaration']==scope['logical_declaration'],'A logical31 was replaced by another group')
        item.update(execution_scope=scope_ref,execution_release=reference,
                    execution_scope_pending_final_pdb=False,excluded_above_first_loss=excluded.get(cid))
    original_ids={row['cell_id'] for row in logical['cells']}
    for row in rows:
        cid=row['cell_id']
        if cid in original_ids:continue
        p.need(cid not in overlay['fresh'],'A appended Eco observation duplicated another source group')
        overlay['fresh'][cid]=dict(row=row,declaration=scope_ref,host_release=release['host_release'],
            host_manifest=release['host_manifest'],existing_original_rate=False,
            reuse_original_rate_first_repeat=False,execution_scope=scope_ref,execution_release=reference,
            execution_scope_pending_final_pdb=False,excluded_above_first_loss=None)
    for ref in (scope_ref,scope['logical_declaration'],scope['final_pdb_terminal_audit'],
                scope['final_pdb_release'],release['qualification_binding'],*scope.get('append_declarations',[])):
        p.checked(ref);sources[ref['path']]=ref['sha256']
    overlay['a_scope']=dict(scope=scope,reference=scope_ref,observations=observations,
        actual_first_loss=limits,release=release,release_reference=reference,
        required_cell_ids=[row['cell_id'] for row in rows],excluded=excluded)
    return overlay


def verify_against_measured(p,overlay,points):
    """Use the collector's independently checked raw PDB results, without live PID checks."""
    item=overlay['a_scope'];by_id={point['cell_id']:point for point in points
                                 if point['model']=='14b' and point.get('measurement_valid')}
    p.need(set(by_id)=={x['row']['cell_id'] for x in item['observations']},
           'A Eco terminal scope omitted or added actual final PDB observations')
    measured=[]
    for record in item['observations']:
        point=by_id[record['row']['cell_id']]
        p.need(point['work_complete'] and point['checkpoint']==record['checkpoint']
               and abs(point['slo_attainment']-record['slo_attainment'])<1e-12,
               'A Eco boundary differs from independently verified work or SLO')
        measured.append(dict(record,slo_attainment=point['slo_attainment']))
    module,_=contract(p)
    p.need(module.boundaries(measured)==item['actual_first_loss'],
           'A Eco exclusions differ from the actual full-work PDB first-loss boundary')
    return dict(verified=True,required=len(item['required_cell_ids']),excluded=len(item['excluded']),
                scope=item['reference'],actual_first_loss=item['actual_first_loss'])
