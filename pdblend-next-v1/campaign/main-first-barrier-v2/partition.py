"""Per-cell source partitions for the explicitly reviewed cooperative queue repair."""
import copy
import importlib.util
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
C=HERE.parent

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v1=load('main_partition_v1',C/'main-first-barrier-v1/barrier.py')
require,read,sha=v1.require,v1.read,v1.sha
OLD_HOST=C.parent/'releases/five-system100-C7B-baseline-v1-runtime'
NEW_HOST=C.parent/'releases/five-system100-C7B-baseline-cooperative-v1-runtime'
OLD_MANIFEST='04ce259ea204299fe3c3a0bcc33a9e2a904d5b2d5a4a94a0974f4b911a3cf597'
NEW_MANIFEST='4adceea7e702f4191196be124c5891ba578eeb940a5ea0b8ae09d3669322a5c8'
ADMISSION='src/ecopadg/serving/admission.py'
OLD_QUEUE='2d59808aad020f05c4d22103a494f773b2d7cc1afef29d6146429807af121a31'
NEW_QUEUE='1f43ebdfedb8d7f379b05af63d60864a5bf524dbb6d8139592edc30b229c6252'


def ref(reader,value):
    require(isinstance(value,dict) and Path(value.get('path','')).is_absolute() and v1.valid_sha(value.get('sha256')),
        'future/unfixed binding reference is not ready')
    return reader.read(value['path'],value['sha256'])


def partition_contract(groups,rows,*,require_bound=True):
    expected={r['cell_id']:r for r in rows};seen=set();ids=set()
    require(len(expected)==len(rows) and rows,'nonempty exact original main domain required')
    for g in groups:
        require(g['id'] not in ids,'duplicate partition id');ids.add(g['id'])
        cells=g.get('cell_ids');require(isinstance(cells,list) and cells and len(set(cells))==len(cells),'empty/duplicate partition cell IDs')
        require(set(cells)<=set(expected) and not seen.intersection(cells),'foreign or overlapping partition cells')
        seen.update(cells)
        require(g['system'] in v1.SYSTEMS and set(g['datasets'])=={expected[c]['dataset'] for c in cells}
            and all(expected[c]['system']==g['system'] and expected[c]['phase']=='main' and expected[c]['slo_scale']==1 for c in cells),
            'partition must name its precise original main cells/system/datasets')
        require(g.get('policy_reference') and v1.valid_sha(g['policy_reference'].get('sha256')),'original policy reference missing')
        if require_bound:
            require(g.get('binding') and v1.valid_sha(g['binding'].get('sha256')),'future partition binding is null/unready')
    require(seen==set(expected),'partition union does not equal every declared main cell')
    return dict(cell_count=len(seen),partitions=len(groups),ready=all(bool(g.get('binding')) for g in groups))


def host_contract(old,current,reader):
    a,b=Path(old['host_release']),Path(current['host_release'])
    am=reader.read(str(a/'manifest.json'));bm=reader.read(str(b/'manifest.json'))
    ah,bh=reader.digest(str(a/'manifest.json')),reader.digest(str(b/'manifest.json'))
    require(set(am['files'])==set(bm['files']),'host files added/removed outside reviewed repair')
    changed=[p for p,h in am['files'].items() if bm['files'][p]!=h]
    if a==b:
        require(ah==bh and not changed,'same host source changed');kind='identical'
    else:
        require(old['model']==current['model']=='7b' and a==OLD_HOST and b==NEW_HOST
            and ah==OLD_MANIFEST and bh==NEW_MANIFEST and changed==[ADMISSION]
            and am['files'][ADMISSION]==OLD_QUEUE and bm['files'][ADMISSION]==NEW_QUEUE,
            'only the exact reviewed C cooperative admission yield host is allowed')
        require(bm.get('parent_release')==str(a) and bm.get('parent_manifest_sha256')==ah,'cooperative host parent differs')
        kind='cooperative-admission-yield-v1'
    for root,manifest,binding in ((a,am,old),(b,bm,current)):
        for name,h in manifest['files'].items():
            path=str(root/name)
            require(binding['files'].get(path)==h,'actual binding does not freeze its complete host source')
            reader.digest(path,h)
    return dict(kind=kind,old_host=str(a),host=str(b),old_manifest_sha256=ah,manifest_sha256=bh,changed_files=changed)


def compatibility(g,reader):
    old,current=ref(reader,g['policy_reference']),ref(reader,g['binding'])
    for key in ('model','system','hostname','protocol_id','deadline_s','instances','large_inputs'):
        require(old.get(key)==current.get(key),'per-cell host repair changed '+key)
    require(current['model'] in v1.MODELS and current['hostname']==v1.HOSTS[current['model']]
        and current['system']==g['system'] and current['protocol_id']==v1.PROTOCOL and current['deadline_s']==v1.DEADLINE,
        'binding model/protocol/deadline differs')
    host=host_contract(old,current,reader);configs={}
    for ds in g['datasets']:
        x,y=old['configs'][ds],current['configs'][ds]
        a=reader.read(x,old['files'][x]);b=reader.read(y,current['files'][y])
        aa,bb=copy.deepcopy(a),copy.deepcopy(b)
        # Only run_cell's journal destination and explicitly bound host metadata.
        aa.pop('journal',None);bb.pop('journal',None)
        if host['kind']!='identical':
            require(aa.get('controller_source_release')==old['host_release'] and bb.get('controller_source_release')==current['host_release'],
                'actual metadata must identify the exact old/new host')
            aa.pop('controller_source_release');bb.pop('controller_source_release')
        require(aa==bb,'algorithm/profile/assignment/prior/budget/SLO/hardtimeout/topology configuration changed')
        def external(value):
            if isinstance(value,dict):
                for x in value.values():external(x)
            elif isinstance(value,list):
                for x in value:external(x)
            elif isinstance(value,str) and value in old['files']:
                require(current['files'].get(value)==old['files'][value],'same configuration path has different profile/external input bytes')
                reader.digest(value,old['files'][value])
        external(aa)
        normalized=copy.deepcopy(aa);normalized.pop('controller_source_release',None)
        configs[ds]=dict(original=x,original_sha256=old['files'][x],actual=y,actual_sha256=current['files'][y],
            policy_normalized_sha256=v1.digest_json(normalized))
    return dict(host=host,configs=configs,scope='source repair provenance, not identical output/SLO/energy or measured benefit'),current


def execution_source(g,binding,reader,parent_path,parent_sha):
    original=reader.read(parent_path,parent_sha)
    require(parent_sha==v1.SOURCE_SHA[binding['model']] and binding['files'].get(parent_path)==parent_sha,'original workload manifest not bound')
    source=g.get('execution_manifest') or {'path':parent_path,'sha256':parent_sha}
    require(binding['host_release']!=str(NEW_HOST) or source['sha256']!=parent_sha,
        'reviewed repair must execute only its exact new-cell subset, never a whole-group rerun')
    manifest=ref(reader,source)
    require(binding['files'].get(source['path'])==source['sha256'],'actual execution manifest not bound')
    if source['sha256']!=parent_sha:
        require(manifest.get('parent_manifest')==parent_path and manifest.get('parent_manifest_sha256')==parent_sha
            and manifest.get('model')==binding['model'] and manifest.get('protocol_id')==v1.PROTOCOL,
            'subset must identify the original frozen source')
        expected={r['cell_id']:r for r in original['cells']};rows=manifest['cells']
        require(len(rows)==len(g['cell_ids']) and {r['cell_id'] for r in rows}==set(g['cell_ids'])
            and all(r==expected[r['cell_id']] for r in rows),'subset changed, omitted or added original partition rows')
    else:require(manifest==original,'full execution source differs')
    return dict(reference=source,parent_manifest=parent_path,parent_manifest_sha256=parent_sha,
        kind='exact-original-row-subset' if source['sha256']!=parent_sha else 'original-full-manifest',
        original_cell_ids=list(g['cell_ids']))


def retained_failure(reader,value):
    proof=ref(reader,value)
    require(proof.get('failed_measurement_retained') is True and proof.get('failed_main_cell_counted_complete') is False
        and proof.get('failed_supervisor_and_runner_stopped') is True and proof.get('failed_measurement_child_stopped') is True,
        'failed attempt must stay explicitly invalid, terminal and preserved')
    files=proof['retained_files'];require(files,'retained failure raw absent')
    for p,h in files.items():reader.digest(p,h)
    matches=[p for p in files if p.endswith('/operations/'+proof['failed_cell_id']+'/receipt.json')]
    require(len(matches)==1,'exact failed receipt missing');rp=matches[0];receipt=reader.read(rp,files[rp])
    require(receipt.get('measurement_valid') is False and receipt.get('child_stopped') is True
        and receipt.get('clock_restore_complete') is True and receipt.get('finished_s')
        and receipt.get('restoration') and all(r.get('complete') is True for r in receipt['restoration'].values()),
        'failed measurement or terminal cleanup falsely relabelled')
    require(receipt['cell_id']==proof['failed_cell_id'],'wrong failed work ID')
    cp=str(Path(rp).parents[2]/'checkpoints'/(proof['failed_cell_id']+'.json'))
    require(not reader.local(cp).exists(),'invalid original attempt cannot have a completed checkpoint')
    require(receipt.get('full_operation_energy_j')==proof.get('full_operation_energy_j')
        and type(proof.get('full_operation_energy_j')) in (int,float) and math.isfinite(proof['full_operation_energy_j'])
        and proof['full_operation_energy_j']>0,'failed-operation energy removed or changed')
    for p,h in proof['valid_prior_checkpoints'].items():reader.digest(p,h)
    return dict(reference=value,failed_cell_id=proof['failed_cell_id'],invalid_output=str(Path(rp).parents[2]),
        raw_files=files,successful_original_checkpoints=proof['valid_prior_checkpoints'],
        primary_observed_energy_j=proof['primary_observed_energy_j'],full_operation_energy_j=proof['full_operation_energy_j'],
        energy_windows_overlap_not_added=True,counts_as_completed=False)


def execution(row,g,binding,reader,failures,*,supplied=None):
    output=Path(binding['output']);cp=reader.read(str(output/'checkpoints'/(row['cell_id']+'.json')))
    receipt=reader.read(cp['receipt'],cp['receipt_sha256'])
    if supplied:paths=[supplied['invocation']]
    else:paths=[str(p) for p in sorted(reader.local(output/'invocations').glob('*.json'))]
    matches=[]
    for path in paths:
        inv=reader.read(path)
        if row['cell_id'] not in inv.get('completed',[]):continue
        source=g['execution_source']
        require(inv.get('binding_sha256')==g['binding']['sha256'] and inv.get('manifest_sha256')==source['reference']['sha256']
            and inv.get('protocol_id')==v1.PROTOCOL and inv.get('system')==row['system'] and inv.get('phase')=='main'
            and inv.get('finished_s') and inv['started_s']<=receipt['started_s']<=receipt['finished_s']<=cp['completed_s']<=inv['finished_s']<=v1.DEADLINE,
            'actual per-cell invocation/binding/source/time differs')
        require(row['dataset'] in inv.get('selected_datasets',v1.DATASETS),'cell outside invocation dataset group')
        digest=reader.digest(path)
        succeeded=inv.get('complete') is True and not inv.get('error')
        if not succeeded:
            retained=[f for f in failures if f['raw_files'].get(path)==digest
                and f['successful_original_checkpoints'].get(str(output/'checkpoints'/(row['cell_id']+'.json')))==reader.digest(str(output/'checkpoints'/(row['cell_id']+'.json')))
                and f['failed_cell_id']==inv.get('current_cell') and f['failed_cell_id']!=row['cell_id']]
            require(inv.get('error') and len(retained)==1,'failed invocation needs its exact preserved failure and successful original CP')
        matches.append(dict(invocation=path,invocation_sha256=digest,invocation_succeeded=bool(succeeded),
            invocation_finished_s=inv['finished_s'],pid=inv['pid'],host_release=binding['host_release'],
            host_manifest_sha256=reader.digest(str(Path(binding['host_release'])/'manifest.json')),
            execution_manifest=source['reference']))
    require(len(matches)==1,'CP must identify exactly one executing invocation; skip is not execution')
    result=matches[0]
    before=reader.read(str(output/'operations'/row['cell_id']/'identity.before.json'))
    after=reader.read(str(output/'operations'/row['cell_id']/'identity.after.json'))
    bi={x['provenance']['instance_id']:x for x in before};ai={x['provenance']['instance_id']:x for x in after}
    require(len(before)==len(after)==len(binding['instances']) and set(bi)==set(ai)=={i['id'] for i in binding['instances']},'missing actual per-cell engine identity')
    actual={}
    for i in binding['instances']:
        a,z=bi[i['id']],ai[i['id']]
        require(a['provenance']==z['provenance'] and all(a['provenance'].get(k)==v for k,v in i['provenance'].items()),'actual engine model/source differs')
        for k in ('Id','Image','Name','Args'):require(a['container'][k]==z['container'][k],'engine identity changed')
        for k in ('StartedAt','Pid'):require(a['container']['State'][k]==z['container']['State'][k],'engine process changed')
        require(a['container']['Id']==i['container']['id'] and a['container']['Image']==i['container']['image']
            and a['container']['State']['StartedAt']==i['container']['StartedAt'] and a['container']['State']['Pid']>0,'actual engine does not match bound process')
        actual[i['id']]=dict(container_id=a['container']['Id'],image=a['container']['Image'],StartedAt=a['container']['State']['StartedAt'],host_pid=a['container']['State']['Pid'],provenance=a['provenance'])
    result['actual_engine_identity']=actual
    if supplied:require(result==supplied,'compact per-cell execution/source differs from actual raw')
    return result


def record(row,g,binding,reader,failures):
    out=Path(binding['output']);cp_path=str(out/'checkpoints'/(row['cell_id']+'.json'));cp=reader.read(cp_path)
    receipt=reader.read(cp['receipt'],cp['receipt_sha256']);summary=receipt['summary'];config=binding['configs'][row['dataset']]
    rec=dict(row=row,checkpoint=cp_path,checkpoint_sha256=reader.digest(cp_path),output=str(out),
        binding=g['binding']['path'],binding_sha256=g['binding']['sha256'],partition_id=g['id'],
        config=config,config_sha256=binding['files'][config],measurement_valid=True,work_complete=summary['work_complete'],
        n_expected=row['n_requests'],good_requests=summary['good_requests'],completed_work_requests=summary['completed_work_requests'],
        energy_j=summary['energy_j'],implementation_variant=summary['implementation_variant'],receipt_sha256=cp['receipt_sha256'],
        actual_config_sha256=cp['artifacts'][str(out/'cells'/row['cell_id']/'runtime_config.json')],finished_s=receipt['finished_s'],child_pid=receipt['child_pid'])
    v1.verify_record(rec,reader)
    rec['execution']=execution(row,g,binding,reader,failures)
    return rec
