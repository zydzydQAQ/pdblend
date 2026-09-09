"""CPU policy/reference contracts. Never contacts a running engine."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parent

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

barrier=load('scale_main_release',CAMPAIGN/'main-first-barrier-v1/barrier.py')
require,read,sha=barrier.require,barrier.read,barrier.sha
PROTOCOL,DEADLINE=barrier.PROTOCOL,barrier.DEADLINE
EXECUTOR=CAMPAIGN/'five-system-execution-v3/run.py'
EXECUTOR_SHA='7c7dbe217243b42a8f93b57476ed457a6111e8f90c71ac4269130c9b46420f92'
DRIVER=HERE/'scale_driver.py'
_parts=CAMPAIGN/'main-first-barrier-v2/partition.py'
_release=CAMPAIGN/'main-first-barrier-v2/barrier.py'
require(sha(_parts)=='36aa52bfc07fc14e25dd48cac3f317b1bcd2966a2da2642c1c5a268ac07f7747'
    and sha(_release)=='f097fe250049113c3b31733bbfb6ed530a03e5042c12b5060306042189e894d4',
    'frozen per-cell verifier changed')
_previous_partition=sys.modules.get('partition')
try:
    sys.modules['partition']=load('scale_partition_v2',_parts)
    released=load('scale_main_release_v2',_release)
finally:
    if _previous_partition is None:sys.modules.pop('partition',None)
    else:sys.modules['partition']=_previous_partition
_model_release=CAMPAIGN/'B32B-qualified-main-release-v1/release.py'
require(sha(_model_release)=='39e408a369c9e7eec9e60c8978524b9a31013d8f76230f3d690baabdc79b0a22','frozen B qualified model release changed')
released=load('scale_B32B_qualified_model_release_v1',_model_release)
revision=released.p


def qualified_gate(binding,datasets):
    """Reconstruct each required mechanism from original HTTP/tokens/events.

    B Eco delegates the explicitly versioned native-trajectory qualification.
    Other systems retain the original ordinary/PD reader and raw exact flags.
    """
    require(binding.get('model')=='32b','this adapter only permits B32B')
    if binding.get('system')=='ecoserve':
        require(datasets and set(datasets)<=set(binding.get('configs',{})),
            'qualified Eco scale dataset configuration missing')
        return released.audit_eco_binding(binding)
    gate=Path(binding['correctness_evidence'])
    reader=load('scale_legacy_gate',CAMPAIGN/'AC-baseline-binding-v2/gate_evidence.py')
    power=load('scale_original_power',HERE/'power_evidence.frozen.py')
    original=reader.files(gate)
    require(original and all(binding['files'].get(p)==h for p,h in original.items()),'gate raw is not entirely SHA-bound')
    strategies={read(binding['configs'][ds])['strategy'] for ds in datasets}
    require(len(strategies)==1,'different algorithms in one physical scale group')
    strategy=next(iter(strategies))
    canonical='dynamollm' if strategy=='dynamollm-resident' else strategy
    require(canonical==binding['system'],'gate strategy/canonical system differs')
    hetero=len({i['tp'] for i in binding['instances']})>1
    require(not hetero or (binding['model']=='14b' and strategy=='distserve' and datasets==['longbench']),
        'heterogeneous oracle only permits original A DistServe LongBench')
    actual,observed=reader.audit(gate,binding['instances'],strategy,power.power_evidence,hetero=hetero)
    first={x['provenance']['instance_id']:x['provenance'] for x in read(gate/'identity.before.json')}
    last={x['provenance']['instance_id']:x['provenance'] for x in read(gate/'identity.after.json')}
    require(first==last,'full imported source/model provenance changed during the fresh gate')
    declared=binding.get('mechanism_proof',{})
    require(declared.get('required')==actual['required'] and declared.get('verified')==actual['verified'],
        'binding mechanism qualification differs from original raw reconstruction')
    require(observed==original,'gate changed during mechanism review')
    return actual

def reference(ref):
    require(isinstance(ref,dict) and ref.get('path') and barrier.valid_sha(ref.get('sha256')),'missing fixed file reference')
    require(sha(ref['path'])==ref['sha256'],'input SHA differs: '+ref['path'])
    return read(ref['path'])

def policy_equal(original,current):
    """Only journal is overwritten by frozen run_cell before construction.

    Dynamo's original template/runtime paths are deliberately NOT normalized.
    A new binding must reference the exact original policy after requalification.
    """
    left=copy.deepcopy(original);right=copy.deepcopy(current)
    left.pop('journal',None);right.pop('journal',None)
    require(left==right,'scale policy differs from original main (including topology/source/budget/roles/prior)')
    return True

def core_instance(instance):
    value=copy.deepcopy(instance)
    value['container'].pop('StartedAt',None)
    value.get('provenance',{}).pop('pid',None) # Container PID may legitimately remain 1 after restart.
    return value

def identity_index(path,binding):
    require(binding['files'].get(str(path))==sha(path),'inspection identity is not in fresh binding SHA set')
    rows=read(path);require(isinstance(rows,list),'full Docker inventory required')
    indexed={x['Id']:x for x in rows}
    ids={i['container']['id'] for i in binding['instances']}
    require(ids<=set(indexed),'binding instance missing from actual inspection')
    for i in binding['instances']:
        x=indexed[i['container']['id']]
        require(x['Name'].lstrip('/')==i['container']['name'] and x['Image']==i['container']['image']
            and x['State']['StartedAt']==i['container']['StartedAt'] and x['State']['Running'] is True
            and type(x['State']['Pid']) is int and x['State']['Pid']>0,'actual inspection does not match live binding identity')
    return indexed

def compatible_bindings(main_ref,scale_ref,datasets,identity_mode):
    require(identity_mode in ('same_process','restarted'),'measurement needs an explicit deployment identity mode')
    main,current=reference(main_ref),reference(scale_ref)
    require(main.get('model')==current.get('model')=='32b','this adapter only permits B32B')
    for key in ('protocol_id','deadline_s','model','system','hostname'):
        require(main[key]==current[key],'scale changed bound '+key)
    require(current['protocol_id']==PROTOCOL and current['deadline_s']==DEADLINE,'wrong scale protocol/deadline')
    require(main.get('large_inputs')==current.get('large_inputs'),'frozen model/large-input identities changed')
    require(set(datasets)<=set(main['configs']) and set(datasets)<=set(current['configs']),'physical group configuration missing')
    source_policy(main,current,datasets)
    if main['output']!=current['output']:
        require(main['model']=='7b' and main['system']=='dynamollm' and current['host_release']==str(revision.NEW_HOST),
            'new output is allowed only for the explicit cooperative C Dynamo continuation')
    original={i['id']:i for i in main['instances']};future={i['id']:i for i in current['instances']}
    require(set(original)==set(future),'physical IDs cannot change implicitly for this continuation')
    for rid in original:require(core_instance(original[rid])==core_instance(future[rid]),'model/TP/GPU/ports/engine source/capability changed')
    before=identity_index(Path(main_ref['path']).parent/'identity.json',main)
    after=identity_index(Path(scale_ref['path']).parent/'identity.json',current)
    for rid,old in original.items():
        new=future[rid];cid=old['container']['id']
        same=(old['container']['StartedAt']==new['container']['StartedAt'] and before[cid]['State']['Pid']==after[cid]['State']['Pid'])
        if identity_mode=='same_process':
            require(same and old.get('provenance')==new.get('provenance'),'same-process claim changed host PID/StartedAt/provenance')
        else:
            require(old['container']['StartedAt']!=new['container']['StartedAt'] and before[cid]['State']['Pid']!=after[cid]['State']['Pid'],
                'restart requires actual fresh StartedAt AND host State.Pid; namespace PID=1 is insufficient')
    if identity_mode=='same_process' and current['system']=='ecoserve':
        qualified_gate(current,datasets)
    if identity_mode=='restarted':
        require(main_ref!=scale_ref and current.get('output_correctness_verified') is True
            and not current.get('correctness_gate_required_before_performance') and current.get('mechanism_proof'),
            'restart requires a newly qualified performance binding')
        gate=Path(current['correctness_evidence'])
        require(str(gate)!=main.get('correctness_evidence'),'old process correctness gate cannot qualify restarted engines')
        qualified_gate(current,datasets)
        gate_after=read(gate/'identity.after.json')
        observed={x['container']['Id']:x for x in gate_after}
        for i in current['instances']:
            cid=i['container']['id'];x=observed[cid]
            require(x['container']['State']['StartedAt']==i['container']['StartedAt']
                and x['container']['State']['Pid']==after[cid]['State']['Pid']
                and all(x['provenance'].get(k)==v for k,v in i['provenance'].items()),
                'fresh gate qualified another process/source')
    return dict(main=main,current=current,identity_mode=identity_mode,
        live_identity_before_controls_required=True,
        scope='CPU/static source and measured gate compatibility; frozen executor still rechecks actual engine before every cell')

def executed_binding(row,refs,source_sha):
    """An original terminal invocation identifies who actually executed a CP.

    A later invocation which merely skipped the CP cannot claim it: the cell
    must occur in its newly-completed list, with matching original binding SHA.
    """
    values=[(ref,reference(ref)) for ref in refs]
    require(len({b['output'] for _,b in values})==1,'continuation changed original output')
    out=Path(values[0][1]['output']);cid=row['cell_id']
    cp=read(out/'checkpoints'/(cid+'.json'));receipt=read(cp['receipt']);matches=[]
    for path in sorted((out/'invocations').glob('*.json')):
        v=read(path)
        if cid not in v.get('completed',[]):continue
        candidates=[(ref,b) for ref,b in values if ref['sha256']==v.get('binding_sha256')]
        require(len(candidates)==1,'CP was executed under an unknown/ambiguous binding SHA')
        require(v.get('phase')=='scale' and v.get('system')==row['system'] and v.get('protocol_id')==PROTOCOL
            and v.get('manifest_sha256')==source_sha and v.get('complete') is True and not v.get('error')
            and v.get('finished_s') is not None,'CP has no clean terminal scale invocation')
        require(v['started_s']<=receipt['started_s']<=receipt['finished_s']<=cp['completed_s']<=v['finished_s']<=DEADLINE,
            'CP measurement is outside its bound invocation/deadline')
        if 'selected_datasets' in v:require(row['dataset'] in v['selected_datasets'],'CP outside actual invocation group')
        ref,b=candidates[0];matches.append((ref,b,path,sha(path)))
    require(len(matches)==1,'CP must have exactly one original terminal executing invocation')
    return matches[0]

def point_process(row,binding,cp,e):
    op=Path(binding['output'])/'operations'/row['cell_id']
    receipt=e.read(cp['receipt'],cp['receipt_sha256'])
    ids={i['id'] for i in binding['instances']}
    before=e.read(op/'identity.before.json');after=e.read(op/'identity.after.json')
    index=lambda rows:{x['provenance']['instance_id']:x for x in rows}
    bi,ai=index(before),index(after)
    require(len(before)==len(after)==len(ids) and set(bi)==set(ai)==ids,'actual full process identity missing')
    require(set(receipt['restoration'])==ids,'actual native instance set differs')
    processes={}
    for i in binding['instances']:
        a,b=bi[i['id']],ai[i['id']]
        for key in ('Id','Image','Name','Args'):
            require(a['container'][key]==b['container'][key],'CP container execution identity changed')
        for key in ('StartedAt','Pid'):
            require(a['container']['State'][key]==b['container']['State'][key],'CP process changed during measurement')
        require(type(a['container']['State']['Pid']) is int and a['container']['State']['Pid']>0
            and a['container']['State']['Running'] is True and b['container']['State']['Running'] is True,'actual live host PID missing')
        require(a['container']['Id']==i['container']['id'] and a['container']['Image']==i['container']['image']
            and a['container']['Name'].lstrip('/')==i['container']['name']
            and a['container']['State']['StartedAt']==i['container']['StartedAt']
            and a['provenance']==b['provenance'] and all(a['provenance'].get(k)==v for k,v in i['provenance'].items()),
            'CP ran on another bound process/model/imported source')
        rr=receipt['restoration'][i['id']];require(rr.get('complete') is True and not rr.get('errors'),'CP actual restoration failed')
        barrier.native(rr['before'],rr['proof'],i)
        resumed=rr['resumed'];control=resumed['control'];r=resumed['after']
        require(r.get('id')==i['id'] and r.get('generation')==r.get('acknowledged_generation')==control['generation']==resumed['before']['generation']+1
            and r.get('accepting') is True and all(r.get(k)==0 for k in ('active','running','waiting')) and not r.get('kv_allocations')
            and not r.get('error') and not r.get('runtime_error') and all(r.get(k)==control[k] for k in ('role','mode','admit_prefill','admit_decode')),
            'CP final native resume was not actually ACKed/idle')
        if i['native_kind']=='v3':require(r.get('scheduler_budget_pending') is None,'CP restore budget still pending')
        if i.get('restore_budget_tokens') is not None:
            require(r.get('scheduler_budget_effective',{}).get('max_num_batched_tokens')==i['restore_budget_tokens'],'CP restored wrong budget')
        if i.get('scheduler_cache_observed'):
            cache=[x.get('controls',{}).get('runtime') for x in r.get('scheduler_io',[])]
            require(len(cache)==i.get('scheduler_cache_count',1) and all(x and x.get('generation')==r['generation'] and not x.get('error') for x in cache),'CP cache ACK missing')
        processes[i['id']]=dict(container_id=a['container']['Id'],StartedAt=a['container']['State']['StartedAt'],host_pid=a['container']['State']['Pid'],provenance=a['provenance'])
    return processes

def verify_existing(row,refs,source_sha,required_binding=None):
    # Deduplicate an unchanged main==scale binding reference.
    refs=list({(r['path'],r['sha256']):r for r in refs}.values())
    ref,binding,invocation,invocation_sha=executed_binding(row,refs,source_sha)
    if required_binding:require(ref==required_binding,'new CP did not execute with the current fresh binding')
    f=barrier.facts();e=f.Evidence();b=copy.deepcopy(binding);b['_sha256']=ref['sha256']
    point=f.point(row,b,e)
    require(point.get('checkpoint_verified') and point.get('metrics_verified'),'existing CP invalid; never overwrite: '+str(point.get('error')))
    cp=e.read(point['checkpoint_path']);processes=point_process(row,binding,cp,e)
    # Baseline binders additionally pin the host PID in their complete inventory.
    inventory=Path(ref['path']).parent/'identity.json'
    if str(inventory) in binding['files']:
        bound=identity_index(inventory,binding)
        for p in processes.values():require(bound[p['container_id']]['State']['Pid']==p['host_pid'],'CP host PID differs from actual bound inventory')
    for path,digest in e.files.items():require(sha(path)==digest,'existing CP changed during scale continuation check')
    require(sha(invocation)==invocation_sha and sha(ref['path'])==ref['sha256'],'CP execution binding/invocation changed during verification')
    return dict(cell_id=row['cell_id'],checkpoint=point['checkpoint_path'],checkpoint_sha256=sha(point['checkpoint_path']),
        work_complete=point['work_complete'],good_requests=point['good_requests'],energy_j=point['energy_j'],reused=True,
        executed_binding_path=ref['path'],executed_binding_sha256=ref['sha256'],executing_invocation=str(invocation),
        executing_invocation_sha256=invocation_sha,actual_processes=processes)

def source_policy(original,current,datasets):
    """Exact policy plus only the explicitly frozen generic cooperative yield."""
    for bound in (original,current):
        manifest=str(Path(bound['host_release'])/'manifest.json')
        require(bound['files'].get(manifest)==sha(manifest),'original/current complete host manifest is not SHA-bound')
    host_changed=original['host_release']!=current['host_release']
    if host_changed:
        require(original['model']==current['model']=='7b' and original['system']==current['system']=='dynamollm',
            'this continuation only enables the reviewed C Dynamo host repair')
    revision.host_contract(original,current,barrier.Reader())
    for ds in datasets:
        a,b=original['configs'][ds],current['configs'][ds]
        require(sha(a)==original['files'][a] and sha(b)==current['files'][b],'main/scale actual configuration SHA changed')
        left,right=read(a),read(b)
        if host_changed:
            require(left.get('controller_source_release')==original['host_release'] and right.get('controller_source_release')==current['host_release'],
                'source metadata does not identify exact reviewed host transition')
            left.pop('controller_source_release');right.pop('controller_source_release')
        policy_equal(left,right)
        def external(value):
            if isinstance(value,dict):
                for k,v in value.items():
                    if k!='journal':external(v)
            elif isinstance(value,list):
                for v in value:external(v)
            elif isinstance(value,str) and value in original['files']:
                require(current['files'].get(value)==original['files'][value],'original profile/external bytes changed at same path')
                require(sha(value)==original['files'][value],'original profile/external input changed')
        external(left)
    return True


def released_main_records(group,model_proof,manifest):
    ds=set(group['datasets']);system=group['system']
    records=[copy.deepcopy(r) for r in model_proof['records'] if r['row']['system']==system and r['row']['dataset'] in ds]
    require(len(records)==10*len(ds),'released exact main domain incomplete')
    byid={r['row']['cell_id']:r for r in records};require(len(byid)==len(records),'duplicate released main row')
    scales=[r for r in manifest['cells'] if r['phase']=='scale' and r['system']==system and r['dataset'] in ds]
    require(len(scales)==6*len(ds),'wrong original scale domain')
    evidence=barrier.Reader();partitions={g['id']:g for g in model_proof.get('groups',[]) if 'id' in g}
    failures=[]
    if model_proof['schema']==2:
        failures=[revision.retained_failure(evidence,f['reference']) for f in model_proof['retained_failures']]
    for r in records:
        barrier.verify_record(r,evidence)
        b=evidence.read(r['binding'],r['binding_sha256'])
        if model_proof['schema']==2:
            g=partitions[r['partition_id']]
            revision.execution(r['row'],g,b,evidence,failures,supplied=r['execution'])
        else:
            g=dict(binding=dict(path=r['binding'],sha256=r['binding_sha256']),execution_source=dict(
                reference=dict(path=model_proof['source_manifest'],sha256=model_proof['source_sha256'])))
            r['execution']=revision.execution(r['row'],g,b,evidence,[])
    for row in scales:
        r=byid.get(row['reuse_main_cell_id']);require(r is not None,'scale references foreign main ID')
        require(all(row[k]==r['row'][k] for k in ('model','system','dataset','rate_rps','seed','trace_sha256','n_requests','content_pairing_sha256')),
            'scale trace/work/model/system/rate changed from actual referenced main')
        require(r['row']['phase']=='main' and r['row']['slo_scale']==1.,'scale cannot reference another scale point')
        require(row['slo_scale'] in (.5,2.) and row['slo_ttft_s']==r['row']['slo_ttft_s']*row['slo_scale']
            and row['slo_tpot_s']==r['row']['slo_tpot_s']*row['slo_scale'],'scale SLO does not match its original main reference')
    evidence.stable()
    return records,scales


def scale_source(current,full_source,records,*,repaired):
    require(current.get('execution_manifest',full_source)==full_source,
        'scale current binding execution_manifest must identify the actual full original source, not a historical main subset')
    if repaired:
        require(current.get('execution_manifest')==full_source,'repaired C scale requires an explicit full-source execution manifest')
        histories={tuple(sorted(r['execution']['execution_manifest'].items())) for r in records}
        require({tuple(sorted(x.items())) for x in current.get('main_execution_manifests',[])}==histories,
            'C scale must retain exact old/full and new/subset main execution sources as history')
        for path_sha in histories:
            ref=dict(path_sha);require(current['files'].get(ref['path'])==ref['sha256'] and sha(ref['path'])==ref['sha256'],
                'historical main execution source not preserved in new scale binding')


def check_group(group,model_proof,manifest):
    require(group['datasets'] and len(set(group['datasets']))==len(group['datasets']),'invalid dataset selection')
    records,rows=released_main_records(group,model_proof,manifest)
    policy_ref=group['policy_reference'];original=reference(policy_ref)
    require(original['system']==group['system'] and original['model']==model_proof['model'],'wrong original online policy')
    refs=list({(r['binding'],r['binding_sha256']):dict(path=r['binding'],sha256=r['binding_sha256']) for r in records}.values())
    require(policy_ref in refs,'original policy must be one of the actual released producers')
    for r in records:
        binding=reference(dict(path=r['binding'],sha256=r['binding_sha256']))
        source_policy(original,binding,[r['row']['dataset']])
    mode=group['identity_mode'];current=None
    require(group['system']!='pdblend' or mode=='reuse_only','original three PDB scale sets are reuse-only')
    if mode!='reuse_only':
        current=compatible_bindings(policy_ref,group['scale_binding'],group['datasets'],mode)['current']
        full_source=dict(path=model_proof['source_manifest'],sha256=model_proof['source_sha256'])
        refs.append(group['scale_binding'])
        repaired=any(reference(x)['host_release']==str(revision.NEW_HOST) for x in refs)
        if repaired:
            require(group['system']=='dynamollm' and group['datasets']==['alpaca','sharegpt','longbench']
                and current['host_release']==str(revision.NEW_HOST),'all18 C Dynamo scale points must use one declared repaired host')
        scale_source(current,full_source,records,repaired=repaired)
        # Original v3 skip behavior is kept: except the new C Dynamo namespace,
        # existing scale CPs must remain under this same system output.
    refs=list({(x['path'],x['sha256']):x for x in refs}.values())
    outputs={reference(x)['output'] for x in refs};reused=[];pending=[]
    for row in rows:
        candidates=[out for out in outputs if (Path(out)/'checkpoints'/(row['cell_id']+'.json')).exists()]
        require(len(candidates)<=1,'duplicate scale attempts in different producer outputs must be reviewed, never chosen by energy')
        if candidates:
            related=[x for x in refs if reference(x)['output']==candidates[0]]
            reused.append(verify_existing(row,related,model_proof['source_sha256']))
            if current is not None:require(candidates[0]==current['output'],'existing scale CP must be skipped in its original shared output, not copied')
        else:
            for out in outputs:
                require(not (Path(out)/'operations'/row['cell_id']).exists() and not (Path(out)/'cells'/row['cell_id']).exists(),
                    'uncheckpointed scale attempt retained; no automatic retry/overwrite')
            pending.append(row)
    require(mode!='reuse_only' or not pending,'reuse-only missing scale CPs cannot trigger new PDB work')
    return dict(group=group,main=original,current=current,rows=rows,reused=reused,pending=pending,main_records=records,producer_bindings=refs)


def check_spec(spec,release_path,release_sha,selected=None):
    require(spec.get('model')=='32b','this adapter only permits B32B')
    checked_release=released.verify_release(release_path,release_sha,expected_protocol_id=PROTOCOL,expected_deadline_s=DEADLINE,expected_model=spec.get('model'))
    release=read(release_path)
    require(spec['schema']==2 and spec['protocol_id']==PROTOCOL and spec['deadline_s']==DEADLINE,'scale spec protocol/deadline differs')
    model=spec['model'];proof=release['models'][model]
    require(spec['hostname']==proof['hostname'] and spec['source']['sha256']==proof['source_sha256'],'wrong model/source/host')
    manifest=reference(spec['source']);barrier.source_rows(manifest,model)
    require(sha(EXECUTOR)==EXECUTOR_SHA,'original v3 source changed')
    groups=spec['groups'];require(len({g['id'] for g in groups})==len(groups),'duplicate scale group ID')
    selected=set(selected or [g['id'] for g in groups]);require(selected and selected<={g['id'] for g in groups},'unknown scale groups')
    chosen=[g for g in groups if g['id'] in selected];coverage=set()
    for g in chosen:
        for ds in g['datasets']:
            key=(g['system'],ds);require(key not in coverage,'overlapping selected system/dataset groups');coverage.add(key)
    checked=[check_group(g,proof,manifest) for g in chosen]
    return dict(release=checked_release,model=model,hostname=proof['hostname'],groups=checked,
        selected_scale_cells=sum(len(g['rows']) for g in checked),reused=sum(len(g['reused']) for g in checked),
        pending=sum(len(g['pending']) for g in checked))
