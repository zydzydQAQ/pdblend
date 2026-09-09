"""Audit fresh new-rate baseline observations against their declarations."""
from pathlib import Path
import source_identity_v2 as source_identity

def reference(cp,key):
    value=cp[key]
    return value if isinstance(value,dict) else dict(path=value,sha256=cp[key+'_sha256'])

def declaration_chain(p,cp,row,logical_ref):
    execution_ref=cp.get('declaration')
    if execution_ref is None or execution_ref==logical_ref:return logical_ref
    p.need(isinstance(execution_ref,dict),'unrecognized baseline execution declaration')
    instruction=p.checked(execution_ref)
    p.need(instruction.get('schema')=='B-baseline-original-repeats-capacity-reconciliation-v1',
           'unknown baseline continuation declaration')
    p.need(instruction['parent_declaration']==logical_ref,'baseline continuation changed parent declaration')
    for path,digest in instruction['source_files'].items():p.need(p.sha(path)==digest,'continuation source changed')
    matches=[r for r in instruction['remaining_cells'] if r['cell_id']==row['cell_id']]
    p.need(matches==[row] and cp['row']==row,'continuation changed an original scientific row')
    observed=[]
    for stage in instruction['prior_stages']:
        status=p.checked(stage['status'])
        p.need(status['attempted']==status['completed'] and status.get('finished_s')
               and status.get('node_lease_held') is False,'previous stage not terminal')
        observed.extend(status['attempted'])
    p.need(len(observed)==len(set(observed))==instruction['already_observed_count'],
           'previous observation accounting differs')
    p.need(row['cell_id'] not in observed,'previously observed baseline replayed')
    return execution_ref

def actual_identity(p,cp,binding,receipt_path):
    directory=Path(receipt_path).parent
    records=[]
    for name in ('identity.before.json','identity.after.json'):
        path=directory/name
        p.need(cp['artifacts'].get(str(path))==p.sha(path),'actual identity missing from checkpoint')
        records.append(p.read(path))
    before,after=records
    a={i['provenance']['instance_id']:i for i in before}
    z={i['provenance']['instance_id']:i for i in after}
    p.need(len(a)==len(before)==len(z)==len(after)==len(binding['instances'])
           and set(a)==set(z)=={i['id'] for i in binding['instances']},'incomplete actual process identity')
    actual={}
    for instance in binding['instances']:
        x,y=a[instance['id']],z[instance['id']]
        p.need(x['provenance']==y['provenance']
               and all(x['provenance'].get(k)==v for k,v in instance['provenance'].items()),
               'actual engine provenance differs')
        p.need(x['container']['Id']==y['container']['Id']==instance['container']['id']
            and x['container']['Image']==y['container']['Image']==instance['container']['image']
            and x['container']['State']['StartedAt']==y['container']['State']['StartedAt']==instance['container']['StartedAt']
            and type(x['container']['State']['Pid']) is int and x['container']['State']['Pid']>0
            and x['container']['State']['Pid']==y['container']['State']['Pid'],
            'actual engine process changed or mismatches binding')
        actual[instance['id']]=dict(container_id=instance['container']['id'],
            started_at=instance['container']['StartedAt'],host_pid=x['container']['State']['Pid'])
    return actual

def inspect(audit,row,checkpoint,declaration_ref,continuation_refs=()):
    p=audit.p
    trace_ref=dict(path=row['trace_path'],sha256=row['trace_sha256'])
    trace=p.checked(trace_ref)
    metadata=dict(row,n_expected=len(trace['requests']),
        expected_generated_tokens=sum(r['output_len'] for r in trace['requests']))
    point={k:metadata[k] for k in p.PAIR_FIELDS}
    point.update(cell_id=row['cell_id'],system=row['system'],repeat=row['repeat'],
        phase='main',slo_scale=row['slo_scale'],new_rate=True,measurement_valid=False,
        status='unmeasured',work_complete=None,error=None,declaration=declaration_ref)
    point.update({k:None for k in audit.METRICS})
    if checkpoint is None:return point
    try:
        cp=p.read(checkpoint);executed=cp['row']
        required=[reference(cp,k)['path'] for k in ('receipt','binding')]+list(cp['artifacts'])
        missing=[path for path in required if not Path(path).is_file()]
        if missing:
            point.update(status='awaiting_mirror',missing_artifact_count=len(missing),
                error='terminal checkpoint seen; referenced evidence has not fully arrived')
            return point
        authorized_ids={row['cell_id']}
        if '/C/boundary-p4/' in declaration_ref['path']:
            authorized_ids.add(row['cell_id'].replace('parallel-rate-p4-explore-','parallel-rate-p4-boundary-'))
        p.need(executed['cell_id'] in authorized_ids,'baseline executed cell differs')
        p.need(executed['n_requests']==row['n_requests']==len(trace['requests']),
            'baseline declared request count differs')
        point.update(declared_cell_id=row['cell_id'],cell_id=executed['cell_id'])
        for key in (*p.PAIR_FIELDS,'system','repeat','slo_scale','arrival_window_s'):
            actual=len(trace['requests']) if key=='n_expected' else executed[key]
            expected=metadata[key]
            p.need(actual==expected,'baseline executed declaration differs: '+key)
        p.need(executed['trace_path']==row['trace_path'],'baseline trace path differs')
        execution_declaration=declaration_chain(p,cp,row,declaration_ref)
        receipt_ref=reference(cp,'receipt');binding_ref=reference(cp,'binding')
        receipt=p.checked(receipt_ref);binding=p.checked(binding_ref)
        # Both current runners freeze this trace and the complete five-system declaration.
        p.need(binding['files'].get(row['trace_path'])==row['trace_sha256'],'baseline trace not frozen in binding')
        p.need(binding['files'].get(declaration_ref['path'])==declaration_ref['sha256'],'baseline declaration not frozen in binding')
        p.need(binding['files'].get(execution_declaration['path'])==execution_declaration['sha256'],
               'baseline execution continuation not frozen in binding')
        verified_continuations=[]
        for instruction_ref in continuation_refs:
            if instruction_ref['path'] not in binding['files']:continue
            p.need(binding['files'][instruction_ref['path']]==instruction_ref['sha256'],
                   'baseline reconciliation reference changed')
            instruction=p.checked(instruction_ref)
            if instruction.get('schema')=='B-baseline-original-repeats-capacity-reconciliation-v1':
                declaration_chain(p,dict(cp,declaration=instruction_ref),row,declaration_ref)
                verified_continuations.append(instruction_ref)
        continuation=binding.get('continuation_instruction')
        if continuation is not None:
            instruction=p.checked(continuation)
            p.need(binding['files'].get(continuation['path'])==continuation['sha256'],'continuation not frozen in binding')
            p.need(instruction['logical_declaration']==declaration_ref,'continuation changed scientific declaration')
            p.need(executed==row and row['cell_id'] in instruction['remaining_cell_ids'],'continuation changed or selected unapproved row')
            p.need(row['cell_id'] not in {v['cell_id'] for v in instruction['executed_checkpoints']},'continuation repeats previously observed cell')
            for previous in instruction['executed_checkpoints']:p.checked(previous)
        for path,digest in cp['artifacts'].items():p.need(p.sha(path)==digest,'baseline raw artifact changed: '+path)
        actual_engine_identity=actual_identity(p,cp,binding,receipt_ref['path'])
        version=source_identity.identity(binding,row['dataset'])
        summary=receipt['summary'];directory=Path(receipt_ref['path']).parents[2]/'cells'/executed['cell_id']
        p.need(p.read(directory/'summary.json')==summary,'baseline summary and receipt disagree')
        p.need(receipt['measurement_valid'] is True and summary['measurement_valid'] is True
            and summary['fixed_window_valid'] is True and summary['gpu_count']==8
            and summary['power_source_verified'] is True and receipt['clock_restore_complete'] is True
            and receipt['child_stopped'] is True and not receipt['outer_cleanup_errors'],
            'baseline measurement or cleanup invalid')
        p.need(all(v.get('complete') is True for v in receipt['restoration'].values()),'baseline native cleanup incomplete')
        p.need(summary['trace_sha256']==row['trace_sha256'],'baseline summary trace differs')
        p.need(summary['comparison_system']==row['system'],'baseline serving system differs')
        proof=audit.audit_raw(summary,directory,dict(trace=trace_ref,original_point=metadata))
        additional=audit.raw_metrics.audit_additional_metrics(summary,directory)
        proof['additional_metrics']=additional
        point.update({k:summary.get(k) for k in audit.METRICS if k!='completion_fraction'})
        point.update(additional['normalized_metrics'])
        point.update({k:summary[k] for k in ('completed_work_requests','generated_tokens',
            'expected_generated_tokens','n_expected','good_requests','measurement_duration_s','work_complete')})
        point.update(version)
        point['actual_engine_identity']=actual_engine_identity
        point['verified_continuations']=verified_continuations
        point.update(completion_fraction=summary['completed_work_requests']/summary['n_expected'],
            measurement_valid=True,status='measured_complete' if summary['work_complete'] else 'measured_incomplete',
            verification=proof,checkpoint=p.ref(checkpoint),receipt=receipt_ref,binding=binding_ref,
            implementation_id=binding['host_release'],energy_measured_gpu_count=8,
            completed_work_throughput_rps=summary['completed_work_requests']/summary['measurement_duration_s'],
            generated_token_throughput_tps=summary['generated_tokens']/summary['measurement_duration_s'])
    except (ValueError,KeyError,TypeError,OSError) as exc:
        point.update(status='observed_invalid',measurement_valid=False,error=str(exc))
    return point

def collect(audit,root,pdb_points,declaration_refs,continuation_refs=()):
    p=audit.p
    # C v1 was stopped before any baseline measurement; retain it as superseded.
    # V2 uses the already-frozen cooperative Dynamo historical continuation.
    declarations=[]
    for reference in declaration_refs:
        p.checked(reference);declarations.append(Path(reference['path']))
    result=[];sources={};seen={}
    # Lookup by exact workload and repeat, never by a guessed filename prefix.
    demand={p.pair_identity(x) for x in pdb_points if x.get('measurement_valid')}
    checkpoints={}
    for node in ('A','B','C'):
        for path in (root/node).rglob('checkpoints/*.json'):
            cp=p.read(path);row=cp.get('row',{})
            if row.get('system') not in p.BASELINES:continue
            cid=row['cell_id']
            if cid in checkpoints and checkpoints[cid]!=path:
                raise ValueError('duplicate baseline checkpoint ID '+cid)
            checkpoints[cid]=path
    for path in declarations:
        if not path.exists():continue
        declaration=p.ref(path);sources[str(path)]=declaration['sha256']
        for row in p.read(path)['cells']:
            if row['system'] not in p.BASELINES or row.get('existing_original_rate'):continue
            cid=row['cell_id']
            if cid in seen:
                p.need(row==seen[cid],'conflicting baseline declaration '+cid);continue
            seen[cid]=row
            metadata=dict(row,n_expected=row['n_requests'])
            cp=checkpoints.get(cid)
            if cp is None and '/C/boundary-p4/' in str(path):
                cp=checkpoints.get(cid.replace('parallel-rate-p4-explore-','parallel-rate-p4-boundary-'))
            selected=p.pair_identity(metadata) in demand
            # Keep conditional higher rates in the declaration, not as false gaps.
            if not selected and cp is None:continue
            point=inspect(audit,row,cp,declaration,continuation_refs)
            result.append(point)
            if cp is not None:sources[str(cp)]=p.sha(cp)
    return result,sources
