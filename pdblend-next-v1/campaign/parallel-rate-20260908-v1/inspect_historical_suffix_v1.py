"""Audit only the eleven predeclared original EcoServe32B SLO-scale suffix rows."""
from pathlib import Path
import source_identity_v2 as source_identity
import final_selected_baseline_v1 as final_baseline

def reference(cp,key):
    value=cp[key]
    return value if isinstance(value,dict) else dict(path=value,sha256=cp[key+'_sha256'])

def inspect(audit,row,checkpoint,declaration_ref):
    p=audit.p
    trace_ref=dict(path=row['trace_path'],sha256=row['trace_sha256'])
    trace=p.checked(trace_ref)
    metadata=dict(row,n_expected=len(trace['requests']),
        expected_generated_tokens=sum(r['output_len'] for r in trace['requests']))
    point={k:metadata[k] for k in p.PAIR_FIELDS}
    point.update(cell_id=row['cell_id'],system=row['system'],repeat=1,
        phase='scale',slo_scale=row['slo_scale'],new_rate=False,historical_scale_suffix=True,measurement_valid=False,
        status='unmeasured',work_complete=None,error=None,declaration=declaration_ref)
    point.update({k:None for k in audit.METRICS})
    if checkpoint is None:return point
    try:
        cp=p.read(checkpoint);executed=cp['row']
        p.need(row['phase']=='scale' and row['model']=='32b' and row['system']=='ecoserve' and row['slo_scale'] in (.5,2), 'not original EcoServe32B scale suffix')
        p.need(executed==row and cp.get('historical_scale_suffix') is True, 'historical row changed or suffix marker absent')
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
        for key in (*p.PAIR_FIELDS,'system','phase','slo_scale','arrival_window_s'):
            actual=len(trace['requests']) if key=='n_expected' else executed[key]
            expected=metadata[key]
            p.need(actual==expected,'baseline executed declaration differs: '+key)
        p.need(executed['trace_path']==row['trace_path'],'baseline trace path differs')
        if isinstance(cp.get('declaration'),dict):p.need(cp['declaration']==declaration_ref,'baseline declaration reference differs')
        receipt_ref=reference(cp,'receipt');binding_ref=reference(cp,'binding')
        receipt=p.checked(receipt_ref);binding=p.checked(binding_ref)
        actual_identity=final_baseline.actual_identity(p,cp,binding,receipt_ref['path'])
        # Suffix binding retains the exact original trace and immutable suffix declaration.
        p.need(binding['files'].get(row['trace_path'])==row['trace_sha256'],'baseline trace not frozen in binding')
        p.need(binding['files'].get(declaration_ref['path'])==declaration_ref['sha256'],'baseline declaration not frozen in binding')
        for path,digest in cp['artifacts'].items():p.need(p.sha(path)==digest,'baseline raw artifact changed: '+path)
        version=source_identity.identity(binding,row['dataset'])
        declaration=p.checked(declaration_ref)
        p.need(row in declaration['cells'], 'suffix row not explicitly declared')
        old_binding=p.checked(declaration['original_binding'])
        expected_version=source_identity.identity(old_binding,row['dataset'])
        for key in ('controller_source_sha256','profile_sha256','policy_sha256'):
            p.need(version[key]==expected_version[key], 'historical source/profile/policy changed: '+key)
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
        point['actual_engine_identity']=actual_identity
        point.update(completion_fraction=summary['completed_work_requests']/summary['n_expected'],
            measurement_valid=True,status='measured_complete' if summary['work_complete'] else 'measured_incomplete',
            verification=proof,checkpoint=p.ref(checkpoint),receipt=receipt_ref,binding=binding_ref,
            implementation_id=binding['host_release'],energy_measured_gpu_count=8,
            completed_work_throughput_rps=summary['completed_work_requests']/summary['measurement_duration_s'],
            generated_token_throughput_tps=summary['generated_tokens']/summary['measurement_duration_s'])
    except (ValueError,KeyError,TypeError,OSError) as exc:
        point.update(status='observed_invalid',measurement_valid=False,error=str(exc))
    return point
