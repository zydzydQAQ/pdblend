"""Independent raw audit for newly declared rates without an old PDB row."""
from pathlib import Path

def inspect(audit,cell,checkpoint):
    p=audit.p;source=cell['source_row'];trace=p.checked(cell['trace'])
    metadata=dict(source,n_expected=len(trace['requests']),
        expected_generated_tokens=sum(r['output_len'] for r in trace['requests']))
    point={k:metadata[k] for k in p.PAIR_FIELDS}
    point.update(cell_id=cell['cell_id'],arm=cell['arm'],repeat=cell['repeat'],stage=cell['stage'],
        original_cell_id=cell['original_cell_id'],new_rate=True,exploration_only=True,
        measurement_valid=False,status='unmeasured',work_complete=None,error=None)
    point.update({k:None for k in audit.METRICS})
    if checkpoint is None:return point
    try:
        cp=p.read(checkpoint)
        p.need(cp['declaration']==cell,'new-rate declaration changed')
        p.need(cp['row']['cell_id']==cell['cell_id'] and cp['row']['trace_sha256']==cell['trace']['sha256'],
               'executed workload differs')
        receipt=p.checked(dict(path=cp['receipt'],sha256=cp['receipt_sha256']))
        binding=p.checked(dict(path=cp['binding'],sha256=cp['binding_sha256']))
        p.need(binding['improvement']['arm']==cell['arm'] and binding['improvement']['repeat']==cell['repeat'],
               'executed arm or repeat differs')
        for path,digest in cp['artifacts'].items():p.need(p.sha(path)==digest,'raw artifact changed: '+path)
        summary=receipt['summary'];directory=Path(cp['receipt']).parents[2]/'cells'/cell['cell_id']
        p.need(p.read(directory/'summary.json')==summary,'summary and receipt disagree')
        p.need(receipt['measurement_valid'] is True and summary['measurement_valid'] is True
            and summary['fixed_window_valid'] is True and summary['gpu_count']==8
            and summary['power_source_verified'] is True and receipt['clock_restore_complete'] is True
            and receipt['child_stopped'] is True and not receipt['outer_cleanup_errors'],
            'measurement or cleanup invalid')
        p.need(all(v.get('complete') is True for v in receipt['restoration'].values()),'native cleanup incomplete')
        p.need(summary['trace_sha256']==cell['trace']['sha256'],'summary trace changed')
        # Supply declared SLO metadata to the unchanged raw arithmetic audit;
        # never synthesize an old observation or a baseline result.
        proof=audit.audit_raw(summary,directory,dict(cell,original_point=metadata))
        additional=audit.raw_metrics.audit_additional_metrics(summary,directory)
        proof['additional_metrics']=additional
        point.update({k:summary.get(k) for k in audit.METRICS if k!='completion_fraction'})
        point.update(additional['normalized_metrics'])
        point.update({k:summary[k] for k in ('completed_work_requests','generated_tokens',
            'expected_generated_tokens','n_expected','good_requests','measurement_duration_s','work_complete')})
        point.update(completion_fraction=summary['completed_work_requests']/summary['n_expected'],
            measurement_valid=True,status='measured_complete' if summary['work_complete'] else 'measured_incomplete',
            verification=proof,checkpoint=p.ref(checkpoint),receipt=p.ref(cp['receipt']),
            implementation_id=binding['improvement']['implementation'])
    except (ValueError,KeyError,TypeError,OSError) as exc:
        point.update(status='observed_invalid',measurement_valid=False,error=str(exc))
    return point
