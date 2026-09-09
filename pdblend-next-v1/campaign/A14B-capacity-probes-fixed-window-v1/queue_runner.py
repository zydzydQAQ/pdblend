"""Shared bounded PDB fixed-window queue; hardware behavior belongs to each Cell.

The caller holds the existing per-node lease. No old request-count/span gate is
used. Completed evidence is immutable; failed attempts require a new package.
"""
import asyncio
import itertools
import json
import math
from pathlib import Path
import time

PROTOCOL='per-dataset-slo-fixed-window-v2'
PHASES=('probe','main','scale')


def require(ok,reason):
    if not ok:raise ValueError(reason)


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def validate_spec(b,spec):
    require(spec.get('protocol_id')==PROTOCOL and spec.get('measurement_schema')==3
        and spec.get('split')=='development' and spec.get('execute_baselines') is False
        and spec.get('formal_eligible') is False,'wrong fixed-window development scope')
    require(spec.get('model') in ('7b','14b','32b'),'model identity required')
    ref=spec['deadline_scope'];require(b.sha(ref['path'])==ref['sha256'],'deadline scope changed')
    deadline=b.read(ref['path'])
    require(deadline['arrival_window_s']==300 and deadline['request_hard_timeout_s']==120
        and deadline['drain_after_arrival_window_s']==120 and deadline['arrival_seeds']==[701,1701]
        and deadline['rates_per_model_dataset']==10 and deadline['minimum_requests'] is None,
        'old workload/deadline protocol cannot enter the new queue')
    require(deadline['phase_hours']==dict(optimization_and_capacity=3,main_matrix=8,slo_scale=6,targeted_reruns_and_reports=7)
        and finite(deadline['created_s']) and finite(deadline['deadline_s'])
        and abs(deadline['deadline_s']-deadline['created_s']-86400)<.001,'invalid global phase allocation')
    budget=spec.get('execution_budget',{})
    require(all(finite(budget.get(key)) and budget[key]>0 for key in ('startup_allowance_s','restore_allowance_s')),
        'explicit finite startup and restoration allowances required')
    rows=spec['cells'];require(len({r['cell_id'] for r in rows})==len(rows),'duplicate cell id')
    for row in rows:
        require(row.get('phase') in PHASES and row.get('model')==spec['model']
            and row.get('arrival_seed') in (701,1701) and row.get('dataset') in deadline['datasets']
            and row.get('trace_duration_s')==300 and type(row.get('n_requests')) is int and row['n_requests']>0,
            'invalid variable-N fixed-window row')
        scale=row.get('slo_scale');base=deadline['dataset_slo_s'][row['dataset']]
        require(scale in (.5,1.,2.) and row['slo_ttft_s']==base['ttft']*scale
            and row['slo_tpot_s']==base['tpot']*scale
            and row.get('slo_attainment_target',row.get('slo',{}).get('attainment_target'))==.9,
            'row SLO differs from actual user scale')
        require(scale==1 if row['phase'] in ('probe','main') else scale in (.5,2.),'wrong phase SLO scale')
    for phase in PHASES:
        selected=[r for r in rows if r['phase']==phase]
        if selected:
            require(len(selected)<=10 if phase=='probe' else len(selected)==(60 if phase=='main' else 36),
                'phase declaration count differs')
            require([r['sequence'] for r in selected]==list(range(1,len(selected)+1)),
                'phase sequence must be contiguous and frozen')
    require(len({r.get('strategy') for r in rows})==1 and all(str(r.get('strategy')).startswith('pdblend') for r in rows),
        'one PDB strategy is required for the whole model')
    configs={r.get('controller_config') for r in rows if r.get('controller_config')}
    require(len(configs)<=1,'dataset-specific controller policies are forbidden')
    main=phase_rows(spec,'main');scale=phase_rows(spec,'scale')
    if main:
        for dataset in deadline['datasets']:
            group=[r for r in main if r['dataset']==dataset];rates={r['rate_rps'] for r in group}
            require(len(rates)==10 and all(finite(r) and r>0 for r in rates)
                and {(r['rate_rps'],r['arrival_seed']) for r in group}==set(itertools.product(rates,(701,1701))),
                'main must cover exactly ten rates and both seeds for every dataset')
    if scale:
        require(bool(main),'scale1 references must belong to the same declared main matrix')
        indexed={r['cell_id']:r for r in main}
        for dataset in deadline['datasets']:
            group=[r for r in scale if r['dataset']==dataset];rates={r['rate_rps'] for r in group}
            require(len(rates)==3 and len(group)==12
                and {(r['rate_rps'],r['arrival_seed'],r['slo_scale']) for r in group}
                    ==set(itertools.product(rates,(701,1701),(.5,2.))),
                'scale must cover three rates, both seeds and both additional scales')
        for row in scale:
            ref=indexed.get(row.get('reuse_main_cell_id'))
            require(ref and all(ref[k]==row[k] for k in ('dataset','rate_rps','arrival_seed','trace_sha256','n_requests')),
                'SLO scale must reuse the exact corresponding scale1 workload')
    return deadline


def phase_record(b,spec,phase,*,now=None,create=False):
    require(phase in PHASES,'unknown phase')
    deadline=validate_spec(b,spec);now=time.time() if now is None else now
    require(finite(now) and now>=deadline['created_s']-1.,'wall clock predates the declared experiment')
    path=Path(spec['deadline_scope']['path']).parent/'phase-ledgers'/spec['model']/(phase+'.json')
    if path.exists():
        record=b.read(path)
        require(record.get('schema')==1 and record.get('phase')==phase and record.get('model')==spec['model']
            and record.get('deadline_scope_sha256')==spec['deadline_scope']['sha256']
            and finite(record.get('started_s')) and record['started_s']<=now+1.,'phase clock identity changed/regressed')
        start=record['started_s']
    else:
        start=now;record=dict(schema=1,phase=phase,model=spec['model'],started_s=start,
            deadline_scope_sha256=spec['deadline_scope']['sha256'])
    maximum={'probe':deadline['created_s']+3*3600,
             'main':deadline['deadline_s']-13*3600,'scale':deadline['deadline_s']-7*3600}[phase]
    duration={'probe':3,'main':8,'scale':6}[phase]*3600
    end=min(start+duration,maximum)
    if 'deadline_s' in record:require(record['deadline_s']==end,'phase deadline changed')
    record['deadline_s']=end
    if create and not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as handle:json.dump(record,handle,indent=2,allow_nan=False);handle.write('\n')
    return record


def cell_limits(spec,phase_record,*,now=None):
    """Return None if another complete window/drain/startup/restore cannot fit."""
    now=time.time() if now is None else now;budget=spec['execution_budget']
    startup=budget['startup_allowance_s'];restore=budget['restore_allowance_s'];end=phase_record['deadline_s']
    require(finite(now) and now>=phase_record['started_s']-1.,'wall clock regressed during phase')
    if now+startup+300+120+restore>end:return None
    arrival=min(now+startup,end-restore-420)
    return dict(schema=1,phase=phase_record['phase'],phase_started_s=phase_record['started_s'],
        phase_deadline_s=end,issued_s=now,startup_allowance_s=startup,restore_allowance_s=restore,
        arrival_window_s=300.,drain_allowance_s=120.,latest_arrival_epoch_s=arrival,
        cell_execution_deadline_s=arrival+420.,restore_deadline_s=arrival+420.+restore)


def artifacts(root,cid):
    root=Path(root);paths=[]
    for folder in ('cells','operations'):
        path=root/folder/cid;require(path.is_dir(),'completed artifact directory missing: '+str(path))
        paths.extend(p for p in path.rglob('*') if p.is_file())
    for folder in ('receipts','identities'):
        path=root/folder/(cid+'.json');require(path.is_file(),'completed artifact missing: '+str(path));paths.append(path)
    return sorted(str(p.relative_to(root)) for p in paths)


def valid_receipt(receipt,row,limits):
    require(receipt.get('screen_valid') is True and receipt.get('outer_cleanup',{}).get('complete') is True,
        'failed execution/cleanup is retained and cannot be checkpointed as complete')
    s=receipt.get('summary',{});w=s.get('fixed_window',{})
    require(s.get('measurement_valid') is True and s.get('measurement_schema')==3
        and s.get('measurement_window_protocol')==PROTOCOL and s.get('fixed_window_valid') is True,
        'old or invalid measurement window')
    require(s.get('n_expected')==row['n_requests'] and s.get('trace_sha256')==row['trace_sha256']
        and s.get('slo_scale')==row['slo_scale'] and w.get('arrival_window_s')==300
        and w.get('effective_slo_s')==dict(ttft=row['slo_ttft_s'],tpot=row['slo_tpot_s']),
        'receipt workload/SLO differs from the frozen row')
    require(s.get('post_measurement_cleanup',{}).get('cleanup_complete') is True,
        'actual controller cleanup proof missing')
    require(finite(w.get('arrival_epoch_s')) and w['arrival_epoch_s']<=limits['latest_arrival_epoch_s']
        and finite(s.get('measurement_end_s')) and s['measurement_end_s']<=limits['cell_execution_deadline_s'],
        'actual dispatch epoch/measurement exceeded the admitted time budget')
    require(finite(receipt.get('finished_s')) and receipt['finished_s']<=limits['restore_deadline_s'],
        'restoration finished beyond the reserved boundary')
    # Attainment and work failures remain observations. There is deliberately no
    # q>=.9 or work_complete gate selecting only favorable cells.
    return s


def phase_rows(spec,phase):return [r for r in spec['cells'] if r['phase']==phase]


def package_path(b):return Path(getattr(b,'PACKAGE_MANIFEST',b.ROOT/'package-manifest.json'))


def inspect_queue(b,phase,*,check_pending=True):
    spec=b.read(b.ROOT/'runspec.json');validate_spec(b,spec);rows=phase_rows(spec,phase)
    require(rows,'phase not declared; freeze real source/rates in a new package first')
    root=b.ROOT;directory=root/'checkpoints'/phase
    paths=sorted(directory.glob('*.json')) if directory.exists() else []
    require(len(paths)<=len(rows),'too many phase checkpoints')
    records=[];previous=None;package=b.sha(package_path(b))
    for index,path in enumerate(paths):
        row=rows[index];cid=row['cell_id'];record=b.read(path)
        require(path.name==f'{index+1:04d}-{cid}.json' and record.get('schema')==2
            and record.get('phase')==phase and record.get('sequence')==index+1
            and record.get('cell_id')==cid,'checkpoint phase/order changed')
        require(record.get('package_manifest_sha256')==package
            and record.get('previous_checkpoint_sha256')==previous
            and record.get('source_trace_sha256')==row['trace_sha256'],'checkpoint source/chain changed')
        expected=artifacts(root,cid)
        require(set(record.get('artifacts',{}))==set(expected),'checkpoint artifact set changed')
        for rel,digest in record['artifacts'].items():require(b.sha(root/rel)==digest,'checkpoint artifact changed: '+rel)
        summary=valid_receipt(b.read(root/'receipts'/(cid+'.json')),row,record['limits'])
        clock=phase_record(b,spec,phase)
        require(record['limits']==cell_limits(spec,clock,now=record['limits']['issued_s']),
            'checkpoint admission deadline changed')
        if phase=='scale':require(record.get('scale_one_reference')==main_reference(b,row),'scale1 reference changed')
        require(summary==b.read(root/'cells'/cid/'summary.json'),'receipt and actual summary differ')
        require(b.sha(row['trace'])==row['trace_sha256'],'source trace changed')
        records.append(record);previous=b.sha(path)
    if check_pending:
        for row in rows[len(records):]:
            cid=row['cell_id']
            require(not any((root/folder/cid).exists() for folder in ('cells','operations'))
                and not any((root/folder/(cid+'.json')).exists() for folder in ('receipts','identities')),
                'uncheckpointed attempt retained; use a new declared package for a targeted retry: '+cid)
    return dict(completed=len(records),remaining=len(rows)-len(records),records=records,
        previous_checkpoint_sha256=previous,phase_execution_complete=len(records)==len(rows))


def main_reference(b,row):
    if row['phase']!='scale':return None
    progress=inspect_queue(b,'main',check_pending=False)
    found=[r for r in progress['records'] if r['cell_id']==row['reuse_main_cell_id']]
    require(len(found)==1,'scale1 main reference not completed; do not manufacture or rerun it')
    record=found[0]
    require(record['source_trace_sha256']==row['trace_sha256'],'scale changed the main workload')
    path=b.ROOT/'checkpoints/main'/f"{record['sequence']:04d}-{record['cell_id']}.json"
    return dict(cell_id=record['cell_id'],checkpoint_path=str(path),checkpoint_sha256=b.sha(path))


def checkpoint(b,row,limits,previous,reference):
    cid=row['cell_id'];summary=valid_receipt(b.read(b.ROOT/'receipts'/(cid+'.json')),row,limits)
    require(summary==b.read(b.ROOT/'cells'/cid/'summary.json'),'receipt and summary differ at checkpoint')
    require(time.time()<=limits['phase_deadline_s'],'cell evidence arrived after the phase deadline')
    values={rel:b.sha(b.ROOT/rel) for rel in artifacts(b.ROOT,cid)}
    path=b.ROOT/'checkpoints'/row['phase']/f"{row['sequence']:04d}-{cid}.json"
    record=dict(schema=2,phase=row['phase'],sequence=row['sequence'],cell_id=cid,completed_s=time.time(),
        package_manifest_sha256=b.sha(package_path(b)),source_trace_sha256=row['trace_sha256'],
        previous_checkpoint_sha256=previous,artifacts=values,limits=limits,scale_one_reference=reference,
        measurement_valid=True,work_complete=summary.get('work_complete'),
        slo_attainment=summary.get('slo_attainment'),good_requests=summary.get('good_requests'),
        energy_j=summary.get('energy_j'),no_formal_baseline_inference=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:json.dump(record,handle,indent=2,allow_nan=False);handle.write('\n')
    return record,b.sha(path)


def next_invocation(root):
    paths=list((Path(root)/'invocations').glob('*.json'))
    return 'invocations/'+f'{max((int(p.stem) for p in paths),default=0)+1:06d}'+'.json'


async def sweep(b,cell_type,*,phase='main',max_cells=1):
    import aiohttp
    import inspect
    from ecopadg.serving.cell import run_cell
    from ecopadg.measure.backends import PynvmlBackend
    require(type(max_cells) is int and max_cells>0,'explicit positive max_cells required')
    require(phase in PHASES,'unknown phase')
    require(Path(inspect.getfile(run_cell)).resolve()==b.HOST/'src/ecopadg/serving/cell.py','wrong immutable host import')
    spec=b.read(b.ROOT/'runspec.json');validate_spec(b,spec)
    require(phase_rows(spec,phase),'selected phase is not yet declared')
    name=next_invocation(b.ROOT)
    status=dict(phase='preflight',selected_phase=phase,complete=False,started_s=time.time(),cells=[],
        baseline_execution=False,protocol_id=PROTOCOL,full_matrix_complete=False,max_cells=max_cells)
    b.write(name,status);freeze=None;progress=None;clock=None
    try:
        if (b.ROOT/'STOP').exists():
            status.update(phase='stopped_by_request',complete=True);return status
        clock=phase_record(b,spec,phase,create=True);status['phase_clock']=clock
        # The first check avoids expensive or hardware preflight for a phase
        # whose reserved window cannot fit. Recheck after all slow preflight.
        if cell_limits(spec,clock) is None:
            status.update(phase='stopped_by_deadline',complete=True);return status
        freeze=await asyncio.to_thread(b.frozen_check,True)
        progress=await asyncio.to_thread(inspect_queue,b,phase)
        selected=phase_rows(spec,phase)[progress['completed']:][:max_cells]
        status.update(previous_completed_cells=progress['completed'],invocation_selected_cells=len(selected))
        if selected and cell_limits(spec,clock) is not None and not (b.ROOT/'STOP').exists():
            hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
            async with aiohttp.ClientSession(trust_env=False) as session:
                for row in selected:
                    if (b.ROOT/'STOP').exists():status['stop_requested']=True;break
                    if cell_limits(spec,clock) is None:status['deadline_reached']=True;break
                    await asyncio.to_thread(b.frozen_check,False)
                    if hasattr(b,'materialize'):await asyncio.to_thread(b.materialize,row)
                    require(b.sha(row['trace'])==row['trace_sha256'],'selected trace changed')
                    reference=await asyncio.to_thread(main_reference,b,row)
                    if (b.ROOT/'STOP').exists():status['stop_requested']=True;break
                    limits=cell_limits(spec,clock)
                    if limits is None:status['deadline_reached']=True;break
                    status.update(phase='cell:'+row['cell_id'],current_limits=limits);b.write(name,status)
                    cell=cell_type(b,session,row,freeze,status,name,hardware,limits=limits)
                    await asyncio.wait_for(cell.execute(),timeout=max(.001,limits['restore_deadline_s']-time.time()))
                    await asyncio.to_thread(b.frozen_check,False)
                    identity=await b.live(session,expected_inventory=freeze['inventory'])
                    b.write('identities/'+row['cell_id']+'.json',identity)
                    record,previous=await asyncio.to_thread(checkpoint,b,row,limits,
                        progress['previous_checkpoint_sha256'],reference)
                    progress['records'].append(record);progress['previous_checkpoint_sha256']=previous;progress['completed']+=1
                    status['checkpointed_cells']=progress['completed'];b.write(name,status)
                b.write('after/'+Path(name).name,await b.live(session,expected_inventory=freeze['inventory']))
        elif selected:
            status['stop_requested']=(b.ROOT/'STOP').exists()
            status['deadline_reached']=not status['stop_requested']
        await asyncio.to_thread(b.frozen_check,True)
        status.update(phase='stopped_by_request' if status.get('stop_requested') else
            'stopped_by_deadline' if status.get('deadline_reached') else 'finished',complete=True,
            checkpointed_cells=progress['completed'],remaining_cells=len(phase_rows(spec,phase))-progress['completed'],
            selected_phase_execution_complete=progress['completed']==len(phase_rows(spec,phase)))
    except BaseException as exc:
        status.update(phase='failed',error=repr(exc));raise
    finally:
        if freeze is not None:
            try:
                for path,digest in freeze['dependencies']['protected_files'].items():
                    require(b.sha(path)==digest,'protected baseline changed: '+path)
                status['baseline_preservation_verified']=True
            except BaseException as exc:status.update(complete=False,phase='failed',baseline_preservation_error=repr(exc))
        status['finished_s']=time.time();b.write(name,status)
    require(status['complete'],'invocation failed; evidence retained')
    return status
