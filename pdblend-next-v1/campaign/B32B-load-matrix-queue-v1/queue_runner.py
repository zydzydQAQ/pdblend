"""Bounded, resumable queue around an existing reviewed Cell implementation.

No automatic retry/adoption: an attempted cell without a valid immutable checkpoint
stops resume. Stop after a completed cell with max_cells; failed raw stays untouched.
"""
import asyncio
import json
from pathlib import Path
import time


def require(ok,reason):
    if not ok:raise ValueError(reason)


def artifacts(root,cid):
    root=Path(root);paths=[]
    for folder in ('cells','operations'):
        directory=root/folder/cid
        require(directory.is_dir(),'completed cell artifacts missing: '+str(directory))
        paths.extend(p for p in directory.rglob('*') if p.is_file())
    for name in (f'receipts/{cid}.json',f'identities/{cid}.json'):
        p=root/name;require(p.is_file(),'completed receipt/identity missing: '+name);paths.append(p)
    return sorted(str(p.relative_to(root)) for p in paths)


def valid_receipt(receipt,row):
    require(receipt.get('screen_valid') is True and receipt.get('outer_cleanup',{}).get('complete') is True,'failed execution/cleanup cannot be skipped')
    summary=receipt.get('summary',{})
    require(summary.get('measurement_valid') is True and summary.get('measurement_schema')==3,'invalid measurement cannot be skipped')
    require(summary.get('n_expected')==row['n_requests'] and summary.get('trace_sha256')==row['trace_sha256'],'receipt work/source identity differs')
    return summary


def inspect_queue(b):
    """Read-only prefix verification; every previously checkpointed artifact is hashed."""
    spec=b.read(b.ROOT/'runspec.json');rows=spec['cells'];root=b.ROOT
    records=[];paths=sorted((root/'checkpoints').glob('*.json')) if (root/'checkpoints').exists() else []
    require(len(paths)<=len(rows),'too many checkpoints')
    previous=None;package=b.sha(root/'package-manifest.json')
    for index,path in enumerate(paths):
        row=rows[index];cid=row['cell_id'];record=b.read(path)
        require(path.name==f'{index+1:04d}-{cid}.json' and record.get('schema')==1
            and record.get('sequence')==index+1 and record.get('cell_id')==cid,'checkpoint queue ordering changed')
        require(record.get('package_manifest_sha256')==package and record.get('previous_checkpoint_sha256')==previous
            and record.get('source_trace_sha256')==row['trace_sha256'],'checkpoint source/chain changed')
        expected=artifacts(root,cid);require(set(record.get('artifacts',{}))==set(expected),'checkpoint artifact set changed')
        for rel,digest in record['artifacts'].items():
            require(b.sha(root/rel)==digest,'checkpoint artifact changed: '+rel)
        summary=valid_receipt(b.read(root/'receipts'/(cid+'.json')),row)
        require(summary==b.read(root/'cells'/cid/'summary.json'),'receipt and actual summary differ')
        require(b.sha(row['trace'])==row['trace_sha256'],'original trace changed')
        records.append(record);previous=b.sha(path)
    for row in rows[len(records):]:
        cid=row['cell_id']
        require(not any((root/folder/cid).exists() for folder in ('cells','operations'))
            and not (root/'receipts'/(cid+'.json')).exists()
            and not (root/'identities'/(cid+'.json')).exists(),
            'uncheckpointed attempted cell retained; no automatic retry/adoption: '+cid)
    return dict(completed=len(records),remaining=len(rows)-len(records),records=records,
        previous_checkpoint_sha256=previous,selected_execution_complete=len(records)==len(rows),
        full_matrix_complete=False)


def checkpoint(b,row,previous):
    cid=row['cell_id'];summary=valid_receipt(b.read(b.ROOT/'receipts'/(cid+'.json')),row)
    require(summary==b.read(b.ROOT/'cells'/cid/'summary.json'),'receipt and summary differ at checkpoint')
    paths=artifacts(b.ROOT,cid);values={rel:b.sha(b.ROOT/rel) for rel in paths}
    sequence=row['sequence'];path=b.ROOT/'checkpoints'/f'{sequence:04d}-{cid}.json'
    require(not path.exists(),'checkpoint already exists; refuse overwrite')
    value=dict(schema=1,sequence=sequence,cell_id=cid,completed_s=time.time(),
        package_manifest_sha256=b.sha(b.ROOT/'package-manifest.json'),source_trace_sha256=row['trace_sha256'],
        previous_checkpoint_sha256=previous,artifacts=values,
        measurement_valid=True,work_complete=summary.get('work_complete'),slo_attainment=summary.get('slo_attainment'),
        long_workload_acceptance='separate raw actual-dispatch/full-work audit required')
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation fails closed. An interrupted write remains visible and
    # blocks resume; it never becomes a silent successful checkpoint.
    with path.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
    return value,b.sha(path)


def next_invocation(root):
    paths=list((Path(root)/'invocations').glob('*.json'))
    numbers=[int(p.stem) for p in paths]
    return 'invocations/'+f'{max(numbers,default=0)+1:06d}'+'.json'


async def sweep(b,cell_type,*,max_cells=None):
    import aiohttp
    import inspect
    from ecopadg.serving.cell import run_cell
    from ecopadg.measure.backends import PynvmlBackend
    require(max_cells is None or type(max_cells) is int and max_cells>0,'max_cells must be positive')
    b.require(Path(inspect.getfile(run_cell)).resolve()==b.HOST/'src/ecopadg/serving/cell.py','wrong frozen host module')
    spec=b.read(b.ROOT/'runspec.json');name=next_invocation(b.ROOT)
    require(not (b.ROOT/name).exists(),'invocation collision')
    status=dict(phase='preflight',complete=False,started_s=time.time(),cells=[],baseline_execution=False,
        selected_cells=len(spec['cells']),declared_matrix_cells=spec.get('declared_matrix_cells',513),
        full_matrix_complete=False,max_cells=max_cells,scope='PDB development only; resumable at completed cell checkpoints; no automatic retries')
    b.write(name,status);freeze=None;progress=None
    try:
        freeze=await asyncio.to_thread(b.frozen_check,True)
        progress=await asyncio.to_thread(inspect_queue,b)
        start=progress['completed'];pending=spec['cells'][start:]
        selected=pending if max_cells is None else pending[:max_cells]
        status.update(previous_completed_cells=start,invocation_selected_cells=len(selected));b.write(name,status)
        # No hardware/session/control is constructed if the queue is already complete.
        if (b.ROOT/'STOP').exists():status['stop_requested']=True
        if selected and not status.get('stop_requested'):
            hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
            async with aiohttp.ClientSession(trust_env=False) as session:
                for row in selected:
                    if (b.ROOT/'STOP').exists():
                        status['stop_requested']=True;break
                    await asyncio.to_thread(b.frozen_check,False)
                    if hasattr(b,'materialize'):await asyncio.to_thread(b.materialize,row)
                    if (b.ROOT/'STOP').exists():
                        status['stop_requested']=True;break
                    b.require(b.sha(row['trace'])==row['trace_sha256'],'selected trace changed')
                    with b.socket.socket() as sock:sock.bind(('127.0.0.1',b.read(b.CONFIG)['port']))
                    status['phase']='cell:'+row['cell_id'];b.write(name,status)
                    cell=cell_type(b,session,row,freeze,status,name,hardware)
                    await cell.execute()
                    await asyncio.to_thread(b.frozen_check,False)
                    identity=await b.live(session,expected_inventory=freeze['inventory'])
                    b.write('identities/'+row['cell_id']+'.json',identity)
                    record,previous=await asyncio.to_thread(checkpoint,b,row,progress['previous_checkpoint_sha256'])
                    progress['records'].append(record);progress['previous_checkpoint_sha256']=previous;progress['completed']+=1
                    status['checkpointed_cells']=progress['completed'];b.write(name,status)
                b.write('after/'+Path(name).name,await b.live(session,expected_inventory=freeze['inventory']))
        await asyncio.to_thread(b.frozen_check,True)
        status.update(phase='stopped_by_request' if status.get('stop_requested') else 'finished',complete=True,checkpointed_cells=progress['completed'],
            remaining_cells=len(spec['cells'])-progress['completed'],
            selected_queue_execution_complete=progress['completed']==len(spec['cells']),full_matrix_complete=False)
    except BaseException as exc:status.update(phase='failed',error=repr(exc));raise
    finally:
        if freeze is not None:
            try:
                for path,digest in freeze['dependencies']['protected_files'].items():
                    b.require(b.sha(path)==digest,'protected baseline changed: '+path)
                status['baseline_preservation_verified']=True
            except BaseException as exc:status.update(complete=False,phase='failed',baseline_preservation_error=repr(exc))
        status['finished_s']=time.time();b.write(name,status)
    b.require(status['complete'],'invocation did not complete; retain failed evidence')
    return status
