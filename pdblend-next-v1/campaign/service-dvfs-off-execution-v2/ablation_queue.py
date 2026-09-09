"""Independent later phase; original main/scale clocks are read only."""
import asyncio
import csv
import inspect
import math
import os
from pathlib import Path
import time

from adapter import PHASE, file_refs, immutable, read, require, sha, source_processes


def checkpoint_header(record, summary, clock):
    require(record.get('schema') == 2
            and type(record.get('completed_s')) in (int,float) and math.isfinite(record['completed_s'])
            and record['limits']['issued_s'] <= record['completed_s'] <= clock['deadline_s'],
            'checkpoint completion timestamp outside actual phase')
    require(record.get('measurement_valid') is True
            and all(field in record and field in summary
                    and type(record[field]) is type(summary[field]) and record[field] == summary[field] for field in
                    ('work_complete','slo_attainment','good_requests','energy_j')),
            'checkpoint metric header differs from verified raw summary')


def phase_record(b, *, now=None, create=False):
    now = time.time() if now is None else now
    spec = read(b.ROOT / 'runspec.json')
    ref = spec['deadline_scope']
    require(sha(ref['path']) == ref['sha256'], 'global deadline identity changed')
    deadline = read(ref['path'])
    require(math.isfinite(now) and deadline['created_s'] <= now,
            'wall clock predates original experiment')
    path = b.ROOT / 'phase-ledger.json'
    prerequisite = read(b.ROOT / 'freeze.json')
    freeze_digest = sha(b.ROOT / 'freeze.json')
    if path.exists():
        record = read(path)
        require(record['schema'] == 1 and record['phase'] == PHASE
                and record['model'] == spec['model'] and record['deadline_scope_sha256'] == ref['sha256']
                and record['deadline_s'] == deadline['deadline_s']
                and record['prerequisite_freeze_sha256'] == freeze_digest
                and math.isfinite(record['started_s'])
                and max(deadline['created_s'], prerequisite['prepared_s']) <= record['started_s'] <= min(now+1, deadline['deadline_s']),
                'ablation phase clock changed or regressed')
    else:
        record = dict(schema=1, phase=PHASE, model=spec['model'], started_s=now,
                      deadline_s=deadline['deadline_s'], deadline_scope_sha256=ref['sha256'],
                      prerequisite_freeze_sha256=freeze_digest)
        require(now >= prerequisite['prepared_s'], 'ablation starts before prerequisite preparation')
        if create and now <= deadline['deadline_s']:
            immutable(path, record)
    return record


def inspect_queue(b):
    spec = b.package_check()
    freeze = read(b.ROOT / 'freeze.json')
    q = b.SOURCE_QUEUE
    paths = sorted((b.ROOT / 'checkpoints' / PHASE).glob('*.json'))
    require(len(paths) <= 18, 'extra ablation checkpoint')
    previous, records = None, []
    clock = phase_record(b) if paths else None
    for index, path in enumerate(paths):
        row = spec['cells'][index]
        cid = row['cell_id']
        record = read(path)
        require(path.name == f'{index+1:04d}-{cid}.json' and record.get('schema') == 2 and record['sequence'] == index+1
                and record['phase'] == PHASE and record['cell_id'] == cid
                and record['package_manifest_sha256'] == sha(b.ROOT / 'package-manifest.json')
                and record['previous_checkpoint_sha256'] == previous
                and record['source_trace_sha256'] == row['trace_sha256'], 'checkpoint chain/source changed')
        require(record['scale_one_reference'] == freeze['terminal_evidence']['references'][row['source_main_cell_id']],
                'paired source reference changed')
        require(record['limits'] == q.cell_limits(spec, clock, now=record['limits']['issued_s']),
                'actual later phase limits changed')
        require(set(record['artifacts']) == set(q.artifacts(b.ROOT, cid)), 'artifact set changed')
        file_refs({str(b.ROOT / p): h for p, h in record['artifacts'].items()})
        summary = q.valid_receipt(read(b.ROOT / 'receipts' / (cid+'.json')), row, record['limits'])
        require(summary == read(b.ROOT / 'cells' / cid / 'summary.json'), 'summary differs from receipt')
        checkpoint_header(record, summary, clock)
        records.append(record)
        previous = sha(path)
    for row in spec['cells'][len(records):]:
        cid = row['cell_id']
        require(not any((b.ROOT / folder / cid).exists() for folder in ('cells', 'operations'))
                and not any((b.ROOT / folder / (cid+'.json')).exists() for folder in ('receipts', 'identities')),
                'uncheckpointed attempt retained; no automatic retry or overwrite')
    return dict(completed=len(records), remaining=18-len(records), records=records,
                previous_checkpoint_sha256=previous, phase_execution_complete=len(records)==18)


def clock_proof(b, cid):
    """Require real all-eight clock samples, not an assertion of fixed hardware MHz."""
    receipt = read(b.ROOT / 'receipts' / (cid+'.json'))
    rows = list(csv.DictReader((b.ROOT / 'operations' / cid / 'power/clocks.csv').open()))
    times = [float(r['t_s']) for r in rows]
    require(len(times) >= 2 and times[0] <= receipt['operation_start_s']
            and receipt['operation_end_s'] <= times[-1] and all(a < z for a, z in zip(times, times[1:])),
            'actual clocks do not bracket complete operation')
    values = [[float(row[f'gpu{i}_sm_mhz']) for i in range(8)] for row in rows]
    require(all(math.isfinite(t) for t in times)
            and all(math.isfinite(v) and v > 0 for row in values for v in row), 'missing/nonfinite actual GPU clock')
    operation=b.ROOT/'operations'/cid
    status=read(operation/'clock-commands-status.json')
    commands=[__import__('json').loads(line) for line in (operation/'clock-commands.jsonl').read_text().splitlines()]
    require(status.get('complete') is True and not status.get('recording_errors')
            and status.get('aliases_restored') is True and status.get('pid') == receipt.get('child_pid')
            and status.get('records') == len(commands)
            and [r['sequence'] for r in commands] == list(range(1,len(commands)+1))
            and all(r['pid']==receipt['child_pid'] and math.isfinite(r['observed_s'])
                    and receipt['operation_start_s']<=r['observed_s']<=receipt['operation_end_s'] for r in commands)
            and math.isfinite(status['finished_s']) and status['finished_s']<=receipt['operation_end_s'],
            'incomplete/foreign clock-command journal')
    acquired=[r for r in commands if r['kind']=='owner_acquired']
    require(len(acquired)==1 and acquired[0]['gpus']==list(range(8))
            and status['owners']==[acquired[0]['owner']], 'original all-eight ClockOwner acquisition missing')
    intents=[r for r in commands if r['kind']=='owner_set']
    writes=[r for r in commands if r['kind']=='hardware_begin']
    require(intents and all(r['frequency_mhz']==2520 for r in intents)
            and writes and all(r['method']=='reset_clock' or (r['method']=='set_clock' and r['frequency_mhz']==2520)
                               for r in writes), 'service DVFS-off issued a nonmaximum clock command')
    intent_ends={r['call']:r for r in commands if r['kind']=='owner_set_end'}
    require(len(intent_ends)==len(intents) and all(intent_ends.get(r['sequence'],{}).get('complete') is True for r in intents),
            'ClockOwner service intent unfinished/failed')
    require(all(type(r['gpu']) is int and r['gpu'] in range(8) for r in writes)
            and all(r.get('owner')==acquired[0]['owner'] for r in intents+writes), 'foreign owner/GPU clock command')
    finished={r['call']:r for r in commands if r['kind']=='hardware_end'}
    require(len(finished)==len(writes) and all(finished.get(r['sequence'],{}).get('complete') is True for r in writes),
            'hardware clock call did not finish successfully')
    closed=[r for r in commands if r['kind']=='owner_close_end']
    require(closed and all(r.get('complete') is True for r in closed), 'child ClockOwner close missing/failed')
    return dict(complete=True, frames=len(rows), all_gpu_indices=list(range(8)),
                service_maximum_commands_verified=True, owner_set_intents=len(intents),
                hardware_set_writes=sum(r['method']=='set_clock' for r in writes),
                hardware_reset_writes=sum(r['method']=='reset_clock' for r in writes),
                actual_physical_fixed_2520_certified=False,
                semantics='active service planner requests maximum; original idle unlock/parking remains unchanged')


async def sweep(b, *, max_cells=1):
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.cell import run_cell
    require(type(max_cells) is int and max_cells > 0, 'positive bounded invocation required')
    require(getattr(b,'LEASE',{}).get('pid')==os.getpid()
            and b.LEASE.get('inherited') is False, 'independent node lease required')
    stat=os.fstat(b.LEASE['fd'])
    require((stat.st_ino,stat.st_dev)==(b.LEASE['inode'],b.LEASE['device']), 'node lease descriptor changed')
    require(Path(inspect.getfile(run_cell)).resolve() == b.HOST / 'src/ecopadg/serving/cell.py',
            'wrong frozen host module')
    q = b.SOURCE_QUEUE
    name = q.next_invocation(b.ROOT)
    status = dict(schema=1, phase='preflight', selected_phase=PHASE, complete=False, pid=os.getpid(),lease=b.LEASE,
                  started_s=time.time(), cells=[], execute_baselines=False, max_cells=max_cells)
    b.write(name, status)
    freeze = None
    try:
        if (b.ROOT / 'STOP').exists():
            status.update(phase='stopped_by_request', complete=True)
            return status
        require(not source_processes(b.ORIGINAL.ROOT), 'original source process still active')
        freeze = await asyncio.to_thread(b.frozen_check, True)
        spec = read(b.ROOT / 'runspec.json')
        progress = await asyncio.to_thread(inspect_queue, b)
        clock = phase_record(b, create=True)
        status['phase_clock'] = clock
        selected = spec['cells'][progress['completed']:][:max_cells]
        if selected and q.cell_limits(spec, clock) is not None:
            hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
            async with aiohttp.ClientSession(trust_env=False) as session:
                for row in selected:
                    if (b.ROOT / 'STOP').exists():
                        status['stop_requested'] = True
                        break
                    if q.cell_limits(spec, clock) is None:
                        status['deadline_reached'] = True
                        break
                    await asyncio.to_thread(b.frozen_check, False)
                    require(not source_processes(b.ORIGINAL.ROOT), 'source process restarted')
                    limits = q.cell_limits(spec, clock)
                    if limits is None:
                        status['deadline_reached'] = True
                        break
                    status.update(phase='cell:'+row['cell_id'], current_limits=limits)
                    b.write(name, status)
                    cell = b.CELL(b, session, row, freeze, status, name, hardware, limits=limits)
                    await asyncio.wait_for(cell.execute(), timeout=max(.001, limits['restore_deadline_s']-time.time()))
                    await asyncio.to_thread(b.frozen_check, False)
                    b.write('identities/'+row['cell_id']+'.json',
                            await b.live(session, expected_inventory=freeze['inventory']))
                    proof = await asyncio.to_thread(clock_proof, b, row['cell_id'])
                    b.write('operations/'+row['cell_id']+'/ablation-clock-evidence.json', proof)
                    reference = freeze['terminal_evidence']['references'][row['source_main_cell_id']]
                    record, previous = await asyncio.to_thread(q.checkpoint, b, row, limits,
                        progress['previous_checkpoint_sha256'], reference)
                    progress['records'].append(record)
                    progress['completed'] += 1
                    progress['previous_checkpoint_sha256'] = previous
                    status['checkpointed_cells'] = progress['completed']
                    b.write(name, status)
        elif selected:
            status['deadline_reached'] = True
        await asyncio.to_thread(b.frozen_check, True)
        status.update(phase='stopped_by_request' if status.get('stop_requested') else
            'stopped_by_deadline' if status.get('deadline_reached') else 'finished', complete=True,
            checkpointed_cells=progress['completed'], remaining_cells=18-progress['completed'],
            selected_phase_execution_complete=progress['completed']==18)
    except BaseException as exc:
        status.update(phase='failed', error=repr(exc))
        raise
    finally:
        if freeze is not None:
            try:
                file_refs(freeze['dependencies']['protected_files'])
                status['baseline_preservation_verified'] = True
            except BaseException as exc:
                status.update(phase='failed', complete=False, baseline_preservation_error=repr(exc))
        status['finished_s'] = time.time()
        b.write(name, status)
    require(status['complete'], 'ablation invocation failed; retain every attempt')
    return status
