"""Exactly three immutable PDB-only original-work replays, no retries."""
import asyncio
import inspect
import time
from pathlib import Path


def cell_limits(spec,row,now):
    # This is the original trace span, not the fixed300 protocol.
    span=row['trace_duration_s'];end=spec['deadline_s']
    if now+60+span+120+120>end:return None
    return dict(schema=1,issued_s=now,arrival_span_s=span,latest_arrival_epoch_s=now+60,
                cell_execution_deadline_s=now+60+span+120,
                restore_deadline_s=now+60+span+120+120,phase_deadline_s=end)


async def sweep(b):
    import aiohttp
    from ecopadg.serving.cell import run_cell
    from ecopadg.measure.backends import PynvmlBackend
    from execution import Cell
    b.require(Path(inspect.getfile(run_cell)).resolve()==b.HOST/'src/ecopadg/serving/cell.py','wrong host')
    b.require(not (b.ROOT/'status.json').exists(),'attempt exists; preserve raw and never retry here')
    spec=b.package_check();status=dict(phase='preflight',complete=False,cells=[],started_s=time.time(),
                                    baseline_execution=False,formal_eligible=False)
    b.write('status.json',status);freeze=None
    try:
        freeze=await asyncio.to_thread(b.frozen_check,True)
        hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            for row in spec['cells']:
                if (b.ROOT/'STOP').exists():status.update(phase='stopped_by_request',complete=True);break
                if cell_limits(spec,row,time.time()) is None:
                    status.update(phase='stopped_by_deadline',complete=True);break
                await asyncio.to_thread(b.frozen_check,False)
                limits=cell_limits(spec,row,time.time())
                if limits is None:status.update(phase='stopped_by_deadline',complete=True);break
                status['phase']='cell:'+row['cell_id'];b.write('status.json',status)
                cell=Cell(b,session,row,freeze,status,'status.json',hardware,limits=limits)
                await asyncio.wait_for(cell.execute(),max(.01,limits['restore_deadline_s']-time.time()))
                await asyncio.to_thread(b.frozen_check,False)
                identity=await b.live(session,expected_inventory=freeze['inventory'])
                b.write('identities/'+row['cell_id']+'.json',identity)
            else:status.update(phase='finished',complete=True)
            b.write('after.json',await b.live(session,expected_inventory=freeze['inventory']))
        await asyncio.to_thread(b.frozen_check,True)
    except BaseException as exc:
        status.update(phase='failed',complete=False,error=repr(exc));raise
    finally:
        if freeze is not None:
            try:
                for p,h in freeze['dependencies']['protected_files'].items():
                    b.require(b.sha(p)==h,'historical baseline changed: '+p)
                status['baseline_preservation_verified']=True
            except BaseException as exc:
                status.update(phase='failed',complete=False,baseline_preservation_error=repr(exc))
        status['finished_s']=time.time();b.write('status.json',status)
    b.require(status['complete'],'replay failure preserved')
