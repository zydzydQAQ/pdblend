"""Independent clean-terminal evidence; native endpoints and clocks are read only."""
import argparse
import asyncio
import csv
import fcntl
import io
import math
import os
from pathlib import Path
import socket
import sys
import time
import pipeline_v4 as m
p=m.p
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def exited(value):
    p.need(value.get('finished_s') and not p.active_owner(value), 'owner is still active or not terminal')
    observed=p.process_identity(value['pid'])
    return dict(pid=value['pid'],startticks=value['startticks'],active=False,
                observed_process=observed,finished_s=value['finished_s'])


def lock_probe(path):
    start=time.time()
    # Open an existing lock; never create an alternative lock namespace.
    with Path(path).open('r+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        acquired=time.time()
        fcntl.flock(stream,fcntl.LOCK_UN)
    return dict(path=str(path),started_s=start,acquired_s=acquired,released_s=time.time(),
                exclusive_acquire_succeeded=True,released=True)


async def collect(reference,out):
    state=p.checked(reference)
    p.need(socket.gethostname()=='iZwz9i5bte3xkpmcoes3t2Z', 'wrong physical host')
    p.need(state['node']=='B' and state['model']=='14b' and state['datasets']==['sharegpt']
           and state['scope']=='five_systems' and state['complete'] and state['five_system_complete']
           and not state.get('error') and not state['node_lease_held'], 'group is not a successful five-system terminal')
    owner=exited(state);child=exited(state['child'])
    last=p.checked(state['last_cell_status'])
    p.need(last['complete'] and not last['failed'] and not last.get('error') and not last['node_lease_held'],
           'final measurement cleanup is incomplete')
    last_owner=exited(last)
    decision=m.decision(state)
    p.need(decision['phase']=='complete' and decision['five_system_complete'] and decision['pdb_boundary_complete'],
           'independent contract replay is incomplete')
    observations=[p.checked(r) for r in state['observations']]
    counts={system:sum(o['system']==system for o in observations) for system in m.contract.SYSTEMS}
    p.need(counts==dict(pdblend=7,mixed=6,distserve=6,dynamollm=6,ecoserve=6), 'unexpected work count')
    metrics=('energy_j','slo_attainment','ttft_avg_s','tpot_avg_s','token_throughput_tps','request_throughput_rps','gpu_util')
    for observation in observations:
        p.need(observation['measurement_host']=='B' and observation['measurement_valid'] and observation['work_complete']
               and observation['strict_slo_recomputed'] and observation['token_throughput_is_exact'],
               'unresolved measurement or token evidence')
        p.need(observation.get('measurement_purpose')!='metric_supplement', 'unexpected supplemental work')
        p.need(all(isinstance(observation.get(k),(int,float)) and math.isfinite(observation[k]) for k in metrics),
               'missing required metric')
        p.need(len(observation['energy_per_gpu_j'])==len(observation['gpu_util_per_gpu'])==8
               and math.isclose(sum(observation['energy_per_gpu_j']),observation['energy_j'],rel_tol=1e-10),
               'eight-GPU energy evidence mismatch')
    for rate in (.25,.5,.75,1.,1.25,1.5):
        rows=[o for o in observations if o['rate_rps']==rate]
        p.need({o['system'] for o in rows}==set(m.contract.SYSTEMS) and len({o['trace_sha256'] for o in rows})==1,
               'same-host exact-trace pairing failed')
    final=observations[-1];checkpoint=p.checked(final['checkpoint']);receipt=p.checked(checkpoint['receipt'])
    p.need(receipt['measurement_valid'] and receipt['child_stopped'] and receipt['clock_restore_complete']
           and not receipt['outer_cleanup_errors'] and len(receipt['restoration'])==8
           and all(x['complete'] for x in receipt['restoration'].values()), 'native or clock restoration is not complete')
    binding=p.checked(checkpoint['binding'])
    p.need(binding['hostname']==socket.gethostname() and binding['model']=='14b', 'foreign final binding')
    expected=p.checked(p.ref(HERE/'node-identity.json'))
    host=Path(binding['host_release'])
    sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
    executor=p.load(ROOT/'common/execution-until-complete-v1/run.py','terminal_frozen_executor')
    import aiohttp
    async with aiohttp.ClientSession(trust_env=False) as session:
        identity=await executor.identity(session,binding)
    p.save(out/'native-identity-and-idle.json',identity)
    gpu_text=await executor.command('nvidia-smi','--query-gpu=index,uuid,clocks.current.sm,clocks.max.sm,clocks.applications.graphics,clocks.default_applications.graphics,utilization.gpu','--format=csv,noheader,nounits')
    rows=list(csv.reader(io.StringIO(gpu_text)))
    p.need(len(rows)==8,'eight physical GPUs not observed')
    gpus=[]
    for row,wanted in zip(rows,expected['GPUs']):
        index=int(row[0]);uuid=row[1].strip()
        p.need(index==wanted['index'] and uuid==wanted['uuid'],'physical GPU identity changed')
        gpus.append(dict(index=index,uuid=uuid,current_sm_mhz=int(row[2]),max_sm_mhz=int(row[3]),
                         applications_graphics=row[4].strip(),default_applications_graphics=row[5].strip(),
                         utilization_pct=int(row[6])))
    p.save(out/'gpu-clock-snapshot.json',dict(observed_s=time.time(),gpus=gpus,raw_csv=gpu_text))
    locks=[lock_probe('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')]
    locks += [lock_probe(Path('/root/workspace/pdblend/new-results/.clock-locks')/f'pdblend-gpu-{i}.lock') for i in range(8)]
    # Recheck all terminal owners after native inspection and the lock probes.
    for value in (state,state['child'],last):exited(value)
    p.need(p.checked(reference)==state and p.checked(state['last_cell_status'])==last, 'terminal changed during independent audit')
    return dict(schema='B14-ShareGPT-independent-clean-five-system-terminal-v1',passed=True,
        node='B',model='14b',dataset='sharegpt',scope='five_systems',pipeline=reference,last_cell=state['last_cell_status'],
        final_checkpoint=final['checkpoint'],final_receipt=checkpoint['receipt'],binding=checkpoint['binding'],
        owner_exit=owner,child_exit=child,last_measurement_owner_exit=last_owner,lock_probes=locks,
        native_idle=p.ref(out/'native-identity-and-idle.json'),gpu_clock_snapshot=p.ref(out/'gpu-clock-snapshot.json'),
        clock_restoration=dict(passed=True,receipt_clock_restore_complete=True,
            source=p.ref(host/'src/ecopadg/serving/backend.py'),backend=p.ref(host/'src/ecopadg/measure/backends.py'),
            semantics='Pinned ClockOwner.close resets every GPU lock and reports any failure; all eight clock locks are independently free. Current idle SM is recorded, not forced to a service frequency.'),
        measurements_by_system=counts,coordinates=30,observations=state['observations'],decision=decision,
        source=p.ref(__file__),finished_s=time.time(),native_or_frequency_writes_issued=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pipeline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();p.need(not args.out.exists(),'new independent evidence directory required');args.out.mkdir()
    proof=asyncio.run(collect(p.ref(args.pipeline),args.out));p.save(args.out/'proof.json',proof);print(p.ref(args.out/'proof.json'))
