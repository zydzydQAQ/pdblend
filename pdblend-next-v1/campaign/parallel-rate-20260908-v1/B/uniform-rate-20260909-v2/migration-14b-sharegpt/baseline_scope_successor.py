"""Resume the original ShareGPT baselines after a completed scope-review pause."""
import argparse
import asyncio
import copy
import fcntl
import os
from pathlib import Path
import signal
import socket
import sys
import time
import baseline_q3_successor as q
import pipeline_v4 as m
import audit_sharegpt_scope as audit
p = m.p
HERE = Path(__file__).resolve().parent


def validate_handoff(plan):
    prior = p.checked(plan['prior_pipeline'])
    p.need(plan['prior_pipeline']['path'] == plan['prior_pipeline_status'], 'predecessor path changed')
    boundary = p.checked(prior['pdb_boundary'])
    decision = q.checked_prior(prior, boundary)
    proof = audit.verify(plan['implementation_scope_audit'])
    p.need(proof['passed'], 'implementation scope remains unresolved')
    marker = p.checked(plan['owned_pause_marker'])
    p.need(marker['schema'] == 'B14-formal-capacity-scope-pause-v1', 'not the reviewed scope pause')
    p.need(plan['owned_pause_marker']['path'] == str(HERE/'STOP'), 'different stop marker')
    p.need(not (HERE.parent/'STOP').exists(), 'independent node STOP remains')
    p.need(not (Path(plan['prior_pipeline_status']).parent/'baseline-deployment-001').exists(),
           'baseline already started; a separate continuation is required')
    return prior, decision


async def adopt(plan, state, out):
    prior, decision = validate_handoff(plan)
    # The only removed stop is the exact SHA-frozen marker created by this review.
    os.replace(plan['owned_pause_marker']['path'], out/'consumed-scope-pause.json')
    state.update(prior_pipeline=plan['prior_pipeline'], pdb_boundary=prior['pdb_boundary'],
                 declaration=prior['declaration'], observations=copy.deepcopy(prior['observations']),
                 inherited_attempts=copy.deepcopy(prior['attempts']), attempts=[],
                 last_cell_status=prior['last_cell_status'], binding=prior['binding'],
                 group_decisions={'sharegpt': decision}, implementation_scope_audit=plan['implementation_scope_audit'],
                 schema='uniform-v2-node-pipeline-status', scope='five_systems')
    p.save(out/'status.json', state)
    return prior


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--run',action='store_true')
    args=parser.parse_args(); reference=p.ref(args.plan); plan=p.checked(reference)
    m.check_files(plan)
    p.need(socket.gethostname()==plan['expected_hostname'], 'wrong physical host')
    p.need(os.environ.get('PYTHONPATH')==plan['runtime_pythonpath'], 'qualification dependency path changed')
    validate_handoff(plan)
    if not args.run:
        print('Original ShareGPT scope, completed boundary and exact owned STOP verified; no GPU work'); return
    p.need(not args.out.exists(), 'new successor output required'); args.out.mkdir()
    lock=Path(plan['supervisor_lock']).open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(schema='uniform-v2-B14-Q3-handoff-guard-status', node='B', model='14b', datasets=['sharegpt'],
               scope='five_systems', plan=reference, declaration=plan['declaration'],
               pid=os.getpid(), startticks=p.process_identity(os.getpid())['startticks'],
               started_s=time.time(), complete=False, node_lease_held=False,
               phase='adopting_completed_scope_review', observations=[], attempts=[])
    p.save(args.out/'status.json',state)
    # Reuse the previously reviewed Q3 continuation with the v4 child environment.
    q.m=m; q.wait_boundary=adopt
    async def controlled():
        task=asyncio.current_task()
        for signum in (signal.SIGTERM,signal.SIGINT): asyncio.get_running_loop().add_signal_handler(signum,task.cancel)
        await q.run(plan,state,args.out)
    try: asyncio.run(controlled())
    except BaseException as error: state.update(error=repr(error),phase='stopped_failure'); raise
    finally:
        state['finished_s']=time.time(); p.save(args.out/'status.json',state); lock.close()


if __name__=='__main__': main()
