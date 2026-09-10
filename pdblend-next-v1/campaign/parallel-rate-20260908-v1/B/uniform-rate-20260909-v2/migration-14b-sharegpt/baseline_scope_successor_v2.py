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
    archived = Path(plan['reviewed_preflight_failure']['path']).parent/'consumed-scope-pause.json'
    p.need(plan['owned_pause_marker']['path'] == str(archived), 'different archived stop marker')
    p.need(not (HERE/'STOP').exists(), 'new scope STOP must not be consumed')
    failed = p.checked(plan['reviewed_preflight_failure'])
    p.need(failed['node']=='B' and failed['model']=='14b' and failed['datasets']==['sharegpt']
           and failed.get('finished_s') and not p.active_owner(failed) and not failed['node_lease_held'],
           'failed preflight owner remains active or foreign')
    p.need(failed['error']=="ValueError('migration child failed; preserve evidence and stop')"
           and failed['child']['exitcode']==1 and not p.active_owner(failed['child']), 'different predecessor failure')
    p.need(failed['declaration']==prior['declaration'] and failed['observations']==prior['observations']
           and failed['last_cell_status']==prior['last_cell_status'], 'PDB evidence differs after preflight')
    operation_ref = plan['failed_preflight_operation']
    operation = p.checked(operation_ref)
    p.need(operation.get('finished_s') and not operation['complete'] and not p.active_owner(operation)
           and operation['error']=="RuntimeError('new port is occupied: 59188')", 'unrecognized deployment failure')
    operation_dir = Path(operation_ref['path']).parent
    p.need(operation_dir == Path(plan['reviewed_preflight_failure']['path']).parent/'baseline-deployment-001',
           'foreign failed deployment')
    for name in ('deployment/deployment-receipt.json','deployment/deployment-power','deployment/creation-intents','bootstrap'):
        p.need(not (operation_dir/name).exists(), 'failed preflight contains hardware action evidence')
    p.need(p.checked(operation['spec'])['predecessor_terminal']==prior['pdb_boundary'], 'failed preflight used another boundary')
    p.need(str(HERE/'baseline_producer_v2.py') in failed['child']['argv'], 'failed child was not the frozen preflight source')

    p.need(not (HERE.parent/'STOP').exists(), 'independent node STOP remains')
    p.need(not (Path(plan['prior_pipeline_status']).parent/'baseline-deployment-001').exists(),
           'baseline already started; a separate continuation is required')
    return prior, decision


async def adopt(plan, state, out):
    prior, decision = validate_handoff(plan)
    # The original pause was already archived by the CPU-only failed predecessor.
    state['reviewed_preflight_failure']=plan['reviewed_preflight_failure']
    state['archived_scope_pause']=plan['owned_pause_marker']
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
