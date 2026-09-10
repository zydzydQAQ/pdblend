"""Wait for the current B14 PDB boundary, then use the corrected native retain URL.

A dedicated marker is consumed only after the prior owner exits at its next
child boundary. Successful PDB work and completed baseline creation are reused.
No active measurement or deployment process receives a cancellation signal.
"""
import argparse
import asyncio
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import pipeline_v3 as m
p=m.p
HERE=Path(__file__).resolve().parent


def checked_prior(prior, boundary):
    p.need(prior['node']=='B' and prior['model']=='14b' and prior['datasets']==['sharegpt'], 'foreign prior migration')
    p.need(prior.get('finished_s') and not p.active_owner(prior) and not prior.get('node_lease_held'), 'prior migration still owns node')
    p.need(prior.get('error')=="ValueError('stop requested before next stage')", 'prior failure is not this planned child boundary')
    p.need(prior.get('child',{}).get('exitcode')==0 and not p.active_owner(prior['child']), 'prior child did not finish cleanly')
    p.need(boundary['complete'] and boundary['pdb_boundary_complete'] and boundary['scope']=='pdblend', 'PDB boundary is not complete')
    p.need(prior['declaration']==boundary['declaration'] and prior['observations']==boundary['observations']
           and prior['last_cell_status']==boundary['last_cell_status'], 'PDB evidence changed at handoff')
    last=p.checked(prior['last_cell_status'])
    p.need(last['complete'] and last['finished_s'] and not last.get('failed') and not last.get('error')
           and not last['node_lease_held'] and not p.active_owner(last), 'last PDB measurement did not cleanly exit')
    decision=m.decision(prior)
    p.need(decision['pdb_boundary_complete'] and decision['phase'] in ('baselines','complete'), 'original fixed-grid PDB boundary does not validate')
    return decision


async def wait_boundary(plan,state,out):
    source=Path(plan['prior_pipeline_status']);marker=HERE/'STOP'
    marker_value=dict(schema='planned-B14-Q3-baseline-handoff-v1',owner_pid=state['pid'],owner_startticks=state['startticks'],
                      plan=state['plan'],prior_pipeline_status=str(source),reason='avoid known old /retain_weights URL; use /retain-weights without repeating valid PDB work')
    marker_bytes=(json.dumps(marker_value,sort_keys=True)+'\n').encode()
    while True:
        old=p.read(source)
        state['prior_progress']=dict(phase=old.get('phase'),observations=len(old.get('observations',[])),checked_s=time.time())
        if old.get('phase')=='creating_14b_baselines':
            p.need(not (HERE.parent/'STOP').exists(), 'node stop requested')
            p.need(old.get('pdb_boundary'), 'PDB boundary snapshot absent')
            p.need(not marker.exists(), 'a different stop marker already exists')
            # The producer checks only the node-wide STOP at startup; this
            # marker is checked by the supervisor before its next child.
            with marker.open('xb') as stream:stream.write(marker_bytes)
            state.update(phase='awaiting_clean_baseline_handoff',requested_stop_s=time.time())
            p.save(out/'stop-request.json',marker_value);p.save(out/'status.json',state)
            break
        p.need(not old.get('finished_s'), 'prior migration ended before the planned PDB boundary')
        p.need(old.get('phase')!='fresh_14b_baseline_qualification', 'old qualification already began; do not interfere')
        p.save(out/'status.json',state);await asyncio.sleep(.5)
    while True:
        old=p.read(source)
        if old.get('finished_s') and not p.active_owner(old):break
        p.save(out/'status.json',state);await asyncio.sleep(.5)
    boundary=p.checked(old['pdb_boundary']);decision=checked_prior(old,boundary)
    p.need(marker.read_bytes()==marker_bytes, 'owned handoff marker was replaced; do not consume external stop')
    p.need(not (HERE.parent/'STOP').exists(), 'node stop requested')
    os.replace(marker,out/'consumed-stop.json')
    previous_ref=p.ref(source);p.need(p.checked(previous_ref)==old,'prior terminal changed while freezing')
    state.update(prior_pipeline=previous_ref,pdb_boundary=old['pdb_boundary'],declaration=old['declaration'],
        observations=copy.deepcopy(old['observations']),attempts=[],last_cell_status=old['last_cell_status'],
        inherited_attempts=copy.deepcopy(old['attempts']),group_decisions={'sharegpt':decision},
        scope='five_systems',schema='uniform-v2-node-pipeline-status')
    p.save(out/'status.json',state)
    return old


async def run(plan,state,out):
    effective=await m.wait_stage_release(plan,state,out)
    previous=await wait_boundary(effective,state,out)
    migration_lock=Path(plan['migration_supervisor_lock']).open('a+')
    fcntl.flock(migration_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    node_lock=Path(plan['predecessor_supervisor_lock']).open('a+')
    fcntl.flock(node_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        m.check_files(effective)
        deployment=Path(plan['prior_pipeline_status']).parent/'baseline-deployment-001'
        if deployment.exists():
            owner=p.read(deployment/'status.json')
            p.need(owner['complete'] and owner['finished_s'] and not owner.get('error') and not owner['node_lease_held']
                   and not p.active_owner(owner), 'previous baseline creation is incomplete or still active')
            bootstrap=p.ref(deployment/'bootstrap/binding.json')
            p.need(p.checked(bootstrap)['model']=='14b', 'foreign baseline deployment')
            state['reused_complete_deployment']=dict(bootstrap=bootstrap,owner=p.ref(deployment/'status.json'))
        else:
            state['phase']='creating_14b_baselines';p.save(out/'status.json',state)
            deployment=out/'baseline-deployment-001'
            await m.child([sys.executable,'-B',effective['baseline_producer']['path'],'--spec',effective['baseline_template']['path'],
                '--predecessor-terminal',state['pdb_boundary']['path'],'--out',deployment,'--run'],out/'baseline-deployment.log',state,out)
            bootstrap=p.ref(deployment/'bootstrap/binding.json')
        state['phase']='fresh_14b_baseline_qualification';p.save(out/'status.json',state)
        qualification=out/'baseline-qualification-001'
        await m.child([sys.executable,'-B',effective['baseline_qualification']['path'],'--bootstrap',bootstrap['path'],'--out',qualification,
             '--node','B','--hostname',effective['expected_hostname'],'--profile',effective['profile']['path'],'--run'],out/'baseline-qualification.log',state,out)
        bindings=p.read(qualification/'bindings.json')
        for system in ('mixed','distserve','dynamollm','ecoserve'):
            state['phase']=system;p.save(out/'status.json',state)
            handoff=await m.meter_handoff(effective,state,out,system,bindings[system],effective['baseline_validator'],[])
            while True:
                selected=m.decision(state);state['group_decisions']['sharegpt']=selected;p.save(out/'status.json',state)
                if selected['phase']=='complete':break
                p.need(selected['phase']=='baselines','PDB boundary incomplete or engineering failure')
                tasks=[t for t in selected['baseline_tasks'] if t['action']=='execute' and t['row']['system']==system]
                if not tasks:break
                tasks.sort(key=lambda t:(t['row']['rate_rps'],t['row'].get('measurement_purpose','normal'),t['row']['repeat']))
                await m.cell(effective,state,out,tasks[0]['row'],handoff)
        final=m.decision(state)
        p.need(final['phase']=='complete' and final['five_system_complete'],'B14 ShareGPT all-five-system scope incomplete')
        state.update(complete=True,phase='complete',five_system_complete=True,group_decisions={'sharegpt':final})
    finally:
        node_lock.close();migration_lock.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    reference=p.ref(a.plan);plan=p.checked(reference);m.check_files(plan)
    p.need(socket.gethostname()==plan['expected_hostname'],'wrong physical host')
    if not a.run:print('Q3 baseline handoff CPU source gate passed; no GPU operation');return
    p.need(not a.out.exists(),'new handoff output required');a.out.mkdir(parents=True)
    lock=Path(plan['supervisor_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(schema='uniform-v2-B14-Q3-handoff-guard-status',node='B',model='14b',datasets=['sharegpt'],scope='five_systems',
        declaration=plan['declaration'],plan=reference,pid=os.getpid(),startticks=p.process_identity(os.getpid())['startticks'],
        started_s=time.time(),complete=False,node_lease_held=False,phase='awaiting_PDB_boundary',observations=[],attempts=[])
    p.save(a.out/'status.json',state)
    async def controlled():
        task=asyncio.current_task()
        for signalnum in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(signalnum,task.cancel)
        await run(plan,state,a.out)
    try:asyncio.run(controlled())
    except BaseException as error:state.update(error=repr(error),phase='stopped_failure');raise
    finally:state['finished_s']=time.time();p.save(a.out/'status.json',state);lock.close()

if __name__=='__main__':main()
