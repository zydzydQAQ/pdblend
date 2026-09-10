"""Single B14B ShareGPT successor: fresh qualification, PDB boundary, four baselines."""
import argparse
import asyncio
import fcntl
import os
from pathlib import Path
import signal
import socket
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
COMMON=ROOT/'common/uniform-rate-20260909-v2'
sys.path.insert(0,str(COMMON))
import support as p
import contract


def check_files(plan):
    for path,digest in plan['files'].items():p.need(p.sha(path)==digest,'frozen migration source changed: '+path)
    p.need(plan['node']=='B' and plan['model']=='14b' and plan['datasets']==['sharegpt'] and plan['scope']=='five_systems','migration scope changed')
    p.need(plan['expected_hostname']=='iZwz9i5bte3xkpmcoes3t2Z','migration physical host changed')
    contract.load_declaration(plan['declaration'])


def check_terminal(saved):
    p.need(saved.get('complete') and saved.get('finished_s') and not saved.get('error') and not saved.get('node_lease_held'),'predecessor not successful terminal')
    p.need(saved['node']=='B' and saved['model']=='32b' and saved['scope']=='five_systems','wrong predecessor model/node/scope')
    p.need(not p.active_owner(saved),'predecessor supervisor still running')
    last=p.checked(saved['last_cell_status'])
    p.need(last['complete'] and last['finished_s'] and not last.get('error') and not last.get('failed') and not last['node_lease_held'] and not p.active_owner(last),'predecessor measurement is not clean and exited')
    for dataset in contract.DATASETS:
        group=contract.resolve_group(saved['declaration'],'32b',dataset,actual_host='B')
        observations=[p.checked(r) for r in saved['observations'] if p.checked(r)['dataset']==dataset]
        decision=contract.select_group(group,observations)
        p.need(decision['phase']=='complete' and decision['five_system_complete'],'32B five-system scope incomplete')
    release=p.checked(last['release']);binding=p.checked(release['binding'])
    p.need(binding['model']=='32b' and binding['hostname']=='iZwz9i5bte3xkpmcoes3t2Z','predecessor binding is foreign')
    return release['binding']


def stopped(plan):return any(Path(x).exists() for x in plan['stop_paths'])


async def wait_predecessor(plan,state,out):
    path=Path(plan['predecessor_terminal_path'])
    while True:
        p.need(not stopped(plan),'stop requested while waiting for predecessor')
        if path.exists():
            saved=p.read(path)
            state['predecessor_progress']=dict(path=str(path),complete=saved.get('complete'),phase=saved.get('phase'),
                observations=len(saved.get('observations',[])),checked_s=time.time())
            if saved.get('finished_s') and saved.get('error'):
                raise ValueError('predecessor failed; automatic model switch forbidden')
            if saved.get('complete') and saved.get('finished_s') and not p.active_owner(saved):
                binding=check_terminal(saved)
                reference=p.ref(path);p.need(p.checked(reference)==saved,'terminal changed during audit')
                state.update(predecessor_terminal=reference,previous_binding=binding)
                p.save(out/'status.json',state);return binding
        p.save(out/'status.json',state)
        await asyncio.sleep(30)


async def child(argv,log,state,out):
    plan=p.checked(state['plan'])
    p.need(not stopped(plan),'stop requested before next stage')
    p.need(len(argv)>=3 and argv[0]==sys.executable and argv[1]=='-B','explicit Python child expected')
    argv=[*argv[:2],HERE/'stdlib_entry.py','--source',*argv[2:]]
    with Path(log).open('xb') as stream:
        proc=await asyncio.create_subprocess_exec(*map(str,argv),stdout=stream,stderr=asyncio.subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        state['child']=dict(pid=proc.pid,argv=list(map(str,argv)),started_s=time.time(),startticks=p.process_identity(proc.pid)['startticks'])
        p.save(out/'status.json',state)
        try:code=await proc.wait()
        except BaseException:
            if proc.returncode is None:proc.terminate();await proc.wait()
            raise
        state['child'].update(exitcode=code,finished_s=time.time());p.save(out/'status.json',state)
        p.need(code==0,'migration child failed; preserve evidence and stop')


def decision(state):
    return contract.select_group(contract.resolve_group(state['declaration'],'14b','sharegpt',actual_host='B'),[p.checked(r) for r in state['observations']])


async def cell(plan,state,out,row,handoff):
    p.need(not stopped(plan),'stop requested at cell boundary')
    name=f'{len(state["attempts"])+1:04d}-{row["system"]}-sharegpt-r{row["rate_rps_decimal"]}-rep{row["repeat"]}-{row.get("measurement_purpose","normal")}'
    attempt=out/name;attempt.mkdir()
    predecessors=list(handoff.get('predecessors',[]))
    if state.get('last_cell_status'):predecessors.append(state['last_cell_status'])
    kwargs=dict(declaration=state['declaration'],qualification=handoff['qualification'],qualification_validator=handoff['qualification_validator'],
        node='B',model='14b',dataset='sharegpt',rate=row['rate_rps_decimal'],system=row['system'],out=str(attempt/'release'),
        scheduling_observations=state['observations'],predecessors=predecessors,repeats=[row['repeat']],
        measurement_purpose=row.get('measurement_purpose','normal'),stop_paths=plan['stop_paths'],extra_files=[state['plan'],state['effective_plan'],state['stage_release'],p.ref(__file__)])
    p.save(attempt/'prepare-request.json',dict(rows=[row],kwargs=kwargs))
    state['attempts'].append(dict(name=name,cell_id=row['cell_id'],started_s=time.time()))
    state.update(phase=row['system'],current_dataset='sharegpt',current_rate_rps=row['rate_rps'])
    await child([sys.executable,'-B',COMMON/'prepare_request.py','--request',attempt/'prepare-request.json','--out',attempt/'release-reference.json'],attempt/'prepare.log',state,out)
    release=p.read(attempt/'release-reference.json')
    await child([sys.executable,'-B',COMMON/'run_cells.py','--release',release['path'],'--out',attempt/'measurement','--run'],attempt/'measurement.log',state,out)
    terminal=p.ref(attempt/'measurement/status.json');saved=p.checked(terminal)
    p.need(saved['complete'] and not saved.get('error') and not saved['failed'] and not saved['node_lease_held'] and not p.active_owner(saved),'measurement did not finish cleanly')
    state['observations'].extend(saved['observations']);state['last_cell_status']=terminal
    state['binding']=p.checked(release)['binding']
    state['attempts'][-1].update(complete=True,status=terminal,finished_s=time.time());p.save(out/'status.json',state)


async def meter_handoff(plan,state,out,system,binding,validator,predecessors):
    target=out/('meter-'+system)
    await child([sys.executable,'-B',plan['meter_binding']['path'],'--binding',binding['path'],'--out',target,'--native-validator',validator['path']],out/('meter-'+system+'.log'),state,out)
    handoff=dict(node='B',model='14b',system=system,qualification=p.ref(target/'qualified.json'),qualification_validator=plan['meter_binding'],predecessors=predecessors)
    p.save(out/('handoff-'+system+'.json'),handoff)
    state.setdefault('handoffs',{})[system]=p.ref(out/('handoff-'+system+'.json'))
    return handoff


async def wait_stage_release(plan,state,out):
    path=Path(plan['stage_release_path'])
    while not path.exists():
        p.need(not stopped(plan),'stop requested before qualification-stage release')
        state['phase']='awaiting_verified_baseline_stage_release'
        p.save(out/'status.json',state)
        await asyncio.sleep(30)
    reference=p.ref(path);stage=p.checked(reference)
    p.need(stage['schema']=='migration-B14B-five-system-stage-release-v1' and stage['node']=='B' and stage['model']=='14b'
           and stage['datasets']==['sharegpt'],'wrong baseline stage release')
    for name in ('baseline_producer','baseline_template','baseline_qualification','baseline_validator','profile','meter_binding'):
        ref=stage[name];p.need(p.sha(ref['path'])==ref['sha256'],'baseline stage source changed: '+name)
    for file,digest in stage['files'].items():p.need(p.sha(file)==digest,'baseline stage dependency changed: '+file)
    state['stage_release']=reference
    effective=dict(plan,**{k:stage[k] for k in ('baseline_producer','baseline_template','baseline_qualification','baseline_validator','profile','meter_binding')})
    p.save(out/'effective-plan.json',dict(effective,source_stage_release=reference))
    state['effective_plan']=p.ref(out/'effective-plan.json')
    p.save(out/'status.json',state)
    return effective


async def execute(plan,state,out):
    previous=await wait_predecessor(plan,state,out)
    plan=await wait_stage_release(plan,state,out)
    # The predecessor's own supervisor lock is acquired only after it has exited.
    # Each hardware child additionally acquires the original exclusive node lease.
    stage_lock=Path(plan['predecessor_supervisor_lock']).open('a+')
    fcntl.flock(stage_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        check_files(plan);p.need(not stopped(plan),'stop requested before restoration')
        state['phase']='restoring_14b_pdblend';p.save(out/'status.json',state)
        await child([sys.executable,'-B',HERE/'restore.py','--parent',plan['pdb_parent']['path'],'--previous',previous['path'],'--out',out/'pdb-restoration-001','--run'],out/'pdb-restoration.log',state,out)
        await child([sys.executable,'-B',HERE/'stages.py','fixed','--input',out/'pdb-restoration-001/bootstrap.json','--out',out/'fixed-spec-001'],out/'fixed-spec.log',state,out)
        state['phase']='fresh_82_shapes_and_cancellation'
        await child([sys.executable,'-B',HERE/'qualify_fixed.py','--spec',out/'fixed-spec-001/spec.json','--out',out/'fixed-qualification-001','--run'],out/'fixed-qualification.log',state,out)
        await child([sys.executable,'-B',HERE/'stages.py','idle','--input',out/'fixed-qualification-001/qualified.json','--out',out/'idle-spec-001'],out/'idle-spec.log',state,out)
        state['phase']='fresh_natural_idle_recovery'
        await child([sys.executable,'-B',HERE/'qualify_idle.py','--spec',out/'idle-spec-001/spec.json','--out',out/'idle-qualification-001','--run'],out/'idle-qualification.log',state,out)
        handoff=await meter_handoff(plan,state,out,'pdblend',p.ref(out/'idle-qualification-001/binding.json'),p.ref(HERE/'verify_binding.py'),[])
        while True:
            selected=decision(state);state['group_decisions']['sharegpt']=selected;p.save(out/'status.json',state)
            if selected.get('pdb_boundary_complete') and selected['phase'] in ('baselines','complete'):break
            if selected['phase']=='extension_declaration_required':
                target=out/('extension-'+str(len(state.get('extensions',[]))+1).zfill(3))
                request=dict(declaration=state['declaration'],next_rate=selected['next_rate_rps_decimal'],out=str(target))
                p.save(out/'extension-request.json',request)
                await child([sys.executable,'-B',HERE/'extend.py','--request',out/'extension-request.json'],out/('extend-'+target.name+'.log'),state,out)
                state.setdefault('extensions',[]).append(p.ref(target/'manifest.json'));state['declaration']=p.ref(target/'declaration.json');continue
            p.need(selected['phase']=='pdblend','PDB engineering failure or undeclared condition requires diagnosis')
            tasks=[t for t in selected['next_tasks'] if t['action']=='execute']
            p.need(tasks,'PDB phase has no executable next task')
            await cell(plan,state,out,tasks[0]['row'],handoff)
        snapshot=dict(state,scope='pdblend',complete=True,pdb_boundary_complete=True,finished_s=time.time())
        p.save(out/'pdb-boundary.json',snapshot);state['pdb_boundary']=p.ref(out/'pdb-boundary.json')
        state['phase']='creating_14b_baselines';p.save(out/'status.json',state)
        await child([sys.executable,'-B',plan['baseline_producer']['path'],'--spec',plan['baseline_template']['path'],'--predecessor-terminal',out/'pdb-boundary.json','--out',out/'baseline-deployment-001','--run'],out/'baseline-deployment.log',state,out)
        state['phase']='fresh_14b_baseline_qualification';p.save(out/'status.json',state)
        await child([sys.executable,'-B',plan['baseline_qualification']['path'],'--bootstrap',out/'baseline-deployment-001/bootstrap/binding.json','--out',out/'baseline-qualification-001','--node','B','--hostname',plan['expected_hostname'],'--profile',plan['profile']['path'],'--run'],out/'baseline-qualification.log',state,out)
        bindings=p.read(out/'baseline-qualification-001/bindings.json')
        for system in ('mixed','distserve','dynamollm','ecoserve'):
            handoff=await meter_handoff(plan,state,out,system,bindings[system],plan['baseline_validator'],[])
            while True:
                selected=decision(state);state['group_decisions']['sharegpt']=selected;p.save(out/'status.json',state)
                if selected['phase']=='complete':break
                p.need(selected['phase']=='baselines','PDB boundary no longer complete or engineering failure')
                tasks=[t for t in selected['baseline_tasks'] if t['action']=='execute' and t['row']['system']==system]
                if not tasks:break
                tasks.sort(key=lambda t:(t['row']['rate_rps'],t['row'].get('measurement_purpose','normal'),t['row']['repeat']))
                await cell(plan,state,out,tasks[0]['row'],handoff)
        final=decision(state);p.need(final['phase']=='complete' and final['five_system_complete'],'B14 SG five-system scope is incomplete')
        state['group_decisions']['sharegpt']=final;state.update(complete=True,phase='complete',five_system_complete=True)
    finally:stage_lock.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    plan_ref=p.ref(a.plan);plan=p.checked(plan_ref);check_files(plan)
    p.need(socket.gethostname()==plan['expected_hostname'],'wrong physical node')
    if not a.run:print('B14 ShareGPT CPU plan passed; no GPU work');return
    p.need(not a.out.exists(),'fresh pipeline output required');a.out.mkdir(parents=True)
    lock=Path(plan['supervisor_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(schema='uniform-v2-node-pipeline-status',plan=plan_ref,pid=os.getpid(),startticks=p.process_identity(os.getpid())['startticks'],
        started_s=time.time(),node='B',model='14b',datasets=['sharegpt'],scope='five_systems',declaration=plan['declaration'],
        complete=False,node_lease_held=False,phase='awaiting_32b_five_system_terminal',observations=[],attempts=[],group_decisions={})
    p.save(a.out/'status.json',state)
    async def controlled():
        task=asyncio.current_task()
        for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,task.cancel)
        await execute(plan,state,a.out)
    try:asyncio.run(controlled())
    except BaseException as e:state.update(error=repr(e),phase='stopped_failure');raise
    finally:state['finished_s']=time.time();p.save(a.out/'status.json',state);lock.close()

if __name__=='__main__':main()
