"""New-A uniform-v2: original dynamic PDB gate, one normal run, then declared baselines."""
import argparse,asyncio,fcntl,json,os,signal,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
def group_observations(state, dataset, *, pdb_only=True):
    result = []
    for reference in state['observations']:
        value = p.checked(reference)
        if value['dataset'] == dataset and (not pdb_only or value['system'] == 'pdblend'):
            result.append(reference)
    return result

def resolve_group(plan, state, contract, dataset):
    group = contract.resolve_group(state['declaration'], plan['model'], dataset, actual_host=plan['node'])
    dynamic = [p.checked(r) for r in plan.get('dynamic_reuse_observations', [])
               if p.checked(r)['dataset'] == dataset]
    return contract.apply_audited_reuse(group, dynamic) if dynamic else group

async def child_command(argv, log, state, status):
    with Path(log).open('xb') as stream:
        child = await asyncio.create_subprocess_exec(*argv, stdout=stream, stderr=asyncio.subprocess.STDOUT,
                                                     start_new_session=True)
        state['child'] = dict(pid=child.pid, argv=argv, started_s=time.time())
        p.save(status, state)
        try:
            code = await child.wait()
        except BaseException:
            if child.returncode is None:
                child.terminate()
                try:
                    await asyncio.wait_for(child.wait(), 130)
                except asyncio.TimeoutError:
                    # A still-live measurement is never treated as cleaned up.
                    state['child_cleanup_unresolved'] = True
            raise
        state['child'].update(exitcode=code, finished_s=time.time())
        p.save(status, state)
        p.need(code == 0, 'child failed; preserve its attempt and diagnose before any successor')

async def run_rate(plan, handoff, selected, dataset, state, out):
    p.need(not any(Path(path).exists() for path in plan['stop_paths']), 'stop requested at rate boundary')
    rows = [task['row'] for task in selected]
    rate, system = rows[0]['rate_rps'], rows[0]['system']
    rate_name = str(rate).replace('.', 'p')
    name = f'{len(state["attempts"])+1:04d}-{system}-{dataset}-r{rate_name}'
    attempt = out / name
    state['attempts'].append(dict(name=name, system=system, dataset=dataset, rate_rps=rate,
                                  cells=[r['cell_id'] for r in rows], started_s=time.time()))
    state.update(phase=system, current_dataset=dataset, current_rate_rps=rate)
    p.save(out / 'status.json', state)
    observations = group_observations(state, dataset, pdb_only=False)
    predecessors = list(handoff.get('predecessors', []))
    if state.get('last_cell_status'):
        predecessors.append(state['last_cell_status'])
    kwargs = dict(declaration=state['declaration'], qualification=handoff['qualification'],
        qualification_validator=handoff['qualification_validator'], node=plan['node'], model=plan['model'],
        dataset=dataset, rate=rate, system=system, out=str(attempt / 'release'),
        scheduling_observations=observations, predecessors=predecessors,
        repeats=tuple(r['repeat'] for r in rows), measurement_executor=handoff.get('measurement_executor'),
        extra_files=handoff.get('extra_files', []), stop_paths=plan['stop_paths'],
        measurement_purpose=rows[0].get('measurement_purpose','normal'))
    kwargs['dynamic_reuse_observations'] = [r for r in plan.get('dynamic_reuse_observations', [])
                                            if p.checked(r)['dataset'] == dataset]
    request = attempt / 'prepare-request.json'
    result = attempt / 'release-reference.json'
    p.save(request, dict(prepare_adapter=handoff.get('prepare_adapter'), rows=rows, kwargs=kwargs))
    await child_command([sys.executable, '-B', str(p.HERE / 'prepare_request.py'),
                         '--request', str(request), '--out', str(result)],
                         attempt / 'prepare.log', state, out / 'status.json')
    release = p.read(result)
    await child_command([sys.executable, '-B', str(p.HERE / 'run_cells.py'), '--release', release['path'],
                         '--out', str(attempt / 'measurement'), '--run'], attempt / 'measurement.log',
                         state, out / 'status.json')
    terminal = p.ref(attempt / 'measurement/status.json')
    saved = p.checked(terminal)
    p.need(saved['complete'] and not saved.get('error') and not saved['failed']
           and not saved['node_lease_held'] and saved['finished_s'] and not p.active_owner(saved),
           'cell stage did not terminate cleanly')
    state['observations'].extend(saved['observations'])
    state['last_cell_status'] = terminal
    state['attempts'][-1].update(finished_s=time.time(), status=terminal, complete=True)
    p.save(out / 'status.json', state)

def save(state,out,**kw):
 state.update(kw,updated_s=time.time());p.save(out/'status.json',state)

def stopping(plan):
 return any(Path(path).exists() for path in plan['stop_paths'])

async def pdb_handoff(plan,out,state):
 path=Path(plan['pdb_qualification']);qstatus=path.parent/'status.json'
 while not path.exists():
  p.need(not stopping(plan),'stop before PDB qualification')
  if qstatus.exists():p.need(p.read(qstatus).get('phase')!='stopped_failure','capacity qualification failed; preserve proof and diagnose')
  save(state,out,phase='awaiting_capacity_qualification',waiting_for=str(path));await asyncio.sleep(5)
 while p.active_owner(p.read(qstatus)):
  p.need(not stopping(plan),'stop before terminal qualification');await asyncio.sleep(1)
 q=p.ref(path);validator=p.ref(HERE/'formal-dynamic-v2/verify.py')
 handoff=dict(node=plan['node'],model=plan['model'],system='pdblend',qualification=q,qualification_validator=validator,
   prepare_adapter=p.ref(HERE/'formal-dynamic-v2/prepare_adapter.py'),measurement_executor=p.ref(HERE/'formal-dynamic-v2/dynamic_measurement.py'),per_cell_release=True,
   predecessors=[p.ref(qstatus)],extra_files=[p.ref(HERE/'formal-dynamic-v2/source-equivalence.json')])
 p.save(out/'pdb-handoff.json',handoff)
 await child_command([sys.executable,'-B',str(p.HERE/'validate_handoff.py'),'--handoff',str(out/'pdb-handoff.json'),'--out',str(out/'pdb-handoff-verification.json')],out/'pdb-handoff-verification.log',state,out/'status.json')
 state['pdb_handoff']=p.ref(out/'pdb-handoff.json');save(state,out)
 return handoff

async def baseline_handoff(plan,stage,out,state):
 ready=Path(stage['ready'])
 while not ready.exists():
  p.need(not stopping(plan),'stop at baseline readiness boundary')
  save(state,out,phase='awaiting_'+stage['name'],waiting_for=str(ready));await asyncio.sleep(5)
 ref=p.ref(ready);definition=p.checked(ref)
 for source in definition['sources']:p.need(p.sha(source['path'])==source['sha256'],'baseline producer source changed')
 save(state,out,phase='qualifying_'+stage['name'])
 await child_command(definition['argv'],out/('qualify-'+stage['name']+'.log'),state,out/'status.json')
 state.setdefault('baseline_stage_releases',{})[stage['name']]=ref;save(state,out)
 return definition['handoffs']

async def execute(plan,out,state):
 contract=p.load(plan['contract'],'newA_uniform2_contract');pdb=None
 for dataset in plan['dataset_order']:
  while True:
   group=resolve_group(plan,state,contract,dataset)
   decision=contract.select_group(group,[p.checked(r) for r in group_observations(state,dataset,pdb_only=False)])
   state.setdefault('group_decisions',{})[dataset]=decision;save(state,out)
   if decision.get('pdb_boundary_complete') or decision['phase'] in ('baselines','complete'):break
   p.need(decision['phase']!='diagnosis','engineering failure is not a capacity boundary')
   if decision['phase']=='extension_declaration_required':
    generator=p.load(plan['generator'],'newA_uniform2_generator');dest=out/('extension-'+dataset+'-'+decision['next_rate_rps_decimal'].replace('.','p'))
    generator.append_point(state['declaration'],plan['model'],dataset,decision['next_rate_rps_decimal'],dest)
    state['declaration']=p.ref(dest/'declaration.json');state.setdefault('extensions',[]).append(state['declaration']);continue
   p.need(decision['phase']=='pdblend','unknown PDB decision')
   if pdb is None:pdb=await pdb_handoff(plan,out,state)
   await run_rate(plan,pdb,decision['next_tasks'][:1],dataset,state,out)
  state.setdefault('completed_pdblend_datasets',[]).append(dataset);save(state,out)
 if pdb is None:
  previous=p.checked(plan['resume_pdblend_terminal'])
  p.need(previous['complete'] and not previous['node_lease_held'] and previous['node']==plan['node'] and previous['model']==plan['model'] and previous['declaration']==state['declaration'],'invalid completed PDB predecessor')
  p.need(all(r in state['observations'] for r in previous['observations']) and all(p.checked(r)['system']=='pdblend' for r in previous['observations']),'completed PDB observations not preserved')
  pdb=p.checked(previous['pdb_handoff']);state['pdb_handoff']=previous['pdb_handoff']
  verification=dict(binding=previous['binding'])
  state['resumed_pdblend_terminal']=plan['resume_pdblend_terminal'];save(state,out)
 else:verification=p.read(out/'pdb-handoff-verification.json')
 terminal=dict(schema='uniform-v2-new-A-pdblend-terminal-v1',complete=True,node_lease_held=False,finished_s=time.time(),node=plan['node'],model=plan['model'],qualification=pdb['qualification'],binding=verification['binding'],last_cell_status=state.get('last_cell_status'),observations=state['observations'],declaration=state['declaration'],pdb_handoff=state['pdb_handoff'])
 p.save(out/'pdblend-terminal.json',terminal);state['pdblend_terminal']=p.ref(out/'pdblend-terminal.json');save(state,out,phase='pdb_boundary_complete')
 for stage in plan['baseline_stages']:
  handoffs=None
  for dataset,system in stage['groups']:
   while True:
    group=resolve_group(plan,state,contract,dataset);decision=contract.select_group(group,[p.checked(r) for r in group_observations(state,dataset,pdb_only=False)])
    state['group_decisions'][dataset]=decision;save(state,out)
    p.need(decision['phase']!='diagnosis','baseline engineering evidence needs diagnosis')
    p.need(decision.get('pdb_boundary_complete') or decision['phase']=='complete','PDB boundary incomplete')
    p.need(decision['phase']!='metric_supplement_declaration_required','exact metric supplement needs an immutable new declaration')
    tasks=[t for t in decision.get('baseline_tasks',[]) if t['row']['system']==system]
    if not tasks:break
    if handoffs is None:handoffs=await baseline_handoff(plan,stage,out,state)
    path=Path(handoffs[dataset+':'+system]);handoff=p.checked(p.ref(path))
    await run_rate(plan,handoff,tasks[:1],dataset,state,out)
  state.setdefault('completed_baseline_stages',[]).append(stage['name']);save(state,out)
 for dataset in plan['dataset_order']:
  group=resolve_group(plan,state,contract,dataset);decision=contract.select_group(group,[p.checked(r) for r in group_observations(state,dataset,pdb_only=False)])
  p.need(decision['phase']=='complete','reached group still has unmeasured/missing metrics');state['group_decisions'][dataset]=decision
 save(state,out,phase='complete',complete=True)

def restore_completed_resident_stage(plan, state):
 prior_ref = plan.get('resume_resident_pipeline_terminal')
 if not prior_ref:
  return
 prior = p.checked(prior_ref)
 p.need(prior.get('finished_s') and not p.active_owner(prior) and not prior.get('node_lease_held'), 'resident predecessor still active')
 p.need(prior['node'] == plan['node'] and prior['model'] == plan['model'] and prior['declaration'] == state['declaration'], 'resident predecessor scope changed')
 p.need('8tp1' in prior.get('completed_baseline_stages', []) and prior['observations'] == state['observations'], 'completed resident observations changed')
 last_ref = prior['last_cell_status']; last = p.checked(last_ref)
 p.need(last['complete'] and last.get('finished_s') and not last.get('failed') and not last.get('node_lease_held') and not p.active_owner(last), 'resident last measurement not clean')
 p.need(last['observations'] and all(ref in state['observations'] for ref in last['observations']), 'resident final observations not preserved')
 state['last_cell_status'] = last_ref
 state['resumed_resident_pipeline_terminal'] = prior_ref

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args();plan=p.checked(p.ref(a.plan))
 p.need(plan['schema']=='new-A-uniform-v2-pipeline-plan' and plan['node']=='Anew20260909' and plan['model']=='14b' and plan['dataset_order']==['alpaca','longbench'],'wrong new-A work assignment')
 for path,sha in plan['files'].items():p.need(p.sha(path)==sha,'dispatcher source changed')
 p.load(plan['contract'],'newA_uniform2_preflight').load_declaration(plan['declaration'])
 if not a.run:print('CPU plan and source validation passed');return
 p.need(socket.gethostname()==plan['expected_hostname'] and not a.out.exists(),'wrong host/reused output')
 owner=Path(plan['supervisor_lock']).open('a+');fcntl.flock(owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);a.out.mkdir(parents=True)
 state=dict(schema='uniform-v2-node-pipeline-status',pid=os.getpid(),startticks=p.process_identity(os.getpid())['startticks'],started_s=time.time(),plan=p.ref(a.plan),declaration=plan['declaration'],node=plan['node'],model=plan['model'],complete=False,node_lease_held=False,phase='starting',attempts=[],observations=list(plan.get('initial_observations',[])))
 async def controlled():
  restore_completed_resident_stage(plan,state)
  task=asyncio.current_task()
  for sig in (signal.SIGTERM,signal.SIGINT):asyncio.get_running_loop().add_signal_handler(sig,task.cancel)
  await execute(plan,a.out,state)
 try:asyncio.run(controlled())
 except BaseException as exc:state.update(error=repr(exc),phase='stopped_failure');raise
 finally:state['finished_s']=time.time();p.save(a.out/'status.json',state);owner.close()
if __name__=='__main__':main()
