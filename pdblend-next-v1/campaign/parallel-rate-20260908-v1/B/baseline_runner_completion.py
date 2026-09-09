"""Single fresh lease for predeclared new-rate cells; no automatic replay."""
import argparse,asyncio,copy,importlib.util,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'execution-completion-p4-002'))
import runner as fixed
p=fixed.p
DECL=ROOT/'completion-rates-p4-001/declaration.json'

def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def predecessor():
 controller=fixed.load(ROOT/'baseline_control_completion.py','b_completion_baseline_controller')
 return controller.finished_pdb()

def contract():
 d=p.read(DECL);p.need(d['schema']=='parallel-rate-completion-five-system-B-v1' and d['deadline_s'] is None and d['campaign_lifecycle']=='until_declared_complete_v1','wrong new-rate declaration')
 release,base,work=fixed.release_contract(ROOT/'completion-release-p4-001/release.json')
 p.need(work['five_system_declaration']==p.ref(DECL),'unfrozen five-system declaration')
 for w in d['workloads']:
  rows=[c for c in d['cells'] if c['workload_id']==w['workload_id']]
  p.need(len(rows)==10 and {c['system'] for c in rows}=={'pdblend',*p.BASELINES},'all five systems need two repeats')
  p.need(p.sha(w['trace_path'])==w['trace_sha256'],'new trace changed')
  for c in rows:p.need(all(c[k]==w[k] for k in ('trace_sha256','trace_path','content_pairing_sha256','n_requests','expected_generated_tokens','slo','seed','arrival_window_s')),'five-system pairing differs')
 status=p.read(ROOT/'fixed-screen-p4-001/status.json');selected={}
 for w in d['workloads']:
  if w.get('existing_original_rate'):continue
  candidate=[c for c in d['cells'] if c['workload_id']==w['workload_id'] and c['system']=='pdblend']
  if all(c['cell_id'] in status['completed'] for c in candidate):selected[w['workload_id']]=w
 d=dict(d);d['cells']=[c for c in d['cells'] if c['workload_id'] in selected]
 p.need(d['cells'],'no selected new-rate baselines need execution')
 return d,release,base

async def execute(a,state):
 d,release,base=contract();previous=predecessor()
 references=p.read(a.bindings) if a.bindings else None
 if a.system_group=='baselines':
  p.need(references and set(references)==set(p.BASELINES),'all four fresh baseline bindings required')
  bases={k:p.checked(v) for k,v in references.items()}
  host=next(iter(bases.values()))['host_release'];p.need(all(b['host_release']==host for b in bases.values()),'baseline runtime mismatch')
 else:host=release['host_release'];bases=None
 common=fixed.runtime(host)
 from ecopadg.serving.campaign import node_lease
 from ecopadg.measure.backends import PynvmlBackend
 import aiohttp
 p.need('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh lease required')
 cells=[c for c in d['cells'] if (c['system']=='pdblend')==(a.system_group=='pdblend') and c['repeat']==a.repeat]
 cells.sort(key=lambda c:(c['repeat'],c['sequence']))
 p.need(not a.out.exists(),'new output required');a.out.mkdir(parents=True)
 output=a.out/'results';output.mkdir()
 p.write(a.out/'declaration-order.json',cells,exclusive=True);p.write(a.out/'predecessor.json',previous,exclusive=True)
 state.update(declared=len(cells),full_five_system_declaration_count=len(d["cells"]),system_group=a.system_group,repeat=a.repeat,remaining=[c['cell_id'] for c in cells])
 def save():state.update(updated_s=time.time());p.write(a.out/'status.json',state)
 save()
 with node_lease():
  state['node_lease_held']=True;hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
  if bases:
   qualifier=fixed.load(ROOT/'baseline_control_completion.py','baseline_p4_qualification')
   for system,binding in bases.items():qualifier.audit_performance(system,binding)
  async with aiohttp.ClientSession(trust_env=False) as session:
   for row in cells:
    if a.stop_requested or (ROOT/'STOP-baselines-completion').exists():
     state.update(phase='stopped_at_boundary',stopped_at_boundary=True);break
    contract()
    if bases:binding=copy.deepcopy(bases[row['system']])
    else:
     cell=dict(arm='fixed2',repeat=row['repeat'],cell_id=row['cell_id'],dataset=row['dataset'],original_cell_id=row['workload_id'],trace={'path':row['trace_path'],'sha256':row['trace_sha256']})
     binding=fixed.point_binding(base,release,cell,output,a.out)
    binding.update(output=str(output),deadline_s=None,campaign_lifecycle='until_declared_complete_v1',formal_eligible=False)
    binding['files'].update({str(DECL):p.sha(DECL),str(Path(__file__).resolve()):p.sha(__file__),row['trace_path']:row['trace_sha256']})
    bp=a.out/'bindings'/(row['cell_id']+'.json');p.write(bp,binding,exclusive=True);common.validate_binding(binding)
    state.update(phase='running',current_cell=row['cell_id']);state['attempted'].append(row['cell_id']);save()
    try:
     receipt=await common.run_one(session,binding,row,output,hardware)
     rp=output/'operations'/row['cell_id']/'receipt.json';artifacts={str(f):p.sha(f) for directory in (rp.parent,output/'cells'/row['cell_id']) for f in directory.rglob('*') if f.is_file()}
     p.write(output/'checkpoints'/(row['cell_id']+'.json'),dict(row=row,declaration=p.ref(DECL),binding=p.ref(bp),receipt=p.ref(rp),artifacts=artifacts,measurement_valid=receipt['measurement_valid'],work_complete=receipt['summary'].get('work_complete'),completed_s=time.time()),exclusive=True)
     state['completed'].append(row['cell_id']);gate=fixed.engineering_gate(receipt);p.write(a.out/'engineering-gates'/(row['cell_id']+'.json'),gate,exclusive=True)
     p.need(gate['passed'],'engineering failure stops expansion '+str(gate['errors']))
    except BaseException as exc:
     rp=output/'operations'/row['cell_id']/'receipt.json';cp=output/'checkpoints'/(row['cell_id']+'.json')
     if rp.exists() and not cp.exists():
      actual=p.read(rp);artifacts={str(f):p.sha(f) for directory in (rp.parent,output/'cells'/row['cell_id']) for f in directory.rglob('*') if f.is_file()}
      p.write(cp,dict(row=row,declaration=p.ref(DECL),binding=p.ref(bp),receipt=p.ref(rp),artifacts=artifacts,measurement_valid=actual.get('measurement_valid') is True,work_complete=actual.get('summary',{}).get('work_complete'),execution_error=repr(exc),failed_observation_preserved=True,completed_s=time.time()),exclusive=True)
     state['failed'].append(dict(cell_id=row['cell_id'],error=repr(exc)));raise
    finally:state['remaining']=[c['cell_id'] for c in cells if c['cell_id'] not in state['attempted']];save()
  state['complete']=len(state['completed'])==len(cells)
  if state['complete']:state['phase']='complete'
 state['node_lease_held']=False;save()

def main():
 q=argparse.ArgumentParser();q.add_argument('--system-group',choices=('baselines',),required=True);q.add_argument('--repeat',type=int,choices=(1,2),required=True);q.add_argument('--bindings',type=Path);q.add_argument('--out',required=True,type=Path);q.add_argument('--run',action='store_true');a=q.parse_args();a.stop_requested=False
 if not a.run:
  d,_,_=contract();print(dict(cpu_passed=True,gpu_executed=False,selected_cells=sum((c['system']=='pdblend')==(a.system_group=='pdblend') and c['repeat']==a.repeat for c in d['cells'])));return
 p.need(not a.out.exists(),'new attempt required')
 def stop(*_):a.stop_requested=True
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 state=dict(schema=1,pid=os.getpid(),started_s=time.time(),phase='starting',complete=False,attempted=[],completed=[],failed=[],automatic_retries=False)
 try:asyncio.run(execute(a,state))
 except BaseException as exc:state.update(phase='failed',error=repr(exc));raise
 finally:
  if a.out.exists():state.update(finished_s=time.time(),node_lease_held=False);state.pop('current_cell',None);p.write(a.out/'status.json',state)

if __name__=='__main__':main()
