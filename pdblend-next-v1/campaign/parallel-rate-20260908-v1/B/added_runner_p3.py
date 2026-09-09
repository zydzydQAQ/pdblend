"""Single fresh lease for predeclared new-rate cells; no automatic replay."""
import argparse,asyncio,copy,importlib.util,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'execution-p3'))
import runner as fixed
p=fixed.p
DECL=ROOT/'added-rates-p3-001/declaration.json'

def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def predecessor():
 out=ROOT/'fixed-screen-p3-001';s=p.read(out/'status.json')
 p.need(s.get('complete') is True and len(s['completed'])==16 and not s['failed'] and not s.get('node_lease_held') and not alive(s['pid']),'complete clean fixed16 predecessor required')
 for cid in s['completed']:
  cp=p.read(out/'results/checkpoints'/(cid+'.json'))
  p.need(all(p.sha(k)==v for k,v in cp['artifacts'].items()),'predecessor raw changed')
  p.need(p.read(out/'engineering-gates'/(cid+'.json'))['passed'] is True,'engineering failure blocks new rate')
 return p.ref(out/'status.json')

def contract():
 d=p.read(DECL);p.need(d['schema']=='parallel-rate-added-five-system-B-p3-v1' and d['deadline_s']==p.DEADLINE,'wrong new-rate declaration')
 old=p.checked(d['source_declaration']);release,base,_=fixed.release_contract(d['runtime_release']['path'])
 p.need(p.sha(d['runtime_release']['path'])==d['runtime_release']['sha256'],'wrong runtime release')
 p.need(len(d['cells'])==30 and len({c['cell_id'] for c in d['cells']})==30,'exact30 new observations required')
 for a,b in zip(d['cells'],old['cells']):
  expected=copy.deepcopy(b);expected['cell_id']=expected['cell_id'].replace('parallel-rate-p1-added','parallel-rate-p3-added')
  p.need(a==expected,'new declaration changed measured work')
 for w in d['workloads']:
  rows=[c for c in d['cells'] if c['workload_id']==w['workload_id']]
  p.need(len(rows)==10 and {c['system'] for c in rows}=={'pdblend',*p.BASELINES},'all five systems need two repeats')
  p.need(p.sha(w['trace_path'])==w['trace_sha256'],'new trace changed')
  for c in rows:
   p.need(all(c[k]==w[k] for k in ('trace_sha256','trace_path','content_pairing_sha256','n_requests','expected_generated_tokens','slo','seed','arrival_window_s')),'five-system pairing differs')
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
 cells=[c for c in d['cells'] if (c['system']=='pdblend')==(a.system_group=='pdblend')]
 cells.sort(key=lambda c:(c['repeat'],c['sequence']))
 p.need(not a.out.exists(),'new output required');a.out.mkdir(parents=True)
 output=a.out/'results';output.mkdir()
 p.write(a.out/'declaration-order.json',cells,exclusive=True);p.write(a.out/'predecessor.json',previous,exclusive=True)
 state.update(declared=len(cells),remaining=[c['cell_id'] for c in cells])
 def save():state.update(updated_s=time.time());p.write(a.out/'status.json',state)
 save()
 with node_lease():
  state['node_lease_held']=True;hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
  if bases:
   qualifier=fixed.load(ROOT/'baseline_control_p3.py','baseline_p3_qualification')
   for system,binding in bases.items():qualifier.audit_performance(system,binding)
  async with aiohttp.ClientSession(trust_env=False) as session:
   for row in cells:
    if a.stop_requested or (ROOT/'STOP-added-p3').exists() or time.time()+400>=p.DEADLINE:
     state.update(phase='stopped_at_boundary',stopped_at_boundary=True);break
    contract()
    if bases:binding=copy.deepcopy(bases[row['system']])
    else:
     cell=dict(arm='fixed2',repeat=row['repeat'],cell_id=row['cell_id'],dataset=row['dataset'],original_cell_id=row['workload_id'],trace={'path':row['trace_path'],'sha256':row['trace_sha256']})
     binding=fixed.point_binding(base,release,cell,output,a.out)
    binding.update(output=str(output),deadline_s=p.DEADLINE,formal_eligible=False)
    binding['files'].update({str(DECL):p.sha(DECL),str(Path(__file__).resolve()):p.sha(__file__),row['trace_path']:row['trace_sha256']})
    bp=a.out/'bindings'/(row['cell_id']+'.json');p.write(bp,binding,exclusive=True);common.validate_binding(binding)
    state.update(phase='running',current_cell=row['cell_id']);state['attempted'].append(row['cell_id']);save()
    try:
     receipt=await common.run_one(session,binding,row,output,hardware)
     rp=output/'operations'/row['cell_id']/'receipt.json';artifacts={str(f):p.sha(f) for directory in (rp.parent,output/'cells'/row['cell_id']) for f in directory.rglob('*') if f.is_file()}
     p.write(output/'checkpoints'/(row['cell_id']+'.json'),dict(row=row,declaration=p.ref(DECL),binding=p.ref(bp),receipt=p.ref(rp),artifacts=artifacts,measurement_valid=receipt['measurement_valid'],work_complete=receipt['summary'].get('work_complete'),completed_s=time.time()),exclusive=True)
     state['completed'].append(row['cell_id']);gate=fixed.engineering_gate(receipt);p.write(a.out/'engineering-gates'/(row['cell_id']+'.json'),gate,exclusive=True)
     p.need(gate['passed'],'engineering failure stops expansion '+str(gate['errors']))
    except BaseException as exc:state['failed'].append(dict(cell_id=row['cell_id'],error=repr(exc)));raise
    finally:state['remaining']=[c['cell_id'] for c in cells if c['cell_id'] not in state['attempted']];save()
  state['complete']=len(state['completed'])==len(cells)
  if state['complete']:state['phase']='complete'
 state['node_lease_held']=False;save()

def main():
 q=argparse.ArgumentParser();q.add_argument('--system-group',choices=('pdblend','baselines'),required=True);q.add_argument('--bindings',type=Path);q.add_argument('--out',required=True,type=Path);q.add_argument('--run',action='store_true');a=q.parse_args();a.stop_requested=False
 if not a.run:contract();print(dict(cpu_passed=True,gpu_executed=False));return
 p.need(not a.out.exists(),'new attempt required')
 def stop(*_):a.stop_requested=True
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 state=dict(schema=1,pid=os.getpid(),started_s=time.time(),phase='starting',complete=False,attempted=[],completed=[],failed=[],automatic_retries=False)
 try:asyncio.run(execute(a,state))
 except BaseException as exc:state.update(phase='failed',error=repr(exc));raise
 finally:
  if a.out.exists():state.update(finished_s=time.time(),node_lease_held=False);state.pop('current_cell',None);p.write(a.out/'status.json',state)

if __name__=='__main__':main()
