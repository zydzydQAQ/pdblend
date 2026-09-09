"""One newly identified adjacent control repeat, after the frozen main selection."""
import asyncio,copy,os,sys,time,json
from pathlib import Path
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'execution-completion-p4-002'))
import runner as fixed
p=fixed.p
DECL=B/'lb025-repeat2-declaration.json'
OUT=B/'adjacent-control-p4-001'
def contract():
 d=p.read(DECL);release,base,old=fixed.release_contract(B/'completion-release-p4-001/release.json')
 original=next(c for c in old['cells'] if c['cell_id']==d['first_repeat_cell_id'])
 expected=copy.deepcopy(original);expected.update(repeat=2,cell_id=original['cell_id'].replace('-repeat1','-repeat2'),rate_selection='adjacent_control_repeat2')
 expected['source_row'].update(repeat=2,cell_id=expected['cell_id'])
 p.need(d['cell']==expected and d['schema']=='B-adjacent-LB025-second-repeat-v1','wrong adjacent control or changed trace/SLO')
 p.need(d['source_release']==p.ref(B/'completion-release-p4-001/release.json') and d['deadline_s'] is None,'source/profile/lifecycle changed')
 for name,digest in d['files'].items():p.need(p.sha(name)==digest,'adjacent source changed')
 return d,release,base
async def execute(state):
 d,release,base=contract();controller=fixed.load(B/'baseline_control_completion.py','lb025_predecessor');previous=controller.finished_pdb()
 first=B/'fixed-screen-p4-001/results/checkpoints'/(d['first_repeat_cell_id']+'.json');cp=p.read(first);r=p.read(cp['receipt'])
 p.need(fixed.engineering_gate(r)['passed'] and r['summary']['slo_attainment']>=.90,'adjacent first repeat not a complete passing control')
 cell=d['cell'];common=fixed.runtime(release['host_release'])
 from ecopadg.serving.campaign import node_lease
 from ecopadg.measure.backends import PynvmlBackend
 import aiohttp
 p.need('PDBLEND_NODE_LOCK_FD' not in os.environ and not OUT.exists(),'new independent adjacent-control stage required')
 OUT.mkdir();output=OUT/'results';output.mkdir();p.write(OUT/'declaration.json',d,exclusive=True);p.write(OUT/'predecessor.json',dict(main=previous,first_repeat=p.ref(first)),exclusive=True)
 def save():state.update(updated_s=time.time());p.write(OUT/'status.json',state)
 save()
 with node_lease():
  state['node_lease_held']=True;save();hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
  binding=fixed.point_binding(base,release,cell,output,OUT);binding['files'].update({str(DECL):p.sha(DECL),str(Path(__file__).resolve()):p.sha(__file__)})
  bp=OUT/'bindings'/(cell['cell_id']+'.json');p.write(bp,binding,exclusive=True);common.validate_binding(binding)
  row=copy.deepcopy(cell['source_row']);row.update(cell_id=cell['cell_id'],original_cell_id=cell['original_cell_id'],improvement_arm=cell['arm'],improvement_repeat=2)
  state.update(phase='running',current_cell=cell['cell_id']);state['attempted'].append(cell['cell_id']);save()
  rp=output/'operations'/cell['cell_id']/'receipt.json';cp=output/'checkpoints'/(cell['cell_id']+'.json')
  try:
   async with aiohttp.ClientSession(trust_env=False) as session:receipt=await common.run_one(session,binding,row,output,hardware)
   artifacts={str(f):p.sha(f) for directory in (rp.parent,output/'cells'/cell['cell_id']) for f in directory.rglob('*') if f.is_file()}
   p.write(cp,dict(row=row,declaration=cell,declaration_reference=p.ref(DECL),binding=str(bp),binding_sha256=p.sha(bp),receipt=str(rp),receipt_sha256=p.sha(rp),artifacts=artifacts,measurement_valid=receipt['measurement_valid'],work_complete=receipt['summary'].get('work_complete'),completed_s=time.time()),exclusive=True)
   gate=fixed.engineering_gate(receipt);p.write(OUT/'engineering-gates'/(cell['cell_id']+'.json'),gate,exclusive=True);state['completed'].append(cell['cell_id'])
   p.need(gate['passed'],'adjacent control request/engineering failure: '+str(gate['errors']))
  except BaseException as exc:
   if rp.exists() and not cp.exists():
    receipt=p.read(rp);artifacts={str(f):p.sha(f) for directory in (rp.parent,output/'cells'/cell['cell_id']) for f in directory.rglob('*') if f.is_file()}
    p.write(cp,dict(row=row,declaration=cell,declaration_reference=p.ref(DECL),binding=str(bp),binding_sha256=p.sha(bp),receipt=str(rp),receipt_sha256=p.sha(rp),artifacts=artifacts,measurement_valid=receipt.get('measurement_valid') is True,work_complete=receipt.get('summary',{}).get('work_complete'),execution_error=repr(exc),failed_observation_preserved=True,completed_s=time.time()),exclusive=True)
   state['failed'].append(dict(cell_id=cell['cell_id'],error=repr(exc)));raise
  state.update(phase='complete',complete=True)
 state['node_lease_held']=False;save()
def main():
 if '--run' not in sys.argv:contract();print(json.dumps(dict(cpu_contract=True,declared=1,gpu_executed=False)));return
 state=dict(schema=1,pid=os.getpid(),started_s=time.time(),phase='starting',complete=False,declared=1,attempted=[],completed=[],failed=[],automatic_retries=False)
 try:asyncio.run(execute(state))
 except BaseException as exc:state.update(phase='failed',error=repr(exc));raise
 finally:state.update(finished_s=time.time(),node_lease_held=False);state.pop('current_cell',None);p.write(OUT/'status.json',state)
if __name__=='__main__':main()
