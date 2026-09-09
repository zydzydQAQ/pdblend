"""Append only the eleven original EcoServe SLO observations after new rates."""
import argparse,asyncio,copy,json,os,signal,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'execution-completion-p4-002'))
import runner as fixed
p=fixed.p
DECL=B/'historical-ecoserve-scale11-declaration-v2.json'
CHAIN=p.REPO/'campaign/main-rate-priority-v2/B/scale_chain.py'
OUT=B/'historical-ecoserve-scale11-002'

def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def contract():
 d=p.read(DECL)
 p.need(d['schema']=='B-historical-ecoserve-scale11-until-complete-v1' and d['deadline_s'] is None and d['campaign_lifecycle']=='until_declared_complete_v1','wrong historical scope/lifecycle')
 prefix=p.checked(d['original_prefix']);spec=p.checked(d['original_spec']);manifest=p.checked(spec['source'])
 eco=next(g for g in prefix['groups'] if g['system']=='ecoserve')
 expected={r['cell_id']:r for r in manifest['cells']}
 p.need(len(eco['pending_ids'])==11 and d['cells']==[expected[cid] for cid in eco['pending_ids']],'historical rows changed or duplicated')
 p.need(all(g['system']=='ecoserve' or not g['pending_ids'] for g in prefix['groups']),'other systems unexpectedly pending')
 for row in d['cells']:
  p.need(row['system']=='ecoserve' and row['model']=='32b' and row['phase']=='scale' and row['slo_scale'] in (.5,2) and row['seed']==701 and row['arrival_window_s']==100,'wrong frozen historical row')
  p.need(p.sha(row['trace_path'])==row['trace_sha256'],'historical trace changed')
 for path,digest in d['source_files'].items():p.need(p.sha(path)==digest,'historical execution source changed '+path)
 return d

def prerequisite():
 out=B/'continuation-completion-002';s=p.read(out/'status.json')
 p.need(s.get('complete') is True and not s.get('node_lease_held') and not alive(s['pid']),'new-rate baseline predecessor must finish and exit')
 p.need([x['phase'] for x in s['steps']]==['lb025_repeat2','restore','gate','qualify','baseline_repeat1','baseline_repeat2'] and all(x['exitcode']==0 for x in s['steps']),'baseline predecessor incomplete')
 refs=[p.ref(out/'status.json')]
 adjacent=p.read(B/'adjacent-control-p4-001/status.json')
 p.need(adjacent['complete'] and not adjacent['failed'] and not adjacent['node_lease_held'] and not alive(adjacent['pid']),'adjacent control failure/incomplete')
 refs.append(p.ref(B/'adjacent-control-p4-001/status.json'))
 for repeat in (1,2):
  q=B/('baselines-completion-r'+str(repeat)+'-001');state=p.read(q/'status.json')
  p.need(state.get('complete') is True and not state['failed'] and not state.get('node_lease_held') and not alive(state['pid']),'baseline repeat failed/incomplete')
  p.need(len(state['completed'])==state['declared'],'missing declared baseline cells')
  for cid in state['completed']:
   cp=p.read(q/'results/checkpoints'/(cid+'.json'))
   p.need(all(p.sha(k)==v for k,v in cp['artifacts'].items()),'prior baseline artifacts changed')
   p.need(p.read(q/'engineering-gates'/(cid+'.json'))['passed'],'prior baseline request/engineering failure')
  refs.append(p.ref(q/'status.json'))
 return refs

def compatible(c,old,new):
 for key in ('protocol_id','model','system','hostname','large_inputs'):
  p.need(old[key]==new[key],'historical policy contract changed '+key)
 p.need(new['deadline_s'] is None and new['campaign_lifecycle']=='until_declared_complete_v1','global deadline not removed')
 c.source_policy(old,new,['alpaca','sharegpt','longbench'])
 before={i['id']:i for i in old['instances']};after={i['id']:i for i in new['instances']}
 p.need(set(before)==set(after),'historical replica IDs changed')
 for key in before:
  left,right=c.core_instance(before[key]),c.core_instance(after[key]);left.pop('host_pid',None);right.pop('host_pid',None)
  p.need(left==right,'historical instance policy/source/layout changed')
 return True

async def execute(a,state):
 d=contract();previous=prerequisite()
 chain=fixed.load(CHAIN,'b_historical_scale_chain');c,prefix,spec,checked=chain.check_prefix(d['original_prefix'])
 eco=next(g for g in checked['groups'] if g['group']['system']=='ecoserve')
 p.need(eco['pending']==d['cells'],'fresh full historical audit differs')
 old=p.checked(eco['group']['scale_binding']);references=p.read(B/'baseline-bindings-completion.json');base=p.checked(references['ecoserve'])
 compatible(c,old,base)
 # Do not silently replay a completed cell or an uncheckpointed measured attempt.
 ids={r['cell_id'] for r in d['cells']}
 for root,dirs,files in os.walk(p.REPO/'campaign'):
  if Path(root).name in ('operations','cells','checkpoints'):
   present=set(dirs) if Path(root).name!='checkpoints' else {Path(f).stem for f in files}
   p.need(not ids.intersection(present),'historical cell already attempted elsewhere; reconcile before execution')
 common=fixed.runtime(base['host_release'])
 from ecopadg.serving.campaign import node_lease
 from ecopadg.measure.backends import PynvmlBackend
 import aiohttp
 p.need('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh exclusive lease required')
 p.need(not OUT.exists(),'new historical output required');OUT.mkdir()
 output=OUT/'results';output.mkdir();p.write(OUT/'predecessor.json',previous,exclusive=True);p.write(OUT/'declaration.json',d,exclusive=True)
 state.update(declared=11,remaining=[r['cell_id'] for r in d['cells']])
 def save():state.update(updated_s=time.time());p.write(OUT/'status.json',state)
 save()
 with node_lease():
  state['node_lease_held']=True;save()
  controller=fixed.load(B/'baseline_control_completion.py','b_historical_fresh_qualification');proof=controller.audit_performance('ecoserve',base)
  p.need(proof.get('passed') is True,'fresh Eco mechanism qualification failed')
  p.write(OUT/'qualification-audit.json',proof,exclusive=True)
  hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
  async with aiohttp.ClientSession(trust_env=False) as session:
   for row in d['cells']:
    if a.stop_requested or (B/'STOP-historical-scale11').exists():state.update(phase='stopped_at_boundary');break
    contract();binding=copy.deepcopy(base);binding.update(output=str(output),deadline_s=None,campaign_lifecycle='until_declared_complete_v1',formal_eligible=False)
    binding['files'].update({str(DECL):p.sha(DECL),str(Path(__file__).resolve()):p.sha(__file__),row['trace_path']:row['trace_sha256']})
    bp=OUT/'bindings'/(row['cell_id']+'.json');p.write(bp,binding,exclusive=True);common.validate_binding(binding)
    state.update(phase='running',current_cell=row['cell_id']);state['attempted'].append(row['cell_id']);save()
    rp=output/'operations'/row['cell_id']/'receipt.json';cp=output/'checkpoints'/(row['cell_id']+'.json')
    try:
     receipt=await common.run_one(session,binding,row,output,hardware)
     artifacts={str(f):p.sha(f) for directory in (rp.parent,output/'cells'/row['cell_id']) for f in directory.rglob('*') if f.is_file()}
     p.write(cp,dict(row=row,declaration=p.ref(DECL),binding=p.ref(bp),receipt=p.ref(rp),artifacts=artifacts,measurement_valid=receipt['measurement_valid'],work_complete=receipt['summary'].get('work_complete'),historical_scale_suffix=True,completed_s=time.time()),exclusive=True)
     state['completed'].append(row['cell_id']);gate=fixed.engineering_gate(receipt);p.write(OUT/'engineering-gates'/(row['cell_id']+'.json'),gate,exclusive=True)
     p.need(gate['passed'],'request/engineering failure stops historical suffix '+str(gate['errors']))
    except BaseException as exc:
     if rp.exists() and not cp.exists():
      receipt=p.read(rp);artifacts={str(f):p.sha(f) for directory in (rp.parent,output/'cells'/row['cell_id']) for f in directory.rglob('*') if f.is_file()}
      p.write(cp,dict(row=row,declaration=p.ref(DECL),binding=p.ref(bp),receipt=p.ref(rp),artifacts=artifacts,measurement_valid=receipt.get('measurement_valid') is True,work_complete=receipt.get('summary',{}).get('work_complete'),execution_error=repr(exc),failed_observation_preserved=True,historical_scale_suffix=True,completed_s=time.time()),exclusive=True)
     state['failed'].append(dict(cell_id=row['cell_id'],error=repr(exc)));raise
    finally:state['remaining']=[r['cell_id'] for r in d['cells'] if r['cell_id'] not in state['attempted']];save()
  state['complete']=len(state['completed'])==11
  if state['complete']:state['phase']='complete'
 state['node_lease_held']=False;save()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');a=parser.parse_args();a.stop_requested=False
 if not a.run:print(json.dumps(dict(cpu_contract_passed=True,declared=len(contract()['cells']),gpu_executed=False)));return
 p.need(not OUT.exists(),'new historical attempt required')
 def stop(*_):a.stop_requested=True
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 state=dict(schema=1,pid=os.getpid(),started_s=time.time(),phase='starting',complete=False,attempted=[],completed=[],failed=[],automatic_retries=False,historical_scale_suffix=True)
 try:asyncio.run(execute(a,state))
 except BaseException as exc:state.update(phase='failed',error=repr(exc));raise
 finally:
  state.update(finished_s=time.time(),node_lease_held=False);state.pop('current_cell',None);p.write(OUT/'status.json',state)

if __name__=='__main__':main()
