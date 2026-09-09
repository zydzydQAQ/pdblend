"""Declare the affected matched pair of replacement idle observations without changing P6 business control."""
from pathlib import Path
import json,hashlib,copy,time
A=Path(__file__).resolve().parent;ROOT=A.parent;CODE=A/'load-p6-idle-code-001';OUT=A/'load-p6-idle-inputs-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def need(v,m):
 if not v:raise RuntimeError(m)
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);need(not p.exists(),'new immutable file required');p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 need(not OUT.exists(),'new idle supplement output required')
 old=A/'load-p6-full-002';status=read(old/'status.json');need(status['complete'] and status['cleanup_complete'] and len(status['completed'])==27 and not status.get('error') and not status.get('cleanup_errors'),'full27 business/cleanup must finish')
 need(not Path('/proc',str(status['pid'])).exists(),'physical predecessor must exit')
 sp=read(A/'load-p6-full-inputs-002/spec.json');cap=read(sp['capacity_binding']['path']);base=read(sp['original_binding']['path'])
 for name,digest in cap['calibrated_source_semantics']['capacity_modules'].items():need(sha(CODE/name)==digest,'P6 controller/capacity module may not change for idle observation repair')
 cpu=read(CODE/'cpu-validation.json');need(cpu['passed'],'idle observer CPU proof required')
 need(sha(CODE/'capacity_load_calibrate.py')==cpu['measurement_source']['sha256'],'idle observer differs from tested source')
 need(all(sha(p)==h for p,h in read(CODE/'manifest.json')['files'].items()),'idle frozen source manifest changed')
 for p,h in cpu.get('files',{}).items():need(sha(p)==h,'idle CPU source changed')
 cap.update(owner_id='ap6idle',http_port_base=31400,kv_port_base=62000,max_creations=1,runtime_dir=str(A/'load-p6-idle-001/runtime'))
 for p in [*CODE.glob('*.py'),CODE/'manifest.json',CODE/'cpu-validation.json',Path(__file__).resolve()]:cap['files'][str(p)]=sha(p)
 cap['files'][sp['profiles']['path']]=sp['profiles']['sha256']
 write(OUT/'capacity-binding.json',cap);capref=ref(OUT/'capacity-binding.json')
 base['files'].update(cap['files']);base['files'][capref['path']]=capref['sha256'];write(OUT/'execution-binding.json',base)
 oldsp=ref(A/'load-p6-full-inputs-002/spec.json');selection=dict(schema='P6-calibration-component-selection-v1',created_s=time.time(),controller_manifest=cap['calibrated_source_semantics']['candidate_manifest'],profile=sp['profiles'],unchanged_identity=cap['identity'],retained_business_results=[],retained_cold_and_remove_results=[],retained_idle_results=[],unselected_original_pair=[],rejected_idle_results=[],replacement_idle_results=[],reason='cycle2-layout3 has one native sample gap1.039372s above unchanged1s threshold; all declared60s endpoints were bracketed and other5idle qualify; replace only entire affected matched pair, retain cycles1/3; no business control change')
 for n in (1,2,3):
  for key,layout in [('low',2),('low40',2),('high2',2),('under_load','2to3'),('high3',3),('low',3),('low40',3)]:selection['retained_business_results'].append(ref(old/f'cycle-{n}-{key}-layout{layout}/result.json'))
  selection['retained_cold_and_remove_results'] += [ref(old/f'cycle-{n}-under_load-layout2to3/result.json'),ref(old/f'cycle-{n}-remove.json')]
  for layout in (2,3):
   r=ref(old/f'cycle-{n}-idle-layout{layout}/result.json')
   if n in (1,3):selection['retained_idle_results'].append(r)
   else:
    selection['unselected_original_pair'].append(r)
    if layout==3:selection['rejected_idle_results'].append(r)
    selection['replacement_idle_results'].append(str(A/f'load-p6-idle-001/cycle-1-idle-layout{layout}/result.json'))
 selection['retained_source_spec']=oldsp;selection['retained_status']=ref(old/'status.json');selection['retained_inventory']=ref(old/'inventory.json')
 selection['sources']=dict(full=dict(output=str(old),spec_ref=oldsp),idle=dict(output=str(A/'load-p6-idle-001'),spec_path=str(OUT/'spec.json')))
 selection['business_results']=selection['retained_business_results']
 selection['excluded_idle_pairs']=[selection['unselected_original_pair']]
 selection['idle_pairs']=[dict(source=ref(old/f'cycle-{n}-idle-layout3/result.json'),target=ref(old/f'cycle-{n}-idle-layout2/result.json'),source_owner='full',target_owner='full') for n in (1,3)]
 selection['idle_pairs'].append(dict(source=dict(path=str(A/'load-p6-idle-001/cycle-1-idle-layout3/result.json')),target=dict(path=str(A/'load-p6-idle-001/cycle-1-idle-layout2/result.json')),source_owner='idle',target_owner='idle',planned=True,replaces_original_cycle=2))
 write(OUT/'selection.json',selection)
 cycles=[]
 for n in (2,):
  cycle={}
  for operation in ('restore','remove'):
   action=read(sp['cycles'][n-1][operation]['path']);action['purpose']='P6 unchanged-control replacement idle observation only; retain actual cold/removal energy separately';p=OUT/f'cycle-{n}-{operation}.json';write(p,action);cycle[operation]=ref(p)
  cycles.append(cycle)
 sp.update(mode='idle_calibration',cycles=cycles,matched_idle_duration_s=60.,original_binding=ref(OUT/'execution-binding.json'),capacity_binding=capref,stop_path=str(A/'STOP_P6_IDLE_SUPPLEMENT'),replacement_selection=ref(OUT/'selection.json'))
 sp['files'].update(base['files'])
 for p in OUT.glob('*.json'):sp['files'][str(p)]=sha(p)
 for cycle in cycles:
  for r in cycle.values():sp['files'][r['path']]=r['sha256']
 write(OUT/'spec.json',sp);print(json.dumps(ref(OUT/'spec.json')))
if __name__=='__main__':main()
