"""Read-only original A/B Eco60 window/failed-request audit, runnable on B CPU."""
import ast,copy,csv,hashlib,json,os,socket,statistics,time
from pathlib import Path
from types import SimpleNamespace as NS
R=Path('/root/workspace/pdblend-next-v1'); B=R/'campaign/parallel-rate-20260908-v1/B'
INPUT=B/'original-AB-Eco60-inputs-v1.json'; OUT=B/'original-AB-Eco60-window-audit-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def pct(values,q):
 a=sorted(values); i=(len(a)-1)*q;lo=int(i);return a[lo]+(a[min(lo+1,len(a)-1)]-a[lo])*(i-lo)
def mean_util(power,gpus,start,end):
 total=0
 for a,b in zip(power,power[1:]):
  t0=float(a['t_s']);t1=float(b['t_s']); l=max(t0,start);h=min(t1,end)
  if h<=l:continue
  v0=sum(float(a[f'gpu{g}_util_pct']) for g in gpus)/len(gpus);v1=sum(float(b[f'gpu{g}_util_pct']) for g in gpus)/len(gpus)
  vl=v0+(v1-v0)*(l-t0)/(t1-t0);vh=v0+(v1-v0)*(h-t0)/(t1-t0);total+=(vl+vh)*(h-l)/2
 assert float(power[0]['t_s'])<=start<end<=float(power[-1]['t_s'])
 return total/(end-start)
def scheduler(source,config):
 tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EcoServeScheduler');ns={};exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
 ids=[i['id'] for i in config['instances']][:config.get('eco_initial_instances',len(config['instances']))]
 return ns['EcoServeScheduler'](None,ids,lower=config.get('eco_macro_lower',2),upper=config.get('eco_macro_upper',3))
def audit(p):
 cp=read(p['checkpoint_path']); cell=Path(p['checkpoint_path']).parents[1]/'cells'/p['cell_id']; cfg=read(cell/'runtime_config.json');summary=read(cell/'summary.json');bench=rows(cell/'bench.csv');power=rows(cell/'power.csv')
 controls=[json.loads(l) for l in (cell/'control.jsonl').read_text().splitlines()];source=Path(cfg['controller_source_release'])/'src/ecopadg/serving/ecoserve.py';sch=scheduler(source,cfg)
 wanted=[cell/n for n in ('runtime_config.json','summary.json','bench.csv','control.jsonl','power.csv','cleanup.json')]
 for f in wanted:assert cp['artifacts'][str(f)]==sha(f),(p['cell_id'],str(f),'raw changed')
 assert len(bench)==p['n_expected'];assert sum(r['success']!='1' for r in bench)==p['failed_requests']
 events={};windows={i['id']:[] for i in cfg['instances']};member=0
 for ix,c in enumerate(controls):
  if 'request_id' in c:events.setdefault(c['request_id'],[]).append((ix,c))
  if c['kind']=='admission':
   plan=c['plan'];target=plan['routes'][0]['decode_id'];sch.committed(NS(routes=(NS(decode_id=target),)),c['at_s'])
   for w in plan.get('windows',[]):windows[w['instance_id']].append(dict(index=ix,at_s=c['at_s'],on=w['admit_prefill'],origin='raw_admission',client_request_id=c['client_request_id']))
  elif c['kind']=='eco_macro_membership':
   before=[tuple(g) for g in c['before']];after=[tuple(g) for g in c['after']];assert sch.groups==before,(p['cell_id'],'membership before mismatch')
   old=set(sum(before,()));new=set(sum(after,()));added=new-old;removed=old-new
   assert len(added)+len(removed)==1,(p['cell_id'],'membership delta')
   if added:sch.add_instance(next(iter(added)))
   else:
    rid=next(iter(removed)); states=[NS(instance_id=i['id'],requests=() if i['id']==rid else ('busy',),running=0,waiting=0,reserved_kv_tokens=0) for i in cfg['instances']]
    assert sch.remove_idle_instance(NS(instances=states))==rid
   assert sch.groups==after and sch.version==c['version'],(p['cell_id'],'exact member replay mismatch')
   chosen={sch.selected.get(g,g[0]) for g in sch.groups}
   for iid in windows:windows[iid].append(dict(index=ix,at_s=c['at_s'],on=iid in chosen,origin='exact_original_membership_method_replay',client_request_id=None))
   member+=1
 adm={c['client_request_id']:(ix,c) for ix,c in enumerate(controls) if c['kind']=='admission'}
 bad=[]
 for r in bench:
  if r['success']=='1':continue
  item={k:r.get(k) for k in ('idx','error','n_text_chunks','first_token_s','actual_dispatch_s','planned_arrival_s','request_deadline_s','finish_s')};pair=adm.get(r['idx']);item['admitted']=pair is not None
  if pair:
   ix,a=pair;rid=a['request_id'];es=events[rid]; end=next((j for j,v in es if v['kind']=='request_end'),len(controls));no_first=not r['first_token_s'] and not any(v['kind']=='first_token' for j,v in es)
   target=a['plan']['routes'][0]['prefill_id']; ws=[w for w in windows[target] if ix<w['index']<end];off=next((w for k,w in enumerate(ws) if w['on'] is False and not any(z['on'] for z in ws[k+1:])),None)
   t=next((v for j,v in es if v['kind']=='request_timing'),{});ids=next(i['gpus'] for i in cfg['instances'] if i['id']==target);end_s=float(r['finish_s']);start=max(float(summary['measurement_start_s']),end_s-30)
   item.update(server_request_id=rid,target_instance=target,target_gpus=ids,admission=a,request_timing=t,no_client_or_server_first_token=no_first,window_events_before_request_end=ws,final_off_without_reopen=off,window_stranding_demonstrated=bool(no_first and off),target_gpu_final30_util_pct=mean_util(power,ids,start,end_s),all8_final30_util_pct=mean_util(power,list(range(8)),start,end_s))
  bad.append(item)
 lat=[float(r['actual_dispatch_s'])-float(r['planned_arrival_s']) for r in bench]; confirmed=[r for r in bad if r.get('window_stranding_demonstrated')]
 return dict(cell_id=p['cell_id'],model=p['model'],dataset=p['dataset'],rate_rps=p['rate_rps'],checkpoint=ref(p['checkpoint_path']),receipt=ref(p['receipt_path']),source=p['executed_source'],raw_files={str(f):sha(f) for f in wanted},original_scheduler=ref(source),n_expected=p['n_expected'],failed_requests=p['failed_requests'],request_timeouts=p['request_timeouts'],energy_j=p['energy_j'],slo_attainment=p['slo_attainment'],work_complete=p['work_complete'],actual_dispatch_lateness_max_s=max(lat),actual_dispatch_lateness_p99_s=pct(lat,.99),membership_exact_replayed_count=member,failed_request_evidence=bad,demonstrated_window_stranding_count=len(confirmed),scientific_quarantine_recommended=bool(confirmed),raw_energy_preserved=True)
def main():
 inputs=read(INPUT)
 for path,digest in inputs['files'].items():assert sha(path)==digest,('input changed',path)
 assert not OUT.exists(),'immutable output exists'
 result=[]
 print(json.dumps(dict(pid=os.getpid(),host=socket.gethostname(),cwd=os.getcwd(),started_s=time.time(),read_only=True)),flush=True)
 for p in inputs['points']:
  result.append(audit(p)); print(json.dumps({k:result[-1][k] for k in ('cell_id','failed_requests','demonstrated_window_stranding_count')}),flush=True)
 d=dict(schema='original-AB-Eco60-window-liveness-readonly-audit-v1',created_s=time.time(),host=socket.gethostname(),pid=os.getpid(),source=ref(Path(__file__).resolve()),inputs=ref(INPUT),points=result,point_count=len(result),original_720_unchanged=True,raw_energy_preserved=True,quarantined_cells=[x['cell_id'] for x in result if x['scientific_quarantine_recommended']],limitations=['Membership journal does not directly log window actions; these are reconstructed by exact frozen EcoServeScheduler add/remove/committed methods and every before/after group/version is required to match. Admission OFF actions are raw observations.','No GPU work or policy mutation. Failed requests without demonstrated OFF/no-reopen remain unclassified by this window audit.','Counterfactual fixed-version performance requires an independently declared actual rerun; no quantitative outcome is inferred from this audit.'])
 with OUT.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
 print(json.dumps(dict(output=ref(OUT),point_count=len(result),quarantined_cells=d['quarantined_cells'])),flush=True)
if __name__=='__main__':main()
