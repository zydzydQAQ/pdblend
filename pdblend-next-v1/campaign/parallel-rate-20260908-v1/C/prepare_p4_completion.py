"""Declare the remaining p4 background points and same-seed confirmation repeats."""
import copy,hashlib,importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 workflow=HERE/'workflow-p4-completion';assert not workflow.exists();workflow.mkdir()
 screen=read(HERE/'workflow-p4-strict/work-declaration.json')
 first=[c for c in screen['cells'] if c['repeat']==1]
 assert len(first)==6
 explore=[];stages=[dict(name='screen-p4',count=6,phase='stopped_at_boundary')]
 for p in sorted(HERE.glob('explore-p4-*-input/declaration.json')):
  d=read(p);out=HERE/p.parent.name.removesuffix('-input')
  assert (out/'status.json').exists(),'unexecuted declaration must not be selected'
  s=read(out/'status.json');assert s['complete'] and not s.get('engineering_gate_failed') and not s['failed']
  cp=next((out/'results/checkpoints').glob('*.json'));receipt=read(read(cp)['receipt']);sm=receipt['summary']
  assert sm['work_complete'] and sm['failed_requests']==0 and sm['request_timeouts']==0
  row=next(c for c in d['cells'] if c['system']=='pdblend')
  explore.append((d,row,ref(p),cp,sm));stages.append(dict(name=out.name,count=1,phase='complete'))
 bounds={}
 for ds in ('alpaca','sharegpt','longbench'):
  candidates=[]
  for cp in (HERE/'screen-p4/results/checkpoints').glob('*.json'):
   c=read(cp)
   if c['row']['dataset']==ds:candidates.append((c['row']['rate_rps'],read(c['receipt'])['summary']['slo_attainment'],str(cp)))
  candidates.extend((r['rate_rps'],sm['slo_attainment'],str(cp)) for d,r,dr,cp,sm in explore if r['dataset']==ds)
  miss=sorted(x for x in candidates if x[1]<.9);assert miss,'must find first complete SLO miss before completion queue'
  upper=miss[0][0];lower=max(x[0] for x in candidates if x[0]<upper and x[1]>=.9)
  assert all(x[0]<=upper for x in candidates),'post-boundary expansion prohibited'
  bounds[ds]=dict(last_pass_rate=lower,first_complete_miss_rate=upper,points=candidates)
 manifest=REPO/'campaign/five-system-fixed-window-v1/sources/C7B/manifest.json'
 original=read(manifest)['cells'];cells=[]
 for c in first:
  x=copy.deepcopy(c);x.update(repeat=2,cell_id=x['cell_id'].replace('repeat1','repeat2').replace('parallel-rate-p4-','parallel-rate-p4-completion-'));cells.append(x)
 for d,r,dr,cp,sm in explore:
  source=copy.deepcopy(r);source['cell_id']=source['cell_id'].replace('repeat1','repeat2')
  cells.append(dict(schema=1,model='7b',dataset=r['dataset'],arm='fixed2',repeat=2,stage='screen_fixed2',source_row=source,
   trace=dict(path=r['trace_path'],sha256=r['trace_sha256']),original_cell_id=r['workload_id']+'-pdblend-slo1',
   cell_id=source['cell_id'].replace('parallel-rate-p4-explore-','parallel-rate-p4-completion-'),new_rate=True))
 for r in sorted(original,key=lambda r:(r['dataset'],r['rate_rps'])):
  if r['system']!='pdblend' or r['phase']!='main' or r['slo_scale']!=1 or r['rate_rps']>bounds[r['dataset']]['first_complete_miss_rate']:continue
  if any(x['source_row']['dataset']==r['dataset'] and x['source_row']['rate_rps']==r['rate_rps'] for x in first):continue
  cells.append(dict(schema=1,model='7b',dataset=r['dataset'],arm='fixed2',repeat=1,stage='screen_fixed2',source_row=copy.deepcopy(r),
   trace=dict(path=r['trace_path'],sha256=r['trace_sha256']),original_cell_id=r['cell_id'],cell_id='parallel-rate-p4-completion-'+r['cell_id']+'-repeat1',background=True))
 assert len({x['cell_id'] for x in cells})==len(cells)
 d=dict(schema='parallel-rate-C-p4-completion-v1',created_s=time.time(),deadline_s=1788872770.0400891,
  original_manifest=ref(manifest),original_six_reused=True,critical_points_two_same_seed_runs=True,background_points_one_run=True,
  newly_explored_rates_two_same_seed_runs=True,predecessor_stages=stages,bounds=bounds,exploration_declarations=[dr for _,_,dr,_,_ in explore],
  cells=cells,seed=701,window_s=100,request_timeout_s=120,all8gpu_power=True,any_request_failure_stops_further_measurement=True)
 write(workflow/'work-declaration.json',d)
 template=HERE/'workflow-p4-strict'
 for name in ['protocol.py','prepare_release.py','operate.py','runner.py']:(workflow/name).write_bytes((template/name).read_bytes())
 p=workflow/'runner.py';s=p.read_text().replace(sha(template/'work-declaration.json'),sha(workflow/'work-declaration.json'))
 marker="    with node_lease():\n        state['node_lease_held'] = True"
 replacement="""    with node_lease():
        for pred in declaration['predecessor_stages']:
            root=p.ROOT.parent/pred['name'];st=p.read(root/'status.json')
            p.need(st['phase']==pred['phase'] and len(st['completed'])==pred['count'] and not st['failed']
                   and not st.get('engineering_gate_failed') and st['node_lease_held'] is False
                   and not Path('/proc/'+str(st['pid'])).exists(),'predecessor stage not cleanly terminal')
            for cp in (root/'results/checkpoints').glob('*.json'):
                c=p.read(cp);r=p.checked(dict(path=c['receipt'],sha256=c['receipt_sha256']));sm=r['summary']
                p.need(p.read(c['binding'])['host_release']==release['host_release'],'predecessor source differs')
                p.need(sm['work_complete'] and sm['failed_requests']==0 and sm['request_timeouts']==0
                       and r['measurement_valid'] and r['child_stopped'] and r['clock_restore_complete']
                       and not r['outer_cleanup_errors'],'predecessor work/cleanup failed')
                for file,h in c['artifacts'].items():p.need(p.sha(file)==h,'predecessor raw artifact changed')
        state['node_lease_held'] = True"""
 assert s.count(marker)==1;p.write_text(s.replace(marker,replacement))
 p=workflow/'prepare_release.py';s=p.read_text();marker="    release = dict(schema='main-slo-improvement-release-v1'"
 repl="""    work=p.read(p.ROOT/'work-declaration.json')
    refs=[work['original_manifest'],*work['exploration_declarations']]
    for dr in work['exploration_declarations']:
        paired=p.checked(dr);refs += [*paired['generators'],paired['source_spec'],paired['cells'][0]['source_300s_trace']]
    for reference in refs:
        p.need(p.sha(reference['path'])==reference['sha256'],'completion provenance changed')
        files[reference['path']]=reference['sha256']
    release = dict(schema='main-slo-improvement-release-v1'"""
 assert s.count(marker)==1;p.write_text(s.replace(marker,repl))
 spec=importlib.util.spec_from_file_location('adapt_completion',HERE/'adapt_workflow_until_complete.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.adapt(workflow)
 print(json.dumps(dict(workflow=str(workflow),cells=len(cells),background=sum(c.get('background',False) for c in cells),bounds=bounds)))
if __name__=='__main__':main()
