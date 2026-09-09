"""Freeze all new-rate baseline pairs; GPU restoration remains a separate stage."""
import copy,hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 decl=HERE/'boundary-p4/declaration.json';assert not decl.exists()
 complete=read(HERE/'workflow-p4-completion/work-declaration.json')
 stages=copy.deepcopy(complete['predecessor_stages']);stages.append(dict(name='p4-completion',count=len(complete['cells']),phase='complete'))
 cells=[];files={};refs=[]
 for dr in complete['exploration_declarations']:
  assert sha(dr['path'])==dr['sha256'];d=read(dr['path']);refs.append(dr);files[dr['path']]=dr['sha256']
  for reference in [*d['generators'],d['source_spec'],d['cells'][0]['source_300s_trace']]:
   assert sha(reference['path'])==reference['sha256'];files[reference['path']]=reference['sha256']
  for repeat in (1,2):
   for cell in d['cells']:
    if cell['system']=='pdblend':continue
    row=copy.deepcopy(cell);row.update(repeat=repeat,sequence=len(cells)+1)
    row['cell_id']=row['cell_id'].replace('repeat1','repeat'+str(repeat));files[row['trace_path']]=row['trace_sha256'];cells.append(row)
 for p in [HERE/'p4-completion-release/release.json',HERE/'workflow-p4-completion/work-declaration.json']:
  files[str(p)]=sha(p)
 assert len(cells)==24 and len({c['cell_id'] for c in cells})==len(cells)
 write(decl,dict(schema='parallel-rate-C-all-new-rate-pairs-p4-v1',created_s=time.time(),deadline_s=None,campaign_lifecycle='until_declared_complete_v1',
  cells=cells,files=files,pdblend_stages=stages,bounds=complete['bounds'],exploration_declarations=refs,
  model='7b',new_rates_all_paired=True,repetitions=2,seed=701,window_s=100,request_timeout_s=120,all8gpu_power=True,
  historical_baselines_byte_unchanged=True,original_trace_baselines_reused=True,any_failed_or_timed_out_request_stops_node_successors=True,
  complete_low_slo_capacity_points_retained=True,main_acceptance='PDB complete work; E lower; PDB SLO >= min(0.90, baseline SLO)',
  strict90_and_jgood_comparison_supplementary=True,serving_host=str(REPO/'campaign/parallel-rate-20260908-v1/hosts/7b-fixed-p4')))
 print(json.dumps(dict(declaration=str(decl),sha256=sha(decl),baseline_cells=len(cells),predecessor_pdb_cells=sum(x['count'] for x in stages))))
if __name__=='__main__':main()
