"""Read-only same-trace comparison projection; immutable raw points remain authoritative."""
import hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
SNAPSHOT=REPO/'campaign/five-system-results-v4/actual-snapshot-006/results.json'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def measured(cp):
 c=read(cp);rp=Path(c['receipt'])
 if not rp.exists():return None
 if sha(rp)!=c['receipt_sha256']:return None
 r=read(rp);b=read(c['binding']);sm=r['summary'];row=c['row']
 return dict(model=row['model'],dataset=row['dataset'],system=row['system'],rate_rps=row['rate_rps'],repeat=row.get('improvement_repeat',row.get('repeat',1)),
  seed=row['seed'],slo_ttft_s=row['slo_ttft_s'],slo_tpot_s=row['slo_tpot_s'],trace_sha256=row['trace_sha256'],content_pairing_sha256=row['content_pairing_sha256'],
  n_expected=sm['n_expected'],expected_generated_tokens=sm['expected_generated_tokens'],completed_work_requests=sm['completed_work_requests'],generated_tokens=sm['generated_tokens'],
  measurement_valid=r.get('measurement_valid') is True,work_complete=sm.get('work_complete') is True,failed_requests=sm.get('failed_requests'),request_timeouts=sm.get('request_timeouts'),
  slo_attainment=sm['slo_attainment'],energy_j=sm['energy_j'],good_requests=sm['good_requests'],energy_per_good_request_j=sm.get('energy_per_good_request_j'),
  receipt=str(rp),receipt_sha256=c['receipt_sha256'],checkpoint=str(cp),checkpoint_sha256=sha(cp),host_release=b['host_release'],
  clock_restore_complete=r.get('clock_restore_complete') is True,child_stopped=r.get('child_stopped') is True,outer_cleanup_errors=r.get('outer_cleanup_errors'))
def key(p):return tuple(p[k] for k in ('model','dataset','rate_rps','seed','trace_sha256','content_pairing_sha256','slo_ttft_s','slo_tpot_s','n_expected','expected_generated_tokens'))
def compare(c,b):
 assert key(c)==key(b),'not exact trace/work/SLO pair'
 target=min(.9,b['slo_attainment']);valid=c['measurement_valid'] and b['measurement_valid']
 cp=c['work_complete'] and c['failed_requests']==0 and c['request_timeouts']==0 and c['completed_work_requests']==c['n_expected'] and c['generated_tokens']==c['expected_generated_tokens']
 energy=c['energy_j']<b['energy_j'];slo=c['slo_attainment']>=target
 j_c=c['energy_j']/c['good_requests'] if c['good_requests'] else None
 j_b=b['energy_j']/b['good_requests'] if b['good_requests'] else None
 return dict(system=b['system'],paired=True,main_pass=bool(valid and cp and energy and slo),pdb_work_complete=cp,
  baseline_work_complete=b['work_complete'],slo_required=target,slo_pass=slo,energy_pass=energy,
  strict90_jgood_pass=bool(valid and cp and b['work_complete'] and c['slo_attainment']>=.9 and b['slo_attainment']>=.9 and energy and j_c is not None and j_b is not None and j_c<j_b),
  energy_saved_pct=100*(1-c['energy_j']/b['energy_j']),pdb_jgood=j_c,baseline_jgood=j_b,baseline_slo=b['slo_attainment'],baseline_energy_j=b['energy_j'],
  baseline_checkpoint=b.get('checkpoint',b.get('checkpoint_path')),baseline_receipt=b.get('receipt',b.get('receipt_path')))
def main():
 assert sha(SNAPSHOT)=='1ddd027859c026ebbea6dce212fe8369098de1505f18d489af2a8f0be3e3febb'
 old={(*key(x),x['system']):x for x in read(SNAPSHOT)['points'] if x['phase']=='main' and x['slo_scale']==1 and x['model']=='7b' and x['system']!='pdblend'}
 roots=[HERE/'screen-p4',HERE/'p4-completion']+[p.parent for p in HERE.glob('explore-p4-*/status.json')]
 candidates=[]
 for root in roots:
  for cp in (root/'results/checkpoints').glob('*.json'):
   try:v=measured(cp)
   except (OSError,KeyError,ValueError):continue
   if v is not None:
    assert v['host_release']==str(REPO/'campaign/parallel-rate-20260908-v1/hosts/7b-fixed-p4');candidates.append(v)
 new={}
 for cp in (HERE/'boundary-baselines-p4').glob('*/results/checkpoints/*.json'):
  try:v=measured(cp)
  except (OSError,KeyError,ValueError):continue
  if v is not None:new[(*key(v),v['system'],v['repeat'])]=v
 rows=[]
 for c in sorted(candidates,key=lambda x:(x['dataset'],x['rate_rps'],x['repeat'])):
  pairs=[]
  for system in ('mixed','distserve','dynamollm','ecoserve'):
   b=old.get((*key(c),system));source='snapshot006 exact-trace reuse'
   if b is None:b=new.get((*key(c),system,c['repeat']));source='new trace same repeat'
   if b is None:pairs.append(dict(system=system,paired=False,main_pass=None));continue
   p=compare(c,b);p['baseline_source']=source;pairs.append(p)
  rows.append(dict(**c,comparisons=pairs,main_baselines_passed=sum(p.get('main_pass') is True for p in pairs),paired_baselines=sum(p['paired'] for p in pairs)))
 out=dict(schema=1,captured_s=time.time(),serving_version='p4',profile_sha256='10594a2058edce72bd7673eb6d5fcaf7fa10b47af45413b934e86b9321c5356a',
  same_seed_repeats_not_independent_seeds=True,no_best_version_or_repeat_selection=True,raw_energy_reintegration_by_root_separate=True,
  main_acceptance='complete PDB work + E lower + SLO >= min(.90, baseline SLO)',strict90_jgood_is_supplementary=True,rows=rows)
 p=HERE/'comparison-p4-latest.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2)+'\n');t.replace(p)
 print(json.dumps(dict(path=str(p),pdb_runs=len(rows),unique_rate_cells=len({(r['dataset'],r['rate_rps']) for r in rows}),fully_paired_runs=sum(r['paired_baselines']==4 for r in rows))))
if __name__=='__main__':main()
