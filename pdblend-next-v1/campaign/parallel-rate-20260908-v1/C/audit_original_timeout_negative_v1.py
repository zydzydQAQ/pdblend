"""Read-only diagnosis of a frozen baseline's original 120-second capacity timeout."""
import csv,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'main-slo-improvement-v1'))
import protocol as p
import final_selected_baseline_v2 as baseline

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(path),sha256=sha(path))
def require(ok,message):
 if not ok:raise ValueError(message)
def validate_timeout_rows(rows):
 failed=[r for r in rows if r['success']!='1'];require(bool(failed),'not a capacity negative')
 lateness=[];overshoot=[];partial=0;chunks=0
 for row in rows:
  planned=float(row['planned_arrival_s']);actual=float(row['actual_dispatch_s']);deadline=float(row['request_deadline_s'])
  require(all(math.isfinite(x) for x in (planned,actual,deadline)) and abs(deadline-planned-120)<.00001,'original 120s request budget changed')
  require(row['open_loop_independent']=='True' and -.00001<=actual-planned<=1.,'arrival dispatch engineering failure')
  lateness.append(actual-planned)
 for row in failed:
  require(row['error']=='request_hard_timeout' and row['request_timeout']=='True' and not row['admission_rejection'] and row['http_status'] in ('','200'),'non-timeout failure requires independent diagnosis')
  elapsed=float(row['finish_s'])-float(row['request_deadline_s']);require(math.isfinite(elapsed) and 0<=elapsed<=1.,'timeout does not coincide with original deadline')
  require(row['token_count_source']=='missing' and row['token_ids_verified']=='0','failed token accounting requires separate review')
  overshoot.append(elapsed);chunks+=int(row['n_text_chunks']);partial+=int(row['n_text_chunks'])>0
 ordered=sorted(lateness);q=(len(ordered)-1)*.99;k=int(q);p99=ordered[k]+(ordered[min(k+1,len(ordered)-1)]-ordered[k])*(q-k)
 require(p99<=.1,'arrival p99 indicates engineering lateness')
 return dict(failed_requests=len(failed),request_timeouts=len(failed),dispatch_lateness_max_s=max(lateness),dispatch_lateness_p99_s=p99,timeout_deadline_overshoot_max_s=max(overshoot),unverified_partial_output_requests=partial,partial_text_chunks=chunks,generated_token_count_complete=False,generated_tokens_semantics='recorded verified output tokens are a lower bound; partial chunks are not tokens')

def audit(cp_path):
 cp=read(cp_path);row=cp['row'];binding=read(cp['binding']);receipt=read(cp['receipt']);summary=receipt['summary']
 files=dict(binding['files']);files.update(cp['artifacts']);files.update({str(cp_path):sha(cp_path),cp['binding']:cp['binding_sha256'],cp['receipt']:cp['receipt_sha256'],str(Path(__file__).resolve()):sha(__file__)})
 for path,h in files.items():require(sha(path)==h,'frozen evidence changed '+path)
 require(summary['measurement_valid'] and summary['fixed_window_valid'] and receipt['measurement_valid'] and summary['runtime_error'] is None,'measurement invalid')
 require(not summary['work_complete'] and summary['admission_rejections']==0 and summary['gpu_count']==8,'different negative case')
 require(receipt['child_stopped'] and receipt['clock_restore_complete'] and not receipt['outer_cleanup_errors'] and all(v['complete'] for v in receipt['restoration'].values()),'cleanup or native restoration invalid')
 directory=Path(cp['receipt']).parents[2]/'cells'/row['cell_id'];bench=list(csv.DictReader((directory/'bench.csv').open()))
 diagnosis=validate_timeout_rows(bench)
 require(diagnosis['failed_requests']==summary['failed_requests']==summary['request_timeouts'],'timeout summary count differs')
 controls=[json.loads(line) for line in (directory/'control.jsonl').read_text().splitlines()]
 timings={v['client_request_id']:v for v in controls if v.get('kind')=='request_timing'}
 require(set(timings)=={r['request_id'] for r in bench},'controller timing coverage differs')
 handler_delays=[]
 for r in bench:
  t=timings[r['request_id']];require(t['planned_arrival_s']==float(r['planned_arrival_s']) and t['actual_dispatch_s']==float(r['actual_dispatch_s']) and t['hard_deadline_s']==float(r['request_deadline_s']),'controller/bench original timing mismatch')
  handler_delays.append(t['handler_arrival_s']-t['actual_dispatch_s']);require(0<=handler_delays[-1]<=1.,'controller handler starvation')
  require(t['cleanup_end_s']<=float(r['request_deadline_s'])+2.,'request cleanup overran original deadline')
 stamps=sorted(v['at_s'] for v in controls if 'at_s' in v);maxgap=max(z-a for a,z in zip(stamps,stamps[1:]))
 require(maxgap<10.,'controller log starvation needs separate diagnosis')
 identity=baseline.actual_identity(p,cp,binding,cp['receipt'])
 spec=importlib.util.spec_from_file_location('c_raw',C/'boundary-continuation-p4v2-005/verify_raw.py');raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
 verified=raw.verify(cp_path)
 config=read(binding['configs'][row['dataset']]);profile=Path(config['profiles'])
 return dict(schema='frozen-baseline-original120-capacity-negative-audit-v1',captured_s=time.time(),passed=True,classification='valid_incomplete_original_policy_capacity_timeout',measurement_valid=True,work_complete=False,equal_work_energy_comparison_eligible=False,checkpoint=ref(cp_path),receipt=ref(cp['receipt']),binding=ref(cp['binding']),auditor_source=ref(__file__),raw_verification=verified,timeout_diagnosis=diagnosis,actual_identity=identity,controller_handler_delay_max_s=max(handler_delays),controller_event_gap_max_s=maxgap,source=ref(Path(binding['host_release'])/'manifest.json'),profile=ref(profile),policy=ref(binding['configs'][row['dataset']]),files=files,scientific_row_unchanged=True,no_retry_authorization=True)
if __name__=='__main__':
 value=audit(Path(sys.argv[1]));dest=Path(sys.argv[2]);require(not dest.exists(),'immutable diagnosis exists');dest.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(json.dumps({k:value[k] for k in ('passed','classification','timeout_diagnosis')}))
