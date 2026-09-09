"""Exact declared rows, unchanged shared execution, and fail-closed timing checks."""
import ast,copy,json
from pathlib import Path
import eco_drained_runner_v1 as runner
B=Path(__file__).resolve().parent

def run():
 d=runner.p.read(runner.DECL); rows=d['cells'];assert len(rows)==35 and len(d['declared_cells'])==37
 assert {(x['row']['dataset'],float(x['row']['rate_rps'])) for x in d['excluded_cells']}=={('sharegpt',2.),('longbench',1.)}
 for r in rows:
  old=r['source_row']
  for k in ('trace_path','trace_sha256','content_pairing_sha256','n_requests','seed','slo_ttft_s','slo_tpot_s','rate_rps','dataset','system'):
   assert r[k]==old[k],(r['cell_id'],k)
  assert r['system']=='ecoserve' and r['arrival_window_s']==100
 assert sum(r['replacement_scope']=='entire-newrates-3x2-group' for r in rows)==6
 a=[dict(planned_arrival_s=10.,actual_dispatch_s=10.,dispatch_delay_s=0.)]
 summary=dict(n_expected=1,dispatch_delay_max_s=0.,dispatch_delay_p99_s=0.)
 assert runner.timing_values(a,summary)['passed']; count=2
 for field,value in [('actual_dispatch_s',9.),('actual_dispatch_s',float('nan')),('dispatch_delay_s',.1)]:
  x=copy.deepcopy(a);x[0][field]=value
  try:runner.timing_values(x,summary)
  except (ValueError,RuntimeError):count+=1
  else:raise AssertionError(field)
 late=[dict(planned_arrival_s=10.,actual_dispatch_s=11.1,dispatch_delay_s=1.1)]
 result=runner.timing_values(late,dict(n_expected=1,dispatch_delay_max_s=1.1,dispatch_delay_p99_s=1.1));assert not result['passed'] and len(result['errors'])==2;count+=1
 old=ast.parse((B/'cooperative_dynamo_runner_v2.py').read_text());new=ast.parse((B/'eco_drained_runner_v1.py').read_text())
 for name in ('timing_values','timing_gate'):
  get=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),include_attributes=False)
  assert get(old)==get(new);count+=1
 text=(B/'eco_drained_runner_v1.py').read_text();assert text.index("p.write(cp,")<text.index("p.need(gate['passed']")
 assert "await common.run_one(session,binding,row,output,hardware)" in text and "gate=c.fixed.engineering_gate(receipt)" in text;count+=1
 return dict(passed=True,count=count,exact35rows_checked=True,declared37=True,excluded2above_existing_boundary=True,whole_newrate6=True,original_shared_measurement_child_unchanged=True,each_failure_preserved_before_stop=True)
if __name__=='__main__':print(json.dumps(run()))
