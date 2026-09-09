"""Check that the drain repair preserved the declared open-loop arrival timing."""
import csv,hashlib,importlib.util,json,math
from pathlib import Path
C=Path(__file__).resolve().parent
def timing(bench,events):
    byid={e['client_request_id']:e for e in events if e.get('kind')=='request_timing'}
    assert len(byid)==len(bench) and set(byid)=={r['request_id'] for r in bench}
    lateness=[];handlers=[]
    for r in bench:
        planned,actual,deadline=map(float,(r['planned_arrival_s'],r['actual_dispatch_s'],r['request_deadline_s']))
        assert all(map(math.isfinite,(planned,actual,deadline))) and abs(deadline-planned-120)<1e-5
        assert r['open_loop_independent']=='True' and -1e-5<=actual-planned<=1
        t=byid[r['request_id']]
        assert t['planned_arrival_s']==planned and t['actual_dispatch_s']==actual and t['hard_deadline_s']==deadline
        delay=t['handler_arrival_s']-actual;assert math.isfinite(delay) and 0<=delay<=1
        lateness.append(actual-planned);handlers.append(delay)
    ordered=sorted(lateness);q=(len(ordered)-1)*.99;k=int(q)
    p99=ordered[k]+(ordered[min(k+1,len(ordered)-1)]-ordered[k])*(q-k)
    assert p99<=.1
    return dict(dispatch_lateness_max_s=max(lateness),dispatch_lateness_p99_s=p99,handler_delay_max_s=max(handlers),request_budget_s=120)
def audit(cp_path):
    cp=json.loads(Path(cp_path).read_text());receipt=json.loads(Path(cp['receipt']).read_text())
    path=C/'boundary-continuation-p4v2-006/verify_raw.py'
    s=importlib.util.spec_from_file_location('c_eco_raw_verify',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    raw=m.verify(cp_path);assert raw['work_complete'] and not raw['failed_requests'] and not raw['request_timeouts']
    cell=Path(cp['receipt']).parents[2]/'cells'/cp['row']['cell_id']
    bench=list(csv.DictReader((cell/'bench.csv').open()));events=[json.loads(s) for s in (cell/'control.jsonl').read_text().splitlines()]
    result=timing(bench,events)
    assert receipt['summary']['runtime_error'] is None and all(v['complete'] for v in receipt['restoration'].values())
    return dict(passed=True,raw=raw,timing=result,all_original_requests_complete=True,no_zero_progress_stranding=True,auditor_source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
