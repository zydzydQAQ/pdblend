"""Diagnose exact original native queue refusal; never change producer status."""
import csv,hashlib,importlib.util,json,sys,time
from pathlib import Path
C=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/C');ROOT=C.parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'main-slo-improvement-v1')]
import protocol as protocol
import final_selected_baseline_v2 as identities
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def queue_proof(bench,events,limit):
    expected='RuntimeError: HTTP 503: decode failed: bounded admission queue full'
    bad=[b for b in bench if b['success']!='1'];assert bad
    byid={e['client_request_id']:e for e in events if e.get('kind')=='request_timing'}
    admissions={str(e['client_request_id']):e for e in events if e.get('kind')=='admission'}
    assert set(byid)==set(admissions)=={r['request_id'] for r in bench}
    good={b['request_id'] for b in bench if b['success']=='1'}
    routes={k:v['plan']['routes'][0]['decode_id'] for k,v in admissions.items()}
    results=[]
    for b in bad:
        assert b['http_status']=='503' and b['error']==expected and b['request_timeout']=='False'
        assert not b['admission_rejection'] and b['n_text_chunks']=='0' and not b['first_token_s']
        k=b['request_id'];t=byid[k];at=t['forward_started_s'];assert at<t['hard_deadline_s']
        active=[rid for rid in good if routes[rid]==routes[k] and byid[rid]['forward_started_s']<=at<=byid[rid]['stream_end_s']]
        # These are successful streams whose observed forward/stream intervals
        # span this refusal, independently of the controller's error label.
        # Controller DONE is observed before the native finally/abort removes
        # streams; this interval count is corroboration, not exact native state.
        assert active,'refusal lacks independently observed concurrent work'
        assert 0<=t['cleanup_end_s']-at<=1
        results.append(dict(request_id=k,controller_request_id=t['request_id'],instance_id=routes[k],native_forward_s=at,
            native_limit=limit,native_guard_fired_by_unique_original_response=True,controller_interval_is_not_exact_native_occupancy=True,simultaneously_active_successful_streams=sorted(active),active_count=len(active),
            refusal_cleanup_delay_s=t['cleanup_end_s']-at,error=b['error']))
    return results
def audit(cp_path):
    cp=read(cp_path)
    # v2 uses structured immutable refs; normalize only their representation.
    for name in ('binding','receipt'):
        if isinstance(cp[name],dict):
            reference=cp[name]
            assert sha(reference['path'])==reference['sha256']
            cp[name]=reference['path'];cp[name+'_sha256']=reference['sha256']
    binding=read(cp['binding']);receipt=read(cp['receipt']);summary=receipt['summary']
    files=dict(binding['files']);files.update(cp['artifacts']);files.update({str(cp_path):sha(cp_path),cp['binding']:cp['binding_sha256'],cp['receipt']:cp['receipt_sha256'],str(Path(__file__).resolve()):sha(__file__)})
    for p,h in files.items():assert sha(p)==h,'changed evidence '+p
    assert summary['measurement_valid'] and summary['fixed_window_valid'] and summary['runtime_error'] is None and not summary['work_complete']
    assert receipt['child_stopped'] and receipt['clock_restore_complete'] and not receipt['outer_cleanup_errors']
    assert all(v['complete'] for v in receipt['restoration'].values())
    actual=identities.actual_identity(protocol,cp,binding,cp['receipt'])
    cell=Path(cp['receipt']).parents[2]/'cells'/cp['row']['cell_id'];bench=list(csv.DictReader((cell/'bench.csv').open()));events=[json.loads(s) for s in (cell/'control.jsonl').read_text().splitlines()]
    engine=Path(next(iter(binding['instances'][0]['provenance']['source_files_at_import'])))
    engine=engine.parent/'engine.py';text=engine.read_text();assert 'if len(self.streams) >= self.config.get("max_pending", 128):' in text and 'raise web.HTTPTooManyRequests(text="bounded admission queue full")' in text
    configs=[read(i['engine_config']) for i in binding['instances']];assert all(c.get('max_pending',128)==128 for c in configs)
    details=queue_proof(bench,events,128)
    timing=load(C/'audit_eco_completed_v1.py','eco_timing').timing(bench,events)
    raw=load(C/'boundary-continuation-p4v2-006/verify_raw.py','eco_native_raw').verify(cp_path)
    assert raw['failed_requests']==len(details)==summary['failed_requests'] and raw['request_timeouts']==summary['request_timeouts']==0
    assert raw['completed_requests']+len(details)==raw['expected_requests']
    stamps=sorted(e['at_s'] for e in events if 'at_s' in e);gap=max(z-a for a,z in zip(stamps,stamps[1:]));assert gap<10
    files[str(C/'audit_eco_completed_v1.py')]=sha(C/'audit_eco_completed_v1.py');files[str(C/'boundary-continuation-p4v2-006/verify_raw.py')]=sha(C/'boundary-continuation-p4v2-006/verify_raw.py')
    return dict(schema='C-Eco-original-native128-queue-negative-audit-v1',passed=True,classification='valid_incomplete_original_native_queue_capacity_negative',
        measurement_valid=True,work_complete=False,equal_work_energy_comparison_eligible=False,not_hardware_saturation_proof=True,native_occupancy_authority='actual frozen unique native error branch; controller successful-stream overlap is corroboration only',
        captured_s=time.time(),checkpoint=ref(cp_path),receipt=ref(cp['receipt']),binding=ref(cp['binding']),source=ref(Path(binding['host_release'])/'manifest.json'),
        original_engine_source=ref(engine),native_limit=128,failed_requests=len(details),every_other_request_completed_exact_output=True,
        no_accepted_request_stranded=True,refusal_details=details,timing=timing,controller_event_gap_max_s=gap,raw=raw,actual_identity=actual,
        auditor_source=ref(__file__),files=files,no_source_config_budget_changed=True,no_retry_authorization=True)
if __name__=='__main__':
    result=audit(Path(sys.argv[1]));out=Path(sys.argv[2]);assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['passed','classification','failed_requests','timing','controller_event_gap_max_s']}))
