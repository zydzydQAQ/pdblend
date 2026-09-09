"""Freeze an untouched-science continuation after the observed original120 timeout."""
import copy,json,time,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parent;OLD=C/'boundary-continuation-p4v2-005';NEW=C/'boundary-continuation-p4v2-006'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
assert not NEW.exists();NEW.mkdir()
v=copy.deepcopy(read(OLD/'continuation.json'));oldstate=read(OLD/'execution/status.json');assert oldstate['phase']=='failed' and oldstate['finished_s']
newcp=sorted((C/'boundary-baseline-continuation-p4v2-005').rglob('checkpoints/*.json'));assert len(newcp)==4
for cp in newcp:v['executed_checkpoints'].append(dict(cell_id=read(cp)['row']['cell_id'],**ref(cp)))
assert len({r['cell_id'] for r in v['executed_checkpoints']})==17
observed={r['cell_id'] for r in v['executed_checkpoints']};v['remaining_cell_ids']=[cid for cid in v['remaining_cell_ids'] if cid not in observed];assert len(v['remaining_cell_ids'])==7
v.update(created_s=time.time(),output_root=str(C/'boundary-baseline-continuation-p4v2-006'),binding_root=str(NEW/'bindings'),previous_control_attempt=ref(OLD/'continuation.json'),previous_terminal_status=ref(OLD/'execution/status.json'),original120_capacity_diagnosis=ref(C/'Dynamo-SG375-first-timeout-independent-001.json'),original120_auditor=ref(C/'audit_original_timeout_negative_v1.py'),original120_cpu=ref(C/'timeout-negative-cpu-001.json'))
v['capacity_negative_successor_rules'].pop('zero_timeouts_and_503_and_other_errors');v['capacity_negative_successor_rules'].update(original120_timeout_allowed_only_after_full_independent_point_audit=True,timeout_bench_controller_exact120_required=True,zero_503_other_errors_or_unknown_physical_states=True,all_original_failures_remain_in_denominator=True,partial_chunks_never_called_zero_tokens=True)
v['authorization']='User authorized completion of the fixed-SLO paired matrix. Original17 observed checkpoints preserved; this suffix contains only original7 unobserved rows. Each known capacity negative requires independent raw/timing/actual identity/cleanup verification before a successor; any new error stops.'
v['control_fix']='Original named429 and separately diagnosed exact original120-second policy capacity timeouts may be recorded after the child has exited and independent raw/native/clock/identity checks pass. No serving/source/profile/trace/budget change.'
write(NEW/'continuation.json',v);h=sha(NEW/'continuation.json');oldh=sha(OLD/'continuation.json')
for name in ('continuation_driver.py','run_one.py','verify_raw.py'):
 text=(OLD/name).read_text().replace(oldh,h)
 if name=='continuation_driver.py':
  text=text.replace("assert len(old)==13 and len(spec['remaining_cell_ids'])==11","assert len(old)==17 and len(spec['remaining_cell_ids'])==7")
  text=text.replace("assert s.get('runtime_error') is None and s['request_timeouts']==0", """assert s.get('runtime_error') is None
    if s['request_timeouts']:
        auditor=load(C/'audit_original_timeout_negative_v1.py','c_original120_auditor')
        proof=auditor.audit(cp)
        target=OUT/'capacity-diagnoses'/(c['row']['cell_id']+'.json')
        assert not target.exists();write(target,proof)
        return dict(classification=proof['classification'],capacity_negative=True,
                    diagnosis=dict(path=str(target),sha256=sha(target)),request_timeouts=s['request_timeouts'])""")
  text=text.replace("spec=read(SPEC);zero=read(spec['zero_attempt_failure']['path']);", "spec=read(SPEC);prior=read(spec['previous_terminal_status']['path']);assert prior['phase']=='failed' and prior['finished_s'] and not Path('/proc/'+str(prior['pid'])).exists();zero=read(spec['zero_attempt_failure']['path']);")
 (NEW/name).write_text(text)
files=dict(read(OLD/'package.json')['files'])
for name in ['final_selected_baseline_v2.py','final_baseline_overlay_v1.py','source_identity_v3.py','source_identity_v2.py','source_identity.py','capacity_calibration_compatibility_v1.py','build_capacity_p5.py']:
 files[str(ROOT/name)]=sha(ROOT/name)
for path in [C/'audit_original_timeout_negative_v1.py',C/'test_original_timeout_negative_v1.py',C/'timeout-negative-cpu-001.json',C/'Dynamo-SG375-first-timeout-independent-001.json',OLD/'execution/status.json',OLD/'package.json',ROOT.parent/'main-slo-improvement-v1/protocol.py',Path(__file__).resolve(),*NEW.glob('*.py'),NEW/'continuation.json',*newcp]:files[str(path)]=sha(path)
assert all(sha(path)==h for path,h in files.items())
write(NEW/'package.json',dict(schema='C-original-baseline-remaining7-original120-capacity-continuation-v1',files=files,serving_sources_unchanged=True,logical_rows_unchanged=True,previous_seventeen_observations_preserved=True))
print(json.dumps(dict(continuation=ref(NEW/'continuation.json'),package=ref(NEW/'package.json'),remaining=len(v['remaining_cell_ids']))))
