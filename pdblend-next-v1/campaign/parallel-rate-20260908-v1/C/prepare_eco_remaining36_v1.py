"""Resume the unchanged Eco37 group after diagnosed native128 queue refusal."""
import ast,copy,hashlib,json,time
from pathlib import Path
C=Path(__file__).resolve().parent;PARENT=C/'eco-drain37-v1';OUT=C/'eco-drain37-continuation-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
    assert not OUT.exists();OUT.mkdir();(OUT/'code').mkdir()
    declaration=PARENT/'declaration.json';d=read(declaration);cp=next((PARENT/'performance/results/checkpoints').glob('*.json'));observed=read(cp)['row']['cell_id']
    status=PARENT/'performance/status.json';old=read(status);assert old['phase']=='engineering_gate_failed' and old['completed']==old['attempted']==[observed] and old['node_lease_held'] is False
    diagnosis=PARENT/'first-native-queue-negative-diagnosis.json';assert read(diagnosis)['passed'] and read(diagnosis)['failed_requests']==42
    independent=C.parent/'B/C-Eco12-queuefull-independent-audit-v1.json';assert independent.is_file()
    remaining=[copy.deepcopy(r) for r in d['cells'] if r['cell_id']!=observed];assert len(remaining)==36
    continuation=dict(schema='C-Eco37-remaining36-native128-negative-continuation-v1',authorized=True,created_s=time.time(),logical_declaration=ref(declaration),
        previous_terminal_status=ref(status),retained_observations=[dict(cell_id=observed,checkpoint=ref(cp),diagnosis=ref(diagnosis),independent_audit=ref(independent))],
        remaining_cells=remaining,remaining_cell_ids=[r['cell_id'] for r in remaining],original_work_complete_not_modified=True,
        source_release=d['baseline_controller_hosts']['ecoserve'],no_observed_cell_replayed=True,no_source_policy_profile_or_budget_changed=True,
        native_queue_negative_may_continue_only_after_exact_original_error_source_raw_timing_identity_cleanup_audit=True,
        any_new_error_type_stops_successors=True,request_timeout_s=120,window_s=100,seed=701,all8gpu_power=True,deadline_s=None,campaign_lifecycle='until_declared_complete_v1')
    write(OUT/'continuation.json',continuation)
    s=(PARENT/'code/run_v2.py').read_text().replace("PACKAGE=HERE/'eco-drain37-v1/package-v2.json'","PACKAGE=HERE/'eco-drain37-continuation-001/package.json'")
    needle="    return binding,cells"
    extra="""    continuation_path=HERE/'eco-drain37-continuation-001/continuation.json'
    continuation=read(continuation_path)
    require(binding['continuation_instruction']==dict(path=str(continuation_path),sha256=sha(continuation_path)),'continuation not bound')
    require(continuation['logical_declaration']==dict(path=str(DECLARATION),sha256=DECLARATION_SHA),'wrong original group')
    prior=read(continuation['previous_terminal_status']['path'])
    require(prior['finished_s'] and not Path('/proc/'+str(prior['pid'])).exists(),'previous owner remains active')
    retained={r['cell_id'] for r in continuation['retained_observations']}
    cells=[row for row in cells if row['cell_id'] not in retained]
    require(cells==continuation['remaining_cells'] and len(cells)==36,'remaining suffix differs')
    return binding,cells"""
    assert s.count(needle)==1;s=s.replace(needle,extra)
    start=s.index("                if not gate['passed']:state.update(")
    end=s.index("                write(args.out/'status.json',state)",start)
    s=s[:start]+"""                checkpoint=output/'checkpoints'/(row['cell_id']+'.json')
                if gate['passed']:
                    independent=load(HERE/'audit_eco_completed_v1.py','c_eco_arrival_audit').audit(checkpoint)
                else:
                    # Preserve the producer's failed engineering gate and raw.
                    # Only this separately diagnosed original native refusal
                    # can be accepted as a measured performance negative.
                    independent=load(HERE/'audit_eco_native_queue_negative_v1.py','c_eco_native_queue_audit').audit(checkpoint)
                    state.setdefault('native_queue_negative_cells',[]).append(row['cell_id'])
                write(output/'independent-audits'/(row['cell_id']+'.json'),independent,True)
"""+s[end:]
    ast.parse(s);(OUT/'code/run.py').write_text(s)
    files=read(PARENT/'package-v2.json')['files'];files.update({str(p):sha(p) for p in [__file__,OUT/'continuation.json',OUT/'code/run.py',cp,status,diagnosis,independent,PARENT/'native-queue-negative-cpu.json',C/'audit_eco_native_queue_negative_v1.py',C/'test_eco_native_queue_negative_v1.py']})
    write(OUT/'package.json',dict(schema='C-Eco37-remaining36-execution-package-v1',files=files,parent_package=ref(PARENT/'package-v2.json'),no_physical_source_change=True))
    print(json.dumps(dict(continuation=ref(OUT/'continuation.json'),package=ref(OUT/'package.json'),runner=ref(OUT/'code/run.py'))))
if __name__=='__main__':main()
