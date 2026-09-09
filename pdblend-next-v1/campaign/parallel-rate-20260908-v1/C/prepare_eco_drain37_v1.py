"""Freeze one complete replacement group before any Eco drain GPU run."""
import copy,csv,hashlib,json,os,subprocess,time
from pathlib import Path
C=Path(__file__).resolve().parent;REPO=C.parents[2]
OUT=C/'eco-drain37-v1';HOST=REPO/'releases/five-system100-C7B-baseline-eco-drain-v2-runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    assert not OUT.exists();OUT.mkdir()
    test=C/'test_eco_prefill_drain_v2.py'
    env=dict(os.environ,PYTHONPATH='/root/workspace/pdblend/.runtime-deps')
    cpu=json.loads(subprocess.check_output(['/usr/bin/python3','-B',str(test)],env=env,text=True));assert cpu['passed']
    cpu.update(source=ref(test),builder=ref(__file__),created_s=time.time());write(OUT/'cpu-validation.json',cpu)
    original=REPO/'campaign/five-system-results-v4/actual-snapshot-006/results.json'
    points=[p for p in read(original)['points'] if p['model']=='7b' and p['system']=='ecoserve' and p['phase']=='main' and p['slo_scale']==1]
    assert len(points)==30 and len({p['cell_id'] for p in points})==30
    new=C/'boundary-p4v2/declaration.json';newrows=[r for r in read(new)['cells'] if r['system']=='ecoserve'];assert len(newrows)==6
    records=[];retired=[]
    for p in sorted(points,key=lambda v:(v['dataset'],v['rate_rps'])):
        cp=read(p['checkpoint_path']);row=copy.deepcopy(cp['row']);row['repeat']=1
        records.append((row,dict(kind='original_main30',old_cell_id=row['cell_id'],checkpoint=ref(p['checkpoint_path']))))
        retired.append(dict(old_cell_id=row['cell_id'],checkpoint=ref(p['checkpoint_path']),reason='whole_source_group_replacement_not_pointwise_best'))
    old12=next(v for v in records if v[0]['dataset']=='alpaca' and v[0]['rate_rps']==12)
    extra=copy.deepcopy(old12);extra[0]['repeat']=2;extra[1]['kind']='authorized_original_alp12_second_repeat';records.append(extra)
    for row in newrows:
        records.append((copy.deepcopy(row),dict(kind='original_new_rate6',old_cell_id=row['cell_id'],declaration=ref(new))))
        retired.append(dict(old_cell_id=row['cell_id'],declaration=ref(new),reason='whole_new_eco6_source_replacement'))
    cells=[];mapping=[];files={str(HOST/'manifest.json'):sha(HOST/'manifest.json'),str(original):sha(original),str(new):sha(new),str(OUT/'cpu-validation.json'):sha(OUT/'cpu-validation.json')}
    for row,origin in records:
        oldcid=row['cell_id'];cid=f"eco-drain37-v1-7b-{row['dataset']}-r{row['rate_rps']:g}-s701-w100-ecoserve-slo1-repeat{row['repeat']}"
        row.update(cell_id=cid,execution_status='not_run',baseline_controller_host=str(HOST))
        assert row['system']=='ecoserve' and row['seed']==701 and row['slo_scale']==1 and row['arrival_window_s']==100
        assert sha(row['trace_path'])==row['trace_sha256'];files[row['trace_path']]=row['trace_sha256']
        cells.append(row);mapping.append(dict(new_cell_id=cid,repeat=row['repeat'],origin=origin))
    priority=lambda r:(0 if r['dataset']=='alpaca' and r['rate_rps']==12 else 1 if r['rate_rps'] in ([15,18.75] if r['dataset']=='alpaca' else [3.75] if r['dataset']=='sharegpt' else []) else 2,r['dataset'],r['rate_rps'],r['repeat'])
    cells.sort(key=priority);assert len(cells)==len({r['cell_id'] for r in cells})==37
    assert len([r for r in cells if priority(r)[0]<2])==8
    d=dict(schema='C-EcoServe-whole-source-replacement-37-v1',model='7b',system='ecoserve',created_s=time.time(),authorized=True,
        cells=cells,files=files,deadline_s=None,campaign_lifecycle='until_declared_complete_v1',baseline_controller_hosts={'ecoserve':str(HOST)},
        source=ref(HOST/'manifest.json'),original_snapshot=ref(original),parent_declaration=ref(new),replacement_mapping=mapping,retired_group=retired,
        priority_first8=[r['cell_id'] for r in cells[:8]],remaining29=[r['cell_id'] for r in cells[8:]],whole_group_required=37,
        original_policy_profiles_unchanged=True,request_timeout_s=120,window_s=100,seed=701,all8gpu_power=True,
        any_request_failure_stops_successors=True,complete_low_slo_negative_retained=True,pointwise_best_selection_forbidden=True,
        reason='Temporal OFF could strand already admitted native prefill; same admission and membership path affects all datasets/rates, so all original30 and new6 are replaced as one source group plus original Alp12 repeat2.',
        algorithm_reference='https://www.usenix.org/system/files/osdi26-du.pdf#page=7',algorithm_section='3.3 instance scheduler; accepted prefill continues before decode; macrocyclic routing does not discard admitted prefill progress')
    write(OUT/'declaration.json',d)
    diag=read(C/'Eco-Alp15-window-stranding-diagnosis-001.json');old=next(p for p in points if p['dataset']=='alpaca' and p['rate_rps']==12)
    quarantine=[]
    for cp_path,why in [(old['checkpoint_path'],'37 admitted requests on cr0 never produced first token; last OFF by client1132 closed cr0 at96.016s, no later ON before original120s timeouts'),(diag['checkpoint']['path'],'26 admitted requests1114..1139 on cr2 never produced first token; client1140 closed cr2 at77.1466s with no subsequent ON;85.200s idle gap precedes original120s timeout')]:
        cp=read(cp_path)
        for path,h in cp['artifacts'].items():assert sha(path)==h
        quarantine.append(dict(cell_id=cp['row']['cell_id'],checkpoint=ref(cp_path),receipt=ref(cp['receipt']),raw_files=cp['artifacts'],scientific_eligible_false=True,scientific_comparison_eligible=False,reason=why,classification='ecoserve_temporal_prefill_window_liveness_defect',raw_arithmetic_verified=True,raw_energy_retained=True))
    write(OUT/'quarantine.json',dict(schema='baseline-engineering-quarantine-v1',created_s=time.time(),points=quarantine,diagnosis=ref(C/'Eco-Alp15-window-stranding-diagnosis-001.json'),replacement_declaration=ref(OUT/'declaration.json'),no_original_raw_modified=True))
    print(json.dumps({'declaration':ref(OUT/'declaration.json'),'quarantine':ref(OUT/'quarantine.json'),'cpu':ref(OUT/'cpu-validation.json'),'count':37}))
if __name__=='__main__':main()
