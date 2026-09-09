"""A Eco source and logical group preparation; no GPU/predecessor inferred."""
import copy,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
A=Path(__file__).resolve().parent;ROOT=A.parent;REPO=ROOT.parents[1];OUT=A/'eco-drain31-v1'
HOST=REPO/'releases/five-system100-A14B-baseline-eco-drain-v1-runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
    assert not OUT.exists();OUT.mkdir();restore=OUT/'restore-code';restore.mkdir();code=OUT/'code';code.mkdir()
    old=ROOT/'B/A-baseline-until-complete-draft'
    for name in ['restore_parent.py','restore_adapter.py','test_draft.py','README.md']:shutil.copyfile(old/name,restore/name)
    subprocess.run(['/usr/bin/python3','-B',str(restore/'test_draft.py')],check=True)
    cpu=json.loads(subprocess.check_output(['/usr/bin/python3','-B',str(A/'test_eco_prefill_drain_v1.py')],env=dict(os.environ,PYTHONPATH='/root/workspace/pdblend/.runtime-deps'),text=True));write(OUT/'source-cpu.json',cpu)
    snapshot=REPO/'campaign/five-system-results-v4/actual-snapshot-006/results.json'
    points=[p for p in read(snapshot)['points'] if p['model']=='14b' and p['system']=='ecoserve' and p['phase']=='main' and p['slo_scale']==1]
    assert len(points)==30;records=[];mapping=[]
    for p in sorted(points,key=lambda r:(r['dataset'],r['rate_rps'])):
        original=read(p['checkpoint_path'])['row'];r=copy.deepcopy(original);r['repeat']=1
        r.update(cell_id=f"eco-drain31-v1-14b-{r['dataset']}-r{r['rate_rps']:g}-s701-w100-ecoserve-slo1-repeat1",execution_status='not_run',baseline_controller_host=str(HOST))
        assert sha(r['trace_path'])==r['trace_sha256']
        records.append(r);mapping.append(dict(old_cell_id=p['cell_id'],new_cell_id=r['cell_id'],repeat=1,checkpoint=ref(p['checkpoint_path'])))
    old12=next(p for p in points if p['dataset']=='alpaca' and p['rate_rps']==12)
    r=copy.deepcopy(next(r for r in records if r['dataset']=='alpaca' and r['rate_rps']==12));r['repeat']=2;r['cell_id']=r['cell_id'].removesuffix('repeat1')+'repeat2';records.append(r)
    mapping.append(dict(old_cell_id=old12['cell_id'],new_cell_id=r['cell_id'],repeat=2,checkpoint=ref(old12['checkpoint_path']),authorized_additional_repeat=True))
    records.sort(key=lambda r:(0 if r['dataset']=='alpaca' and r['rate_rps']==12 else 1,r['dataset'],r['rate_rps'],r['repeat']))
    write(OUT/'declaration.json',dict(schema='A-Eco-whole-source-logical31-v1',created_s=time.time(),model='14b',system='ecoserve',cells=records,source=ref(HOST/'manifest.json'),original_snapshot=ref(snapshot),replacement_mapping=mapping,
        logical_declared_count=31,ready_for_gpu=False,actual_final_pdb_terminal_required=True,execution_scope_pending_final_pdb=True,
        final_first_loss_exclusions_require_explicit_pinned_scope=True,new_rate_rows_require_explicit_append_declaration=True,
        original_policy_profiles_unchanged=True,window_s=100,request_timeout_s=120,seed=701,all8gpu_power=True,deadline_s=None,campaign_lifecycle='until_declared_complete_v1',
        source_cpu=ref(OUT/'source-cpu.json'),restore_cpu=ref(restore/'cpu-validation.json'),no_future_predecessor_or_qualification_inferred=True))
    # Future binder source is prepared now; the spec needs the actual measured
    # restore operation and P8 final native identities before it can be built.
    s=(ROOT/'C/eco-drain37-v1/code/bind.py').read_text().replace('five-system100-C7B-baseline-eco-drain-v2-runtime','five-system100-A14B-baseline-eco-drain-v1-runtime')
    s=s.replace("SYSTEMS=('mixed','distserve','ecoserve','dynamollm','dynamollm-resident')","SYSTEMS=('ecoserve',)")
    (code/'bind.py').write_text(s)
    shutil.copyfile(ROOT/'C/baseline-until-complete-v1/validate.py',code/'validate.py')
    files={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    files.update({str(A/n):sha(A/n) for n in ['build_eco_prefill_drain_v1.py','test_eco_prefill_drain_v1.py','prepare_eco_cpu_v1.py']});files[str(HOST/'manifest.json')]=sha(HOST/'manifest.json')
    write(OUT/'manifest.json',dict(schema='A-Eco-CPU-preparation-only-v1',files=files,host=ref(HOST/'manifest.json'),restore_parent_manifest=ref(old/'manifest.json'),gpu_executed=False,ready_for_gpu=False,actual_final_pdb_terminal_required=True))
    print(json.dumps(dict(out=str(OUT),manifest=ref(OUT/'manifest.json'),declaration=ref(OUT/'declaration.json'),host=ref(HOST/'manifest.json'))))
if __name__=='__main__':main()
