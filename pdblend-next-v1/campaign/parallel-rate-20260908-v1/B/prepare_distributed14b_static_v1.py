"""Freeze assigned B ShareGPT fixed2 policy and real local qualification refs."""
import argparse, copy, importlib.util, json, socket, sys
from pathlib import Path
B=Path(__file__).resolve().parent; R=B.parent; W=R.parent.parent
sys.path.insert(0,str(B)); import distributed14b_static_v1 as run
read,sha,ref,need,write=run.read,run.sha,run.ref,run.need,run.write

def main():
    p=argparse.ArgumentParser(); p.add_argument('--profile-status',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    need(socket.gethostname()=='iZwz9i5bte3xkpmcoes3t2Z','target-B only')
    jobs_path=R/'distributed-14b-v1/B-jobs.json'; jobs=read(jobs_path)
    d=B/'distributed-14b-v1'; br=ref(d/'pdb-deployment-001/binding-base.json'); base=run.checked(br)
    native_path=d/'pdb-ordinary-001/status.json'; native=read(native_path)
    need(all(native.get(k) is True for k in ('passed','complete','measurement_valid','native_cleanup_complete','clock_restore_complete')) and not native['errors'],'actual ordinary qualification incomplete')
    lineage=read(native_path.parent/'lineage.json'); need(lineage['binding']==br,'ordinary actual binding differs')
    profile=read(a.profile_status)
    need(profile.get('passed') is True and profile.get('measurement_valid') is True and profile.get('binding')==br,'actual B profile/frequency qualification incomplete or foreign binding')
    for path,h in profile['files'].items(): need(sha(path)==h,'profile evidence changed')
    need(not a.out.exists(),'immutable new release required'); a.out.mkdir(parents=True)
    original=R/'A/final-p9-release-001/configs/sharegpt.json'; cfg=read(original)
    config=copy.deepcopy(cfg); config['instances']=copy.deepcopy(base['instances'])
    config.update(capacity_integration_v1=False,controller_source_release=base['host_release'],host_source_release=base['host_release'],engine_source_release=str(Path(base['instances'][0]['engine_config']).parent),journal=str(a.out/'unused-control.jsonl'))
    config.pop('capacity_binding_path',None); config.pop('capacity_binding_sha256',None)
    # Operational source labels match the actual imported package recorded by deployment.
    spec=read(d/'deployment/pdblend/spec.json')
    imported=[v.split('=',1)[1] for v in spec['instances'][0]['environment'] if v.startswith('PYTHONPATH=')]
    need(len(imported)==1,'exact imported native source path required')
    config['engine_source_release']=str(Path(imported[0].split(':')[0]).parent)
    allowed={'instances','capacity_integration_v1','capacity_binding_path','capacity_binding_sha256','controller_source_release','host_source_release','engine_source_release','journal'}
    need({k:v for k,v in config.items() if k not in allowed}=={k:v for k,v in cfg.items() if k not in allowed},'original PDB policy changed')
    config_path=a.out/'configs/sharegpt.json'; write(config_path,config,True)
    base.update(system='pdblend',configs={'sharegpt':str(config_path)},output_correctness_verified=True,correctness_gate_required_before_performance=False,actual_arm='fixed2',capacity_integration_v1=False)
    files=base['files']; files[str(config_path)]=sha(config_path); files[str(jobs_path)]=sha(jobs_path)
    for p in (original,Path(jobs['profile_reference']['path']),R/'A/p9-nonalpaca-capacity-noaction-001/validation.json',Path(__file__),B/'distributed14b_static_v1.py',native_path,a.profile_status): files[str(p)]=sha(p)
    for p,h in profile['files'].items(): files[p]=h
    for f in native_path.parent.rglob('*'):
        if f.is_file(): files[str(f)]=sha(f)
    # Resolve only measured/configuration evidence inputs; the journal is a new output.
    for key in ('profiles','interconnect','transfer_evidence'):
        p=Path(config[key]); need(p.is_file(),'required policy evidence missing '+str(p)); files[str(p)]=sha(p)
    for p in config.get('frequency_evidence',[]): need(Path(p).is_file(),'frequency reference missing'); files[p]=sha(p)
    rows=[run.flat(c) for c in jobs['pdb_cells']]
    # Reuse the declared lowest-rate point as the real controller first-admission gate.
    first=next(row for row in rows if row['rate_rps']==.05 and row['repeat']==1)
    rows=[first]+[row for row in rows if row is not first]
    for row in rows: files[row['trace']]=row['trace_sha256']
    binding_path=a.out/'binding.json'; write(binding_path,base,True)
    rules=dict(schema='distributed14b-static-execution-rules-v1',jobs=ref(jobs_path),whole_assigned_group=True,actual_arm='fixed2',capacity_integration_v1=False,first_admission_original_cell=first['cell_id'],arrival_window_s=100,request_timeout_s=120,cleanup_local_budget_s=90,global_deadline=None,arrival_dispatch_max_limit_s=1.,arrival_dispatch_p99_limit_s=.1,stop_on_any_request_failure=True,stop_above_first_complete_pdb_slo_below=.9,source=ref(B/'distributed14b_static_v1.py'))
    rules_path=a.out/'execution-rules.json'; write(rules_path,rules,True)
    release=dict(schema='distributed14b-static-release-v1',node='B',model='14b',dataset='sharegpt',system='pdblend',jobs=ref(jobs_path),binding=ref(binding_path),actual_arm='fixed2',capacity_integration_v1=False,rows=rows,execution_rules=ref(rules_path),qualification_refs={'ordinary':ref(native_path),'profile_frequency':ref(a.profile_status)},policy_parent=ref(original),files={**files,str(binding_path):sha(binding_path),str(rules_path):sha(rules_path)})
    release_path=a.out/'release.json'; write(release_path,release,True); run.load_release(release_path)
    write(a.out/'preparation.json',dict(passed=True,hardware_actions=False,release=ref(release_path),declared=len(rows),policy_fields_unchanged=True),True)
    print(json.dumps(dict(release=ref(release_path),ready_for_GPU=True)))
if __name__=='__main__': main()
