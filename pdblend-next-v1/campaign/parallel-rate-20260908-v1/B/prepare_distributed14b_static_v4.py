"""B ShareGPT P12 factory; only real completed local profile evidence is accepted."""
import argparse,asyncio,copy,importlib.util,json,socket,sys
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parent
sys.path.insert(0,str(B));import distributed14b_static_v5 as run
read,sha,ref,need,write=run.read,run.sha,run.ref,run.need,run.write

def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def manifest_files(path):
    m=read(path);root=Path(path).parent
    result={str(Path(p) if Path(p).is_absolute() else root/p):h for p,h in m['files'].items()}
    need(all(sha(p)==h for p,h in result.items()),'source manifest changed')
    result[str(path)]=sha(path);return result

def successor_identity(successor, qualification):
    need(successor['max_service_frequency_mhz']==qualification['max_service_frequency_mhz'],'successor frequency differs from actual qualification')
    need(successor['profile']==successor['actual_profile']==qualification['profile'],'successor profile differs from actual qualification')
    need(successor['profile_registration']==qualification['profile_registration']['registration'],'successor raw registration differs')

def prepare(registration,profile,out):
    need(socket.gethostname()=='iZwz9i5bte3xkpmcoes3t2Z','target B only')
    d=B/'distributed-14b-v1';jobs_path=R/'distributed-14b-v1/B-jobs.json';jobs=read(jobs_path)
    parent_binding=ref(d/'pdb-deployment-001/binding-base.json');base=run.checked(parent_binding)
    regdir=R/'common/distributed14b-frequency-registration-v3';regmod=regdir/'registration.py'
    registered=load(regmod,'actual_B2100_registration').verify(ref(registration),ref(profile))
    need(registered['derived_profile']['frequency_registration']['hostname']==base['hostname'],'registered hardware belongs to another node')
    out=Path(out).resolve();need(not out.exists(),'new immutable P12 release required');out.mkdir(parents=True)
    host=R/'hosts/14b-capacity-p12';host_manifest=ref(host/'manifest.json');cpu=ref(R/'cpu-validation-p12.json')
    original=R/'A/final-p9-release-001/configs/sharegpt.json';parent_config=read(original);cfg=copy.deepcopy(parent_config)
    cfg.update(instances=copy.deepcopy(base['instances']),profiles=str(Path(profile).resolve()),max_service_frequency_mhz=2100,
        capacity_integration_v1=False,idle_domain_reacquire_v1=True,idle_domain_reacquire_timeout_s=1.5,controller_source_release=str(host),host_source_release=str(host),journal=str(out/'unused-control.jsonl'))
    cfg.pop('capacity_binding_path',None);cfg.pop('capacity_binding_sha256',None)
    native_spec=read(d/'deployment/pdblend/spec.json')
    native_paths=[v.split('=',1)[1] for v in native_spec['instances'][0]['environment'] if v.startswith('PYTHONPATH=')]
    need(len(native_paths)==1,'actual imported native package ambiguous')
    cfg['engine_source_release']=str(Path(native_paths[0].split(':')[0]).parent)
    costkey=lambda c:(c['tp'],c['source_mhz'],c['target_mhz'])
    costs={costkey(c):copy.deepcopy(c) for c in parent_config['frequency_costs'] if max(c['source_mhz'],c['target_mhz'])<2100}
    costs.update({costkey(c):copy.deepcopy(c) for c in registered['frequency_costs']});cfg['frequency_costs']=list(costs.values())
    need(len(registered['frequency_costs'])==6 and all(c['tp']==1 and c['source_mhz'] in (900,1500,2100) and c['target_mhz'] in (900,1500,2100) for c in registered['frequency_costs']),'six real TP1 directed transitions required')
    reg_input=read(registration);measurement=Path(reg_input['measurement']['path']).parent
    cfg['frequency_evidence']=list(parent_config.get('frequency_evidence',[]))+[str(measurement/f'transitions-gpu{g}/raw.json') for g in (6,7)]
    validator_path=R/'common/distributed14b-qualification-v3/verify.py';validator=load(validator_path,'new_B_saved_qualification')
    validator.policy_identity(cfg,parent_config,'pdblend')
    config_path=out/'configs/sharegpt.json';write(config_path,cfg,True)
    hardware_path=out/'hardware-identity.json';write(hardware_path,asyncio.run(run.hardware_identity('B')),True)
    group_path=out/'execution-group.json'
    write(group_path,dict(schema='distributed14b-dataset-actual-execution-group-v1',node='B',hostname=base['hostname'],model='14b',dataset='sharegpt',jobs=ref(jobs_path),max_service_frequency_mhz=2100,
        systems={'pdblend':dict(host_manifest=host_manifest,profile=ref(profile),configuration=ref(config_path))},future_systems=['mixed','distserve','dynamollm','ecoserve'],
        future_group_append_must_preserve_existing_system_objects=True),True)
    successor_path=out/'source-successor.json'
    write(successor_path,dict(schema='distributed14b-hardware-profile-successor-v1',jobs=ref(jobs_path),parent_controller_manifest=jobs['common_controller_manifest'],host_manifest=host_manifest,cpu_validation=cpu,
        original_profile_reference=jobs['profile_reference'],actual_profile=ref(profile),profile=ref(profile),profile_registration=ref(registration),max_service_frequency_mhz=2100,
        capacity_integration_v1=False,actual_arm='fixed2',original_scientific_rows_unchanged=True,
        noaction_proof=ref(R/'A/p9-nonalpaca-capacity-noaction-001/validation.json')),True)
    p10_manifest=ref(R/'hosts/14b-capacity-p11/manifest.json')
    idle_validator=R/'common/postpark-idle-transition-qualification-v1/verify.py'
    idle_measurement=ref(d/'postpark-idle-probe-001/evidence-manifest.json')
    idle_proof=load(idle_validator,'B_actual_saved_idle_transitions').verify(idle_measurement)
    idle_qualification=dict(validator=ref(idle_validator),measurement=idle_measurement)
    successor=read(successor_path)
    successor['idle_domain_reacquisition']=dict(enabled=True,flag='idle_domain_reacquire_v1',parent_controller_manifest=p10_manifest,timeout_s=1.5,idle_transition_qualification=idle_qualification)
    # This factory has not been published yet; finish the single successor object
    # before any binding or qualification ref is frozen below.
    write(successor_path,successor)
    lineage_path=out/'controller-successor.json'
    write(lineage_path,dict(schema='distributed14b-P11-to-P12-idle-recovery-successor-v1',
        parent_manifest=p10_manifest,host_manifest=host_manifest,cpu_validation=cpu,
        profile=ref(profile),profile_registration=ref(registration),idle_domain_reacquire_timeout_s=1.5,idle_transition_qualification=idle_qualification,
        prior_engineering_diagnosis=ref(d/'p11-late-idle-observation-diagnosis-001.json'),
        failed_P11_observation_preserved=True,whole_assigned_group_remeasured_under_P12=True),True)
    source_package=R/'common/distributed14b-pdb-qualification-source-v3/manifest.json'
    files={**base['files'],**registered['sources'],**idle_proof['files'],**manifest_files(R/'hosts/14b-capacity-p11/manifest.json'),**manifest_files(R/'hosts/14b-capacity-p10/manifest.json'),**manifest_files(host/'manifest.json'),**manifest_files(regdir/'manifest.json'),**manifest_files(source_package)}
    for path in (original,config_path,hardware_path,group_path,successor_path,lineage_path,jobs_path,Path(profile),Path(registration),validator_path,Path(__file__),B/'distributed14b_static_v5.py',Path(cpu['path'])):files[str(path)]=sha(path)
    ordinary=d/'pdb-ordinary-001';frequencies=[]
    for directory in (ordinary,d/'profile-validation-001',d/'profile-validation-002',d/'frequency2400-feasibility-001'):
        for path in directory.rglob('*'):
            if path.is_file():files[str(path)]=sha(path)
        status=read(directory/'status.json')
        if 'spec' in status:files.update(run.checked(status['spec'])['files'])
    native_validator=ref(R/'common/distributed14b-profile-validation-v1/validate.py');files[native_validator['path']]=native_validator['sha256']
    for directory,point_ids in (
        ('profile-validation-001',[f'gpu6-mid16-{f}' for f in (900,1500,2100)]),
        ('profile-validation-002',[f'gpu7-mid16-{f}' for f in (900,1500,2100)])):
        status_ref=ref(d/directory/'status.json');s=run.checked(status_ref)
        for point in point_ids:frequencies.append(dict(status=status_ref,binding=s['binding'],raw=ref(d/directory/point/'raw.json'),point_id=point,validator=native_validator))
    rows=[run.flat(c) for c in jobs['pdb_cells']];first=next(c for c in rows if c['rate_rps']==.05 and c['repeat']==1);rows=[first]+[c for c in rows if c is not first]
    for row in rows:files[row['trace']]=row['trace_sha256']
    for path in cfg['frequency_evidence']+[cfg['interconnect'],cfg['transfer_evidence']]:files[path]=sha(path)
    base.update(host_release=str(host),system='pdblend',configs={'sharegpt':str(config_path)},files=dict(files),output_correctness_verified=True,correctness_gate_required_before_performance=False,actual_arm='fixed2',capacity_integration_v1=False)
    binding_path=out/'binding.json';write(binding_path,base,True);files[str(binding_path)]=sha(binding_path)
    qpath=out/'qualification.json'
    qualification=dict(schema='distributed14b-qualified-execution-v1',model='14b',node='B',hostname=base['hostname'],dataset='sharegpt',system='pdblend',
        binding=parent_binding,controller_successor=ref(lineage_path),host_manifest=host_manifest,measurement_host_manifest=ref(R/'hosts/14b-capacity-p9/manifest.json'),profile=ref(profile),
        max_service_frequency_mhz=2100,qualified_frequencies_mhz=[900,1500,2100],execution_group=ref(group_path),hardware_identity=ref(hardware_path),actual_configuration=ref(config_path),policy_parent=ref(original),
        native=dict(kind='pdb_ordinary',status=ref(ordinary/'status.json'),validator=native_validator),frequency_cases=frequencies,
        profile_registration=dict(validator=ref(regmod),registration=ref(registration)),files=dict(files))
    successor_identity(read(successor_path),qualification)
    run.verify_successor_qualification(read(successor_path),qualification)
    write(qpath,qualification,True)
    rebuilt=validator.verify(ref(qpath),ref(binding_path));write(out/'qualification-reconstruction.json',rebuilt,True)
    rules_path=out/'execution-rules.json'
    write(rules_path,dict(schema='distributed14b-static-execution-rules-v2',jobs=ref(jobs_path),source_successor=ref(successor_path),whole_assigned_group=True,actual_arm='fixed2',capacity_integration_v1=False,
        first_admission_original_cell=first['cell_id'],arrival_window_s=100,request_timeout_s=120,cleanup_local_budget_s=90,global_deadline=None,
        arrival_dispatch_max_limit_s=1.,arrival_dispatch_p99_limit_s=.1,stop_on_any_request_failure=True,stop_above_first_complete_pdb_slo_below=.9,source=ref(B/'distributed14b_static_v5.py')),True)
    files.update({str(qpath):sha(qpath),str(rules_path):sha(rules_path)})
    release=dict(schema='distributed14b-static-release-v2',node='B',model='14b',dataset='sharegpt',system='pdblend',jobs=ref(jobs_path),binding=ref(binding_path),actual_arm='fixed2',capacity_integration_v1=False,rows=rows,
        execution_rules=ref(rules_path),source_successor=ref(successor_path),qualification=ref(qpath),qualification_validator=ref(validator_path),policy_parent=ref(original),files=files)
    release_path=out/'release.json';write(release_path,release,True);run.load_release(release_path)
    write(out/'preparation.json',dict(passed=True,hardware_actions=False,release=ref(release_path),declared=len(rows),actual_arm='fixed2',actual_profile=ref(profile)),True)
    return ref(release_path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.registration,a.profile,a.out)))
