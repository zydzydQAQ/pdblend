"""Read-only raw requalification of fresh 14B legacy engines on their actual node."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
NATIVE_AUDIT=p.REPO/'campaign/AC-baseline-binding-v2/gate_evidence.py'
NATIVE_AUDIT_SHA='b84a113563f1b064be5ef7a8cbf2006b0790f3b60bb1be3851ce66114dd9e9e6'
HOSTNAMES={'B':'iZwz9i5bte3xkpmcoes3t2Z','Anew20260909':'iZwz9274emxme9019d2sjgZ'}
NATIVE_PARENTS={'resident':'945f69e71a64f8780e7b6e04bf26b5cdcd1d13a1626d97032b7e213ad2414dd7',
                'heterogeneous':'e6c56d3a2759ca1c146b2b4e39b901a40b5ab298384b29efbd8758dd94be22c2'}


def pinned_files(files):
    for path,digest in files.items():p.need(p.sha(path)==digest,'frozen native/policy evidence changed: '+path)


def terminal_owner(owner):
    p.need(owner['complete'] and owner.get('finished_s') and not owner.get('error') and not owner['node_lease_held'],'qualification owner not successful terminal')
    p.need(not p.active_owner(owner),'qualification producer is still active')
    child=owner.get('child')
    if child:p.need(child.get('exitcode')==0 and child.get('finished_s') and not p.active_owner(child),'qualification child not clean and exited')


def native_source(reference,heterogeneous):
    proof=p.checked(reference);layout='heterogeneous' if heterogeneous else 'resident';row=proof[layout]
    p.need(row['original']['sha256']==NATIVE_PARENTS[layout],'unexpected native gate parent')
    for ref in (row['original'],row['actual']):p.need(p.sha(ref['path'])==ref['sha256'],'native gate source changed')
    original=Path(row['original']['path']).read_text()
    expected=original.replace('clocks.set(range(8),2520,verify_rise=False)','clocks.set(range(8),2100,verify_rise=False)')
    expected=expected.replace('asyncio.to_thread(ClockOwner,hardware,tuple(range(8)))','asyncio.to_thread(ClockOwner,hardware,tuple(range(8)),max_frequency=2100)')
    if heterogeneous:
        campaign=p.REPO/'campaign'
        expected=expected.replace('from checks import require,write',"sys.path.insert(0,'"+str(campaign/'A14B-legacy-heterogeneous-correctness-v1')+"')\nfrom checks import require,write")
        expected=expected.replace("COMMON=ROOT.parent/'five-system-execution-v2/run.py'","COMMON=Path('"+str(ROOT/'common/execution-until-complete-v1/run.py')+"')")
        expected=expected.replace("COMMON_SHA='ddc634e0b826d1873ed0bb7e3bd9088ba1412476725d8ec1e371414ccce54ad2'","COMMON_SHA='77bcbbb68e20419e5bc469a838c71e1abfa789dc167d901501715fda0ff4a8a9'")
        expected=expected.replace("    require(time.time()+510<m.GLOBAL_DEADLINE,'insufficient global time for gate plus cleanup')\n",'')
        expected=expected.replace('cleanup_end=time.monotonic()+max(0,min(90,m.GLOBAL_DEADLINE-time.time()))','cleanup_end=time.monotonic()+90')
    p.need(Path(row['actual']['path']).read_text()==expected,'native mechanism or cleanup source changed outside explicit clock/platform binding')
    return {ref['path']:ref['sha256'] for ref in (reference,row['original'],row['actual'])}


def platform(host,files):
    host=Path(host);manifest_ref=p.ref(host/'manifest.json');manifest=p.checked(manifest_ref)
    p.need(manifest['configuration_key']=='max_service_frequency_mhz' and manifest['max_service_frequency_mhz_default']==2520
        and manifest['explicit_target_domain_mhz']==[900,1500,2100],'declared platform domain differs')
    parent=Path(manifest['parent_release']);parent_ref=dict(path=str(parent/'manifest.json'),sha256=manifest['parent_manifest_sha256'])
    old=p.checked(parent_ref);p.need(set(old['files'])==set(manifest['files']),'runtime file set changed')
    builder=p.load(manifest['builder'],'fresh_legacy_saved_platform_builder')
    p.need(manifest['builder']['sha256']=='3dc179595756817b2df1ff01b59d3363870306a97ba1e4695dce0712b9e83b11','unreviewed platform transformation')
    collector=manifest['collector'];p.need(p.sha(collector['path'])==collector['sha256'],'collector source changed')
    for name,digest in old['files'].items():
        oldfile=parent/name;newfile=host/name;p.need(p.sha(oldfile)==digest and p.sha(newfile)==manifest['files'][name],'runtime source differs')
        data=oldfile.read_bytes()
        if name in manifest['platform_adaptation']:
            text,proof=builder.edits(Path(name).name,data.decode());proof['parent_sha256']=digest
            p.need(json.loads(json.dumps(proof))==manifest['platform_adaptation'][name],'recorded platform transformation differs')
            data=text.encode()
        elif name=='benchmarks/scripts/bench_vllm.py':data=Path(collector['path']).read_bytes()
        p.need(newfile.read_bytes()==data,'undeclared runtime/control source change: '+name)
        files[str(oldfile)]=digest;files[str(newfile)]=manifest['files'][name]
    verification_ref=p.ref(host.parent/'verification-002.json');verification=p.checked(verification_ref)
    p.need(verification['passed'] and verification['gpu_executed'] is False and verification['formal_qualification_required']
        and verification['original_baseline_points_preserved']==357
        and all(verification[k] for k in ('clock_fallback_and_deferred_target_covered','planner_worker_thread_covered','dynamo_full_topology_search_covered')),
        'platform CPU equivalence coverage incomplete')
    for reference in (manifest_ref,parent_ref,manifest['builder'],collector,verification_ref):files[reference['path']]=reference['sha256']


def policy(reference,bootstrap,dataset,system,profile_ref,adapter,files):
    saved=p.checked(reference);pinned_files(saved['files'])
    p.need(saved['instances']==bootstrap['instances'] and saved['hostname']==bootstrap['hostname'] and saved['model']=='14b'
        and saved['system']==system and set(saved['configs'])=={dataset},'mapped policy identity differs')
    mapping_ref=saved['policy_adaptation'];mapping=p.checked(mapping_ref)
    p.need(mapping['model']=='14b' and mapping['dataset']==dataset and mapping['system']==system and mapping['profile']==profile_ref
        and mapping['policy_flow_sorting_budgets_unchanged'] and mapping['historical_metadata_is_not_new_node_qualification'], 'policy map declaration differs')
    directory=Path(reference['path']).parent
    config,evidence,generated,mapped=adapter.configuration(bootstrap,dataset,system,directory,profile_ref,saved['host_release'],
        topology_ref=mapping['topology'],retained_weights_ref=mapping.get('retained_weights'))
    p.need(evidence==mapping and config==p.read(saved['configs'][dataset]),'historical routing/admission/batching configuration differs from exact mechanical mapping')
    p.need(mapped==p.read(directory/'historical-mapped-config.json'),'historical mapped configuration was altered')
    for path,value in generated.items():p.need(p.read(path)==value,'actual engine template/route differs from policy generation')
    p.need(config['max_service_frequency_mhz']==2100,'wrong actual service frequency domain')
    platform(saved['host_release'],files)
    files.update(saved['files']);files[reference['path']]=reference['sha256']
    return saved


def verify(reference):
    binding=p.checked(reference);pinned_files(binding['files']);e=binding['fresh_legacy_qualification']
    p.need(e['model']=='14b' and binding['model']=='14b' and e['node'] in HOSTNAMES and binding['hostname']==HOSTNAMES[e['node']], 'wrong actual model/node')
    p.need(e['old_node_qualification_inherited'] is False,'old physical qualification inherited')
    boot=p.checked(e['bootstrap']);profile=p.checked(e['profile'])
    p.need(boot['hostname']==binding['hostname'] and boot['model']==binding['model'] and boot['instances']==binding['instances']
        and boot['fresh_node_native_identity'] and boot['old_node_qualification_inherited'] is False,'fresh native bootstrap identity differs')
    deployed=p.checked(e['deployment_bootstrap'])
    p.need({k:v for k,v in boot.items() if k != 'files'}=={k:v for k,v in deployed.items() if k != 'files'}
        and all(boot['files'].get(f)==h for f,h in deployed['files'].items()),'qualification input changed beyond frozen source closure')
    p.need(not boot['output_correctness_verified'] and boot['correctness_gate_required_before_performance'],'bootstrap was not correctness-only')
    owner_ref=p.ref(e['owner_status']);owner=p.checked(owner_ref);terminal_owner(owner)
    p.need(owner['node']==e['node'] and owner['model']=='14b' and owner['hostname']==binding['hostname'] and owner['bootstrap']==e['bootstrap']
        and owner['profile']==e['profile'],'qualification owner/input identity differs')
    files=dict(binding['files']);files[owner_ref['path']]=owner_ref['sha256']
    files.update(native_source(e['source_equivalence'],e['heterogeneous']))
    p.need(p.sha(e['qualifier']['path'])==e['qualifier']['sha256'],'qualification producer source changed')
    freq_ref=e.get('frequency_validator') or p.ref(Path(e['qualifier']['path']).parent/'verify_frequency.py')
    p.need(binding['files'].get(freq_ref['path'])==freq_ref['sha256'],'frequency auditor not frozen in native binding')
    frequency=p.load(freq_ref,'fresh_legacy_independent_frequency').verify(Path(e['frequency_gate']),e['bootstrap'],e['profile'])
    p.need(frequency==p.checked(e['frequency_audit']) and frequency['passed'] and frequency['independently_recomputed'],'frequency evidence differs from independent replay')
    files.update(frequency['files'])
    p.need(p.sha(NATIVE_AUDIT)==NATIVE_AUDIT_SHA,'original native auditor source differs')
    sys.path[:0]=[str(Path(boot['host_release'])/'src'),str(Path(boot['host_release'])),'/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.measurement import power_evidence
    audit=p.load(NATIVE_AUDIT,'fresh_legacy_independent_original_gate')
    proof,raw=audit.audit(Path(e['native_gate']),boot['instances'],binding['system'],power_evidence,hetero=e['heterogeneous'])
    p.need(proof==e['mechanism_proof']==binding['mechanism_proof'],'native mechanism proof differs from raw')
    files.update(raw);files[str(NATIVE_AUDIT)]=NATIVE_AUDIT_SHA
    adapter=p.load(e['policy_adapter'],'fresh_legacy_independent_policy')
    datasets=e['datasets'];p.need(set(datasets)==set(binding['configs'])==set(e['policy_bindings']) and len(datasets)==len(set(datasets)),'dataset policy closure differs')
    mapped=[policy(e['policy_bindings'][d],boot,d,binding['system'],e['profile'],adapter,files) for d in datasets]
    expected=copy.deepcopy(mapped[0]);expected['configs']={d:x['configs'][d] for d,x in zip(datasets,mapped)}
    measured=p.read(Path(e['frequency_gate'])/'status.json')
    retained=measured.get('retained_weights')
    if retained:expected.setdefault('large_inputs',{}).update(retained['rank_files'])
    if binding['system']=='dynamollm':
        p.need(retained and all(p.checked(x['policy_adaptation'])['retained_weights']==retained['manifest'] for x in mapped),'Dynamo cache was not freshly retained by this measured qualification')
    p.need(all(p.checked(x['policy_adaptation'])['topology']==measured['topology'] for x in mapped),'policy topology is not the measured current-node topology')
    allowed={'files','output_correctness_verified','correctness_gate_required_before_performance','correctness_evidence','mechanism_proof','fresh_legacy_qualification'}
    p.need({k:v for k,v in binding.items() if k not in allowed}=={k:v for k,v in expected.items() if k not in allowed},'qualified binding changed outside exact policy merge and native proof')
    p.need(binding['output_correctness_verified'] and binding['correctness_gate_required_before_performance'] is False
        and binding['correctness_evidence']==e['native_gate'],'native qualification not attached to the actual binding')
    for item in mapped:
        p.need(all(binding['files'].get(f)==h for f,h in item['files'].items()),'mapped policy source closure missing')
    for ref in (reference,e['bootstrap'],e['deployment_bootstrap'],e['profile'],e['frequency_audit'],freq_ref,e['qualifier'],e['policy_adapter'],p.ref(__file__)):
        files[ref['path']]=ref['sha256']
    return dict(passed=True,independently_recomputed=True,node=e['node'],model='14b',system=binding['system'],binding=reference,
        datasets=datasets,native_mechanism=proof,frequency_cases=frequency['cases'],fresh_native_identity=True,old_node_qualification_inherited=False,files=files)
