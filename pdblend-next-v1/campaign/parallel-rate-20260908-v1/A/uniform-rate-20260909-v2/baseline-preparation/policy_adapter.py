"""Mechanical historical policy binding; physical qualification is separate."""
import copy
from pathlib import Path
import sys

ROOT=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
POLICY=p.REPO/'campaign/AC-baseline-100s-preparation-v1'


def configuration(bootstrap,dataset,system,out,profile_ref,host_release,*,topology_ref,retained_weights_ref=None):
    p.need(bootstrap['model']=='14b' and dataset in ('alpaca','sharegpt','longbench')
           and system in ('mixed','distserve','dynamollm','ecoserve'),'unsupported historical identity')
    old=p.load(POLICY/'bind.py','new_node_historical_policy')
    historical,note=old.configuration('14b',dataset,system)
    roles={(i['tp'],tuple(i['gpus'])):i['role'] for i in historical['instances']}
    layout=[]
    for i in bootstrap['instances']:
        key=i['tp'],tuple(i['gpus'])
        p.need(key in roles,'native TP/GPU geometry differs from historical policy')
        layout.append(dict(id=i['id'],tp=i['tp'],gpus=i['gpus'],port=i['port'],kv_port=i['kv_port'],url=i['url'],
            container_name=i['container']['name'],role=roles[key]))
    deployment=dict(model='14b',historical_engine_observation_only=True,
        layouts={system+':'+dataset:layout},topology_source_path_verified=system=='dynamollm')
    mapped,note=old.configuration('14b',dataset,system,deployment=deployment,out=out/'unused')
    original_profile=p.read(historical['profiles'])
    profile=p.checked(profile_ref)
    p.need(profile['points']==[x for x in original_profile['points'] if x['frequency_mhz']<=2100],
           'baseline must retain all original <=2100 points in their original order and values')
    p.need(topology_ref and p.sha(topology_ref['path'])==topology_ref['sha256'],'explicit current-node topology required')
    result=copy.deepcopy(mapped)
    result.update(profiles=profile_ref['path'],interconnect=topology_ref['path'],
        max_service_frequency_mhz=2100,comparison_system=system,controller_source_release=str(host_release))
    generated={}
    if system=='dynamollm':
        p.need(retained_weights_ref,'full Dynamo requires an actual fresh retained-weight manifest')
        cache=p.checked(retained_weights_ref)
        p.need(cache.get('complete') is True and cache.get('ranks') and cache.get('tp') in (1,2,4,8),
               'fresh retained-weight manifest incomplete')
        result['retained_weights']=str(Path(retained_weights_ref['path']).parent)
        engine=p.read(bootstrap['instances'][0]['engine_config'])
        imported=bootstrap['instances'][0]['provenance']['source_files_at_import']
        entries=[path for path in imported if Path(path).name=='engine.py']
        p.need(len(entries)==1,'one actually imported legacy engine entry required')
        engine.update(observation_engine_entry=entries[0],observation_engine_sha256=imported[entries[0]])
        target=out/'engine-template.json';generated[target]=engine
        result['topology'].update(runtime_dir=str(out/'dynamic-runtime'),engine_template=str(target))
    permitted={'profiles','interconnect','max_service_frequency_mhz','comparison_system','controller_source_release'}
    if system=='dynamollm':permitted.update(('retained_weights','topology'))
    p.need({k for k in set(result)|set(mapped) if result.get(k)!=mapped.get(k)}<=permitted,'unexpected controller policy change')
    evidence=dict(schema='uniform-v2-historical-baseline-policy-map',model='14b',dataset=dataset,system=system,
        historical_policy=note,profile_parent=p.ref(historical['profiles']),profile=profile_ref,
        maximum_service_frequency_mhz=2100,original_eligible_point_count=len(profile['points']),
        topology=topology_ref,retained_weights=retained_weights_ref,policy_flow_sorting_budgets_unchanged=True,
        historical_metadata_is_not_new_node_qualification=True)
    return result,evidence,generated,mapped


def build(binding_ref,dataset,system,out,profile_ref,host_release=None,*,topology_ref=None,retained_weights_ref=None):
    out=Path(out).resolve();p.need(not out.exists(),'fresh mapped policy directory required')
    bootstrap=p.checked(binding_ref)
    host=Path(host_release) if host_release else HERE/'platform-domain'/('runtime-baseline-eco-drain-002' if system=='ecoserve' else 'runtime-baseline-002')
    manifest=p.read(host/'manifest.json')
    p.need(manifest['max_service_frequency_mhz_default']==2520 and manifest['configuration_key']=='max_service_frequency_mhz',
           'explicit service ceiling source required')
    config,evidence,generated,mapped=configuration(bootstrap,dataset,system,out,profile_ref,host,
        topology_ref=topology_ref,retained_weights_ref=retained_weights_ref)
    out.mkdir(parents=True)
    config_path=out/'configs'/(dataset+'.json');p.save(config_path,config)
    p.save(out/'policy-map.json',evidence);p.save(out/'historical-mapped-config.json',mapped)
    for path,value in generated.items():p.save(path,value)
    frozen=dict(bootstrap['files'])
    frozen.update({str(host/name):digest for name,digest in manifest['files'].items()})
    refs=[binding_ref,profile_ref,topology_ref,p.ref(host/'manifest.json'),p.ref(__file__),
          p.ref(out/'policy-map.json'),p.ref(out/'historical-mapped-config.json'),p.ref(POLICY/'bind.py')]
    if retained_weights_ref:refs.append(retained_weights_ref)
    for ref in refs:frozen[ref['path']]=ref['sha256']
    for path in [config_path,*generated]:frozen[str(path)]=p.sha(path)
    def external(value,key=''):
        if isinstance(value,dict):
            for k,v in value.items():external(v,k)
        elif isinstance(value,list):
            for v in value:external(v,key)
        elif isinstance(value,str) and value.startswith('/') and key!='journal' and Path(value).is_file():
            frozen[value]=p.sha(value)
    external(config)
    binding=copy.deepcopy(bootstrap)
    binding.update(system=system,implementation_variant=system,host_release=str(host),configs={dataset:str(config_path)},
        output=str(out/'results'),files=frozen,policy_adaptation=p.ref(out/'policy-map.json'),
        correctness_gate_required_before_performance=True,output_correctness_verified=False,formal_eligible=False,
        platform_source=p.ref(host/'manifest.json'))
    # This object is intentionally not a performance qualification. The caller
    # binds its real native/frequency gate and independently verifies the result.
    p.save(out/'binding.json',binding)
    return dict(binding=p.ref(out/'binding.json'),policy_map=p.ref(out/'policy-map.json'),
                qualification_required=True)
