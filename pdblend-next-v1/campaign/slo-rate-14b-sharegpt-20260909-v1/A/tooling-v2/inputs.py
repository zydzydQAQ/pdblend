from pathlib import Path
import copy,json,hashlib,os,socket,subprocess,time
import bootstrap as b
import power_selftest as p
E=p.HERE.parents[1]/'env'
N=E.parent
COMMON=E/'run.py'
def ref(x):return p.ref(x)
def files():
 d=p.source_check()
 for x in p.HERE.glob('*.py'):d[str(x)]=p.sha(x)
 for x in [COMMON, E/'profiles.json',E/'model-manifest.json',E/'sharegpt-template.json',E/'source-mapping.json']:
  d[str(x)]=p.sha(x)
 return d

def validate_bootstrap(s):
 assert s['schema']=='slo14-fresh-native-bootstrap-v1' and s['node']==p.NODE and s['hostname']==p.EXPECTED_HOSTNAME
 assert s['model']=='14b' and s['old_node_qualifications_inherited'] is False
 assert all(p.sha(x)==v for x,v in s['files'].items())
 pre=b.checked(s['environment_preflight']);assert pre['passed'] and pre['hostname']==p.EXPECTED_HOSTNAME
 assert pre['model_manifest']==s['model_manifest'] and pre['model_all_shards_verified']
 assert s['image']=='sha256:0bb51d143b7fcaaea2e794dd6e207cf4165a4f21522a2e932a4bd4a117074bc2'
 assert [i['gpus'] for i in s['instances']]==[[6],[7]] and all(i['tp']==1 for i in s['instances'])
 return s

def bootstrap_spec():
 out=N/p.NODE/'preparation';out.mkdir(parents=True,exist_ok=True)
 pre=p.ref(out/'preflight.json');b.checked(pre)
 ids=[]
 for g in [6,7]:ids.append(dict(id=f'slo14{p.NODE.lower()}{g}',tp=1,gpus=[g],role='mixed',port=35300+g,kv_port=55000+g*16,url=f'http://127.0.0.1:{35300+g}',container_name=f'slo14-{p.NODE}-pdb-gpu{g}',native_kind='v3',scheduler_cache_observed=True,scheduler_cache_count=1,service_budget_tokens=2048,restore_budget_tokens=8192))
 env=dict(PYTHONPATH=str(E/'native-runtime/src'),PYTHONUNBUFFERED='1',PDBLEND_ASYNC_IO='1',PDBLEND_ENGINE_TIMING='1',NCCL_CUMEM_ENABLE='0',NCCL_DEBUG='WARN',NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='1',VLLM_HOST_IP='127.0.0.1')
 original=b.checked(ref(E/'original-bootstrap-spec.json'))
 source_files={str(E/'native-runtime/src/ecopadg/serving'/Path(x).name):p.sha(E/'native-runtime/src/ecopadg/serving'/Path(x).name) for x in original['expected_provenance']['source_files_at_import']}
 spec=dict(schema='slo14-fresh-native-bootstrap-v1',node=p.NODE,hostname=p.EXPECTED_HOSTNAME,model='14b',image='sha256:0bb51d143b7fcaaea2e794dd6e207cf4165a4f21522a2e932a4bd4a117074bc2',engine_module='ecopadg.serving.engine',environment=env,instances=ids,expected_provenance=dict(model='/models/Qwen2.5-14B-Instruct',tp=1,source_files_at_import=source_files),old_node_qualifications_inherited=False,environment_preflight=pre,node_identity=ref(p.IDENTITY),model_manifest=ref(E/'model-manifest.json'),engine_template=ref(E/'native-template.json'),numerical_reference=ref(E/'numerical-reference.json'),common_executor=ref(COMMON),files=files())
 for k in ['environment_preflight','node_identity','model_manifest','engine_template','numerical_reference']:
  spec['files'][spec[k]['path']]=spec[k]['sha256']
 b.save(out/'bootstrap-spec.json',spec);validate_bootstrap(spec);return ref(out/'bootstrap-spec.json')

def boot_record(statuspath):
 statusref=ref(statuspath);s=b.checked(statusref)
 assert s['complete'] and s['ordinary_passed'] and s['setup_measurement']['measurement_valid']
 boot=dict(schema='slo14-cold-bootstrap-proof-v1',node=p.NODE,model='14b',hostname=s['hostname'],instances=s['instances'],complete=True,ordinary_passed=True,node_lease_held=False,setup_measurement=s['setup_measurement'],ordinary=s['ordinary'],status=statusref,spec=s['spec'],files=files(),old_node_qualification_inherited=False)
 for x in Path(statuspath).parent.rglob('*'):
  if x.is_file() and 'native' not in x.relative_to(Path(statuspath).parent).parts and x.name not in ['bootstrap.json','bootstrap-proof-v2.json']:boot['files'][str(x)]=p.sha(x)
 path=Path(statuspath).parent/'bootstrap-proof-v2.json';b.save(path,boot);return ref(path)

def fixed(bootref,out):
 import restore,qualify_fixed as q
 restore.audit(bootref);boot=b.checked(bootref);out=Path(out);assert not out.exists();out.mkdir(parents=True)
 profile=ref(E/'profiles.json');shape=q.shapes(b.checked(profile));assert len(shape)==41
 template=ref(E/'sharegpt-template.json');cfg=b.checked(template);assert cfg['idle_domain_reacquire_timeout_s']==1.5 and cfg['capacity_integration_v1'] is False
 oracle=dict(schema='slo14-cancellation-input-v1',cases=[dict(prompt_length=128,prompt=([9707,1879,13]*43)[:128])],native_restoration=bootref)
 b.save(out/'cancellation-input.json',oracle)
 fs=files();fs.update(boot['files'])
 for r in [bootref,template,boot['ordinary'],ref(out/'cancellation-input.json')]:fs[r['path']]=r['sha256']
 s=dict(schema='migration-B-fixed14B-qualification-spec-v1',bootstrap=bootref,profile=profile,stream=ref(p.HERE/'stream.py'),model_manifest=ref(E/'model-manifest.json'),numerical_reference=ref(out/'cancellation-input.json'),common_executor=ref(COMMON),config_templates={'sharegpt':template},source_config_template=template,authorized_slo=dict(ttft_s=5.,tpot_s=.15),shapes=[list(v) for v in shape],files=fs,all_mixed_tp1_shapes_fresh_observation_required=True,old_node_qualifications_inherited=False)
 b.save(out/'spec.json',s);q.validate(s);return ref(out/'spec.json')

def idle(prior,out):
 import verify_fixed
 v=verify_fixed.verify(prior);assert v['native_shape_cases']==82 and v['v3_cancellations']==2 and v['node']==p.NODE
 out=Path(out);assert not out.exists();out.mkdir(parents=True);q=b.checked(prior)
 fs=files();fs.update(q['files']);fs.update(q['source_files']);fs[prior['path']]=prior['sha256']
 s=dict(schema='migration-B-fixed14B-idle-spec-v1',previous_qualification=prior,common_executor=ref(COMMON),idle_timeout_s=1.5,files=fs)
 b.save(out/'spec.json',s);return ref(out/'spec.json')
