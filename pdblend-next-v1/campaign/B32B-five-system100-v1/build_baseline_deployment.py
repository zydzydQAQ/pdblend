"""Build a B-specific spec for the shared frozen deployment machinery; CPU only."""
import argparse,copy,hashlib,json,pathlib
R=pathlib.Path('/root/workspace/pdblend-next-v1');P=R/'campaign/B32B-five-system100-v1'
IMAGE='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'
DEADLINE=1788872770.0400891

def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise RuntimeError(why)
def write(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def build(out,deploy_release,executor):
    require(not out.exists(),'new deployment output required')
    for pkg in (deploy_release,executor):require((pkg/'manifest.json').is_file(),'shared deployment/executor must be frozen first')
    binding_path=P/'binding.pdblend.r2.json';b=read(binding_path);workloads=R/'campaign/five-system-fixed-window-v1/sources/B32B/manifest.json'
    require(b['model']=='32b' and b['system']=='pdblend' and read(workloads)['model']=='32b','B source identity differs')
    old=read(P/'baseline-container-templates.json');require(len(old)==4,'four original templates required')
    entry=R/'campaign/B32B-baseline-fixed-window-preparation-v1/legacy-observation-candidate/engine.py'
    require(sha(entry)=='c4763c3761acd9fff20bad4ac3d254a1ea3ddef41c7732b63ab698dfd1f5b146','observational engine changed')
    require(all(c['Image']==IMAGE and not c['State']['Running'] and c['HostConfig']['NetworkMode']=='host' and c['HostConfig']['IpcMode']=='host' for c in old),'old baseline image/state/network differs')
    instances=[];out.mkdir(parents=True)
    for j,c in enumerate(old):
        require(c['Name']==f'/pdb-v2-b32q{j}','historical instance order differs')
        cfg=read(P/f'engine-configs/engine-{j}.json');gpus=[2*j,2*j+1];rid=f'base100b{j}'
        require(cfg['id']==rid and cfg['tp']==2 and cfg['model']=='/models/Qwen2.5-32B-Instruct' and cfg['max_model_len']==cfg['max_num_batched_tokens']==8192 and cfg['max_num_seqs']==32,'engine static work limits changed')
        # Independent output owns all newly started runtime state.
        cfg['runtime_dir']=str(out/'runtime');config=out/'engines'/f'{rid}.json';write(config,cfg)
        env=list(c['Config']['Env']);require(f'CUDA_VISIBLE_DEVICES={2*j},{2*j+1}' in env,'actual original GPU env differs')
        env=[v for v in env if not v.startswith('PYTHONDONTWRITEBYTECODE=')]+['PYTHONDONTWRITEBYTECODE=1']
        instances.append(dict(id=rid,tp=2,gpus=gpus,role='mixed',port=cfg['port'],kv_port=cfg['kv_port'],url=f"http://127.0.0.1:{cfg['port']}",container_name='pdb-v2-'+rid,config=str(config),engine_entry=str(entry),image=IMAGE,environment=env,mounts=c['Mounts'],historical_container_id=c['Id'],historical_engine_id=f'b32q{j}',native_kind='legacy_sync_put',scheduler_cache_observed=False))
    files={}
    def freeze(path,digest=None):
        path=pathlib.Path(path).resolve();h=sha(path);require(digest is None or h==digest,'source changed: '+str(path));files[str(path)]=h
    for path in (pathlib.Path(__file__),binding_path,workloads,P/'baseline-container-templates.json',entry.parent/'manifest.json'):
        freeze(path)
    for name,h in read(entry.parent/'manifest.json')['files'].items():freeze(entry.parent/name,h)
    host=pathlib.Path(b['host_release']);freeze(host/'manifest.json')
    for name,h in read(host/'manifest.json')['files'].items():freeze(host/name,h)
    for pkg in (deploy_release,executor):
        freeze(pkg/'manifest.json')
        for name,h in read(pkg/'manifest.json')['files'].items():freeze(pkg/name,h)
    for i in instances:freeze(i['config'])
    for path in (P/'configs').glob('*.json'):freeze(path)
    freeze(P/'inputs/profiles.baseline.json')
    correctness=R/'campaign/B32B-legacy-baseline-correctness-v1';freeze(correctness/'manifest.json')
    for name,h in read(correctness/'manifest.json')['files'].items():freeze(correctness/name,h)
    spec=dict(schema=1,protocol_id=b['protocol_id'],model='32b',layout='resident',hostname=b['hostname'],deadline_s=DEADLINE,out=str(out),host_release=str(host),executor_release=str(executor),instances=instances,required_predecessors=[dict(binding=str(binding_path),manifest=str(workloads),system='pdblend',datasets=['alpaca','sharegpt','longbench'])],previous_binding=str(binding_path),pdb_binding=str(binding_path),workloads=str(workloads),image=IMAGE,files=files,supported_datasets=['alpaca','sharegpt','longbench'],source_entry=str(entry),deployment_budget_s=720,cleanup_budget_s=120,does_not_certify_kv_output_correctness=True,correctness_gate={'package':str(correctness),'manifest_sha256':sha(correctness/'manifest.json'),'separate_node_lease':True},algorithm_scope='four TP2 original resident baselines; original d114 engine with observational fields only; no v3 temporal substitution',deployment_implementation=str(deploy_release/'deploy.py'))
    write(out/'deployment.json',spec);return spec
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--deploy-release',type=pathlib.Path,required=True);p.add_argument('--executor',type=pathlib.Path,required=True);a=p.parse_args();s=build(a.out.resolve(),a.deploy_release.resolve(),a.executor.resolve());print(json.dumps(dict(spec=str(a.out/'deployment.json'),sha256=sha(a.out/'deployment.json'),hardware_actions=False)))
