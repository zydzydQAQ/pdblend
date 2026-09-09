"""Target-B CPU asset verification and immutable PDB deployment declaration."""
import copy,hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parent;W=R.parent.parent
D=B/'distributed-14b-v1';CODE=R/'common/distributed14b-deployment-v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def need(ok,msg):
 if not ok:raise ValueError(msg)
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 need(socket.gethostname()=='iZwz9i5bte3xkpmcoes3t2Z','target-B asset identities required');need(not (D/'assets-release.json').exists(),'immutable new preparation')
 manifest=read(CODE/'manifest.json');need(all(sha(p)==h for p,h in manifest['files'].items()),'generic deployment source differs')
 jobs=R/'distributed-14b-v1/B-jobs.json';assignment=read(jobs);need(assignment['node']=='B' and assignment['dataset']=='sharegpt' and assignment['capacity_integration_v1'] is False,'only assigned B ShareGPT')
 parent=assignment['parent'];need(parent['sha256']=='913a2d5834dbcc3466cff9d6a45e30cba574069ed50d36c89704ffff89b80438' and sha(parent['path'])==parent['sha256'],'exact assigned scientific group required')
 terminal=B/'eco-drain37-v1/final-observations/completion-audit.json';old=read(terminal);need(old['all_required_observations_complete'] and old['remaining_required_count']==0,'old32B is not terminal')
 external=W/'campaign/A14B-sharegpt-slo90-v1';source=read(external/'prepared-003/release.json');prior=read(external/'execution/deployment/pdblend/bindings/pdblend/binding.json')
 release=dict(schema='distributed14b-B-predeployment-assets-v1',model='14b',node='B',redistribution_jobs=ref(jobs),deployment_root=str(D/'deployment'),common_dir=str(R/'common/execution-until-complete-v1'),host_releases=dict(pdblend=str(Path(assignment['common_controller_manifest']['path']).parent),baselines=str(W/'releases/five-system100-A14B-baseline-v1-runtime')),model_root='/root/workspace/models',engine_source_release=source['engine_source_release'],baseline_engine_entry=source['baseline_engine_entry'],baseline_engine_pythonpath=source['baseline_engine_pythonpath'],container_prefix='slo90-distributed14b-B',required_inputs=source['required_inputs'],verified_large_inputs=prior['large_inputs'],old32_terminal=ref(terminal),source_package=ref(CODE/'manifest.json'),native_import_source_must_be_verified_after_start=True,cross_host_reference_profile=assignment['profile_reference'],cross_host_profile_qualified=False)
 write(D/'assets-release.json',release)
 sys.path.insert(0,str(CODE));import deploy
 result=deploy.prepare_spec('pdblend',D/'assets-release.json',socket.gethostname(),B/'redistribute14b-readonly-inventory-001/native-runtime-readonly.json',B/'eco-drain-qualification-001/ecoserve/binding.json')
 write(D/'assets-preparation.json',dict(complete=True,hardware_actions=False,host=socket.gethostname(),pid=os.getpid(),time_s=time.time(),pdb_deployment_spec=result,assets_release=ref(D/'assets-release.json')))
 print(json.dumps(dict(complete=True,hardware_actions=False,pid=os.getpid(),spec=result)),flush=True)
if __name__=='__main__':main()
