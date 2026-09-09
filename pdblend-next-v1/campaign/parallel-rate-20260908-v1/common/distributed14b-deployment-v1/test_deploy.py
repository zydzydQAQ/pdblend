"""CPU counterexamples use synthetic GPU-free objects and unchanged physical ASTs."""
import ast,asyncio,copy,fcntl,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import deploy
PARENT=deploy.REPO/'campaign/A14B-sharegpt-slo90-v1/deploy.py'
class Deployment(unittest.TestCase):
 def test_physical_control_and_meter_AST_unchanged(self):
  def functions(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(p.read_text()).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
  before,after=functions(PARENT),functions(HERE/'deploy.py')
  for name in ('require_lease','freeze_assets','docker_start_arguments','command','inventory','verify_inventory','capture_previous','free_gpu_snapshot','wait_free','ready','actual_binding','Measurement','execute','measured_ordinary','restore_original'):
   with self.subTest(name=name):self.assertEqual(before[name],after[name])
 def test_original_protocol_and_large_stat_contract(self):
  self.assertEqual(deploy.adapter.PROTOCOL,'per-dataset-slo-five-system-fixed-window-v1')
  self.assertEqual(set(deploy.adapter.stat_identity(__file__)),{'size','mtime_ns','inode','device'})
 def test_actual_lease_required(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'lock'
   with p.open('w') as f,patch.object(deploy,'LOCK_PATH',p):
    with self.assertRaises(ValueError):deploy.require_lease(f)
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);self.assertEqual(deploy.require_lease(f),f.fileno())
 def fixture(self,node='B',stage='pdblend',system=None):
  gpus=[(1,[6]),(1,[7])] if stage=='pdblend' else [(1,[j]) for j in range(8)]
  if node=='C' and system=='distserve':gpus=[(1,[j]) for j in range(5)]+[(2,[6,7])]
  host='/synthetic/host';jobs_ref=dict(path='/synthetic/jobs.json',sha256='jobs');parent=dict(path='/synthetic/parent.json',sha256='913a2d5834dbcc3466cff9d6a45e30cba574069ed50d36c89704ffff89b80438');hostref=dict(path=host+'/manifest.json',sha256='host')
  docs={jobs_ref['path']:dict(node=node,model='14b',dataset='sharegpt' if node=='B' else 'longbench',parent=parent,common_controller_manifest=hostref)};hashes={jobs_ref['path']:'jobs',parent['path']:parent['sha256'],hostref['path']:'host'}
  instances=[]
  for j,(tp,g) in enumerate(gpus):
   cfg='/synthetic/engine'+str(j)+'.json';iid='instance'+str(j);docs[cfg]=dict(id=iid,model='/models/Qwen2.5-14B-Instruct',tp=tp,max_model_len=8192,max_num_seqs=32,max_num_batched_tokens=8192)
   instances.append(dict(id=iid,tp=tp,gpus=g,container_name='slo90-new-'+iid,image=deploy.PDB_IMAGE if stage=='pdblend' else deploy.BASELINE_IMAGE,config=cfg,expected_provenance=dict(source_files_at_import={'source':'hash'})))
  spec=dict(protocol_id=deploy.adapter.PROTOCOL,model='14b',stage=stage,node=node,hostname={'B':'iZwz9i5bte3xkpmcoes3t2Z','C':'iZwz9gfq11hx1sbob59yrgZ'}[node],redistribution_jobs=jobs_ref,files={jobs_ref['path']:'jobs'},instances=instances,host_release=host,host_manifest=hostref,large_inputs={},deployment_budget_s=720,cleanup_budget_s=120,campaign_deadline_s=None,remove_containers=False,baseline_system=system)
  return spec,docs,hashes
 def validate(self,spec,docs,hashes):
  with patch.object(deploy,'read',side_effect=lambda p:docs[str(p)]),patch.object(deploy,'sha',side_effect=lambda p:hashes[str(p)]):deploy.validate_spec(spec,check_host=False)
 def test_B_C_authorized_geometries(self):
  for node,stage,system in [('B','pdblend',None),('C','pdblend',None),('B','baselines','ecoserve'),('C','baselines','mixed'),('C','baselines','distserve')]:self.validate(*self.fixture(node,stage,system))
 def test_wrong_node_or_geometry_rejected(self):
  for kind in ('node','model','host','geometry','image','budget','deadline','remove','source'):
   with self.subTest(kind=kind):
    spec,docs,hashes=self.fixture()
    if kind=='node':spec['node']='A'
    elif kind=='model':spec['model']='32b'
    elif kind=='host':spec['hostname']='elsewhere'
    elif kind=='geometry':spec['instances'][0]['gpus']=[0]
    elif kind=='image':spec['instances'][0]['image']='unknown'
    elif kind=='budget':spec['cleanup_budget_s']=10
    elif kind=='deadline':spec['campaign_deadline_s']=1
    elif kind=='remove':spec['remove_containers']=True
    else:spec['host_manifest']['sha256']='other'
    with self.assertRaises((ValueError,KeyError)):self.validate(spec,docs,hashes)
 def test_B_heterogeneous_not_allowed(self):
  spec,docs,hashes=self.fixture('C','baselines','distserve');spec.update(node='B',hostname='iZwz9i5bte3xkpmcoes3t2Z');docs['/synthetic/jobs.json'].update(node='B',dataset='sharegpt')
  with self.assertRaises(ValueError):self.validate(spec,docs,hashes)
 def test_no_gpu_in_cpu_execute(self):
  with patch.object(deploy,'read',return_value={}),patch.object(deploy,'validate_spec'),patch.object(deploy,'require_lease',side_effect=AssertionError('must not acquire')):
   self.assertFalse(asyncio.run(deploy.execute('/synthetic','/unused',run=False))['hardware_actions'])
 def test_existing_name_and_foreign_container_refused(self):
  row=dict(Name='/old',Id='id',Image='image',State=dict(Running=True,StartedAt='old',Pid=1));previous=dict(instances=[dict(container=dict(name='old',id='id',image='image',StartedAt='old'))]);new=[dict(container_name='slo90-new')]
  self.assertEqual(len(deploy.verify_inventory(previous,[row],new)),1)
  for extra in (dict(row,Name='/foreign'),dict(row,Name='/slo90-new',State=dict(Running=False))):
   with self.assertRaises(ValueError):deploy.verify_inventory(previous,[row,extra],new)
if __name__=='__main__':unittest.main()
