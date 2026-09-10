"""Read-only reconstruction of terminal dynamic evidence; companion metadata only."""
import argparse,copy,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p

def audit(checkpoint,out):
 cp=p.checked(checkpoint);receipt=p.checked(cp['receipt']);row=cp['row'];operation=Path(cp['receipt']['path']).parent
 p.need(cp['measurement_valid'] and receipt['measurement_valid'] and receipt['child_stopped'] and not cp['execution_error'] and not receipt['outer_cleanup_errors'],'clean terminal measurement required')
 for file,digest in cp['artifacts'].items():p.need(p.sha(file)==digest,'original CP artifact changed')
 artifacts=dict(cp['artifacts'])
 for file,digest in receipt['dynamic_artifacts'].items():p.need(p.sha(file)==digest,'external dynamic artifact changed');artifacts[file]=digest
 invocation=p.read(operation/'invocation.json');authority=p.read(operation/'authority.json');job=p.read(operation/'job.json');config=p.read(operation/'parent-bound-config.json');binding=p.checked(cp['binding'])
 p.need(invocation['binding']==binding and invocation['row']==row and p.checked(invocation['template_config'])==p.read(binding['configs'][row['dataset']]),'invocation basis differs')
 p.need(authority['invocation']==p.ref(operation/'invocation.json') and authority['schema']=='capacity-parent-lease-authority-v1','authority invocation differs')
 p.need(job['row']==row and job['config']==str(operation/'parent-bound-config.json') and job['inventory_path']==invocation['capacity_inventory_path']==authority['capacity_inventory_path'],'actual job/inventory mapping differs')
 p.need(authority['expected_job_path']==config['capacity_job_path']==str(operation/'job.json') and config['capacity_lease_authority']==p.ref(operation/'authority.json'),'actual config authority mapping differs')
 for k,v in [('fd','fd'),('holder_pid','holder_pid'),('holder_start_ticks','holder_start_ticks'),('lock_device','device'),('lock_inode','inode')]:p.need(authority[k]==job['lease'][v],'job inherited parent lease differs')
 expected=dict(p.checked(invocation['template_config']),capacity_job_path=str(operation/'job.json'),capacity_lease_authority=p.ref(operation/'authority.json'))
 p.need(config==expected and receipt['parent_authority_verified'],'control config changed outside original parent authority fields')
 p.need(job['initial_instances']==binding['instances'],'actual initial native identity differs')
 source=Path(invocation['executor']['path']);p.need(p.sha(source)==invocation['executor']['sha256'],'invoked measurement source differs')
 sys.path.insert(0,str(source.parent));owner=p.load(source.parent/'dynamic_ownership.py','uniform_dynamic_terminal_owner')
 inventory_path=Path(job['inventory_path']);inventory=owner.inventory(inventory_path,binding['instances'],child_pid=receipt['child_pid'],identity=job['capacity_identity']);owner.validate_terminal(inventory,binding['instances'])
 for file,digest in owner.transition_artifacts(inventory).items():p.need(p.sha(file)==digest,'transition proof changed');artifacts[file]=digest
 artifacts[str(inventory_path)]=p.sha(inventory_path)
 actual=copy.deepcopy(binding);actual['configs']={row['dataset']:job['config']};actual['files']=dict(binding['files'])
 for file in (operation/'invocation.json',operation/'authority.json',operation/'parent-bound-config.json',operation/'job.json'):
  artifacts[str(file)]=p.sha(file);actual['files'][str(file)]=p.sha(file)
 p.need(not out.exists(),'fresh evidence closure required');out.mkdir(parents=True);p.save(out/'actual-invocation-binding.json',actual)
 value=dict(schema='uniform-v2-dynamic-terminal-evidence-closure',passed=True,independently_recomputed=True,checkpoint=checkpoint,
  receipt=cp['receipt'],basis_binding=cp['binding'],actual_invocation_binding=p.ref(out/'actual-invocation-binding.json'),
  invocation=p.ref(operation/'invocation.json'),authority=p.ref(operation/'authority.json'),job=p.ref(operation/'job.json'),
  inventory=p.ref(inventory_path),artifacts=artifacts,expired_parent_lease_not_reacquired=True,
  actual_lease_evidence='original child verifies live inherited FLOCK; terminal audit matches pinned job/authority/holder identity',
  generated_s=time.time())
 p.save(out/'audit.json',value);return p.ref(out/'audit.json')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pipeline',type=Path,required=True);a=ap.parse_args();meta=a.pipeline/'metadata.json'
 state=dict(schema='uniform-v2-node-evidence-companion',setup_energy_index=p.ref(HERE/'setup-energy-index-001.json'),evidence_closures=[],checked_checkpoint_paths=[],errors=[])
 p.save(meta,state)
 while True:
  status=p.read(a.pipeline/'status.json')
  for reference in status.get('observations',[]):
   observation=p.checked(reference)
   if observation['system']!='pdblend':continue
   cp=observation['checkpoint']
   if cp['path'] in state['checked_checkpoint_paths']:continue
   try:
    closure=audit(cp,a.pipeline/'evidence-closures'/observation['cell_id']);state['evidence_closures'].append(closure)
   except BaseException as exc:state['errors'].append(dict(checkpoint=cp,error=repr(exc)))
   state['checked_checkpoint_paths'].append(cp['path']);state['updated_s']=time.time();p.save(meta,state)
  if status.get('finished_s'):
   state['complete']=not state['errors'];state['finished_s']=time.time();p.save(meta,state);return
  time.sleep(5)
if __name__=='__main__':main()
