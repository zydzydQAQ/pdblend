"""CPU-review draft: exact A retained restart, parameterized final predecessor."""
import argparse,asyncio,copy,hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
CAMPAIGN=Path('/root/workspace/pdblend-next-v1/campaign')
ORIGINAL=CAMPAIGN/'AC-baseline-deployment-prepared-v1/A-resident'
PARENT=HERE/'restore_parent.py'
COMMON=CAMPAIGN/'parallel-rate-20260908-v1/common/execution-until-complete-v1/run.py'
HOST='iZwz92bdfqihqp38tekqjyZ'
DEADLINE=None
OUT=PREVIOUS=RELEASE=TERMINAL=INVENTORY=None
REFS={}
def require(ok,why):
 if not ok:raise RuntimeError(why)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def pinned(r):require(sha(r['path'])==r['sha256'],'pinned input changed '+r['path']);return read(r['path'])
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False
def checked_reference(value,digest=None):return pinned(value if isinstance(value,dict) else dict(path=value,sha256=digest))

def terminal_contract(audit):
 require(audit['schema']=='A-final-pdb-terminal-for-baseline-restore-v1' and audit['passed'] is True and audit['model']=='14b','explicit final A terminal audit required')
 require(audit['release']==REFS['release'] and audit['previous_binding']==REFS['previous'],'terminal audit references another release/physical binding')
 require(audit['files'] and all(sha(p)==h for p,h in audit['files'].items()),'terminal audit raw inputs changed')
 require(audit.get('capacity_cleanup_verified') is True and audit.get('capacity_cleanup_evidence'),'capacity cleanup audit/evidence must be explicit')
 for item in audit['capacity_cleanup_evidence']:
  value=pinned(item)
  require(audit['files'].get(item['path'])==item['sha256'],'capacity cleanup evidence outside audit SHA set')
  # The owner's final capacity audit may reference a full operation, a final
  # controller drain, or a serving receipt. No future success is inferred here.
  if 'summary' in value:
   value=value['summary'].get('post_measurement_cleanup',{})
  require(value.get('cleanup_complete') is True or value.get('drain_complete') is True,'actual capacity cleanup is not complete')
  require(not value.get('cleanup_errors') and not value.get('errors'),'capacity cleanup retains errors')
 require(audit['stages'],'no measured final PDB stages declared')
 proofs=[]
 for stage in audit['stages']:
  status=pinned(stage['status']);pinned(stage['declaration'])
  require(status.get('complete') is True and not status.get('failed') and not status.get('error') and not status.get('engineering_gate_failed') and status.get('node_lease_held') is False and not alive(status['pid']),'PDB stage has not naturally completed and exited')
  ids=set(stage['expected_cell_ids']);require(ids and ids==set(status['completed']) and len(ids)==len(stage['expected_cell_ids']),'final PDB measured set differs')
  cpdir=Path(stage['checkpoint_root']);require({p.stem for p in cpdir.glob('*.json')}==ids,'final checkpoint set differs')
  for cid in sorted(ids):
   path=cpdir/(cid+'.json');cp=read(path)
   require(audit['files'].get(str(path))==sha(path),'final checkpoint is not SHA-bound by terminal audit')
   require(cp.get('measurement_valid') is True and cp.get('work_complete') is True and cp['artifacts'] and all(sha(p)==h for p,h in cp['artifacts'].items()),'PDB checkpoint/raw work invalid')
   receipt=checked_reference(cp['receipt'],cp.get('receipt_sha256'));binding=checked_reference(cp['binding'],cp.get('binding_sha256'))
   require(binding['model']=='14b' and binding['system']=='pdblend','wrong actual measured model/system')
   require(all(sha(p)==h for p,h in binding['files'].items()),'actual measured binding inputs changed')
   require(receipt.get('measurement_valid') is True and receipt.get('child_stopped') is True and receipt.get('clock_restore_complete') is True and receipt.get('child_exitcode')==0 and not receipt.get('outer_cleanup_errors') and not receipt.get('sampling_error'),'PDB receipt cleanup/clock/child failed')
   summary=receipt['summary'];require(summary.get('work_complete') is True and not summary.get('failed_requests') and not summary.get('request_timeouts') and not summary.get('admission_rejections'),'PDB request failure blocks restoration')
   require(summary.get('post_measurement_cleanup',{}).get('cleanup_complete') is True,'PDB native cleanup incomplete')
   gate=pinned(stage['engineering_gates'][cid]);require(gate.get('passed') is True and not gate.get('errors') and not gate.get('faults'),'PDB engineering gate failed')
  proofs.append(dict(status=stage['status'],count=len(ids)))
 return dict(passed=True,stages=proofs,audit=REFS['terminal'])

def process_scan():
 others=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:argv=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  except (OSError,UnicodeError):continue
  if 'python' in argv and '/campaign/' in argv and '-c ' not in argv and 'ecopadg.serving.engine' not in argv:others.append(dict(pid=proc.name,argv=argv))
 return dict(no_live_serving_child=not others,competing_processes=others)
def verify_release(path,digest):
 require(dict(path=str(Path(path)),sha256=digest)==REFS['release'],'wrong declared final release')
 release=pinned(REFS['release']);require(release.get('files') and all(sha(p)==h for p,h in release['files'].items()),'released files changed')
 result=terminal_contract(pinned(REFS['terminal']));scan=process_scan();require(scan['no_live_serving_child'],'competing campaign process '+str(scan));return result
def package_check():
 m=read(OUT/'adapter-manifest.json')
 for p,h in m['files'].items():require(sha(p)==h,'restore source changed '+p)
 return m
def expected():
 original=read(ORIGINAL/'deployment.json');receipt=read(ORIGINAL/'deployment-receipt.json');inventory={r['Name'].lstrip('/'):r for r in read(ORIGINAL/'containers.after.json')}
 require(original['model']=='14b' and original['hostname']==HOST and original['layout']=='resident','original A resident deployment required')
 require(receipt['complete'] and receipt['measurement_valid'],'original retained deployment invalid')
 require([(i['tp'],i['gpus']) for i in original['instances']]==[(1,[j]) for j in range(8)],'exact original eightTP1 layout required')
 snapshot=pinned(REFS['inventory']);rows=snapshot['containers'];actual={r['Name'].lstrip('/'):r for r in rows};created={r['name']:r['container_id'] for r in receipt['created']}
 require(snapshot['hostname']==HOST,'retained snapshot belongs to another host')
 parent=load(PARENT,'a_restore_draft_static');containers={};provenance={}
 for i in original['instances']:
  name=i['container_name'];old=inventory[name];current=actual[name];require(old['Id']==created[name]==current['Id'] and current['State']['Running'] is False,'same retained stopped baseline ID required')
  parent.static_container(current,old);containers[name]=old;provenance[i['id']]=receipt['new_provenance'][i['id']]
 previous=pinned(REFS['previous']);require(previous['model']=='14b' and previous['system']=='pdblend' and previous['hostname']==HOST and previous['deadline_s'] is None and previous['campaign_lifecycle']=='until_declared_complete_v1','actual final no-deadline PDB binding required')
 require(previous==read(PREVIOUS),'prepared previous binding changed')
 files=dict(original['files']);files.update(previous['files']);files.update(package_check()['files']);files.update(pinned(REFS['terminal'])['files'])
 for r in REFS.values():files[r['path']]=r['sha256']
 for q in (PREVIOUS,OUT/'adapter-manifest.json',ORIGINAL/'deployment.json',ORIGINAL/'deployment-receipt.json',ORIGINAL/'containers.after.json'):files[str(q)]=sha(q)
 result=copy.deepcopy(original);result.update(schema=2,out=str(OUT),operation='restart-retained-residents',previous_binding=str(PREVIOUS),pdb_binding=str(PREVIOUS),model_main_release=str(RELEASE),model_main_release_sha256=REFS['release']['sha256'],deadline_s=None,campaign_lifecycle='until_declared_complete_v1',executor_release=str(COMMON.parent),files=files,expected_containers=containers,expected_provenance=provenance,required_predecessors=[dict(terminal_audit=REFS['terminal'])],new_container_creation_allowed=False,output_correctness_verified=False,fresh_correctness_gate_required=True,baseline_policies_and_engine_source_unchanged=True)
 return result
def validate_spec(spec):require(spec==expected(),'exact A restore declaration changed')
def bind_parent(spec):
 parent=load(PARENT,'a_restore_draft_hardware');parent.HOST=HOST;parent.COMMON=COMMON;parent.package_check=package_check;parent.validate_spec=validate_spec
 parent.validate_previous=lambda value:require(value==read(PREVIOUS),'actual PDB identity changed')
 old_load=parent.load
 def wrapped(path,name):
  if Path(path)==parent.BARRIER:return sys.modules[__name__]
  m=old_load(path,name)
  if Path(path)==parent.DEPLOY:
   def terminal(binding,manifest,**kwargs):
    require(Path(binding)==PREVIOUS and Path(manifest)==Path(spec['workloads']),'unexpected actual predecessor arguments')
    return verify_release(spec['model_main_release'],spec['model_main_release_sha256'])
   m.terminal_group=terminal
  return m
 parent.load=wrapped
 # native_dispatch below preserves the AST of A's existing p4v2 adapter.
 legacy_native=parent.native
 async def native_dispatch(session,instance,limit,records,common,result):
     if instance['native_kind']=='legacy_sync_put':
         return await legacy_native(session,instance,limit,records,common,result)
     require(instance['native_kind']=='v3' and instance['id'] in {i['id'] for i in read(PREVIOUS)['instances']},
             'v3 is allowed only for the exact current PDB predecessor')
     result.update(complete=False,errors=[],native_protocol='original_v3_full_rank_barrier_with_bounded_owner_resume')
     try:
         before=await parent.idle(session,instance,limit,records,common);result['before']=before
         proof=await parent.http(session,instance,'/drain',dict(expected_generation=before['generation']),limit,records)
         result['proof']=proof;common.barrier(before,proof,instance)
         paused=await parent.idle(session,instance,limit,records,common)
         require(paused['generation']==proof['generation'] and paused.get('accepting') is False,'v3 drain generation changed')
         payload=dict(generation=paused['generation']+1,role=instance.get('role','mixed'),mode='continuous',admit_prefill=True,admit_decode=True)
         tokens=instance.get('restore_budget_tokens')
         if tokens is not None:payload['scheduler_budget']=dict(schema_version=1,max_num_batched_tokens=tokens,max_num_seqs=32)
         result['resumed']=dict(before=paused,control=payload)
         reply=await parent.http(session,instance,'/control',payload,limit,records)
         after=await parent.idle(session,instance,limit,records,common);result['resumed'].update(reply=reply,after=after)
         require(reply.get('generation')==payload['generation']==after['generation'] and after.get('accepting') is True,'v3 owner resume not acknowledged')
         for key in ('role','mode','admit_prefill','admit_decode'):require(after.get(key)==payload[key],'v3 resume state mismatch: '+key)
         if tokens is not None:
             effective=after.get('scheduler_budget_effective',{})
             require(effective.get('max_num_batched_tokens')==tokens and effective.get('max_num_seqs')==32,'v3 restored budget not effective')
         result['complete']=True
     except BaseException as exc:
         result['errors'].append(repr(exc));raise
     finally:result['finished_s']=time.time()
 parent.native=native_dispatch
 return parent

def main():
 global OUT,PREVIOUS,RELEASE,TERMINAL,INVENTORY,REFS
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--out',type=Path,required=True)
 for name in ('previous-binding','release','terminal-audit','retained-inventory'):
  parser.add_argument('--'+name,type=Path);parser.add_argument('--'+name+'-sha256')
 a=parser.parse_args();require(not(a.prepare and a.run),'one mode only');OUT=a.out.resolve();PREVIOUS=OUT/'previous-binding.json'
 if a.prepare:
  require(not OUT.exists(),'new independent restoration preparation required')
  names={'previous':'previous_binding','release':'release','terminal':'terminal_audit','inventory':'retained_inventory'}
  for key,name in names.items():
   path=getattr(a,name);digest=getattr(a,name+'_sha256');require(path and digest,'explicit '+name+' path/SHA required')
   REFS[key]=dict(path=str(path.resolve()),sha256=digest);pinned(REFS[key])
 else:REFS=read(OUT/'input-references.json')
 RELEASE=Path(REFS['release']['path']);TERMINAL=Path(REFS['terminal']['path']);INVENTORY=Path(REFS['inventory']['path'])
 if a.prepare:
  terminal_contract(pinned(REFS['terminal']))
  OUT.mkdir();PREVIOUS.write_bytes(Path(REFS['previous']['path']).read_bytes());write(OUT/'input-references.json',REFS)
  original_parent=CAMPAIGN/'A14B-resident-restore-v3';m=read(original_parent/'manifest.json');files=dict(m['dependencies']);files.update({str(original_parent/n):h for n,h in m['files'].items()});files[str(original_parent/'manifest.json')]=sha(original_parent/'manifest.json')
  for path in (PARENT,Path(__file__).resolve(),OUT/'input-references.json',COMMON,COMMON.parent/'manifest.json',COMMON.parent/'child.py',COMMON.parent/'cpu-validation.json'):files[str(path)]=sha(path)
  write(OUT/'adapter-manifest.json',dict(files=files,parent_source=ref(original_parent/'restore.py'),operation_budgets_s=dict(preflight=120,restore=720,cleanup=120),total_deadline_s=None))
  write(OUT/'deployment.json',expected());print(json.dumps(dict(prepared=True,cpu_only=True,spec=ref(OUT/'deployment.json'),hardware_actions=False)));return
 package_check();spec=read(OUT/'deployment.json');validate_spec(spec)
 if not a.run:print(json.dumps(dict(cpu_only=True,passed=True,hardware_actions=False)));return
 require('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh exclusive node lease required')
 host=Path(spec['host_release']);sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
 from ecopadg.serving.campaign import node_lease
 async def execute():
  task=asyncio.current_task();loop=asyncio.get_running_loop();sent=False
  def stop():
   nonlocal sent
   if not sent:sent=True;task.cancel()
  for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,stop)
  return await bind_parent(spec).launch(OUT/'deployment.json')
 with node_lease():result=asyncio.run(execute())
 print(json.dumps(dict(restored=result['complete'],measurement_valid=result['measurement_valid'],fresh_mechanism_gate_required=True)))
if __name__=='__main__':main()
