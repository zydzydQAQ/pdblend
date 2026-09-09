"""Measured retained restart, original27 gate, then fresh mechanism bindings."""
import argparse,asyncio,copy,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
C=Path('/root/workspace/pdblend-next-v1/campaign')
sys.path.insert(0,str(ROOT/'execution-completion-p4-002'))
import runner as fixed
p=fixed.p
ECO=C/'B32B-ecoserve-qualified-main-v1/binding.json'
GATE=ROOT/'baseline-gate-until-complete-v1/validate.py'
RESTORE=ROOT/'baseline-return-completion-001'
QUAL=ROOT/'baseline-qualification-completion-001'
POLICIES={'mixed':C/'B32B-baseline-sequence-v1/attempt-001/bindings/mixed/binding.json',
 'distserve':C/'B32B-baseline-main-first-sequence-v1/attempt-001/bindings/distserve/binding.json',
 'dynamollm':C/'B32B-baseline-main-first-sequence-v1/attempt-001/bindings/dynamollm-resident/binding.json'}

def modules():
 source=p.read(ROOT/'baseline-completion-source-manifest.json')
 for path,digest in source['files'].items():p.need(p.sha(path)==digest,'baseline prepared source changed: '+path)
 target=p.read(ECO);common=fixed.runtime(target['host_release'])
 execution=fixed.load(ROOT/'baseline-return-completion-source/execution.py','b_new_retained_restore')
 qualifier=fixed.load(ROOT/'qualify_restored_baseline_completion.py','b_new_fresh_eco_qualifier')
 return target,common,execution,qualifier

def finished_pdb():
 out=ROOT/'fixed-screen-p4-001';s=p.read(out/'status.json')
 try:alive=Path('/proc',str(s['pid']),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:alive=False
 p.need(s.get('complete') is True and not s['failed'] and not s.get('node_lease_held') and not alive,'complete measured PDB selection, all local cleanup and actual process exit required')
 p.need(set(s['first_complete_breach'])==set(p.DATASETS),'all three first complete SLO boundaries required')
 for cid in s['completed']:
  cp=p.read(out/'results/checkpoints'/(cid+'.json'))
  p.need(all(p.sha(k)==v for k,v in cp['artifacts'].items()) and p.read(out/'engineering-gates'/(cid+'.json'))['passed'],'PDB raw or engineering predecessor invalid')
 return p.ref(out/'status.json')

async def restore():
 predecessor=finished_pdb();target,common,execution,_=modules()
 release,previous,_=fixed.release_contract(ROOT/'completion-release-p4-001/release.json')
 from ecopadg.serving.campaign import node_lease
 p.need('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh independent restoration lease required')
 with node_lease():
  binding=await execution.restore_core(common,target,previous,RESTORE)
 ready=dict(schema='parallel-rate-B-measured-baseline-return-p4',ready=True,created_s=time.time(),
  original_baseline_binding=p.ref(ECO),fresh_restoration_binding=p.ref(RESTORE/'binding.json'),
  fresh_inventory=p.ref(RESTORE/'containers.after.json'),measured_restore_status=p.ref(RESTORE/'status.json'),
  pdb_predecessor=predecessor,fresh_qualification_still_required=True)
 p.write(ROOT/'baseline-return-ready-completion.json',ready,exclusive=True)
 return ready

def bootstrap():
 ready=p.read(ROOT/'baseline-return-ready-completion.json');p.need(ready['ready'] is True,'measured restore absent')
 for k in ('original_baseline_binding','fresh_restoration_binding','fresh_inventory','measured_restore_status','pdb_predecessor'):p.checked(ready[k])
 status=p.checked(ready['measured_restore_status']);p.need(status['complete'] and status['clock_restore_complete'] and not status['errors'] and not status.get('sampling_error'),'restore terminal quality failed')
 b=copy.deepcopy(p.checked(ready['fresh_restoration_binding']))
 b['files'].update(p.read(ROOT/'baseline-completion-source-manifest.json')['files'])
 b['files'][str(ROOT/'baseline-completion-source-manifest.json')]=p.sha(ROOT/'baseline-completion-source-manifest.json')
 b.update(configs={},output=str(QUAL/'future-gate-work'),output_correctness_verified=False,
  correctness_gate_required_before_performance=True,old_correctness_is_historical_only=True,formal_eligible=False,
  identity_file=ready['fresh_inventory']['path'],experiment_scope='measured retained restart; original27 qualification pending',
  restored_priority=dict(binding=ready['fresh_restoration_binding'],status=ready['measured_restore_status'],inventory=ready['fresh_inventory']))
 for key in ('mechanism_proof','correctness_evidence','fresh_ablation_binding','ablation'):b.pop(key,None)
 for r in (ready['fresh_restoration_binding'],ready['measured_restore_status'],ready['fresh_inventory']):b['files'][r['path']]=r['sha256']
 for f in (Path(__file__).resolve(),ROOT/'qualify_restored_baseline_completion.py'):b['files'][str(f)]=p.sha(f)
 p.write(QUAL/'fresh-bootstrap.json',b,exclusive=True)
 return QUAL/'fresh-bootstrap.json'

def gate():
 bp=bootstrap();b=p.read(bp);runtime={p.read(i['engine_config'])['runtime_dir'] for i in b['instances']};p.need(len(runtime)==1,'legacy runtime layout differs')
 out=QUAL/'original27';cmd=[sys.executable,str(GATE),'--binding',str(bp),'--runtime-dir',runtime.pop(),'--out',str(out),'--run']
 # Original validator owns its own fresh lease and bounded native cleanup.
 with (QUAL/'original27.log').open('xb') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 p.write(QUAL/'original27-invocation.json',dict(argv=cmd,exitcode=result.returncode,finished_s=time.time(),gate_source=p.ref(GATE)),exclusive=True)
 p.need(result.returncode in (0,1),'original validator invocation failed')
 status=p.read(out/'status.json');p.need(status['complete'] and status['measurement_valid'] and status['native_cleanup_complete'] and status['clock_restore_complete'] and not status.get('cleanup_errors') and not status.get('sampling_error'),'fresh original27 incomplete or cleanup failed')
 return dict(original_gate_passed=status['passed'],fresh_gate=str(out))

def bind_other(system,original_ref,bootstrap_ref,gate_dir,out):
 original=p.checked(original_ref);boot=p.checked(bootstrap_ref)
 p.need(original['system']==system and original['deadline_s']==1788872770.0400891,'wrong original baseline policy')
 fresh={i['id']:i for i in boot['instances']};instances=[]
 q=fixed.load(ROOT/'qualify_restored_baseline_completion.py','b_fresh_identity_contract')
 for old in original['instances']:
  item=copy.deepcopy(fresh[old['id']]);item['role']=old['role'];item['provenance']={k:item['provenance'][k] for k in old['provenance']};instances.append(item)
 q.same_policy(original['instances'],instances)
 evidence=fixed.load(C/'AC-baseline-binding-v2/gate_evidence.py','b_fresh_mechanism_raw')
 from ecopadg.serving.measurement import power_evidence
 proof,raw_files=evidence.audit(gate_dir,instances,system,power_evidence)
 b=copy.deepcopy(original);b.update(deadline_s=None,campaign_lifecycle='until_declared_complete_v1',instances=instances,output=str(out/'results'),identity_file=boot['identity_file'],
  correctness_evidence=str(gate_dir),mechanism_proof=proof,output_correctness_verified=True,
  correctness_gate_required_before_performance=False,historical_binding_only=False,formal_eligible=False)
 b['files'].update(boot['files']);b['files'].update(raw_files)
 for ref in (original_ref,bootstrap_ref):b['files'][ref['path']]=ref['sha256']
 b['files'][str(C/'AC-baseline-binding-v2/gate_evidence.py')]=p.sha(C/'AC-baseline-binding-v2/gate_evidence.py')
 b['files'][str(Path(__file__).resolve())]=p.sha(__file__)
 b['fresh_baseline_p4']=dict(original=original_ref,bootstrap=bootstrap_ref,gate_dir=str(gate_dir),output_root=str(out),original_policy_bytes_unchanged=True)
 return b

def qualify():
 _,common,_,q=modules();ready=p.read(ROOT/'baseline-return-ready-completion.json');bp=p.ref(QUAL/'fresh-bootstrap.json');gate_dir=QUAL/'original27'
 eco_ref=q.qualify(bp,gate_dir,p.ref(ECO),ready['fresh_inventory'],QUAL/'ecoserve');q.audit_binding(eco_ref)
 refs={'ecoserve':eco_ref}
 for system,original in POLICIES.items():
  out=QUAL/system;p.need(not out.exists(),'new baseline binding required');b=bind_other(system,p.ref(original),bp,gate_dir,out);common.validate_binding(b);p.write(out/'binding.json',b,exclusive=True);refs[system]=p.ref(out/'binding.json');audit_performance(system,b)
 p.write(ROOT/'baseline-bindings-completion.json',refs,exclusive=True)
 return refs

def audit_performance(system,binding):
 if system=='ecoserve':
  q=fixed.load(ROOT/'qualify_restored_baseline_completion.py','b_added_eco_qualification')
  ref=p.ref(Path(binding['qualification']['path']).parent/'binding.json');p.need(p.checked(ref)==binding,'Eco binding changed');return q.audit_binding(ref)
 inputs=binding['fresh_baseline_p4'];expected=bind_other(system,inputs['original'],inputs['bootstrap'],Path(inputs['gate_dir']),Path(inputs['output_root']))
 p.need(expected==binding,'fresh baseline raw reconstruction differs')
 return dict(passed=True,system=system)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=('check','restore','gate','qualify'));a=parser.parse_args()
 if a.action=='check':target,common,e,q=modules();q.sources();print(dict(cpu_only=True,restore_source=p.ref(e.__file__),target=p.ref(ECO),qualification_source=p.ref(q.__file__)))
 elif a.action=='restore':print(json.dumps(asyncio.run(restore()),indent=2))
 elif a.action=='gate':print(json.dumps(gate(),indent=2))
 else:print(json.dumps(qualify(),indent=2))
