"""Separate fresh native gate and immutable B Dynamo cooperative source binding."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import baseline_control_after_external_v2 as c
p=c.p;HOST=p.REPO/'releases/five-system100-B32B-baseline-cooperative-v1-runtime'
DECL=B/'cooperative-dynamo8-001/declaration.json';OUT=B/'cooperative-dynamo-qualification-001'

def host_contract():
 d=p.read(DECL);p.need(d['host_release']==str(HOST) and d['host_manifest']==p.ref(HOST/'manifest.json'),'declared cooperative source differs')
 m=p.checked(d['host_manifest']);parent=Path(m['parent_release']);old=p.read(parent/'manifest.json')
 p.need(p.sha(parent/'manifest.json')==m['parent_manifest_sha256'] and set(old['files'])==set(m['files']),'cooperative parent/source set changed')
 p.need([f for f in old['files'] if m['files'][f]!=old['files'][f]]==['src/ecopadg/serving/admission.py'],'only common cooperative queue may change')
 for root,files in [(HOST,m['files']),(parent,old['files'])]:
  for f,h in files.items():p.need(p.sha(root/f)==h,'frozen host changed '+f)
 p.need(p.checked(d['cpu_validation'])['passed'] is True,'cooperative CPU evidence failed')
 return d,{str(HOST/f):h for f,h in m['files'].items()}|{str(HOST/'manifest.json'):p.sha(HOST/'manifest.json')}

SCOPE=B/'fixed-slo-scope-update-v1.json'
SCOPE_SHA='fcc178ccd6360a0e749e589ed8f396362b0f7e087344e21c3763d514ea963ae7'

def prerequisite():
 reference=B/'cooperative-dynamo8-001/qualification-prerequisite.json';d=p.read(reference)
 p.need(d['schema']=='B-cooperative-after-original-nonDyn-fixedSLO-v2','wrong stage prerequisite')
 p.need(p.sha(SCOPE)==SCOPE_SHA and d['current_scope']==p.ref(SCOPE),'fixed SLO scope correction changed')
 scope=p.checked(d['current_scope'])
 p.need(scope['historical_scale11_required_for_this_task'] is False and scope['historical_scale11_unmeasured'] is True and scope['cooperative_group']==p.ref(DECL),'historical SLO scale is excluded from this task; exact cooperative eight retained')
 import baseline_reconciliation_v3 as ledger
 p.checked(d['completed_non_dynamo_ledger']);checked=ledger.audit(d['completed_non_dynamo_ledger']['path'])
 p.need(checked['remaining_cells']==[] and checked['already_observed_count']==21 and len(checked['retired_unexecuted_cells'])==3,'all eighteen original non-Dynamo observations must finish before cooperative group')
 for entry in checked['prior_stages']:
  state=p.checked(entry['status'])
  p.need(state['finished_s'] and not state.get('node_lease_held'),'prior node owner not released')
  try:alive=Path('/proc',str(state['pid']),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
  except OSError:alive=False
  p.need(not alive,'prior owner still alive')
 return p.ref(reference)

def bootstrap():
 d,files=host_contract();previous=prerequisite();refs=p.read(B/'baseline-bindings-after-external-001.json');original=p.checked(refs['dynamollm'])
 c.modules();p.need(c.audit_performance('dynamollm',original)['passed'],'original native binding qualification failed')
 b=copy.deepcopy(original);b.update(host_release=str(HOST),configs={},output=str(OUT/'correctness-work'),output_correctness_verified=False,correctness_gate_required_before_performance=True,formal_eligible=False)
 b['files'].update(files);b['files'].update({str(DECL):p.sha(DECL),str(Path(__file__).resolve()):p.sha(__file__),previous['path']:previous['sha256'],refs['dynamollm']['path']:refs['dynamollm']['sha256']})
 b['cooperative_qualification']=dict(created_s=time.time(),declaration=p.ref(DECL),predecessor=previous,original_fresh_binding=refs['dynamollm'],no_policy_profile_change=True)
 for key in ('mechanism_proof','correctness_evidence','fresh_baseline_p4'):b.pop(key,None)
 p.write(OUT/'bootstrap.json',b,exclusive=True);return p.ref(OUT/'bootstrap.json')

def gate():
 bp=bootstrap();b=p.checked(bp);runtime={p.read(i['engine_config'])['runtime_dir'] for i in b['instances']};p.need(len(runtime)==1,'original owner runtime differs')
 argv=['/usr/bin/python3','-u',str(c.GATE),'--binding',bp['path'],'--runtime-dir',runtime.pop(),'--out',str(OUT/'original27'),'--run']
 started=time.time()
 with (OUT/'original27.log').open('xb') as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
 p.write(OUT/'original27-invocation.json',dict(argv=argv,exitcode=result.returncode,started_s=started,finished_s=time.time(),bootstrap=bp,gate_source=p.ref(c.GATE),host_manifest=p.ref(HOST/'manifest.json')),exclusive=True)
 status=p.read(OUT/'original27/status.json');p.need(result.returncode in (0,1) and status['complete'] and status['measurement_valid'] and status['native_cleanup_complete'] and status['clock_restore_complete'] and not status.get('cleanup_errors') and not status.get('sampling_error'),'fresh native gate incomplete/cleanup failed')
 # Preserve the original exact temporal output flag. Dynamo resident admission
 # needs the reconstructed ordinary mechanism; no temporal policy is enabled.
 return dict(original_gate_passed=status['passed'],gate=str(OUT/'original27'),native_gate_executed=True)

def original_binding_lineage(boot):
 reference=boot['cooperative_qualification']['original_fresh_binding']
 p.need(boot['files'].get(reference['path'])==reference['sha256'],'original fresh binding must be explicitly frozen')
 original=p.checked(reference)
 p.need(original['instances']==boot['instances'],'cooperative bootstrap differs from actual fresh original instances')
 return original


def derive():
 _,files=host_contract();c.modules();bp=p.ref(OUT/'bootstrap.json');boot=p.checked(bp);original_binding_lineage(boot)
 invocation=p.read(OUT/'original27-invocation.json');status=p.read(OUT/'original27/status.json');runtime={p.read(i['engine_config'])['runtime_dir'] for i in boot['instances']};p.need(len(runtime)==1,'gate owner runtime differs')
 expected=['/usr/bin/python3','-u',str(c.GATE),'--binding',bp['path'],'--runtime-dir',runtime.pop(),'--out',str(OUT/'original27'),'--run']
 p.need(invocation['argv']==expected and invocation['bootstrap']==bp and invocation['gate_source']==p.ref(c.GATE) and invocation['host_manifest']==p.ref(HOST/'manifest.json') and invocation['exitcode'] in (0,1),'gate did not execute this fresh cooperative bootstrap/source')
 p.need(boot['cooperative_qualification']['created_s']<=invocation['started_s']<=status['started_s']<=status['finished_s']<=invocation['finished_s'],'gate is not fresh after cooperative bootstrap')
 base=c.bind_other('dynamollm',p.ref(c.POLICIES['dynamollm']),bp,OUT/'original27',OUT/'dynamollm')
 p.need(base['configs']==p.read(c.POLICIES['dynamollm'])['configs'],'Dynamo policy references changed')
 base.update(host_release=str(HOST),cooperative_dynamo=dict(declaration=p.ref(DECL),host_manifest=p.ref(HOST/'manifest.json'),original_admission_sha256='2d59808aad020f05c4d22103a494f773b2d7cc1afef29d6146429807af121a31',only_queue_yield_changed=True,fresh_gate=p.ref(OUT/'original27/status.json'),bootstrap=bp))
 base['files'][str(OUT/'original27-invocation.json')]=p.sha(OUT/'original27-invocation.json')
 base['files'].update(files);base['files'].update({str(DECL):p.sha(DECL),str(Path(__file__).resolve()):p.sha(__file__)})
 return base

def qualify():
 b=derive();common=c.fixed.runtime(str(HOST));common.validate_binding(b);p.write(OUT/'dynamollm/binding.json',b,exclusive=True)
 audit(p.ref(OUT/'dynamollm/binding.json'));return p.ref(OUT/'dynamollm/binding.json')

def audit(reference):
 b=p.checked(reference);p.need(b==derive(),'cooperative fresh raw-derived binding changed')
 return dict(passed=True,source=p.ref(HOST/'manifest.json'),fresh_native_qualification=p.ref(OUT/'original27/status.json'),ordinary_dynamo_gate_reconstructed=True,temporal_exact_flag_not_relabelled=True)

if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('action',choices=('check','gate','qualify','audit'));a=q.parse_args()
 if a.action=='check':d,files=host_contract();print(json.dumps(dict(cpu_only=True,source_files=len(files),ready_for_GPU=False)))
 elif a.action=='gate':print(json.dumps(gate()))
 elif a.action=='qualify':print(json.dumps(qualify()))
 else:print(json.dumps(audit(p.ref(OUT/'dynamollm/binding.json'))))
