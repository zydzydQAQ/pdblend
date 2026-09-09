"""Fresh TP2 native qualification for exact B Eco window-only source replacement."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import baseline_control_after_external_v2 as c
import qualify_eco_drained_baseline_v1 as q
p=c.p;DECL=B/'eco-drain37-v1/declaration.json';OUT=B/'eco-drain-qualification-001';HOST=p.REPO/'releases/five-system100-B32B-baseline-eco-drain-v1-runtime'
DECL_SHA='b71286d6423ca7195c99fe3040bec9ebe59887c33b1708d5e3aa36fd5364332d'
def contract():
 p.need(p.sha(DECL)==DECL_SHA,'Eco whole-group declaration changed');d=p.read(DECL);m=p.checked(d['host_manifest']);old=p.read(Path(m['parent_release'])/'manifest.json')
 p.need(d['host_release']==str(HOST) and m['parent_manifest_sha256']==p.sha(Path(m['parent_release'])/'manifest.json'),'wrong original host lineage')
 p.need(set(m['files'])==set(old['files']) and [f for f in m['files'] if m['files'][f]!=old['files'][f]]==['src/ecopadg/serving/runtime.py'],'only Eco guard runtime may change')
 for root,files in ((HOST,m['files']),(Path(m['parent_release']),old['files'])):
  for f,h in files.items():p.need(p.sha(root/f)==h,'host bytes changed')
 p.need(p.checked(d['cpu_validation'])['passed'] and len(d['cells'])==35 and len(d['declared_cells'])==37 and len(d['excluded_cells'])==2,'whole group/CPU differs')
 prior=p.checked(d['completed_previous_cooperative']);p.need(prior['all_eight_observed'] and prior['remaining_count']==0 and prior['final_owners_exited'] and not prior['node_lease_held'],'previous queue must be terminal and clean')
 return d,{str(HOST/f):h for f,h in m['files'].items()}|{str(HOST/'manifest.json'):p.sha(HOST/'manifest.json')}
def bootstrap():
 d,files=contract();refs=p.read(B/'baseline-bindings-after-external-001.json');reference=refs['ecoserve'];original=p.checked(reference);c.audit_performance('ecoserve',original)
 b=copy.deepcopy(original);b.update(host_release=str(HOST),configs={},output=str(OUT/'native-work'),output_correctness_verified=False,correctness_gate_required_before_performance=True,formal_eligible=False)
 b['files'].update(files)
 for f in (DECL,Path(__file__).resolve(),Path(q.__file__).resolve()):b['files'][str(f)]=p.sha(f)
 b['files'][reference['path']]=reference['sha256'];b['new_eco_qualification']=dict(created_s=time.time(),whole_group=p.ref(DECL),original_fresh_binding=reference,actual_host_manifest=d['host_manifest'])
 for k in ('qualification','mechanism_proof','correctness_evidence','qualifier_source','priority_qualification_inputs'):b.pop(k,None)
 p.write(OUT/'bootstrap.json',b,exclusive=True);return p.ref(OUT/'bootstrap.json')
def gate():
 bp=bootstrap();b=p.checked(bp);runtime={p.read(i['engine_config'])['runtime_dir'] for i in b['instances']};p.need(len(runtime)==1,'runtime domain changed');argv=['/usr/bin/python3','-u',str(c.GATE),'--binding',bp['path'],'--runtime-dir',runtime.pop(),'--out',str(OUT/'original27'),'--run'];started=time.time()
 with (OUT/'original27.log').open('xb') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
 p.write(OUT/'original27-invocation.json',dict(argv=argv,started_s=started,finished_s=time.time(),exitcode=r.returncode,bootstrap=bp,host_manifest=p.ref(HOST/'manifest.json'),gate_source=p.ref(c.GATE)),exclusive=True)
 s=p.read(OUT/'original27/status.json');p.need(r.returncode in (0,1) and s['complete'] and s['measurement_valid'] and s['native_cleanup_complete'] and s['clock_restore_complete'] and not s.get('cleanup_errors') and not s.get('sampling_error'),'fresh TP2 gate failed/incomplete');return p.ref(OUT/'original27/status.json')
def invocation():
 contract();bp=p.ref(OUT/'bootstrap.json');b=p.checked(bp);v=p.read(OUT/'original27-invocation.json');s=p.read(OUT/'original27/status.json');runtime={p.read(i['engine_config'])['runtime_dir'] for i in b['instances']};p.need(len(runtime)==1,'runtime domain');expected=['/usr/bin/python3','-u',str(c.GATE),'--binding',bp['path'],'--runtime-dir',runtime.pop(),'--out',str(OUT/'original27'),'--run']
 p.need(v['argv']==expected and v['bootstrap']==bp and v['host_manifest']==p.ref(HOST/'manifest.json') and v['gate_source']==p.ref(c.GATE) and v['exitcode'] in (0,1),'actual fresh invocation/source differs')
 p.need(b['new_eco_qualification']['created_s']<=v['started_s']<=s['started_s']<=s['finished_s']<=v['finished_s'],'fresh new source gate chronology failed')
 return b
def qualify():
 b=invocation();bp=p.ref(OUT/'bootstrap.json');inventory=b['restored_priority']['inventory'];reference=q.qualify(bp,OUT/'original27',p.ref(c.ECO),inventory,OUT/'ecoserve');q.audit_binding(reference);common=c.fixed.runtime(str(HOST));common.validate_binding(p.checked(reference));return reference
def audit(reference):
 invocation();p.need(Path(reference['path'])==OUT/'ecoserve/binding.json','wrong Eco source binding');return q.audit_binding(reference)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=('check','gate','qualify','audit'));x=a.parse_args()
 if x.action=='check':contract();q.sources();print(json.dumps(dict(cpu_only=True,declared=37,execute=35,ready_for_GPU=False)))
 elif x.action=='gate':print(json.dumps(gate()))
 elif x.action=='qualify':print(json.dumps(qualify()))
 else:print(json.dumps(audit(p.ref(OUT/'ecoserve/binding.json'))))
