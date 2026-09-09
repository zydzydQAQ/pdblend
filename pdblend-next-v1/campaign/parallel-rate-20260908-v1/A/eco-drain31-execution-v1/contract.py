"""Fixed-SLO A Eco scope and actual fresh eight-TP1 qualification contract."""
import copy,hashlib,importlib.util,json,math,os,socket,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;A=HERE.parent;ROOT=A.parent
LOGICAL=A/'eco-drain31-v1/declaration.json';LOGICAL_SHA='d8bb8fb072658a9e28c489647130082bb44f80a48b3394af8d25bce18b9f67b2'
HOST=ROOT.parent.parent/'releases/five-system100-A14B-baseline-eco-drain-v1-runtime';HOST_SHA='6a66a0697d29974ae725c3667b4fde9b88d0828744101b645e8951f391e5e50d'
COMMON=ROOT/'common/execution-until-complete-v1/run.py';COMMON_SHA='77bcbbb68e20419e5bc469a838c71e1abfa789dc167d901501715fda0ff4a8a9'
GATE=ROOT.parent/'AC-baseline-binding-v2/gate_evidence.py';GATE_SHA='b84a113563f1b064be5ef7a8cbf2006b0790f3b60bb1be3851ce66114dd9e9e6'
PAIR=('model','dataset','trace_sha256','content_pairing_sha256','seed','rate_rps','slo_ttft_s','slo_tpot_s','slo_scale','n_requests')

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def checked(r):need(sha(r['path'])==r['sha256'],'reference bytes changed '+r['path']);return read(r['path'])
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def cp_reference(cp,key):return cp[key] if isinstance(cp[key],dict) else dict(path=cp[key],sha256=cp[key+'_sha256'])
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def terminal(reference):
 t=checked(reference);adapter=A/'eco-drain31-v1/restore-code/restore_adapter.py'
 package=read(A/'eco-drain31-v1/manifest.json');need(sha(adapter)==package['files'][str(adapter)],'original terminal restore auditor changed')
 m=load(adapter,'a_ecodrain_terminal_contract');m.REFS=dict(release=t['release'],previous=t['previous_binding'],terminal=reference)
 proof=m.terminal_contract(t);observed=[]
 for stage in t['stages']:
  for cid in stage['expected_cell_ids']:
   path=Path(stage['checkpoint_root'])/(cid+'.json');cp=read(path);receipt=checked(cp_reference(cp,'receipt'));row=cp['row'];sm=receipt['summary']
   need(sm['work_complete'] and sm['failed_requests']==sm['request_timeouts']==0,'PDB failure is not a complete first SLO loss')
   observed.append(dict(row=row,checkpoint=ref(path),slo_attainment=sm['slo_attainment']))
 return t,observed,proof

def boundaries(observed):
 result={d:None for d in ('alpaca','sharegpt','longbench')}
 for point in observed:
  row=point['row'];need(row['model']=='14b' and row['system']=='pdblend' and row['slo_scale']==1.,'wrong final PDB fixed-SLO scope')
  q=point['slo_attainment'];need(math.isfinite(q) and 0<=q<=1,'invalid actual PDB SLO')
  if q<.9:
   previous=result[row['dataset']];result[row['dataset']]=row['rate_rps'] if previous is None else min(previous,row['rate_rps'])
 return result

def select_rows(logical,scope,observed,appends):
 original={r['cell_id']:r for r in logical['cells']};need(len(original)==31,'exact original logical31 required')
 rows=scope['required_cells'];ids=[r['cell_id'] for r in rows];excluded=scope['excluded_cells'];excluded_ids=[v['cell_id'] for v in excluded]
 need(rows and len(ids)==len(set(ids)) and len(excluded_ids)==len(set(excluded_ids)) and not set(ids)&set(excluded_ids) and (set(ids)&set(original))|set(excluded_ids)==set(original),'logical31 must be exactly partitioned, no omitted original')
 limits=boundaries(observed);need(scope['final_first_miss_by_dataset']==limits,'scope first-loss rates must be recomputed from actual complete PDB observations')
 for entry in excluded:
  row=original[entry['cell_id']];limit=limits[row['dataset']]
  need(entry['dataset']==row['dataset'] and entry['rate_rps']==row['rate_rps'] and entry['reason']=='above_first_complete_PDB_SLO_loss','only exact original above-boundary exclusions allowed')
  need(limit is not None and row['rate_rps']>limit and entry['first_complete_slo_miss_rate']==limit,'cannot exclude a retained or unbounded original rate')
 for row in rows:
  need(row['model']=='14b' and row['system']=='ecoserve' and row['seed']==701 and row['slo_scale']==1. and row['arrival_window_s']==100.,'fixed100/seed701/SLO1 original work required')
  limit=limits[row['dataset']];need(limit is None or row['rate_rps']<=limit,'selected rate exceeds first complete loss')
  if row['cell_id'] in original:
   need(row==original[row['cell_id']],'original logical row was changed')
   need(any(all(v['row'][k]==row[k] for k in PAIR) and (row['repeat']==1 or v['row'].get('repeat',v['row'].get('improvement_repeat'))==row['repeat']) for v in observed),'original Eco row has no actual final PDB comparison')
   continue
  matches=[d for d in appends.values() if any(r==row for r in d['cells'])]
  need(len(matches)==1,'new Eco row requires one exact explicit append source row')
  peers=[r for r in matches[0]['cells'] if all(r[k]==row[k] for k in PAIR)]
  need(len(peers)==10 and {(r['system'],r['repeat']) for r in peers}=={(system,k) for system in ('pdblend','mixed','distserve','dynamollm','ecoserve') for k in (1,2)},'new comparison rate requires five systems and two repeats on identical trace')
  need(any(all(v['row'][k]==row[k] for k in PAIR) and v['row'].get('repeat',v['row'].get('improvement_repeat'))==row['repeat'] for v in observed),'new Eco rate/repeat has no exact final PDB observation')
 # Entire selected append rate must preserve both predefined Eco repeats.
 added=[r for r in rows if r['cell_id'] not in original]
 for row in added:need({r['repeat'] for r in added if all(r[k]==row[k] for k in PAIR)}=={1,2},'both new-rate Eco repeats are required')
 return rows,limits

def qualification(reference):
 b=checked(reference);need(b['model']=='14b' and b['system']=='ecoserve' and b['host_release']==str(HOST),'actual A Eco new host required')
 need(b['deadline_s'] is None and b['campaign_lifecycle']=='until_declared_complete_v1','no total campaign deadline')
 need(len(b['instances'])==8 and [(i['tp'],i['gpus']) for i in b['instances']]==[(1,[k]) for k in range(8)],'A fresh qualification is eight TP1 /55 requests')
 need(b['output_correctness_verified'] is True and b['correctness_gate_required_before_performance'] is False,'actual fresh native gate required')
 for path,h in b['files'].items():need(sha(path)==h,'fresh binding dependency changed '+path)
 gate=Path(b['correctness_evidence']);status=read(gate/'status.json');deployment=read(b['deployment_receipt'])
 need(b['files'].get(str(gate/'status.json'))==sha(gate/'status.json') and b['files'].get(b['deployment_receipt'])==sha(b['deployment_receipt']),'gate and restored deployment must be frozen')
 need(status['passed'] is True and status['measurement_valid'] and status['native_cleanup_complete'] and status['clock_restore_complete'],'A original strict legacy gate must actually pass')
 need(deployment['complete'] and deployment['measurement_valid'] and status['started_s']>=deployment['finished_s'],'gate must follow actual retained restoration')
 need(sha(GATE)==GATE_SHA and sha(HOST/'manifest.json')==HOST_SHA,'actual gate/source auditor changed')
 program='''import sys,json,importlib.util\nfrom pathlib import Path\nhost,auditor,bp=sys.argv[1:];sys.path[:0]=[str(Path(host)/'src'),'/root/workspace/pdblend/.runtime-deps']\nfrom ecopadg.serving.measurement import power_evidence\ns=importlib.util.spec_from_file_location('actual_A_native_gate',auditor);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);b=json.loads(Path(bp).read_text());proof,files=m.audit(Path(b['correctness_evidence']),b['instances'],'ecoserve',power_evidence);n=len(json.loads((Path(b['correctness_evidence'])/'checks/checks.json').read_text())['requests']);print(json.dumps(dict(proof=proof,files=files,requests=n)))'''
 process=subprocess.run([sys.executable,'-I','-c',program,str(HOST),str(GATE),reference['path']],capture_output=True,text=True,timeout=120)
 need(process.returncode==0,'actual A native raw qualification rejected '+process.stderr[-1000:]);actual=json.loads(process.stdout)
 need(actual['requests']==55 and actual['proof']==b['mechanism_proof'] and all(b['files'].get(p)==h for p,h in actual['files'].items()),'A native55 proof or frozen raw differs')
 identity=load(HERE/'policy_identity.py','a_ecodrain_original_policy_identity');logical=checked(dict(path=str(LOGICAL),sha256=LOGICAL_SHA));old_identities={};original_refs=[]
 for mapping in logical['replacement_mapping']:
  cp=checked(mapping['checkpoint']);dataset=cp['row']['dataset']
  if dataset in old_identities:continue
  old_ref=cp_reference(cp,'binding');old=checked(old_ref);old_identities[dataset]=identity.identity(old,dataset);original_refs.extend([mapping['checkpoint'],old_ref])
 need(set(old_identities)=={'alpaca','sharegpt','longbench'},'all original dataset policies required')
 actual_identities={dataset:identity.identity(b,dataset) for dataset in old_identities}
 for dataset,expected in old_identities.items():
  need(all(actual_identities[dataset][k]==expected[k] for k in ('profile_sha256','policy_sha256')),'new Eco changed original dataset policy/profile '+dataset)
 return b,dict(binding=reference,gate=ref(gate/'status.json'),request_count=55,raw_gate_recomputed=True,legacy_gate_passed=True,proof=actual['proof'],original_policy_references=original_refs,actual_policy_identities=actual_identities)

def validate_scope(reference,binding_reference):
 scope=checked(reference);need(scope['schema']=='A-Eco-final-PDB-execution-scope-v1' and scope['approved'] is True,'explicit final scope required')
 need(scope['logical_declaration']==dict(path=str(LOGICAL),sha256=LOGICAL_SHA),'original logical31 reference changed')
 logical=checked(scope['logical_declaration']);t,observed,proof=terminal(scope['final_pdb_terminal_audit'])
 appends={v['path']:checked(v) for v in scope.get('append_declarations',[])}
 rows,limits=select_rows(logical,scope,observed,appends)
 need(scope['final_pdb_release']==t['release'],'scope changed final PDB source release')
 base,qualified=qualification(binding_reference)
 need(base['files'].get(scope['final_pdb_terminal_audit']['path'])==scope['final_pdb_terminal_audit']['sha256'] and base['files'].get(scope['final_pdb_release']['path'])==scope['final_pdb_release']['sha256'],'fresh restoration/binding must freeze exact final predecessor')
 need(base['hostname']=='iZwz92bdfqihqp38tekqjyZ','A observations must remain on original A physical node')
 return scope,base,dict(terminal=proof,qualification=qualified,first_complete_PDB_SLO_loss=limits,required_count=len(rows),logical_count=31)
