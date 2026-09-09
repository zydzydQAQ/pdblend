"""Pure evidence gates and frozen imports for one B diagnostic attempt."""
import copy,hashlib,importlib.util,json,math,os,socket,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT.parents[1]
CANDIDATE=R/'campaign/B32B-temporal-observation-candidate-v1'
CANDIDATE_SHA='78cae0571ff7852a9a9f2fca6f3458d2823b08104f0d3b1a181c1672c297fac8'
DEPLOYMENT=R/'campaign/B32B-five-system100-baseline-deployment-v1/deployment.json'
DEPLOYMENT_SHA='c3caa8ea6efcf46c08ec8d703891148e8c4e09c12249034d382bf94bf728f382'
SEQUENCE=R/'campaign/B32B-baseline-main-first-sequence-v1/attempt-001'
IMAGE='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'
HOST=R/'releases/five-system100-B32B-v1-runtime'
NODE='iZwz9i5bte3xkpmcoes3t2Z'
DEADLINE=1788872770.0400891
PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
MODEL='/models/Qwen2.5-32B-Instruct'
MODULES={
 'checks':(R/'campaign/B32B-legacy-baseline-correctness-v1/checks.py','e0965f922ae42245b275d9c17342c31689290c61bcac5dfaf33a2a352fd29931'),
 'executor':(R/'campaign/five-system-execution-v2/run.py','ddc634e0b826d1873ed0bb7e3bd9088ba1412476725d8ec1e371414ccce54ad2'),
 'deployment':(R/'campaign/AC-baseline-deployment-v1/deploy.py','f1b6589d45135cfd90acdc896421ec1b66c2b6e6119cce4b83507517cc52e3ae')}

def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def verify_files(files):
    for p,h in files.items():require(Path(p).is_absolute() and sha(p)==h,'changed frozen input: '+str(p))
def module(name):
    p,h=MODULES[name]
    if h is not None:require(sha(p)==h,'frozen helper changed: '+name)
    spec=importlib.util.spec_from_file_location('b_temporal_'+name,p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

OBSERVATION_SCOPE=R/'campaign/B32B-temporal-solo-pair-observation-v1'
OBSERVATION_SCOPE_SHA='a9a9d75dc79b4a8a0f95cd91f95124f8b13e9b4465f1952c48bd8768c26ae01d'
REQUEST_SPEC=OBSERVATION_SCOPE/'spec.json'
REQUEST_SPEC_SHA='77ce7b8d8011ae1779f9e0d4e986517b2d61dc3508e5debbf5bc4517f37417e1'

def observation_declaration():
    require(sha(OBSERVATION_SCOPE/'manifest.json')==OBSERVATION_SCOPE_SHA and sha(REQUEST_SPEC)==REQUEST_SPEC_SHA,'new observed UUID declaration changed')
    old=read(CANDIDATE/'specs/original-vs-continuous.json');new=read(REQUEST_SPEC)
    by={r['label']:r for r in old['requests']}
    require(set(old)==set(new) and {k for k in old if old[k]!=new[k]}=={'request_ids'},'only observer UUID selection may change')
    require(new['request_ids']==[by['golden-second']['request_uuid'],by['temporal-second']['request_uuid']],'must observe solo192 and temporal-second by actual labels')
    return new

def package_check():
    manifest=read(ROOT/'manifest.json');verify_files({str(ROOT/p):h for p,h in manifest['files'].items()})
    verify_files(manifest['external_files'])
    require(sha(CANDIDATE/'manifest.json')==CANDIDATE_SHA,'observation candidate version changed')
    c=read(CANDIDATE/'manifest.json');verify_files({str(CANDIDATE/p):h for p,h in c['files'].items()})
    verify_files(read(CANDIDATE/'source-contract.json')['files'])
    require(sha(DEPLOYMENT)==DEPLOYMENT_SHA,'original deployment changed')
    observation_declaration()
    return manifest

def pid_live(pid,proc=Path('/proc')):
    require(type(pid) is int and pid>0,'missing real process identifier')
    try:
        text=(proc/str(pid)/'stat').read_text();state=text[text.rfind(')')+2:].split()[0]
        return bool((proc/str(pid)/'cmdline').read_bytes()) and state!='Z'
    except FileNotFoundError:return False

def process_gate(status,live=pid_live):
    require(status.get('complete') is True and status.get('phase') in ('main_complete','main_incomplete_correctness')
        and finite(status.get('finished_s')),'main supervisor is not terminal')
    require(not live(status['pid']),'main supervisor still live')
    require(status.get('steps'),'main stage journal missing')
    for row in status['steps']:
        require(row.get('complete') is True and row.get('exitcode')==0 and type(row.get('exitcode')) is int,
                'main stage did not terminate cleanly')
        require(not live(row['pid']),'main stage process still live')

def main_gate(proof_path,*,live=pid_live):
    """Three eligible main groups only; deliberately does not release global scale."""
    proof_path=Path(proof_path).resolve();require(proof_path==SEQUENCE/'main-proof.json','unexpected main proof namespace')
    status=read(SEQUENCE/'status.json');process_gate(status,live)
    require(status['main_proof']==str(proof_path) and status['main_proof_sha256']==sha(proof_path),'unbound main proof')
    proof=read(proof_path);require(proof.get('model')=='32b' and proof.get('hostname')==NODE
        and proof.get('protocol_id')==PROTOCOL and proof.get('deadline_s')==DEADLINE,'foreign main proof')
    source=Path(proof['source_manifest']);require(sha(source)==proof['source_sha256'],'main workload changed')
    workload=read(source);require(workload.get('model')=='32b' and workload.get('protocol_id')==PROTOCOL,'wrong source model/protocol')
    results={};files={str(proof_path):sha(proof_path),str(SEQUENCE/'status.json'):sha(SEQUENCE/'status.json'),str(source):sha(source)}
    for system in ('mixed','dynamollm','distserve'):
        p=proof['baseline_systems'].get(system);require(p and p.get('complete') is True and p.get('completed')==30,'required baseline main30 incomplete: '+system)
        binding=Path(p['binding']);b=read(binding);files[str(binding)]=sha(binding)
        require(b.get('model')=='32b' and b.get('hostname')==NODE and b.get('system')==system and b.get('deadline_s')==DEADLINE
            and b.get('protocol_id')==PROTOCOL and b['files'].get(str(source))==sha(source),'main binding scope mismatch')
        rows=[r for r in workload['cells'] if r['system']==system and r['phase']=='main'];require(len(rows)==30,'main source count differs')
        records={r['cell_id']:r for r in p['records']};require(set(records)=={r['cell_id'] for r in rows},'main proof ID set differs')
        for row in rows:
            rec=records[row['cell_id']];cp_path=Path(b['output'])/'checkpoints'/(row['cell_id']+'.json')
            require(rec['checkpoint']==str(cp_path) and sha(cp_path)==rec['checkpoint_sha256'],'foreign/changed main checkpoint')
            cp=read(cp_path);require(cp['row']==row and cp.get('measurement_valid') is True,'invalid main checkpoint')
            receipt_path=Path(cp['receipt']);require(sha(receipt_path)==cp['receipt_sha256']==rec['receipt_sha256'],'receipt changed')
            receipt=read(receipt_path);require(receipt.get('measurement_valid') is True and receipt.get('child_stopped') is True
                and receipt.get('clock_restore_complete') is True and finite(receipt.get('finished_s')),'main native/clock/child not terminal')
            require(receipt.get('restoration') and all(x.get('complete') is True for x in receipt['restoration'].values()),'main native cleanup not complete')
            require(str(receipt_path) in cp.get('artifacts',{}),'main raw not bound');verify_files(cp['artifacts'])
            child=receipt.get('child_pid');require(not live(child),'main HTTP child still live')
            files[str(cp_path)]=sha(cp_path);files.update(cp['artifacts'])
        invs=[(p,read(p)) for p in (Path(b['output'])/'invocations').glob('*.json')]
        invs=[(p,v) for p,v in invs if v.get('phase')=='main' and v.get('system')==system]
        require(invs,'main invocation absent')
        for path,v in invs:
            require(v.get('complete') is True and finite(v.get('finished_s')) and not v.get('error'),'main invocation not terminal')
            require(not live(v['pid']),'main runner still live');files[str(path)]=sha(path)
        results[system]=dict(binding=str(binding),binding_sha256=sha(binding),main_complete=30)
    return dict(groups=results,files=files,global_scale_released=False,ecoserve_gate_passed=False)

def binding_scope(b):
    require(b.get('model')=='32b' and b.get('hostname')==NODE and b.get('deadline_s')==DEADLINE,'wrong B binding')
    require([i['gpus'] for i in b['instances']]==[[0,1],[2,3],[4,5],[6,7]],'four original TP2 pairs required')
    for i in b['instances']:
        require(i['tp']==2 and i['container']['image']==IMAGE and i['native_kind']=='legacy_sync_put'
            and i.get('scheduler_cache_observed') is False,'original baseline engine scope differs')
        cfg=read(i['engine_config']);require(b['files'].get(i['engine_config'])==sha(i['engine_config']),'engine config not frozen')
        require(all(cfg.get(k)==v for k,v in dict(tp=2,max_num_batched_tokens=8192,max_num_seqs=32,max_model_len=8192,model=MODEL).items()),'original static work contract differs')
        require(i['provenance'].get('model')==MODEL and i['provenance'].get('max_model_len')==8192,'model/work identity differs')

def actual_source_mapping(diagnostic=False):
    files=read(CANDIDATE/'source-contract.json')['files'];result={}
    for p,h in files.items():
        for marker in ('/actual-sources/','/source-evidence/'):
            if marker in p and p.endswith('.py'):result['/usr/local/lib/python3.10/dist-packages/vllm/'+p.split(marker,1)[1]]=h
    if diagnostic:
        result['/usr/local/lib/python3.10/dist-packages/vllm/worker/model_runner.py']=sha(CANDIDATE/'model_runner.py')
        result['/usr/local/lib/python3.10/dist-packages/vllm/pdblend_diagnostics.py']=sha(CANDIDATE/'pdblend_diagnostics.py')
    return result

def capture_gate(child,terminal):
    require(terminal.get('http_child_exited') is True and terminal.get('diagnostic_container_stopped') is True
        and terminal.get('diagnostic_gpu_workers_gone') is True,'capture cannot freeze before real process exit')
    require(child.get('complete') is True and child.get('cleanup_complete') is True,'HTTP/native cleanup incomplete')
    require(child.get('completed_requests')==6,'all six HTTP outputs required')
    # exact_passed is deliberately not a gate: mismatching full outputs are the evidence.

def frozen_capture(spec_path,source,dest,outputs,child,terminal):
    capture_gate(child,terminal);source=Path(source);dest=Path(dest)
    require(not dest.exists(),'capture snapshot already exists');paths=sorted(source.glob('*'))
    require(paths and all(p.is_file() and not p.is_symlink() for p in paths),'unexpected capture objects')
    require(sum(p.stat().st_size for p in paths)<=2*1024*1024,'capture exceeds bounded output size')
    before={str(p):sha(p) for p in paths};dest.mkdir()
    for p in paths:(dest/p.name).write_bytes(p.read_bytes())
    require(before=={str(p):sha(p) for p in paths},'capture changed after process exit')
    # Explicitly resolve the original verifier's import to the frozen original helper.
    name='pdblend_diagnostics';previous=sys.modules.get(name)
    helper=importlib.util.spec_from_file_location(name,CANDIDATE/'pdblend_diagnostics.py')
    module=importlib.util.module_from_spec(helper);helper.loader.exec_module(module);sys.modules[name]=module
    try:
        loader=importlib.util.spec_from_file_location('b_frozen_capture_verifier',CANDIDATE/'verify_capture.py')
        v=importlib.util.module_from_spec(loader);loader.loader.exec_module(v)
        report=v.verify(Path(spec_path),dest,Path(outputs))
    finally:
        if previous is None:sys.modules.pop(name,None)
        else:sys.modules[name]=previous
    write(dest/'freeze.json',dict(frozen_s=time.time(),terminal=terminal,source_sha256=before,
        files={p.name:sha(p) for p in dest.iterdir()},result=report,performance_evidence=False))
    return report
