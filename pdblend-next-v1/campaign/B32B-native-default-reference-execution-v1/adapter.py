"""Small isolated adapter to the already executed 900-second operation.

No original source, result path, global module alias, or existing claim is changed.
Default checks do not import CUDA. The frozen parent source supplies restoration,
native transport proof, actual free-GPU checks, clocks, and eight-board energy.
"""
import copy, hashlib, importlib.util, json, os, sys, time, types
from pathlib import Path

ROOT=Path(__file__).resolve().parent
C=ROOT.parent
PARENT=C/'B32B-temporal-observation-execution-v4'
PARENT_SHA='e55ff19a91c523f60b107ecb93ff48a101ec78f0c84d3d067c9c12f698342ed7'
ATTEMPT='B32B-native-default-reference-attempt-001'
BOOTSTRAP=C/'B32B-temporal-observation-attempt-003/results/restored-bootstrap.binding.json'
BOOTSTRAP_SHA='4b33bf20218a38892952ba92c0902a2310f80464b7bc267d06e68ed3a679fb29'
SCHEDULER_SHA='572cdcb93e1af27439cf31534a7d1777eaf70f4b73432abd14f08ddfcfe9a691'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def alias_imports(mapping, fn):
    prior={k:sys.modules.get(k) for k in mapping};sys.modules.update(mapping)
    try:return fn()
    finally:
        for k,v in prior.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v

def check():
    require(sha(PARENT/'manifest.json')==PARENT_SHA,'frozen restoration parent changed')
    m=read(ROOT/'manifest.json')
    for rel,h in m['files'].items():require(sha(ROOT/rel)==h,'reference file changed '+rel)
    for p,h in m['external_files'].items():require(sha(p)==h,'reference dependency changed '+p)
    candidate=C/'B32B-temporal-observation-candidate-v1'
    require(sha(candidate/'manifest.json')=='78cae0571ff7852a9a9f2fca6f3458d2823b08104f0d3b1a181c1672c297fac8','original observation source binding')
    for rel,h in read(candidate/'manifest.json')['files'].items():require(sha(candidate/rel)==h,'original observation helper changed '+rel)
    for p,h in read(candidate/'source-contract.json')['files'].items():require(sha(p)==h,'original source evidence changed '+p)
    require(sha(ROOT/'image-context/scheduler.py')==SCHEDULER_SHA,'official scheduler changed')
    actual=read(ROOT/'request-spec.json');old=read(C/'B32B-temporal-observation-attempt-003/observation-spec.json')
    require(actual['requests']==old['requests'][:4] and actual['request_ids']==old['request_ids'],
            'four original complete requests and explicit capture targets required')
    return m

def capture_gate(child,terminal):
    require(all(terminal.get(k) is True for k in ('http_child_exited','diagnostic_container_stopped','diagnostic_gpu_workers_gone')),
            'actual child/container/workers must exit before capture')
    require(child.get('complete') is True and child.get('cleanup_complete') is True and child.get('completed_requests')==4,
            'four outputs and actual native cleanup required')

def owner_evidence(spec,out):
    cfg=read(spec['diagnostic_instance']['config'])
    source=Path(cfg['runtime_dir'])/'native-reference.events.jsonl'
    data=source.read_bytes();require(len(data)<=4*1024*1024 and data.endswith(b'\n'),'bounded full owner evidence')
    target=Path(out)/'diagnostic-owner.events.jsonl';target.write_bytes(data)
    events=[json.loads(x) for x in data.splitlines()]
    rows=[x for x in events if x['kind']=='executed_step']
    requests=read(spec['observation_spec'])['requests'];ids=[r['request_uuid'] for r in requests]
    expected=[]
    for r in requests[:2]:expected += [(1,0,r['prompt_length'],[r['request_uuid']])]+[(0,1,1,[r['request_uuid']])]*63
    expected += [(1,0,96,[ids[2]])]+[(0,1,1,[ids[2]])]*4+[(1,0,192,[ids[3]])]+[(0,2,2,ids[2:])]*59+[(0,1,1,[ids[3]])]*4
    require([(r['prefill'],r['decode'],r['tokens'],r['request_ids']) for r in rows]==expected,
            'actual full197 owner steps must match predeclared native reference')
    require(all(r['finished_s']>=r['started_s'] and not r['preempted'] and not r['blocks_to_swap_in']
        and not r['blocks_to_swap_out'] and not r['blocks_to_copy'] for r in rows),'actual cache movement/step failure')
    require(len([r for r in events if r['kind']=='output'])==256,'four64 actual driver outputs')
    return dict(complete=True,file=str(target),sha256=sha(target),native_default=True,steps=197,pair_steps=69,
                original_temporal_gate_changed=False,performance_evidence=False)

def load_operation():
    c=load('native_reference_common',PARENT/'common.py')
    c.ROOT=ROOT;c.package_check=check;c.capture_gate=capture_gate
    c.observation_declaration=lambda:read(ROOT/'request-spec.json')
    archive=alias_imports({'common':c},lambda:load('native_reference_archive',PARENT/'archive.py'))
    technical=alias_imports({'common':c},lambda:load('native_reference_loader',PARENT/'technical.py'))
    # Exactly two source-level substitutions to the untouched parent operation:
    # truthful phase wording and probe the actual new ports, not old attempt001.
    source=(PARENT/'run.py').read_text()
    require(sha(PARENT/'run.py')=='def7f63ef560ade09fe33370b1d4f3e2030d539ffb9a85e0420846e362cb8f05','actual operation source changed')
    old='for port in [34501,*range(34732,34764)]:'
    new="for port in [self.spec['diagnostic_instance']['port'],*range(self.spec['diagnostic_instance']['kv_port'],self.spec['diagnostic_instance']['kv_port']+32)]:"
    require(source.count(old)==1,'single port probe substitution')
    source=source.replace(old,new).replace('six_http_requests','four_native_reference_requests').replace('six-request child/cleanup','four-request child/cleanup')
    base=types.ModuleType('native_reference_operation');base.__file__=str(PARENT/'run.py')
    alias_imports({'common':c,'archive':archive,'technical':technical},lambda:exec(compile(source,str(PARENT/'run.py'),'exec'),base.__dict__))
    base.owner_evidence=owner_evidence

    class Operation(base.Operation):
        async def source_identity(self,session,i,inspection,diagnostic=False):
            actual=await self.engine.http(session,i,'/provenance',timeout=self.remaining(3))
            entry=Path(self.spec['diagnostic_instance']['engine_entry'] if diagnostic else i['engine_entry'])
            expected=dict(instance_id=i['id'],tp=2,model=c.MODEL,dtype='bfloat16',max_model_len=8192,
                cuda_visible_devices=','.join(map(str,i['gpus'])),
                source_files_at_import={str(p):sha(p) for p in entry.parent.glob('*.py')})
            require(all(actual.get(k)==v for k,v in expected.items()),'actual native/original imported provenance changed')
            cfg=read(i.get('config',i.get('engine_config')))
            require(all(cfg.get(k)==v for k,v in dict(tp=2,max_num_batched_tokens=8192,max_num_seqs=32,
                max_model_len=8192,model=c.MODEL).items()),'native/original work budget changed')
            sources=await self.sources(inspection['Name'].lstrip('/'),self.spec['installed_diagnostic_sources' if diagnostic else 'installed_parent_sources'])
            return dict(container=inspection,provenance=actual,installed_sources=sources)
    return c,base,technical,Operation

def eligibility(c):
    auth=read(ROOT/'eligibility.json');c.verify_files(auth['files'])
    require(auth['restored_binding']==str(BOOTSTRAP) and sha(BOOTSTRAP)==BOOTSTRAP_SHA,'fresh attempt003 bootstrap required')
    status=read(C/'B32B-temporal-observation-attempt-003/results/status.json')
    require(status['complete'] and status['all_original_restored'] and status['measurement_valid'] and status['capture_complete'],
            'previous real observation/restoration incomplete')
    require(status['exact_passed'] is False,'original numerical failure must remain explicit')
    for pid in auth['terminal_pids']:require(not c.pid_live(pid),'previous actual diagnostic producer still live')
    actual={str(p):sha(p) for p in C.glob('B32B-temporal-observation-execution-*/execution-once.json')}
    require(len(actual)==3 and actual==auth['previous_claims'],'exact three retained prior attempts required')
    require(not (ROOT/'execution-once.json').exists(),'new native reference claim already exists')
    helper=load('native_reference_producers',C/'B32B-temporal-launch-producer-review-v1/readiness_producer.py')
    gate=c.main_gate(c.SEQUENCE/'main-proof.json');producer=helper.verify_main_producers(c.SEQUENCE/'main-proof.json')
    require(producer['actual_main_producers']==90,'actual completed producer90 required')
    return dict(files={**gate['files'],**producer['files'],**auth['files']},producer=producer)
