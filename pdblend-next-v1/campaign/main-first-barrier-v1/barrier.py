"""Read-only main completion evidence; no HTTP, subprocess, GPU or serving import.

An endpoint creates a main proof under a fresh node lease. The coordinator must
deeply verify all raw files before publishing. Consumers must pin the publication
SHA explicitly; compact verification trusts that exact coordinator publication.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import time

HERE = Path(__file__).resolve().parent
PROTOCOL = 'per-dataset-slo-five-system-fixed-window-v1'
DEADLINE = 1788872770.0400891
MODELS = ('7b', '14b', '32b')
BASELINES = ('mixed', 'distserve', 'dynamollm', 'ecoserve')
SYSTEMS = ('pdblend',) + BASELINES
DATASETS = ('alpaca', 'sharegpt', 'longbench')
HOSTS = {'7b':'iZwz9gfq11hx1sbob59yrgZ', '14b':'iZwz92bdfqihqp38tekqjyZ', '32b':'iZwz9i5bte3xkpmcoes3t2Z'}
SOURCE_SHA = {'7b':'b5beb606905cbb944434b4e8ca06d0c08d8a258eb35921e9250d120980f16176',
 '14b':'a0a2193e9504b77bcef1113b0fc7de13a8c2f3f7838f24e27f5a91e24106f327',
 '32b':'4b9494c6b0a38cb9d44dbc490530d88e9a0eec76b44854f6e40db0328bbfe5ed'}
LEASE = Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')
DEFAULT_RELEASE = HERE / 'release/global-main-release.json'

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def digest_json(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
def valid_sha(value): return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def facts():
    p=HERE/'point_evidence.frozen.py'
    s=importlib.util.spec_from_file_location('main_barrier_frozen_facts',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def native(before,proof,instance):
    s=importlib.util.spec_from_file_location('main_barrier_native',HERE/'native_barrier.frozen.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.barrier(before,proof,instance)

def package_check():
    p=HERE/'manifest.json'
    require(p.is_file(),'barrier implementation has not been frozen')
    for name,h in read(p)['files'].items(): require(sha(HERE/name)==h,'barrier source changed: '+name)

class Reader:
    """Per-model path relocation; absolute originals are labels, not identities."""
    def __init__(self, path_map=None): self.path_map=path_map or {};self.files={}
    def local(self,path): return Path(self.path_map.get(str(path),str(path)))
    def digest(self,path,expected=None):
        value=sha(self.local(path));key=str(path)
        if expected is not None:require(value==expected,'raw SHA differs: '+key)
        if key in self.files:require(self.files[key]==value,'raw changed during verification: '+key)
        self.files[key]=value;return value
    def read(self,path,expected=None):
        data=self.local(path).read_bytes();value=hashlib.sha256(data).hexdigest();key=str(path)
        if expected is not None:require(value==expected,'JSON SHA differs: '+key)
        if key in self.files:require(self.files[key]==value,'raw changed during read: '+key)
        self.files[key]=value;return json.loads(data)
    def stable(self):
        for p,h in self.files.items():require(sha(self.local(p))==h,'evidence changed before publication: '+p)

def source_rows(source,model):
    m=facts();rows=m.validate_manifest(source)
    require(source['model']==model,'wrong source model')
    main=[r for r in rows if r['phase']=='main']
    require(len(main)==150,'main declaration is not 150 cells')
    for s in SYSTEMS:
        require(sum(r['system']==s for r in main)==30,'missing main system domain')
    return main

def verify_invocations(group,reader,source_sha):
    rows=[]
    for p,h in group['terminal_invocations'].items():
        v=reader.read(p,h)
        require(v.get('phase')=='main' and v.get('system')==group['system']
            and set(v.get('selected_datasets',DATASETS))==set(group['datasets'])
            and v.get('manifest_sha256')==source_sha and v.get('protocol_id')==PROTOCOL
            and v.get('finished_s'),'main invocation source/group/terminal evidence differs')
        rows.append(v)
    require(rows,'group has no invocation')
    latest=max(rows,key=lambda v:v['started_s'])
    require(latest.get('complete') is True and not latest.get('error')
        and latest.get('binding_sha256')==group['binding_sha256'],'latest bound main invocation not clean terminal')

def process_scan():
    """Processes are sampled before/after hashing under the same node lease.

    Idle engine/model processes and the waiting supervisor are allowed. Actual
    execution drivers and their measurement children are not allowed.
    """
    live=[];seen=0
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:data=path.read_bytes()
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:raise ValueError('cannot establish complete local process view')
        seen+=1;parts=[x.decode(errors='replace') for x in data.split(b'\0') if x]
        is_execution=any('/campaign/five-system-execution-' in x and x.endswith(('/run.py','/child.py')) for x in parts)
        if is_execution and ('--run' in parts or any(x.endswith('/child.py') for x in parts)):
            live.append(dict(pid=int(path.parent.name),argv=parts))
    return dict(hostname=socket.gethostname(),observed_s=time.time(),processes_read=seen,
        no_live_serving_child=not live,live_serving_children=live,
        scope='five-system run.py --run and measurement child.py; idle model engines and supervisors excluded')

@contextmanager
def fresh_lease():
    require(not os.environ.get('PDBLEND_NODE_LOCK_FD'),'inherited node lease prohibited')
    with LEASE.open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield dict(path=str(LEASE),exclusive=True,inherited=False)

def verify_record(record, reader):
    """Recompute contracts from original CP/config/receipt/trace and full raw SHA."""
    row=record['row'];cp=reader.read(record['checkpoint'],record['checkpoint_sha256'])
    require(cp['row']==row and cp.get('measurement_valid') is True,'checkpoint is not the declared measured main cell')
    require(row['phase']=='main' and row['slo_scale']==1.,'scale cell cannot substitute for main')
    out=Path(record['output']);cid=row['cell_id'];op=out/'operations'/cid;cell=out/'cells'/cid
    require(cp['receipt']==str(op/'receipt.json'),'foreign checkpoint receipt')
    artifacts=cp.get('artifacts');require(isinstance(artifacts,dict) and artifacts,'missing raw artifact map')
    required={str(cell/name) for name in ('summary.json','runtime_config.json','bench.csv','power.csv','power_source.json',
        'power_metadata.jsonl','arrival_window.json','cleanup.json','control.jsonl')}
    required|={str(op/name) for name in ('receipt.json','job.json','identity.before.json','identity.after.json','controls.before.json',
        'actual-epoch.json','dispatch.jsonl','child.log','power/power.csv','power/power_source.json','power/power_metadata.jsonl','power/clocks.csv')}
    require(required<=set(artifacts),'raw checkpoint is missing required clock/identity/config/epoch/cleanup evidence')
    require(all(Path(p).is_absolute() and '..' not in Path(p).parts and (Path(p).is_relative_to(op) or Path(p).is_relative_to(cell))
        for p in artifacts),'artifact escaped its original cell')
    for p,h in artifacts.items():reader.digest(p,h)
    receipt=reader.read(cp['receipt'],cp['receipt_sha256']);summary=reader.read(str(cell/'summary.json'))
    require(receipt.get('summary')==summary,'receipt and original summary disagree')
    binding=reader.read(record['binding'],record['binding_sha256'])
    require(binding['model']==row['model'] and binding['system']==row['system'] and binding['hostname']==HOSTS[row['model']]
        and binding['protocol_id']==PROTOCOL and binding['deadline_s']==DEADLINE,'record binding differs')
    require(binding['configs'][row['dataset']]==record['config'] and binding['files'][record['config']]==record['config_sha256'],
        'actual dataset config was not in group binding')
    require(set(receipt['restoration'])=={i['id'] for i in binding['instances']},'restoration instance set differs')
    for instance in binding['instances']:
        rr=receipt['restoration'][instance['id']]
        require(rr.get('complete') is True and not rr.get('errors'),'native restoration failed')
        native(rr['before'],rr['proof'],instance)
        resumed=rr['resumed'];after=resumed['after'];control=resumed['control']
        require(after.get('id')==instance['id'] and after.get('generation')==control['generation']==resumed['before']['generation']+1
            and after.get('acknowledged_generation')==after['generation'] and after.get('accepting') is True
            and all(after.get(k)==0 for k in ('active','running','waiting')) and not after.get('kv_allocations')
            and not after.get('error') and not after.get('runtime_error'),'actual resumed owner is not idle/ACK/accepting')
        require(all(after.get(k)==control[k] for k in ('role','mode','admit_prefill','admit_decode')),'resume control not applied')
        if instance['native_kind']=='v3':
            require(after.get('scheduler_budget_pending') is None,'budget pending after restoration')
        if instance.get('restore_budget_tokens') is not None:
            require(after.get('scheduler_budget_effective',{}).get('max_num_batched_tokens')==instance['restore_budget_tokens'],
                'restore token budget differs')
        if instance.get('scheduler_cache_observed'):
            cache=[x.get('controls',{}).get('runtime') for x in after.get('scheduler_io',[])]
            require(len(cache)==instance.get('scheduler_cache_count',1) and all(x and x.get('generation')==after['generation'] and not x.get('error') for x in cache),
                'actual scheduler cache did not ACK restore')
    before_id=reader.read(str(op/'identity.before.json'));after_id=reader.read(str(op/'identity.after.json'))
    require(len(before_id)==len(after_id)==len(binding['instances']),'full instance identities absent')
    def identity_index(rows):return {x['provenance']['instance_id']:x for x in rows}
    bi,ai=identity_index(before_id),identity_index(after_id)
    require(set(bi)==set(ai)=={i['id'] for i in binding['instances']},'identity instance set changed')
    for i in binding['instances']:
        a,b=bi[i['id']],ai[i['id']]
        for key in ('Id','Image','Name','Args'):
            require(a['container'][key]==b['container'][key],'container identity changed')
        for key in ('StartedAt','Pid'):
            require(a['container']['State'][key]==b['container']['State'][key],'container process changed')
        require(a['provenance']==b['provenance'],'import/model provenance changed')
        require(a['container']['Id']==i['container']['id'] and a['container']['Image']==i['container']['image']
            and a['container']['State']['StartedAt']==i['container']['StartedAt'],'actual identity differs from bound deployment')
    config=reader.read(record['config'],record['config_sha256'])
    actual=reader.read(str(cell/'runtime_config.json'))
    expected=copy.deepcopy(config);expected.update(journal=str(cell/'control.jsonl'),slo_scale=1.,slo_protocol='per-dataset-slo-v1',
        slo_attainment_target=.9,slo_ttft_s=row['slo_ttft_s'],slo_tpot_s=row['slo_tpot_s'],comparison_system=row['system'])
    require(actual==expected,'actual Controller configuration differs from bound config/SLO')
    strategy=config['strategy'];canonical='pdblend' if strategy.startswith('pdblend') else 'dynamollm' if strategy=='dynamollm-resident' else strategy
    require(canonical==row['system'] and summary.get('implementation_variant')==strategy,'actual mechanism label changed')
    trace=reader.read(row['trace'],row['trace_sha256'])
    require(trace['protocol_id']==PROTOCOL and trace['seed']==701 and trace['arrival_window_s']==100
        and len(trace['requests'])==trace['n_requests']==row['n_requests'],'trace work/window differs')
    facts().verify_summary(row,summary,receipt)
    require(summary.get('gpu_count')==8 and len(summary.get('gpu_util_per_gpu',[]))==8,'not an all-eight measurement')
    require(summary.get('incomplete_drain') is False and summary.get('drain_complete') is True,'primary drain incomplete')
    require(receipt['finished_s']<=DEADLINE and cp['completed_s']<=DEADLINE,'main work checkpoint exceeds unchanged deadline')
    result=dict(measurement_valid=True,work_complete=summary['work_complete'],n_expected=row['n_requests'],
        good_requests=summary['good_requests'],completed_work_requests=summary['completed_work_requests'],energy_j=summary['energy_j'],
        implementation_variant=strategy,receipt_sha256=cp['receipt_sha256'],actual_config_sha256=artifacts[str(cell/'runtime_config.json')],
        finished_s=receipt['finished_s'],child_pid=receipt['child_pid'])
    require(all(record.get(k)==v for k,v in result.items()),'compact record differs from actual raw')
    return result

def proof_contract(proof):
    model=proof.get('model');require(model in MODELS,'unknown model proof')
    require(proof.get('schema')==1 and proof.get('kind')=='host-main-proof' and proof.get('protocol_id')==PROTOCOL
        and proof.get('deadline_s')==DEADLINE and proof.get('hostname')==HOSTS[model],'host proof contract differs')
    require(proof.get('source_sha256')==SOURCE_SHA[model] and proof.get('baseline_systems')==list(BASELINES),'source or four-baseline set differs')
    require(proof.get('source_manifest') and valid_sha(proof.get('source_sha256')),'model source identity missing')
    rows=source_rows(proof['source_declaration'],model)
    require(digest_json(proof['source_declaration'])==proof['source_declaration_canonical_sha256'],'embedded declaration changed')
    records=proof.get('records',[]);require(len(records)==150 and len({r['row']['cell_id'] for r in records})==150
        and len({r['checkpoint'] for r in records})==150,'not 150 distinct actual main records')
    evidence=proof.get('evidence_files');require(isinstance(evidence,dict) and evidence,'host raw SHA map absent')
    require(all(valid_sha(h) for h in evidence.values()),'invalid host raw SHA reference')
    require(evidence.get(proof['source_manifest'])==proof['source_sha256'],'source bytes missing from raw evidence map')
    expected={r['cell_id']:r for r in rows}
    require({r['row']['cell_id'] for r in records}==set(expected),'main domain missing (including Eco)')
    groups=proof.get('groups',[]);covered=set()
    for g in groups:
        require(g['system'] in SYSTEMS and g['datasets'] and set(g['datasets'])<=set(DATASETS),'invalid physical group')
        for ds in g['datasets']:
            key=(g['system'],ds);require(key not in covered,'duplicate physical group domain');covered.add(key)
        require(valid_sha(g['binding_sha256']) and g['terminal_invocations'],'group binding/terminal evidence missing')
        require(evidence.get(g['binding'])==g['binding_sha256'] and all(valid_sha(h) and evidence.get(p)==h
            for p,h in g['terminal_invocations'].items()),'group raw binding/invocation SHA membership missing')
    require(covered=={(s,d) for s in SYSTEMS for d in DATASETS},'missing physical group domain')
    if model=='14b':
        dist=[set(g['datasets']) for g in groups if g['system']=='distserve']
        require(len(dist)==2 and {'alpaca','sharegpt'} in dist and {'longbench'} in dist,'A DistServe must bind its two actual physical layouts')
    for r in records:
        require(r['row']==expected[r['row']['cell_id']] and r.get('measurement_valid') is True,'wrong/missing measured main row')
        for k in ('checkpoint_sha256','receipt_sha256','config_sha256','actual_config_sha256','binding_sha256'):require(valid_sha(r.get(k)),'record SHA missing: '+k)
        matching=[g for g in groups if g['system']==r['row']['system'] and r['row']['dataset'] in g['datasets']]
        require(len(matching)==1 and r['binding']==matching[0]['binding'] and r['binding_sha256']==matching[0]['binding_sha256'],
            'record not tied to its actual physical group')
        require(type(r.get('n_expected')) is int and r['n_expected']==r['row']['n_requests'],'offered work differs')
        require(type(r.get('completed_work_requests')) is int and type(r.get('good_requests')) is int
            and 0<=r['good_requests']<=r['completed_work_requests']<=r['n_expected'],'bad work/good denominator')
        require(type(r.get('energy_j')) in (int,float) and math.isfinite(r['energy_j']) and r['energy_j']>=0,'failed work energy missing')
        require(type(r.get('work_complete')) is bool,'missing work-completion fact')
        for key,hkey in (('checkpoint','checkpoint_sha256'),('binding','binding_sha256'),('config','config_sha256')):
            require(evidence.get(r[key])==r[hkey],'compact record missing original raw SHA membership')
        require(evidence.get(r['row']['trace'])==r['row']['trace_sha256'],'compact trace not in raw evidence map')
    for side in ('before','after'):
        p=proof['process_evidence'][side]
        require(p['hostname']==HOSTS[model] and p['no_live_serving_child'] is True and p['live_serving_children']==[],
            'host did not observe no live serving child')
    require(proof['process_evidence']['before']['observed_s']<=proof['process_evidence']['after']['observed_s']<=proof['created_s'],
        'host process evidence order invalid')
    require(proof['node_lease']==dict(path=str(LEASE),exclusive=True,inherited=False),'fresh node lease not proved')
    return model

def prove_main(spec_path,out):
    package_check();reader=Reader();spec_path=str(Path(spec_path).resolve());spec=reader.read(spec_path);model=spec['model']
    require(model in MODELS and socket.gethostname()==spec['hostname']==HOSTS[model],'prove-main must run on actual model host')
    require(spec['schema']==1 and spec['protocol_id']==PROTOCOL and spec['deadline_s']==DEADLINE,'spec protocol/deadline differs')
    require(spec['source_sha256']==SOURCE_SHA[model],'unfrozen workload source')
    require(not Path(out).exists(),'proof output already exists')
    source=reader.read(spec['source_manifest'],spec['source_sha256']);rows=source_rows(source,model)
    records=[];groups=[];coverage=set()
    with fresh_lease() as lease:
        before=process_scan();require(before['no_live_serving_child'],'main serving child still live')
        for g in spec['groups']:
            system=g['system'];datasets=g['datasets'];binding=reader.read(g['binding'],g['binding_sha256'])
            require(binding['hostname']==HOSTS[model] and binding['model']==model and binding['system']==system
                and binding['protocol_id']==PROTOCOL and binding['deadline_s']==DEADLINE,'group binding identity differs')
            require(binding['files'].get(spec['source_manifest'])==spec['source_sha256'],'source not bound by actual config')
            selected=[r for r in rows if r['system']==system and r['dataset'] in datasets]
            require(len(selected)==10*len(datasets),'invalid selected main group')
            for ds in datasets:
                key=(system,ds);require(key not in coverage,'overlapping physical groups');coverage.add(key)
            output=Path(binding['output']);inv=[]
            for p in sorted((output/'invocations').glob('*.json')):
                value=read(p)
                if value.get('phase')=='main' and value.get('system')==system and set(value.get('selected_datasets',DATASETS))==set(datasets):
                    reader.digest(str(p));inv.append((p,value))
            require(inv,'no real main invocation for this physical group')
            latest=max(inv,key=lambda x:x[1]['started_s'])
            require(latest[1].get('complete') is True and latest[1].get('finished_s') and not latest[1].get('error'),
                'latest physical-group main invocation is not clean terminal')
            require(all(v.get('finished_s') for _,v in inv),'unclosed historical main invocation needs explicit evidence review')
            gr=dict(system=system,datasets=datasets,binding=g['binding'],binding_sha256=g['binding_sha256'],
                terminal_invocations={str(p):reader.digest(str(p)) for p,_ in inv})
            verify_invocations(gr,reader,spec['source_sha256'])
            groups.append(gr)
            for row in selected:
                cp_path=output/'checkpoints'/(row['cell_id']+'.json');cp=reader.read(str(cp_path))
                receipt=reader.read(cp['receipt'],cp['receipt_sha256']);summary=receipt['summary'];config=binding['configs'][row['dataset']]
                rec=dict(row=row,checkpoint=str(cp_path),checkpoint_sha256=reader.digest(str(cp_path)),output=str(output),
                    binding=g['binding'],binding_sha256=g['binding_sha256'],
                    config=config,config_sha256=binding['files'][config],measurement_valid=True,work_complete=summary['work_complete'],
                    n_expected=row['n_requests'],good_requests=summary['good_requests'],completed_work_requests=summary['completed_work_requests'],
                    energy_j=summary['energy_j'],implementation_variant=summary['implementation_variant'],receipt_sha256=cp['receipt_sha256'],
                    actual_config_sha256=cp['artifacts'][str(output/'cells'/row['cell_id']/'runtime_config.json')],
                    finished_s=receipt['finished_s'],child_pid=receipt['child_pid'])
                verify_record(rec,reader);records.append(rec)
        require(coverage=={(s,d) for s in SYSTEMS for d in DATASETS},'all four baseline120 and PDB30 main cells required')
        reader.stable();after=process_scan();require(after['no_live_serving_child'],'new serving child appeared before proof publication')
        proof=dict(schema=1,kind='host-main-proof',protocol_id=PROTOCOL,deadline_s=DEADLINE,model=model,hostname=HOSTS[model],
            source_manifest=spec['source_manifest'],source_sha256=spec['source_sha256'],source_declaration=source,
            source_declaration_canonical_sha256=digest_json(source),baseline_systems=list(BASELINES),groups=groups,
            records=records,process_evidence=dict(before=before,after=after),node_lease=lease,created_s=time.time(),
            spec_path=spec_path,spec_sha256=reader.files[spec_path],evidence_files=reader.files,
            scope='All declared main experiments measured and drained; low SLO/incomplete work retained, not a performance win.')
        proof_contract(proof);write_new(out,proof)
    return dict(main_proof=str(Path(out).resolve()),sha256=sha(out),model=model,baseline_main=120,pdblend_main=30)

def verify_model_proof(proof,*,deep=False,path_map=None):
    model=proof_contract(proof)
    if deep:
        reader=Reader(path_map)
        source=reader.read(proof['source_manifest'],proof['source_sha256'])
        require(source==proof['source_declaration'],'embedded source differs from frozen bytes')
        for p,h in proof['evidence_files'].items():reader.digest(p,h)
        for rec in proof['records']:verify_record(rec,reader)
        for group in proof['groups']:
            b=reader.read(group['binding'],group['binding_sha256'])
            require(b['model']==model and b['hostname']==proof['hostname'] and b['system']==group['system'],'raw group identity changed')
            verify_invocations(group,reader,proof['source_sha256'])
        reader.stable()
    return model

def assemble_release(proof_refs,out,*,path_maps=None):
    package_check();require(len(proof_refs)==3,'three real host proofs required')
    require(not Path(out).exists(),'release is immutable; choose a new publication path')
    models={};refs={}
    for path,expected in proof_refs:
        require(valid_sha(expected) and sha(path)==expected,'unfixed or changed host proof')
        proof=read(path);model=verify_model_proof(proof,deep=True,path_map=(path_maps or {}).get(proof['model']))
        require(model not in models,'duplicate model proof');models[model]=proof;refs[model]=dict(path=str(Path(path).resolve()),sha256=expected,
            canonical_sha256=digest_json(proof))
    require(set(models)==set(MODELS),'all three model hosts required')
    require(time.time()<DEADLINE,'global deadline elapsed; no release can restart it')
    for model,ref in refs.items():
        require(sha(ref['path'])==ref['sha256'],'proof changed during global assembly')
        final=Reader((path_maps or {}).get(model))
        for p,h in models[model]['evidence_files'].items():final.digest(p,h)
    release=dict(schema=1,kind='global-main-release',protocol_id=PROTOCOL,deadline_s=DEADLINE,baseline_systems=list(BASELINES),
        created_s=time.time(),coordinator_hostname=socket.gethostname(),models=models,proof_refs=refs,
        coordinator_deep_verification=True,main_records=450,baseline_main_records=360,pdblend_main_records=90,
        scale_authorization='All model main domains complete; original local deadline/lease/scale work/ref gates still apply.',
        compact_trust_scope='Explicitly pinned coordinator publication; consumers do not claim remote raw reread.')
    write_new(out,release);return dict(release=str(Path(out).resolve()),sha256=sha(out),main_records=450)

def verify_release(path,expected_sha256,*,expected_protocol_id=PROTOCOL,expected_deadline_s=DEADLINE,deep=False,path_maps=None):
    require(valid_sha(expected_sha256),'an explicit published release SHA is mandatory')
    require(time.time()<expected_deadline_s,'release deadline elapsed; local phase cannot be started')
    require(sha(path)==expected_sha256,'release SHA differs or release is missing')
    release=read(path)
    require(release.get('schema')==1 and release.get('kind')=='global-main-release' and
        release.get('protocol_id')==expected_protocol_id==PROTOCOL and release.get('deadline_s')==expected_deadline_s==DEADLINE,
        'release protocol/deadline differs')
    require(release.get('baseline_systems')==list(BASELINES) and set(release.get('models',{}))==set(MODELS)
        and release.get('coordinator_deep_verification') is True,'global full main verification absent')
    require(release.get('main_records')==450 and release.get('baseline_main_records')==360 and release.get('pdblend_main_records')==90,'global main domain differs')
    for model,proof in release['models'].items():
        require(model==verify_model_proof(proof,deep=deep,path_map=(path_maps or {}).get(model)),'model proof swapped')
        ref=release['proof_refs'][model];require(valid_sha(ref['sha256']),'host proof SHA absent')
        require(ref.get('canonical_sha256')==digest_json(proof),'embedded proof differs from coordinator-attested object')
        # Exact original JSON bytes are bound at assembly. Compact embedded objects
        # are additionally tied by canonical hash so no remote original path is needed.
        require(proof['created_s']<=release['created_s']<DEADLINE,'proof/publication time exceeds deadline')
    require(sha(path)==expected_sha256,'release changed while reading')
    return dict(released=True,protocol_id=PROTOCOL,deadline_s=DEADLINE,main_records=450,
        verified_scope='all_model_raw_files_and_contract' if deep else 'root_verified_release_sha_and_contract',
        root_release_sha256=expected_sha256,local_deadline_and_lease_still_required=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prove-main');q.add_argument('--spec',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('assemble-release');q.add_argument('--proof',nargs=2,action='append',metavar=('PATH','SHA256'),required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--path-maps',type=Path)
    q=sub.add_parser('verify-release');q.add_argument('--release',type=Path,default=DEFAULT_RELEASE);q.add_argument('--sha256',required=True);q.add_argument('--deep',action='store_true');q.add_argument('--path-maps',type=Path)
    a=p.parse_args()
    if a.command=='prove-main':result=prove_main(a.spec,a.out)
    elif a.command=='assemble-release':result=assemble_release(a.proof,a.out,path_maps=read(a.path_maps) if a.path_maps else None)
    else:
        package_check();result=verify_release(a.release,a.sha256,deep=a.deep,path_maps=read(a.path_maps) if a.path_maps else None)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
