"""Read-only qualification and original Eco binding. Never controls an engine."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
C=Path('/root/workspace/pdblend-next-v1/campaign')
PLAN=C/'B32B-ecoserve-main-scale-preparation-v1'
HOST=C.parent/'releases/five-system100-B32B-v1-runtime'
PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
CORRECTNESS='legacy-temporal-default-trajectory-exact-v2'
DEADLINE=None
IMAGE='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'
EXPECTED_HOST='iZwz9i5bte3xkpmcoes3t2Z'
PERFORMANCE_OUTPUT=C/'B32B-baseline-main-first-sequence-v1/attempt-001/bindings/ecoserve/results'
QUALIFIER_DIR=C/'B32B-temporal-qualification-v2'
NEEDED=('ordinary','pd','cancel','temporal_native_trajectory_exact','native_cleanup','identity','all8_measurement','clock')

def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(path):return json.loads(Path(path).read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)
def pinned(path,digest):
    path=Path(path).resolve();require(valid_sha(digest) and sha(path)==digest,'explicit file SHA differs: '+str(path));return read(path)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def verify_files(files):
    require(isinstance(files,dict),'SHA mapping required')
    for path,digest in files.items():require(valid_sha(digest) and sha(path)==digest,'input changed: '+path)
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def package_check():
    manifest=read(HERE/'manifest.json')
    verify_files({str(HERE/p):h for p,h in manifest['files'].items()})
    verify_files(manifest.get('dependencies',{}))
def plan_module():
    manifest=read(PLAN/'manifest.json')
    require(sha(PLAN/'manifest.json')=='d3688efdc224be6d55b5dc3dd263cef113b26d666947e2ba181bed85017f8861','frozen original-policy plan changed')
    verify_files({str(PLAN/p):h for p,h in manifest['files'].items()})
    return load('qualified_eco_original_plan',PLAN/'prepare.py')

def bootstrap_contract(binding):
    require(binding.get('schema')==1 and binding.get('model')=='32b' and binding.get('hostname')==EXPECTED_HOST
        and binding.get('protocol_id')==PROTOCOL and binding.get('deadline_s') is None and binding.get('campaign_lifecycle')=='until_declared_complete_v1'
        and binding.get('host_release')==str(HOST),'original B host/model/protocol required')
    require(binding.get('configs')=={} and binding.get('correctness_gate_required_before_performance') is True
        and binding.get('output_correctness_verified') is False,'unqualified configs-empty bootstrap required')
    require(binding.get('window_s')==100 and binding.get('seeds')==[701],'original100/701 required')
    ii=binding.get('instances',[])
    require([(i['id'],i['tp'],i['gpus']) for i in ii]==[(f'base100b{k}',2,[2*k,2*k+1]) for k in range(4)],'actual original4 TP2 mapping required')
    for i in ii:
        require(i.get('native_kind')=='legacy_sync_put' and i.get('scheduler_cache_observed') is False
            and i['container']['image']==IMAGE and i['container']['StartedAt'],'B legacy identity/capability differs')
        p=i['provenance']
        require(p.get('instance_id')==i['id'] and p.get('tp')==2 and p.get('model')=='/models/Qwen2.5-32B-Instruct'
            and p.get('dtype')=='bfloat16' and p.get('max_model_len')==8192
            and p.get('cuda_visible_devices')==','.join(map(str,i['gpus'])),'actual B model/source/TP differs')
        require(p.get('source_files_at_import') and all(binding['files'].get(k)==v for k,v in p['source_files_at_import'].items()),
            'imported engine sources must be individually frozen')
    return True

def inventory_contract(binding,rows):
    require(isinstance(rows,list) and len(rows)==4,'full four-container inspect inventory required')
    index={r['Id']:r for r in rows};require(len(index)==4,'duplicate actual container')
    for i in binding['instances']:
        require(i['container']['id'] in index,'bound container absent from inventory')
        r=index[i['container']['id']];s=r['State']
        require(r['Name'].lstrip('/')==i['container']['name'] and r['Image']==i['container']['image']
            and s['StartedAt']==i['container']['StartedAt'] and s.get('Running') is True
            and type(s.get('Pid')) is int and s['Pid']>0 and not s.get('Paused') and not s.get('Restarting'),
            'actual host PID/StartedAt/image is not the bound restored process')
    return index

def full_inventory(raw,binding):
    if raw and all(isinstance(x,dict) and 'container' in x for x in raw):
        by={x['provenance']['instance_id']:x for x in raw}
        require(len(by)==4,'full original4 restore observations required')
        for i in binding['instances']:
            require(i['id'] in by and all(by[i['id']]['provenance'].get(k)==v for k,v in i['provenance'].items()),
                'restored imported provenance differs')
        rows=[x['container'] for x in raw]
    else:rows=raw
    inventory_contract(binding,rows);return copy.deepcopy(rows)

def derive_bootstrap(a):
    """Add explicit full inspect provenance to a new JSON; never edit raw restore."""
    require(not a.out.exists(),'new bootstrap output required')
    b=pinned(a.bootstrap,a.bootstrap_sha256);bootstrap_contract(b);verify_files(b['files'])
    raw=pinned(a.identity,a.identity_sha256);inventory=full_inventory(raw,b)
    restore_path=Path(b['restoration_evidence']).resolve()
    status=pinned(restore_path,a.restore_status_sha256)
    require(a.identity.resolve().parent==restore_path.parent
        and a.identity.name in ('restored-ready.json','restored-identity.after.json'),
        'identity must be an original native restore observation')
    other=restore_path.parent/('restored-identity.after.json' if a.identity.name=='restored-ready.json' else 'restored-ready.json')
    other_inventory=full_inventory(read(other),b)
    first_by=inventory_contract(b,inventory);last_by=inventory_contract(b,other_inventory)
    for cid in first_by:
        require(first_by[cid]['State']['Pid']==last_by[cid]['State']['Pid']
            and first_by[cid]['State']['StartedAt']==last_by[cid]['State']['StartedAt'],
            'resident process changed across final restore observations')
    require(status.get('restored_binding')==str(a.bootstrap.resolve()) and status.get('restored_binding_sha256')==a.bootstrap_sha256,
        'restore terminal receipt does not identify this raw bootstrap')
    require(status.get('all_original_restored') is True and status.get('measurement_valid') is True
        and status.get('clock_restore_complete') is True and not status.get('sampling_error')
        and status.get('finished_s') and status.get('restored_native',{}).get('complete') is True,
        'complete measured native restoration required')
    require(type(status.get('full_operation_energy_j')) in (int,float)
        and math.isfinite(status['full_operation_energy_j']) and status['full_operation_energy_j']>=0,
        'actual full operation energy must be retained')
    new=copy.deepcopy(b);files=dict(b['files'])
    for path,digest in [(a.bootstrap,a.bootstrap_sha256),(a.identity,a.identity_sha256),(other,sha(other)),(restore_path,a.restore_status_sha256)]:
        path=str(path.resolve());require(files.get(path,digest)==digest,'conflicting bootstrap source');files[path]=digest
    verify_files(files);a.out.mkdir(parents=True)
    identity=a.out/'identity.json';write(identity,inventory);files[str(identity.resolve())]=sha(identity)
    new.update(identity_file=str(identity.resolve()),files=files,
        derived_from_bootstrap=dict(path=str(a.bootstrap.resolve()),sha256=a.bootstrap_sha256),
        identity_observation=dict(path=str(a.identity.resolve()),sha256=a.identity_sha256))
    bp=a.out/'binding.json';verify_files(files);write(bp,new)
    write(a.out/'derivation.json',dict(complete=True,performance_eligible=False,binding=dict(path=str(bp.resolve()),sha256=sha(bp)),
        no_controls_or_serving_requests=True,original_restore_unchanged=True))
    return dict(binding=str(bp.resolve()),sha256=sha(bp),performance_eligible=False)

def legacy_exact_evidence(checks):
    temporal=checks.get('temporal',{})
    require(temporal.get('complete') is True,'complete original temporal outputs required')
    references,outputs=temporal.get('reference_token_ids'),temporal.get('token_ids')
    require(isinstance(references,list) and isinstance(outputs,list) and len(references)==len(outputs)==2,
        'original two solos/two pair outputs required')
    require(all(isinstance(tokens,list) and len(tokens)==64 and all(type(t) is int and t>=0 for t in tokens)
        for tokens in references+outputs),'original four full64 outputs required')
    differences=[next((dict(position_one_based=k,reference=x,observed=y) for k,(x,y) in enumerate(zip(a,b),1) if x!=y),None)
        for a,b in zip(references,outputs)]
    require(any(differences) and temporal.get('first_differences')==differences,'actual legacy mismatch must be retained exactly')
    flags=checks.get('checks',{})
    present='temporal_exact' in flags
    require(not present or flags['temporal_exact'] is False,'existing legacy exact flag must remain false')
    return dict(header_present=present,header_value=flags.get('temporal_exact'),
        recomputed_exact=False,first_differences=differences,
        missing_header_scope='Original Checks raises on mismatch before setting its exact flag; absence is preserved, not rewritten.')

def qualification_contract(q,binding,gate,inventory):
    require(q.get('schema')==2 and q.get('kind')=='temporal-default-trajectory-qualification'
        and q.get('protocol_id')==CORRECTNESS and q.get('passed') is True
        and q.get('eligible_systems',{}).get('ecoserve') is True,'registered native trajectory qualification required')
    require(all(q.get('verified',{}).get(k) is True for k in NEEDED),'all eight measured mechanism/identity/energy conditions required')
    require(q.get('legacy_single_vs_pair_exact') is False,'legacy single-versus-pair failure must remain false')
    require(q.get('legacy_failure_preserved') is True,'legacy failure evidence must be retained')
    require(isinstance(q.get('files'),dict) and q['files'],'qualification must bind all original inputs')
    old=read(gate/'status.json');checks=read(gate/'checks/checks.json')
    require(old.get('complete') is True and old.get('measurement_valid') is True
        and old.get('native_cleanup_complete') is True and old.get('clock_restore_complete') is True
        and not old.get('sampling_error') and not old.get('cleanup_errors')
        and old.get('finished_s') is not None,'original fresh gate must finish measured/native clean')
    require(old.get('passed') is False and old.get('mechanism_gate',{}).get('temporal') is False,
        'do not rewrite original failed exact gate')
    legacy=legacy_exact_evidence(checks)
    require(q.get('original_mechanism_gate')==old['mechanism_gate'],'qualification changed original mechanism flags')
    bound=inventory_contract(binding,inventory)
    original=[]
    for filename in ('identity.before.json','identity.after.json'):
        rows=read(gate/filename);require(len(rows)==4,'fresh gate needs all four instances')
        full_inventory(rows,binding)
        by={r['container']['Id']:r for r in rows};original.append(by)
        for cid,ref in bound.items():
            actual=by[cid]['container']
            require(actual['State']['Pid']==ref['State']['Pid'] and actual['State']['StartedAt']==ref['State']['StartedAt'],
                'fresh gate ran on another host process')
            for key in ('Id','Image','Name','Path','Args','Config','HostConfig'):
                require(actual.get(key)==ref.get(key),'actual container settings changed: '+key)
            norm=lambda x:sorted(json.dumps(v,sort_keys=True) for v in x.get('Mounts',[]))
            require(norm(actual)==norm(ref),'actual mounts changed')
    for cid in bound:require(original[0][cid]['provenance']==original[1][cid]['provenance'],'full imported provenance changed during gate')
    rawfiles={str(p.resolve()):sha(p) for p in gate.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(rawfiles and all(q['files'].get(p)==h for p,h in rawfiles.items()),'qualification omitted or changed fresh gate raw')
    return old,checks,rawfiles,legacy

def load_qualifier(path,digest):
    path=path.resolve();require(path.parent==QUALIFIER_DIR and valid_sha(digest) and sha(path)==digest,
        'explicit final qualifier source SHA/path required')
    manifest=path.parent/'manifest.json';require(manifest.is_file(),'qualifier must have a final frozen manifest')
    m=read(manifest);files={str((path.parent/p).resolve()) if not Path(p).is_absolute() else p:h for p,h in m['files'].items()}
    require(files.get(str(path))==digest,'qualifier source not in its frozen manifest');verify_files(files)
    for key in ('dependencies','external_files'):
        verify_files(m.get(key,{}));files.update(m.get(key,{}))
    files[str(manifest)]=sha(manifest)
    module=load('explicit_b_frozen_qualifier',path)
    require(callable(getattr(module,'audit_fresh_gate',None)),'final audit_fresh_gate API missing')
    return module,files

def bind(a):
    require(not a.out.exists() and not PERFORMANCE_OUTPUT.exists(),'new Eco binding/output namespace required; no retry or overwrite')
    plan=plan_module();rows,configs,policy_files=plan.inputs()
    b=pinned(a.bootstrap,a.bootstrap_sha256);bootstrap_contract(b);verify_files(b['files'])
    require(b.get('identity_file') and b['files'].get(b['identity_file'])==sha(b['identity_file']),'derive explicit full-identity bootstrap first')
    inventory=full_inventory(read(b['identity_file']),b)
    oracle=pinned(a.oracle,a.oracle_sha256)
    module,module_files=load_qualifier(a.qualifier,a.qualifier_sha256)
    # This is the only eligibility source. No cached qualification is trusted.
    q=module.audit_fresh_gate(a.gate.resolve(),b,oracle)
    require(q.get('inputs',{}).get('gate_dir')==str(a.gate.resolve())
        and q['inputs'].get('binding_canonical_sha256')==canonical(b)
        and q['inputs'].get('oracle_canonical_sha256')==canonical(oracle),
        'qualification revalidated another bootstrap/oracle/gate')
    original,checks,gate_files,legacy=qualification_contract(q,b,a.gate.resolve(),inventory)
    files=dict(b['files'])
    for mapping in (policy_files,module_files,q['files'],gate_files):
        for path,digest in mapping.items():
            require(files.get(path,digest)==digest,'conflicting frozen identity/source: '+path);files[path]=digest
    for path,digest in [(a.bootstrap,a.bootstrap_sha256),(a.oracle,a.oracle_sha256),(Path(__file__),sha(__file__))]:
        files[str(path.resolve())]=digest
    roles={i['id']:i['role'] for i in next(iter(configs.values()))['instances']}
    ii=copy.deepcopy(b['instances'])
    for cfg in configs.values():
        require({i['id']:i['role'] for i in cfg['instances']}==roles,'original dataset-specific roles differ')
        for i in cfg['instances']:
            actual=next(x for x in ii if x['id']==i['id'])
            require(all(i[k]==actual[k] for k in ('tp','gpus','url','port','kv_port')),'actual routes changed original policy')
    for i in ii:i['role']=roles[i['id']]
    verify_files(files);a.out.mkdir(parents=True)
    config_paths={}
    for ds,cfg in configs.items():
        p=a.out/'configs'/f'{ds}.json';write(p,cfg);files[str(p.resolve())]=sha(p);config_paths[ds]=str(p.resolve())
    identity=a.out/'identity.json';write(identity,inventory);files[str(identity.resolve())]=sha(identity)
    qp=a.out/'qualification.json';write(qp,q);files[str(qp.resolve())]=sha(qp)
    result=copy.deepcopy(b)
    result.update(system='ecoserve',implementation_variant='ecoserve',output=str(PERFORMANCE_OUTPUT),configs=config_paths,
        instances=ii,files=files,identity_file=str(identity.resolve()),correctness_evidence=str(a.gate.resolve()),
        correctness_gate_required_before_performance=False,output_correctness_verified=True,
        correctness_protocol_id=CORRECTNESS,correctness_scope='native default trajectory exact v2; original single-versus-pair exact remains false',
        qualification=dict(path=str(qp.resolve()),sha256=sha(qp)),
        qualifier_source=dict(path=str(a.qualifier.resolve()),sha256=a.qualifier_sha256),
        qualified_bootstrap=dict(path=str(a.bootstrap.resolve()),sha256=a.bootstrap_sha256),
        oracle=dict(path=str(a.oracle.resolve()),sha256=a.oracle_sha256),
        legacy_output_correctness_verified=False,legacy_single_vs_pair_exact=False,
        legacy_exact_evidence=legacy,
        mechanism_proof=dict(required=['ordinary','temporal_native_trajectory_exact'],verified=copy.deepcopy(q['verified']),
            legacy_verified=copy.deepcopy(original['mechanism_gate']),overall_runtime_gate_passed=False,
            qualified_under_explicit_protocol=CORRECTNESS,original_failure_preserved=True),
        formal_eligible=False,execution_manifest=dict(path=str(plan.SOURCE),sha256=plan.SOURCE_SHA))
    verify_files(files);bp=a.out/'binding.json';write(bp,result)
    receipt=dict(complete=True,binding=dict(path=str(bp.resolve()),sha256=sha(bp)),system='ecoserve',main_cells=30,
        future_scale_cells=18,qualification=result['qualification'],correctness_protocol_id=CORRECTNESS,
        original_gate_passed=False,original_temporal_exact=False,legacy_exact_evidence=legacy,measurement_or_controls_started=False,
        actual_live_recheck_still_required=True,finished_s=time.time())
    write(a.out/'binding-receipt.json',receipt);return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    s.add_parser('check')
    d=s.add_parser('derive-bootstrap');f=s.add_parser('bind')
    for parser in (d,f):
        parser.add_argument('--bootstrap',type=Path,required=True);parser.add_argument('--bootstrap-sha256',required=True)
        parser.add_argument('--out',type=Path,required=True)
    d.add_argument('--identity',type=Path,required=True);d.add_argument('--identity-sha256',required=True)
    d.add_argument('--restore-status-sha256',required=True)
    for key in ('oracle','qualifier'):
        f.add_argument('--'+key,type=Path,required=True);f.add_argument('--'+key+'-sha256',required=True)
    f.add_argument('--gate',type=Path,required=True)
    a=p.parse_args();package_check()
    if a.command=='check':
        rows,_,_=plan_module().inputs()
        print(json.dumps(dict(cpu_check=True,main_cells=30,future_scale_cells=18,actual_binding_written=False,rows=len(rows))))
        return
    a.out=a.out.resolve()
    print(json.dumps(derive_bootstrap(a) if a.command=='derive-bootstrap' else bind(a),indent=2))

if __name__=='__main__':main()
