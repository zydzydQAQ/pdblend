"""CPU-only B scale binding derivation. A real pinned B150 release is required to write."""
import argparse,copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT.parent
SCALE=C/'scale-only-continuation-B32B-v1';SCALE_SHA='c3dbb24d736e25024c1430742d798fb53a814ec55ff3a4591b2bea406c16333b'
RELEASE=C/'B32B-qualified-main-release-v1';RELEASE_SHA='c801e9a84204affbd814682cd4407b5fb725529270dfa72661b5e95e598c92ed'
FRESH=C/'B32B-temporal-matched-shape-bootstrap-v1/binding.json';FRESH_SHA='8afbe0d60ed5e86bff55ab27f29b2508bfc87d2be98e8cdc15c7680b1b73d014'
GATE=C/'B32B-temporal-matched-shape-fresh-gate-v1'
ECO=C/'B32B-ecoserve-qualified-main-v1/binding.json';ECO_SHA='87dbf43fcf8076dc1b4bc588e14fad717a2d2b3eeb83cae1514f3fe2d078d119'
DATASETS=['alpaca','sharegpt','longbench']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def package_check():
    m=read(ROOT/'manifest.json')
    for name,h in m['files'].items():require(sha(ROOT/name)==h,'scale binding helper source changed')
    return {str(ROOT/name):h for name,h in m['files'].items()}|{str(ROOT/'manifest.json'):sha(ROOT/'manifest.json')}

def modules():
    require(sha(SCALE/'manifest.json')==SCALE_SHA and sha(RELEASE/'manifest.json')==RELEASE_SHA,'frozen B scale/release packages changed')
    c=load('B_scale_fresh_contract',SCALE/'contract.py')
    previous=sys.modules.get('contract');sys.modules['contract']=c
    try:r=load('B_scale_fresh_reference_map',SCALE/'reference_map.py')
    finally:
        if previous is None:sys.modules.pop('contract',None)
        else:sys.modules['contract']=previous
    r.package_check();return c,r

def inputs():
    require(sha(FRESH)==FRESH_SHA and sha(ECO)==ECO_SHA,'fixed actual fresh27/Eco binding changed')
    d=read(RELEASE/'draft-spec.json');policies={g['system']:g['policy_reference'] for g in d['groups']}
    policies['ecoserve']=ref(ECO)
    require(set(policies)=={'pdblend','mixed','distserve','dynamollm','ecoserve'},'five original systems')
    for v in policies.values():require(sha(v['path'])==v['sha256'],'actual main policy binding changed')
    return d,policies,read(FRESH)

def candidate(system,main,fresh,c):
    """In-memory derivation only. Keep main configs/output/model stat identities exact."""
    require(system in ('mixed','dynamollm','distserve','ecoserve') and main['system']==system,'baseline system only')
    if system=='ecoserve':return copy.deepcopy(main),'same_process'
    new=copy.deepcopy(main);by={i['id']:i for i in fresh['instances']}
    for instance in new['instances']:
        actual=by[instance['id']]
        for key in ('id','tp','gpus','url','port','kv_port','engine_config','native_kind','scheduler_cache_observed'):
            require(instance[key]==actual[key],'physical/source capability changed '+key)
        oldcore=c.core_instance(instance)
        instance['container']=copy.deepcopy(actual['container'])
        # Preserve the original binding's declared provenance schema; the fresh
        # full response (including engine_version) remains separately SHA-bound.
        instance['provenance']={k:copy.deepcopy(actual['provenance'][k]) for k in instance['provenance']}
        require(c.core_instance(instance)==oldcore,'only original fresh process identity may change')
    require(new.get('large_inputs')==fresh.get('large_inputs'),'same original model stat/SHA bindings required')
    # The old full overall gate failure stays explicit; reconstruct only this system's needed mechanisms.
    reader=c.load('B_fresh_scale_gate',C/'AC-baseline-binding-v2/gate_evidence.py')
    power=c.load('B_fresh_scale_power',SCALE/'power_evidence.frozen.py')
    strategy='dynamollm-resident' if system=='dynamollm' else system
    actual,raw=reader.audit(GATE,new['instances'],strategy,power.power_evidence,hetero=False)
    files=dict(new['files'])
    for mapping in (fresh['files'],raw):
        for p,h in mapping.items():require(files.get(p,h)==h,'conflicting historical/fresh input '+p);files[p]=h
    files[str(FRESH)]=FRESH_SHA
    new.update(files=files,correctness_evidence=str(GATE),output_correctness_verified=True,
        correctness_gate_required_before_performance=False,
        mechanism_proof=dict(required=actual['required'],verified=actual['verified'],
            overall_runtime_gate_passed=False,original_failure_preserved=True),
        scale_identity_source=ref(FRESH),fresh_gate_reuses_current_resident_processes=True,
        fresh_full_provenance={i['id']:copy.deepcopy(i['provenance']) for i in fresh['instances']})
    return new,'restarted'

def check():
    c,r=modules();d,policies,fresh=inputs();out=[]
    for system in ('mixed','dynamollm','distserve','ecoserve'):
        main=c.reference(policies[system]);new,mode=candidate(system,main,fresh,c)
        for ds in DATASETS:c.policy_equal(read(main['configs'][ds]),read(new['configs'][ds]))
        out.append(dict(system=system,identity_mode=mode,output_unchanged=new['output']==main['output'],
            policy_reference=policies[system],scale_cells=18,configs_unchanged=new['configs']==main['configs'],
            mechanism_proof=new['mechanism_proof']))
    return dict(cpu_only=True,bindings_written=False,release_created=False,scale_started=False,
                groups=out,pdblend_scale_reuse_only=18,additional_container_restart_required=False,
                executor_fresh_live_identity_and_lease_still_required=True)

def prepare(release_path,release_sha,out):
    require(not os.environ.get('PDBLEND_NODE_LOCK_FD'),'CPU binder cannot inherit ownership')
    implementation=package_check()
    c,r=modules();d,policies,fresh=inputs();out=Path(out).resolve();require(not out.exists(),'new immutable binding/spec directory')
    # This precedes any output. Null, invented, incomplete or wrong-model releases fail closed.
    c.released.verify_release(release_path,release_sha,expected_model='32b',deep=True)
    source=dict(path=d['source_manifest'],sha256=d['source_sha256'])
    inventory=c.identity_index(Path(fresh['identity_file']),fresh)
    groups=[dict(id='B-pdblend-reuse18',system='pdblend',datasets=DATASETS,
                 policy_reference=policies['pdblend'],identity_mode='reuse_only')]
    out.mkdir()
    for system in ('mixed','dynamollm','distserve','ecoserve'):
        original=c.reference(policies[system]);new,mode=candidate(system,original,fresh,c)
        new=r.binding_metadata(new,source,release_path,release_sha,DATASETS)
        new['files'].update(implementation)
        new['scale_binding_derivation']=ref(ROOT/'manifest.json')
        directory=out/system;directory.mkdir();identity=directory/'identity.json'
        write(identity,list(inventory.values()));new['identity_file']=str(identity);new['files'][str(identity)]=sha(identity)
        bp=directory/'binding.json';write(bp,new)
        c.compatible_bindings(policies[system],ref(bp),DATASETS,mode)
        groups.append(dict(id='B-'+system+'-scale18',system=system,datasets=DATASETS,
                           policy_reference=policies[system],identity_mode=mode,scale_binding=ref(bp)))
    spec=dict(schema=2,model='32b',hostname=d['hostname'],protocol_id=c.PROTOCOL,deadline_s=c.DEADLINE,
        source=source,groups=groups,release=ref(release_path),no_additional_gpu_deployment=True)
    checked=c.check_spec(spec,release_path,release_sha)
    require(checked['selected_scale_cells']==90 and checked['reused']>=18,'full original scale domain/reuse')
    sp=out/'spec.json';write(sp,spec)
    result=dict(complete=True,spec=ref(sp),selected_scale_cells=checked['selected_scale_cells'],reused=checked['reused'],pending=checked['pending'],
        cpu_only=True,performance_started=False,live_identity_per_cell_required=True)
    write(out/'receipt.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--release',type=Path);p.add_argument('--release-sha256');p.add_argument('--out',type=Path);a=p.parse_args()
    package_check()
    if a.prepare:
        require(a.release and a.release_sha256 and a.out,'actual release/SHA and new output required')
        print(json.dumps(prepare(a.release,a.release_sha256,a.out)))
    else:print(json.dumps(check(),indent=2))
if __name__=='__main__':main()
