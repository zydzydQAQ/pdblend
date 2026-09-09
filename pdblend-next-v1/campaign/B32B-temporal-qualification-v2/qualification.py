"""Explicit versioned, read-only EcoServe temporal qualification.

No GPU calls and no performance runner. Original gate flags are never rewritten.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PROTOCOL='legacy-temporal-default-trajectory-exact-v2'
DEADLINE=1788872770.0400891

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def modules():
    g=load('qualified_shape_gate',ROOT/'shape_gate.py')
    old=sys.modules.get('shape_gate')
    try:
        sys.modules['shape_gate']=g;o=load('qualified_native_oracle',ROOT/'oracle.py')
    finally:
        if old is None:sys.modules.pop('shape_gate',None)
        else:sys.modules['shape_gate']=old
    return g,o,load('qualified_instant_power',ROOT/'power_source.py')

def check_package():
    manifest=read(ROOT/'manifest.json');files={str(ROOT/'manifest.json'):sha(ROOT/'manifest.json')}
    for name,digest in manifest['files'].items():
        p=(ROOT/name).resolve();require(ROOT in p.parents and sha(p)==digest,'qualification source changed')
        files[str(p)]=digest
    for path,digest in manifest['external_files'].items():
        require(sha(path)==digest,'frozen qualification dependency changed');files[path]=digest
    return files

def audit_fresh_gate(gate_dir,binding,oracle):
    """Parsed binding/oracle objects. Re-read all oracle and gate inputs.

    Caller separately pins the actual binding and oracle file SHAs and checks
    live identity under its own fresh lease immediately before/after binding.
    This reader proves recorded process identity, not current process liveness.
    """
    files=check_package();g,o,power=modules();gate=Path(gate_dir).resolve()
    require(isinstance(binding,dict) and isinstance(oracle,dict),'parsed actual binding/oracle objects required')
    require(oracle.get('kind')=='registered-native-default-temporal-oracle' and oracle.get('protocol_id')==PROTOCOL,
        'unregistered oracle/CPU fixture cannot qualify')
    actual=o.verify(oracle['inputs']);require(actual==oracle,'oracle differs from complete raw revalidation')
    files.update(actual['files']);files.update(power.SOURCES)
    restored=o.read(o.ATTEMPT/'results/restored-bootstrap.binding.json')
    require(binding.get('model')=='32b' and binding.get('deadline_s')==DEADLINE and binding.get('configs')=={}
        and binding.get('instances')==restored['instances'],'fresh correctness-only bootstrap differs from native restored processes')
    require(binding.get('window_s')==100 and binding.get('seeds')==[701]
        and binding.get('correctness_gate_required_before_performance') is True
        and binding.get('output_correctness_verified') is False,'bootstrap is not the declared unqualified 100s protocol')
    require(binding.get('identity_file') and binding.get('files',{}).get(binding['identity_file'])==sha(binding['identity_file']),
        'fresh identity file not bound')
    files[binding['identity_file']]=sha(binding['identity_file'])
    for i in binding['instances']:
        require(i['tp']==2 and i['native_kind']=='legacy_sync_put' and i.get('scheduler_cache_observed') is False,
            'original legacy TP2 scope changed')
        for path,digest in i['provenance']['source_files_at_import'].items():
            require(binding['files'].get(path)==digest==sha(path),'actual imported source not frozen');files[path]=digest
        path=i['engine_config'];require(binding['files'].get(path)==sha(path),'actual engine config not frozen');files[path]=sha(path)
    status=read(gate/'status.json')
    require(actual['physical_evidence']['full_operation_end_s']<=status['started_s']<=status['measurement_start_s']
        <status['measurement_end_s']<=status['finished_s']<=DEADLINE,'gate precedes fresh restore or exceeds original deadline')
    inspected=g.inspect_fresh_gate(gate,binding,actual['reference_tokens_by_label'],power.load())
    files.update(inspected['files']);t=inspected['temporal'];original=inspected['original_gate']
    require(original['verified']['ordinary'] and original['verified']['pd'] and t['temporal_native_trajectory_exact'],
        'required original checks or new trajectory failed')
    for path,digest in files.items():require(o.sha(path)==digest,'qualification input changed during audit')
    return dict(schema=2,kind='temporal-default-trajectory-qualification',protocol_id=PROTOCOL,passed=True,
        eligible_systems={'ecoserve':True},verified=dict(ordinary=True,pd=True,cancel=True,
            temporal_native_trajectory_exact=True,native_cleanup=True,identity=True,all8_measurement=True,clock=True),
        original_mechanism_gate=original['verified'],legacy_single_vs_pair_exact=t['legacy_single_vs_pair_exact'],
        legacy_first_differences=t['legacy_first_differences'],legacy_failure_preserved=True,
        inputs=dict(gate_dir=str(gate),binding_canonical_sha256=canonical(binding),
            oracle_canonical_sha256=canonical(oracle),oracle_inputs=actual['inputs']),
        temporal=t,physical_evidence=original['physical'],files=files,
        verified_identity_scope='recorded fresh four processes; caller must check live identity under own lease',
        numerical_scope='prescribed full64 outputs exact against independent matched native shape; not bitwise KV or general numerical correctness',
        performance_outcomes_certified=False)

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true')
    p.add_argument('--register-oracle',action='store_true');p.add_argument('--native-spec',type=Path)
    p.add_argument('--native-spec-sha256');p.add_argument('--native-package-sha256')
    p.add_argument('--gate',type=Path);p.add_argument('--binding',type=Path);p.add_argument('--binding-sha256')
    p.add_argument('--oracle',type=Path);p.add_argument('--oracle-sha256');p.add_argument('--out',type=Path);a=p.parse_args()
    check_package()
    if a.check or not (a.register_oracle or a.gate):
        require(not a.out,'default check cannot create an actual qualification')
        print(json.dumps(dict(package_valid=True,ready=False,registered_oracle=None,gpu_executed=False)));return
    require(a.out is not None and not a.out.exists(),'new explicit output required')
    if a.register_oracle:
        require(not a.gate and a.native_spec and a.native_spec_sha256 and a.native_package_sha256,'explicit actual native inputs required')
        _,o,_=modules();result=o.verify(o.contract(a.native_package_sha256,a.native_spec,a.native_spec_sha256))
    else:
        require(a.binding and a.oracle and sha(a.binding)==a.binding_sha256 and sha(a.oracle)==a.oracle_sha256,
            'explicit actual bootstrap/oracle file SHA required')
        result=audit_fresh_gate(a.gate,read(a.binding),read(a.oracle))
    write_new(a.out,result);print(json.dumps(dict(output=str(a.out.resolve()),sha256=sha(a.out),gpu_executed=False)))

if __name__=='__main__':main()
