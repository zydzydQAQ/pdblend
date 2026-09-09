"""B-only real150 main proof and release with explicit temporal qualification.

Original per-cell measurement/source/producer checks remain in frozen v2.
No GPU operations and no release from a declaration or partial main domain.
"""
import argparse
import copy
import importlib.util
from pathlib import Path
import json
import socket
import sys
import time

HERE=Path(__file__).resolve().parent
C=HERE.parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
import hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
PARENT=C/'main-first-barrier-v2'
assert sha(PARENT/'partition.py')=='36aa52bfc07fc14e25dd48cac3f317b1bcd2966a2da2642c1c5a268ac07f7747'
assert sha(PARENT/'barrier.py')=='f097fe250049113c3b31733bbfb6ed530a03e5042c12b5060306042189e894d4'
old=sys.modules.get('partition')
try:
    sys.modules['partition']=load('b_main_qualified_partition',PARENT/'partition.py')
    per_cell=load('b_main_qualified_parent',PARENT/'barrier.py')
finally:
    if old is None:sys.modules.pop('partition',None)
    else:sys.modules['partition']=old
p,v1=per_cell.p,per_cell.v1
read,require=v1.read,v1.require
PROTOCOL,DEADLINE=per_cell.PROTOCOL,per_cell.DEADLINE
CORRECTNESS='legacy-temporal-default-trajectory-exact-v2'
KIND='model-main-release-per-cell-qualified-temporal-v2'
QUALIFIER=C/'B32B-temporal-qualification-v2/qualification.py'
QUALIFIER_SHA='693d7b47ae5473c6fae0e0b21b3a6083b7599980a0873f22271c4fc350ffd1a1'
QUALIFIER_MANIFEST_SHA='fe69a81abe3cddd789dd91041faddbae2dc6f82bada79cb7375ad93542dc6633'
NEEDED=('ordinary','pd','cancel','temporal_native_trajectory_exact','native_cleanup','identity','all8_measurement','clock')
HOST=C.parent/'releases/five-system100-B32B-v1-runtime'
ECO_CONFIG_SHA={'alpaca':'34fc552fce6d8fd2dcaf31ed31c3a5ab9d56bf842a5e95391495b16c68997e41',
    'sharegpt':'9a3c39c0708de47e600155c4710aecef6324c9d77cbc193747863f35abaac98c',
    'longbench':'ea9af3b79ff4970c8ec6cf11f22a6b1b92d74cb6147d77cd89915e554cf9fa68'}
SOURCE=C/'five-system-fixed-window-v1/sources/B32B/manifest.json'
ORIGINAL_BINDINGS={
    'pdblend':C/'B32B-five-system100-v1/binding.pdblend.r2.json',
    'mixed':C/'B32B-baseline-sequence-v1/attempt-001/bindings/mixed/binding.json',
    'dynamollm':C/'B32B-baseline-main-first-sequence-v1/attempt-001/bindings/dynamollm-resident/binding.json',
    'distserve':C/'B32B-baseline-main-first-sequence-v1/attempt-001/bindings/distserve/binding.json'}

def prepare_spec(out,eco_binding=None,eco_sha=None):
    """A declaration only. Missing future Eco binding is explicitly null."""
    require(not Path(out).exists(),'new declaration path required')
    reader=v1.Reader();source=reader.read(str(SOURCE),v1.SOURCE_SHA['32b']);rows=v1.source_rows(source,'32b')
    refs={s:dict(path=str(p),sha256=reader.digest(str(p))) for s,p in ORIGINAL_BINDINGS.items()}
    if eco_binding is not None:
        require(v1.valid_sha(eco_sha),'actual Eco binding needs a fixed SHA')
        ref=dict(path=str(Path(eco_binding).resolve()),sha256=eco_sha);b=p.ref(reader,ref);audit_eco_binding(b,reader);refs['ecoserve']=ref
    groups=[]
    for system in v1.SYSTEMS:
        ref=refs.get(system)
        groups.append(dict(id='B-'+system+'-original-main30',system=system,datasets=list(v1.DATASETS),
            cell_ids=[r['cell_id'] for r in rows if r['system']==system],binding=ref,policy_reference=ref))
    value=dict(schema=2,kind='B-qualified-main-declaration-not-proof',model='32b',hostname=v1.HOSTS['32b'],
        protocol_id=PROTOCOL,deadline_s=DEADLINE,source_manifest=str(SOURCE),source_sha256=v1.SOURCE_SHA['32b'],
        groups=groups,retained_failure_refs=[],actual_main_completion_asserted=False,actual_release_created=False,
        missing_future_bindings=[g['system'] for g in groups if g['binding'] is None])
    if eco_binding is not None:p.partition_contract(groups,rows)
    reader.stable();v1.write_new(out,value);return dict(spec=str(Path(out).resolve()),sha256=sha(out),
        missing_future_bindings=value['missing_future_bindings'],actual_main_completion_asserted=False)

def package_check():
    m=read(HERE/'manifest.json')
    for path,h in m['files'].items():require(sha(HERE/path)==h,'B release source changed')
    for path,h in m['dependencies'].items():require(sha(path)==h,'B release dependency changed')
    per_cell.package_check()

def audit_eco_binding(binding,reader=None):
    """Pure recorded-evidence audit; no lease/network/GPU. Returns compact proof.

    This only handles the explicit qualified B Eco binding. Other strategies
    keep their existing ordinary/PD/temporal readers. It never writes old flags.
    """
    reader=reader or v1.Reader()
    require(binding.get('model')=='32b' and binding.get('system')=='ecoserve'
        and binding.get('protocol_id')==PROTOCOL and binding.get('deadline_s')==DEADLINE
        and binding.get('correctness_protocol_id')==CORRECTNESS,'only explicit B Eco qualified protocol')
    require(binding.get('output_correctness_verified') is True and binding.get('legacy_output_correctness_verified') is False
        and binding.get('legacy_single_vs_pair_exact') is False,'old exact gate must remain failed')
    require(binding.get('host_release')==str(HOST) and set(binding.get('configs',{}))==set(ECO_CONFIG_SHA),'original Eco host/three dataset policy required')
    reader.digest(str(HOST/'manifest.json'),'158605f4ff65c97028c975760b78c97b4f3241bb8655c05f62746487483291bd')
    for ds,digest in ECO_CONFIG_SHA.items():
        path=str(C/'B32B-five-system100-v1/configs'/(ds+'.ecoserve.json'))
        expected=reader.read(path,digest);expected.update(controller_source_release=str(HOST),comparison_system='ecoserve')
        observed=reader.read(binding['configs'][ds],binding['files'][binding['configs'][ds]])
        expected.pop('journal',None);observed.pop('journal',None)
        require(observed==expected,'qualified Eco algorithm/profile/roles/prior/limits/SLO changed')
    refs={k:binding[k] for k in ('qualification','qualified_bootstrap','oracle','qualifier_source')}
    require(refs['qualifier_source']==dict(path=str(QUALIFIER),sha256=QUALIFIER_SHA),'reviewed qualifier source required')
    objects={}
    for name,ref in refs.items():
        require(binding['files'].get(ref['path'])==ref['sha256'],'qualified source not in performance binding')
        if name=='qualifier_source':reader.digest(ref['path'],ref['sha256'])
        else:objects[name]=p.ref(reader,ref)
    reader.digest(str(QUALIFIER.parent/'manifest.json'),QUALIFIER_MANIFEST_SHA)
    module=load('b_release_exact_qualifier',QUALIFIER)
    actual=module.audit_fresh_gate(binding['correctness_evidence'],objects['qualified_bootstrap'],objects['oracle'])
    require(actual==objects['qualification'],'saved qualification differs from all original raw')
    require(actual.get('passed') is True and actual.get('eligible_systems')=={'ecoserve':True}
        and all(actual.get('verified',{}).get(k) is True for k in NEEDED),'complete explicitly qualified mechanisms required')
    bootstrap=objects['qualified_bootstrap']
    def identity_only(i):return {k:v for k,v in i.items() if k!='role'}
    require([identity_only(i) for i in binding['instances']]==[identity_only(i) for i in bootstrap['instances']],
        'performance engine process differs from qualified fresh27')
    inventory=reader.read(binding['identity_file'],binding['files'][binding['identity_file']])
    original_inventory=reader.read(bootstrap['identity_file'],bootstrap['files'][bootstrap['identity_file']])
    norm=lambda x:sorted((json.dumps(r,sort_keys=True) for r in x))
    require(norm(inventory)==norm(original_inventory),'complete physical process inventory changed')
    proof=binding['mechanism_proof']
    require(proof.get('verified')==actual['verified'] and proof.get('legacy_verified')==actual['original_mechanism_gate']
        and proof.get('overall_runtime_gate_passed') is False and proof.get('original_failure_preserved') is True
        and proof.get('qualified_under_explicit_protocol')==CORRECTNESS,'binding rewrote original gate qualification')
    for path,h in actual['files'].items():
        require(binding['files'].get(path)==h,'qualification raw omitted from binding');reader.digest(path,h)
    status=reader.read(str(Path(binding['correctness_evidence'])/'status.json'))
    reader.stable()
    return dict(protocol_id=CORRECTNESS,qualifier_source=refs['qualifier_source'],qualifier_manifest_sha256=QUALIFIER_MANIFEST_SHA,
        qualification=refs['qualification'],qualified_bootstrap=refs['qualified_bootstrap'],oracle=refs['oracle'],
        qualification_canonical_sha256=v1.digest_json(actual),verified=actual['verified'],eligible_systems={'ecoserve':True},
        original_mechanism_gate=actual['original_mechanism_gate'],legacy_single_vs_pair_exact=False,legacy_failure_preserved=True,
        gate_dir=binding['correctness_evidence'],fresh_gate_finished_s=status['finished_s'],
        raw_temporal_exact_header_present='temporal_exact' in reader.read(str(Path(binding['correctness_evidence'])/'checks/checks.json'))['checks'])

def proof_contract(proof):
    require(per_cell.proof_contract(proof)=='32b','B model150 proof required')
    groups=[g for g in proof['groups'] if g['system']=='ecoserve']
    require(len(groups)==1 and len(groups[0]['cell_ids'])==30,'exact original Eco30 partition required')
    quals=proof.get('correctness_qualifications',{});require(set(quals)=={'ecoserve'},'explicit Eco qualification missing')
    q=quals['ecoserve'];group=groups[0];e=proof['evidence_files']
    require(q['binding']==group['binding'] and q['cell_ids']==group['cell_ids']
        and q['protocol_id']==CORRECTNESS and q['qualifier_source']==dict(path=str(QUALIFIER),sha256=QUALIFIER_SHA)
        and q['qualifier_manifest_sha256']==QUALIFIER_MANIFEST_SHA,'wrong qualification source/partition')
    require(q['eligible_systems']=={'ecoserve':True} and all(q['verified'].get(k) is True for k in NEEDED)
        and q['original_mechanism_gate']==dict(ordinary=True,pd=True,temporal=False)
        and q['legacy_single_vs_pair_exact'] is False and q['legacy_failure_preserved'] is True
        and type(q['raw_temporal_exact_header_present']) is bool,'old/new correctness protocols conflated')
    for key in ('binding','qualifier_source','qualification','qualified_bootstrap','oracle'):
        r=q[key];require(v1.valid_sha(r['sha256']) and e.get(r['path'])==r['sha256'],'qualification provenance absent from main evidence')
    require(e.get(str(QUALIFIER.parent/'manifest.json'))==QUALIFIER_MANIFEST_SHA
        and v1.valid_sha(q['qualification_canonical_sha256']),'qualified implementation hash missing')
    for r in proof['records']:
        if r['row']['system']=='ecoserve':
            require(q['fresh_gate_finished_s']<=r['finished_s']<=r['execution']['invocation_finished_s']<=proof['created_s'],
                'Eco producer precedes qualified fresh gate')
    return '32b'

def prove_main(spec_path,out):
    package_check();reader=v1.Reader();spec_path=str(Path(spec_path).resolve());spec=reader.read(spec_path)
    require(spec.get('model')=='32b' and socket.gethostname()==spec['hostname']==v1.HOSTS['32b'],'actual B host required')
    require(spec['schema']==2 and spec['protocol_id']==PROTOCOL and spec['deadline_s']==DEADLINE
        and spec['source_sha256']==v1.SOURCE_SHA['32b'] and time.time()<DEADLINE,'original protocol/source/deadline required')
    require(not Path(out).exists(),'new actual proof path required')
    source=reader.read(spec['source_manifest'],spec['source_sha256']);rows=v1.source_rows(source,'32b')
    p.partition_contract(spec['groups'],rows)
    require(not spec.get('retained_failure_refs'),'B ordinary main partitions do not use C failed-prefix repair')
    records=[];groups=[];qualifications={}
    with v1.fresh_lease() as lease:
        before=v1.process_scan();require(before['no_live_serving_child'],'actual main producer still active')
        for sg in spec['groups']:
            compatibility,binding=p.compatibility(sg,reader)
            ex=p.execution_source(sg,binding,reader,spec['source_manifest'],spec['source_sha256'])
            group=dict(sg,compatibility=compatibility,execution_source=ex);groups.append(group)
            if sg['system']=='ecoserve':
                qualifications['ecoserve']=dict(audit_eco_binding(binding,reader),binding=sg['binding'],cell_ids=sg['cell_ids'])
            for row in rows:
                if row['cell_id'] in sg['cell_ids']:
                    rec=p.record(row,group,binding,reader,[])
                    if sg['system']=='ecoserve':
                        receipt=reader.read(str(Path(binding['output'])/'operations'/row['cell_id']/'receipt.json'))
                        require(receipt['started_s']>=qualifications['ecoserve']['fresh_gate_finished_s'],'Eco ran before qualification')
                    records.append(rec)
        reader.stable();after=v1.process_scan();require(after['no_live_serving_child'],'new serving producer appeared')
        result=dict(schema=2,kind='host-main-proof-per-cell',protocol_id=PROTOCOL,deadline_s=DEADLINE,
            model='32b',hostname=v1.HOSTS['32b'],source_manifest=spec['source_manifest'],source_sha256=spec['source_sha256'],
            source_declaration=source,source_declaration_canonical_sha256=v1.digest_json(source),baseline_systems=list(v1.BASELINES),
            groups=groups,records=records,retained_failures=[],correctness_qualifications=qualifications,
            process_evidence=dict(before=before,after=after),node_lease=lease,created_s=time.time(),spec_path=spec_path,
            spec_sha256=reader.files[spec_path],evidence_files=reader.files,
            scope='Original150 main CPs with exact producer/source; Eco uses explicit matched-native temporal qualification and preserves old exact failure.')
        proof_contract(result);v1.write_new(out,result)
    return dict(main_proof=str(Path(out).resolve()),sha256=sha(out),model='32b',baseline_main=120,pdblend_main=30)

def verify_model_proof(proof,*,deep=False,path_map=None):
    proof_contract(proof)
    if deep:
        per_cell.verify_model_proof(proof,deep=True,path_map=path_map);reader=v1.Reader(path_map)
        group=next(g for g in proof['groups'] if g['system']=='ecoserve')
        binding=p.ref(reader,group['binding']);q=dict(audit_eco_binding(binding,reader),binding=group['binding'],cell_ids=group['cell_ids'])
        require(q==proof['correctness_qualifications']['ecoserve'],'main qualification differs from actual gate')
        for r in proof['records']:
            if r['row']['system']=='ecoserve':
                cp=reader.read(r['checkpoint'],r['checkpoint_sha256']);receipt=reader.read(cp['receipt'],cp['receipt_sha256'])
                require(receipt['started_s']>=q['fresh_gate_finished_s'],'Eco measurement precedes qualified gate')
        reader.stable()
    return '32b'

def release_contract(value,*,expected_model=None,now_s=None):
    require(expected_model in (None,'32b') and value.get('model')=='32b','release only authorizes B32B')
    require(value.get('schema')==2 and value.get('kind')==KIND and value.get('protocol_id')==PROTOCOL
        and value.get('deadline_s')==DEADLINE and value.get('global_release') is False,'explicit B-only release required')
    require(set(value.get('models',{}))==set(value.get('proof_refs',{}))=={'32b'}
        and value.get('baseline_systems')==list(v1.BASELINES) and value.get('coordinator_deep_verification') is True
        and (value.get('main_records'),value.get('baseline_main_records'),value.get('pdblend_main_records'))==(150,120,30),
        'all actual150 main, including all Eco30, required')
    proof=value['models']['32b'];proof_contract(proof);ref=value['proof_refs']['32b']
    require(v1.valid_sha(ref['sha256']) and ref['canonical_sha256']==v1.digest_json(proof),'actual proof hash differs')
    require(proof['created_s']<=value['created_s']<=(time.time() if now_s is None else now_s)<DEADLINE,'publication/deadline differs')
    return '32b'

def assemble_release(proof_path,proof_sha,out,*,path_map=None):
    package_check();require(not Path(out).exists() and v1.valid_sha(proof_sha) and sha(proof_path)==proof_sha,'new output/pinned proof required')
    proof=read(proof_path);verify_model_proof(proof,deep=True,path_map=path_map)
    require(sha(proof_path)==proof_sha,'proof changed during complete raw review')
    result=dict(schema=2,kind=KIND,model='32b',protocol_id=PROTOCOL,deadline_s=DEADLINE,models={'32b':proof},
        proof_refs={'32b':dict(path=str(Path(proof_path).resolve()),sha256=proof_sha,canonical_sha256=v1.digest_json(proof))},
        baseline_systems=list(v1.BASELINES),created_s=time.time(),coordinator_hostname=socket.gethostname(),
        coordinator_deep_verification=True,main_records=150,baseline_main_records=120,pdblend_main_records=30,global_release=False,
        scale_authorization='Only B after all real B main150; original scale rows, own lease/deadline, fresh binding and exact main references still required.')
    release_contract(result,expected_model='32b');v1.write_new(out,result)
    return dict(release=str(Path(out).resolve()),sha256=sha(out),model='32b',kind=KIND,main_records=150,global_release=False)

def verify_release(path,expected_sha256,*,expected_model=None,expected_protocol_id=PROTOCOL,expected_deadline_s=DEADLINE,deep=False,path_maps=None):
    require(v1.valid_sha(expected_sha256) and sha(path)==expected_sha256,'explicit fixed release SHA required')
    require(expected_protocol_id==PROTOCOL and expected_deadline_s==DEADLINE,'original scale protocol/deadline')
    value=read(path);release_contract(value,expected_model=expected_model)
    if deep:verify_model_proof(value['models']['32b'],deep=True,path_map=(path_maps or {}).get('32b'))
    require(sha(path)==expected_sha256,'release changed during read')
    return dict(released=True,kind=KIND,model='32b',protocol_id=PROTOCOL,deadline_s=DEADLINE,main_records=150,
        global_release=False,verified_scope='this_model_raw_files_and_contract' if deep else 'pinned_model_release_sha_and_contract',
        root_release_sha256=expected_sha256,local_deadline_and_lease_still_required=True,per_cell_main_sources=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    s=sub.add_parser('prepare-spec');s.add_argument('--out',type=Path,required=True);s.add_argument('--eco-binding',type=Path);s.add_argument('--eco-binding-sha256')
    s=sub.add_parser('prove-main');s.add_argument('--spec',type=Path,required=True);s.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('assemble-release');s.add_argument('--proof',type=Path,required=True);s.add_argument('--proof-sha256',required=True);s.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('verify-release');s.add_argument('--release',type=Path,required=True);s.add_argument('--sha256',required=True);s.add_argument('--model',default='32b',choices=['32b']);s.add_argument('--deep',action='store_true')
    a=parser.parse_args();package_check()
    if a.command=='check':result=dict(package_valid=True,actual_release_created=False,actual_main150_asserted=False)
    elif a.command=='prepare-spec':result=prepare_spec(a.out,a.eco_binding,a.eco_binding_sha256)
    elif a.command=='prove-main':result=prove_main(a.spec,a.out)
    elif a.command=='assemble-release':result=assemble_release(a.proof,a.proof_sha256,a.out)
    else:result=verify_release(a.release,a.sha256,expected_model=a.model,deep=a.deep)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
