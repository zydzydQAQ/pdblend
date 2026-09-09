"""Version2 main release: exact cell partitions, preserving all actual sources."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import time
import partition as p

v1=p.v1
require,read,sha=v1.require,v1.read,v1.sha
PROTOCOL,DEADLINE=v1.PROTOCOL,v1.DEADLINE
HERE=Path(__file__).resolve().parent
DEFAULT_RELEASE=HERE/'release/global-main-release.json'


def package_check():
    m=read(HERE/'manifest.json')
    for path,h in m['files'].items():require(sha(HERE/path)==h,'v2 source changed: '+path)
    for path,h in m['dependencies'].items():require(sha(path)==h,'original verifier dependency changed: '+path)


def proof_contract(proof):
    if proof.get('schema')==1:return v1.proof_contract(proof)
    model=proof.get('model')
    require(model in v1.MODELS and proof.get('schema')==2 and proof.get('kind')=='host-main-proof-per-cell'
        and proof.get('protocol_id')==PROTOCOL and proof.get('deadline_s')==DEADLINE
        and proof.get('hostname')==v1.HOSTS[model],'wrong v2 host/protocol/deadline')
    require(proof['source_sha256']==v1.SOURCE_SHA[model] and proof['baseline_systems']==list(v1.BASELINES),'wrong original main source/system set')
    rows=v1.source_rows(proof['source_declaration'],model);expected={r['cell_id']:r for r in rows}
    require(v1.digest_json(proof['source_declaration'])==proof['source_declaration_canonical_sha256'],'source declaration altered')
    p.partition_contract(proof['groups'],rows)
    groups={g['id']:g for g in proof['groups']};records=proof['records'];e=proof['evidence_files']
    require(len(records)==150 and len({r['row']['cell_id'] for r in records})==150 and len({r['checkpoint'] for r in records})==150
        and {r['row']['cell_id'] for r in records}==set(expected),'all150 true main records including Eco/PDB required')
    require(e and all(v1.valid_sha(h) for h in e.values()) and e.get(proof['source_manifest'])==proof['source_sha256'],'raw SHA/source map missing')
    for g in groups.values():
        for ref in (g['binding'],g['policy_reference']):require(e.get(ref['path'])==ref['sha256'],'partition original/actual binding SHA not in evidence')
        host=g['compatibility']['host'];require(host['kind'] in ('identical','cooperative-admission-yield-v1'),'unreviewed host exception')
        if host['kind']=='cooperative-admission-yield-v1':
            require(model=='7b' and host['old_host']==str(p.OLD_HOST) and host['host']==str(p.NEW_HOST)
                and host['old_manifest_sha256']==p.OLD_MANIFEST and host['manifest_sha256']==p.NEW_MANIFEST
                and host['changed_files']==[p.ADMISSION],'repair identity/exception broadened')
        else:require(host['host']==host['old_host'] and host['manifest_sha256']==host['old_manifest_sha256'] and not host['changed_files'],'identical host claim differs')
        require(e.get(str(Path(host['host'])/'manifest.json'))==host['manifest_sha256'],'actual host source manifest absent')
        ex=g['execution_source'];require(e.get(ex['reference']['path'])==ex['reference']['sha256']
            and ex['parent_manifest']==proof['source_manifest'] and ex['parent_manifest_sha256']==proof['source_sha256']
            and ex['original_cell_ids']==g['cell_ids'],'actual execution subset source missing')
        if host['kind']=='cooperative-admission-yield-v1':
            original=[x for x in groups.values() if x['system']==g['system'] and x['binding']==g['policy_reference']]
            require(original,'repaired partition lacks the retained actual original policy partition')
            for ds,cfg in g['compatibility']['configs'].items():
                peers=[x['compatibility']['configs'][ds] for x in original if ds in x['datasets']]
                require(peers and all(x['policy_normalized_sha256']==cfg['policy_normalized_sha256'] for x in peers),
                    'same-dataset source partitions changed original online policy')
    for r in records:
        row=r['row'];g=groups[r['partition_id']]
        require(row==expected[row['cell_id']] and row['cell_id'] in g['cell_ids']
            and r['binding']==g['binding']['path'] and r['binding_sha256']==g['binding']['sha256']
            and r['measurement_valid'] is True,'record not in its exact executing partition')
        for key,hkey in (('checkpoint','checkpoint_sha256'),('binding','binding_sha256'),('config','config_sha256')):
            require(v1.valid_sha(r[hkey]) and e.get(r[key])==r[hkey],'per-cell raw reference missing')
        require(e.get(row['trace'])==row['trace_sha256'],'per-cell original trace SHA absent')
        require(type(r['n_expected']) is int and r['n_expected']==row['n_requests']
            and type(r['completed_work_requests']) is int and type(r['good_requests']) is int
            and 0<=r['good_requests']<=r['completed_work_requests']<=r['n_expected']
            and type(r['work_complete']) is bool and type(r['energy_j']) in (int,float)
            and math.isfinite(r['energy_j']) and r['energy_j']>=0,'offered/good/full-energy evidence differs')
        ex=r['execution'];require(e.get(ex['invocation'])==ex['invocation_sha256']
            and ex['execution_manifest']==g['execution_source']['reference']
            and ex['host_release']==g['compatibility']['host']['host']
            and ex['host_manifest_sha256']==g['compatibility']['host']['manifest_sha256']
            and ex['actual_engine_identity'] and r['finished_s']<=ex['invocation_finished_s']<=DEADLINE,
            'actual invocation/host/engine identity absent')
    for f in proof['retained_failures']:
        require(f['counts_as_completed'] is False and f['energy_windows_overlap_not_added'] is True
            and e.get(f['reference']['path'])==f['reference']['sha256']
            and all(e.get(path)==h for path,h in f['raw_files'].items()),'original invalid attempt was hidden or counted')
    for side in ('before','after'):
        x=proof['process_evidence'][side]
        require(x['hostname']==v1.HOSTS[model] and x['no_live_serving_child'] is True and x['live_serving_children']==[],
            'real host has no final no-serving-child proof')
    require(proof['process_evidence']['before']['observed_s']<=proof['process_evidence']['after']['observed_s']<=proof['created_s']<DEADLINE,
        'actual host proof time differs')
    require(proof['node_lease']==dict(path=str(v1.LEASE),exclusive=True,inherited=False),'fresh real host lease missing')
    return model


def prove_main(spec_path,out):
    package_check();reader=v1.Reader();spec_path=str(Path(spec_path).resolve());spec=reader.read(spec_path);model=spec['model']
    require(model in v1.MODELS and socket.gethostname()==spec['hostname']==v1.HOSTS[model],'must run on the actual model host')
    require(spec['schema']==2 and spec['protocol_id']==PROTOCOL and spec['deadline_s']==DEADLINE
        and spec['source_sha256']==v1.SOURCE_SHA[model] and time.time()<DEADLINE,'original protocol/source/deadline required')
    require(not Path(out).exists(),'proof output must be new')
    source=reader.read(spec['source_manifest'],spec['source_sha256']);rows=v1.source_rows(source,model)
    p.partition_contract(spec['groups'],rows) # Null future bindings reject before host lease/publication.
    failures=[p.retained_failure(reader,r) for r in spec.get('retained_failure_refs',[])]
    records=[];groups=[]
    with v1.fresh_lease() as lease:
        before=v1.process_scan();require(before['no_live_serving_child'],'main serving process still active')
        for source_group in spec['groups']:
            compatibility,binding=p.compatibility(source_group,reader)
            require(binding['files'].get(spec['source_manifest'])==spec['source_sha256'],'work source not bound')
            ex=p.execution_source(source_group,binding,reader,spec['source_manifest'],spec['source_sha256'])
            group=dict(source_group,compatibility=compatibility,execution_source=ex);groups.append(group)
            for row in rows:
                if row['cell_id'] in group['cell_ids']:records.append(p.record(row,group,binding,reader,failures))
        reader.stable();after=v1.process_scan();require(after['no_live_serving_child'],'new serving process appeared')
        proof=dict(schema=2,kind='host-main-proof-per-cell',protocol_id=PROTOCOL,deadline_s=DEADLINE,model=model,hostname=v1.HOSTS[model],
            source_manifest=spec['source_manifest'],source_sha256=spec['source_sha256'],source_declaration=source,
            source_declaration_canonical_sha256=v1.digest_json(source),baseline_systems=list(v1.BASELINES),
            groups=groups,records=records,retained_failures=failures,process_evidence=dict(before=before,after=after),
            node_lease=lease,created_s=time.time(),spec_path=spec_path,spec_sha256=reader.files[spec_path],evidence_files=reader.files,
            scope='Actual per-cell original source retained; all150 measured main CPs, not a single-version performance victory.')
        proof_contract(proof);v1.write_new(out,proof)
    return dict(main_proof=str(Path(out).resolve()),sha256=sha(out),model=model,baseline_main=120,pdblend_main=30)


def verify_model_proof(proof,*,deep=False,path_map=None):
    if proof.get('schema')==1:return v1.verify_model_proof(proof,deep=deep,path_map=path_map)
    model=proof_contract(proof)
    if deep:
        reader=v1.Reader(path_map)
        require(reader.read(proof['source_manifest'],proof['source_sha256'])==proof['source_declaration'],'embedded source differs')
        for path,h in proof['evidence_files'].items():reader.digest(path,h)
        failures=[p.retained_failure(reader,f['reference']) for f in proof['retained_failures']]
        require(failures==proof['retained_failures'],'retained failed measurement facts changed')
        groups={}
        for g in proof['groups']:
            compat,binding=p.compatibility(g,reader);require(compat==g['compatibility'],'host/config policy compatibility differs')
            ex=p.execution_source(g,binding,reader,proof['source_manifest'],proof['source_sha256'])
            require(ex==g['execution_source'],'actual source subset derivation differs')
            groups[g['id']]=(g,binding)
        for r in proof['records']:
            v1.verify_record(r,reader);g,binding=groups[r['partition_id']]
            p.execution(r['row'],g,binding,reader,failures,supplied=r['execution'])
        reader.stable()
    return model


def assemble_release(refs,out,*,path_maps=None):
    package_check();require(len(refs)==3 and not Path(out).exists(),'three original host proofs and new release path required')
    models={};proof_refs={}
    for path,h in refs:
        require(v1.valid_sha(h) and sha(path)==h,'host proof SHA missing/changed')
        proof=read(path);model=verify_model_proof(proof,deep=True,path_map=(path_maps or {}).get(proof['model']))
        require(model not in models,'duplicate model proof');models[model]=proof
        proof_refs[model]=dict(path=str(Path(path).resolve()),sha256=h,canonical_sha256=v1.digest_json(proof))
    require(set(models)==set(v1.MODELS) and time.time()<DEADLINE,'all450 main cells and original deadline required')
    for model,ref in proof_refs.items():
        require(sha(ref['path'])==ref['sha256'],'host proof changed during assembly')
        reader=v1.Reader((path_maps or {}).get(model))
        for path,h in models[model]['evidence_files'].items():reader.digest(path,h)
    release=dict(schema=2,kind='global-main-release-per-cell',protocol_id=PROTOCOL,deadline_s=DEADLINE,
        baseline_systems=list(v1.BASELINES),models=models,proof_refs=proof_refs,created_s=time.time(),
        coordinator_hostname=socket.gethostname(),coordinator_deep_verification=True,main_records=450,
        baseline_main_records=360,pdblend_main_records=90,
        compact_trust_scope='Explicit pinned root SHA; other hosts do not claim remote raw reread.',
        scale_authorization='Complete real main domain only; exact per-cell references and explicitly compatible source still required.')
    v1.write_new(out,release);return dict(release=str(Path(out).resolve()),sha256=sha(out),main_records=450)


def verify_release(path,expected_sha256,*,expected_protocol_id=PROTOCOL,expected_deadline_s=DEADLINE,deep=False,path_maps=None):
    require(v1.valid_sha(expected_sha256) and sha(path)==expected_sha256,'explicit fixed global release SHA required')
    value=read(path)
    if value.get('schema')==1:
        return v1.verify_release(path,expected_sha256,expected_protocol_id=expected_protocol_id,expected_deadline_s=expected_deadline_s,deep=deep,path_maps=path_maps)
    require(time.time()<DEADLINE and value.get('schema')==2 and value.get('kind')=='global-main-release-per-cell'
        and value.get('protocol_id')==expected_protocol_id==PROTOCOL and value.get('deadline_s')==expected_deadline_s==DEADLINE,
        'per-cell release protocol/deadline differs')
    require(value.get('baseline_systems')==list(v1.BASELINES) and set(value.get('models',{}))==set(v1.MODELS)
        and value.get('coordinator_deep_verification') is True and value.get('main_records')==450
        and value.get('baseline_main_records')==360 and value.get('pdblend_main_records')==90,'450 complete main domain absent')
    for model,proof in value['models'].items():
        require(model==verify_model_proof(proof,deep=deep,path_map=(path_maps or {}).get(model)),'swapped model proof')
        r=value['proof_refs'][model]
        require(v1.valid_sha(r['sha256']) and r['canonical_sha256']==v1.digest_json(proof)
            and proof['created_s']<=value['created_s']<DEADLINE,'original proof/publication differs')
    require(sha(path)==expected_sha256,'release changed during read')
    return dict(released=True,protocol_id=PROTOCOL,deadline_s=DEADLINE,main_records=450,
        verified_scope='all_model_raw_files_and_contract' if deep else 'root_verified_release_sha_and_contract',
        root_release_sha256=expected_sha256,local_deadline_and_lease_still_required=True,per_cell_main_sources=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prove-main');q.add_argument('--spec',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('assemble-release');q.add_argument('--proof',nargs=2,action='append',required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--path-maps',type=Path)
    q=sub.add_parser('verify-release');q.add_argument('--release',type=Path,required=True);q.add_argument('--sha256',required=True);q.add_argument('--deep',action='store_true');q.add_argument('--path-maps',type=Path)
    a=parser.parse_args();package_check()
    if a.cmd=='prove-main':result=prove_main(a.spec,a.out)
    elif a.cmd=='assemble-release':result=assemble_release(a.proof,a.out,path_maps=read(a.path_maps) if a.path_maps else None)
    else:result=verify_release(a.release,a.sha256,deep=a.deep,path_maps=read(a.path_maps) if a.path_maps else None)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
