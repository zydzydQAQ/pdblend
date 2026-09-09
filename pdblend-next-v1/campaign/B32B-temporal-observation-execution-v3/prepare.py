"""Default: CPU/hash check. --prepare freezes a future single attempt after 90 mains."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
import common as c
import technical

def prepare(out,proof):
    c.package_check();c.require(not os.environ.get('PDBLEND_NODE_LOCK_FD'),'inherited node lease forbidden')
    gate=c.main_gate(proof);out=Path(out).resolve();c.require(not out.exists(),'new prepared directory required')
    c.require(out.parent==c.ROOT.parent and out.name.startswith('B32B-temporal-observation-attempt-'),'independent sibling attempt namespace required')
    authorization=technical.verify();c.require(out.name==authorization['attempt_name'],'only authorized attempt002')
    previous=Path(authorization['restored_binding']);binding=c.read(previous);c.binding_scope(binding)
    deployment=c.read(c.DEPLOYMENT)
    c.verify_files(binding['files'])
    # Full weights are read once during offline preparation; execution checks same inode/stat.
    # This can warm OS cache. The diagnostic is not a disk-cold startup benchmark.
    for p,v in binding.get('large_inputs',{}).items():
        c.require(c.sha(p)==v['sha256'],'model weight/source identity changed: '+p)
        c.require(c.module('executor').stat_identity(p)==v['stat'],'model file stat changed: '+p)
    out.mkdir(parents=True);obs=c.read(c.CANDIDATE/'specs/original-vs-continuous.json')
    obs['output_dir']=str(out/'results/capture-live');technical.compact_spec(out/'observation-spec.json',obs)
    c.write(out/'loader-preflight.json',technical.spec_preflight(out/'observation-spec.json'))
    instance=copy.deepcopy(deployment['instances'][1]);cfg=c.read(instance['config'])
    iid='temporalobsb2';port=34502;kv=34764
    cfg.update(id=iid,port=port,kv_port=kv,runtime_dir=str(out/'results/runtime'),initial_generation=0,
        role='mixed',peers={iid:dict(host='127.0.0.1',tp=2,kv_port=kv)})
    c.write(out/'engine.json',cfg)
    instance.update(id=iid,port=port,kv_port=kv,url=f'http://127.0.0.1:{port}',role='mixed',
        config=str(out/'engine.json'),container_name='pdb-v2-temporalobsb2',image=None)
    instance['environment'] += ['PDBLEND_DIAGNOSTIC_SPEC='+str(out/'observation-spec.json'),
        'PDBLEND_DIAGNOSTIC_SPEC_SHA256='+c.sha(out/'observation-spec.json')]
    context=out/'image-context';context.mkdir()
    for name in ('model_runner.py','pdblend_diagnostics.py'):shutil.copyfile(c.CANDIDATE/name,context/name)
    tag='pdblend-temporal-parent-d114:'+c.sha(c.CANDIDATE/'manifest.json')[:12]
    (context/'Dockerfile').write_text('FROM '+tag+'\nCOPY model_runner.py /usr/local/lib/python3.10/dist-packages/vllm/worker/model_runner.py\nCOPY pdblend_diagnostics.py /usr/local/lib/python3.10/dist-packages/vllm/pdblend_diagnostics.py\n')
    files=dict(gate['files']);files.update(binding['files']);files.update(authorization['files'])
    files.update({str(p):c.sha(p) for p in out.rglob('*') if p.is_file()})
    files[str(c.ROOT/'manifest.json')]=c.sha(c.ROOT/'manifest.json')
    spec=dict(schema=1,purpose='one bounded B temporal metadata/logits observation',hardware_executed=False,
        hostname=c.NODE,model='32b',deadline_s=c.DEADLINE,prepared_s=time.time(),results=str(out/'results'),
        historical_main_qualification_only=True,technical_correction_authorization=str(technical.AUTH),
        main_proof=str(Path(proof).resolve()),main_scope=['mixed','dynamollm','distserve'],
        previous_binding=str(previous),previous_binding_sha256=c.sha(previous),original_deployment=str(c.DEPLOYMENT),
        original_deployment_sha256=c.DEPLOYMENT_SHA,host_release=str(c.HOST),candidate_manifest_sha256=c.CANDIDATE_SHA,
        observation_spec=str(out/'observation-spec.json'),observation_spec_sha256=c.sha(out/'observation-spec.json'),
        image_context=str(context),parent_image=c.IMAGE,parent_tag=tag,diagnostic_instance=instance,
        installed_parent_sources=c.actual_source_mapping(),installed_diagnostic_sources=c.actual_source_mapping(True),
        files=files,large_inputs=binding.get('large_inputs',{}),total_cap_s=900,startup_cap_s=150,work_cap_s=390,
        cleanup_cap_s=90,restore_cap_s=240,tail_cap_s=30,
        attempt_claim=str(c.ROOT/'execution-once.json'),
        restore_all_original_containers=True,fresh_correctness_and_binding_required_after_restore=True,
        old_binding_reusable_after_restart=False,performance_evidence=False,automatic_retries=False)
    c.write(out/'spec.json',spec)
    return dict(cpu_only=True,prepared=True,spec=str(out/'spec.json'),spec_sha256=c.sha(out/'spec.json'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--main-proof',type=Path,default=c.SEQUENCE/'main-proof.json');a=p.parse_args();c.package_check()
    if not a.prepare:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'actual-spec.json';value=c.read(c.CANDIDATE/'specs/original-vs-continuous.json');value['output_dir']=str(c.ROOT.parent/'B32B-temporal-observation-attempt-002/results/capture-live');technical.compact_spec(path,value);proof=technical.spec_preflight(path)
        print(json.dumps(dict(cpu_only=True,hardware_actions=False,prepared=False,loader_preflight=proof)));return
    c.require(a.out is not None,'new output path required');print(json.dumps(prepare(a.out,a.main_proof)))
if __name__=='__main__':main()
