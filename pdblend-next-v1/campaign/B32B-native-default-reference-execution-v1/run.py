"""Default CPU check; explicit --prepare/--run are separate future operations."""
import argparse, asyncio, copy, json, os, shutil, socket, sys, tempfile, time
from pathlib import Path
import adapter as a

def prepare(out):
    a.check();c,base,loader,Operation=a.load_operation()
    a.require(not os.environ.get('PDBLEND_NODE_LOCK_FD'),'no inherited lease')
    before=a.eligibility(c);out=Path(out).resolve()
    a.require(out==a.C/a.ATTEMPT and not out.exists(),'unique native reference output')
    previous=a.read(a.BOOTSTRAP);c.binding_scope(previous);c.verify_files(previous['files'])
    for p,v in previous.get('large_inputs',{}).items():
        a.require(c.sha(p)==v['sha256'] and c.module('executor').stat_identity(p)==v['stat'],'original model identity')
    out.mkdir();obs=a.read(a.ROOT/'request-spec.json');obs['output_dir']=str(out/'results/capture-live')
    loader.compact_spec(out/'observation-spec.json',obs)
    c.write(out/'loader-preflight.json',loader.spec_preflight(out/'observation-spec.json'))
    instance=copy.deepcopy(c.read(c.DEPLOYMENT)['instances'][1]);cfg=c.read(instance['config'])
    iid,port,kv='nativeref100b1',34504,34828
    cfg.update(id=iid,port=port,kv_port=kv,runtime_dir=str(out/'results/runtime'),initial_generation=0,
        role='mixed',peers={iid:dict(host='127.0.0.1',tp=2,kv_port=kv)})
    c.write(out/'engine.json',cfg)
    instance.update(id=iid,port=port,kv_port=kv,url=f'http://127.0.0.1:{port}',role='mixed',
        config=str(out/'engine.json'),engine_entry=str(a.ROOT/'engine/engine.py'),
        container_name='pdb-v2-nativeref100b1',image=None)
    instance['environment'] += ['PDBLEND_DIAGNOSTIC_SPEC='+str(out/'observation-spec.json'),
        'PDBLEND_DIAGNOSTIC_SPEC_SHA256='+a.sha(out/'observation-spec.json')]
    context=out/'image-context';context.mkdir()
    for f in ('model_runner.py','pdblend_diagnostics.py','scheduler.py'):shutil.copyfile(a.ROOT/'image-context'/f,context/f)
    tag='pdblend-native-reference-parent-d114:'+a.sha(a.ROOT/'manifest.json')[:12]
    (context/'Dockerfile').write_text('FROM '+tag+'\nCOPY model_runner.py /usr/local/lib/python3.10/dist-packages/vllm/worker/model_runner.py\nCOPY pdblend_diagnostics.py /usr/local/lib/python3.10/dist-packages/vllm/pdblend_diagnostics.py\nCOPY scheduler.py /usr/local/lib/python3.10/dist-packages/vllm/core/scheduler.py\n')
    installed=c.actual_source_mapping(True);installed['/usr/local/lib/python3.10/dist-packages/vllm/core/scheduler.py']=a.SCHEDULER_SHA
    files=dict(previous['files']);files.update(before['files'])
    files.update({str(a.ROOT/p):h for p,h in a.read(a.ROOT/'manifest.json')['files'].items()})
    files.update(a.read(a.ROOT/'manifest.json')['external_files']);files[str(a.ROOT/'manifest.json')]=a.sha(a.ROOT/'manifest.json')
    files.update({str(p):a.sha(p) for p in out.rglob('*') if p.is_file()})
    after=a.eligibility(c);a.require(before==after,'producer/eligibility changed during preparation')
    spec=dict(schema=1,purpose='one native default same-trajectory counterfactual; not a replacement baseline oracle',
        hostname=c.NODE,model='32b',deadline_s=c.DEADLINE,prepared_s=time.time(),results=str(out/'results'),
        main_proof=str(c.SEQUENCE/'main-proof.json'),previous_binding=str(a.BOOTSTRAP),previous_binding_sha256=a.BOOTSTRAP_SHA,
        original_deployment=str(c.DEPLOYMENT),original_deployment_sha256=c.DEPLOYMENT_SHA,host_release=str(c.HOST),
        candidate_manifest_sha256=c.CANDIDATE_SHA,reference_package_manifest_sha256=a.sha(a.ROOT/'manifest.json'),
        observation_spec=str(out/'observation-spec.json'),observation_spec_sha256=a.sha(out/'observation-spec.json'),
        image_context=str(context),parent_image=c.IMAGE,parent_tag=tag,diagnostic_instance=instance,
        installed_parent_sources=c.actual_source_mapping(False),installed_diagnostic_sources=installed,
        files=files,large_inputs=previous.get('large_inputs',{}),total_cap_s=900,startup_cap_s=150,work_cap_s=390,
        cleanup_cap_s=90,restore_cap_s=240,tail_cap_s=30,attempt_claim=str(a.ROOT/'execution-once.json'),
        original_temporal_gate_changed=False,fresh_correctness_and_binding_required_after_restore=True,
        restore_all_original_containers=True,performance_evidence=False,automatic_retries=False)
    c.write(out/'spec.json',spec);return dict(prepared=True,cpu_only=True,spec=str(out/'spec.json'),spec_sha256=a.sha(out/'spec.json'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--run',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--spec-sha256');args=p.parse_args()
    a.require(not(args.prepare and args.run),'separate prepare and actual execution');a.check()
    c,base,loader,Operation=a.load_operation()
    if args.prepare:
        a.require(args.out is not None,'explicit new preparation output required');print(json.dumps(prepare(args.out)));return
    if not args.run:
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'spec.json';loader.compact_spec(path,a.read(a.ROOT/'request-spec.json'))
            proof=loader.spec_preflight(path)
        print(json.dumps(dict(cpu_only=True,hardware_actions=False,prepared=False,loader_preflight=proof)));return
    a.require(args.spec is not None and args.spec_sha256 is not None and a.sha(args.spec)==args.spec_sha256,'explicit actual spec SHA')
    spec=a.read(args.spec);c.verify_files(spec['files'])
    a.require(args.spec.resolve()==a.C/a.ATTEMPT/'spec.json' and spec['results']==str(a.C/a.ATTEMPT/'results')
        and spec['attempt_claim']==str(a.ROOT/'execution-once.json'),'isolated output/claim required')
    a.require(spec['hostname']==socket.gethostname()==c.NODE and spec['deadline_s']==c.DEADLINE,'actual B/global deadline')
    a.require(spec['reference_package_manifest_sha256']==a.sha(a.ROOT/'manifest.json') and spec['previous_binding']==str(a.BOOTSTRAP)
        and spec['previous_binding_sha256']==a.BOOTSTRAP_SHA,'exact new package/fresh original binding')
    a.require(not os.environ.get('PDBLEND_NODE_LOCK_FD'),'fresh exclusive node lease only')
    a.require(not (a.ROOT/'STOP').exists() and not Path(spec['results']).exists(),'STOP or existing attempt')
    loader.spec_preflight(spec['observation_spec']);a.eligibility(c)
    sys.path[:0]=[str(c.HOST/'src'),str(c.HOST),'/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.campaign import node_lease
    with node_lease():
        a.eligibility(c);base.deadline_limits(time.time());c.verify_files(spec['files'])
        with Path(spec['attempt_claim']).open('x') as f:json.dump(dict(spec=str(args.spec.resolve()),sha256=a.sha(args.spec),pid=os.getpid(),claimed_s=time.time()),f)
        result=asyncio.run(Operation(spec).run())
    a.require(result['observation_completed'],'reference incomplete; preserve all failure evidence')

if __name__=='__main__':main()
