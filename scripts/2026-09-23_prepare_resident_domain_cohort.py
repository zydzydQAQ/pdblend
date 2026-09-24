#!/usr/bin/env python3
"""Freeze executable profile epoch jobs and run their pinned CPU-only preflight."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
IMAGE_TOKEN='pdblend:l20-cu128-vllm-v1@'+IMAGE
VERIFY=ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cohort-spec',type=Path,help='Existing final cohort.json shared with other independent profilers')
    p.add_argument('--resume-short',action='append',default=[],metavar='MODEL=PATH')
    p.add_argument('--cache-manifest',type=Path,default=ROOT/'results/2026-09-23/quad-compiler-cache-archive/manifest.json')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    prepare_argv=['/home/pdblend/.venv/bin/python','-B',str(ROOT/'scripts/2026-09-23_prepare_short_resident.py'),
        '--out',str(out/'packages')]
    for value in args.resume_short:prepare_argv.extend(['--resume-short',value])
    prep=subprocess.run(prepare_argv,check=True,capture_output=True,text=True)
    (out/'prepare.stdout').write_text(prep.stdout)
    review=json.loads((out/'packages/review.json').read_text())
    members={'pdblend-'+name.split('-')[0]:value for name,value in review['members'].items()}
    spec=importlib.util.spec_from_file_location('resident_domain_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    source,source_sha=helper.freeze_source(ROOT/'src',out/'sources')
    helper.verify_snapshot(source,json.loads((source/'manifest.json').read_text())['files'])
    if args.cohort_spec:
        if args.cohort_spec.name!='cohort.json':raise ValueError('shared cohort filename must be cohort.json')
        epochs=args.cohort_spec.resolve().parent;cohort_spec=json.loads(args.cohort_spec.read_text())
        if (not set(members)<=set(cohort_spec['members']) or len(set(cohort_spec['members']))!=len(cohort_spec['members'])):
            raise ValueError('final cohort missing resident profile members')
        cohort=cohort_spec['cohort_id']
    else:
        cohort='pdblend-resident-domains-'+source_sha[:20]
        epochs=out/'epochs';epochs.mkdir()
        write(epochs/'cohort.json',dict(schema=1,cohort_id=cohort,members=list(members),coordinator=True,
            qualification_limit=.05,window_boundary_epochs=True,allow_external_peers=False))
        cohort_spec=json.loads((epochs/'cohort.json').read_text())
    caches=json.loads(args.cache_manifest.read_text())['members']
    jobs=[];preflights={}
    for index,(name,m) in enumerate(members.items()):
        recipe=json.loads(Path(m['recipe']).read_text());job_id='resident-domain-'+name+'-'+source_sha[:16]
        paths=dict(source_manifest=source/'manifest.json',model_verification=VERIFY,cohort=epochs/'cohort.json',
            short_manifest=Path(m['package'])/'manifest.json')
        if m['long_package']:paths['long_manifest']=Path(m['long_package'])/'manifest.json'
        binding=dict(source_sha256=source_sha,image_digest=IMAGE,sampling_cohort=cohort,
            exact_inputs_sha256={k:sha(v) for k,v in paths.items()})
        input_manifest=out/'inputs'/f'{name}.json';write(input_manifest,binding)
        matches=[c for c in caches.values() if (c['model_id'],c['tp'],c['pp'],c['image_digest'])==(m['model_id'],m['tp'],1,IMAGE)]
        if len(matches)!=1:raise ValueError('cache must match exact model topology image')
        cache=matches[0]
        if any(sha(Path(cache['path'])/f)!=h for f,h in cache['files'].items()):raise ValueError('cache archive bytes changed')
        working_cache=out/'compiler-cache'/name;shutil.copytree(cache['path'],working_cache)
        mounts=[(source,'/opt/pdblend-src','ro'),(source/'manifest.json','/source-manifest.json','ro'),
            (Path('/home/models'),'/models','ro'),(VERIFY,'/verification/model-verification.json','ro'),
            ('{attempt_dir}','/output','rw'),(epochs,'/epochs','rw'),(epochs,'/coord','rw'),
            (input_manifest,'/spec/input-manifest.json','ro'),(working_cache,'/root/.cache/vllm','rw')]
        mounts += [(Path(v),v,'ro') for v in sorted(set(m['readonly_roots']))]
        argv=['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN','--ipc=host',
            '--network=host','--shm-size=16g','--ulimit','nofile=65536:65536','--entrypoint','/opt/venv/bin/python']
        for host,target,mode in mounts:argv += ['-v',f'{host}:{target}:{mode}']
        env=dict(PYTHONPATH='/opt/pdblend-src',PYTHONDONTWRITEBYTECODE='1',PDBLEND_MODELS_DIR='/models',
            PDBLEND_MODEL_VERIFICATION_RECEIPT='/verification/model-verification.json',PDBLEND_SOURCE_SHA256=source_sha,
            PDBLEND_SOURCE_MANIFEST='/source-manifest.json',PDBLEND_IMAGE_ID=IMAGE,PDBLEND_HARDWARE_ID='8xL20-lease',
            PDBLEND_VLLM_VERSION='0.10.1.1',CUDA_VERSION='12.8.1',PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',
            CUDA_VISIBLE_DEVICES='{lease_local_indices}',PDBLEND_COORD_DIR='/coord',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        env.update(PDBLEND_SAMPLING_EPOCH_ROOT='/epochs',PDBLEND_PROFILE_MEMBER=name)
        for key,value in env.items():argv += ['-e',key+'='+value]
        argv += ['-e','PDBLEND_CONCURRENCY_ENVIRONMENT','-e','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256',IMAGE_TOKEN,
            '-B','-m','pdblend.profile.resident_domain_job','--model','/models/'+m['model_id'],
            '--gpus',*map(str,range(m['tp'])),'--base-port','{lease_port}','--out','/output',
            '--epochs-root','/epochs','--member',name,'--short-package',m['package'],'--input-manifest','/spec/input-manifest.json']
        if m['long_package']:argv += ['--long-package',m['long_package']]
        payload=dict(schema=1,system='pdblend',model_id=m['model_id'],tp=m['tp'],pp=1,gpu_count=m['tp'],
            source_snapshot=str(source),source_sha256=source_sha,image_digest=IMAGE,argv=argv,container_name=job_id,
            sampling_cohort=cohort,cohort_dir=str(epochs),cohort_member=name,cohort_members=cohort_spec['members'],
            immutable_input_sha256=sha(input_manifest),input_manifest=dict(path=str(input_manifest),sha256=sha(input_manifest)),
            depends_on=[],after_terminal=[],exclusive=False,global_lock=False,formal_eligible=False,energy_comparable=False,
            timeout_s=10800,required_receipts=['completion.json'],completion_is_formal_qualification=False,
            measurement_scope=recipe,compiler_cache_reuse=dict(archive=cache['path'],working_copy=str(working_cache),
                manifest_sha256=sha(args.cache_manifest)),qualification_protocol='SamplingEpochs per-window guard and real isolated/parallel probe')
        jobs.append(dict(job_id=job_id,priority=445-index,max_attempts=1,payload=payload))
        cpu_out=out/'cpu-preflight'/name;cpu_out.mkdir(parents=True)
        cpu=[];i=0
        while i<len(argv):
            value=argv[i]
            if i<argv.index(IMAGE_TOKEN) and value in ('--gpus','--cap-add'):i+=2;continue
            value=value.replace(job_id,job_id+'-cpu').replace('{attempt_dir}',str(cpu_out))
            value=value.replace('{lease_local_indices}',','.join(map(str,range(m['tp'])))).replace('{lease_port}',str(19700+index*100))
            value=value.replace('{lease_gpu_uuids}','cpu-preflight-no-gpu');cpu.append(value);i+=1
        cpu[cpu.index(IMAGE_TOKEN):cpu.index(IMAGE_TOKEN)]=['-e','NVIDIA_VISIBLE_DEVICES=void']
        cpu += ['--preflight-only'];write(cpu_out/'command.json',dict(argv=cpu,hardware_executed=False))
        process=subprocess.run(cpu,capture_output=True,text=True,timeout=120)
        (cpu_out/'stdout').write_text(process.stdout);(cpu_out/'stderr').write_text(process.stderr)
        receipt=cpu_out/'preflight.json'
        preflights[name]=dict(returncode=process.returncode,receipt=str(receipt),receipt_sha256=sha(receipt) if receipt.exists() else None)
    write(out/'jobs.json',jobs)
    passed=all(x['returncode']==0 for x in preflights.values())
    write(out/'review.json',dict(schema=1,status='prepare_only_cpu_container_passed' if passed else 'preflight_failed',
        source_snapshot=str(source),source_sha256=source_sha,image_digest=IMAGE,preflights=preflights,
        sampling_cohort=cohort,members=members,jobs_sha256=sha(out/'jobs.json'),gpu_count=sum(j['payload']['gpu_count'] for j in jobs),
        hardware_executed=False,queue_modified=False,formal_eligible=False,energy_comparable=False,
        long_training_reused=36,new_long_training_points=0,long_holdout_points=24,
        limits=review['limitations']))
    print(json.dumps(dict(status='passed' if passed else 'failed',review=str(out/'review.json'),jobs=str(out/'jobs.json')),indent=2))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
