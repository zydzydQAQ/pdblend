#!/usr/bin/env python3
"""Prepare and CPU-check an immutable 2+2+4-GPU DistServe profile cohort.

This command neither queues jobs nor initializes a GPU. Real isolated/concurrent
qualification is deferred to the original native collector on its owned lease.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from pdblend.bench.resident_session import digest, write_new
from pdblend_baselines.distserve.stage_cohort import MODELS, point_plan, sha
from pdblend_baselines.distserve.stage_collect import load_point_plan
from pdblend_baselines.distserve.stage_ports import CONTRACT

IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
VERIFY=ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'


def ref(path,root):return dict(path=str(path.relative_to(root)),sha256=sha(path))


def job_for(member,row,*,out,source,source_sha,cohort,cohort_sha,image,verification,corpus_root):
    job_id='distserve-targeted-'+member.removeprefix('distserve-')+'-cohort-'+cohort_sha[:16]
    image_token='pdblend:l20-cu128-vllm-v1@'+image
    argv=['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN','--ipc=host',
        '--network=host','--shm-size=16g','--ulimit','nofile=65536:65536','--entrypoint','/opt/venv/bin/python']
    mounts=[(source,'/opt/pdblend-src','ro'),(source/'manifest.json','/source-manifest.json','ro'),
        (Path('/home/models'),'/models','ro'),(verification,'/verification/model-verification.json','ro'),
        (out,'/spec','ro'),(out/'coord','/coord','rw'),(corpus_root,corpus_root,'ro'),('{attempt_dir}','/output','rw'),
        (Path('/proc'),'/host/proc','ro')]
    for host,target,mode in mounts:argv+=['-v',f'{host}:{target}:{mode}']
    env=dict(PYTHONPATH='/opt/pdblend-src',PYTHONDONTWRITEBYTECODE='1',PDBLEND_MODELS_DIR='/models',
        PDBLEND_MODEL_VERIFICATION_RECEIPT='/verification/model-verification.json',PDBLEND_SOURCE_SHA256=source_sha,
        PDBLEND_SOURCE_MANIFEST='/source-manifest.json',PDBLEND_IMAGE_ID=image,PDBLEND_VLLM_VERSION='0.10.1.1',
        CUDA_VERSION='12.8.1',PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',CUDA_VISIBLE_DEVICES='{lease_local_indices}',
        PDBLEND_RESIDENT_LOAD_LOCK='/coord/model-load.lock',PDBLEND_PROFILE_WAVE='/coord',
        PDBLEND_PROFILE_MEMBER=member,PDBLEND_DIST_PORT_GUARD='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    for key,value in env.items():argv+=['-e',key+'='+value]
    argv+=['-e','PDBLEND_CONCURRENCY_ENVIRONMENT','-e','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256',image_token,
        '-B','-m','pdblend_baselines.distserve.stage_cohort','--cohort-inputs','/spec/cohort-inputs.json','--member',member,
        '--model',row['model_id'],'--tp',str(row['tp']),'--gpus','{lease_local_indices}',
        '--base-port','{lease_port}','--point-plan','/spec/'+row['point_plan']['path'],
        '--input-manifest','/spec/plans/'+member+'/input-manifest.json','--out','/output/dist-stage']
    argv=[sys.executable,'-B',str(source/'pdblend_baselines/distserve/stage_ports.py'),
        '--host-out','{attempt_dir}/startup-ports-host','--base-port','{lease_port}','--tp',str(row['tp']),
        '--',*argv]
    payload=dict(schema='distserve-targeted-stage-cohort-docker/v1',system='distserve',model_id=row['model_id'],
        tp=row['tp'],pp=1,gpu_count=row['gpu_count'],exclusive=False,global_lock=False,reserve_host=False,
        sampling_cohort=cohort,cohort_dir=str(out/'coord'),cohort_member=member,cohort_sha256=cohort_sha,
        source_snapshot=str(source),source_sha256=source_sha,image_digest=image,container_name=job_id,argv=argv,
        depends_on=[],after_terminal=[],formal_eligible=False,energy_comparable=False,
        required_receipts=['dist-stage/cohort-preflight.json','dist-stage/completion.json',
            'startup-ports-host/before.json','startup-ports-host/after.json'],timeout_s=18000,
        scope='model-owned B1/2/4 six-frequency stage timing, power and independent holdout',
        sampling_complete_is_calibration_pass=False)
    return dict(job_id=job_id,priority=390,max_attempts=1,payload=payload)


def cpu_command(job,out,index):
    source=job['payload']['argv'];token='pdblend:l20-cu128-vllm-v1@'+job['payload']['image_digest']
    boundary=source.index(token);argv=[];position=0
    while position<len(source):
        arg=source[position]
        if position<boundary and arg in ('--gpus','--cap-add'):
            position+=2;continue
        for old,new in [('{attempt_dir}',str(out)),('{lease_port}',str(19000+index*100)),
                        ('{lease_local_indices}',','.join(str(i) for i in range(job['payload']['gpu_count']))),
                        ('{lease_gpu_uuids}','cpu-preflight-no-gpu'),(job['job_id'],job['job_id']+'-cpu')]:
            arg=arg.replace(old,new)
        argv.append(arg);position+=1
    boundary=argv.index(token);argv[boundary:boundary]=[
        '--runtime','runc','--cpus','1','--memory','2g','-e','NVIDIA_VISIBLE_DEVICES=void']
    return argv+['--preflight-only']


def prepare(out,corpus_root,verification,image,base_source_manifest=None):
    out,corpus_root,verification=out.resolve(),corpus_root.resolve(),verification.resolve()
    out.mkdir(parents=True,exist_ok=False)
    members={}
    for member,(model,tp) in MODELS.items():
        size=member.removeprefix('distserve-')
        value=point_plan(corpus_root/('2026-09-22-'+size+'-v1'),model,tp)
        path=out/'plans'/member/'points.json';write_new(path,value)
        points=load_point_plan(path,model,tp)
        members[member]=dict(model_id=model,tp=tp,pp=1,gpu_count=2*tp,point_plan=ref(path,out),
            points=len(points),raw_windows=sum(p['repeats'] for p in points))
    spec=importlib.util.spec_from_file_location('dist_cohort_source_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer=importlib.util.module_from_spec(spec);spec.loader.exec_module(freezer)
    if base_source_manifest is None:
        source,source_sha=freezer.freeze_source(ROOT/'src',out/'sources')
    else:
        base_source_manifest=Path(base_source_manifest).resolve();base=json.loads(base_source_manifest.read_text())
        if digest(base['files'])!=base['source_sha256']:raise ValueError('base source manifest identity differs')
        overlays={'pdblend_baselines/distserve/'+name+'.py' for name in ('stage_collect','stage_cohort','stage_ports')}
        files=dict(base['files']);files.update({name:sha(ROOT/'src'/name) for name in overlays})
        source_sha=digest(files);source=out/'sources'/source_sha;source.mkdir(parents=True)
        for name,value in files.items():
            previous=ROOT/'src'/name if name in overlays else base_source_manifest.parent/name
            if sha(previous)!=value:raise ValueError('source input bytes changed: '+name)
            destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(previous,destination)
        write_new(source/'manifest.json',dict(schema=1,source_sha256=source_sha,files=files))
        write_new(out/'source-base-overlay.json',dict(base_manifest=dict(path=str(base_source_manifest),sha256=sha(base_source_manifest)),
            overlays={name:files[name] for name in sorted(overlays)},source_sha256=source_sha,
            all_other_source_files_unchanged=True))
    cohort='distserve-three-model-'+digest(dict(source=source_sha,members=members))[:20]
    wave=dict(cohort_id=cohort,coordinator=True,members=list(MODELS),startup_port_contract=CONTRACT,
              synchronize_parallel_windows=True,keep_peers_resident_until_all_done=True)
    write_new(out/'coord/wave.json',wave)
    binding=dict(schema='distserve-three-model-stage-cohort/v1',cohort_id=cohort,members=members,
        source_sha256=source_sha,image_digest=image,wave_spec=ref(out/'coord/wave.json',out),
        gpu_count=8,formal_eligible=False,parallel_qualified=False,startup_port_contract=CONTRACT)
    write_new(out/'cohort-inputs.json',binding);cohort_sha=sha(out/'cohort-inputs.json')
    jobs=[]
    for member,row in members.items():
        write_new(out/'plans'/member/'input-manifest.json',dict(source_sha256=source_sha,image_digest=image,
            cohort_sha256=cohort_sha,cohort_member=member,exact_inputs_sha256=dict(
                point_plan=row['point_plan']['sha256'],source_manifest=sha(source/'manifest.json'),
                model_verification=sha(verification))))
        jobs.append(job_for(member,row,out=out,source=source,source_sha=source_sha,cohort=cohort,
            cohort_sha=cohort_sha,image=image,verification=verification,corpus_root=corpus_root))
    write_new(out/'jobs.json',jobs)
    return jobs,binding,source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--corpus-root',type=Path,default=ROOT/'datasets/prepared')
    parser.add_argument('--verification',type=Path,default=VERIFY)
    parser.add_argument('--image',default=IMAGE)
    parser.add_argument('--base-source-manifest',type=Path,
        help='Keep every base file unchanged except the three private DistServe port-guard modules')
    args=parser.parse_args();out=args.out.resolve()
    jobs,cohort,source=prepare(out,args.corpus_root,args.verification,args.image,args.base_source_manifest)
    checks=[]
    for index,job in enumerate(jobs):
        member=job['payload']['cohort_member'];cpu_out=out/'cpu-preflight'/member;cpu_out.mkdir(parents=True)
        argv=cpu_command(job,cpu_out,index)
        write_new(cpu_out/'command.json',dict(argv=argv,hardware_executed=False))
        proc=subprocess.run(argv,capture_output=True,text=True,timeout=180)
        (cpu_out/'stdout').write_text(proc.stdout);(cpu_out/'stderr').write_text(proc.stderr)
        refs={}
        for name in ('preflight.json','cohort-preflight.json'):
            path=cpu_out/'dist-stage'/name
            if path.is_file():refs[name]=ref(path,out)
        checks.append(dict(member=member,returncode=proc.returncode,receipts=refs))
    passed=all(c['returncode']==0 and len(c['receipts'])==2 for c in checks)
    review=dict(schema='distserve-three-model-stage-preparation/v1',status='cpu_preflight_passed' if passed else 'preflight_failed',
        cohort_id=cohort['cohort_id'],source_manifest=dict(path=str(source/'manifest.json'),sha256=sha(source/'manifest.json')),
        jobs=ref(out/'jobs.json',out),cohort=ref(out/'cohort-inputs.json',out),cpu_preflights=checks,
        planned_gpu_partitions=[2,2,4],planned_total_gpu_count=8,
        raw_windows_by_model={member:row['raw_windows'] for member,row in cohort['members'].items()},
        qualification=dict(status='pending_hardware',parallel_qualified=False,engines=6,repeats=3,
            isolated='one model and one local engine at a time; all peers resident and idle',
            parallel='all six engines settled before every shared measurement',
            common_window_minimum_s=2.,timing_limit=.05,power_limit=.05,
            fallback='serialize cohort members AND both local roles; peers remain resident until all done'),
        capacity_policy='actual engine KV/token/batch limits; unsupported_engine excluded from model fitting',
        selection_splits=['calibration','tuning'],evaluation_used_for_selection=False,
        queue_modified=False,hardware_executed=False,formal_eligible=False,
        limitations=['CPU preflight does not grant concurrent GPU qualification.',
            'Every model needs independent holdout and measured-domain coverage; collection complete is not calibration passed.',
            'Short settled decode contexts or unsupported capacity may leave missing_profile.'])
    write_new(out/'review.json',review)
    print(json.dumps(dict(status=review['status'],out=str(out),jobs=len(jobs),hardware_executed=False)),flush=True)
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
