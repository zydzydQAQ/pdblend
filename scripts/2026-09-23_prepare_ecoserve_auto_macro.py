#!/usr/bin/env python3
"""Freeze three native automatic EcoServe jobs and run CPU CLI preflights only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
VERIFY=ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/2026-09-23/ecoserve-auto-macro-v2')
    parser.add_argument('--revision',type=int,default=2)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.revision<1:raise ValueError('positive immutable job revision required')
    old=json.loads((ROOT/'results/2026-09-23/dist-eco-resident-v1/jobs.json').read_text())
    spec=importlib.util.spec_from_file_location('eco_source_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    source,source_sha=helper.freeze_source(ROOT/'src',out/'sources')
    helper.verify_snapshot(source,json.loads((source/'manifest.json').read_text())['files'])
    coord=out/'coord';coord.mkdir()
    jobs,reviews=[],[]
    for previous in old:
        p=previous['payload'];model=p['model_id'];tp=p['tp'];key=model.split('-')[1].lower()
        a=p['argv'];profile=Path(a[a.index('--eco-profile')+1]);item=out/key;item.mkdir()
        counts=(18,12,12) if tp==2 else (48,24,24)
        phases=[dict(at_s=t,requests=n,input_tokens=7168,output_tokens=512,phase='pressure') for t,n in zip((0,20,40),counts)]
        phases += [dict(at_s=t,requests=1,input_tokens=128,output_tokens=512,phase='recovery') for t in range(110,286,2)]
        rng=random.Random(701);rows=[]
        for phase in phases:
            for _ in range(phase['requests']):
                rows.append(dict(idx=len(rows),arrival_s=phase['at_s'],
                                 prompt=[rng.randrange(100,5000) for _ in range(phase['input_tokens'])],
                                 max_tokens=phase['output_tokens'],source='ecoserve-auto-'+phase['phase']+'-seed701'))
        write(item/'trace.json',dict(seed=701,service_duration_s=300,requests=rows,phases=phases))
        checksums=dict(trace=sha(item/'trace.json'),csv=sha(profile),csv_manifest=sha(str(profile)+'.manifest.json'),
                       source_manifest=sha(source/'manifest.json'),model_verification=sha(VERIFY))
        binding=dict(source_sha256=source_sha,image_digest=IMAGE,exact_inputs_sha256=checksums)
        write(item/'input-manifest.json',binding)
        job_id=f'ecoserve-auto-macro-{key}-original-period-qualification-v{args.revision}';count=4*tp
        argv=['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN','--ipc=host',
              '--network=host','--shm-size=16g','--ulimit','nofile=65536:65536','--entrypoint','/opt/venv/bin/python']
        mounts=[(source,'/opt/pdblend-src','ro'),(source/'manifest.json','/source-manifest.json','ro'),
                (Path('/home/models'),'/models','ro'),(VERIFY,'/verification/model-verification.json','ro'),
                (profile.parent,str(profile.parent),'ro'),(item/'trace.json','/spec/trace.json','ro'),
                (item/'input-manifest.json','/spec/input-manifest.json','ro'),(coord,'/coord','rw'),
                ('{attempt_dir}','/output','rw')]
        for host,target,mode in mounts:argv+=['-v',f'{host}:{target}:{mode}']
        environment=dict(PYTHONPATH='/opt/pdblend-src',PYTHONDONTWRITEBYTECODE='1',PDBLEND_MODELS_DIR='/models',
                         PDBLEND_MODEL_VERIFICATION_RECEIPT='/verification/model-verification.json',
                         PDBLEND_SOURCE_SHA256=source_sha,PDBLEND_SOURCE_MANIFEST='/source-manifest.json',
                         PDBLEND_IMAGE_ID=IMAGE,PDBLEND_VLLM_VERSION='0.10.1.1',CUDA_VERSION='12.8.1',
                         PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',CUDA_VISIBLE_DEVICES='{lease_local_indices}',
                         PDBLEND_LEASE_PORT='{lease_port}',PDBLEND_RESIDENT_LOAD_LOCK='/coord/model-load.lock')
        for name,value in environment.items():argv+=['-e',f'{name}={value}']
        argv+=['-e','PDBLEND_CONCURRENCY_ENVIRONMENT','-e','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256']
        token='pdblend:l20-cu128-vllm-v1@'+IMAGE
        argv += [token,'-B','-m','pdblend_baselines.ecoserve.auto_macro','--model',model,'--tp',str(tp),
                 '--gpus','{lease_local_indices}','--base-port','{lease_port}','--eco-profile',str(profile),
                 '--trace','/spec/trace.json','--input-manifest','/spec/input-manifest.json',
                 '--duration','300','--out','/output/eco-auto']
        payload=dict(schema='ecoserve-automatic-macro-docker-v1',system='ecoserve',model_id=model,tp=tp,pp=1,
                     gpu_count=count,source_snapshot=str(source),container_name=job_id,argv=argv,**binding,
                     depends_on=[],after_terminal=[],exclusive=False,global_lock=False,formal_eligible=False,
                     energy_comparable=False,completion_is_formal_qualification=False,
                     required_receipts=['eco-auto/completion.json'],timeout_s=3600,
                     service_duration_s=300,automatic_policy_required=True,manual_primitive_allowed=False,
                     control=dict(initial_active=3,initial_parked=1,period_s=5,history_window_s=60,
                                  macro_lower=2,macro_upper=3,slo_ttft_s=5,slo_tpot_s=.15),
                     trace=dict(path=str(item/'trace.json'),sha256=checksums['trace'],requests=len(rows),seed=701),
                     independent_author_profile=dict(path=str(profile),sha256=checksums['csv'],
                                                     manifest_sha256=checksums['csv_manifest']))
        job=dict(job_id=job_id,priority=430,max_attempts=1,payload=payload);jobs.append(job)
        cpu_out=item/'cpu-preflight';cpu_out.mkdir();cpu=[];i=0
        while i<len(argv):
            value=argv[i]
            if value=='--cap-add' or value=='--gpus' and i<argv.index(token):i+=2;continue
            value=value.replace(job_id,job_id+'-cpu-preflight').replace('{attempt_dir}',str(cpu_out))
            value=value.replace('{lease_local_indices}',','.join(map(str,range(count)))).replace('{lease_port}','19400')
            value=value.replace('{lease_gpu_uuids}','cpu-preflight-no-gpu');cpu.append(value);i+=1
        index=cpu.index(token);cpu[index:index]=['-e','NVIDIA_VISIBLE_DEVICES=void'];cpu+=['--preflight-only']
        write(item/'preflight-command.json',dict(argv=cpu,hardware_executed=False))
        proc=subprocess.run(cpu,capture_output=True,text=True,timeout=120)
        (item/'preflight.stdout').write_text(proc.stdout);(item/'preflight.stderr').write_text(proc.stderr)
        receipt_path=cpu_out/'eco-auto/preflight.json'
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        review=dict(model_id=model,gpu_count=count,job_id=job_id,**binding,
                    status='prepare_only_cpu_container_passed' if proc.returncode==0 else 'preflight_failed',
                    cpu_container_preflight=dict(returncode=proc.returncode,receipt=str(receipt_path),
                                                 receipt_sha256=sha(receipt_path) if receipt else None),
                    cpu_candidates=[dict(decode_step_assumption_s=r['decode_step_assumption_s'],
                                         split=r['split_candidate'],merge=r['merge_candidate'],candidates=r['candidates'])
                                    for r in receipt.get('cpu_replays',[])],
                    queue_modified=False,hardware_executed=False)
        write(item/'review.json',review);reviews.append(review)
        print(json.dumps(dict(model=model,status=review['status'],returncode=proc.returncode)),flush=True)
    write(out/'jobs.json',jobs)
    write(out/'review.json',dict(schema='ecoserve-automatic-macro-three-model-review-v1',
                                status='prepare_only_cpu_container_passed' if all(r['status']=='prepare_only_cpu_container_passed' for r in reviews) else 'preflight_failed',
                                jobs_sha256=sha(out/'jobs.json'),source_sha256=source_sha,source_snapshot=str(source),
                                image_digest=IMAGE,models=reviews,queue_modified=False,hardware_executed=False,
                                formal_eligible=False,energy_comparable=False,
                                required_gpu_gates=['automatic split with mean_ttft trigger','automatic merge with saved_tpot trigger',
                                                    'matching native clock/park ACK','actual held output then client flush',
                                                    'live KV block-prefix/generation/rank ACK continuity for split and merge',
                                                    'all outputs exactly equal native token stream; complete final drains'],
                                limitations=['CPU models predict candidates, not observed GPU actions. Decoder latency and capacity '
                                             'assumptions are explicit in each preflight receipt.','Missing native actions or continuity yield inconclusive, regardless of elapsed duration.']))
    return 0 if all(r['status']=='prepare_only_cpu_container_passed' for r in reviews) else 2


if __name__=='__main__':raise SystemExit(main())
