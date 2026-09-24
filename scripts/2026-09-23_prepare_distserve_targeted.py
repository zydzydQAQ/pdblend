#!/usr/bin/env python3
"""Prepare a bounded 7B DistServe sampling job; never enqueue or use a GPU."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from pdblend_baselines.distserve.stage_collect import load_point_plan
from pdblend_baselines.native_profile import DIST_FREQS

IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
VERIFY=ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'
MODEL='Qwen2.5-7B-Instruct'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def point_plan(corpus):
    bindings=[];lengths=[];ends=[]
    for dataset in ('alpaca','sharegpt','longbench'):
        path=corpus/(dataset+'.json');data=json.loads(path.read_text())
        if data.get('model_name')!=MODEL:raise ValueError('model-specific corpus required')
        records=data.get('calibration',[])+data.get('tuning',[])
        if not data.get('calibration') or not data.get('tuning'):raise ValueError('independent calibration/tuning splits required')
        lengths.extend(r['input_tokens'] for r in records)
        ends.extend(r['input_tokens']+r['output_tokens'] for r in records)
        bindings.append(dict(dataset=dataset,path=str(path),sha256=sha(path),splits=['calibration','tuning']))
    lo=min(lengths);hi=max(lengths);mid=sorted(lengths)[len(lengths)//2]
    if not 1<=lo<=mid<hi<=7168:raise ValueError('targeted 7B shape envelope differs')
    # Complete-prefill token budget is 8192. A four-request high homogeneous
    # batch therefore uses 2048, independently of the decode shape bound.
    pre=[(lo,),(hi,),(lo,lo),(lo,hi),(lo,)*4,(hi,lo,lo,lo),(2048,)*4]
    # Decoder lengths below are starting prompts, not claimed measured context.
    # All actual scheduler context vectors, including drift during settle,
    # determine the final measured hull and the subsequent missing-profile gate.
    upper=min(7808,max(ends)+128)
    dec=[(lo,),(upper,),(lo,lo),(lo,upper),(lo,)*4,(upper,lo,lo,lo),(upper,)*4]
    hold_pre=[(mid,),(lo,hi),(2048,)*4]
    hold_dec=[(max(mid,lo+128),),(lo+128,upper-256),(upper-256,)*4]
    if any(sum(s)>8192 for s in pre+hold_pre):raise ValueError('targeted prefill candidate exceeds engine budget')
    points=[dict(frequency_mhz=f,role=role,purpose=purpose,lengths=list(shape),repeats=3)
            for f in DIST_FREQS for role,train,hold in [('prefill',pre,hold_pre),('decode',dec,hold_dec)]
            for purpose,shapes in [('training',train),('holdout',hold)] for shape in shapes]
    return dict(schema='distserve-targeted-stage-plan-v1',system='distserve',model_id=MODEL,tp=1,pp=1,
        selection_split='calibration',selection_splits=['calibration','tuning'],evaluation_used_for_selection=False,
        inputs=bindings,points=points,shape_selection='bounded representative vertices from independent calibration/tuning lengths',
        batch_candidates=[1,2,4],frequency_candidates=list(DIST_FREQS),formal_eligible=False,
        limitations=['Only actually measured per-request feature hull qualifies; no maximum-shape shortcut.',
                     'Short decode contexts may remain missing after the required settled window.',
                     'Batches above four, other TP, PP and unmeasured shapes remain missing_profile.',
                     'Independent holdout may fail; successful collection alone does not qualify calibration.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'results/2026-09-23/distserve-targeted-7b-v1')
    p.add_argument('--corpus',type=Path,default=ROOT/'datasets/prepared/2026-09-22-7b-v1')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    value=point_plan(args.corpus.resolve());write(out/'points.json',value)
    points=load_point_plan(out/'points.json',MODEL,1)
    spec=importlib.util.spec_from_file_location('dist_source_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    source,source_sha=helper.freeze_source(ROOT/'src',out/'sources')
    helper.verify_snapshot(source,json.loads((source/'manifest.json').read_text())['files'])
    binding=dict(source_sha256=source_sha,image_digest=IMAGE,exact_inputs_sha256=dict(
        point_plan=sha(out/'points.json'),source_manifest=sha(source/'manifest.json'),model_verification=sha(VERIFY)))
    write(out/'input-manifest.json',binding)
    coord=out/'coord';coord.mkdir();cohort='distserve-targeted-7b-'+source_sha[:16]
    write(coord/'wave.json',dict(cohort_id=cohort,coordinator=True,members=['distserve-7b'],
        synchronize_parallel_windows=True,keep_peers_resident_until_all_done=True))
    job_id='distserve-targeted-7b-'+source_sha[:16]
    token='pdblend:l20-cu128-vllm-v1@'+IMAGE
    argv=['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN','--ipc=host',
        '--network=host','--shm-size=16g','--ulimit','nofile=65536:65536','--entrypoint','/opt/venv/bin/python']
    for host,target,mode in [(source,'/opt/pdblend-src','ro'),(source/'manifest.json','/source-manifest.json','ro'),
        (Path('/home/models'),'/models','ro'),(VERIFY,'/verification/model-verification.json','ro'),
        (out/'points.json','/spec/points.json','ro'),(out/'input-manifest.json','/spec/input-manifest.json','ro'),
        (coord,'/coord','rw'),('{attempt_dir}','/output','rw')]:argv+=['-v',f'{host}:{target}:{mode}']
    env=dict(PYTHONPATH='/opt/pdblend-src',PYTHONDONTWRITEBYTECODE='1',PDBLEND_MODELS_DIR='/models',
        PDBLEND_MODEL_VERIFICATION_RECEIPT='/verification/model-verification.json',PDBLEND_SOURCE_SHA256=source_sha,
        PDBLEND_SOURCE_MANIFEST='/source-manifest.json',PDBLEND_IMAGE_ID=IMAGE,PDBLEND_VLLM_VERSION='0.10.1.1',
        CUDA_VERSION='12.8.1',PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',CUDA_VISIBLE_DEVICES='{lease_local_indices}',
        PDBLEND_RESIDENT_LOAD_LOCK='/coord/model-load.lock',PDBLEND_PROFILE_WAVE='/coord',
        PDBLEND_PROFILE_MEMBER='distserve-7b',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    for k,v in env.items():argv+=['-e',f'{k}={v}']
    argv+=['-e','PDBLEND_CONCURRENCY_ENVIRONMENT','-e','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256',token,
        '-B','-m','pdblend_baselines.distserve.stage_collect','--model',MODEL,'--tp','1',
        '--gpus','{lease_local_indices}','--base-port','{lease_port}','--point-plan','/spec/points.json',
        '--input-manifest','/spec/input-manifest.json','--out','/output/dist-stage']
    payload=dict(schema='distserve-targeted-stage-docker-v1',system='distserve',model_id=MODEL,tp=1,pp=1,
        gpu_count=2,exclusive=False,global_lock=False,reserve_host=False,sampling_cohort=cohort,
        source_snapshot=str(source),container_name=job_id,argv=argv,**binding,
        depends_on=[],after_terminal=[],formal_eligible=False,energy_comparable=False,
        required_receipts=['dist-stage/completion.json'],timeout_s=7200,
        scope='bounded B1/2/4 TP1 native stage timing, group power and independent holdout',
        sampling_complete_is_calibration_pass=False)
    job=dict(job_id=job_id,priority=390,max_attempts=1,payload=payload)
    write(out/'jobs.json',[job])
    cpu_out=out/'cpu-preflight';cpu_out.mkdir();cpu=[];i=0
    while i<len(argv):
        v=argv[i]
        if v=='--cap-add' or v=='--gpus' and i<argv.index(token):i+=2;continue
        v=v.replace(job_id,job_id+'-cpu').replace('{attempt_dir}',str(cpu_out)).replace('{lease_local_indices}','0,1')
        v=v.replace('{lease_port}','19400').replace('{lease_gpu_uuids}','cpu-preflight-no-gpu')
        cpu.append(v);i+=1
    index=cpu.index(token);cpu[index:index]=['-e','NVIDIA_VISIBLE_DEVICES=void'];cpu+=['--preflight-only']
    write(out/'preflight-command.json',dict(argv=cpu,hardware_executed=False))
    proc=subprocess.run(cpu,capture_output=True,text=True,timeout=180)
    (out/'preflight.stdout').write_text(proc.stdout);(out/'preflight.stderr').write_text(proc.stderr)
    receipt=cpu_out/'dist-stage/preflight.json'
    audit=json.loads((ROOT/'results/2026-09-23/distserve-native-search-audit/7b.json').read_text())
    review=dict(schema='distserve-targeted-preparation-review-v1',status='cpu_preflight_passed' if proc.returncode==0 else 'preflight_failed',
        job_id=job_id,job_sha256=sha(out/'jobs.json'),source_sha256=source_sha,image_digest=IMAGE,
        cpu_preflight=dict(returncode=proc.returncode,receipt=str(receipt),sha256=sha(receipt) if receipt.exists() else None),
        model_id=MODEL,tp=1,pp=1,gpu_count=2,raw_windows=sum(x['repeats'] for x in points),
        windows_per_role=sum(x['repeats'] for x in points if x['role']=='decode'),
        estimated_seconds=dict(pure_parallel_settle_and_measure=1260,pure_serial_fallback=2520,
            qualification_minimum=63,loading_and_cleanup_estimate=[90,240],
            parallel_total_estimate=[1500,2100],serial_total_estimate=[2800,3600]),
        queue_modified=False,hardware_executed=False,formal_eligible=False,
        reuse=dict(native_profile=audit['profiles'],native_capacity=audit['native_capacity_receipts'],
            old_windows=126,status='retained_mechanism_evidence',reason='old windows lack independent holdout, power, repetitions and per-request heterogeneous vectors'),
        required_gates=['owned two-GPU UUID lease; no unqualified peer job may run during sampling',
            'all-rank request/shape alignment and native CUDA timing',
            'three repeats, >=2s settle, >=5s measurement, >=8 decode steps, >=2 power and >=1 clock samples',
            'actual isolated/concurrent group timing and power differences <=5%; serialize local roles if failed',
            'independent holdout max <=10%; timing <=10%; power MAPE <=10%, max <=15%',
            'trace-specific coverage and deployment receipts required separately'],
        scheduling_note='Single two-GPU cohort. To use the other six GPUs, prepare and bind additional qualification participants before freezing a new cohort; arbitrary functional peers are forbidden.',
        limitations=value['limitations'])
    write(out/'review.json',review)
    print(json.dumps(dict(status=review['status'],job_id=job_id,raw_windows=review['raw_windows'],returncode=proc.returncode)),flush=True)
    return 0 if proc.returncode==0 else 2


if __name__=='__main__':raise SystemExit(main())
