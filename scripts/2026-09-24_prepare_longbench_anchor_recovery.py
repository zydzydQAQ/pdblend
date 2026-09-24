#!/usr/bin/env python3
"""Prepare and CPU-preflight a LongBench-only recovery job; never enqueue."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from pdblend.bench.longbench_anchor_recovery import MODEL,SCHEMA,binding,bound,candidate_rates,prior_inputs
from pdblend.bench.resident_session import write_new,file_sha,digest

IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
VERIFY=ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'
PRIOR=ROOT/'results/2026-09-22/three-model/queue-attempts/mixed-rate-anchor-32b-e954f2cc36b8/attempt-0001-a33b59da895d45e0b86113a2a1d2979c/anchor'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prior',type=Path,default=PRIOR)
    parser.add_argument('--minimum-rate',type=float,default=.03125)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    refs=prior_inputs(args.prior)
    failed=bound(refs['failed_tuning']['completion'])
    rates=candidate_rates(failed['rate_rps'],args.minimum_rate)
    spec=importlib.util.spec_from_file_location('recovery_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer=importlib.util.module_from_spec(spec);spec.loader.exec_module(freezer)
    source,source_sha=freezer.freeze_source(ROOT/'src',out/'sources')
    corpus=ROOT/'datasets/prepared/2026-09-22-32b-v1'
    plan=dict(schema=SCHEMA,model_id=MODEL,prior=refs,minimum_rate_rps=args.minimum_rate,
        candidate_rates_rps=rates,slo=dict(ttft_s=15.,tpot_s=.2),
        selection_splits=['calibration','tuning'],evaluation_used_for_selection=False,
        calibration_seed=9701,tuning_seed=9702,window_s=60,confirm_window_s=120,
        source_manifest=binding(source/'manifest.json'),image_digest=IMAGE,
        policy='halve prior failed tuning rate; calibration before each independent tuning; stop at first confirmed rate or floor',
        limitations=['Measured passing lower bound, not exact capacity.',
                     'The original two passing dataset anchors retain their original source and confirmation bytes.',
                     'No evaluation data selects candidate rates; every failed candidate is retained.'])
    path=out/'plan.json';write_new(path,plan)
    job_id='mixed-longbench-recovery-32b-'+digest(plan)[:16]
    argv=['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN',
        '--ipc=host','--network=host','--shm-size=16g','--ulimit','nofile=65536:65536',
        '--entrypoint','/opt/venv/bin/python']
    for host,target,mode in [(ROOT,ROOT,'ro'),(source,'/opt/pdblend-src','ro'),
            ('/home/models','/models','ro'),('{attempt_dir}','/output','rw'),
            ('/tmp/pdblend-physical-clock-owners','/tmp/pdblend-physical-clock-owners','rw')]:
        argv+=['-v',f'{host}:{target}:{mode}']
    env=dict(PYTHONPATH='/opt/pdblend-src',PYTHONDONTWRITEBYTECODE='1',PDBLEND_MODELS_DIR='/models',
        PDBLEND_MODEL_VERIFICATION_RECEIPT=str(VERIFY),PDBLEND_SOURCE_MANIFEST=str(source/'manifest.json'),
        PDBLEND_SOURCE_SHA256=source_sha,PDBLEND_IMAGE_ID=IMAGE,PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',
        CUDA_VISIBLE_DEVICES='{lease_local_indices}',CUDA_VERSION='12.8.1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',PDBLEND_CLOCK_LOCK_DIR='/tmp/pdblend-physical-clock-owners')
    for key,value in env.items():argv+=['-e',key+'='+value]
    argv+=[IMAGE,'-B','-m','pdblend.bench.longbench_anchor_recovery','--plan',str(path),
        '--plan-sha256',file_sha(path),'--model','/models/'+MODEL,'--tp','2',
        '--gpus','{lease_local_indices}','--corpus',str(corpus),'--out','/output/anchor','--base-port','{lease_port}']
    job=dict(job_id=job_id,priority=798,max_attempts=1,payload=dict(argv=argv,container_name=job_id,
        model_id=MODEL,system='mixed',tp=2,pp=1,gpu_count=8,exclusive=True,global_lock=True,reserve_host=True,
        source_snapshot=str(source),source_sha256=source_sha,image_digest=IMAGE,
        recovery_plan=binding(path),prior_completion=refs['receipts']['completion'],
        depends_on=[],after_terminal=[],timeout_s=7200,required_receipts=['anchor/completion.json'],
        scope='longbench_only_independent_calibration_tuning_recovery',formal_eligible=False,energy_comparable=False))
    write_new(out/'jobs.json',[job])
    cpu=out/'cpu-preflight';cpu.mkdir()
    check=[];i=0
    while i<len(argv):
        value=argv[i]
        if i<argv.index(IMAGE) and value in ('--gpus','--cap-add'):
            i+=2;continue
        value=value.replace(job_id,job_id+'-cpu').replace('{attempt_dir}',str(cpu))
        value=value.replace('{lease_gpu_uuids}','').replace('{lease_local_indices}','0,1,2,3,4,5,6,7')
        value=value.replace('{lease_port}','19200');check.append(value);i+=1
    index=check.index(IMAGE);check[index:index]=['-e','NVIDIA_VISIBLE_DEVICES=void']
    check+=['--preflight-only'];write_new(out/'preflight-command.json',dict(argv=check,hardware_executed=False))
    proc=subprocess.run(check,capture_output=True,text=True,timeout=180)
    (out/'preflight.stdout').write_text(proc.stdout);(out/'preflight.stderr').write_text(proc.stderr)
    receipt=cpu/'anchor/preflight.json'
    review=dict(status='cpu_preflight_passed' if proc.returncode==0 else 'preflight_failed',
        returncode=proc.returncode,jobs=binding(out/'jobs.json'),plan=binding(path),
        source_sha256=source_sha,preflight=binding(receipt) if receipt.is_file() else None,
        queue_modified=False,hardware_executed=False,inherited_datasets=['alpaca','sharegpt'],
        new_dataset='longbench',candidate_rates_rps=rates,
        maximum_service_s=len(rates)*180,job_id=job_id,implementation=binding(Path(__file__)))
    write_new(out/'review.json',review)
    print(json.dumps(review))
    return 0 if proc.returncode==0 else 2


if __name__=='__main__':raise SystemExit(main())
