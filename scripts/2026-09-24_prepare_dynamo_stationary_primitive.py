#!/usr/bin/env python3
"""Freeze one short, model-free CUDA IPC primitive job; never enqueue it."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from pdblend_baselines.dynamollm.stationary_ipc import digest
from pdblend_baselines.dynamollm.stationary_probe import PROBE_PLAN,write_new

IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
SOURCE_FILES=['pdblend_baselines/__init__.py','pdblend_baselines/dynamollm/__init__.py',
              *['pdblend_baselines/dynamollm/'+name+'.py' for name in
                ('gpu_weights','stationary_tensors','stationary_ipc','stationary_admission','stationary_probe')]]


def prepare(out,priority=151):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    hashes={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    source_sha=digest(hashes);source=out/'sources'/source_sha;source.mkdir(parents=True)
    for name in SOURCE_FILES:
        destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'src'/name,destination)
    write_new(source/'manifest.json',dict(schema=1,source_sha256=source_sha,files=hashes))
    config=dict(schema='dynamo-stationary-primitive-job-config/v1',plan=PROBE_PLAN,
        source_snapshot=str(source),source_sha256=source_sha,image_id=IMAGE,
        qualification_scope='same-GPU CUDA IPC original synthetic shards and consumer crash isolation only',
        actual_model_loaded=False,full_tp_switch_qualified=False,formal_eligible=False)
    config_path=out/'config.json';write_new(config_path,config)
    config_sha=hashlib.sha256(config_path.read_bytes()).hexdigest();job_id='dynamo-stationary-ipc-'+config_sha[:16]
    env=dict(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PDBLEND_IMAGE_ID=IMAGE,PDBLEND_PRIMITIVE_CONFIG_SHA256=config_sha,
        PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',CUDA_VISIBLE_DEVICES='{lease_local_indices}',
        PDBLEND_CONCURRENCY_ENVIRONMENT='{attempt_dir}/concurrency-environment.json')
    argv=['docker','run','--rm','--name',job_id,'--gpus','all','--ipc=host',
          '--entrypoint','/opt/venv/bin/python','-v',str(ROOT)+':'+str(ROOT)+':ro',
          '-v','{attempt_dir}:{attempt_dir}:rw']
    for key,value in env.items():argv+=['-e',key+'='+value]
    argv+=[IMAGE,'-B','-m','pdblend_baselines.dynamollm.stationary_probe',
           '--config',str(config_path),'--out','{attempt_dir}/stationary-primitive']
    job=dict(job_id=job_id,priority=priority,max_attempts=1,payload=dict(argv=argv,cwd=str(ROOT),
        container_name=job_id,system='dynamollm',model_id='synthetic-no-model',gpu_count=1,tp=1,pp=1,
        exclusive=False,reserve_host=False,depends_on=[],after_terminal=[],timeout_s=180,
        source_snapshot=str(source),source_sha256=source_sha,image_digest=IMAGE,
        config_path=str(config_path),config_sha256=config_sha,prepare_only=True,
        kind='same_gpu_stationary_cuda_ipc_primitive',formal_eligible=False,energy_comparable=False,
        model_loads=0,required_receipts=['stationary-primitive/completion.json','stationary-primitive/gpu-after.json'],
        concurrency_requirement='Only on its leased empty GPU; do not join an existing profile sampling cohort',
        invalidates_no_previous_results=True))
    jobs=out/'jobs.json';write_new(jobs,[job])
    run_env=dict(os.environ,**env);run_env['PYTHONPATH']=str(source)
    check=subprocess.run([sys.executable,'-B','-m','pdblend_baselines.dynamollm.stationary_probe',
                          '--config',str(config_path),'--preflight'],env=run_env,text=True,capture_output=True,timeout=30)
    write_new(out/'cpu-preflight.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr,
        gpu_executed=False,source_sha256=source_sha,config_sha256=config_sha,
        prepare_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if check.returncode:raise RuntimeError(check.stderr or check.stdout)
    return dict(jobs_path=str(jobs),job_id=job_id,source_sha256=source_sha,config_sha256=config_sha,
                host_cpu_preflight_passed=True,gpu_executed=False,enqueued=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--priority',type=int,default=151)
    args=parser.parse_args();print(json.dumps(prepare(args.out,args.priority),indent=2,sort_keys=True))


if __name__=='__main__':main()
