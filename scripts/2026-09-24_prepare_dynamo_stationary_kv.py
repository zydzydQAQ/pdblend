#!/usr/bin/env python3
"""Freeze an unexecuted one-load source-KV probe; never enqueue or touch GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.dont_write_bytecode = True
from pdblend_baselines.dynamollm.stationary_ipc import digest
from pdblend_baselines.dynamollm.stationary_kv_probe import PROBE_PLAN, write_new

IMAGE = 'sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
OVERLAYS = ['pdblend_baselines/dynamollm/'+name+'.py' for name in
    ('gpu_weights','stationary_tensors','stationary_ipc','stationary_worker','stationary_probe',
     'stationary_admission','stationary_kv','stationary_kv_worker','stationary_service','stationary_kv_probe')]


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def prepare(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    base_path = args.base_source_manifest.resolve(); base = json.loads(base_path.read_text())
    files = dict(base['files'])
    files.update({name: hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in OVERLAYS})
    source_sha = digest(files); source = out/'sources'/source_sha; source.mkdir(parents=True)
    for name, sha in files.items():
        old = ROOT/'src'/name if name in OVERLAYS else base_path.parent/name
        need_sha = hashlib.sha256(old.read_bytes()).hexdigest()
        if need_sha != sha: raise ValueError('source input bytes changed: '+name)
        new = source/name; new.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(old,new)
    write_new(source/'manifest.json',dict(schema=1,source_sha256=source_sha,files=files))
    config = dict(schema='dynamo-source-kv-owned-probe-config/v1', source_snapshot=str(source), source_sha256=source_sha,
        image_digest=IMAGE, plan=PROBE_PLAN, model_id='Qwen2.5-7B-Instruct', model_path='/models/Qwen2.5-7B-Instruct',
        model_verification=ref(args.model_verification), tp=1,pp=1,gpu_count=2,max_num_seqs=32,
        required_prior_qualification_job=args.ipc_primitive_job,
        same_resident_load_for_all_stages=True, no_extra_target_model=True, formal_eligible=False)
    config_path=out/'config.json'; write_new(config_path,config); config_sha=ref(config_path)['sha256']
    job_id='dynamo-stationary-kv-'+config_sha[:16]
    env=dict(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PDBLEND_SOURCE_SHA256=source_sha,PDBLEND_SOURCE_MANIFEST=str(source/'manifest.json'),PDBLEND_IMAGE_ID=IMAGE,
        PDBLEND_KV_PROBE_CONFIG_SHA256=config_sha,PDBLEND_MODEL_VERIFICATION_RECEIPT=str(args.model_verification.resolve()),
        PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',CUDA_VISIBLE_DEVICES='0,1',DYNAMO_GENERATION='0',
        VLLM_USE_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',TOKENIZERS_PARALLELISM='false',
        NCCL_CUMEM_ENABLE='0',NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='0',
        PDBLEND_CONCURRENCY_ENVIRONMENT='{attempt_dir}/concurrency-environment.json')
    argv=['docker','run','--rm','--name',job_id,'--gpus','"device={lease_gpu_uuids}"','--ipc=host','--network=host',
        '--shm-size=16g','--entrypoint','/opt/venv/bin/python','-v',str(ROOT)+':'+str(ROOT)+':ro',
        '-v','/home/models:/models:ro','-v','{attempt_dir}:{attempt_dir}:rw']
    for key,value in env.items():argv += ['-e',key+'='+value]
    argv += [IMAGE,'-B','-m','pdblend_baselines.dynamollm.stationary_kv_probe','--config',str(config_path),
        '--base-port','{lease_port}','--out','{attempt_dir}/source-kv-probe']
    job=dict(job_id=job_id,priority=args.priority,max_attempts=1,payload=dict(argv=argv,cwd=str(ROOT),
        container_name=job_id,system='dynamollm',model_id=config['model_id'],gpu_count=2,tp=1,pp=1,
        exclusive=False,reserve_host=False,timeout_s=1800,depends_on=[args.ipc_primitive_job],after_terminal=[],
        source_snapshot=str(source),source_sha256=source_sha,image_digest=IMAGE,config_path=str(config_path),
        config_sha256=config_sha,prepare_only=True,model_loads=2,formal_eligible=False,energy_comparable=False,
        kind='source_KV_lifetime_only_not_TP_conversion',required_receipts=['source-kv-probe/completion.json',
            'source-kv-probe/gpu-after.json','source-kv-probe/probe/completion.json'],
        concurrency_requirement='Do not insert into a running profile sampling cohort; root alone schedules.'))
    write_new(out/'jobs.json',[job])
    run_env=dict(os.environ,**env)
    command=[sys.executable,'-B','-m','pdblend_baselines.dynamollm.stationary_kv_probe',
             '--config',str(config_path),'--preflight']
    checked=subprocess.run(command,env=run_env,text=True,capture_output=True,timeout=30)
    write_new(out/'cpu-preflight.json',dict(command=command,returncode=checked.returncode,
        stdout=checked.stdout,stderr=checked.stderr,hardware_executed=False))
    if checked.returncode: raise RuntimeError(checked.stderr or checked.stdout)
    write_new(out/'manifest.json',dict(schema='dynamo-source-kv-prepared/v1',base_source_manifest=ref(base_path),
        source_manifest=ref(source/'manifest.json'),config=ref(config_path),jobs=ref(out/'jobs.json'),
        overlays={name:ref(ROOT/'src'/name) for name in OVERLAYS},builder=ref(Path(__file__)),
        host_cpu_preflight_passed=True,hardware_executed=False,enqueued=False,formal_eligible=False,
        alternatives='Existing future private Fleet may call probe_on_resident with its actual bound source and peer capabilities; that avoids both model loads.',
        remaining=['source KV GPU proof','consumer IPC and target fragmented binding integration','target TP isolation',
                   'missing fragment direct GPU transport','physical peak memory qualification','1800/300/5s original mechanism']))
    print(json.dumps(dict(job_id=job_id,jobs=str(out/'jobs.json'),source_sha256=source_sha,enqueued=False),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--base-source-manifest',type=Path,required=True)
    parser.add_argument('--model-verification',type=Path,required=True)
    parser.add_argument('--ipc-primitive-job',required=True,
        help='Exact immutable 1-GPU IPC primitive job whose success is required before this probe')
    parser.add_argument('--priority',type=int,default=149)
    prepare(parser.parse_args())


if __name__=='__main__':main()
