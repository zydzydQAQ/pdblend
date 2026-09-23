#!/usr/bin/env python3
"""Queue model-owned predictor training and native GPU correctness probes.

Both follow the current calibration cohort. Higher priority places functional
work before queued TP4 holdouts without cancelling or invalidating measurements.
These are mechanism/development artifacts, never an energy ranking.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pdblend.experimentation.lease import GPULeaseQueue


def build_jobs(kind, *, root, source, source_hash, image, verification, depends):
    jobs=[]
    for size in ('7B','14B','32B'):
        model=f'Qwen2.5-{size}-Instruct'
        tp=2 if size=='32B' else 1
        count=1 if kind=='predictor' else tp*{'dynamo':3,'probe':2,'eco4':4}[kind]
        identity=dict(kind=kind,model_id=model,source_sha256=source_hash,image_digest=image,seed=701)
        suffix=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
        name=f'native-{kind}-{size.lower()}-{suffix}'
        argv=['docker','run','--rm','--name',name,'--gpus','all','--cap-add','SYS_ADMIN',
              '--ipc=host','--network','host','--shm-size','16g','--ulimit','nofile=65536:65536',
              '--entrypoint','/opt/venv/bin/python',
              '-v',f'{source}:/opt/pdblend-src:ro','-v','/home/models:/models:ro',
              '-v',f'{verification}:/verification/model-verification.json:ro',
              '-v','{attempt_dir}:/output:rw','-e','PYTHONPATH=/opt/pdblend-src',
              '-e','PDBLEND_MODELS_DIR=/models','-e','PDBLEND_GPU_UUIDS={lease_gpu_uuids}',
              '-e','PDBLEND_MODEL_VERIFICATION_RECEIPT=/verification/model-verification.json',
              '-e',f'PDBLEND_SOURCE_SHA256={source_hash}','-e',f'PDBLEND_IMAGE_ID={image}',
              '-e','PDBLEND_HARDWARE_ID=8xL20-lease','-e','CUDA_VERSION=12.8.1',
              '-e','TOKENIZERS_PARALLELISM=false','-e','OMP_NUM_THREADS=4']
        if kind=='predictor':
            corpus=ROOT/f'datasets/prepared/2026-09-22-{size.lower()}-v1'
            argv+=['-v',f'{corpus}:/corpus:ro',image,'-B','-m','pdblend_baselines.dynamollm.train_v1',
                   '--model',f'/models/{model}','--corpus-root','/corpus','--encoder','/models/bert-base-uncased',
                   '--out','/output/predictor','--report','/output/training.json','--device','cuda']
        elif kind in ('probe','eco4'):
            argv +=[image,'-B','-m','pdblend_runtime.'+('probe' if kind=='probe' else 'ecoserve_probe'),'--model',f'/models/{model}',
                    '--tp',str(tp),'--gpus',','.join(map(str,range(count))),
                    '--base-port','{lease_port}','--out','/output']
        else:
            argv +=[image,'-B','-m','pdblend_baselines.dynamollm.transition_probe_v1',
                    '--model',f'/models/{model}','--gpus',','.join(map(str,range(count))),
                    '--base-port','{lease_port}','--out','/output/dynamo','--profile-target']
        payload=dict(identity,argv=argv,gpu_count=count,exclusive=False,global_lock=False,
            depends_on=depends,container_name=name,timeout_s=2700,
            required_receipts=['dynamo/completion.json' if kind=='dynamo' else 'completion.json'],
            formal_eligible=False,energy_comparable=False,evidence_class='independent_predictor_training' if kind=='predictor' else 'native_execution_probe')
        jobs.append(dict(job_id=name,payload=payload,priority={'predictor':410,'probe':400,'dynamo':390,'eco4':380}[kind],max_attempts=1))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=('predictor','probe','dynamo','eco4'))
    p.add_argument('--root',type=Path,default=ROOT/'results/2026-09-22/three-model')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();root=a.root.resolve()
    q=json.loads((root/'queue.json').read_text())
    depends=sorted(k for k,j in q['jobs'].items() if k.startswith('holdout-') and
                   j['status'] in ('running','succeeded') and j['priority']==300)
    if len(depends)!=5:raise ValueError(f'expected exactly five first-cohort calibration dependencies: {depends}')
    spec=importlib.util.spec_from_file_location('freeze',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    source,sha=helper.freeze_source(ROOT/'src',root/'native-acceptance-sources')
    receipt,_=helper.freeze_receipt(root/'model-verification.json',root/'profile-receipts')
    image=subprocess.check_output(['docker','image','inspect','pdblend:l20-cu128-vllm-v1','--format','{{.Id}}'],text=True).strip()
    jobs=build_jobs(a.kind,root=root,source=source,source_hash=sha,image=image,verification=receipt,depends=depends)
    if a.kind in ('dynamo','eco4'):
        for job in jobs:
            prerequisites=[key for key,j in q['jobs'].items() if key.startswith('native-probe-') and
                j['status'] in ('queued','running','succeeded') and j['payload']['model_id']==job['payload']['model_id']]
            if len(prerequisites)!=1:raise ValueError('one current native probe required before dependent baseline mechanisms')
            job['payload']['depends_on']=prerequisites
    output=root/f'native-{a.kind}-spec-{sha[:16]}.json'
    output.write_text(json.dumps(jobs,indent=2)+'\n')
    if not a.prepare_only:
        queue=GPULeaseQueue(root/'queue.json')
        for job in jobs:queue.enqueue(**job)
    print(json.dumps(dict(prepared_only=a.prepare_only,spec=str(output),jobs=[j['job_id'] for j in jobs],depends=depends),indent=2))


if __name__=='__main__':main()
