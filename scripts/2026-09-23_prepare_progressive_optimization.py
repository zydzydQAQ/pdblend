#!/usr/bin/env python3
"""Freeze and CPU-preflight native safety checks plus a fresh eight-GPU cohort."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
IMAGE_TOKEN = 'pdblend:l20-cu128-vllm-v1@'+IMAGE
PYTHON = '/home/pdblend/.venv/bin/python'
VERIFY = ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--queue', type=Path, default=ROOT/'results/2026-09-22/three-model/queue.json')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('source_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source, source_sha = helper.freeze_source(ROOT/'src', out/'sources')
    helper.verify_snapshot(source, json.loads((source/'manifest.json').read_text())['files'])
    queue = json.loads(args.queue.read_text())
    # Terminal dependency means failure does not strand the independent audit;
    # the queue still waits for physical lease release before starting it.
    waiting = sorted(k for k, v in queue['jobs'].items()
                     if v['status'] in ('queued', 'running', 'pending'))
    models = {}
    for name in ('7b', '14b', '32b'):
        path = ROOT/f'results/2026-09-23/resident-domain-prepared-v3/packages/{name}-short-package/manifest.json'
        old = json.loads(path.read_text())
        for binding in old['inputs'].values():
            if sha(binding['path']) != binding['sha256']:
                raise ValueError('immutable input changed: '+binding['path'])
        models[name] = old
    names = [f'{name}-{frequency}' for frequency in (1500, 2520) for name in models]
    epochs = out/'epochs'
    cohort = 'pdblend-optimization-'+source_sha[:20]
    write(epochs/'cohort.json', dict(schema=1, cohort_id=cohort, members=names,
        coordinator=True, qualification_limit=.05, window_boundary_epochs=True, allow_external_peers=False))
    jobs, preflights = [], {}

    def command(job_id, mounts, extra_env, module_args):
        argv = ['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN',
                '--ipc=host','--network=host','--shm-size=16g','--ulimit','nofile=65536:65536',
                '--entrypoint','/opt/venv/bin/python']
        common = [(source,'/opt/pdblend-src','ro'), (source/'manifest.json','/source-manifest.json','ro'),
                  (Path('/home/models'),'/models','ro'), (VERIFY,'/verification/model-verification.json','ro'),
                  ('{attempt_dir}','/output','rw')]
        for host, target, mode in common+mounts:
            argv += ['-v',f'{host}:{target}:{mode}']
        env = dict(PYTHONPATH='/opt/pdblend-src', PYTHONDONTWRITEBYTECODE='1',
            PDBLEND_MODELS_DIR='/models', PDBLEND_MODEL_VERIFICATION_RECEIPT='/verification/model-verification.json',
            PDBLEND_SOURCE_SHA256=source_sha, PDBLEND_SOURCE_MANIFEST='/source-manifest.json',
            PDBLEND_IMAGE_ID=IMAGE, PDBLEND_HARDWARE_ID='8xL20-lease', PDBLEND_VLLM_VERSION='0.10.1.1',
            CUDA_VERSION='12.8.1', PDBLEND_GPU_UUIDS='{lease_gpu_uuids}', CUDA_VISIBLE_DEVICES='{lease_local_indices}',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', **extra_env)
        for key, value in env.items():
            argv += ['-e',f'{key}={value}']
        argv += ['-e','PDBLEND_CONCURRENCY_ENVIRONMENT','-e','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256',
                 IMAGE_TOKEN,'-B','-m',*module_args]
        return argv

    def add(job_id, argv, payload, gpu_count):
        payload.update(source_snapshot=str(source), source_sha256=source_sha, image_digest=IMAGE,
            argv=argv, container_name=job_id, gpu_count=gpu_count, depends_on=[],
            exclusive=False, global_lock=False, formal_eligible=False, energy_comparable=False,
            completion_is_formal_qualification=False, required_receipts=['completion.json'])
        jobs.append(dict(job_id=job_id, priority=430, max_attempts=1, payload=payload))
        cpu_out = out/'cpu-preflight'/job_id
        cpu_out.mkdir(parents=True)
        cpu, i = [], 0
        while i < len(argv):
            value = argv[i]
            if i < argv.index(IMAGE_TOKEN) and value in ('--gpus','--cap-add'):
                i += 2
                continue
            value = value.replace(job_id, job_id+'-cpu').replace('{attempt_dir}',str(cpu_out))
            value = value.replace('{lease_local_indices}',','.join(map(str,range(gpu_count))))
            value = value.replace('{lease_gpu_uuids}','cpu-preflight-no-gpu').replace('{lease_port}','19000')
            cpu.append(value)
            i += 1
        cpu[cpu.index(IMAGE_TOKEN):cpu.index(IMAGE_TOKEN)] = ['-e','NVIDIA_VISIBLE_DEVICES=void']
        cpu += ['--preflight-only']
        write(cpu_out/'command.json',dict(argv=cpu,hardware_executed=False))
        proc = subprocess.run(cpu,capture_output=True,text=True,timeout=120)
        (cpu_out/'stdout').write_text(proc.stdout)
        (cpu_out/'stderr').write_text(proc.stderr)
        receipt = cpu_out/'preflight.json'
        preflights[job_id] = dict(returncode=proc.returncode,receipt=str(receipt),
            receipt_sha256=sha(receipt) if receipt.exists() else None)
        print(job_id, 'CPU preflight', proc.returncode, flush=True)

    functional_ids = []
    for name, m in models.items():
        base = Path(m['inputs']['base_candidate']['path'])
        job_id = 'pdblend-online-safety-'+name+'-'+source_sha[:16]
        functional_ids.append(job_id)
        argv = command(job_id,[(base, str(base),'ro')],{}, ['pdblend.bench.online_qualification',
            '--model','/models/'+m['model_id'],'--profile',str(base),'--tp',str(m['tp']),
            '--gpus',','.join(map(str,range(m['tp']*2))),'--base-port','{lease_port}','--out','/output'])
        add(job_id,argv,dict(schema=1,system='pdblend',model_id=m['model_id'],tp=m['tp'],pp=1,
            after_terminal=waiting,timeout_s=3600,scope='native_disconnect_cancel_kv_dvfs_park_functional',
            profile_inputs=m['inputs'],seed=701),m['tp']*2)
    for frequency in (1500,2520):
        for name, m in models.items():
            member = f'{name}-{frequency}'
            package = out/'packages'/member
            base, raw = (m['inputs'][k]['path'] for k in ('base_candidate','identity_raw'))
            env = dict(os.environ,PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([PYTHON,'-B','-m','pdblend.profile.collection.optimization_profiles','prepare',
                '--base-candidate',base,'--identity-raw',raw,'--out',str(package),'--frequencies',str(frequency)],
                env=env,check=True,capture_output=True,text=True)
            job_id = 'pdblend-energy-gaps-'+member+'-'+source_sha[:16]
            mounts = [(package,str(package),'ro'), (base,base,'ro'), (raw,raw,'ro'),
                      (epochs,'/epochs','rw'),(epochs,'/coord','rw')]
            argv = command(job_id,mounts,dict(PDBLEND_COORD_DIR='/coord',PDBLEND_SAMPLING_EPOCH_ROOT='/epochs',
                PDBLEND_PROFILE_MEMBER=member),['pdblend.profile.collection.optimization_profiles','run',
                '--package',str(package),'--model','/models/'+m['model_id'],
                '--gpus',*map(str,range(m['tp'])),'--base-port','{lease_port}','--out','/output',
                '--epochs-root','/epochs','--member',member])
            add(job_id,argv,dict(schema=1,system='pdblend',model_id=m['model_id'],tp=m['tp'],pp=1,
                after_terminal=waiting+functional_ids,timeout_s=10800,sampling_cohort=cohort,
                cohort_dir=str(epochs),cohort_member=member,cohort_members=names,
                package_manifest=dict(path=str(package/'manifest.json'),sha256=sha(package/'manifest.json')),
                scope='low_batch_and_mixed_window_energy_training_independent_holdout',seed=701),m['tp'])
    write(out/'jobs.json',jobs)
    passed = all(x['returncode']==0 and x['receipt_sha256'] for x in preflights.values())
    write(out/'review.json',dict(status='cpu_preflight_passed' if passed else 'preflight_failed',
        source_sha256=source_sha,source_snapshot=str(source),image_digest=IMAGE,queue_modified=False,
        jobs_sha256=sha(out/'jobs.json'),preflights=preflights,existing_dependencies=waiting,
        functional_gpu_groups=[2,2,4],sampling_gpu_groups=[1,1,2,1,1,2],repeats_per_point=3,
        formal_eligible=False,hardware_executed=False,
        scope='functional validation and bounded component measurement; no formal energy comparison'))
    print(json.dumps(dict(out=str(out),passed=bool(passed),jobs=len(jobs))))
    return 0 if passed else 1


if __name__=='__main__':
    sys.exit(main())
