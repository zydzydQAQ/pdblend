#!/usr/bin/env python3
"""Freeze the priority PDBlend smoke/A-B campaign and its scheduling revision."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
VERIFY = ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'
RATES = {'7b': (1., 4., 1.), '14b': (.5, 2., .5), '32b': (.25, 1., .25)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')


def trace(name, split):
    seed = 701 if split == 'evaluation' else 9701
    rates = RATES[name] if split == 'evaluation' else (RATES[name][0],)
    length = 100. if split == 'evaluation' else 60.
    rng = random.Random(seed)
    content = random.Random(seed * 7919 + 1)
    prompts = {n: [content.randrange(100, 5000) for _ in range(n)] for n in (512, 1024, 2048)}
    rows = []
    for stage, rate in enumerate(rates):
        t = stage * length + rng.expovariate(rate)
        while t < (stage+1)*length:
            n = (512, 1024, 2048)[len(rows) % 3]
            rows.append(dict(idx=len(rows), arrival_s=t, prompt=prompts[n], max_tokens=128,
                             source=f'synthetic-development-{name}-{split}'))
            t += rng.expovariate(rate)
    return dict(schema=1, model_id=f'Qwen2.5-{name.upper()}-Instruct', split=split,
                seed=seed, sampling_seed=701, duration_s=length*len(rates),
                stage_duration_s=length, rates_rps=rates, requests=rows,
                formal_eligible=False, energy_comparable=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--queue', type=Path, default=ROOT/'results/2026-09-22/three-model/queue.json')
    parser.add_argument('--enqueue', action='store_true')
    parser.add_argument('--compile-cache', type=Path,
        help='Verified per-model vLLM compile caches; copied for each new container')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    source, revision = helper.freeze_source(ROOT/'src', out/'sources')
    helper.verify_snapshot(source, json.loads((source/'manifest.json').read_text())['files'])
    state = json.loads(args.queue.read_text())
    if any(l.get('status') == 'active' for l in state['leases'].values()):
        raise RuntimeError('finish verified current-job cleanup before publishing the priority wave')
    cohort = out/'cohort'
    write(cohort/'cohort.json', dict(schema=1, cohort_id='native-ab-'+revision[:16],
          members=['7b', '14b', '32b'], qualification_limit=.05))
    pd_cohort = out/'pd-eight-cohort'
    write(pd_cohort/'cohort.json', dict(schema=1, cohort_id='native-pd8-'+revision[:16],
          members=['7b-pd8'], qualification_limit=.05))
    new_ids = {n: f'pdblend-quick-ab-{n}-{revision[:16]}' for n in RATES}
    pd_id = f'pdblend-quick-pd8-ab-{revision[:16]}'
    previous_pd = [key for key,j in state['jobs'].items()
                   if key.startswith('pdblend-quick-pd8-ab-') and j['status']=='queued']
    if len(previous_pd) > 1:
        raise ValueError('multiple pending eight-GPU smoke revisions require explicit reconciliation')
    jobs, preflights = [], {}
    model_inputs = {}
    for name in RATES:
        original = ROOT/f'results/2026-09-23/resident-domain-prepared-v3/packages/{name}-short-package/manifest.json'
        data = json.loads(original.read_text())
        base = Path(data['inputs']['base_candidate']['path'])
        if sha(base) != data['inputs']['base_candidate']['sha256']:
            raise ValueError('base profile identity changed')
        package = out/'inputs'/name
        evaluation, tuning = package/'trace.json', package/'tuning.json'
        write(evaluation, trace(name, 'evaluation')); write(tuning, trace(name, 'tuning'))
        manifest = package/'manifest.json'
        write(manifest, dict(schema=1, model_id=data['model_id'], tp=data['tp'], pp=1,
              image_digest=IMAGE, source_sha256=revision,
              inputs={k:dict(path=str(p), sha256=sha(p)) for k,p in
                      [('profile',base),('trace',evaluation),('tuning_trace',tuning),('model_verification',VERIFY)]}))
        model_inputs[name] = (data, base, package, manifest)

    for name, is_pd in [('7b',False),('14b',False),('32b',False),('7b',True)]:
        data, base, package, manifest = model_inputs[name]
        count = 8 if is_pd else data['tp']*2
        job_id = pd_id if is_pd else new_ids[name]
        wave, member = (pd_cohort,'7b-pd8') if is_pd else (cohort,name)
        mounts = [(source,'/opt/pdblend-src','ro'),(source/'manifest.json','/source-manifest.json','ro'),
                  (Path('/home/models'),'/models','ro'),(VERIFY,VERIFY,'ro'),
                  (base,base,'ro'),(package,package,'ro'),(wave,wave,'rw'),('{attempt_dir}','/output','rw')]
        cache_identity = None
        if args.compile_cache:
            cache_root = args.compile_cache.resolve()/name
            cache_manifest = json.loads((cache_root/'manifest.json').read_text())
            if cache_manifest['image_digest'] != IMAGE:
                raise ValueError('compile cache image differs')
            for rel,digest in cache_manifest['files'].items():
                candidate=(cache_root/rel).resolve()
                if not candidate.is_relative_to(cache_root) or sha(candidate)!=digest:
                    raise ValueError('compile cache changed: '+rel)
            cache_copy = out/'compile-cache'/job_id
            shutil.copytree(cache_root/'vllm',cache_copy)
            cache_identity = dict(manifest=str(cache_root/'manifest.json'),
                                  sha256=sha(cache_root/'manifest.json'), destination=str(cache_copy))
            mounts.append((cache_copy,'/root/.cache/vllm','rw'))
        argv = ['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN',
                '--ipc=host','--network=host','--shm-size=16g','--ulimit','nofile=65536:65536',
                '--entrypoint','/opt/venv/bin/python']
        for host,target,mode in mounts: argv += ['-v',f'{host}:{target}:{mode}']
        env = dict(PYTHONPATH='/opt/pdblend-src',PYTHONDONTWRITEBYTECODE='1',PDBLEND_MODELS_DIR='/models',
                   PDBLEND_MODEL_VERIFICATION_RECEIPT=str(VERIFY),PDBLEND_SOURCE_SHA256=revision,
                   PDBLEND_SOURCE_MANIFEST='/source-manifest.json',PDBLEND_IMAGE_ID=IMAGE,
                   PDBLEND_HARDWARE_ID='8xL20-lease',PDBLEND_VLLM_VERSION='0.10.1.1',CUDA_VERSION='12.8.1',
                   PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',CUDA_VISIBLE_DEVICES='{lease_local_indices}',
                   TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        for key,value in env.items(): argv += ['-e',f'{key}={value}']
        argv += ['-e','PDBLEND_CONCURRENCY_ENVIRONMENT','-e','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256',
                 IMAGE,'-B','-m','pdblend.bench.native_ab_smoke','--model','/models/'+data['model_id'],
                 '--profile',str(base),'--tp',str(data['tp']),'--gpus','{lease_local_indices}',
                 '--base-port','{lease_port}','--out','/output','--cohort-root',str(wave),'--member',member,
                 '--trace',str(package/'trace.json'),'--tuning-trace',str(package/'tuning.json'),
                 '--input-manifest',str(manifest)]
        if is_pd: argv += ['--pd-eight']
        payload=dict(schema=1,system='pdblend',model_id=data['model_id'],tp=data['tp'],pp=1,seed=701,
                     source_snapshot=str(source),source_sha256=revision,image_digest=IMAGE,
                     exact_inputs_sha256={str(manifest):sha(manifest)},argv=argv,container_name=job_id,
                     gpu_count=count,exclusive=is_pd,global_lock=is_pd,reserve_host=is_pd,
                     depends_on=[new_ids['7b']] if is_pd else [],
                     after_terminal=list(new_ids.values()) if is_pd else [],
                     required_receipts=['completion.json'],timeout_s=5400,
                     formal_eligible=False,energy_comparable=False,
                     scope='automatic_pd_latency_ab' if is_pd else 'native_functional_and_periodic_latency_ab',
                     queue_receipt_semantics='functional correctness, data completeness and cleanup; SLO efficacy audited separately')
        if cache_identity:
            payload['compile_cache'] = cache_identity
        if is_pd and previous_pd:
            payload['supersedes_job_id'] = previous_pd[0]
        if not is_pd:
            old = f'pdblend-online-safety-{name}-7bc52ec01720186b'
            if old in state['jobs'] and state['jobs'][old]['status']=='queued': payload['supersedes_job_id']=old
            payload['sampling_cohort']='native-ab-'+revision[:16]
        jobs.append(dict(job_id=job_id,priority=1100 if is_pd else 1000,max_attempts=1,payload=payload))
        cpu_out=out/'cpu-preflight'/job_id;cpu_out.mkdir(parents=True)
        cpu=[];i=0
        while i<len(argv):
            if i<argv.index(IMAGE) and argv[i] in ('--gpus','--cap-add'):
                i+=2;continue
            value=argv[i].replace('{attempt_dir}',str(cpu_out)).replace('{lease_local_indices}',','.join(map(str,range(count))))
            value=value.replace('{lease_gpu_uuids}','cpu-preflight-no-gpu').replace('{lease_port}','19000')
            if value==job_id:value=job_id+'-cpu'
            cpu.append(value);i+=1
        cpu[cpu.index(IMAGE):cpu.index(IMAGE)]=['--runtime','runc','-e','NVIDIA_VISIBLE_DEVICES=void']
        cpu+=['--preflight-only']
        write(cpu_out/'command.json',dict(argv=cpu,hardware_executed=False))
        proc=subprocess.run(cpu,capture_output=True,text=True,timeout=180)
        (cpu_out/'stdout').write_text(proc.stdout);(cpu_out/'stderr').write_text(proc.stderr)
        receipt=cpu_out/'preflight.json'
        preflights[job_id]=dict(returncode=proc.returncode,receipt=str(receipt),receipt_sha256=sha(receipt) if receipt.exists() else None)
        print(job_id,'CPU preflight',proc.returncode,flush=True)
        if proc.returncode: raise RuntimeError(f'CPU preflight failed: {cpu_out}')

    # Rewrite only unexecuted scheduling specs and preserve their immutable predecessors.
    replacements={job['payload']['supersedes_job_id']:job['job_id'] for job in jobs if job['payload'].get('supersedes_job_id')}
    deferred=[(key,j) for key,j in state['jobs'].items() if j['status']=='queued' and key not in replacements]
    for key,j in deferred: replacements[key]=key+'-after-quick-'+revision[:8]
    for key,old in deferred:
        payload=json.loads(json.dumps(old['payload']));new_id=replacements[key]
        payload['supersedes_job_id']=key
        payload['after_terminal']=sorted(set([replacements.get(x,x) for x in payload.get('after_terminal',[])]+[pd_id]))
        payload['depends_on']=[replacements.get(x,x) for x in payload.get('depends_on',[])]
        old_container=payload.get('container_name',key);payload['container_name']=new_id
        payload['argv']=[new_id if item==old_container else item for item in payload['argv']]
        jobs.append(dict(job_id=new_id,priority=old.get('priority',0),max_attempts=old.get('max_attempts',1),payload=payload))
    write(out/'jobs.json',jobs)
    review=dict(status='cpu_preflight_passed',source_sha256=revision,source_snapshot=str(source),
                image_digest=IMAGE,preflights=preflights,jobs_sha256=sha(out/'jobs.json'),
                campaign_jobs=list(new_ids.values())+[pd_id],deferred_jobs=[replacements[k] for k,_ in deferred],
                layout=[2,2,4],profile_recollection_required=False,formal_eligible=False,queue_modified=False)
    write(out/'review.json',review)
    if args.enqueue:
        from pdblend.experimentation.lease import GPULeaseQueue
        q=GPULeaseQueue(args.queue)
        q.enqueue_replacements(jobs)
        write(out/'enqueue.json',dict(status='published',jobs=[x['job_id'] for x in jobs]))
    print(json.dumps(dict(out=str(out),campaign_jobs=review['campaign_jobs'],deferred=len(deferred))))


if __name__=='__main__':main()
