#!/usr/bin/env python3
"""Prepare-only useful incremental profile and loaded-drain jobs; never enqueue."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

ROOT=Path('/home/pdblend4')
BASE=ROOT/'results/2026-09-23/dynamo-functional-sources/05ce62a146b4a02a9ccb0b9ceba07642b4c5273c47f7078a72ee7717b873ab49'
DEPS=ROOT/'results/2026-09-23/dynamo-python-deps-v1'
OVERLAY=('history_provenance.py','validation.py','transition_evidence.py','prediction_cache.py',
         'profile_v1.py','profile_epochs.py','portable_profile.py','loaded_drain.py','transition_probe_v1.py',
         'transition_costs.py','runtime.py','asset_preflight.py')
COORDINATION=('sampling_epochs.py','wave.py','parallel.py')


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):result.update(chunk)
    return result.hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def prepare(out, *, cohort=None, profiles_only=False):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    helper=load('gap_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    pins=load('gap_pins',ROOT/'scripts/2026-09-23_prepare_dynamo_functional_jobs.py')
    helper.verify_snapshot(BASE,json.loads((BASE/'manifest.json').read_text())['files'])
    cohort=Path(cohort).resolve() if cohort else None
    cohort_spec=json.loads((cohort/'cohort.json').read_text()) if cohort else None
    with tempfile.TemporaryDirectory(prefix='dynamo-gap-source-') as directory:
        staging=Path(directory)/'src';shutil.copytree(BASE,staging)
        for name in OVERLAY:
            shutil.copyfile(ROOT/'src/pdblend_baselines/dynamollm'/name,staging/'pdblend_baselines/dynamollm'/name)
        for name in COORDINATION:
            shutil.copyfile(ROOT/'src/pdblend/profile'/name,staging/'pdblend/profile'/name)
        source,digest=helper.freeze_source(staging,ROOT/'results/2026-09-23/dynamo-asset-sources')
    jobs=[]
    for key,(model,tp,_) in pins.MODELS.items():
        for kind in (('profile',) if profiles_only else ('profile','loaded_transition')):
            directory=out/(key+'-'+kind);directory.mkdir()
            gpu_count=tp if kind=='profile' else tp*3
            config=dict(kind=kind,model_id=model,model_path='/models/'+model,tp=tp,pp=1,
                gpus=list(range(gpu_count)),source_snapshot=str(source),source_sha256=digest,
                dependencies_manifest='/deps/manifest.json',
                immutable_inputs={'/deps/manifest.json':sha(DEPS/'manifest.json')},formal_eligible=False)
            if kind=='profile':
                points=[dict(frequency_mhz=f,input_tokens=n,output_tokens=o,batch=1)
                        for f in (900,1200,1500,1800,2100,2520) for n in (16,7168) for o in (74,512)]
                plan=dict(schema='dynamo-missing-profile-cells-v1',system='dynamollm',model_id=model,
                    tp=tp,pp=1,fit_from_holdout=False,points=points,
                    coverage_scope='initial-TP input/output boundaries at B1; no larger batch qualification',
                    input_bounds=[16,7168],output_bounds=[74,512],
                    missing_gates=['batch>1 coverage','independent interpolation holdout',
                                   'runtime parallel interference qualification','rate anchor'],
                    prior_18_cells_recollect=False,formal_eligible=False)
                planpath=directory/'plan.json';save(planpath,plan)
                config['points_file']='/spec/plan.json';config['immutable_inputs']['/spec/plan.json']=sha(planpath)
                module='pdblend_baselines.dynamollm.profile_v1';output='profile'
                args=['--model','/models/'+model,'--tp',str(tp),'--gpus',','.join(map(str,range(tp))),
                    '--points-file','/spec/plan.json','--base-port','{lease_port}',
                    '--out','/output/profile','--resume','--settle','2','--measure','5']
                if cohort:
                    member='dynamo-'+key
                    if member not in cohort_spec['members']:
                        raise ValueError('Dynamo member missing from frozen sampling cohort')
                    config.update(sampling_epoch_root='/sampling-epoch',sampling_member=member,
                        sampling_cohort=cohort_spec['cohort_id'])
                    config['immutable_inputs']['/sampling-epoch/cohort.json']=sha(cohort/'cohort.json')
                    args+=['--sampling-epoch-root','/sampling-epoch','--sampling-member',member,
                           '--require-sampling-epochs','--job-config','/spec/config.json']
                timeout=14400
            else:
                config.update(drain_input=7168,drain_output=512,drain_batch=4)
                module='pdblend_baselines.dynamollm.transition_probe_v1';output='dynamo'
                args=['--model','/models/'+model,'--gpus',','.join(map(str,range(gpu_count))),
                    '--base-port','{lease_port}','--out','/output/dynamo','--drain-input','7168',
                    '--drain-output','512','--drain-batch','4','--transition-timeout','900']
                timeout=2400
            save(directory/'config.json',config)
            identity=dict(source=digest,model=model,kind=kind,config=sha(directory/'config.json'),image=pins.IMAGE)
            suffix=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
            job_id=f'dynamo-gap-{key}-{kind.replace("_","-")}-{suffix}'
            mounts=[(str(source),'/opt/pdblend-src','ro'),(str(source/'manifest.json'),'/source-manifest.json','ro'),
                ('/home/models','/models','ro'),(str(pins.VERIFY),'/verification/model-verification.json','ro'),
                (str(directory),'/spec','ro'),(str(DEPS),'/deps','ro')]
            if kind=='profile' and cohort:
                mounts += [(str(cohort),'/sampling-epoch','rw'),
                           (str(cohort/'cohort.json'),'/sampling-epoch/cohort.json','ro')]
            argv=['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN',
                  '--ipc=host','--network=host','--shm-size=16g','--entrypoint','/opt/venv/bin/python']
            for host,target,mode in mounts:argv+=['-v',f'{host}:{target}:{mode}']
            argv+=['-v','{attempt_dir}:/output:rw']
            env={'PYTHONPATH':'/opt/pdblend-src:/deps','PYTHONDONTWRITEBYTECODE':'1',
                 'PDBLEND_MODEL_VERIFICATION_RECEIPT':'/verification/model-verification.json',
                 'PDBLEND_MODELS_DIR':'/models','PDBLEND_SOURCE_SHA256':digest,'PDBLEND_IMAGE_ID':pins.IMAGE,
                 'PDBLEND_SOURCE_MANIFEST':'/source-manifest.json','PDBLEND_VLLM_VERSION':'0.10.1.1',
                 'PDBLEND_HARDWARE_ID':'8xL20-lease','CUDA_VERSION':'12.8.1',
                 'CUDA_VISIBLE_DEVICES':'{lease_local_indices}','PDBLEND_GPU_UUIDS':'{lease_gpu_uuids}',
                 'PDBLEND_LEASE_PORT':'{lease_port}','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4'}
            for name,value in env.items():argv+=['-e',name+'='+value]
            for name in ('PDBLEND_CONCURRENCY_ENVIRONMENT','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256'):
                argv+=['-e',name]
            argv+=[pins.IMAGE,'-B','-m',module,*args]
            job=dict(job_id=job_id,priority=150,max_attempts=1,payload=dict(schema='dynamo-gap-job-v1',
                model_id=model,tp=tp,pp=1,gpu_count=gpu_count,exclusive=False,
                source_snapshot=str(source),source_sha256=digest,source_revision=digest,
                image_digest=pins.IMAGE,argv=argv,mounts=mounts,timeout_s=timeout,
                required_receipts=[output+'/completion.json'],container_name=job_id,
                config_path=str(directory/'config.json'),kind=kind,
                exact_inputs_sha256={'config':sha(directory/'config.json')},
                formal_eligible=False,energy_comparable=False,prepare_only=True,
                expected_scope='hardware measurement only; independent policy/full qualification remains separate'))
            if kind=='profile' and cohort:
                job['payload'].update(sampling_cohort=cohort_spec['cohort_id'],
                    sampling_cohort_member=member,sampling_cohort_members=cohort_spec['members'],
                    sampling_cohort_path=str(cohort/'cohort.json'),
                    sampling_cohort_sha256=sha(cohort/'cohort.json'),
                    measurement_qualification='fresh isolated/parallel probes, repeat guards, safe retirement',
                    required_receipts=['profile/completion.json','profile/epoch-drain.json','profile/epoch-release.json'])
            jobs.append(job);save(directory/'job.json',job)
    save(out/'jobs.json',jobs)
    save(out/'manifest.json',dict(source_snapshot=str(source),source_sha256=digest,base=str(BASE),
        overlay=list(OVERLAY),coordination_only_shared=list(COORDINATION),
        image_digest=pins.IMAGE,jobs_sha256=sha(out/'jobs.json'),
        queue_modified=False,gpu_started=False,prepare_only=True))
    return out/'jobs.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cohort',type=Path)
    parser.add_argument('--profiles-only',action='store_true')
    args=parser.parse_args()
    print(prepare(args.out,cohort=args.cohort,profiles_only=args.profiles_only))
