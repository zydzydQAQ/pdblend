#!/usr/bin/env python3
"""Prepare only missing Dynamo B16 corners with a fresh shared sampling cohort."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from pdblend.bench.resident_session import file_sha,digest,write_new
from pdblend_baselines.dynamollm.profile_v1 import measurement_points
from types import SimpleNamespace

IMAGE='sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
VERIFY=ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'
DEPS=ROOT/'results/2026-09-23/dynamo-python-deps-v1'
BASE=ROOT/'results/2026-09-24/resident-comparison-eco-v3/sources/d7d9372cbfb57f21830a342448db9c14a3b5b2ca094665bb41298ddb2e21f5c3'
ATTEMPTS=ROOT/'results/2026-09-22/three-model/queue-attempts'
PRIORS={
 '7b':'dynamo-gap-7b-profile-b75b87d56104d4ce/attempt-0001-e6a765d70529479c910157867ef23129/profile',
 '14b':'dynamo-gap-14b-profile-9c614491adbe70f6/attempt-0001-42784a2977ed4ca3a9daa4aa28edcfeb/profile',
 '32b':'dynamo-gap-32b-profile-6e5acd8b00661e64/attempt-0001-2e0d1691562b485ba8515f7606b0b1e0/profile'}
TARGET_PRIORS={
 '7b':'native-dynamo-7b-43cbb1f425c37905-outfix-51be08d6dcd3/attempt-0001-9dd44f6e1ec94c62960a184090fd4ef2/dynamo/target-profile',
 '14b':'native-dynamo-14b-1cb15f112c3158a1-outfix-51be08d6dcd3/attempt-0001-964f81eb5d7c4948ad83d7b35f367f53/dynamo/target-profile'}


def ref(path):return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


def missing_plan(prior, model, tp):
    prior=Path(prior);complete=json.loads((prior/'completion.json').read_text())
    measured=json.loads((prior/'profile.json').read_text())
    if (complete.get('status')!='passed' or complete.get('complete') is not True or complete.get('cleanup_errors')
            or measured.get('model_id')!=model or measured.get('tp')!=tp
            or measured.get('independent_profile') is not True or measured.get('measurement')!='hardware'):
        raise ValueError('complete same-model independent prior profile required')
    points=[dict(frequency_mhz=p['frequency_mhz'],input_tokens=p['input_tokens'],
                 output_tokens=p['context_tokens']-p['input_tokens'],batch=p['batch']) for p in measured['points']]
    expected={(f,n,o,1) for f in (900,1200,1500,1800,2100,2520) for n in (16,7168) for o in (74,512)}
    keys=lambda p:(p['frequency_mhz'],p['input_tokens'],p['output_tokens'],p['batch'])
    if len(points)!=24 or set(map(keys,points))!=expected or complete.get('points')!=24:
        raise ValueError('prior must be the already measured 24-corner B1 envelope')
    return dict(schema='dynamo-missing-profile-cells-v1',system='dynamollm',model_id=model,tp=tp,pp=1,
        fit_from_holdout=False,points=[dict(p,batch=16) for p in sorted(points,key=keys)],
        reused_profile=ref(prior/'profile.json'),reused_completion=ref(prior/'completion.json'),
        prior_cells_recollect=False,selection_used_evaluation_outputs=False,
        max_num_seqs_source='unchanged original independent Dynamo admission limit 16',
        scope='only the missing batch boundary; raw geometry is not formal interpolation qualification',
        formal_eligible=False,remaining_gates=['independent interpolation/heterogeneous-batch holdout',
            'additional target TP coverage','loaded transition coverage beyond measured B4/direction',
            'original 1800/300/5-second actions','original stationary-weight retention'])


def target_plan(prior, model):
    """The six measured TP2 points do not bound the new request/batch geometry."""
    prior=Path(prior);complete=json.loads((prior/'completion.json').read_text())
    measured=json.loads((prior/'profile.json').read_text())
    if (complete.get('status')!='passed' or complete.get('complete') is not True or complete.get('cleanup_errors')
            or measured.get('model_id')!=model or model not in ('Qwen2.5-7B-Instruct','Qwen2.5-14B-Instruct')
            or measured.get('tp')!=2 or measured.get('independent_profile') is not True
            or measured.get('measurement')!='hardware'):
        raise ValueError('complete same-model independent TP2 target profile required')
    actual={(p['frequency_mhz'],p['input_tokens'],p['context_tokens']-p['input_tokens'],p['batch'])
            for p in measured['points']}
    expected={(f,512,64,1) for f in (900,1200,1500,1800,2100,2520)}
    if len(measured['points'])!=6 or complete.get('points')!=6 or actual!=expected:
        raise ValueError('target gap requires the recorded six singleton TP2 points')
    return dict(schema='dynamo-missing-profile-cells-v1',system='dynamollm',model_id=model,tp=2,pp=1,
        fit_from_holdout=False,points=[dict(frequency_mhz=f,input_tokens=n,output_tokens=o,batch=b)
            for f in (900,1200,1500,1800,2100,2520) for n in (16,7168) for o in (74,512) for b in (1,16)],
        reused_profile=ref(prior/'profile.json'),reused_completion=ref(prior/'completion.json'),
        prior_cells_recollect=False,selection_used_evaluation_outputs=False,
        scope='missing own target TP2 bounds for native ScaleShard candidate evaluation',
        original_target_geometry=[[512,64,1]],formal_eligible=False,
        remaining_gates=['independent interpolation/heterogeneous-batch holdout','remaining target TP4 coverage',
            'loaded transition coverage beyond measured B4/direction','original 1800/300/5-second actions',
            'original stationary-weight retention'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--include-target-tp',action='store_true',
        help='add only missing 7B/14B TP2 boundary profiles in one five-member eight-GPU cohort')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    loader=importlib.util.spec_from_file_location('dynamo_bridge_freezer',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper=importlib.util.module_from_spec(loader);loader.loader.exec_module(helper)
    base_manifest=json.loads((BASE/'manifest.json').read_text());helper.verify_snapshot(BASE,base_manifest['files'])
    with tempfile.TemporaryDirectory(prefix='dynamo-batch-bridge-') as temp:
        stage=Path(temp)/'src';shutil.copytree(BASE,stage)
        for name in ('asset_preflight.py','trace_coverage.py','profile_v1.py'):
            shutil.copyfile(ROOT/'src/pdblend_baselines/dynamollm'/name,stage/'pdblend_baselines/dynamollm'/name)
        source,source_sha=helper.freeze_source(stage,out/'sources')
    members=['dynamo-32b','dynamo-14b','dynamo-7b'];counts={'dynamo-32b':2,'dynamo-14b':1,'dynamo-7b':1}
    jobs_to_prepare=[(key,key,2 if key=='32b' else 1,ATTEMPTS/PRIORS[key],missing_plan)
                     for key in ('7b','14b','32b')]
    if args.include_target_tp:
        for key in ('7b','14b'):
            member='dynamo-'+key+'-tp2';members.append(member);counts[member]=2
            jobs_to_prepare.append((key+'-tp2',key,2,ATTEMPTS/TARGET_PRIORS[key],target_plan))
    cohort_dir=out/'sampling-cohort';cohort_dir.mkdir()
    cohort=dict(cohort_id=out.name+'-'+source_sha[:12],gpu_budget=sum(counts.values()),members=members,member_gpu_counts=counts,
        independent_system_profiles=True,qualification_common_window_minimum_s=2.,qualification_measure_s=8.,
        require_isolated_parallel_limit=.05,release_protocol='window_boundary_retire_cleanup_release_requalify',
        recovery_policy='reuse only completed own B1 input bindings; all new B16 cells need fresh cohort qualification')
    write_new(cohort_dir/'cohort.json',cohort)
    jobs=[];configs=[];preflights=[]
    for key,model_key,tp,prior,builder in jobs_to_prepare:
        model='Qwen2.5-'+model_key.upper()+'-Instruct'
        directory=out/key;directory.mkdir()
        plan=builder(prior,model,tp) if builder is missing_plan else builder(prior,model)
        plan.update(native_capacity_check='fresh native total_kv_tokens before every cell; unsupported is never fitted',
                    prior_native_capability=ref(prior/'capability.json'))
        plan_path=directory/'plan.json';write_new(plan_path,plan)
        measurement_points(SimpleNamespace(points_file=plan_path,tp=tp),model)
        config=dict(kind='profile',model_id=model,model_path='/models/'+model,tp=tp,pp=1,gpus=list(range(tp)),
            source_snapshot=str(source),source_sha256=source_sha,dependencies_manifest=str(DEPS/'manifest.json'),
            points_file=str(plan_path),sampling_epoch_root=str(cohort_dir),sampling_member='dynamo-'+key,
            sampling_cohort=cohort['cohort_id'],formal_eligible=False,
            immutable_inputs={str(p):file_sha(p) for p in (plan_path,DEPS/'manifest.json',cohort_dir/'cohort.json',
                Path(plan['reused_profile']['path']),Path(plan['reused_completion']['path']),prior/'capability.json')})
        config_path=directory/'config.json';write_new(config_path,config);configs.append(ref(config_path))
        job_id='dynamo-batch16-'+key+'-'+digest(config)[:16]
        argv=['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN',
            '--ipc=host','--network=host','--shm-size=16g','--entrypoint','/opt/venv/bin/python']
        for host,target,mode in [(ROOT,ROOT,'ro'),(source,'/opt/pdblend-src','ro'),('/home/models','/models','ro'),
                (cohort_dir,cohort_dir,'rw'),(cohort_dir/'cohort.json',cohort_dir/'cohort.json','ro'),
                ('{attempt_dir}','{attempt_dir}','rw'),
                ('/tmp/pdblend-physical-clock-owners','/tmp/pdblend-physical-clock-owners','rw')]:
            argv+=['-v',f'{host}:{target}:{mode}']
        env=dict(PYTHONPATH='/opt/pdblend-src:'+str(DEPS),PYTHONDONTWRITEBYTECODE='1',
            PDBLEND_MODELS_DIR='/models',PDBLEND_MODEL_VERIFICATION_RECEIPT=str(VERIFY),
            PDBLEND_SOURCE_MANIFEST=str(source/'manifest.json'),PDBLEND_SOURCE_SHA256=source_sha,
            PDBLEND_IMAGE_ID=IMAGE,PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',CUDA_VISIBLE_DEVICES='{lease_local_indices}',
            PDBLEND_VLLM_VERSION='0.10.1.1',PDBLEND_HARDWARE_ID='8xL20-lease',CUDA_VERSION='12.8.1',
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
            PDBLEND_CONCURRENCY_ENVIRONMENT='{attempt_dir}/concurrency-environment.json',
            PDBLEND_CLOCK_LOCK_DIR='/tmp/pdblend-physical-clock-owners')
        for name,value in env.items():argv+=['-e',name+'='+value]
        argv+=[IMAGE,'-B','-m','pdblend_baselines.dynamollm.profile_v1','--model',config['model_path'],
            '--tp',str(tp),'--gpus','{lease_local_indices}','--base-port','{lease_port}',
            '--points-file',str(plan_path),'--job-config',str(config_path),
            '--out','{attempt_dir}/profile','--settle','2','--measure','5',
            '--sampling-epoch-root',str(cohort_dir),'--sampling-member','dynamo-'+key,'--require-sampling-epochs']
        job=dict(job_id=job_id,priority=150,max_attempts=1,payload=dict(argv=argv,container_name=job_id,cwd=str(ROOT),
            system='dynamollm',model_id=model,tp=tp,pp=1,gpu_count=tp,exclusive=False,reserve_host=False,
            depends_on=[],after_terminal=[],timeout_s=14400,source_snapshot=str(source),source_sha256=source_sha,
            image_digest=IMAGE,sampling_cohort=cohort['cohort_id'],sampling_cohort_member='dynamo-'+key,
            sampling_cohort_members=members,sampling_cohort_path=str(cohort_dir/'cohort.json'),
            sampling_cohort_sha256=file_sha(cohort_dir/'cohort.json'),
            required_receipts=['profile/completion.json','profile/epoch-drain.json','profile/epoch-release.json'],
            config_path=str(config_path),kind='missing_target_TP2_profile_boundary' if builder is target_plan
                else 'missing_B16_profile_boundary',formal_eligible=False,
            measurement_qualification='fresh isolated/parallel probes, repeat guards, safe retirement',
            prior_cells_recollect=False,prepare_only=True))
        jobs.append(job)
        check=[];i=0
        while i<argv.index(IMAGE):
            if argv[i] in ('--gpus','--cap-add'):i+=2;continue
            check.append(argv[i].replace(job_id,job_id+'-cpu').replace('{attempt_dir}',str(directory/'cpu-output'))
                .replace('{lease_gpu_uuids}','').replace('{lease_local_indices}',','.join(map(str,range(tp)))))
            i+=1
        (directory/'cpu-output').mkdir()
        check+=['-e','NVIDIA_VISIBLE_DEVICES=void',IMAGE,'-B','-m','pdblend_baselines.dynamollm.asset_preflight',
                '--config',str(config_path)]
        write_new(directory/'cpu-command.json',dict(argv=check,hardware_executed=False))
        proc=subprocess.run(check,capture_output=True,text=True,timeout=180)
        (directory/'cpu.stdout').write_text(proc.stdout);(directory/'cpu.stderr').write_text(proc.stderr)
        if proc.returncode:raise RuntimeError('CPU preflight failed: '+key+' '+proc.stderr[-1000:])
        receipt=json.loads(proc.stdout);write_new(directory/'cpu-preflight.json',receipt);preflights.append(ref(directory/'cpu-preflight.json'))
    write_new(out/'jobs.json',jobs)
    write_new(out/'review.json',dict(status='cpu_preflight_passed',hardware_executed=False,queue_modified=False,
        jobs=ref(out/'jobs.json'),source_manifest=ref(source/'manifest.json'),source_sha256=source_sha,
        source_base_manifest=ref(BASE/'manifest.json'),configs=configs,preflights=preflights,
        cohort=ref(cohort_dir/'cohort.json'),new_cells=168 if args.include_target_tp else 72,reused_B1_boundary_cells=72,
        reused_target_TP2_singleton_cells=12 if args.include_target_tp else 0,
        total_leased_gpu_count=sum(counts.values()),schedule='all cohort jobs together after current host-exclusive measurement',
        formal_eligible=False,scope='necessary independent batch geometry; additional formal gates remain explicit'))
    print(json.dumps(dict(jobs=str(out/'jobs.json'),jobs_sha256=file_sha(out/'jobs.json'),
                         new_cells=168 if args.include_target_tp else 72,queue_modified=False)))


if __name__=='__main__':main()
