#!/usr/bin/env python3
"""Prepare a fresh 7B/32B TP4 power cohort; never enqueue or change old jobs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from pdblend.profile.power_calibration import load_package,digest,write_immutable,timing_package_binding
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('power_source_freeze',ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
freeze=importlib.util.module_from_spec(spec);spec.loader.exec_module(freeze)


def after_quad(state):
    selected={}
    for name,job in state['jobs'].items():
        payload=job['payload']
        if (payload.get('cohort_id','').startswith('quad-profile-4-2-1-1-') and
                job['status'] in ('queued','running','succeeded')):
            member=payload.get('profile_wave_member')
            if member in selected:raise ValueError('ambiguous live quad member; refresh review instead of changing it')
            selected[member]=name
    if len(selected)!=4:raise ValueError('exactly four existing quad members must precede the independent power pair')
    return sorted(selected.values())


def mounts_for_package(package,manifest):
    paths=[package]
    inputs=manifest['inputs']
    for name,binding in inputs.items():
        path=Path(binding['path'])
        if digest(path)!=binding['sha256']:raise ValueError('immutable package input checksum mismatch')
        if name in ('original_raw','original_completion','base_candidate','original_manifest'):
            paths.append(path.parent)
        else:paths.append(path)
    unique=[]
    for path in sorted(set(paths),key=lambda p:(len(p.parts),str(p))):
        if not any(parent.is_dir() and path.is_relative_to(parent) for parent in unique):unique.append(path)
    for binding in inputs.values():
        path=Path(binding['path'])
        if not any(path==p or (p.is_dir() and path.is_relative_to(p)) for p in unique):
            raise AssertionError('missing read-only exact-path mount')
    return unique


def build_jobs(packages,state,source,source_hash,receipt,image,out,timing_package=None):
    loaded=[(p,*load_package(p)[:2]) for p in packages]
    if [(m['model_id'],m['tp']) for _,m,_ in loaded]!=[('Qwen2.5-7B-Instruct',4),('Qwen2.5-32B-Instruct',4)]:
        raise ValueError('power pair must contain distinct model-owned 7B and 32B TP4 packages')
    dependencies=after_quad(state)
    identity=dict(source_sha256=source_hash,image_digest=image,package_manifest_sha256=[digest(p/'manifest.json') for p in packages],
                  depends_on=dependencies,qualification='new_2100_MHz_paired_ProfileWave')
    timing_binding=timing_package_binding(timing_package) if timing_package is not None else None
    if timing_binding is not None:
        from pdblend.profile.timing_calibration import load_package as load_timing_package
        load_timing_package(Path(timing_package))
        identity['timing_package_binding']=timing_binding
    suffix=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    cohort='power-holdout-4-4-'+suffix;members=['7b-tp4-power','32b-tp4-power']
    wave=dict(cohort_id=cohort,coordinator=True,members=members,purpose='independent_power_holdouts_4_plus_4',
        keep_peers_resident_until_all_done=True,synchronize_parallel_windows=True,
        qualification_frequency_mhz=2100,profile_frequency_coverage=[900,1200,1500,1800,2100,2520],
        representative_layout_check_only=True,existing_quad_unchanged=True)
    wave_dir=out/'wave';jobs=[];read_only=[]
    for (package,manifest,plan),member in zip(loaded,members):
        name=f'power-holdout-{member}-{suffix}'
        mounts=mounts_for_package(package,manifest)
        timing_member=timing_binding is not None and manifest['model_id']=='Qwen2.5-32B-Instruct'
        if timing_member:
            timing_root=Path(timing_binding['package'])
            timing_manifest=json.loads((timing_root/'manifest.json').read_text())
            for path in mounts_for_package(timing_root,timing_manifest):
                if not any(path==old or old.is_dir() and path.is_relative_to(old) for old in mounts):mounts.append(path)
        argv=['docker','run','--rm','--name',name,'--gpus','all','--cap-add','SYS_ADMIN','--ipc=host','--network','host',
            '--shm-size','16g','--ulimit','nofile=65536:65536','--entrypoint','/opt/venv/bin/python',
            '-v',f'{source}:/opt/pdblend-src:ro','-v','/home/models:/models:ro',
            '-v',f'{receipt}:/verification/model-verification.json:ro','-v','{attempt_dir}:/output:rw',
            '-v',f'{ROOT}/results/2026-09-22/three-model/profile-wave:/coord:rw','-v',f'{wave_dir}:/wave:rw']
        for path in mounts:argv+=['-v',f'{path}:{path}:ro']
        for env in ['PYTHONPATH=/opt/pdblend-src','PDBLEND_MODELS_DIR=/models',
            'PDBLEND_MODEL_VERIFICATION_RECEIPT=/verification/model-verification.json',
            'PDBLEND_SOURCE_SHA256='+source_hash,'PDBLEND_IMAGE_ID='+image,'PDBLEND_HARDWARE_ID=8xL20-lease',
            'PDBLEND_VLLM_VERSION=0.10.1.1','CUDA_VERSION=12.8.1','PDBLEND_GPU_UUIDS={lease_gpu_uuids}',
            'PDBLEND_COORD_DIR=/coord','PDBLEND_PROFILE_WAVE=/wave','PDBLEND_PROFILE_MEMBER='+member,
            'PDBLEND_CONCURRENCY_ENVIRONMENT','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256',
            'TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=4']:
            argv+=['-e',env]
        argv +=[image,'-B','-m','pdblend.profile.power_job','--package',str(package),'--model','/models/'+manifest['model_id'],
            '--gpus','0','1','2','3','--base-port','{lease_port}','--out','/output']
        if timing_member:argv+=['--timing-package',timing_binding['package']]
        payload=dict(schema=1,system='pdblend',model_id=manifest['model_id'],tp=4,pp=1,gpu_count=4,
            exclusive=False,global_lock=False,source_sha256=source_hash,source_snapshot=str(source),image_digest=image,
            argv=argv,container_name=name,timeout_s=5400,required_receipts=['queue-completion.json'],
            depends_on=dependencies,cohort_id=cohort,profile_wave_members=members,profile_wave_member=member,
            cohort_dir=str(wave_dir),cohort_member=member,topology=dict(model_id=manifest['model_id'],tp=4,pp=1,gpu_count=4),
            package_dir=str(package),package_manifest_sha256=digest(package/'manifest.json'),
            candidate_sha256=manifest['candidate_sha256'],plan_sha256=manifest['plan_sha256'],
            evidence_class='independent_decode_power_holdout',expected_power_points=24,expected_power_windows=72,
            reused_timing_passed=manifest['timing_component_passed'],timing_failures_are_not_power_failures=True,
            formal_eligible=False,energy_comparable=False,completion_is_formal_qualification=False,
            measurement_receipt_semantics='completed sampling; power/timing/composite status remain separately gated',
            scheduling_proxy_seconds=plan['cpu_scheduling_proxy']['combined_proxy_seconds'])
        if timing_member:
            payload.update(timing_package_binding=timing_binding,timing_overlay_requested=True,
                           timing_scope='independent_resident_panel_separate_from_original_power_and_timing_results',
                           timing_overlay_expected_points=18,timing_overlay_expected_windows=54,
                           scheduling_proxy_excludes_timing_panel=True)
        jobs.append(dict(job_id=name,payload=payload,priority=297,max_attempts=1))
        read_only.append(dict(job_id=name,exact_path_mounts=[str(x) for x in mounts],bound_inputs=manifest['inputs']))
    return dict(schema=1,mode='prepare_only',live_queue_modified=False,automatic_enqueue_allowed=False,
        source_frozen=True,identity=identity,jobs=jobs,wave=wave,wave_dir=str(wave_dir),read_only_inputs=read_only,
        predecessors_unchanged=dependencies,after_existing_quad=True,
        coordinator_steps=['After review, enqueue both jobs together at equal priority after all four existing quad members succeed.',
            'Two workers claim disjoint four-GPU UUID leases; the existing shared load lock staggers model loading.',
            'Both members reach ready. At 2100 MHz each runs B8/context1024 while the peer waits idle, then both run the same probe concurrently.',
            'Each parallel repeat waits until every member has active settled decode, then starts sampling; timing and group-power differences must each be <=5%, with >=2s common windows. Otherwise ProfileWave serializes the cohort.',
            'No existing quad/native member is counted as a peer; no one-member wave is labelled parallel.',
            'When requested, only the 32B member runs its separately bound 18-point timing panel on the same engine before publishing done; original power and timing receipts stay unchanged.',
            'A finished member stays resident with its GPU lease until every member writes done; only then may either member unload/reset/release.',
            'Any future downstream work must depend on both power jobs. Power completion does not promote failed 32B timing or formal ranking.'],
        restart_policy='No automatic retries: after interruption prepare a fresh wave directory and reviewed resume mapping; never reuse stale wave ready/done files.',
        qualifications_limit='One 2100-MHz representative check, not six-frequency interference validation or whole-host energy comparison.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--queue',type=Path,default=ROOT/'results/2026-09-22/three-model/queue.json')
    p.add_argument('--packages',type=Path,nargs=2,metavar=('PACKAGE_7B','PACKAGE_32B'),
        help='new immutable packages when bound implementation hashes change')
    p.add_argument('--timing-package',type=Path,help='optional independent 32B timing panel on its resident engine')
    a=p.parse_args();out=a.out.resolve();state=json.loads(a.queue.read_text())
    packages=[path.resolve() for path in a.packages] if a.packages else [ROOT/'results/2026-09-23/7b-tp4-power-override-package',ROOT/'results/2026-09-23/32b-power-training-review-final/tp4-package']
    for package in packages:load_package(package)
    source,source_hash=freeze.freeze_source(ROOT/'src',ROOT/'results/2026-09-23/power-holdout-sources')
    image=json.loads((packages[0]/'manifest.json').read_text())['original_timing_environment']['image_digest']
    receipt=ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'
    if not receipt.is_file():raise ValueError('pinned model verification receipt missing')
    review=build_jobs(packages,state,source,source_hash,receipt,image,out,timing_package=a.timing_package)
    write_immutable(out/'wave'/'wave.json',review['wave']);write_immutable(out/'job-review.json',review)
    write_immutable(out/'jobs.json',review['jobs'])
    print(json.dumps(dict(review=str(out/'job-review.json'),source_sha256=source_hash,jobs=[j['job_id'] for j in review['jobs']],
        depends_on=review['predecessors_unchanged'],enqueue=False),indent=2))

if __name__=='__main__':main()
