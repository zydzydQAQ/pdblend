#!/usr/bin/env python3
"""Prepare one immutable combined job and CPU-preflight it; never modify a queue."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.dont_write_bytecode = True
from pdblend.bench import longbench_anchor_recovery as recovery
from pdblend.bench import longbench_mixed_combo as combo
from pdblend.bench.resident_session import write_new, file_sha, digest

PARENT = ROOT/'results/2026-09-24/resident-comparison-eco-v3/campaign.json'
PRIOR = ROOT/'results/2026-09-22/three-model/queue-attempts/mixed-rate-anchor-32b-e954f2cc36b8/attempt-0001-a33b59da895d45e0b86113a2a1d2979c/anchor'
VERIFY = ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent-campaign', type=Path, default=PARENT)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--minimum-rate', type=float, default=.03125)
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    parent_ref = recovery.binding(args.parent_campaign); parent = recovery.bound(parent_ref)
    group = combo.parent_group(parent)
    baseline_ref = group['points'][0]['source_manifest']
    if any(p['source_manifest'] != baseline_ref for p in group['points']):
        raise ValueError('inherited Mixed source snapshots differ')
    execution_inputs_ref = recovery.binding(args.parent_campaign.resolve().parent/'execution-inputs.json')
    execution_inputs = recovery.bound(execution_inputs_ref)
    parent_source_ref = recovery.binding(Path(execution_inputs['source'])/'manifest.json')
    parent_source = recovery.bound(parent_source_ref)
    if parent_source['source_sha256'] != execution_inputs['source_sha256']:
        raise ValueError('parent execution source identity differs')
    spec = importlib.util.spec_from_file_location('combo_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(spec); spec.loader.exec_module(freezer)
    parent_source_root = Path(parent_source_ref['path']).parent
    freezer.verify_snapshot(parent_source_root, parent_source['files'])
    stage = out/'source-staging'; stage.mkdir()
    for name in parent_source['files']:
        target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent_source_root/name, target)
    for name in combo.OVERLAYS:
        target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/'src'/name, target)
    source, source_sha = freezer.freeze_source(stage, out/'sources')
    shutil.rmtree(stage)
    source_ref = recovery.binding(source/'manifest.json')
    protection = combo.validate_source(parent_source_ref, source_ref)
    baseline_protection = combo.validate_baseline_source(baseline_ref, source_ref)
    refs = recovery.prior_inputs(args.prior)
    rates = recovery.candidate_rates(recovery.bound(refs['failed_tuning']['completion'])['rate_rps'], args.minimum_rate)
    image = group['engine_identity']['image_digest']
    recovery_plan = dict(schema=recovery.SCHEMA, model_id=recovery.MODEL, prior=refs,
        minimum_rate_rps=args.minimum_rate, candidate_rates_rps=rates, slo=dict(ttft_s=15., tpot_s=.2),
        selection_splits=['calibration', 'tuning'], evaluation_used_for_selection=False,
        calibration_seed=9701, tuning_seed=9702, window_s=60, confirm_window_s=120,
        source_manifest=source_ref, image_digest=image,
        policy='halve prior failed tuning rate; fresh calibration then independent tuning; first confirmation or floor')
    write_new(out/'recovery-plan.json', recovery_plan)
    plan = dict(schema=combo.SCHEMA, model_id=recovery.MODEL, parent_campaign=parent_ref,
        parent_group_sha256=digest(group), parent_source_manifest=parent_source_ref,
        baseline_source_manifest=baseline_ref, parent_execution_inputs=execution_inputs_ref,
        source_manifest=source_ref, recovery_plan=recovery.binding(out/'recovery-plan.json'),
        fallback='only complete predeclared SLO-only exhaustion after verified physical cleanup',
        evaluation=dict(seed=701, duration_s=150, scales=list(combo.SCALES), generate_only_after_confirmed_anchor=True),
        input_preservation='all inherited parent points and trace bytes remain unchanged',
        max_engine_load_cycles=2, extra_anchor_load_allocated_to_service=False)
    path = out/'plan.json'; write_new(path, plan)
    job_id = 'comparison-32b-anchor-combined-'+digest(plan)[:16]
    argv = ['docker', 'run', '--rm', '--name', job_id, '--gpus', 'all', '--cap-add', 'SYS_ADMIN',
        '--ipc=host', '--network=host', '--shm-size=16g', '--ulimit', 'nofile=65536:65536',
        '--entrypoint', '/opt/venv/bin/python']
    for host,target,mode in [(ROOT,ROOT,'ro'), (source,'/opt/pdblend-src','ro'),
            ('/home/models','/models','ro'), ('{attempt_dir}','{attempt_dir}','rw'),
            ('/tmp/pdblend-physical-clock-owners','/tmp/pdblend-physical-clock-owners','rw')]:
        argv += ['-v', f'{host}:{target}:{mode}']
    env = dict(group['engine_identity']['environment'], PYTHONPATH='/opt/pdblend-src', PYTHONDONTWRITEBYTECODE='1',
        PDBLEND_MODELS_DIR='/models', PDBLEND_MODEL_VERIFICATION_RECEIPT=str(VERIFY),
        PDBLEND_SOURCE_MANIFEST=str(source/'manifest.json'), PDBLEND_SOURCE_SHA256=source_sha,
        PDBLEND_IMAGE_ID=image, PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',
        PDBLEND_CONCURRENCY_ENVIRONMENT='{attempt_dir}/concurrency-environment.json',
        CUDA_VISIBLE_DEVICES='{lease_local_indices}', CUDA_VERSION='12.8.1', OMP_NUM_THREADS='4',
        PDBLEND_CLOCK_LOCK_DIR='/tmp/pdblend-physical-clock-owners')
    for key,value in env.items(): argv += ['-e', key+'='+value]
    corpus = ROOT/'datasets/prepared/2026-09-22-32b-v1'
    argv += [image, '-B', '-m', 'pdblend.bench.longbench_mixed_combo', '--plan', str(path),
        '--plan-sha256', file_sha(path), '--model', '/models/'+recovery.MODEL, '--tp', '2',
        '--gpus', '{lease_local_indices}', '--corpus', str(corpus),
        '--out', '{attempt_dir}/combined', '--base-port', '{lease_port}']
    job = dict(job_id=job_id, priority=798, max_attempts=1, payload=dict(argv=argv, container_name=job_id,
        cwd=str(ROOT), model_id=recovery.MODEL, system='mixed', tp=2, pp=1, gpu_count=8,
        exclusive=True, global_lock=True, reserve_host=True, depends_on=[], after_terminal=[], timeout_s=24000,
        source_snapshot=str(source), source_sha256=source_sha, image_digest=image,
        comparison_campaign=str(args.parent_campaign.resolve()), combined_plan=recovery.binding(path),
        overlay_campaign_output='combined/campaign.json', session_output='combined/session',
        required_receipts=['combined/completion.json', 'combined/session/completion.json'],
        scope='independent_longbench_recovery_then_frozen_mixed_resident_comparison'))
    write_new(out/'jobs.json', [job]); cpu = out/'cpu-preflight'; cpu.mkdir()
    check=[]; i=0
    while i<len(argv):
        value=argv[i]
        if i<argv.index(image) and value in ('--gpus', '--cap-add'):
            i+=2; continue
        value=value.replace(job_id,job_id+'-cpu').replace('{attempt_dir}',str(cpu))
        value=value.replace('{lease_gpu_uuids}','').replace('{lease_local_indices}','0,1,2,3,4,5,6,7')
        value=value.replace('{lease_port}','19200'); check.append(value); i+=1
    index=check.index(image); check[index:index]=['-e', 'NVIDIA_VISIBLE_DEVICES=void']
    check+=['--preflight-only']; write_new(out/'preflight-command.json',dict(argv=check,hardware_executed=False))
    proc=subprocess.run(check,capture_output=True,text=True,timeout=180)
    (out/'preflight.stdout').write_text(proc.stdout); (out/'preflight.stderr').write_text(proc.stderr)
    receipt=cpu/'combined/preflight.json'
    review=dict(status='cpu_preflight_passed' if proc.returncode==0 else 'preflight_failed',
        returncode=proc.returncode, jobs=recovery.binding(out/'jobs.json'), plan=recovery.binding(path),
        source_sha256=source_sha, source_protection=protection, baseline_source_protection=baseline_protection,
        parent_campaign=parent_ref,
        preflight=recovery.binding(receipt) if receipt.is_file() else None, queue_modified=False,
        hardware_executed=False, job_id=job_id, priority=798, candidate_rates_rps=rates,
        conditional_points=dict(confirmed=12,slo_exhausted=8), implementation=recovery.binding(Path(__file__)))
    write_new(out/'review.json',review); print(json.dumps(review))
    return 0 if proc.returncode==0 else 2


if __name__=='__main__': raise SystemExit(main())
