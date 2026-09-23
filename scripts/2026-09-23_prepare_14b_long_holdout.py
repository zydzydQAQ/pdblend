#!/usr/bin/env python3
"""Prepare immutable 14B TP1 holdout inputs and a non-enqueued GPU recipe."""
import argparse
import json
from pathlib import Path

from pdblend.profile.long_holdout_only import prepare,load_package
from pdblend.profile.power_calibration import write_immutable,digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.training is None:
        report=json.loads((ROOT/'results/2026-09-23/incremental-wave-closeout-v1/audit.json').read_text())
        args.training=Path(report['members']['14b-tp1-longctx']['root'])
    if args.out.exists():raise FileExistsError('use a new immutable job package directory')
    args.out.mkdir(parents=True)
    package=args.out/'package';manifest=prepare(training=args.training,out=package)
    _,_,plan=load_package(package)
    recipe=dict(schema=1,kind='14b_tp1_long_holdout_only',model_id=manifest['model_id'],system='pdblend',tp=1,pp=1,
        gpu_count=1,exclusive=False,max_attempts=2,timeout_s=3600,required_receipts=['completion.json'],
        argv_template=['python','-m','pdblend.profile.long_holdout_only','run','--package',str(package.resolve()),
            '--model','/models/Qwen2.5-14B-Instruct','--gpus','0','--base-port','{lease_port}','--out','{attempt_dir}'],
        readonly_roots=[str(args.training.resolve()),str(package.resolve())],
        package_manifest_sha256=digest(package/'manifest.json'),
        implementation_sha256=manifest['implementation_sha256'],
        source_files=['src/pdblend/profile/long_holdout_only.py'],
        requires_source_freeze=True,requires_fresh_concurrency_qualification=True,
        qualification_interface='ProfileWave.from_environment / qualify_external / measurement or run_existing caller protocol',
        measurement_scope=dict(kind='holdout_only',expected_training_points=0,reused_training_points=36,
            expected_holdout_points=24,exact_batches=[1,4],batch_interpolation_qualified=False,
            window_lower_bound_s=504,estimated_elapsed_s_range=[1140,1740]),
        enqueued=False,hardware_started=False,formal_eligible=False,energy_comparable=False)
    write_immutable(args.out/'job-recipe.json',recipe)
    write_immutable(args.out/'review.json',dict(status='cpu_preflight_passed',package=str(package.resolve()),
        candidate_sha256=manifest['candidate_sha256'],plan_sha256=manifest['plan_sha256'],
        recipe_sha256=digest(args.out/'job-recipe.json'),points=len(plan['points']),missing_points=plan['missing_points'],
        training_reused=True,new_training_points=0,formal_eligible=False,hardware_started=False))
    print(json.dumps(dict(package=str(package),recipe=str(args.out/'job-recipe.json'))))


if __name__=='__main__':main()
