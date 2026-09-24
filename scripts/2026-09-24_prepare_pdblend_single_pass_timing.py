#!/usr/bin/env python3
"""Freeze two single-pass development timing jobs; never enqueue or run GPUs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pdblend.profile.collection.native_timing_plan import binding, read_bound
from pdblend.profile.collection.native_timing_single_pass import build_plan, LEVEL


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def prepare(out, parent_plans, *, after_terminal):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('new immutable package directory required')
    if set(parent_plans) != {'7b', '14b'} or not isinstance(after_terminal, str) or not after_terminal:
        raise ValueError('explicit 7B/14B parents and one predecessor job required')
    plans = {size: build_plan(binding(path)) for size, path in parent_plans.items()}
    if any(plan['model_id'] != f'Qwen2.5-{size.upper()}-Instruct' for size, plan in plans.items()):
        raise ValueError('parent plan model differs from package key')
    freezer = module('single_pass_source_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    builder = module('single_pass_native_builder', ROOT/'scripts/2026-09-24_prepare_pdblend_native_timing.py')
    out.mkdir(parents=True)
    source, source_sha = freezer.freeze_source(ROOT/'src', out/'base-sources')
    rows = []; jobs = []
    for size in ('7b', '14b'):
        plan = plans[size]
        path = out/size/'single-pass-plan.json'
        write(path, plan)
        report = builder.prepare(out/size/'collection', plan['query_ledger']['path'],
            plan['query_provenance']['path'], point_plan=path, source_base=source)
        prepared_source = read_bound(report['source_manifest'])
        if prepared_source['source_sha256'] != source_sha:
            raise ValueError('workspace changed while freezing full source; prepare a new package')
        job = read_bound(report['jobs'])[0]
        job.update(priority=1000, depends_on=[])
        job['payload']['after_terminal'] = [after_terminal]
        job['payload'].update(qualification_level=LEVEL, original_design_qualified=False,
                              component_qualified=False, parallel_qualified=False)
        jobs.append(job)
        rows.append(dict(model=size, parent_point_plan=plan['parent_point_plan'],
            point_plan=report['point_plan'], input_manifest=job['payload']['input_manifest'],
            source_manifest=report['source_manifest'], job_id=job['job_id'],
            training_points=report['training_points'], holdout_points=report['holdout_points'],
            unique_measurement_windows=report['measurement_windows'], interference_windows=0,
            theoretical_balanced_sampling_floor_s=report['minimum_sampling_wall_s'],
            duration_excludes_model_load_warmup_cleanup_and_scheduling=True))
    write(out/'jobs.json', jobs)
    report = dict(schema='pdblend-single-pass-timing-package/v1', qualification_level=LEVEL,
        enqueued=False, hardware_executed=False, formal_eligible=False,
        full_profile_qualified=False, component_qualified=False, parallel_qualified=False,
        original_design_qualified=False, source_manifest=binding(source/'manifest.json'),
        source_sha256=source_sha, jobs=binding(out/'jobs.json'), models=rows,
        after_terminal=[after_terminal], reused_existing_domains=['legacy_model_curves',
            'native_power_component_raw', 'native_runtime_component_raw'],
        reuse_semantics='existing observations retained under their original qualification; no recollection or CUDA relabeling',
        uncollected=['repeatability', 'isolated_vs_concurrent_interference'],
        interpretation='single observed CUDA window per unique predeclared shape; independent holdout diagnostics only',
        builder=binding(__file__))
    write(out/'manifest.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent-7b', type=Path, required=True)
    parser.add_argument('--parent-14b', type=Path, required=True)
    parser.add_argument('--after-terminal', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, {'7b': args.parent_7b, '14b': args.parent_14b},
                             after_terminal=args.after_terminal), indent=2))
