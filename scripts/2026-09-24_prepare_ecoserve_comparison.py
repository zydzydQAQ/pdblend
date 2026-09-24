#!/usr/bin/env python3
"""Extend an immutable comparison manifest with scoped EcoServe inputs."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

from pdblend.bench.comparison_campaign import binding, group_points, load_bound
from pdblend.bench.comparison_ecoserve_inputs import validate_ecoserve_inputs
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest, write_new

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-base', type=Path, help='An immutable source snapshot to extend')
    p.add_argument('--overlay', action='append', default=[], help='Explicit src-relative implementation to overlay')
    p.add_argument('--readiness', type=Path, default=ROOT/'results/2026-09-23/resident-comparison-readiness/readiness.json')
    args = p.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    base_ref = binding(args.base); base = load_bound(base_ref)
    ready_ref = binding(args.readiness); ready = load_bound(ready_ref)
    execution = json.loads((args.base.parent/'execution-inputs.json').read_text())
    spec = importlib.util.spec_from_file_location('source_freezer', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(spec); spec.loader.exec_module(freezer)
    source_input = ROOT/'src'
    if args.source_base:
        frozen = json.loads((args.source_base/'manifest.json').read_text())
        freezer.verify_snapshot(args.source_base, frozen['files'])
        source_input = out/'staged-source'; source_input.mkdir()
        for name in frozen['files']:
            target = source_input/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.source_base/name,target)
        for name in args.overlay:
            target = source_input/name
            if not target.resolve().is_relative_to(source_input.resolve()):
                raise ValueError('overlay leaves the source tree')
            shutil.copyfile(ROOT/'src'/name,target)
        write_new(out/'source-extension.json',dict(base_manifest=binding(args.source_base/'manifest.json'),
            overlays={name:binding(ROOT/'src'/name) for name in args.overlay}))
    elif args.overlay:
        raise ValueError('--overlay requires --source-base')
    source, source_sha = freezer.freeze_source(source_input, out/'sources')
    if args.source_base:
        shutil.rmtree(source_input)
    source_ref = binding(source/'manifest.json'); files = load_bound(source_ref)['files']
    runtime = {k:v for k,v in files.items() if k.startswith(('pdblend_runtime/','pdblend/engine/'))}
    measurement = {k:v for k,v in files.items() if k.startswith('pdblend/measure/') or k in (
        'pdblend/bench/comparison_metrics.py','pdblend/bench/comparison_metering.py','pdblend/bench/client.py')}
    points, prepared, checks = deepcopy(base['points']), [], []
    for point in points:
        if point['system'] != 'ecoserve' or point['model_id'] not in ready['models'] or not point['trace']:
            continue
        evidence = ready['models'][point['model_id']]['ecoserve']
        mixed = next(p for p in base['points'] if p['model_id']==point['model_id'] and p['system']=='mixed' and p.get('engine_identity'))
        identity = deepcopy(mixed['engine_identity'])
        if (identity['runtime_source_sha256'] != digest(runtime)
                or identity['measurement_source_sha256'] != digest(measurement)):
            raise ValueError('public engine/measurement implementation differs from frozen baseline')
        for i, row in enumerate(identity['instances']):
            row['instance_id'] = f'eco{i}'
            row['launch_options']['kv_connector'] = 'P2pNcclConnector'
        config = dict(system='ecoserve',model_id=point['model_id'],model_path='/models/'+point['model_id'],
            model_verification_receipt=execution['model_verification']['path'],
            instances=[dict(id=r['instance_id'],gpus=[identity['fleet_gpu_uuids'].index(u) for u in r['gpu_uuids']],
                            tp=r['tp'],pp=r['pp']) for r in identity['instances']],
            eco_prefill_csv=evidence['profile_csv']['path'],eco_profile_sha256=evidence['profile_csv']['sha256'],
            slo_ttft_s=point['slo']['ttft_s'],slo_tpot_s=point['slo']['tpot_s'],
            eco_initial_instances=8,eco_macro_lower=2,eco_macro_upper=3,eco_scale_period_s=5.,
            eco_history_window_s=60.,eco_state_poll_s=.05,eco_drain_timeout_s=120.,request_timeout_s=180.,
            eco_active_frequency_mhz=2520,park_idle=True)
        config_path=out/'configs'/(point['name']+'.json'); write_new(config_path,config)
        inputs=dict(system_config=binding(config_path),trace=point['trace'],source_manifest=source_ref,
            eco_profile_source_manifest=ready['ecosystem_source_continuity']['profile']['source_manifest'],
            eco_mechanism_source_manifest=ready['ecosystem_source_continuity']['mechanism']['source_manifest'],
            **{'eco_'+k:evidence[v] for k,v in dict(profile_csv='profile_csv',profile_manifest='profile_manifest',
                mechanism_completion='mechanism_completion',automatic_completion='automatic_completion',
                mechanism_review='independent_mechanism_review').items()})
        point.update(inputs=inputs,engine_identity=identity,source_manifest=source_ref,revision=source_sha,
                     qualification_mode='ecoserve_native_bootstrap',blockers=[],status='prepared')
        checked=validate_ecoserve_inputs(point,identity,source_manifest=source_ref)
        checks.append(dict(point=point['name'],**checked))
        if not checked['preflight_ready']:
            point.update(status='blocked',blockers=checked['missing_gates'])
        else:
            prepared.append(point['name'])
    groups=group_points(points)
    current=dict(base,campaign_id=out.name,parent_campaign=base_ref,readiness=ready_ref,
        execution_campaigns=sorted(set(base.get('execution_campaigns',[])+[str(args.base.resolve())])),
        points=points,groups=groups,summary=dict(points=len(points),prepared_points=sum(not p['blockers'] for p in points),
            resident_sessions=len(groups),new_ecoserve_points=len(prepared),pure_service_s=150*len(points)))
    write_new(out/'campaign.json',current);write_new(out/'input-preflight.json',checks)
    jobs=[]
    for group in groups:
        if not all(p['name'] in prepared for p in group['points']):
            continue
        path=out/'groups'/(group['session_id']+'.json');write_new(path,group)
        jobs.append(resident_job(group,path,root=ROOT,source=source,image=execution['image_digest'],
            verification=execution['model_verification']['path'],campaign=out/'campaign.json',
            priority=801 if '7B-' in group['model_id'] else 799))
    write_new(out/'jobs.json',jobs)
    write_new(out/'execution-inputs.json',dict(execution,source=str(source),source_sha256=source_sha))
    print(json.dumps(dict(**current['summary'],jobs=len(jobs),out=str(out)),indent=2))


if __name__=='__main__':main()
