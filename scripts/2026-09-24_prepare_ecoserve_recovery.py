#!/usr/bin/env python3
"""Prepare a new EcoServe attempt with isolated sampling; preserve prior evidence."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

from pdblend.bench.comparison_campaign import binding, group_points, load_bound
from pdblend.bench.comparison_ecoserve_inputs import (validate_ecoserve_inputs,
    LIFECYCLE_MODE,LIFECYCLE_REVIEW_SHA256)
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest, write_new, file_sha

ROOT = Path(__file__).resolve().parents[1]


def lifecycle_review(path):
    if path is None:return None
    ref=binding(path);review=load_bound(ref)
    if (ref['sha256']!=LIFECYCLE_REVIEW_SHA256 or review.get('mode')!=LIFECYCLE_MODE
            or review.get('hardware_qualified') is not False):
        raise ValueError('unreviewed EcoServe comparison lifecycle')
    return ref


def unattempted_selection(base, sessions, queue_path):
    """Bind terminal ownership and exclude both valid and invalid observations."""
    queue=json.loads(Path(queue_path).read_text()); current={p['name']:p for p in base['points']}
    selected={}; proofs=[]; candidates={};observed=set();directories=set()
    for directory in sessions:
        directory=Path(directory).resolve();attempt=directory.parent
        if directory in directories:raise ValueError('duplicate terminal session selection')
        directories.add(directory)
        manifest_ref=binding(attempt/'manifest.json');manifest=load_bound(manifest_ref)
        execution_ref=binding(attempt/'execution.json');execution=load_bound(execution_ref)
        completion_ref=binding(directory/'completion.json');report=load_bound(completion_ref)
        job=queue['jobs'][manifest['job_id']];lease=queue['leases'][manifest['lease_id']]
        if (manifest.get('immutable') is not True or job['status'] not in ('failed','succeeded')
                or job.get('lease_id') is not None or job['payload']!=manifest['payload']
                or lease['status']!=job['status'] or lease['job_id']!=job['job_id']
                or Path(lease['attempt_dir']).resolve()!=attempt
                or execution.get('status') not in ('failed','passed')
                or execution.get('finished_s',-1)<report.get('finished_s',float('inf'))):
            raise ValueError('unattempted selection requires a released terminal queue attempt')
        if (report.get('schema')!='resident-group-session/v1' or report.get('status') not in ('failed','passed')
                or report.get('cleanup_errors') or report.get('cleanup',{}).get('passed') is not True
                or report['cleanup'].get('process_cleanup_verified') is not True):
            raise ValueError('unattempted selection requires verified terminal session cleanup')
        argv=manifest['payload']['argv'];group_ref=binding(argv[argv.index('--group')+1]);group=load_bound(group_ref)
        planned={p['name']:digest(p) for p in group['points']}
        expected_job='comparison-'+group['model_id'].split('-')[1].lower()+'-'+digest(group)[:16]
        if (manifest['job_id']!=expected_job or job['job_id']!=expected_job
                or manifest['payload'].get('container_name')!=expected_job
                or manifest['payload'].get('session_id')!=group['session_id']):
            raise ValueError('terminal resident job identity does not bind the actual group')
        if (report['group_sha256']!=digest(group) or report['planned_points']!=planned
                or group['session_id']!=report['session_id']
                or any(p['system']!='ecoserve' for p in group['points'])):
            raise ValueError('terminal group/point inventory differs')
        seen=set()
        for row in report['windows']:
            name=row['point'];ref={k:row[k] for k in ('path','sha256')}
            expected=directory/'windows'/name/'receipt.json'
            if Path(ref['path']).resolve()!=expected.resolve() or name not in planned or name in seen:
                raise ValueError('terminal window inventory differs')
            receipt=load_bound(ref)
            if receipt['point_sha256']!=planned[name]:raise ValueError('terminal receipt point binding differs')
            seen.add(name)
        actual={p.parent.name for p in (directory/'windows').glob('*/receipt.json')}
        if actual!=seen:raise ValueError('terminal report omits an actual window')
        for row in report['skipped']:
            name=row['point'];receipt=load_bound(row['frozen_receipt'])
            if (name not in planned or name in seen or receipt['point_sha256']!=planned[name]
                    or receipt.get('evidence_valid') is not True or receipt.get('baseline_frozen') is not True):
                raise ValueError('terminal skipped baseline proof differs')
            seen.add(name)
        remaining=set(planned)-seen
        for name in remaining:
            candidates.setdefault(name,set()).add(planned[name])
        observed.update(seen)
        proofs.append(dict(session=completion_ref,attempt=manifest_ref,execution=execution_ref,group=group_ref,
            job_id=job['job_id'],terminal_status=job['status'],lease_released=True,
            attempted_or_skipped=sorted(seen),unattempted=sorted(remaining)))
    for name,hashes in candidates.items():
        if name in observed:continue
        if len(hashes)!=1 or digest(current.get(name)) not in hashes:
            raise ValueError('unattempted point is inconsistent or changed in parent campaign')
        selected[name]=next(iter(hashes))
    return dict(schema='ecoserve-unattempted-only-selection/v1',points=selected,sessions=proofs,
                invalid_observations_retried=False,frozen_baselines_retried=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execution-inputs', type=Path,
                   help='Explicit execution inputs when the parent is a verified combined-session campaign')
    p.add_argument('--source-base', type=Path, help='An immutable source snapshot to extend')
    p.add_argument('--overlay', action='append', default=[], help='Explicit src-relative implementation to overlay')
    p.add_argument('--isolated-meter', action='store_true', help='Use the unchanged sampler in an independent process')
    p.add_argument('--models',nargs='+',choices=['7b','14b'],default=['7b','14b'],
                   help='Only prepare terminal/ready model groups; do not revise an in-flight model')
    p.add_argument('--continue-frequency-rejections',action='store_true',
                   help='Keep invalid clock evidence unranked and require a fresh verified reset before the next point')
    p.add_argument('--lifecycle-review',type=Path,
                   help='Explicit reviewed comparison-only cancellation and serial-close wrapper')
    p.add_argument('--previous', type=Path, action='append', required=True,
                   help='Prior EcoServe session roots; all valid baseline points are preserved and skipped')
    p.add_argument('--unattempted-from',type=Path,action='append',default=[],
                   help='Only execute points absent from these verified terminal session windows/skips')
    p.add_argument('--queue',type=Path,default=ROOT/'results/2026-09-22/three-model/queue.json')
    p.add_argument('--readiness', type=Path, default=ROOT/'results/2026-09-23/resident-comparison-readiness/readiness.json')
    args = p.parse_args()
    lifecycle_ref=lifecycle_review(args.lifecycle_review)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    base_ref = binding(args.base); base = load_bound(base_ref)
    selection=None
    if args.unattempted_from:
        if not {p.resolve() for p in args.unattempted_from}<={p.resolve() for p in args.previous}:
            raise ValueError('unattempted sessions must also be supplied as --previous')
        selection=unattempted_selection(base,args.unattempted_from,args.queue)
        write_new(out/'unattempted-selection.json',selection)
    frozen_receipts = {}
    for prior in args.previous:
        if not prior.is_dir():
            raise ValueError('prior session directory missing: '+str(prior))
        for receipt_path in prior.glob('windows/*/receipt.json'):
            receipt=json.loads(receipt_path.read_text())
            if not (receipt.get('evidence_valid') is True and receipt.get('baseline_frozen') is True
                    and receipt.get('cleanup_passed') is True):
                continue
            artifacts=receipt.get('artifacts',{})
            if not artifacts or 'point.json' not in artifacts or 'result.json' not in artifacts:
                raise ValueError('frozen baseline lacks bound artifacts')
            for name,expected in artifacts.items():
                target=(receipt_path.parent/name).resolve()
                if not target.is_relative_to(receipt_path.parent.resolve()) or file_sha(target)!=expected:
                    raise ValueError('frozen baseline bytes changed: '+str(target))
            point=json.loads((receipt_path.parent/'point.json').read_text())
            result=json.loads((receipt_path.parent/'result.json').read_text())
            if (point.get('system')!='ecoserve' or digest(point)!=receipt['point_sha256']
                    or result!=receipt.get('result') or result.get('evidence_valid') is not True):
                raise ValueError('frozen EcoServe point/result identity differs')
            parent=next((p for p in base['points'] if p['name']==point['name']),None)
            if parent!=point or point['name'] in frozen_receipts:
                raise ValueError('parent does not preserve a unique frozen EcoServe observation')
            frozen_receipts[point['name']]=binding(receipt_path)
    ready_ref = binding(args.readiness); ready = load_bound(ready_ref)
    execution = json.loads((args.execution_inputs or args.base.parent/'execution-inputs.json').read_text())
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
        if point['model_id'].split('-')[1].lower() not in args.models:
            continue
        if point['name'] in frozen_receipts:
            continue
        if selection is not None and point['name'] not in selection['points']:
            continue
        evidence = ready['models'][point['model_id']]['ecoserve']
        mixed = next(p for p in base['points'] if p['model_id']==point['model_id'] and p['system']=='mixed' and p.get('engine_identity'))
        identity = deepcopy(mixed['engine_identity'])
        if (identity['runtime_source_sha256'] != digest(runtime)
                or identity['measurement_source_sha256'] != digest(measurement)):
            raise ValueError('public engine/measurement implementation differs from frozen baseline')
        if args.isolated_meter:
            identity['metering_execution'] = 'isolated_process'
        if lifecycle_ref:identity['eco_comparison_lifecycle']=LIFECYCLE_MODE
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
        if lifecycle_ref:config['eco_comparison_lifecycle']=LIFECYCLE_MODE
        config_path=out/'configs'/(point['name']+'.json'); write_new(config_path,config)
        inputs=dict(system_config=binding(config_path),trace=point['trace'],source_manifest=source_ref,
            eco_profile_source_manifest=ready['ecosystem_source_continuity']['profile']['source_manifest'],
            eco_mechanism_source_manifest=ready['ecosystem_source_continuity']['mechanism']['source_manifest'],
            **{'eco_'+k:evidence[v] for k,v in dict(profile_csv='profile_csv',profile_manifest='profile_manifest',
                mechanism_completion='mechanism_completion',automatic_completion='automatic_completion',
                mechanism_review='independent_mechanism_review').items()})
        point.update(inputs=inputs,engine_identity=identity,source_manifest=source_ref,revision=source_sha,
                     qualification_mode='ecoserve_native_bootstrap',blockers=[],status='prepared')
        if lifecycle_ref:
            point['eco_comparison_lifecycle']=LIFECYCLE_MODE
            inputs['eco_lifecycle_review']=lifecycle_ref
        if args.isolated_meter:
            point['metering_execution'] = 'isolated_process'
        if args.continue_frequency_rejections:
            point['observation_failure_policy']='continue_after_verified_frequency_rejection'
        checked=validate_ecoserve_inputs(point,identity,source_manifest=source_ref)
        checks.append(dict(point=point['name'],**checked))
        if not checked['preflight_ready']:
            point.update(status='blocked',blockers=checked['missing_gates'])
        else:
            prepared.append(point['name'])
    groups=group_points([p for p in points if p['name'] in prepared] if selection is not None else points)
    current=dict(base,campaign_id=out.name,parent_campaign=base_ref,readiness=ready_ref,
        execution_source_manifest=source_ref,
        inherited_execution_source_manifest=base.get('execution_source_manifest'),
        execution_campaigns=sorted(set(base.get('execution_campaigns',[])+[str(args.base.resolve())])),
        points=points,groups=groups,preserved_baseline_receipts=frozen_receipts,
        summary=dict(points=len(points),prepared_points=sum(not p['blockers'] for p in points),
            resident_sessions=len(groups),new_ecoserve_points=len(prepared),pure_service_s=150*len(points)))
    if selection is not None:current['unattempted_selection']=binding(out/'unattempted-selection.json')
    write_new(out/'campaign.json',current);write_new(out/'input-preflight.json',checks)
    jobs=[]
    for group in groups:
        if not any(p['name'] in prepared for p in group['points']):
            continue
        path=out/'groups'/(group['session_id']+'.json');write_new(path,group)
        job=resident_job(group,path,root=ROOT,source=source,image=execution['image_digest'],
            verification=execution['model_verification']['path'],campaign=out/'campaign.json',
            priority=801 if '7B-' in group['model_id'] else 799)
        for prior in args.previous:
            job['payload']['argv']+=['--previous',str(prior.resolve())]
        job['payload']['prior_sessions']=[str(p.resolve()) for p in args.previous]
        jobs.append(job)
    write_new(out/'jobs.json',jobs)
    write_new(out/'execution-inputs.json',dict(execution,source=str(source),source_sha256=source_sha))
    print(json.dumps(dict(**current['summary'],jobs=len(jobs),out=str(out)),indent=2))


if __name__=='__main__':main()
