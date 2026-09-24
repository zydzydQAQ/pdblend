#!/usr/bin/env python3
"""Prepare a separate 32B four-TP2 EcoServe group from an immutable campaign; never enqueue."""
import argparse
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil

from pdblend.bench.comparison_campaign import binding, group_points, load_bound
from pdblend.bench.comparison_ecoserve32_inputs import validate_ecoserve_inputs
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest, engine_signature, file_sha, write_new

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERLAYS = ('pdblend/bench/comparison_runtime.py', 'pdblend/bench/comparison_ecoserve32_inputs.py',
                    'pdblend/bench/comparison_ecoserve32_acceptance.py')


def is_ecoserve32(point):
    return point.get('system') == 'ecoserve' and point.get('model_id') == 'Qwen2.5-32B-Instruct'


def frozen_points(base, previous):
    """Verify terminal prior sessions before changing any campaign point or source."""
    parents = {p['name']: p for p in base['points']}
    if len(parents) != len(base['points']):
        raise ValueError('parent campaign has duplicate point names')
    prepared = [p for p in base['points'] if is_ecoserve32(p) and (
        p.get('engine_identity') or p.get('qualification_mode') or p.get('metering_execution')
        or p.get('status') == 'prepared' or p.get('evidence_valid') or p.get('baseline_frozen'))]
    inherited = {name: ref for name, ref in base.get('preserved_baseline_receipts', {}).items()
                 if name in parents and is_ecoserve32(parents[name])}
    if not previous and (prepared or inherited):
        raise ValueError('EcoServe32 already has prepared/measurement identity; --previous is required')
    receipts, covered = {}, set()
    for prior in previous:
        prior = Path(prior).resolve()
        completion_path = prior/'completion.json'
        if not prior.is_dir() or not completion_path.is_file():
            raise ValueError('prior session must have a terminal completion.json: '+str(prior))
        completion = json.loads(completion_path.read_text())
        if completion.get('status') not in ('passed', 'failed') or not completion.get('finished_s'):
            raise ValueError('prior session has not terminated: '+str(prior))
        covered.add(completion.get('engine_signature'))
        recorded = {Path(row['path']).resolve(): row['sha256'] for row in completion.get('windows', [])}
        if len(recorded) != len(completion.get('windows', [])):
            raise ValueError('prior completion has duplicate window receipts')
        paths = set(prior.glob('windows/*/receipt.json'))
        if paths != set(recorded):
            raise ValueError('prior completion window inventory differs from receipt files')
        for receipt_path in sorted(paths):
            if file_sha(receipt_path) != recorded[receipt_path]:
                raise ValueError('prior receipt checksum differs: '+str(receipt_path))
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('evidence_valid') is not True:
                if receipt.get('baseline_frozen') is True:
                    raise ValueError('frozen baseline claims invalid evidence')
                continue
            if receipt.get('baseline_frozen') is not True or receipt.get('cleanup_passed') is not True:
                raise ValueError('valid baseline lacks its freeze/cleanup receipt')
            artifacts = receipt.get('artifacts', {})
            if not {'point.json', 'result.json'} <= artifacts.keys():
                raise ValueError('frozen baseline lacks bound point/result artifacts')
            for name, expected in artifacts.items():
                target = (receipt_path.parent/name).resolve()
                if (not target.is_relative_to(receipt_path.parent.resolve())
                        or not target.is_file() or file_sha(target) != expected):
                    raise ValueError('frozen baseline bytes changed: '+str(target))
            point = json.loads((receipt_path.parent/'point.json').read_text())
            result = json.loads((receipt_path.parent/'result.json').read_text())
            name = point.get('name')
            if (not is_ecoserve32(point) or receipt.get('point') != name
                    or receipt_path.parent.name != name or digest(point) != receipt.get('point_sha256')
                    or result != receipt.get('result') or result.get('evidence_valid') is not True
                    or engine_signature(point['engine_identity']) != receipt.get('engine_signature')
                    or receipt.get('engine_signature') != completion.get('engine_signature')):
                raise ValueError('frozen EcoServe32 point/result/engine identity differs')
            if parents.get(name) != point or name in receipts:
                raise ValueError('parent does not preserve a unique frozen EcoServe32 observation')
            receipts[name] = binding(receipt_path)
        # Recovery sessions can skip observations from an earlier session.
        # Require that original session too, so runtime --previous can find them.
        for row in completion.get('skipped', []):
            ref = row['frozen_receipt']
            if file_sha(ref['path']) != ref['sha256']:
                raise ValueError('skipped baseline receipt checksum differs')
            inherited.setdefault(row['point'], ref)
    for point in prepared:
        if engine_signature(point['engine_identity']) not in covered:
            raise ValueError('prepared EcoServe32 engine has no matching terminal --previous session')
    for name, ref in inherited.items():
        if receipts.get(name) != ref:
            raise ValueError('include the original --previous session for every preserved EcoServe32 point')
    return receipts


def planned_points(points, datasets, preserved):
    selected = [p for p in points if is_ecoserve32(p) and p['dataset'] in datasets and p.get('trace')]
    counts = {dataset: sum(p['dataset'] == dataset for p in selected) for dataset in datasets}
    if (len(selected) not in (8, 12) or any(n not in (0, 4) for n in counts.values())
            or any(counts.get(dataset) != 4 for dataset in ('alpaca', 'sharegpt'))
            or any({p['scale'] for p in selected if p['dataset'] == dataset} != {.25, .5, .75, 1.}
                   for dataset in datasets if counts[dataset])):
        raise ValueError('prepare complete four-scale datasets only: '+repr(counts))
    return [p['name'] for p in selected if p['name'] not in preserved]


def pending_groups(groups, prepared):
    return [g for g in groups if any(p['name'] in prepared for p in g['points'])]


def bind_previous(job, previous):
    for prior in previous:
        job['payload']['argv'] += ['--previous', str(Path(prior).resolve())]
    job['payload']['prior_sessions'] = [str(Path(p).resolve()) for p in previous]


def queued_replacement(base, queue_path, job_id, *, previous=()):
    """Read-only exception for an exactly bound, never-claimed twelve-point job.

    The scheduling owner must still atomically recheck the queue before replacing
    it. This preparation never supersedes a job or grants measurement validity.
    """
    if previous:
        raise ValueError('queued replacement cannot use executed --previous sessions')
    queue_path=Path(queue_path).resolve();raw=queue_path.read_bytes();queue=json.loads(raw)
    job=queue.get('jobs',{}).get(job_id,{})
    if (job.get('job_id')!=job_id or job.get('status')!='queued'
            or type(job.get('attempts')) is not int or job['attempts']!=0 or job.get('lease_id') is not None):
        raise ValueError('replacement requires an unclaimed queued job with attempts=0 and no lease')
    if any(lease.get('job_id')==job_id for lease in queue.get('leases',{}).values()):
        raise ValueError('queued replacement has active or historical lease evidence')
    attempt_root=queue_path.parent/'queue-attempts'
    if any((attempt_root/job_id).glob('attempt-*')):
        raise ValueError('queued replacement already has an attempt directory')
    points=[point for point in base['points'] if is_ecoserve32(point)]
    names={point['name'] for point in points}
    if (len(points)!=12 or len(names)!=12
            or set(planned_points(points,('alpaca','sharegpt','longbench'),{}))!=names):
        raise ValueError('queued replacement requires exactly twelve complete EcoServe32 points')
    if (any(p.get('evidence_valid') is True or p.get('baseline_frozen') is True for p in points)
            or names & set(base.get('preserved_baseline_receipts',{}))):
        raise ValueError('queued replacement cannot revise valid frozen EcoServe32 evidence')
    for receipt_path in attempt_root.glob('*/attempt-*/session/windows/32b-ecoserve-*/receipt.json'):
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('evidence_valid') is True or receipt.get('baseline_frozen') is True:
            raise ValueError('queued replacement found existing valid/frozen EcoServe32 receipt: '+str(receipt_path))
    payload=job.get('payload',{});argv=payload.get('argv',[])
    flags=[i for i,arg in enumerate(argv) if arg=='--group' or arg.startswith('--group=')]
    if (len(flags)!=1 or argv[flags[0]]!='--group' or flags[0]+1>=len(argv)
            or 'pdblend.bench.comparison_runtime' not in argv or '--previous' in argv
            or any(arg.startswith('--previous=') for arg in argv) or payload.get('prior_sessions')):
        raise ValueError('old queued payload lacks one original resident group invocation')
    group_ref=binding(Path(argv[flags[0]+1]));group=load_bound(group_ref)
    expected=[g for g in group_points(base['points']) if {p['name'] for p in g['points']}==names]
    if (len(expected)!=1 or group!=expected[0] or job_id!='comparison-32b-'+digest(group)[:16]
            or payload.get('session_id')!=group['session_id']
            or payload.get('system')!='ecoserve' or payload.get('model_id')!='Qwen2.5-32B-Instruct'
            or payload.get('gpu_count')!=8 or payload.get('exclusive') is not True
            or payload.get('reserve_host') is not True):
        raise ValueError('old queued group/payload does not bind the exact twelve parent points')
    if any(p.get('revision')!=payload.get('source_sha256') for p in group['points']):
        raise ValueError('old queued group source differs from payload')
    return dict(schema='queued-ecoserve32-replacement-preflight/v1',supersedes_job_id=job_id,
        queue_path=str(queue_path),queue_sha256_at_read=hashlib.sha256(raw).hexdigest(),
        old_job=deepcopy(job),old_job_sha256=digest(job),old_group=group_ref,
        point_sha256={p['name']:digest(p) for p in points},no_valid_frozen_ecoserve32_found=True,
        queue_modified=False,atomic_scheduling_recheck_required=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-base', type=Path, required=True, help='Reviewed immutable source to extend, including the accepted isolated-meter startup fix; public baseline bytes remain unchanged')
    p.add_argument('--execution-inputs', type=Path, help='Defaults beside --base')
    p.add_argument('--after-terminal', required=True, help='Existing 32B combined job; root owns scheduling')
    p.add_argument('--previous', type=Path, action='append', default=[],
                   help='Terminal prior EcoServe32 sessions; preserve and skip every valid point, including SLO failures')
    p.add_argument('--datasets', nargs='+', choices=['alpaca','sharegpt','longbench'], default=['alpaca','sharegpt','longbench'])
    p.add_argument('--overlay', action='append', default=None,
                   help='Explicit overlays replace the historical three-overlay default')
    p.add_argument('--replace-queued-job', help='Prepare a replacement only for the exactly bound unclaimed job')
    p.add_argument('--queue', type=Path, help='Read-only queue snapshot; required with --replace-queued-job')
    p.add_argument('--continue-frequency-rejections', action='store_true',
                   help='Keep frequency-only invalid observations unranked and reset before the next point')
    p.add_argument('--readiness', type=Path, default=ROOT/'results/2026-09-24/ecoserve32-readiness-v1/readiness.json')
    args = p.parse_args(argv)
    args.overlay = list(DEFAULT_OVERLAYS) if args.overlay is None else args.overlay
    if bool(args.replace_queued_job) != bool(args.queue):
        raise ValueError('--replace-queued-job and --queue must be supplied together')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not {'alpaca','sharegpt'} <= set(args.datasets):
        raise ValueError('prepare the eight confirmed Alpaca/ShareGPT points together')
    base_ref = binding(args.base); base = load_bound(base_ref)
    replacement = (queued_replacement(base,args.queue,args.replace_queued_job,previous=args.previous)
                   if args.replace_queued_job else None)
    if replacement and set(args.datasets)!={'alpaca','sharegpt','longbench'}:
        raise ValueError('queued replacement must retain all twelve points across three datasets')
    preserved = {} if replacement else frozen_points(base, args.previous)
    pending = set(planned_points(base['points'], args.datasets, preserved))
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    if replacement:write_new(out/'queued-replacement-preflight.json',replacement)
    ready_ref = binding(args.readiness); ready = load_bound(ready_ref)
    execution_ref = binding(args.execution_inputs or args.base.parent/'execution-inputs.json')
    execution = load_bound(execution_ref)
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
            target.parent.mkdir(parents=True,exist_ok=True)
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
        if point['name'] not in pending:
            continue
        evidence = ready['models'][point['model_id']]['ecoserve']
        mixed = next(p for p in base['points'] if p['model_id']==point['model_id'] and p['system']=='mixed' and p.get('engine_identity'))
        identity = deepcopy(mixed['engine_identity'])
        identity['metering_execution'] = 'isolated_process'
        if len(identity['instances']) != 4 or any(r['tp'] != 2 or r['pp'] != 1 for r in identity['instances']):
            raise ValueError('32B comparison requires exactly four TP2/PP1 engines')
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
            eco_initial_instances=4,eco_macro_lower=2,eco_macro_upper=3,eco_scale_period_s=5.,
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
                     qualification_mode='ecoserve_native_bootstrap',metering_execution='isolated_process',blockers=[],status='prepared')
        if args.continue_frequency_rejections:
            point['observation_failure_policy']='continue_after_verified_frequency_rejection'
        checked=validate_ecoserve_inputs(point,identity,source_manifest=source_ref)
        checks.append(dict(point=point['name'],**checked))
        if not checked['preflight_ready']:
            raise ValueError('32B input qualification failed for '+point['name']+': '+json.dumps(checked['gate_failures']))
        else:
            prepared.append(point['name'])
    if set(prepared) != pending:
        raise ValueError('prepared point inventory differs from the verified recovery plan')
    groups=group_points(points)
    write_new(out/'execution-inputs.json',dict(execution,source=str(source),source_sha256=source_sha))
    current=dict(base,campaign_id=out.name,parent_campaign=base_ref,readiness=ready_ref,
        execution_inputs=binding(out/'execution-inputs.json'),execution_source_manifest=source_ref,
        parent_execution_inputs=execution_ref,parent_execution_source_manifest=base.get('execution_source_manifest'),
        execution_campaigns=sorted(set(base.get('execution_campaigns',[])+[str(args.base.resolve())])),
        preserved_baseline_receipts={**base.get('preserved_baseline_receipts', {}), **preserved},
        points=points,groups=groups,summary=dict(points=len(points),prepared_points=sum(not p['blockers'] for p in points),
            resident_sessions=len(groups),new_ecoserve_points=len(prepared),preserved_ecoserve32_points=len(preserved),
            pure_service_s=150*len(points)))
    write_new(out/'campaign.json',current);write_new(out/'input-preflight.json',checks)
    jobs=[]
    for group in pending_groups(groups, prepared):
        path=out/'groups'/(group['session_id']+'.json');write_new(path,group)
        job=resident_job(group,path,root=ROOT,source=source,image=execution['image_digest'],
            verification=execution['model_verification']['path'],campaign=out/'campaign.json',priority=797)
        job['payload']['after_terminal']=[args.after_terminal]
        job['payload']['model_id']='Qwen2.5-32B-Instruct'
        job['payload']['system']='ecoserve'
        bind_previous(job, args.previous)
        if replacement:job['payload']['supersedes_job_id']=args.replace_queued_job
        jobs.append(job)
    if replacement and len(jobs)!=1:
        raise ValueError('one queued twelve-point group must produce exactly one replacement job')
    write_new(out/'jobs.json',jobs)
    print(json.dumps(dict(**current['summary'],jobs=len(jobs),out=str(out)),indent=2))


if __name__=='__main__':main()
