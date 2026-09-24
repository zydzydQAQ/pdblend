#!/usr/bin/env python3
"""Predeclare exactly three independent PD/four-baseline repeats; never enqueue.

Only 0--3% initial all-baseline wins are repeated. Every condition keeps the
original trace, policy, startup inputs, image and actual resident job template.
Diagnostics and retries cannot replace any of the three declared observations.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from pdblend.bench.cohort_dominance import (BASELINE_SYSTEMS, declare_paired_repeats,
    evaluate_declared_repeats, identity, repeat_reference)
from pdblend.bench.comparison_campaign import binding,load_bound
from pdblend.bench.measurement_compatibility import hydrate_receipt_evidence,load_compatibility
from pdblend.bench.resident_session import digest,write_new


def helper(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def canonical_receipt(path):
    path=Path(path).resolve();ref=binding(path);receipt=load_bound(ref)
    point=load_bound(dict(path=str(path.parent/'point.json'),sha256=receipt['artifacts']['point.json']))
    if digest(point)!=receipt['point_sha256'] or receipt['point']!=point['name']:
        raise ValueError('receipt and immutable executed point differ')
    load_bound(point['trace']);source=load_bound(point['source_manifest'])
    if source['source_sha256']!=point['revision']:raise ValueError('executed source and point revision differ')
    engine=point['engine_identity'];metrics=receipt['result']['metrics']
    if point['slo']!={'ttft_s':metrics['slo_ttft_s'],'tpot_s':metrics['slo_tpot_s']}:
        raise ValueError('measured SLO thresholds differ from original frozen point')
    raw=dict(metrics,system=point['system'],model_id=point['model_id'],dataset=point['dataset'],
        rate_scale=point['scale'],offered_rps=point['rate_rps'],seed=point['seed'],duration_s=point['duration_s'],
        trace_sha256=point['trace']['sha256'],revision=point['revision'],point_id=point['name'],
        receipt_path=str(path),receipt_sha256=ref['sha256'],
        measurement_protocol_version=point['measurement_protocol_version'],
        evidence_valid=receipt['result'].get('evidence_valid'),formal_eligible=receipt['result'].get('formal_eligible'))
    raw.update({key:engine[key] for key in ('model_hash','tokenizer_hash','image_digest',
        'runtime_source_sha256','measurement_source_sha256')})
    reducer=helper(ROOT/'scripts/report_cohort_energy.py','paired_energy_canonical_reducer')
    series='pd_repeat_'+point['revision'][:12] if point['system']=='pdblend' else point['system']
    row,_,provenance=reducer.analyze_point(dict(ref,series=series,receipt_sha256=ref['sha256']),raw)
    row=hydrate_receipt_evidence(row)
    if point.get('paired_repeat'):row['repeat_id']=point['paired_repeat']['repeat_id']
    return row,point,provenance


def verify_source(ref):
    manifest=load_bound(ref);source=Path(ref['path']).parent.resolve()
    if manifest['source_sha256']!=source.name or digest(manifest['files'])!=source.name:
        raise ValueError('template source is not its content-addressed frozen inventory')
    for name,sha in manifest['files'].items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('template frozen source bytes differ')


def template_catalog(job_files):
    catalog={};refs=[]
    for path in job_files:
        ref=binding(Path(path));refs.append(ref)
        for job in load_bound(ref):
            payload=job['payload'];argv=payload['argv']
            if (argv.count('--group')!=1 or payload.get('gpu_count')!=8 or payload.get('exclusive') is not True
                    or payload.get('reserve_host') is not True or payload.get('required_receipts')!=['session/completion.json']):
                raise ValueError('repeat support requires the real exclusive eight-GPU resident job template')
            group_ref=binding(Path(argv[argv.index('--group')+1]));group=load_bound(group_ref)
            if payload['session_id']!=group['session_id']:raise ValueError('template job/group identity differs')
            for point in group['points']:
                key=(point['name'],point['revision']);entry=dict(job=job,group=group,group_ref=group_ref,jobs_ref=ref,point=point)
                if key in catalog and catalog[key]!=entry:raise ValueError('ambiguous real job template for '+point['name'])
                catalog[key]=entry
    return catalog,refs


def replace_arg(argv,name,value):
    argv=list(argv)
    if argv.count(name)!=1:raise ValueError('template needs exactly one '+name)
    argv[argv.index(name)+1]=value;return argv


def actual_execution_claim(receipt_path):
    receipt_path=Path(receipt_path).resolve()
    for parent in receipt_path.parents:
        path=parent/'manifest.json'
        if not path.is_file():continue
        value=json.loads(path.read_text())
        if value.get('schema')==1 and value.get('immutable') is True and isinstance(value.get('payload'),dict) and value.get('job_id'):
            return binding(path),value
    raise ValueError('selected receipt lacks its actual immutable execution claim')


def verify_executed_template(template,reference):
    claim_ref,claim=actual_execution_claim(reference['receipt_path'])
    expected=deepcopy(claim['payload']);provided=deepcopy(template['job']['payload'])
    # Scheduling dependencies can be newly declared; all actual execution
    # fields, including command/module/env/mounts, must remain identical.
    for payload in (expected,provided):
        payload.pop('depends_on',None);payload.pop('after_terminal',None)
    if claim['job_id']!=template['job']['job_id'] or expected!=provided:
        raise ValueError('job template differs from the actual original execution payload')
    return claim_ref


def clone_observation(template,reference,*,design_ref,condition_id,repeat,output,index,predecessor):
    original=template['point'];point=deepcopy(original)
    if original['name']!=reference['point_id'] or original['revision']!=reference['revision']:
        raise ValueError('repeat template differs from preselected observation')
    execution_claim=verify_executed_template(template,reference)
    verify_source(original['source_manifest'])
    payload=template['job']['payload'];argv=payload['argv']
    if (payload['source_sha256']!=original['revision'] or payload['image_digest']!=original['engine_identity']['image_digest']
            or template['group']['engine_identity']!=original['engine_identity']):
        raise ValueError('template changed the independent engine, image or source')
    manifest_args=[a.split('=',1)[1] for a in argv if a.startswith('PDBLEND_SOURCE_MANIFEST=')]
    if manifest_args!=[original['source_manifest']['path']]:raise ValueError('template source environment differs')
    # Startup/profile/config bindings are reused verbatim, including each
    # baseline's independent implementation and metering adapter.
    for name in ('system_config','offline_choice','planning_trace','trace'):
        if name in original.get('inputs',{}):load_bound(original['inputs'][name])
    for ref in original.get('inputs',{}).get('profiles',[]):load_bound(ref)
    point.update(name=original['name']+f'-paired-r{repeat}-'+design_ref['sha256'][:10],
        run_id=Path(output).name,status='prepared',blockers=[],formal_eligible=False,
        result_policy='all_recorded_windows/v1')
    point['paired_repeat']=dict(design=design_ref,condition_id=condition_id,repeat_id=repeat,
        original_point_sha256=digest(original),original_receipt=reference['receipt_path'],
        startup='one_fresh_resident_session_same_frozen_startup_inputs',retry_substitution_allowed=False)
    group=deepcopy(template['group']);group['points']=[point]
    group['session_id']='paired-'+digest(dict(point=point,design=design_ref))[:24]
    group_path=Path(output)/'groups'/(group['session_id']+'.json');write_new(group_path,group)
    point_path=Path(output)/'points'/(point['name']+'.json');write_new(point_path,point)
    job=deepcopy(template['job']);job_id='paired-'+digest(group)[:24]
    job.update(job_id=job_id,priority=600,max_attempts=1)
    target=job['payload'];new_argv=replace_arg(argv,'--group',str(group_path.resolve()))
    target.update(argv=replace_arg(new_argv,'--name',job_id),container_name=job_id,session_id=group['session_id'],
        comparison_campaign=str((Path(output)/'campaign.json').resolve()),depends_on=[],
        after_terminal=[predecessor] if predecessor else [],paired_repeat_design=design_ref,
        paired_repeat_id=repeat,paired_repeat_condition=condition_id)
    plan=dict(condition_id=condition_id,repeat_id=repeat,system=point['system'],point_id=point['name'],
        revision=point['revision'],comparison_identity=reference['comparison_identity'],point=binding(point_path),
        group=binding(group_path),template_jobs=template['jobs_ref'],template_group=template['group_ref'],
        original_execution_claim=execution_claim,job_id=job_id,sequence=index,original_receipt=reference['receipt_path'])
    return point,group,job,plan


def prepare(points_path,job_files,out,*,candidate_revision,compatibility=(),after_terminal=None,driver_serialized=False):
    out=Path(out).resolve()
    if out.exists():raise ValueError('repeat preparation requires a new immutable output directory')
    snapshot_ref=binding(Path(points_path));snapshot=load_bound(snapshot_ref)
    # Recompute the initial comparison from the selected raw receipts, without
    # choosing a different attempt because its SLO or energy looks better.
    initial=[];originals={};provenance=[]
    for source_row in snapshot:
        if source_row.get('system') not in BASELINE_SYSTEMS and not (
                source_row.get('system')=='pdblend' and source_row.get('revision')==candidate_revision):continue
        path=Path(source_row['receipt_path'])
        if binding(path)['sha256']!=source_row.get('receipt_sha256'):raise ValueError('initial snapshot receipt changed')
        row,point,proof=canonical_receipt(path);initial.append(row);provenance.append(proof)
        originals[(point['name'],point['revision'])]=point
    reviews=[load_compatibility(Path(path)) for path in compatibility]
    protocol=declare_paired_repeats(initial,candidate_revision=candidate_revision,declared_at_s=time.time(),
                                    measurement_compatibility=reviews)
    catalog,job_refs=template_catalog(job_files)
    out.mkdir(parents=True,exist_ok=False)
    design=dict(protocol,initial_points=snapshot_ref,template_jobs=job_refs,
        measurement_compatibility=[review['manifest_binding'] for review in reviews],
        system_order=[list(order) for order in (('pdblend',*BASELINE_SYSTEMS),
                     (*BASELINE_SYSTEMS,'pdblend'),('dynamollm','ecoserve','pdblend','mixed','distserve'))],
        queue_order='one_job_after_prior_released_lease_including_blocked' if driver_serialized else 'after_terminal_chain',
        hardware_executed=False,enqueued=False)
    write_new(out/'design.json',design);design_ref=binding(out/'design.json')
    points=[];groups=[];jobs=[];plans=[];previous=after_terminal
    for case in design['conditions']:
        if not case['repeat_required']:continue
        for repeat in (1,2,3):
            for system in design['system_order'][repeat-1]:
                reference=case['candidate'] if system=='pdblend' else case['baselines'][system]
                key=(reference['point_id'],reference['revision']);template=catalog.get(key)
                if not template or template['point']!=originals[key]:raise ValueError('real executed point/job template not supplied: '+str(key))
                point,group,job,plan=clone_observation(template,reference,design_ref=design_ref,
                    condition_id=case['condition_id'],repeat=repeat,output=out,index=len(jobs),predecessor=previous)
                if driver_serialized:
                    job['payload']['after_terminal']=[after_terminal] if not jobs and after_terminal else []
                    job['payload']['predeclared_driver_sequence']=len(jobs)
                points.append(point);groups.append(group);jobs.append(job);plans.append(plan);previous=job['job_id']
    protocol.update(design=design_ref,planned_observations=plans)
    # Validate declaration completeness now, before any job can run.
    evaluate_declared_repeats(protocol,[],measurement_compatibility=reviews)
    write_new(out/'protocol.json',protocol);write_new(out/'jobs.json',jobs)
    write_new(out/'campaign.json',dict(scope='predeclared_same_trace_paired_energy_repeats/v1',
        design=design_ref,protocol=binding(out/'protocol.json'),points=points,groups=groups,formal_eligible=False))
    write_new(out/'initial-raw-provenance.json',provenance)
    write_new(out/'preparation.json',dict(design=design_ref,protocol=binding(out/'protocol.json'),
        jobs=binding(out/'jobs.json'),campaign=binding(out/'campaign.json'),preparer=binding(Path(__file__)),
        conditions=36,repeated_conditions=sum(case['repeat_required'] for case in design['conditions']),
        observations=len(points),hardware_executed=False,enqueued=False,container_preflight_pending=True,
        goal_complete=False))
    return dict(conditions=36,repeated_conditions=sum(case['repeat_required'] for case in design['conditions']),jobs=len(jobs),enqueued=False)


def evaluate(package,queue_path,out):
    package=Path(package).resolve();protocol_ref=binding(package/'protocol.json');protocol=load_bound(protocol_ref)
    design=load_bound(protocol['design']);reviews=[load_compatibility(ref['path']) for ref in design['measurement_compatibility']]
    for key in ('schema','declared_at_s','candidate_revision','repeat_ids','conditions','selection_rule','acceptance_rule','energy_scope'):
        if protocol.get(key)!=design.get(key):raise ValueError('protocol target differs from its bound predeclared design: '+key)
    for ref in design['measurement_compatibility']:load_bound(ref)
    plans={plan['point_id']:plan for plan in protocol['planned_observations']};rows=[];refused=[];proofs=[]
    preparation=load_bound(binding(package/'preparation.json'))
    if preparation['protocol']!=protocol_ref:raise ValueError('preparation protocol was replaced')
    specs={job['job_id']:job for job in load_bound(preparation['jobs'])}
    queue_ref=binding(Path(queue_path));queue=load_bound(queue_ref) # No reclaim, probing or mutation.
    for name,plan in plans.items():
        try:
            spec=specs[plan['job_id']];job=queue.get('jobs',{}).get(plan['job_id'])
            leases=[lease for lease in queue.get('leases',{}).values() if lease.get('job_id')==plan['job_id']]
            if (not job or job.get('payload')!=spec['payload'] or job.get('max_attempts')!=1
                    or len(leases)!=1 or leases[0].get('attempt')!=1):
                raise ValueError('missing or repeated lease; diagnostic/retry cannot replace predeclared attempt')
            lease=leases[0]
            if job.get('status')!='succeeded' or job.get('lease_id') is not None or lease.get('status')!='succeeded':
                raise ValueError('declared lease did not successfully complete and release')
            attempt=Path(lease['attempt_dir']);attempt_ref=binding(attempt/'manifest.json');actual=load_bound(attempt_ref)
            if (actual.get('job_id')!=plan['job_id'] or actual.get('attempt')!=1 or actual.get('payload')!=spec['payload']
                    or sorted(actual.get('gpu_uuids',[]))!=plan['comparison_identity']['gpu_uuids']):
                raise ValueError('actual claimed execution or complete fleet differs from declaration')
            session=attempt/'session';complete_ref=binding(session/'completion.json');complete=load_bound(complete_ref)
            if (complete.get('status')!='passed' or complete.get('complete') is not True
                    or complete.get('cleanup',{}).get('passed') is not True
                    or complete.get('cleanup',{}).get('process_cleanup_verified') is not True):
                raise ValueError('session completion or physical cleanup failed')
            path=session/'windows'/name/'receipt.json';row,point,proof=canonical_receipt(path)
            if point!=load_bound(plan['point']):raise ValueError('executed repeat point differs from declared config/source/startup')
            if point['paired_repeat']['design']!=protocol['design']:raise ValueError('repeat used a different predeclaration')
            rows.append(row);proofs.append(dict(proof,session_completion=complete_ref,claimed_execution=attempt_ref))
        except (ValueError,KeyError,TypeError,OSError,AssertionError) as exc:
            refused.append(dict(point_id=name,job_id=plan['job_id'],reason=str(exc)))
            rows.append(dict(point_id=name,system=plan['system'],repeat_id=plan['repeat_id']))
    result=evaluate_declared_repeats(protocol,rows,measurement_compatibility=reviews)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    write_new(out/'queue-snapshot.json',queue)
    for name,value in [('acceptance.json',dict(result,protocol=protocol_ref,source_queue_at_capture=queue_ref,
                        queue_snapshot=binding(out/'queue-snapshot.json'),refused=refused)),
                       ('points.json',rows),('provenance.json',proofs)]:write_new(out/name,value)
    return dict(result['status_counts'],goal_complete=False,requires_result_review=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prepare_cli=sub.add_parser('prepare');prepare_cli.add_argument('--points',type=Path,required=True)
    prepare_cli.add_argument('--jobs',type=Path,action='append',required=True);prepare_cli.add_argument('--out',type=Path,required=True)
    prepare_cli.add_argument('--candidate-revision',required=True);prepare_cli.add_argument('--after-terminal')
    prepare_cli.add_argument('--driver-serialized',action='store_true',help='predeclare one-at-a-time submission after each released lease, including failed attempts')
    prepare_cli.add_argument('--measurement-compatibility',type=Path,action='append',default=[])
    evaluate_cli=sub.add_parser('evaluate');evaluate_cli.add_argument('--package',type=Path,required=True)
    evaluate_cli.add_argument('--queue',type=Path,required=True);evaluate_cli.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.points,args.jobs,args.out,candidate_revision=args.candidate_revision,
        compatibility=args.measurement_compatibility,after_terminal=args.after_terminal,driver_serialized=args.driver_serialized)
    else:result=evaluate(args.package,args.queue,args.out)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
