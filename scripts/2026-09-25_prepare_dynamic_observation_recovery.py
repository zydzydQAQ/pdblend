#!/usr/bin/env python3
"""Host-only resume after an attempted dynamic PD window fails execution.

Never enqueue, rewrite a measurement, or retry the failed observation.  The
projection archives only the failed active entry; every decision, observation,
point and receipt remains hash-bound and visible in the publication campaign.
"""
from copy import deepcopy
from pathlib import Path

from pdblend.bench import comparison_campaign as cc
from pdblend.bench.resident_session import digest, write_new
from pdblend.bench.single_observation_slo_boundary import (
    observation_verdict, read_extension_manifest, states_with_endpoints)

KIND = 'failed_dynamic_observation_resume/v1'
PLAN_SCHEMA = 'saturation-execution-recovery/v1'


def completed_receipt(point, receipt):
    return (receipt.get('point') == point['name']
            and receipt.get('point_sha256') == digest(point)
            and receipt.get('cleanup_passed') is True
            and not receipt.get('error')
            and receipt.get('recorded_window_complete') is True
            and observation_verdict(point, receipt)['verdict'] in ('pass', 'fail'))


def project_manifest(original, completion_ref, group, *, load=cc.load_bound):
    """Pure projection. No successful entry or observation may disappear."""
    completion = load(completion_ref)
    if (completion.get('complete') is not False
            or completion.get('group_sha256') != digest(group)
            or completion.get('engine_signature') != group['engine_signature']
            or completion.get('cleanup', {}).get('passed') is not True
            or completion.get('cleanup', {}).get('process_cleanup_verified') is not True
            or completion.get('cleanup', {}).get('errors')
            or completion.get('cleanup_errors')
            or completion.get('extension_manifest') != original['manifest_ref']):
        raise ValueError('failed session lacks matching completed process cleanup')
    head = original['manifest']
    if head['policy'] != group['extension_policy']:
        raise ValueError('failed session changed extension policy')
    observation_by_receipt = {r['receipt']['sha256']: r for r in head['observations']}
    active, archived, retained = [], [], []
    for entry in head['points']:
        if not entry.get('receipt'):
            raise ValueError('pending dynamic point must not be silently discarded')
        point, receipt = load(entry['point']), load(entry['receipt'])
        observed = observation_by_receipt.get(entry['receipt']['sha256'])
        if (observed is None or observed['receipt'] != entry['receipt']
                or load(observed['point']) != point):
            raise ValueError('dynamic entry lacks its exact immutable observation')
        if completed_receipt(point, receipt):
            active.append(deepcopy(entry))
        else:
            if (observation_verdict(point, receipt) !=
                    {'verdict': 'incomplete', 'reason': 'execution_or_drain_failure'}
                    or observed.get('verdict') != 'incomplete'
                    or observed.get('reason') != 'execution_or_drain_failure'):
                raise ValueError('only explicit execution/drain failures may be archived')
            archived.append(deepcopy(entry))
    if not archived:
        raise ValueError('no attempted dynamic execution failure to archive')
    dynamic_failures = {entry['receipt']['sha256'] for entry in archived}
    by_name = {}
    for observation in head['observations']:
        point, receipt = load(observation['point']), load(observation['receipt'])
        if point['name'] in by_name:
            raise ValueError('duplicate attempted point cannot be selected for recovery')
        by_name[point['name']] = (point, receipt)
        if completed_receipt(point, receipt):
            retained.append(dict(point=observation['point'], receipt=observation['receipt']))
        elif observation['receipt']['sha256'] not in dynamic_failures:
            raise ValueError('non-dynamic or unaccounted failed observation cannot be retried')
    for point in group['points']:
        prior = by_name.get(point['name'])
        if prior is None or prior[0] != point or not completed_receipt(*prior):
            raise ValueError('every original group point must already be completed unchanged')
    # Completion references must bind all new attempts, never a foreign receipt.
    report_refs = {(str(Path(r['path']).resolve()), r['sha256']) for r in completion['windows']}
    expected_refs = {(str(Path(r['receipt']['path']).resolve()), r['receipt']['sha256'])
                     for r in head['observations']}
    skipped_refs = {(str(Path(r['frozen_receipt']['path']).resolve()), r['frozen_receipt']['sha256'])
                    for r in completion.get('skipped', [])}
    if report_refs | skipped_refs != expected_refs or len(report_refs) != len(completion['windows']):
        raise ValueError('completion does not account for the exact observation ledger')
    projected = deepcopy(head)
    projected.update(points=active, previous_manifest=original['manifest_ref'],
                     stop_reason=None, continuation_required=False)
    projected['resume_projection'] = dict(schema=KIND, failed_completion=completion_ref,
        source_head=original['manifest_ref'], archived_attempted_entries=archived,
        reason='failed execution is retained as incomplete; only other sequences may advance')
    if (projected['observations'] != head['observations']
            or projected['decisions'] != head['decisions']
            or projected['boundaries'] != head['boundaries']):
        raise ValueError('projection must preserve the full boundary evidence ledger')
    for entry in archived:
        dataset = load(entry['point'])['dataset']
        state = projected['boundaries'][dataset]
        if state['status'] != 'incomplete_observation' or state.get('next_rate_scale') is not None:
            raise ValueError('failed sequence is not blocked against a retry')
    return projected, retained, archived


def verify_projection(ref, *, load=cc.load_bound):
    proof = load(ref)
    if proof.get('schema') != KIND:
        raise ValueError('unknown dynamic recovery projection')
    original = read_extension_manifest(proof['source_head'], load_bound=load)
    group = load(proof['original_group'])
    expected, retained, archived = project_manifest(original, proof['failed_completion'], group, load=load)
    actual = read_extension_manifest(proof['resume_manifest'], load_bound=load)
    if (actual['manifest'] != expected or proof['retained_completed'] != retained
            or proof['archived_attempted_entries'] != archived):
        raise ValueError('dynamic recovery projection changed an attempted observation')
    root = Path(proof['resume_root'])
    for row in retained:
        point = load(row['point'])
        window = root/'windows'/point['name']
        original_window = Path(row['receipt']['path']).parent.resolve()
        if not window.is_symlink() or window.resolve() != original_window:
            raise ValueError('resume does not reference the original completed window')
    expected_names = {load(row['point'])['name'] for row in retained}
    if {p.name for p in (root/'windows').iterdir()} != expected_names:
        raise ValueError('resume contains extra attempted or failed window directories')
    pointer = load(cc.binding(root/'extensions/latest.json'))
    if pointer.get('manifest') != proof['resume_manifest']:
        raise ValueError('resume latest differs from the approved projection')
    return proof


def recovery_group(original, proof, *, load=cc.load_bound):
    group=deepcopy(original)
    names={p['name'] for p in group['points']}
    for row in proof['retained_completed']:
        point=load(row['point'])
        if point['name'] not in names:
            group['points'].append(point);names.add(point['name'])
    group['session_id']='dynamic-failure-resume-'+digest(proof)[:20]
    return group


def recovery_job(original, group, out, resume):
    new_job=deepcopy(original)
    new_job['job_id']='comparison-'+group['model_id'].split('-')[1].lower()+'-'+digest(group)[:16]
    payload=new_job['payload']; args=payload['argv']
    args[args.index('--name')+1]=new_job['job_id'];args[args.index('--group')+1]=str(out/'group.json')
    while '--previous' in args:
        at=args.index('--previous');del args[at:at+2]
    args+=['--previous',str(resume)]
    payload.update(container_name=new_job['job_id'],session_id=group['session_id'],
        comparison_campaign=str(out/'campaign.json'),after_terminal=[])
    new_job.update(priority=3000,max_attempts=1)
    return new_job


def prepare(parent_campaign, original_job, session, out, *, root_path='/home/pdblend4'):
    """Create immutable job/plan; caller remains the sole queue owner."""
    out, session, root = Path(out).resolve(), Path(session).resolve(), Path(root_path).resolve()
    if out.exists():
        raise ValueError('preserve existing or partial preparation; choose a new directory')
    parent_ref = cc.binding(parent_campaign); parent = cc.load_bound(parent_ref)
    argv = original_job['payload']['argv']
    group_ref = cc.binding(argv[argv.index('--group')+1]); group = cc.load_bound(group_ref)
    completion_ref = cc.binding(session/'completion.json')
    completion = cc.load_bound(completion_ref)
    original = read_extension_manifest(completion['extension_manifest'])
    if (original_job['payload'].get('system') != 'pdblend'
            or original_job['payload'].get('run_id') != parent['run_id']
            or original['manifest']['run_id'] != parent['run_id']
            or original['manifest']['model_id'] != group['model_id']
            or group['extension_policy'] not in parent.get('active_extension_policy_refs', [])
            or group['extension_policy'] not in parent.get('extension_policy_refs', [])):
        raise ValueError('recovery is not authorized by the active model/policy/campaign')
    projected, retained, archived = project_manifest(original, completion_ref, group)
    out.mkdir(parents=True)
    write_new(out/'original-job.json',original_job)
    resume = out/'resume'; (resume/'windows').mkdir(parents=True)
    write_new(resume/'extensions/manifest-000000.json', projected)
    resume_ref = cc.binding(resume/'extensions/manifest-000000.json')
    write_new(resume/'extensions/latest.json', dict(schema=projected['schema'], mode='pointer', manifest=resume_ref))
    for row in retained:
        point = cc.load_bound(row['point'])
        (resume/'windows'/point['name']).symlink_to(Path(row['receipt']['path']).parent.resolve(), target_is_directory=True)
    proof = dict(schema=KIND, original_group=group_ref, original_job_id=original_job['job_id'],
        original_job=cc.binding(out/'original-job.json'),
        source_head=original['manifest_ref'], failed_completion=completion_ref,
        resume_root=str(resume), resume_manifest=resume_ref, retained_completed=retained,
        archived_attempted_entries=archived, raw_artifact_revalidation='frozen runtime performs full verification at resume')
    write_new(out/'projection.json', proof); proof_ref=cc.binding(out/'projection.json')
    verify_projection(proof_ref)
    new_group=recovery_group(group,proof)
    write_new(out/'group.json',new_group)
    publication=deepcopy(parent)
    variants={(p['name'],digest(p)) for p in publication['points']}
    for entry in archived:
        point=cc.load_bound(entry['point'])
        if (point['name'],digest(point)) not in variants:
            publication['points'].append(point);variants.add((point['name'],digest(point)))
    publication.update(parent_campaign=parent_ref,campaign_id=out.name,groups=[new_group],
        dynamic_observation_recovery_projections=list(parent.get('dynamic_observation_recovery_projections',[]))+[proof_ref])
    write_new(out/'campaign.json',publication)
    new_job=recovery_job(original_job,new_group,out,resume)
    write_new(out/'job.json',new_job)
    datasets={d:dict(state,lower=None,upper=None,unique_boundary=False)
              for d,state in original['manifest']['boundaries'].items()}
    write_new(out/'prior-selection.json',dict(schema='observed-boundary-selection/v1',
        run_id=parent['run_id'],model_id=group['model_id'],manifest=original['manifest_ref'],
        datasets=datasets,baseline_outcomes_used_for_selection=False))
    plan=dict(schema=PLAN_SCHEMA,recovery_kind=KIND,run_id=parent['run_id'],model_id=group['model_id'],
        prior_baseline_campaign=parent_ref,prior_selection=cc.binding(out/'prior-selection.json'),
        recovery_campaign=cc.binding(out/'campaign.json'),pd_recovery_job=cc.binding(out/'job.json'),
        projection=proof_ref,selection_generation=2,finish_before_job_ids=[],
        preparation_helper=cc.binding(__file__),gpu_source_changed=False,enqueue_performed=False)
    write_new(out/'plan.json',plan)
    return dict(plan=cc.binding(out/'plan.json'),job=cc.binding(out/'job.json'),projection=proof_ref)


def verify_plan(runner, ref):
    plan=cc.load_bound(ref)
    if (plan.get('schema')!=PLAN_SCHEMA or plan.get('recovery_kind')!=KIND
            or plan.get('run_id')!=runner.package.name or plan.get('selection_generation')!=2):
        raise ValueError('dynamic recovery plan scope differs')
    proof=verify_projection(plan['projection'])
    parent=cc.load_bound(plan['prior_baseline_campaign'])
    campaign=cc.load_bound(plan['recovery_campaign'])
    job=cc.load_bound(plan['pd_recovery_job'])
    group=cc.load_bound(cc.binding(job['payload']['argv'][job['payload']['argv'].index('--group')+1]))
    original_group=cc.load_bound(proof['original_group'])
    expected=recovery_group(original_group,proof)
    if group!=expected:
        raise ValueError('recovery altered the original group/source/configuration')
    expected_job=recovery_job(cc.load_bound(proof['original_job']),expected,
        Path(plan['recovery_campaign']['path']).parent,Path(proof['resume_root']))
    if job!=expected_job:
        raise ValueError('recovery changed frozen launch arguments or original job options')
    if (campaign.get('parent_campaign')!=plan['prior_baseline_campaign']
            or plan['projection'] not in campaign.get('dynamic_observation_recovery_projections',[])
            or plan['model_id']!=group['model_id'] or campaign['run_id']!=plan['run_id']
            or job['payload'].get('run_id')!=plan['run_id']
            or job['payload'].get('system')!='pdblend'
            or job['payload'].get('comparison_campaign')!=plan['recovery_campaign']['path']
            or job['job_id']!='comparison-'+group['model_id'].split('-')[1].lower()+'-'+digest(group)[:16]):
        raise ValueError('recovery publication/job identity differs')
    required={(p['name'],digest(p)) for p in parent['points']}
    required.update((cc.load_bound(e['point'])['name'],digest(cc.load_bound(e['point'])))
                    for e in proof['archived_attempted_entries'])
    if required!={(p['name'],digest(p)) for p in campaign['points']}:
        raise ValueError('publication lost historical or failed measurement point identity')
    return plan,job


def recover_model(runner, ref):
    """Existing driver dispatch hook; no baseline has run for this model yet."""
    plan,job=verify_plan(runner,ref);model=plan['model_id']
    progress=runner.state.setdefault('execution_recovery_state',{}).setdefault(ref['sha256'],{})
    if progress.get('complete'):
        if model not in runner.state['completed_models']:raise ValueError('completed recovery lost model checkpoint')
        return
    if progress.get('combined_prepared'):
        marker=cc.load_bound(progress['combined_prepared']);runner.campaign=Path(marker['campaign']['path'])
    else:runner.campaign=Path(plan['recovery_campaign']['path'])
    projection=cc.load_bound(plan['projection'])
    for manifest in (projection['source_head'],projection['resume_manifest']):
        if manifest not in runner.manifests:runner.manifests.append(manifest)
    runner.state.update(model=model,phase='pdblend_dynamic_observation_recovery')
    runner.emit('dynamic_observation_recovery_started',plan=ref)
    if not progress.get('pd_manifest'):
        result=runner.pd(job)
        if not result:raise ValueError('dynamic recovery failed without an immutable manifest')
        progress['pd_manifest']=result;runner.emit('dynamic_observation_recovery_observed',manifest=result)
    selected=runner.selection(model,progress['pd_manifest'],generation=plan['selection_generation'],supersedes=plan['prior_selection'])
    runner.state.setdefault('boundary_selections',{})[model]=cc.binding(selected)
    out,prepared,jobs=runner.prepare_baselines(model,selected)
    progress['combined_prepared']=cc.binding(out/'prepared.json')
    runner.campaign=out/'campaign.json';runner.state['phase']='recovered_baseline_supplements_and_endpoints'
    runner.emit('recovered_baselines_prepared',plan=ref,prepared=progress['combined_prepared'],
        windows=sum(len(g['points']) for g in prepared['groups']),groups=len(jobs))
    for baseline in jobs:runner.run_baseline(baseline)
    runner.publish(force=True);progress['complete']=True
    if model not in runner.state['completed_models']:runner.state['completed_models'].append(model)
    runner.emit('recovered_model_complete',model_id=model,plan=ref)
