#!/usr/bin/env python3
"""Prepare-only 4+2+1+1 PDBlend holdout/training/repair cohort. Never enqueue."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def module(name, file):
    spec=importlib.util.spec_from_file_location(name, ROOT/'scripts'/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


paired=module('quad_paired', '2026-09-23_prepare_paired_long_context.py')
costing=module('quad_costing', '2026-09-23_prepare_long_context_phase1.py')


def subset_plan(path, *, model_id, tp, batches):
    """Keep selected measurements verbatim; explicitly exclude deferred regions."""
    full=json.loads(path.read_text());raw_path=Path(full['training_source'])
    if paired.sha256(raw_path)!=full['training_source_sha256']:
        raise ValueError('training source checksum mismatch')
    raw=json.loads(raw_path.read_text())
    identity=dict(system='pdblend',model_id=model_id,tp=tp,pp=1)
    if any(full.get(k)!=v or raw.get(k)!=v for k,v in identity.items()):
        raise ValueError('incremental plan and training identity differ')
    if full.get('fit_existing_holdout') is not False:
        raise ValueError('training must not consume existing holdout')
    selected=[copy.deepcopy(p) for p in full['training'] if p['batch'] in batches]
    expected={(f,c,b) for f in costing.FREQUENCIES for c in (5120,7168) for b in batches}
    if len(selected)!=len(expected) or {(p['freq_mhz'],p['context_tokens'],p['batch']) for p in selected}!=expected:
        raise ValueError('selected long-context shapes are not in the original legal plan')
    capacity=raw['kv_capacity_tokens'];limit=.9*capacity
    if any(p['batch']*(p['context_tokens']+p['max_tokens'])>limit or
           p['context_tokens']+p['max_tokens']>8192 or p.get('purpose')!='training_extension' or
           p['repeats']!=3 or p['settle_s']<2 or p['measure_s']<5 for p in selected):
        raise ValueError('selected shape violates live-capacity reservation or sample gates')
    deferred=[dict(copy.deepcopy(p),scheduling_status='deferred',coverage_status='missing_profile',
                   reason='phase1_prioritizes_low_batch_long_context') for p in full['training'] if p['batch'] not in batches]
    holdout=[copy.deepcopy(p) for p in full['holdout'] if p['batch']<=max(batches)]
    deferred_holdout=[dict(copy.deepcopy(p),scheduling_status='deferred',coverage_status='missing_profile')
                      for p in full['holdout'] if p['batch']>max(batches)]
    unsupported=[]
    if model_id=='Qwen2.5-32B-Instruct' and tp==2:
        for f in costing.FREQUENCIES:
            for c in (5120,7168):
                reservation=8*(c+1024)
                if reservation<=limit:
                    raise ValueError('B8 memory conclusion changed; explicitly re-review this plan')
                unsupported.append(dict(freq_mhz=f,batch=8,context_tokens=c,max_tokens=1024,
                    status='unsupported_memory',coverage_status='missing_profile',allowed_in_planner=False,
                    requested_reservation_tokens=reservation,measured_kv_capacity_tokens=capacity,
                    usable_capacity_tokens=limit,reason='full_output_reservation_exceeds_measured_capacity'))
    phase=copy.deepcopy(full)
    phase.update(phase='phase1_low_batch_long_context',training=selected,holdout=holdout,
        deferred_training=deferred,deferred_holdout=deferred_holdout,unsupported_requested_points=unsupported,
        original_full_plan_path=str(path.resolve()),original_full_plan_sha256=paired.sha256(path),
        original_training_point_count=len(full['training']),selected_training_point_count=len(selected),
        deferred_training_point_count=len(deferred),completion_scope=f'only_the_{len(selected)}_declared_phase1_training_points',
        full_training_matrix_complete=False,formal_eligible=False,
        selected_training_minimum_window_seconds=sum(p['repeats']*(p['settle_s']+p['measure_s']) for p in selected),
        minimum_window_seconds=sum(p['repeats']*(p['settle_s']+p['measure_s']) for p in selected+holdout),
        coverage_constraints=dict(long_context_batch_max=max(batches),actual_contexts_only=True,
            selected_prompt_context_tokens=[5120,7168],high_batch_long_context_status='missing_profile',
            excludes=[dict(batch_min=max(batches)+1,context_tokens_min_exclusive=4096,status='missing_profile')],
            forbid_low_batch_long_context_extrapolation_to_high_batch=True,
            forbid_rectangular_domain_union_with_short_context_high_batch=True,
            promotion_requires='new_shape_bounded_fit_and_fresh_independent_holdout'))
    phase['domain_note']=full.get('domain_note','')+f' Phase one excludes batches greater than {max(batches)} at contexts greater than 4096; the full original matrix remains incomplete.'
    costs=[]
    for p in selected:
        seconds,method,anchors=costing.prefill_proxy(raw,p['freq_mhz'],p['context_tokens'])
        windows=p['repeats']*(p['settle_s']+p['measure_s'])
        costs.append(dict(freq_mhz=p['freq_mhz'],batch=p['batch'],context_tokens=p['context_tokens'],
            single_request_prefill_seconds=seconds,prefill_method=method,anchors=anchors,
            serial_prefill_proxy_seconds=p['repeats']*p['batch']*seconds,
            decode_window_seconds=windows,scheduling_proxy_seconds=p['repeats']*p['batch']*seconds+windows))
    report=dict(model_id=model_id,tp=tp,points=len(selected),rows=costs,
        serial_prefill_proxy_seconds=sum(x['serial_prefill_proxy_seconds'] for x in costs),
        decode_window_seconds=sum(x['decode_window_seconds'] for x in costs),
        scheduling_proxy_seconds=sum(x['scheduling_proxy_seconds'] for x in costs),
        exclusions='model loading, 16-token barrier, qualification, frequency changes, cleanup, retries and waits',
        actual_batch_runtime_measured=False,guaranteed_runtime_bound=False,
        measured_kv_capacity_tokens=capacity,usable_capacity_tokens=limit,
        largest_selected_reservation_tokens=max(p['batch']*(p['context_tokens']+p['max_tokens']) for p in selected))
    return phase,report


def rewritten_argv(payload, *, name, member, mounts, command):
    """Retain immutable runtime/model identity; replace only role-specific mounts."""
    image=payload['image_digest'];before=payload['argv'][:payload['argv'].index(image)]
    result=[];index=0
    while index<len(before):
        value=before[index]
        if value=='--name':result.extend([value,name]);index+=2;continue
        if value=='-e':
            env=before[index+1]
            result.extend([value,'PDBLEND_PROFILE_MEMBER='+member if env.startswith('PDBLEND_PROFILE_MEMBER=') else env])
            index+=2;continue
        if value=='-v':
            mapping=before[index+1];host,target,mode=mapping.rsplit(':',2)
            if target not in ('/plan/long-context.json','/training/raw.json','/candidate','/repair/plan.json','/original'):
                result.extend([value,mapping])
            index+=2;continue
        result.append(value);index+=1
    for host,target in mounts:result.extend(['-v',f'{host}:{target}:ro'])
    return result+[image,'-B','-m']+command


def build_review(state, *, seven_tp4_plan, phase_plans, repair_plan, repair_candidate,
                 original_holdout, review_dir, snapshot, source_hash):
    # Reuse the reviewed old-job guards, Docker runtime preservation, candidate
    # and source-snapshot checks. This scaffold is never written or enqueued.
    base=paired.build_review(state,plan_path=seven_tp4_plan,review_dir=review_dir,
                             snapshot=snapshot,source_hash=source_hash)
    if not base['source_frozen']:raise ValueError('quad review requires the root frozen source')
    files=json.loads((Path(snapshot)/'manifest.json').read_text())['files']
    if 'pdblend/profile/calibration_repair.py' not in files:
        raise ValueError('frozen source lacks bounded repair entry point')
    members=['14b-tp4-holdout','32b-tp2-longctx','7b-tp1-longctx','14b-tp1-repair']
    configs=[('Qwen2.5-32B-Instruct',2,24),('Qwen2.5-7B-Instruct',1,36)]
    plans=[json.loads(p.read_text()) for p in phase_plans]
    for plan,(model,tp,n) in zip(plans,configs):
        if (plan['model_id'],plan['tp'],len(plan['training']))!=(model,tp,n):
            raise ValueError('quad training matrix differs from the explicit 24/36-point decision')
        if paired.sha256(Path(plan['training_source']))!=plan['training_source_sha256']:
            raise ValueError('phase source changed')
    repair=json.loads(repair_plan.read_text());candidate=json.loads((repair_candidate/'manifest.json').read_text())
    previous=json.loads((original_holdout/'completion.json').read_text())
    from pdblend.profile.calibration_repair import repair_points
    from pdblend.profile.model import PerfModel
    repair_points(repair,PerfModel.load(repair_candidate/'candidate.json'))
    if (paired.sha256(repair_candidate/'candidate.json')!=repair['candidate_sha256'] or
        candidate['candidate_sha256']!=repair['candidate_sha256'] or
        candidate['training_raw_sha256']!=repair['training_raw_sha256'] or
        previous.get('candidate_sha256')!=repair['candidate_sha256'] or not previous.get('complete') or
        previous.get('raw_sha256')!=paired.sha256(original_holdout/'raw.json')):
        raise ValueError('repair original archive/candidate identity mismatch')
    identity=dict(base_source_sha256=source_hash,image_digest=base['identity']['image_digest'],
        holdout_candidate_sha256=base['identity']['candidate_sha256'],
        original_holdout_payload_sha256=base['identity']['original_holdout_payload_sha256'],
        phase_plan_sha256=[paired.sha256(p) for p in phase_plans],repair_plan_sha256=paired.sha256(repair_plan),
        repair_original_raw_sha256=previous['raw_sha256'])
    suffix=paired.object_hash(identity)[:20];cohort=f'quad-profile-4-2-1-1-{suffix}'
    ids=[f'holdout-14b-tp4-quad-{suffix}',f'longctx-32b-tp2-quad-{suffix}',
         f'longctx-7b-tp1-quad-{suffix}',f'repair-14b-tp1-quad-{suffix}']
    shared=dict(cohort_id=cohort,profile_wave_members=members,qualification_frequency_mhz=2100)
    holdout=copy.deepcopy(base['jobs'][0]);holdout['job_id']=ids[0]
    holdout['payload'].update(**shared,container_name=ids[0],profile_wave_member=members[0])
    # Only rename the inherited holdout argv; preserve all its exact ro mounts.
    hi=holdout['payload']['argv'].index('--name');holdout['payload']['argv'][hi+1]=ids[0]
    jobs=[holdout]
    template=base['jobs'][1]['payload']
    for index,(plan,path,config) in enumerate(zip(plans,phase_plans,configs),1):
        model,tp,n=config;payload=copy.deepcopy(template)
        payload.update(**shared,container_name=ids[index],profile_wave_member=members[index],model_id=model,
            tp=tp,gpu_count=tp,topology=dict(model_id=model,tp=tp,pp=1,gpu_count=tp),
            training_plan_sha256=paired.sha256(path),training_raw_sha256=plan['training_source_sha256'],
            expected_training_points=n,collection_scope=plan['completion_scope'],
            original_training_points=plan['original_training_point_count'],
            deferred_training_points=plan['deferred_training_point_count'],
            unsupported_memory_points=len(plan['unsupported_requested_points']),
            coverage_constraints=copy.deepcopy(plan['coverage_constraints']))
        payload['argv']=rewritten_argv(template,name=ids[index],member=members[index],
            mounts=[(path,'/plan/long-context.json'),(Path(plan['training_source']),'/training/raw.json')],
            command=['pdblend.profile.long_context_collect','--plan','/plan/long-context.json','--training-raw','/training/raw.json',
                     '--model','/models/'+model,'--gpus']+[str(i) for i in range(tp)]+['--base-port','{lease_port}','--out','/output'])
        jobs.append(dict(job_id=ids[index],payload=payload,priority=holdout['priority'],max_attempts=1))
    payload=copy.deepcopy(template)
    for key in ('training_plan_sha256','training_raw_sha256','expected_training_points','deferred_training_points',
                'original_training_points','holdout_points_consumed','collection_scope','coverage_constraints'):
        payload.pop(key,None)
    payload.update(**shared,container_name=ids[3],profile_wave_member=members[3],model_id='Qwen2.5-14B-Instruct',
        tp=1,gpu_count=1,topology=dict(model_id='Qwen2.5-14B-Instruct',tp=1,pp=1,gpu_count=1),
        evidence_class='independent_holdout_domain_repair',independent_holdout=True,expected_decode_points=4,
        candidate_sha256=repair['candidate_sha256'],original_holdout_raw_sha256=previous['raw_sha256'],
        repair_plan_sha256=paired.sha256(repair_plan),fit_performed=False)
    payload['argv']=rewritten_argv(template,name=ids[3],member=members[3],
        mounts=[(repair_candidate,'/candidate'),(repair_plan,'/repair/plan.json'),(original_holdout,'/original')],
        command=['pdblend.profile.calibration_repair','--candidate-dir','/candidate','--repair-plan','/repair/plan.json',
                 '--original-holdout','/original','--model','/models/Qwen2.5-14B-Instruct','--gpus','0',
                 '--base-port','{lease_port}','--out','/output'])
    jobs.append(dict(job_id=ids[3],payload=payload,priority=holdout['priority'],max_attempts=1))
    if sum(j['payload']['gpu_count'] for j in jobs)!=8 or any(j['payload']['exclusive'] or j['payload']['global_lock'] for j in jobs):
        raise ValueError('quad layout must use exactly eight disjoint leased GPUs')
    replacement=copy.deepcopy(base['replacement'])
    replacement.pop('additional_job_id',None)
    replacement.update(replacement_job_id=ids[0],additional_job_ids=ids[1:],
        reason='Unstarted holdout shares qualified 4+2+1+1 GPUs with independent 32B/7B long-context training and bounded 14B repair.')
    old=paired.choose_unstarted_holdout(state);original_candidate=Path(old['payload']['candidate_dir'])
    original_manifest=json.loads((original_candidate/'manifest.json').read_text())
    immutable=[original_candidate/'candidate.json',original_candidate/'manifest.json',Path(original_manifest['training_raw']),
        *phase_plans,*[Path(p['training_source']) for p in plans],repair_plan,repair_candidate/'candidate.json',
        repair_candidate/'manifest.json',original_holdout/'raw.json',original_holdout/'completion.json']
    wave=dict(cohort_id=cohort,coordinator=True,members=members,qualification_frequency_mhz=2100,
        representative_layout_check_only=True,profile_frequency_coverage=list(costing.FREQUENCIES),
        qualification_note='Existing 2100 MHz isolated-then-concurrent B8/context1024 protocol across four members; no claim of six-frequency interference qualification.')
    return dict(schema=1,mode='prepare_only',live_queue_modified=False,source_frozen=True,
        ready_for_root_final_review=True,automatic_enqueue_allowed=False,identity=identity,jobs=jobs,
        replacement=replacement,wave=wave,wave_dir=str((review_dir/'wave').resolve()),
        immutable_inputs=[dict(path=str(p.resolve()),sha256=paired.sha256(p)) for p in dict.fromkeys(immutable)],
        scheduling_notes=['All four members inherit the same predecessor dependencies and priority.',
            'Model loads remain serialized by the existing shared load lock.',
            'Existing ProfileWave serializes measurement if 5% interference or overlap gates fail.',
            'Completed short members release their GPU leases; replacement work requires fresh qualification.',
            'Completion refers only to selected training points; all deferred/unsupported regions remain unavailable.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,default=ROOT/'results/2026-09-22/three-model/queue.json')
    parser.add_argument('--source-snapshot',type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    plan_root=ROOT/'results/2026-09-23/long-context-plan';plans=[];costs=[]
    for model,tp,batches in [('Qwen2.5-32B-Instruct',2,(1,4)),('Qwen2.5-7B-Instruct',1,(1,4,8))]:
        original=plan_root/f'{model}-tp{tp}.json'
        plan,cost=subset_plan(original,model_id=model,tp=tp,batches=batches)
        dest=out/'plans'/f'{model}-tp{tp}-phase1.json'
        paired.write_immutable(dest,plan);paired.write_immutable(out/'plans'/f'{model}-tp{tp}-original-full.json',json.loads(original.read_text()))
        plans.append(dest);costs.append(cost)
    repair_candidate=ROOT/'results/2026-09-22/three-model/calibration-candidates/14b-tp1-a52c3dc4e13305790e9b'
    original_holdout=ROOT/'results/2026-09-22/three-model/queue-attempts/holdout-14b-tp1-a52c3dc4e13305790e9b/attempt-0001-0a0e17e40255461aa9b103eef72c5135'
    repair_plan=ROOT/'results/2026-09-23/14b-tp1-holdout-domain-repair-plan.json'
    review=build_review(json.loads(args.queue.read_text()),
        seven_tp4_plan=ROOT/'results/2026-09-23/long-context-phase1/Qwen2.5-7B-Instruct-tp4-phase1.json',
        phase_plans=plans,repair_plan=repair_plan,repair_candidate=repair_candidate,
        original_holdout=original_holdout,review_dir=out,snapshot=args.source_snapshot,source_hash=args.source_sha256)
    from pdblend.profile.model import PerfModel
    from pdblend.profile.window_sampling import plan_shared_prefill_windows
    holdout_dir=Path(review['jobs'][0]['payload']['candidate_dir'])
    holdout_manifest=json.loads((holdout_dir/'manifest.json').read_text())
    holdout_raw=json.loads(Path(holdout_manifest['training_raw']).read_text())
    holdout_model=PerfModel.load(holdout_dir/'candidate.json')
    holdout_rows=[]
    for p in holdout_manifest['plan']['decode']:
        b,c,f=p['batch'],p['context_tokens'],p['freq_mhz']
        sharing=plan_shared_prefill_windows(holdout_raw,holdout_model,batch=b,context=c,freq_mhz=f)
        prefills=1 if sharing['status']=='ready' else 3
        seconds,method,anchors=costing.prefill_proxy(holdout_raw,f,c)
        holdout_rows.append(dict(freq_mhz=f,batch=b,context_tokens=c,prefill_runs=prefills,
            sharing_status=sharing['status'],single_request_prefill_seconds=seconds,
            scheduling_proxy_seconds=prefills*b*seconds+21))
    repair_manifest=json.loads((repair_candidate/'manifest.json').read_text())
    repair_training=json.loads(Path(repair_manifest['training_raw']).read_text())
    repair_rows=[]
    for p in json.loads(repair_plan.read_text())['points']:
        seconds,method,anchors=costing.prefill_proxy(repair_training,p['freq_mhz'],256)
        repair_rows.append(dict(freq_mhz=p['freq_mhz'],batch=96,context_tokens=256,
                               scheduling_proxy_seconds=3*96*seconds+21))
    holdout_cost=sum(r['scheduling_proxy_seconds'] for r in holdout_rows)
    repair_cost=sum(r['scheduling_proxy_seconds'] for r in repair_rows)
    report=dict(evidence_class='cpu_scheduling_estimate',actual_batch_runtime_measured=False,
        training=costs,qualification_minimum_decode_window_seconds=105,
        qualification_note='4 isolated ×21s + 1 concurrent ×21s before prefill/barriers; failure fallback can serialize the cohort.',
        repair_minimum_decode_window_seconds=84,
        repair=dict(rows=repair_rows,scheduling_proxy_seconds=repair_cost),
        holdout=dict(decode_rows=holdout_rows,scheduling_proxy_seconds=holdout_cost,
            planned_prefill_grid_points=len(holdout_manifest['plan']['prefill']),
            shared_prefill_points=sum(r['prefill_runs']==1 for r in holdout_rows),
            exclusions='prefill grid, mixed grid, model loading, barriers, cancellation and qualification'),
        critical_path_proxy_seconds=max(holdout_cost,repair_cost,*[x['scheduling_proxy_seconds'] for x in costs]),
        guaranteed_runtime_bound=False,formal_eligible=False)
    review['cost_report']='cpu-cost-report.json'
    for name,data in [('review.json',review),('jobs.json',review['jobs']),('replacement.json',review['replacement']),
                      ('wave/wave.json',review['wave']),('cpu-cost-report.json',report)]:
        paired.write_immutable(out/name,data)
    print(json.dumps(dict(output=str(out),source_sha256=args.source_sha256,live_queue_modified=False,
        jobs=[dict(job_id=j['job_id'],gpu_count=j['payload']['gpu_count'],evidence_class=j['payload']['evidence_class']) for j in review['jobs']],
        training_proxy_minutes={f'{r["model_id"]}-tp{r["tp"]}':r['scheduling_proxy_seconds']/60 for r in costs}),indent=2))


if __name__=='__main__':main()
