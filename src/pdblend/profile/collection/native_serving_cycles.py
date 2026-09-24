"""Whole-request energy discovery on an existing model-owned eight-GPU fleet.

This collector never loads engines, creates pure-role power nodes, selects on
evaluation, or grants formal qualification. Training and frozen-candidate
holdout are separate calls so a CPU fit can occur while the fleet stays loaded.
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict
import math
from pathlib import Path
import random
import time

import aiohttp
from .native_timing_audit import need,finite
from .native_timing_plan import binding,digest,read_bound
from .native_timing_plan_v2 import MODEL_TP
from .native_timing_replay import Resolver
from .native_timing_collect import WORKER,_request
from .native_runtime_collect import NativeRuntimeCollector,validate_inventory,snapshot_sampler,write_new
from .native_runtime_audit import power_slice
from pdblend.profile.query.native_query_replay import tuning_groups

SCHEMA='pdblend-native-request-cycle-plan/v1'


def build_cycle_plan(ledger_ref,provenance_ref,*,model_id,dataset='alpaca'):
    """Minimal short-prompt dataset: rate endpoints train, interior rates hold out.

    Both arrival patterns have exactly the same request count and token cohort.
    Training/holdout use disjoint prompt/output identities from tuning content.
    A complete profile may require more datasets and deployed role topologies.
    """
    tp=MODEL_TP.get(model_id);need(tp is not None,'unknown model-owned cycle topology')
    groups=tuning_groups(ledger_ref,provenance_ref,Resolver(),model_id,tp)
    selected=[g for g in groups if g['dataset']==dataset]
    need(len(selected)==4,'selected calibration dataset missing from complete tuning inventory')
    byscale={g['rate_scale']:g for g in selected};points=[]
    for purpose,scales,seed in [('training',(.25,1.),9701),('holdout',(.5,.75),9702)]:
        for scale in scales:
            for frequency in (1500,2520):
                for family in ('paced','burst4'):
                    points.append(dict(purpose=purpose,seed=seed,rate_scale=scale,
                        target_rate_rps=byscale[scale]['target_rate_rps'],frequency_mhz=frequency,
                        arrival_family=family,duration_s=60.,parent_trace=byscale[scale]['trace_binding']))
    plan=dict(schema=SCHEMA,model_id=model_id,tp=tp,pp=1,resident_instances=8//tp,dataset=dataset,
        query_ledger=ledger_ref,query_provenance=provenance_ref,points=points,
        max_num_seqs=32,role='all_mixed',evaluation_used_for_selection=False,
        request_semantics='real_tuning_prompt_and_output_pairs_disjoint_content_hash_partition',
        occupancy_semantics='sampled_native_scheduler_running_requests_not_decode_batch',
        mean_power_semantics='whole_fleet_wall_clock_request_cycle_including_idle_and_admission',
        core_gpu_seconds_per_phase=8*8*60,service_seconds_per_phase=8*60,
        minimum_scope='one_bound_dataset_two_rates_two_arrival_patterns_two_exact_frequencies',
        candidate_required_for_holdout=True,formal_eligible=False,full_profile_qualified=False,
        duration_scope='predeclared_60s_calibration_only_no_150s_or_long_cycle_validation_claim',
        remaining_gates=['independent_raw_replay','training_only_cycle_model_design_and_identifiability',
            'freeze_candidate_before_holdout','independent_native_serving_holdout',
            'complete_tuning_candidate_replay','actual_selected_role_topology_holdouts'])
    # Fail before a lease is used if partitioning cannot supply real requests.
    for point in points:cycle_trace(plan,point)
    return plan


def cycle_trace(plan,point):
    parent=read_bound(point['parent_trace']);purpose=point['purpose']
    need(purpose in ('training','holdout') and point['seed']==(9701 if purpose=='training' else 9702),
         'cycle train/holdout seeds differ')
    unique={}
    for row in parent['requests']:
        need(row.get('source')==plan['dataset'],'request cycle cannot use a different dataset')
        key=digest(dict(prompt=row['prompt'],max_tokens=row['max_tokens']))
        if int(key,16)%2==(0 if purpose=='training' else 1):unique[key]=row
    need(unique,'real calibration content partition is empty')
    count=round(point['target_rate_rps']*point['duration_s']);need(count>=4,'at least four real requests required')
    pool=[unique[k] for k in sorted(unique)];rng=random.Random(point['seed']);rng.shuffle(pool)
    requests=[];duration=point['duration_s']
    for i in range(count):
        source=pool[i%len(pool)]
        arrival=(i+.5)*duration/count if point['arrival_family']=='paced' else (i//4+.5)*duration/math.ceil(count/4)
        requests.append(dict(req_id=i,arrival_s=arrival,prompt=list(source['prompt']),
            max_tokens=source['max_tokens'],source=source['source'],request_content_sha256=
                digest(dict(prompt=source['prompt'],max_tokens=source['max_tokens']))))
    need(all(0<=r['arrival_s']<duration and r['prompt'] and 2<=r['max_tokens']<=512
             and len(r['prompt'])+r['max_tokens']<=8192 for r in requests),'cycle request exceeds native domain')
    return dict(schema='pdblend-native-request-cycle-trace/v1',model_id=plan['model_id'],dataset=plan['dataset'],
        selection_split='calibration_'+purpose,seed=point['seed'],duration_s=duration,
        rate_rps=count/duration,parent_trace=point['parent_trace'],arrival_family=point['arrival_family'],
        requests=requests,evaluation_used_for_selection=False)


def validate_cycle_plan(plan):
    expected=build_cycle_plan(plan['query_ledger'],plan['query_provenance'],model_id=plan['model_id'],dataset=plan['dataset'])
    need(plan==expected,'request-cycle plan differs from original actual-query design')
    return plan


def validate_candidate(candidate_ref,plan,training_ref):
    """Require training-only provenance and an already frozen predictor artifact."""
    need(candidate_ref is not None and training_ref is not None,
         'holdout blocked: freeze a training-only request-cycle candidate while the fleet remains resident')
    candidate=read_bound(candidate_ref);training=read_bound(training_ref)
    need(candidate.get('kind')=='pdblend_native_request_cycle_candidate_v1'
         and candidate.get('model_id')==plan['model_id'] and candidate.get('tp')==plan['tp']
         and candidate.get('training_completion')==training_ref and candidate.get('plan_sha256')==digest(plan)
         and candidate.get('evaluation_used_for_fit') is False and candidate.get('holdout_used_for_fit') is False
         and finite(candidate.get('frozen_s')) and candidate['frozen_s']<time.time(),
         'request-cycle candidate is not bound to training-only model/plan/source')
    need(training.get('phase')=='training' and training.get('collection_complete') is True
         and training.get('safe_restore_passed') is True and training.get('plan_sha256')==digest(plan)
         and training['finished_s']<=candidate['frozen_s'],'candidate was frozen before training safely completed')
    # Binding every training file now prevents a self-declared completion from
    # silently substituting another cohort. Independent replay/fit remains a gate.
    seen=set()
    for row in training['windows']:
        raw=read_bound(row['raw']);need(raw.get('status')=='measured' and raw['point']['purpose']=='training',
                                     'candidate training inventory contains an incomplete or held-out window')
        key=digest(raw['point']);need(key not in seen,'duplicate candidate training raw window');seen.add(key)
        from .native_serving_cycles_audit import audit_cycle_window
        replay=audit_cycle_window(raw,plan)
        need(replay['passed'] and replay==row['audit'],'candidate training raw audit does not reproduce')
    need(seen=={digest(p) for p in plan['points'] if p['purpose']=='training'},
         'candidate training inventory incomplete')
    return candidate


async def collect_cycle_window(runner,specs,plan,point,path,*,candidate_ref=None,trace_builder=None):
    trace=(trace_builder or cycle_trace)(plan,point);origin=None;tasks=[];state_task=None
    raw=dict(schema='pdblend-native-request-cycle-window/v1',system='pdblend',status='failed',point=point,
        plan_sha256=digest(plan),trace=trace,candidate=candidate_ref,lease=runner.lease,
        actual_launch=[dict(spec=asdict(s),argv=s.command()) for s in specs],capabilities={},before={},
        resume={},clocks={},measurement_start={},measurement_stop={},after={},samples={},
        state_observations=[],routes=[],client_requests=[],cleanup_errors=[],hardware_executed=True,
        power_scope='complete_request_cycle_not_pure_decode_or_active_prefill_kernel',formal_eligible=False)
    counts={s.instance_id:0 for s in specs};stop_states=asyncio.Event()
    async def observe_states():
        while not stop_states.is_set():
            started=time.monotonic()
            for spec in specs:
                state=await runner.state(spec)
                raw['state_observations'].append(dict(instance_id=spec.instance_id,received_s=time.time(),state=state))
            try:await asyncio.wait_for(stop_states.wait(),timeout=max(.01,.5-(time.monotonic()-started)))
            except asyncio.TimeoutError:pass
    async def submit(request):
        await asyncio.sleep(max(0,origin+request['arrival_s']-time.time()))
        # Event-loop serialized acquisition/release is the actual least-load
        # routing rule; no background sampler or fictional admission timestamp.
        spec=min(specs,key=lambda s:(counts[s.instance_id],s.instance_id))
        rid='pd-cycle-'+digest(point)[:12]+'-'+str(request['req_id'])
        row=dict(request_id=rid,req_id=request['req_id'],instance_id=spec.instance_id,seen_tokens=0,
                 scheduled_s=origin+request['arrival_s'])
        raw['client_requests'].append(row)
        raw['routes'].append(dict(event='acquire',request_id=rid,instance_id=spec.instance_id,
                                  at_s=time.time(),before=dict(counts)))
        counts[spec.instance_id]+=1
        try:
            await _request(runner.session,spec.base_url,dict(request_id=rid,prompt=request['prompt'],
                max_tokens=request['max_tokens'],temperature=0,seed=point['seed'],ignore_eos=True),row)
        finally:
            counts[spec.instance_id]-=1
            raw['routes'].append(dict(event='release',request_id=rid,instance_id=spec.instance_id,
                                      at_s=time.time(),after=dict(counts)))
    try:
        for spec in specs:
            iid=spec.instance_id
            raw['capabilities'][iid]=await runner.capability(spec)
            raw['before'][iid]=await runner.drain(spec)
            raw['resume'][iid]=await runner.resume(spec)
            raw['clocks'][iid]=await runner.clock(spec,point['frequency_mhz'])
            raw['measurement_start'][iid]=await runner.request(spec,'measurement/start',dict(system='pdblend',scope='runner'))
            need(raw['measurement_start'][iid].get('acknowledged') is True,'idle native cycle measurement arm lacks ACK')
        await asyncio.sleep(2.)
        # A fresh all-rank boundary observation brackets the sampled occupancy.
        for spec in specs:
            state=await runner.state(spec)
            raw['state_observations'].append(dict(instance_id=spec.instance_id,received_s=time.time(),state=state))
        origin=raw['service_started_s']=time.time();raw['service_end_s']=origin+point['duration_s']
        state_task=asyncio.create_task(observe_states())
        tasks=[asyncio.create_task(submit(request)) for request in trace['requests']]
        await asyncio.sleep(point['duration_s'])
        await asyncio.wait_for(asyncio.gather(*tasks),timeout=120.)
        raw['cohort_finished_s']=time.time()
        for spec in specs:
            iid=spec.instance_id;raw['after'][iid]=await runner.drain(spec)
            raw['samples'][iid]=await runner.request(spec,'measurement/samples')
            raw['measurement_stop'][iid]=await runner.stop_measurement(spec)
        raw['tail_end_s']=time.time();raw['status']='measured'
    except BaseException as exc:raw['error']=repr(exc)
    finally:
        # Every unfinished HTTP request receives native cancellation before the
        # task is discarded. Cleanup failure stops subsequent collection.
        if raw['status']!='measured':
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            raw['cancellations']={}
            for spec in specs:
                try:
                    state=await runner.state(spec)
                    for rid in state['all_queue']:
                        raw['cancellations'][rid]=await runner.request(spec,'cancel',dict(request_id=rid))
                except BaseException as exc:raw['cleanup_errors'].append(repr(exc))
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        stop_states.set()
        if state_task:
            try:await state_task
            except BaseException as exc:raw['cleanup_errors'].append('native occupancy observation: '+repr(exc))
        for spec in specs:
            try:
                if spec.instance_id not in raw['after']:raw['after'][spec.instance_id]=await runner.drain(spec)
                if spec.instance_id not in raw['measurement_stop']:
                    raw['measurement_stop'][spec.instance_id]=await runner.stop_measurement(spec)
            except BaseException as exc:raw['cleanup_errors'].append(repr(exc))
        if origin is not None:
            raw.setdefault('tail_end_s',time.time());end=raw['tail_end_s'];deadline=time.monotonic()+5
            while runner.sampler.error is None and time.monotonic()<deadline:
                if runner.sampler.power_metadata and min(runner.sampler.power_metadata[-1]['read_finished_s'])>end:break
                await asyncio.sleep(.05)
            raw['power']=power_slice(snapshot_sampler(runner.sampler,runner.uuids),origin,end)
        if raw['cleanup_errors']:raw['status']='failed'
        write_new(path,raw)
    return raw


async def restore_cycle_fleet(runner, specs, fleet, meter, sampler, gpu_uuids):
    """Verify recovery before another window, preserving every failure."""
    restored=[]; errors=[]; lease=None
    for spec in specs:
        try:
            need(fleet[spec.instance_id].alive(),'cycle collection lost resident engine')
            stopped=await runner.stop_measurement(spec);drain=await runner.drain(spec)
            clock=await runner.clock(spec,2520);resume=await runner.resume(spec)
            restored.append(dict(instance_id=spec.instance_id,measurement_stop=stopped,drain=drain,
                                 clock=clock,resume=resume))
        except BaseException as exc:errors.append(spec.instance_id+': '+repr(exc))
    try:lease=validate_inventory(specs,fleet,meter,sampler,gpu_uuids)
    except BaseException as exc:errors.append(repr(exc))
    return dict(instances=restored,errors=errors,lease=lease,passed=not errors)


async def collect_serving_cycles(specs,fleet,meter,sampler,out,*,gpu_uuids,plan,phase='training',
                                  candidate_ref=None,training_ref=None):
    """Public same-Fleet entry; holdout requires an already frozen CPU candidate."""
    validate_cycle_plan(plan);need(phase in ('training','holdout'),'unknown request-cycle phase')
    candidate=validate_candidate(candidate_ref,plan,training_ref) if phase=='holdout' else None
    need(phase!='training' or candidate_ref is None,'training cannot consume a held-out candidate')
    need(len(specs)==plan['resident_instances'] and all(s.tp==plan['tp'] and s.pp==1
         and Path(s.model).name==plan['model_id'] and WORKER in s.extra_args for s in specs),
         'request-cycle resident topology/worker differs')
    out=Path(out);runner=NativeRuntimeCollector(specs,fleet,meter,sampler,out,gpu_uuids=gpu_uuids)
    runner.lease=validate_inventory(specs,fleet,meter,sampler,gpu_uuids)
    out.mkdir(parents=True,exist_ok=False);(out/'journal.jsonl').touch(exist_ok=False)
    report=dict(schema='pdblend-native-request-cycle-collection/v1',phase=phase,plan_sha256=digest(plan),
        started_s=time.time(),collection_complete=False,safe_restore_passed=False,ready_for_timing=False,
        formal_eligible=False,full_profile_qualified=False,pure_power_component_qualified=False,
        candidate=candidate_ref,training=training_ref,windows=[],cleanup_errors=[],lease=runner.lease,
        qualification_gaps=[],invalid_window_restorations=[],all_observations_valid=False)
    if candidate:need(candidate['frozen_s']<report['started_s'],'holdout started before candidate freeze')
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        runner.session=session
        try:
            for index,point in enumerate(plan['points']):
                if point['purpose']!=phase:continue
                path=out/'windows'/f'{index:02d}.json'
                raw=await collect_cycle_window(runner,specs,plan,point,path,candidate_ref=candidate_ref)
                from .native_serving_cycles_audit import audit_cycle_window
                audit=audit_cycle_window(raw,plan)
                report['windows'].append(dict(raw=binding(path),audit=audit))
                if (audit.get('error_kind')=='expected_measurement_qualification_gap'
                        and audit.get('qualification_gap')=='observed_frequency_mismatch'
                        and audit.get('non_frequency_checks_passed') is True
                        and audit.get('frequency_data_complete') is True and audit['passed'] is False):
                    report['qualification_gaps'].append(dict(raw=binding(path),
                        kind=audit['qualification_gap'],invalid_window_preserved=True,
                        frequency_evidence=audit['frequency_evidence']))
                    recovery=await restore_cycle_fleet(runner,specs,fleet,meter,sampler,gpu_uuids)
                    report['invalid_window_restorations'].append(dict(raw=binding(path),**recovery))
                    need(recovery['passed'],'request-cycle invalid-window recovery failed: '+repr(recovery['errors']))
                    continue
                need(audit['passed'],'raw native request-cycle evidence incomplete: '+repr(audit.get('errors')))
            report['collection_complete']=True
            report['all_observations_valid']=not report['qualification_gaps']
        except BaseException as exc:report['error']=repr(exc)
        finally:
            recovery=await restore_cycle_fleet(runner,specs,fleet,meter,sampler,gpu_uuids)
            report['cleanup_errors'].extend(recovery['errors']);report['restored_lease']=recovery['lease']
            report['restoration']=recovery['instances'];report['safe_restore_passed']=recovery['passed']
            report['ready_for_timing']=report['safe_restore_passed'] and 'error' not in report
            report['finished_s']=time.time();report['remaining_gates']=plan['remaining_gates']
            write_new(out/'completion.json',report)
    return report


async def collect_and_fit_serving_cycles(specs,fleet,meter,sampler,out,*,gpu_uuids,plan):
    """Same-lease training -> immutable CPU candidate -> independent holdout.

    A failed power prediction is a calibration result. It may continue unrelated
    timing only after safe restoration. HTTP/native/measurement/cleanup failures
    stop the caller. This function never starts/stops engines or the sampler.
    """
    from pdblend.profile.query.native_cycle_model import fit_cycle_candidate,replay_cycle_component
    out=Path(out);need(not out.exists(),'request-cycle revision output already exists')
    report=dict(schema='pdblend-native-request-cycle-revision/v1',plan_sha256=digest(plan),
        started_s=time.time(),status='failed',safe_restore_passed=False,ready_for_timing=False,
        component_qualified=False,formal_eligible=False,full_profile_qualified=False,
        expected_model_gap=False,operational_failure=False)
    try:
        training=await collect_serving_cycles(specs,fleet,meter,sampler,out/'training',gpu_uuids=gpu_uuids,plan=plan)
        report['training']=training_ref=binding(out/'training/completion.json')
        report['safe_restore_passed']=training['safe_restore_passed']
        need(training['collection_complete'] and training['ready_for_timing'],'request-cycle training operationally incomplete')
        if training.get('qualification_gaps'):
            report.update(status='frequency_qualification_failed',expected_model_gap=True,
                qualification_gaps=training['qualification_gaps'],failed_qualification_phase='training',
                holdout_not_collected='training contains unqualified frequency observations; no candidate was fitted')
            return report
        try:
            candidate_ref=fit_cycle_candidate(training_ref,plan,out/'candidate.json')
        except ValueError as exc:
            if 'endpoint mean power decreased' not in str(exc):raise
            report.update(status='unsupported_monotone_model',expected_model_gap=True,error=str(exc))
            return report
        report['candidate']=candidate_ref
        holdout=await collect_serving_cycles(specs,fleet,meter,sampler,out/'holdout',gpu_uuids=gpu_uuids,
            plan=plan,phase='holdout',candidate_ref=candidate_ref,training_ref=training_ref)
        report['holdout']=holdout_ref=binding(out/'holdout/completion.json')
        report['safe_restore_passed']=holdout['safe_restore_passed']
        need(holdout['collection_complete'] and holdout['ready_for_timing'],'request-cycle holdout operationally incomplete')
        if holdout.get('qualification_gaps'):
            report.update(status='frequency_qualification_failed',expected_model_gap=True,
                qualification_gaps=holdout['qualification_gaps'],failed_qualification_phase='holdout')
            return report
        audit=replay_cycle_component(training_ref,candidate_ref,holdout_ref,plan)
        report['component']=write_new(out/'component.json',audit)
        report.update(component_qualified=audit['component_qualified'],
            status='component_passed' if audit['component_qualified'] else 'holdout_failed',
            expected_model_gap=not audit['component_qualified'])
    except BaseException as exc:report.update(error=repr(exc),operational_failure=True)
    finally:
        report['ready_for_timing']=report['safe_restore_passed'] and not report['operational_failure']
        report['finished_s']=time.time();write_new(out/'completion.json',report)
    return report
