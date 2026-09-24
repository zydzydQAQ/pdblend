"""Opt-in PD whole-layout energy: native Poisson training and 150s holdout.

The first revision owns only Qwen32B's complete canonical M4/TP2 inventory.
It does not turn whole-request watts into kernel power or reuse discovery
paced/burst4 curves. The caller owns the already resident eight-GPU lease.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import time

import aiohttp
from pdblend.bench.client import poisson_trace,SLOS
from pdblend.bench.comparison_metrics import reduce_comparison
from .native_timing_audit import need,finite
from .native_timing_plan import binding,digest,read_bound
from .native_timing_replay import Resolver
from .native_runtime_collect import NativeRuntimeCollector,validate_inventory,write_new
from .native_serving_cycles import collect_cycle_window
from .native_serving_cycles_audit import _audit
from pdblend.profile.query.native_query_replay import tuning_groups
from .native_frequency_domain import (validate_domain,domain_fields,identity_frequencies,
                                     point_fields,with_domain,require_same_domain)

SCHEMA='pdblend-native-layout-energy-plan/v1'
MODEL='Qwen2.5-32B-Instruct'
REVISION='pdblend_layout_energy_v1'
DOMAIN_REVISION='pdblend_layout_energy_1500_2100_v2'
SCALES=(.25,.5,.75,1.)


def layout_revision(identity):
    frequencies=identity_frequencies(identity)
    if 'frequency_domain' not in identity:return REVISION
    need(identity.get('model_id')==MODEL and (identity.get('tp'),identity.get('pp'))==(2,1)
         and frequencies==(1500,2100),'explicit layout frequency revision differs')
    return DOMAIN_REVISION


def build_layout_plan(ledger_ref,provenance_ref,*,model_id=MODEL,frequency_domain_ref=None):
    need(model_id==MODEL,'missing_profile: layout revision initially supports only canonical TP2/M4')
    domain=validate_domain(read_bound(frequency_domain_ref)) if frequency_domain_ref else None
    identity=dict(model_id=model_id,tp=2,pp=1)
    if domain:identity=with_domain(identity,domain)
    frequencies=identity_frequencies(identity);revision=layout_revision(identity)
    groups=tuning_groups(ledger_ref,provenance_ref,Resolver(),model_id,2,domain);points=[]
    for phase,scales,duration,seed in [('training',(.25,1.),60.,9801),('holdout',SCALES,150.,9802)]:
        for group in groups:
            if group['rate_scale'] not in scales:continue
            for frequency in frequencies:
                points.append(dict(purpose=phase,dataset=group['dataset'],rate_scale=group['rate_scale'],
                    target_rate_rps=group['target_rate_rps'],frequency_mhz=frequency,duration_s=duration,
                    seed=seed,arrival_family='poisson',parent_trace=group['trace_binding'],initial_roles={'M':4},**point_fields(identity)))
    plan=dict(schema=SCHEMA,revision=revision,model_id=model_id,tp=2,pp=1,resident_instances=4,
        gpu_count=8,max_num_seqs=32,min_m_instances=4,query_ledger=ledger_ref,query_provenance=provenance_ref,
        points=points,selection_split='calibration_training_and_independent_holdout',
        routing='least_load_all_M',power_semantics='whole_eight_gpu_wall_clock_mean_w',
        arrival_generator=dict(module='pdblend.bench.client.poisson_trace',sha256=
            binding(Path(__import__('pdblend.bench.client',fromlist=['x']).__file__))['sha256']),
        content_partition='SHA256(prompt,max_tokens) parity; even training, odd holdout',
        formal_arrival_family='poisson',formal_duration_s=150.,
        frequency_semantics=dict(requested_mhz=list(frequencies),observed_tolerance_mhz=30,
            observed_max_gap_s=1.,scope='every_physical_board_service_and_drain_tail',missing_filled=False),
        training_service_s=sum(p['duration_s'] for p in points if p['purpose']=='training'),
        holdout_service_s=sum(p['duration_s'] for p in points if p['purpose']=='holdout'),
        holdout_inventory='both exact frequencies at all three datasets and four confirmed rates',
        evaluation_used_for_selection=False,formal_eligible=False,full_profile_qualified=False,
        remaining_gates=['complete_native_raw_training','frozen_training_only_candidate',
            'full_timing_feasibility_and_energy_candidate_replay','independent_150s_raw_energy_holdout',
            'selected_layout_independent_150s_SLO_holdout','source_compatibility','formal_raw_window_acceptance'])
    if domain:plan.update(frequency_domain_ref=frequency_domain_ref,**domain_fields(domain))
    for point in points:layout_trace(plan,point)
    return plan


def validate_layout_plan(plan):
    need(plan==build_layout_plan(plan['query_ledger'],plan['query_provenance'],model_id=plan['model_id'],
                                frequency_domain_ref=plan.get('frequency_domain_ref')),
         'layout energy plan differs from immutable tuning-only design')
    return plan


def layout_trace(plan,point):
    need(point['frequency_mhz'] in identity_frequencies(plan)
         and point.get('frequency_domain_sha256')==plan.get('frequency_domain_sha256'),
         'layout point frequency/domain differs')
    need(plan['model_id']==MODEL and plan['tp']==2 and plan['resident_instances']==4
         and point['initial_roles']=={'M':4} and point['arrival_family']=='poisson',
         'unsupported layout or arrival family')
    phase=point['purpose'];need(phase in ('training','holdout') and point['seed']==(9801 if phase=='training' else 9802)
        and point['duration_s']==(60. if phase=='training' else 150.),'layout phase/seed/service duration differs')
    parent=read_bound(point['parent_trace']);records={}
    for request in parent['requests']:
        need(request['source']==point['dataset'],'layout request belongs to a different dataset')
        key=digest(dict(prompt=request['prompt'],max_tokens=request['max_tokens']))
        if int(key,16)%2==(0 if phase=='training' else 1):
            records[key]=dict(prompt=request['prompt'],output_tokens=request['max_tokens'])
    need(records,'independent real layout content partition is empty')
    requests=[dict(asdict(r),req_id=r.idx) for r in poisson_trace([records[k] for k in sorted(records)],point['target_rate_rps'],
        point['duration_s'],point['seed'],point['dataset'])]
    need(requests and all(r['prompt'] and 2<=r['max_tokens']<=512 and
        len(r['prompt'])+r['max_tokens']<=8192 for r in requests),'native layout request domain unavailable')
    return dict(schema='pdblend-native-layout-energy-trace/v1',model_id=MODEL,dataset=point['dataset'],
        selection_split='calibration_'+phase,seed=point['seed'],duration_s=point['duration_s'],
        rate_rps=point['target_rate_rps'],realized_arrival_rate_rps=len(requests)/point['duration_s'],
        parent_trace=point['parent_trace'],arrival_family='poisson',requests=requests,
        evaluation_used_for_selection=False,slo=dict(zip(('ttft_s','tpot_s'),SLOS[point['dataset']])))


def audit_layout_window(raw,plan):
    try:
        need(plan.get('schema')==SCHEMA and plan.get('revision')==layout_revision(plan) and plan['model_id']==MODEL
             and plan['tp']==2 and plan['resident_instances']==4 and raw['point']['arrival_family']=='poisson',
             'discovery cycle evidence is outside the layout revision')
        result=_audit(raw,plan,trace_builder=layout_trace)
        from .native_runtime_topology import validate_clock
        physical=dict(zip(raw['lease']['gpu_ids'],raw['lease']['gpu_uuids']))
        for launch in raw['actual_launch']:
            spec=launch['spec'];validate_clock(raw['clocks'][spec['instance_id']],spec,physical,raw['point']['frequency_mhz'])
        trace=raw['trace'];rows=[]
        for request in raw['client_requests']:
            events=request['events'];tokens=[e for e in events if e['token_ids']]
            rows.append(dict(idx=request['req_id'],scheduled_s=request['scheduled_s'],
                completion_tokens=request['completion_tokens'],first_token_s=tokens[0]['received_s'],
                last_token_s=tokens[-1]['received_s'],terminal_s=events[-1]['received_s'],
                finished_s=request['finished_s'],terminal=True,token_events_complete=True,
                token_events=[dict(received_s=e['received_s'],count=len(e['token_ids']),exact=True) for e in tokens]))
        metrics=reduce_comparison(trace,rows,service_started_s=raw['service_started_s'],
            duration_s=trace['duration_s'],slo=SLOS[trace['dataset']])
        need(metrics['token_timing_complete'] and not metrics['failed_requests'] and not metrics['unresolved_requests'],
             'layout canonical exact-token cohort is incomplete')
        if 'frequency_domain' in plan:
            need(read_bound(plan['frequency_domain_ref'])==plan['frequency_domain'],'layout bound domain bytes differ')
            result['identity']=with_domain(result['identity'],plan['frequency_domain'])
        result.update(revision=layout_revision(plan),metrics=metrics,scope='bound_Poisson_all_M4_TP2_layout',
                      profile_qualified=False,duration_s=trace['duration_s'])
        return result
    except (ValueError,RuntimeError,KeyError,TypeError,IndexError,OSError) as exc:
        return dict(passed=False,errors=[str(exc)],formal_eligible=False,full_profile_qualified=False)


async def collect_layout_energy(specs,fleet,meter,sampler,out,*,gpu_uuids,plan,phase='training',
                                candidate_ref=None,selection_ref=None):
    """Same-Fleet phase; all 24 held-out layouts follow a frozen selection.

    Only collection completeness is reported here. Fit or energy/SLO prediction
    failure is assessed by independent CPU replay, never hidden by restoration.
    """
    validate_layout_plan(plan);need(phase in ('training','holdout'),'unknown layout collection phase')
    candidate=None
    if phase=='holdout':
        from pdblend.profile.query.native_layout_model import validate_frozen_selection
        candidate=validate_frozen_selection(plan,candidate_ref,selection_ref)
    else:need(candidate_ref is None and selection_ref is None,'training cannot consume selection/holdout')
    need(len(specs)==4 and all(s.tp==2 and s.pp==1 and Path(s.model).name==MODEL for s in specs),
         'layout energy requires four actual TP2 replicas')
    out=Path(out);runner=NativeRuntimeCollector(specs,fleet,meter,sampler,out,gpu_uuids=gpu_uuids,
                                               frequency_domain_ref=plan.get('frequency_domain_ref'))
    runner.lease=validate_inventory(specs,fleet,meter,sampler,gpu_uuids)
    out.mkdir(parents=True,exist_ok=False);(out/'journal.jsonl').touch(exist_ok=False)
    report=dict(schema='pdblend-native-layout-energy-collection/v1',phase=phase,plan_sha256=digest(plan),
        started_s=time.time(),collection_complete=False,safe_restore_passed=False,ready_for_next=False,
        formal_eligible=False,candidate=candidate_ref,selection=selection_ref,windows=[],cleanup_errors=[],lease=runner.lease)
    if candidate:need(candidate['frozen_s']<report['started_s'],'holdout predates candidate freeze')
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        runner.session=session
        try:
            for index,point in enumerate(plan['points']):
                if point['purpose']!=phase:continue
                path=out/'windows'/f'{index:02d}.json'
                raw=await collect_cycle_window(runner,specs,plan,point,path,candidate_ref=candidate_ref,trace_builder=layout_trace)
                audit=audit_layout_window(raw,plan);report['windows'].append(dict(raw=binding(path),audit=audit))
                need(audit['passed'],'native layout raw collection invalid: '+repr(audit.get('errors')))
            report['collection_complete']=True
        except BaseException as exc:report['error']=repr(exc)
        finally:
            restored=[]
            for spec in specs:
                try:
                    need(fleet[spec.instance_id].alive(),'layout collector lost resident process')
                    stopped=await runner.stop_measurement(spec);drain=await runner.drain(spec)
                    clock=await runner.clock(spec,max(identity_frequencies(plan)));resume=await runner.resume(spec)
                    restored.append(dict(instance_id=spec.instance_id,measurement_stop=stopped,drain=drain,clock=clock,resume=resume))
                except BaseException as exc:report['cleanup_errors'].append(spec.instance_id+': '+repr(exc))
            try:report['restored_lease']=validate_inventory(specs,fleet,meter,sampler,gpu_uuids)
            except BaseException as exc:report['cleanup_errors'].append(repr(exc))
            report['restoration']=restored;report['safe_restore_passed']=not report['cleanup_errors']
            report['operational_failure']='error' in report or not report['safe_restore_passed']
            report['ready_for_next']=report['safe_restore_passed'] and not report['operational_failure']
            report['finished_s']=time.time();write_new(out/'completion.json',report)
    return report


async def collect_and_replay_layout_energy(specs,fleet,meter,sampler,out,*,gpu_uuids,plan,timing_profile_ref):
    """Runtime -> timing already passed; use their real bound profile now.

    CPU qualification failure before collection raises with no GPU mutation.
    After collection starts, operational failures stop further GPU work. A
    complete raw observation that rejects the candidate is an expected model
    gap; safely restored unrelated work may continue. No formal grant occurs.
    """
    from pdblend.profile.query.native_layout_profile import load_layout_timing
    from pdblend.profile.query.native_layout_model import freeze_layout_candidate,freeze_layout_selection,replay_layout_component
    validate_layout_plan(plan)
    timing_model,timing_audit=load_layout_timing(timing_profile_ref)
    if 'frequency_domain' in plan:
        require_same_domain(plan,timing_model.calibration_identity)
    out=Path(out);need(not out.exists(),'layout revision output already exists')
    report=dict(schema='pdblend-native-layout-revision/v1',plan_sha256=digest(plan),timing_profile=timing_profile_ref,
        timing_runtime=timing_audit,started_s=time.time(),status='failed',safe_restore_passed=False,
        ready_for_next=False,operational_failure=False,expected_model_gap=False,formal_eligible=False)
    try:
        training=await collect_layout_energy(specs,fleet,meter,sampler,out/'training',gpu_uuids=gpu_uuids,plan=plan)
        report['training']=training_ref=binding(out/'training/completion.json')
        report['safe_restore_passed']=training['safe_restore_passed']
        need(training['collection_complete'] and training['ready_for_next'],'layout training operationally incomplete')
        try:
            candidate_ref=freeze_layout_candidate(training_ref,plan,out/'candidate.json');report['candidate']=candidate_ref
            selection_ref=freeze_layout_selection(plan,candidate_ref,timing_model,timing_profile_ref=timing_profile_ref,
                                                  out=out/'selection.json');report['selection']=selection_ref
        except ValueError as exc:
            if 'missing_profile:' not in str(exc):raise
            report.update(status='unsupported_candidate',expected_model_gap=True,error=str(exc))
            return report
        held=await collect_layout_energy(specs,fleet,meter,sampler,out/'holdout',gpu_uuids=gpu_uuids,plan=plan,
            phase='holdout',candidate_ref=candidate_ref,selection_ref=selection_ref)
        report['holdout']=holdout_ref=binding(out/'holdout/completion.json');report['safe_restore_passed']=held['safe_restore_passed']
        need(held['collection_complete'] and held['ready_for_next'],'layout holdout operationally incomplete')
        result=replay_layout_component(plan,candidate_ref,selection_ref,holdout_ref,timing_model)
        report['component']=write_new(out/'component.json',result)
        report.update(status='component_passed' if result['component_qualified'] else 'holdout_failed',
            component_qualified=result['component_qualified'],expected_model_gap=not result['component_qualified'])
    except BaseException as exc:report.update(error=repr(exc),operational_failure=True)
    finally:
        report['ready_for_next']=report['safe_restore_passed'] and not report['operational_failure']
        report['finished_s']=time.time();write_new(out/'completion.json',report)
    return report
