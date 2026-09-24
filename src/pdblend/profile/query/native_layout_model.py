"""Training-only whole-layout mean-W predictor and independent native replay."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
import statistics
import time

from pdblend.profile.collection.native_timing_audit import need,finite
from pdblend.profile.collection.native_timing_plan import read_bound,digest,binding
from pdblend.profile.collection.native_runtime_collect import write_new
from pdblend.profile.collection.native_layout_energy import audit_layout_window,validate_layout_plan,MODEL,REVISION,layout_revision
from pdblend.profile.collection.native_frequency_domain import identity_frequencies,require_same_domain,domain_fields
from .native_power_components import LIMITS
from .native_query_replay import tuning_groups,QueryLog
from pdblend.profile.collection.native_timing_replay import Resolver

KIND='pdblend_native_layout_mean_w_candidate_v1'


def read_layout_collection(reference,plan,phase):
    report=read_bound(reference)
    need(report.get('schema')=='pdblend-native-layout-energy-collection/v1' and report.get('phase')==phase
         and report.get('plan_sha256')==digest(plan) and report.get('collection_complete') is True
         and report.get('safe_restore_passed') is True and not report.get('cleanup_errors') and not report.get('error'),
         'native layout phase is incomplete or restoration failed')
    expected={digest(p) for p in plan['points'] if p['purpose']==phase};seen=set();rows=[];intervals=[]
    for entry in report['windows']:
        raw=read_bound(entry['raw']);key=digest(raw['point'])
        need(key in expected and key not in seen,'native layout point inventory differs');seen.add(key)
        audit=audit_layout_window(raw,plan)
        need(audit['passed'] and audit==entry['audit'],'native layout raw audit does not reproduce')
        rows.append(dict(point=raw['point'],raw=entry['raw'],audit=audit,candidate=raw.get('candidate'),
            start_s=raw['service_started_s'],tail_end_s=raw['tail_end_s']))
        intervals.append((raw['service_started_s'],raw['tail_end_s']))
    need(seen==expected and rows,'native layout complete phase inventory required')
    intervals.sort();need(report['started_s']<=intervals[0][0] and intervals[-1][1]<=report['finished_s']
        and all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])), 'native layout windows overlap or exceed phase')
    identity=rows[0]['audit']['identity'];need(all(r['audit']['identity']==identity for r in rows),'layout source identity changed')
    return dict(report=report,rows=rows,identity=identity)


def layout_training_fit(training_ref,plan):
    evidence=read_layout_collection(training_ref,plan,'training');nodes=[]
    for dataset in ('alpaca','sharegpt','longbench'):
        for frequency in identity_frequencies(plan):
            rows=sorted((r for r in evidence['rows'] if r['point']['dataset']==dataset and
                r['point']['frequency_mhz']==frequency),key=lambda r:r['point']['rate_scale'])
            need(len(rows)==2 and [r['point']['rate_scale'] for r in rows]==[.25,1.], 'both raw training endpoints required')
            powers=[r['audit']['measured']['service_mean_power_w'] for r in rows]
            need(all(finite(p) and p>0 for p in powers) and powers[1]>=powers[0],
                 'missing_profile: nonpositive or decreasing endpoint mean power; no clamping')
            nodes.append(dict(dataset=dataset,parent_trace=rows[0]['point']['parent_trace'],
                frequency_mhz=frequency,rates_rps=[r['point']['target_rate_rps'] for r in rows],power_w=powers,
                raw=[r['raw'] for r in rows]))
    require_same_domain(plan,evidence['identity'])
    return dict(kind=KIND,revision=layout_revision(plan),model_id=MODEL,tp=2,pp=1,identity=evidence['identity'],
        **(domain_fields(plan['frequency_domain']) if 'frequency_domain' in plan else {}),
        plan_sha256=digest(plan),training=training_ref,training_finished_s=evidence['report']['finished_s'],
        initial_roles={'M':4},physical_gpu_count=8,arrival_family='poisson',routing='least_load_all_M',
        prediction_semantics='whole_eight_gpu_wall_clock_mean_w',nodes=nodes,training_duration_s=60.,
        rate_semantics='declared_Poisson_intensity; realized_count_per_duration_retained_in_raw_trace',
        requested_serving_duration_s=150.,duration_transfer_qualified=False,formal_eligible=False,
        full_profile_qualified=False,evaluation_used_for_fit=False,holdout_used_for_fit=False)


def freeze_layout_candidate(training_ref,plan,out):
    validate_layout_plan(plan);value=layout_training_fit(training_ref,plan)
    need(value['training_finished_s']<=time.time(),'training finish is in the future')
    value['frozen_s']=time.time();return write_new(Path(out),value)


class NativeLayoutEnergyModel:
    def __init__(self,candidate):
        need(candidate.get('kind')==KIND and candidate.get('prediction_semantics')=='whole_eight_gpu_wall_clock_mean_w',
             'wrong layout prediction semantics')
        self.candidate=deepcopy(candidate)
        self.frequencies=identity_frequencies(candidate)
        self.revision=layout_revision(candidate)
        if 'frequency_domain' in candidate:
            require_same_domain(candidate,candidate['identity'])
            need(candidate.get('revision')==self.revision,'layout candidate revision/domain differs')

    def predict_layout_mean_w(self,*,model_id,tp,pp,counts,rate_rps,frequency_mhz,dataset,parent_trace,
                              arrival_family,service_duration_s,routing='least_load_all_M'):
        need(model_id==self.candidate['model_id']==MODEL and (tp,pp)==(2,1) and counts=={'M':4}
             and arrival_family=='poisson' and service_duration_s==150. and routing=='least_load_all_M'
             and frequency_mhz in self.frequencies,
             'missing_profile: unmeasured native layout/model/arrival/duration/routing')
        node=next((n for n in self.candidate['nodes'] if n['dataset']==dataset and n['parent_trace']==parent_trace
            and n['frequency_mhz']==frequency_mhz),None)
        need(node is not None and finite(rate_rps) and node['rates_rps'][0]<=rate_rps<=node['rates_rps'][1],
             'missing_profile: unmeasured native layout dataset/frequency/rate')
        lo,hi=node['rates_rps'];a,b=node['power_w']
        need(hi>lo>0 and finite(a) and finite(b) and 0<a<=b,'invalid fitted layout endpoints')
        return a+(rate_rps-lo)/(hi-lo)*(b-a)


def build_layout_selection(plan,candidate_ref,timing_model,*,timing_profile_ref):
    """CPU-only all-group/all-frequency replay, before inspecting any holdout."""
    from pdblend.bench.client import Request,SLOS
    from pdblend.bench.run import offline_forecast
    from pdblend.planner.pool import PlannerConfig,SLO
    from pdblend.online.policies import get_policy
    from pdblend.planner.native_layout import NativeLayoutPlanner
    candidate=read_bound(candidate_ref);need(candidate['plan_sha256']==digest(plan),'candidate layout plan differs')
    read_bound(timing_profile_ref)
    need(timing_model.profile_key.get('layout_timing_selection_sha256')==timing_profile_ref['sha256']
         and all(candidate['identity'].get(k)==v for k,v in timing_model.calibration_identity.items() if k!='system'),
         'layout selection timing object does not bind the actual timing/runtime profile identity')
    require_same_domain(plan,candidate);require_same_domain(candidate,timing_model.calibration_identity)
    energy=NativeLayoutEnergyModel(candidate);groups=tuning_groups(plan['query_ledger'],plan['query_provenance'],Resolver(),MODEL,2,
                                                                plan.get('frequency_domain'))
    rows=[]
    for group in groups:
        forecast=replace(offline_forecast([Request(**r) for r in group['trace']['requests']]),rate_rps=group['target_rate_rps'])
        config=get_policy('pdblend').planner_config(PlannerConfig(slots=4,slo=SLO(*SLOS[group['dataset']]),
            freqs=identity_frequencies(plan),max_num_seqs=32))
        scope=dict(dataset=group['dataset'],parent_trace=group['trace_binding'],arrival_family='poisson',service_duration_s=150.)
        logged=QueryLog(timing_model);planner=NativeLayoutPlanner(logged,config,energy,workload_scope=scope)
        plans=planner.candidates(forecast)
        need(plans,'missing_profile: no feasible native layout for '+group['dataset']+' '+str(group['rate_scale']))
        need(not planner.unsupported,'missing_profile: an otherwise feasible layout lacks its energy/timing domain')
        rows.append(dict(dataset=group['dataset'],rate_scale=group['rate_scale'],target_rate_rps=group['target_rate_rps'],
            parent_trace=group['trace_binding'],candidates=[asdict(p) for p in plans],chosen=asdict(plans[0]),
            timing_queries=logged.rows,unsupported_alternatives=planner.unsupported,
            feasible_frequency_count=len(plans)))
    return dict(schema='pdblend-native-layout-selection/v1',revision=layout_revision(plan),plan_sha256=digest(plan),candidate=candidate_ref,
        timing_profile=timing_profile_ref,groups=rows,evaluation_used_for_selection=False,
        min_m_instances=4,canonical_layout_domain_complete=True,formal_eligible=False)


def freeze_layout_selection(plan,candidate_ref,timing_model,*,timing_profile_ref,out):
    value=build_layout_selection(plan,candidate_ref,timing_model,timing_profile_ref=timing_profile_ref)
    value['frozen_s']=time.time();return write_new(Path(out),value)


def validate_frozen_selection(plan,candidate_ref,selection_ref):
    need(candidate_ref is not None and selection_ref is not None,'holdout requires frozen layout candidate and selection')
    candidate=read_bound(candidate_ref);selection=read_bound(selection_ref)
    need(candidate['plan_sha256']==selection['plan_sha256']==digest(plan) and selection['candidate']==candidate_ref
         and selection.get('schema')=='pdblend-native-layout-selection/v1'
         and candidate.get('evaluation_used_for_fit') is False and candidate.get('holdout_used_for_fit') is False
         and selection.get('evaluation_used_for_selection') is False
         and finite(candidate.get('frozen_s')) and finite(selection.get('frozen_s'))
         and candidate['training_finished_s']<=candidate['frozen_s']<=selection['frozen_s']<time.time(),
         'holdout requires frozen training-only candidate and complete prior layout selection')
    expected={(d,s) for d in ('alpaca','sharegpt','longbench') for s in (.25,.5,.75,1.)}
    need(len(selection['groups'])==12 and {(r['dataset'],r['rate_scale']) for r in selection['groups']}==expected,
         'layout selection group inventory incomplete')
    rebuilt=layout_training_fit(candidate['training'],plan)
    need({k:v for k,v in candidate.items() if k!='frozen_s'}==rebuilt,'layout candidate differs from raw training-only refit')
    read_bound(selection['timing_profile'])
    return candidate


def replay_layout_component(plan,candidate_ref,selection_ref,holdout_ref,timing_model):
    candidate=validate_frozen_selection(plan,candidate_ref,selection_ref);selection=read_bound(selection_ref)
    rebuilt=build_layout_selection(plan,candidate_ref,timing_model,timing_profile_ref=selection['timing_profile'])
    need({k:v for k,v in selection.items() if k!='frozen_s'}==rebuilt,'full candidate timing/energy replay differs')
    held=read_layout_collection(holdout_ref,plan,'holdout');report=held['report']
    need(held['identity']==candidate['identity'] and report['candidate']==candidate_ref and report['selection']==selection_ref
         and report['started_s']>selection['frozen_s'],'held-out raw source/selection/freeze binding differs')
    energy=NativeLayoutEnergyModel(candidate);checks=[];selected_checks=[]
    selected={(r['dataset'],r['rate_scale']):r['chosen'] for r in selection['groups']}
    for row in held['rows']:
        point=row['point'];need(row['candidate']==candidate_ref and row['start_s']>selection['frozen_s'],
                              'raw holdout predates selected candidate freeze')
        predicted=energy.predict_layout_mean_w(model_id=MODEL,tp=2,pp=1,counts={'M':4},rate_rps=point['target_rate_rps'],
            frequency_mhz=point['frequency_mhz'],dataset=point['dataset'],parent_trace=point['parent_trace'],
            arrival_family='poisson',service_duration_s=150.)
        observed=row['audit']['measured']['service_mean_power_w'];error=abs(predicted/observed-1.)
        chosen=selected[(point['dataset'],point['rate_scale'])]['f_M']==point['frequency_mhz']
        check=dict(point=point,raw=row['raw'],predicted_power_w=predicted,observed_power_w=observed,
            relative_error=error,selected=chosen,slo_pass=row['audit']['metrics']['slo_pass'])
        checks.append(check)
        if chosen:selected_checks.append(check)
    errors=[r['relative_error'] for r in checks]
    power_pass=statistics.mean(errors)<=LIMITS['mean_relative_error'] and max(errors)<=LIMITS['max_relative_error']
    power_pass=power_pass and all(e<=LIMITS['each_window_relative_error'] for e in errors)
    selected_pass=len(selected_checks)==12 and all(r['slo_pass'] for r in selected_checks)
    return dict(schema='pdblend-native-layout-component/v1',candidate=candidate_ref,selection=selection_ref,holdout=holdout_ref,
        comparisons=checks,limits=LIMITS,mean_relative_error=statistics.mean(errors),max_relative_error=max(errors),
        power_holdout_passed=power_pass,selected_150s_slo_passed=selected_pass,
        component_qualified=power_pass and selected_pass,duration_transfer_qualified=power_pass,
        formal_eligible=False,full_profile_qualified=False,remaining_gates=['qualified_timing_runtime_source_composition',
            'explicit_revision_runtime_integration','independent_formal_raw_window_acceptance'])
