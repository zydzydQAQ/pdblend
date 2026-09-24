"""Scoped complete-request-cycle model, fitted before independent native holdout.

An interpolation curve describes whole-eight-board mean watts for one real
content distribution, fixed all-M topology, 60s duration and arrival family.
It has no kernel-power or arbitrary mean-decode-batch interface. The formal PD
planner is not changed and may not inherit this limited calibration scope.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import statistics
import time

from pdblend.profile.collection.native_timing_audit import need,finite
from pdblend.profile.collection.native_timing_plan import read_bound,digest
from pdblend.profile.collection.native_runtime_collect import write_new
from pdblend.profile.collection.native_serving_cycles_audit import audit_cycle_window
from .native_power_components import LIMITS

KIND='pdblend_native_request_cycle_candidate_v1'


def read_cycle_collection(reference,plan,phase):
    report=read_bound(reference)
    need(report.get('schema')=='pdblend-native-request-cycle-collection/v1' and report.get('phase')==phase
         and report.get('plan_sha256')==digest(plan) and report.get('collection_complete') is True
         and report.get('safe_restore_passed') is True and not report.get('cleanup_errors') and not report.get('error'),
         'request-cycle collection did not completely and safely finish')
    expected={digest(p) for p in plan['points'] if p['purpose']==phase};seen=set();rows=[];identities=[];intervals=[]
    for entry in report['windows']:
        raw=read_bound(entry['raw']);key=digest(raw['point'])
        need(key in expected and key not in seen,'request-cycle raw training/holdout inventory differs');seen.add(key)
        audit=audit_cycle_window(raw,plan)
        need(audit['passed'] and audit==entry['audit'],'request-cycle independent raw replay does not reproduce')
        identities.append(audit['identity']);intervals.append((raw['service_started_s'],raw['tail_end_s']))
        rows.append(dict(point=raw['point'],raw=entry['raw'],identity=audit['identity'],trace=raw['trace'],
            observed_power_w=audit['measured']['service_mean_power_w'],
            observed_energy_j=audit['measured']['energy_service_j'],
            scheduler_occupancy=audit['occupancy_by_replica'],
            candidate=raw.get('candidate'),start_s=raw['service_started_s'],tail_end_s=raw['tail_end_s']))
    need(seen==expected and len(expected)==8,'complete eight-window native cycle phase required')
    need(all(i==identities[0] for i in identities),'cycle training/holdout replica source identity differs')
    intervals.sort()
    need(finite(report.get('started_s')) and finite(report.get('finished_s'))
         and report['started_s']<=intervals[0][0] and intervals[-1][1]<=report['finished_s']
         and all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),
         'cycle windows overlap or lie outside the completed collection interval')
    return dict(report=report,rows=rows,identity=identities[0])


def training_fit(training_ref,plan):
    evidence=read_cycle_collection(training_ref,plan,'training');nodes=[]
    for frequency in (1500,2520):
        for family in ('paced','burst4'):
            rows=sorted((r for r in evidence['rows'] if r['point']['frequency_mhz']==frequency
                and r['point']['arrival_family']==family),key=lambda r:r['trace']['rate_rps'])
            need(len(rows)==2 and [r['point']['rate_scale'] for r in rows]==[.25,1.],
                 'cycle predictor requires both distinct training rate endpoints')
            low,high=rows
            need(high['trace']['rate_rps']>low['trace']['rate_rps']>0 and
                 all(finite(r['observed_power_w']) and r['observed_power_w']>0 for r in rows),
                 'cycle power endpoint is missing/nonpositive')
            # No coefficient clipping or synthetic monotonicity. A decreasing
            # observation remains in raw evidence but cannot qualify this model.
            need(high['observed_power_w']>=low['observed_power_w'],
                 'cycle endpoint mean power decreased: this monotone revision is unsupported')
            nodes.append(dict(frequency_mhz=frequency,arrival_family=family,
                rates_rps=[r['trace']['rate_rps'] for r in rows],power_w=[r['observed_power_w'] for r in rows],
                sampled_scheduler_occupancy=[r['scheduler_occupancy'] for r in rows],
                raw=[r['raw'] for r in rows]))
    return dict(kind=KIND,model_id=plan['model_id'],tp=plan['tp'],pp=1,identity=evidence['identity'],
        plan_sha256=digest(plan),training_completion=training_ref,training_finished_s=evidence['report']['finished_s'],
        dataset=plan['dataset'],parent_trace=plan['points'][0]['parent_trace'],nodes=nodes,
        initial_roles={'M':plan['resident_instances']},gpu_count=8,duration_s=60.,
        prediction_semantics='whole_fleet_wall_clock_request_cycle_mean_power_w',
        interpolation='linear_rate_within_measured_endpoints_exact_frequency_and_arrival_family',
        mean_decode_batch_interface=False,pure_decode_power_interface=False,active_prefill_power_interface=False,
        evaluation_used_for_fit=False,holdout_used_for_fit=False,
        formal_eligible=False,full_profile_qualified=False)


def fit_cycle_candidate(training_ref,plan,out):
    """CPU-only freeze after training. Returns a new immutable candidate binding."""
    candidate=training_fit(training_ref,plan)
    need(candidate['training_finished_s']<=time.time(),'training completion lies in the future')
    candidate['frozen_s']=time.time()
    return write_new(Path(out),candidate)


class RequestCycleModel:
    def __init__(self,candidate):
        need(candidate.get('kind')==KIND and candidate.get('prediction_semantics')==
             'whole_fleet_wall_clock_request_cycle_mean_power_w','unknown complete-cycle power semantics')
        self.candidate=deepcopy(candidate)

    def predict_cycle_power_w(self,*,rate_rps,frequency_mhz,arrival_family,model_id,tp,dataset,parent_trace,
                              duration_s,initial_roles):
        wanted=dict(model_id=model_id,tp=tp,dataset=dataset,parent_trace=parent_trace,duration_s=duration_s,initial_roles=initial_roles)
        need(all(self.candidate.get(k)==v for k,v in wanted.items()),
             'missing_profile: request-cycle model/source/content/topology/duration scope differs')
        row=next((n for n in self.candidate['nodes'] if n['frequency_mhz']==frequency_mhz
                  and n['arrival_family']==arrival_family),None)
        need(row is not None and finite(rate_rps) and row['rates_rps'][0]<=rate_rps<=row['rates_rps'][1],
             'missing_profile: request-cycle frequency/arrival family/rate outside measured domain')
        lo,hi=row['rates_rps'];a,b=row['power_w']
        return a+(rate_rps-lo)/(hi-lo)*(b-a)


def replay_cycle_component(training_ref,candidate_ref,holdout_ref,plan):
    """The saved qualification flags never determine component acceptance."""
    candidate=read_bound(candidate_ref);rebuilt=training_fit(training_ref,plan)
    need({k:v for k,v in candidate.items() if k!='frozen_s'}==rebuilt,
         'request-cycle candidate differs from raw training-only refit')
    frozen=candidate.get('frozen_s');need(finite(frozen) and frozen>=rebuilt['training_finished_s'],
         'cycle candidate predates training completion')
    holdout=read_cycle_collection(holdout_ref,plan,'holdout')
    need(holdout['identity']==candidate['identity'] and holdout['report']['candidate']==candidate_ref
         and holdout['report']['training']==training_ref and holdout['report']['started_s']>frozen,
         'cycle holdout source or candidate freeze binding differs')
    model=RequestCycleModel(candidate);checks=[]
    for row in holdout['rows']:
        need(row['candidate']==candidate_ref and row['start_s']>frozen,'raw holdout precedes frozen candidate')
        point=row['point'];trace=row['trace']
        prediction=model.predict_cycle_power_w(rate_rps=trace['rate_rps'],frequency_mhz=point['frequency_mhz'],
            arrival_family=point['arrival_family'],model_id=plan['model_id'],tp=plan['tp'],dataset=plan['dataset'],
            parent_trace=point['parent_trace'],duration_s=point['duration_s'],initial_roles={'M':plan['resident_instances']})
        error=abs(prediction/row['observed_power_w']-1.)
        checks.append(dict(point=point,raw=row['raw'],predicted_power_w=prediction,
            observed_power_w=row['observed_power_w'],relative_error=error,
            sampled_scheduler_occupancy=row['scheduler_occupancy']))
    errors=[r['relative_error'] for r in checks]
    passed=statistics.mean(errors)<=LIMITS['mean_relative_error'] and max(errors)<=LIMITS['max_relative_error']
    passed=passed and all(e<=LIMITS['each_window_relative_error'] for e in errors)
    return dict(schema='pdblend-native-request-cycle-component/v1',component_qualified=passed,
        identity=candidate['identity'],candidate=candidate_ref,training=training_ref,holdout=holdout_ref,
        limits=LIMITS,comparisons=checks,mean_relative_error=statistics.mean(errors),max_relative_error=max(errors),
        exact_scope={k:candidate[k] for k in ('model_id','tp','dataset','parent_trace','initial_roles','duration_s','prediction_semantics')},
        formal_eligible=False,full_profile_qualified=False,planner_automatically_wired=False,
        remaining_gates=['complete_cycle_aware_PD_revision_integration_without_kernel_power_substitution',
            'complete_tuning_candidate_replay','actual_selected_role_topology_holdouts',
            'remaining_dataset_and_arrival_family_scope','formal_150s_window_acceptance'])
