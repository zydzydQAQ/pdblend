"""Independent all-eight-replica energy holdout for a frozen native composition.

These serving windows validate the composed predictor. Their mixed-role power
is never inserted into the pure-decode calibration table.
"""
from __future__ import annotations
import math
from pathlib import Path
import statistics

from pdblend.bench.comparison_metrics import canonical_outcomes,reduce_comparison
from pdblend.bench.comparison_metering import summarize_comparison
from pdblend.online.native_control import validate_state
from pdblend.planner.forecast import Forecast,percentile
from pdblend.planner.pool import PoolPlanner,PlannerConfig,SLO
from pdblend.profile.collection.native_timing_audit import need,finite
from pdblend.profile.collection.native_timing_plan import digest
from pdblend.profile.collection.native_timing_collect import WORKER
from .native_power_components import LIMITS


def replay_serving_holdout(evidence,resolver,model,*,candidate_sha256,sources,required_deployments=()):
    need(evidence.get('schema')=='pdblend-native-serving-energy-holdout/v1'
         and evidence.get('candidate_sha256')==candidate_sha256,'frozen native serving-energy holdout required')
    plan=resolver.read(evidence['plan'])
    need(plan.get('schema')=='pdblend-native-serving-energy-plan/v1'
         and plan.get('candidate_sha256')==candidate_sha256 and plan.get('evaluation_used_for_selection') is False
         and plan.get('selection_split')=='calibration_holdout' and plan.get('seed')==9702,
         'independent predeclared serving-energy design required')
    frozen=plan.get('candidate_frozen_s');need(finite(frozen),'candidate freeze time missing')
    expected={digest(p):p for p in plan['points']};need(expected and len(expected)==len(plan['points']),
        'missing/duplicate native energy holdout points')
    checked=[];seen=set();intervals=[]
    for reference in evidence['windows']:
        raw=resolver.read(reference);point=raw['point'];key=digest(point)
        need(key in expected and key not in seen,'native energy holdout point differs or repeats');seen.add(key)
        need(raw.get('schema')=='pdblend-native-serving-energy-window/v1' and raw.get('complete') is True
             and raw.get('hardware_executed') is True and raw.get('candidate_sha256')==candidate_sha256
             and not raw.get('cleanup_errors'),'serving-energy raw collection incomplete')
        trace=resolver.read(raw['trace']);origin=raw['service_started_s'];end=origin+150.
        need(finite(origin) and frozen<origin and trace.get('selection_split')=='calibration_holdout'
             and trace.get('seed')==9702 and trace.get('duration_s')==150
             and trace.get('model_id')==model.model and point.get('trace')==raw['trace'],
             'native held-out trace identity or freeze order differs')
        parent=resolver.read(trace['calibration_parent'])
        need(parent.get('selection_split') in ('calibration','tuning') and parent.get('model_id')==model.model,
             'evaluation requests cannot select or fit the native energy composition')
        requests=trace['requests']
        need(requests and all(0<=r['arrival_s']<150 and r['prompt'] and 2<=r['max_tokens']<=512
             and len(r['prompt'])+r['max_tokens']<=8192 for r in requests),'invalid native calibration request domain')
        frequency=point['frequency_mhz'];need(frequency in (1500,2520),'exact native frequency required')
        lease=resolver.read(raw['lease_manifest']);uuids=lease['gpu_uuids']
        need(len(uuids)==len(set(uuids))==8 and lease['payload'].get('gpu_count')==8
             and lease['payload'].get('exclusive') is True and lease['payload'].get('reserve_host') is True,
             'native holdout must own the exclusive eight-GPU fleet')
        from pdblend.profile.collection.native_timing_plan_v2 import MODEL_TP
        tp=MODEL_TP.get(model.model);need(model.tp==tp and model.pp==1,'model-owned TP topology required')
        slots=8//tp
        caps=raw['capabilities'];launches=raw['actual_launch'];need(len(caps)==len(launches)==slots,'full native holdout fleet missing')
        devices=[]
        from pdblend_runtime.probe import NativeSpec
        for launch in launches:
            spec=NativeSpec(**launch['spec']);cap=caps[spec.instance_id]
            need(spec.tp==tp and spec.pp==1 and spec.max_num_seqs==32 and spec.max_model_len==8192
                 and Path(spec.model).name==model.model and spec.kv_connector=='P2pNcclConnector'
                 and WORKER in spec.extra_args and launch['argv'][1:]==spec.command()[1:],
                 'native calibration actual launch differs from declared fleet')
            need(cap.get('supported') is True and all(cap.get(k)==model.identity[k] for k in model.identity if k!='system')
                 and cap.get('source_revision') in sources['calibration_source_revisions']
                 and cap.get('gpu_uuids')==[uuids[g] for g in spec.gpus],
                 'native calibration source/model/physical identity differs')
            devices.extend(spec.gpus)
            validate_state(raw['before'][spec.instance_id],generation=spec.generation,tp=tp,drained=True)
            validate_state(raw['after'][spec.instance_id],generation=spec.generation,tp=tp,drained=True,observed_after_s=end)
        need(sorted(devices)==list(range(8)),'native serving holdout did not bind every physical board once')
        outcomes=resolver.read(raw['outcomes']);events_path=resolver.path(raw['events']['path'])
        # The resolver validates the journal checksum without parsing it as one
        # JSON object; the frozen reader preserves its original event stream.
        from pdblend.profile.collection.native_timing_plan import binding
        need(binding(events_path)['sha256']==raw['events']['sha256'],'native calibration client journal changed')
        from pdblend.results.journal import iter_journal
        canonical=canonical_outcomes('mixed',trace,outcomes,service_started_s=origin,journal=iter_journal(events_path))
        metrics=reduce_comparison(trace,canonical,service_started_s=origin,duration_s=150,
                                 slo=(trace['slo']['ttft_s'],trace['slo']['tpot_s']))
        need(metrics['token_timing_complete'] and metrics['unresolved_requests']==0 and metrics['failed_requests']==0,
             'serving-energy holdout lacks a complete successful exact-token cohort')
        power=resolver.read(raw['power'])
        measured=summarize_comparison(power,gpu_uuids=uuids,origin_s=origin,tail_end_s=raw['tail_end_s'],
            duration_s=150,gpu_uuid_binding_verified=raw.get('gpu_uuid_binding_verified') is True)
        need(measured['energy_comparable'] is True,'native energy holdout physical sampling has a gap or wrong source')
        clocks=[r for r in power['frequency_samples'] if origin<=r[0]<=end]
        need(clocks and clocks[0][0]<=origin+1 and clocks[-1][0]>=end-1
             and all(0<b[0]-a[0]<=1 for a,b in zip(clocks,clocks[1:]))
             and all(len(v)==8 and all(finite(f) and abs(f-frequency)<=30 for f in v) for _,v in clocks),
             'native energy holdout frequency is incomplete or nonuniform')
        lengths=tuple(len(r['prompt']) for r in requests);outputs=tuple(r['max_tokens'] for r in requests)
        forecast=Forecast(rate_rps=trace['rate_rps'],trend_rps=0.,input_mean=statistics.mean(lengths),
            input_p95=percentile(lengths,.95),output_mean=statistics.mean(outputs),inflight=0,
            inputs=lengths,outputs=outputs,length_pairs=tuple(zip(lengths,outputs)))
        config=PlannerConfig(slots=slots,slo=SLO(trace['slo']['ttft_s'],trace['slo']['tpot_s']),freqs=(frequency,),
                             max_num_seqs=32,peak_batch_cap=32,min_m_instances=4)
        prediction=PoolPlanner(model,config)._mixed_pool(forecast.rate_rps,forecast,forecast.input_mean,
                                                       forecast.input_p95,slots,frequency)
        need(prediction is not None and finite(prediction['power_w']) and prediction['power_w']>0,
             'frozen composed model cannot predict this native holdout workload')
        energy=prediction['power_w']*150.;observed=measured['energy_service_j'];error=abs(energy/observed-1)
        checked.append(dict(dataset=trace['dataset'],frequency_mhz=frequency,arrival_family=point['arrival_family'],
            predicted_energy_j=energy,observed_energy_j=observed,relative_error=error,mean_occupancy=prediction['batch'],
            rate_rps=trace['rate_rps'],calibration_parent=trace['calibration_parent'],counts={'M':slots},
            raw=reference,scope='all_mixed_eight_gpu_calibration_only',pure_decode_node_created=False))
        intervals.append((origin,raw['tail_end_s']))
    need(seen==set(expected),'native energy holdout point inventory incomplete')
    intervals.sort();need(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])), 'independent full-fleet energy windows overlap')
    need({r['frequency_mhz'] for r in checked}=={1500,2520},'both exact frequencies need held-out serving-energy evidence')
    for dataset in {r['dataset'] for r in checked}:
        for frequency in (1500,2520):
            need(len({r['arrival_family'] for r in checked if r['dataset']==dataset and r['frequency_mhz']==frequency})>=2,
                 'two independent arrival patterns are required per held-out dataset/frequency')
    errors=[r['relative_error'] for r in checked]
    need(statistics.mean(errors)<=LIMITS['mean_relative_error'] and max(errors)<=LIMITS['max_relative_error']
         and all(e<=LIMITS['each_window_relative_error'] for e in errors),'native serving-energy composition holdout failed')
    for required in required_deployments:
        chosen=required['chosen'];counts={r:n for r,n in chosen['counts'].items() if n}
        need(counts=={'M':8//model.tp},
             'selected deployment needs its own P/D/park/off energy holdout; all-M evidence cannot substitute')
        matching=[r for r in checked if r['dataset']==required['dataset'] and r['frequency_mhz']==chosen['f_M']
                  and r['rate_rps']==required['target_rate_rps'] and r['calibration_parent']==required['trace']]
        need(len({r['arrival_family'] for r in matching})>=2,
             'selected tuning deployment lacks exact model/topology/rate and independent arrival-family holdouts')
    return dict(passed=True,windows=checked,limits=LIMITS,
                scope='bound_all_mixed_calibration_workload_families_only',evaluation_energy_not_used=True,
                dynamic_role_topology_qualified=False,qualified_deployments=list(required_deployments))
