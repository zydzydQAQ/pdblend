"""Independent raw replay for scripted online discovery, never qualification."""
from __future__ import annotations
from dataclasses import asdict

from pdblend.engine.native import NativeSpec
from pdblend.online.native_control import validate_state
from pdblend.bench.comparison_metrics import reduce_comparison
from pdblend.bench.comparison_metering import summarize_comparison
from .native_timing_audit import need,finite
from .native_timing_plan import digest,read_bound
from .native_runtime_topology import validate_topology,validate_clock
from .native_frequency_domain import with_domain
from .native_online_causal import replay_causal_forecasts
from .native_online_plan import online_trace,validate_prior,MODEL,REVISION


def audit_online_window(raw,plan):
    try:return _audit(raw,plan)
    except (ValueError,RuntimeError,KeyError,TypeError,IndexError,OSError) as exc:
        return dict(raw_complete=False,errors=[str(exc)],formal_eligible=False,online_policy_qualified=False)


def _audit(raw,plan):
    point=raw['point'];trace=online_trace(plan,point)
    need(raw.get('schema')=='pdblend-native-online-discovery-window/v1' and raw.get('revision')==REVISION
         and raw.get('status')=='measured' and not raw.get('error') and not raw.get('cleanup_errors')
         and raw.get('plan_sha256')==digest(plan) and raw['trace']==trace
         and raw.get('formal_eligible') is False and raw.get('online_policy_qualified') is False,
         'online discovery window incomplete or plan/trace differs')
    start,end,tail=raw['service_started_s'],raw['service_end_s'],raw['tail_end_s']
    need(all(finite(v) for v in (start,end,tail)) and end==start+point['duration_s'] and tail>=end,
         'online service and tail clocks differ')
    prior=validate_prior(raw['prior'],point,start)
    if point['purpose']=='holdout':
        candidate=read_bound(raw['candidate'])
        need(candidate['plan_sha256']==digest(plan) and candidate['frozen_s']<start,'online holdout is not bound to preexisting candidate')
    else:need(raw.get('candidate') is None,'online training used a candidate')
    specs={r['spec']['instance_id']:r['spec'] for r in raw['actual_launch']}
    need(len(specs)==len(raw['actual_launch'])==4,'online replica inventory differs')
    for launch in raw['actual_launch']:
        spec=NativeSpec(**launch['spec'])
        need(spec.command()==launch['argv'] and spec.tp==2 and spec.pp==1,'actual online native launch binding differs')
    physical=validate_topology(list(specs.values()),raw['lease'],power=raw['power'],capabilities=raw['capabilities'])
    identities=[]
    for iid,spec in specs.items():
        cap=raw['capabilities'][iid]
        identity={k:cap.get(k) for k in ('model_id','tp','pp','model_hash','tokenizer_hash','image_digest','source_revision','engine_revision')}
        need(all(identity.values()) and identity['model_id']==MODEL,'online physical model/source identity missing')
        identities.append(identity)
        for value,after in [(raw['before'][iid],None),(raw['native_after'][iid],end)]:
            validate_state(value,generation=spec['generation'],tp=2,pp=1,drained=True,observed_after_s=after)
            need(value.get('acknowledged') is True and value.get('drained') is True,'online native drain ACK missing')
        need(raw['before'][iid]['native_at_s']<=start and raw['native_after'][iid]['native_at_s']<=tail,
             'online native drain is outside its real window boundary')
        validate_clock(raw['initial_clocks'][iid],spec,physical,point['actions'][0]['frequency_mhz'])
        for key in ('measurement_start','measurement_stop'):
            rows=raw[key][iid]['ranks']
            need(len(rows)==2 and {r['rank'] for r in rows}=={0,1} and all(r.get('acknowledged') is True for r in rows),
                 'online native measurement ACK omitted TP rank')
        ranks=raw['samples'][iid]['ranks'];need(len(ranks)==2 and {r['rank'] for r in ranks}=={0,1},'online CUDA rank omitted')
        ordered=sorted(ranks,key=lambda r:r['rank']);keys=[]
        for rank in ordered:
            events=rank['samples'];signature=[]
            for event in events:
                need(event.get('system')=='pdblend' and event.get('measurement_scope')=='runner'
                     and event.get('rank')==rank['rank'] and event.get('tp')==2 and event.get('pp')==1
                     and finite(event.get('at_s')) and finite(event.get('gpu_elapsed_ms')) and event['gpu_elapsed_ms']>0
                     and not event.get('failed'),'online native CUDA event identity invalid')
                signature.append((event['role'],event['batch'],event['request_ids'],event['prompt_lengths'],event['context_lengths'],event['scheduled_lengths']))
            keys.append(signature)
        need(keys[0]==keys[1],'online CUDA rank shape/order differs')
    need(all(value==identities[0] for value in identities),'online replica sources/models differ')
    need(raw['causal_forecaster']['initial']==prior['forecast'] and raw['causal_forecaster']['prior_ref']==raw['prior'],
         'actual online forecaster did not use the declared tuning prior')
    causal=replay_causal_forecasts(raw['causal_forecaster'])
    routes={row['request_id']:row for row in raw['routes']};requests={f'r{row["idx"]}':row for row in trace['requests']}
    need(len(routes)==len(raw['routes']) and set(routes)==set(requests),'online router cohort incomplete')
    arrivals=[r for r in raw['causal_forecaster']['events'] if r['kind']=='arrive']
    finishes=[r for r in raw['causal_forecaster']['events'] if r['kind']=='finish']
    need(len(arrivals)==len(finishes)==len(routes) and {r['request_id'] for r in arrivals}==set(routes)
         and {r['request_id'] for r in finishes}==set(routes),'actual forecaster callback cohort incomplete')
    for arrival in arrivals:
        rid=arrival['request_id'];request=requests[rid];route=routes[rid]
        finish=next(r for r in finishes if r['request_id']==rid)
        need(arrival['input_tokens']==len(request['prompt']) and arrival['known_max_tokens']==request['max_tokens']
             and route['path']=='M' and route['prefill_instance']==route['decode_instance'] in specs
             and route['terminal_state']=='completed' and not route.get('error')
             and route['completion_tokens']==request['max_tokens']==finish['output_tokens']
             and route['submitted_s']<=arrival['observed_s']<=finish['observed_s']<=tail,
             'online Router request/causal prefix/finish binding differs')
    need(all(n==0 for n in raw['router_inflight'].values()),'online Router still owns requests')
    metrics=reduce_comparison(trace,raw['outcomes'],service_started_s=start,duration_s=point['duration_s'],
        slo=(trace['slo']['ttft_s'],trace['slo']['tpot_s']))
    need(metrics['token_timing_complete'] and not metrics['failed_requests'] and not metrics['unresolved_requests'],
         'online exact-token client cohort incomplete')
    actions=raw['actions'];need(len(actions)==len(point['actions'])-1,'online scripted action inventory differs')
    boundaries=[];stable_start=start;frequency=point['actions'][0]['frequency_mhz']
    for expected,actual in zip(point['actions'][1:],actions):
        need(actual['planned']==expected and actual['status']=='passed'
             and start+expected['offset_s']<=actual['requested_s']<=actual['finished_s']<end
             and set(actual['frequencies_mhz'])==set(specs)
             and set(actual['frequencies_mhz'].values())=={expected['frequency_mhz']},
             'online actual Controller action binding differs')
        boundaries.append((stable_start,actual['requested_s'],frequency))
        stable_start=actual['finished_s']+plan['clock_settle_s'];frequency=expected['frequency_mhz']
    boundaries.append((stable_start,tail,frequency))
    clocks=[(t,values) for t,values in raw['power']['frequency_samples'] if start<=t<=tail]
    need(clocks and clocks[0][0]<=start+1 and clocks[-1][0]>=tail-1
         and all(0<b[0]-a[0]<=1 for a,b in zip(clocks,clocks[1:]))
         and all(len(v)==8 and all(finite(f) and f>0 for f in v) for _,v in clocks),
         'online actual all-board clock coverage missing')
    stable=[]
    for left,right,frequency in boundaries:
        samples=[(t,v) for t,v in clocks if left<=t<right]
        need(right>left and samples and samples[0][0]<=left+1 and samples[-1][0]>=right-1
             and all(abs(f-frequency)<=30 for _,values in samples for f in values),
             'online stable actual clock differs from endpoint')
        stable.append(dict(start_s=left,end_s=right,frequency_mhz=frequency,samples=len(samples)))
    transitions=[r for r in raw['controller_events'] if r.get('kind')=='transition_complete']
    need(len(transitions)==len(point['actions']) and all(r.get('validation_scope')=='native_drain_resume_and_proxy_publication' for r in transitions),
         'actual Controller transition completion receipt missing')
    meter=summarize_comparison(raw['power'],gpu_uuids=raw['lease']['gpu_uuids'],origin_s=start,tail_end_s=tail,
        duration_s=point['duration_s'],gpu_uuid_binding_verified=True)
    need(meter['energy_comparable'],'online whole-window power missing')
    return dict(raw_complete=True,identity=with_domain(identities[0],plan['frequency_domain']),query_count=len(causal['queries']),
        metrics=metrics,metering=meter,stable_clock_intervals=stable,formal_eligible=False,online_policy_qualified=False,
        scope='observed_scripted_actions_and_causal_queries_only',energy_prediction_qualified=False,
        native_clock_transition_model_qualified=False,external_prior_binding_replayed=True,
        hysteresis_policy_executed=False,queries_are_energy_model_labels=False)


def replay_online_collection(reference,plan,*,phase):
    report=read_bound(reference)
    need(report.get('schema')=='pdblend-native-online-discovery-collection/v1' and report.get('phase')==phase
         and report.get('plan_sha256')==digest(plan) and report.get('collection_complete') is True
         and report.get('safe_restore_passed') is True and not report.get('operational_failure') and not report.get('error'),
         'online discovery collection or restoration incomplete')
    expected={digest(p) for p in plan['points'] if p['purpose']==phase};seen=set();queries=0
    for row in report['windows']:
        raw=read_bound(row['raw']);key=digest(raw['point']);need(key in expected and key not in seen,'online point inventory differs')
        seen.add(key);audited=_audit(raw,plan)
        need(audited==row['audit'] and row['restoration']['passed'],'online raw audit or restore does not reproduce')
        queries+=audited['query_count']
    need(seen==expected,'online phase point inventory incomplete')
    return dict(raw_complete=True,query_count=queries,finished_s=report['finished_s'],formal_eligible=False,
                online_policy_qualified=False,energy_prediction_qualified=False)
