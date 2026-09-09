"""Source-ordered logical context from complete owner events; never physical KV.

The caller verifies full before/after engine/source identity and the upstream
prefills-first metadata ordering. This module performs no network or GPU work.
"""
import hashlib
import json
import math
import statistics


def require(ok,why):
    if not ok:raise ValueError(why)


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def reconstruct(raw,events,*,generation,token_budget,source_order_verified=False):
    require(source_order_verified is True,'unverified scheduler/logger ordering; cannot assign mixed event phases')
    require(type(generation) is int and generation>=0 and type(token_budget) is int and token_budget>0,'actual owner generation/budget missing')
    rows=raw['requests'];spec=raw['spec'];batch=spec['batch_size']
    require(type(batch) is int and batch>0 and len(rows)==batch,'complete declared batch missing')
    requests={r.get('request_id'):r for r in rows}
    require(len(requests)==batch and all(isinstance(k,str) and k for k in requests),'request IDs missing/duplicated')
    lengths={};outputs={}
    for rid,r in requests.items():
        n=len(r.get('prompt_token_ids',[]));out=r.get('requested_output_tokens')
        require(type(out) is int and out>=2 and n>0 and r.get('success') is True and r.get('done_marker') is True
            and len(r.get('output_token_ids',[]))==len(r.get('token_received_s',[]))==out
            and r.get('usage',{}).get('prompt_tokens')==n and r.get('usage',{}).get('completion_tokens')==out,
            'full prescribed request input/output/usage absent')
        require(n==spec['input_tokens'] and out==spec['output_tokens'],'point declaration differs from actual request work')
        lengths[rid]=n;outputs[rid]=out
    counts={rid:0 for rid in requests};pf_seen=set();steps=[];previous_end=None;prefill_tokens=0
    runs=[];run=[]
    for index,event in enumerate(events):
        ids=event.get('request_ids');pf,dc=event.get('prefill'),event.get('decode');tokens=event.get('tokens')
        start,end=event.get('started_s'),event.get('finished_s')
        require(isinstance(ids,list) and len(ids)==len(set(ids)) and set(ids)<=set(requests)
            and type(pf) is int and type(dc) is int and pf>=0 and dc>=0 and pf+dc==len(ids)
            and type(tokens) is int and 0<tokens<=token_budget,'invalid/foreign owner IDs or phase/token counts')
        require(event.get('generation')==generation and event.get('role')=='mixed' and event.get('mode')=='continuous'
            and finite(start) and finite(end) and raw['measurement_start_s']<=start<end<=raw['measurement_end_s']
            and (previous_end is None or start>=previous_end-1e-6),'owner generation/identity or complete event order differs')
        previous_end=end
        # This split is justified by independently verified frozen source order,
        # not inferred from client receive times or sorted request identifiers.
        prefills,decodes=ids[:pf],ids[pf:]
        require((pf>0 and tokens>dc) or (pf==0 and tokens==dc),'phase/token accounting differs')
        if pf:prefill_tokens+=tokens-dc
        for rid in prefills:
            require(counts[rid]==0,'prefill/recompute after request began decode; context proof unsupported')
            pf_seen.add(rid)
        contexts={}
        for rid in decodes:
            require(rid in pf_seen,'request decode begins before recorded prefill')
            counts[rid]+=1;ordinal=counts[rid]
            require(ordinal<=outputs[rid]-1,'duplicate/extra owner decode step for request')
            contexts[rid]=dict(decode_ordinal=ordinal,computed_before=lengths[rid]+ordinal-1,
                               attention_after=lengths[rid]+ordinal)
        item=dict(owner_event_index=index,canonical_event_sha256=hashlib.sha256(json.dumps(event,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            started_s=start,finished_s=end,prefill_request_ids=prefills,decode_request_ids=decodes,logical_context=contexts)
        steps.append(item)
        if pf==0 and dc==batch and set(decodes)==set(requests):run.append(item)
        else:
            if run:runs.append(run)
            run=[]
    if run:runs.append(run)
    require(pf_seen==set(requests) and prefill_tokens==sum(lengths.values()),'complete prefills missing, recomputed or altered')
    require(all(counts[rid]==outputs[rid]-1 for rid in requests),'each real request needs its full decode tail; aggregate total is insufficient')
    eligible=[r for r in runs if len(r)>=65]
    require(eligible,'no consecutive full-batch run of65 actual owner steps for late-window export')
    late=eligible[-1][-65:]
    spacings=[b['finished_s']-a['finished_s'] for a,b in zip(late,late[1:])]
    per_request={}
    for rid in requests:
        values=[r['logical_context'][rid] for r in late]
        per_request[rid]=dict(input_tokens=lengths[rid],prescribed_output_tokens=outputs[rid],
            total_observed_decode_steps=counts[rid],computed_before_min=min(v['computed_before'] for v in values),
            computed_before_max=max(v['computed_before'] for v in values),attention_after_min=min(v['attention_after'] for v in values),
            attention_after_max=max(v['attention_after'] for v in values),
            complete_work_attention_upper=lengths[rid]+outputs[rid]-1,
            declared_bucket_edge=lengths[rid]+outputs[rid])
    p95=sorted(spacings)[math.ceil(.95*len(spacings))-1]
    return dict(schema=1,valid=True,source_order_verified=True,profile_point_generated=False,
        scope='source-order reconstructed logical token context; not directly observed physical KV',
        physical_kv_blocks_observed=False,entire_declared_context_domain_covered=False,
        full_owner_event_count=len(events),full_event_sequence_used=True,per_request=per_request,
        complete_prefill_token_sum=prefill_tokens,complete_decode_token_sum=sum(counts.values()),
        late_window=dict(start_s=late[0]['started_s'],end_s=late[-1]['finished_s'],steps=65,finish_spacings=64,
            request_ids=sorted(requests),owner_event_indices=[r['owner_event_index'] for r in late],
            logical_context_by_step=late,owner_wall_sum_s=sum(r['finished_s']-r['started_s'] for r in late),
            continuous_span_s=late[-1]['finished_s']-late[0]['started_s'],
            finish_spacing_values_s=spacings,finish_spacing_mean_s=statistics.mean(spacings),
            finish_spacing_p95_s=p95,finish_spacing_max_s=max(spacings),
            owner_step_mean_s=statistics.mean(r['finished_s']-r['started_s'] for r in late),
            owner_step_max_s=max(r['finished_s']-r['started_s'] for r in late)),
        empirical_latency_observations_only=True,certified_future_latency_upper=False,
        uncertainty='Repeated empirical envelope and independent serving validation still required before planner registration')


def attach_power(result,power,clocks,*,target_gpu,frequency_mhz):
    from ecopadg.metrics import clip_power_window
    from ecopadg.measure.power import trapezoid_energy
    w=result['late_window'];start,end=w['start_s'],w['end_s'];window=clip_power_window(power,start,end,pad_s=0)
    require(all(len(v)==8 and all(finite(x) and x>=0 for x in v) for _,v in window),'late-window eight-board power missing')
    require(clocks and all(finite(t) for t,_ in clocks) and all(b[0]>a[0] for a,b in zip(clocks,clocks[1:]))
        and clocks[0][0]<=start<end<=clocks[-1][0],'clock stream does not bracket the complete late window')
    active=[(t,v) for t,v in clocks if start<=t<=end]
    require(len(active)>=3 and all(len(v)==8 and all(finite(x) and x>=0 for x in v) for _,v in active),'late-window actual eight-board clock observations missing')
    target=[v[target_gpu] for _,v in active]
    require(all(abs(v-frequency_mhz)<=15 for v in target),'actual target frequency differs during late window')
    target_j=trapezoid_energy([(t,[v[target_gpu]]) for t,v in window])
    w.update(all_eight_gpu_energy_j=trapezoid_energy(window),target_gpu=target_gpu,target_gpu_energy_j=target_j,
        target_gpu_mean_power_w=target_j/(end-start),energy_includes_interstep_gaps=True,
        energy_is_not_net_incremental=True,frequency_command_mhz=frequency_mhz,
        actual_target_clock_min_mhz=min(target),actual_target_clock_max_mhz=max(target),
        clock_samples=len(active),clock_sample_timestamps_s=[t for t,_ in active])
    return result
