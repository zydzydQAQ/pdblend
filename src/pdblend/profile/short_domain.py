"""Experimental short-workload fits; no promotion of segmented power to decode.

Timing is HTTP/SSE timing, with every original token arrival preserved. Board
power belongs to complete repeated request windows. Short decode-only sensor
windows cannot establish the original continuous five-second power contract.
"""
from __future__ import annotations
import math
import statistics
import numpy as np

FREQUENCIES=(900,1200,1500,1800,2100,2520)
KIND='short_request_segmented_timing_and_mixed_power_v1'
IDENTITY=('system','model_id','model_hash','tokenizer_hash','tp','pp')


def make_plan(identity):
    if identity['system']!='pdblend' or identity['pp']!=1 or (identity['model_id'],identity['tp']) not in (
        ('Qwen2.5-7B-Instruct',1),('Qwen2.5-14B-Instruct',1),('Qwen2.5-32B-Instruct',2)):
        raise ValueError('short panel requires independent own minimum feasible PP1 topology')
    def points(purpose,prefill,decode):
        return [dict(purpose=purpose,role=role,freq_mhz=f,input_tokens=n,output_tokens=o,batch=1,
            repeats=3 if o>1 else 1,settle_s=2.0,measure_s=5.0,decode_accumulation_s=5.0 if o>1 else 0.0)
            for f in FREQUENCIES for role,inputs,o in (('prefill_timing',prefill,1),('short_mixed',decode,64)) for n in inputs]
    return dict(schema=1,kind=KIND,**{k:identity[k] for k in IDENTITY},training=points('training',(29,64,128),(29,224)),
        holdout=points('independent_holdout',(30,80,128),(29,128,224)),exact_batches=[1],
        output_work={'prefill_timing':1,'short_mixed':64},experimental_sampling=True,
        original_continuous_decode_power_protocol_passed=False,formal_eligible=False,
        fit_uses_holdout=False,seed=701,power_scope='complete repeated-request window including prefill and inter-request gaps',
        decode_power_scope='diagnostic segments only; unqualified sensor lag; never pure-decode calibration',
        minimum_window_seconds=5,decode_repeats=3,prefill_minimum_requests_per_window=3)


def point_key(point):return '/'.join(str(point[k]) for k in ('purpose','role','freq_mhz','input_tokens','output_tokens','batch'))


def summarize(evidence):
    """Rebuild summaries from real arrivals; reject shortened or incomplete work."""
    p=evidence['point'];requests=evidence['requests'];start,end=evidence['start_s'],evidence['end_s']
    if (end-start<p['measure_s'] or evidence['settle_end_s']-evidence['settle_start_s']<p['settle_s'] or
            not evidence['warmup_requests'] or len(requests)<3):raise ValueError('short panel stable window too small')
    ttfts=[];steps=[];segments=[];decode_s=0.;power=evidence['power'];frequency=evidence['frequency']
    for request in requests+evidence['warmup_requests']:
        times=request['token_times_s']
        if (request.get('error') or request.get('stream_done') is not True or request.get('usage_received') is not True or
            request['completion_tokens']!=p['output_tokens'] or len(times)!=p['output_tokens'] or
            any(not math.isfinite(t) for t in times) or any(b<a for a,b in zip(times,times[1:])) or
            not request['submitted_s']<=times[0]<=times[-1]<=request['finished_s']):
            raise ValueError('short workload SSE/token count or timestamp evidence is invalid')
    for request in requests:
        times=request['token_times_s']
        if request['submitted_s']<start or request['finished_s']>end:raise ValueError('request crosses saved measurement window')
        ttfts.append(times[0]-request['submitted_s'])
        for j in range(1,len(times)):steps.append(dict(context=p['input_tokens']+j,seconds=times[j]-times[j-1]))
        if len(times)>1:
            duration=times[-1]-times[0];decode_s+=duration
            samples=[x for x in power if times[0]<=x[0]<=times[-1]]
            clocks=[x for x in frequency if times[0]<=x[0]<=times[-1]]
            segments.append(dict(start_s=times[0],end_s=times[-1],decode_steps=len(times)-1,
                observed_context_min=p['input_tokens']+1,observed_context_max=p['input_tokens']+len(times)-1,
                power_samples=len(samples),frequency_samples=len(clocks),
                diagnostic_power_w=statistics.fmean(sum(v) for _,v in samples) if samples else None,
                power_sensor_lag_qualified=False,power_qualified=False))
    if p['output_tokens']>1 and (decode_s<p['decode_accumulation_s'] or any(x['decode_steps']<8 for x in segments)):
        raise ValueError('short decode accumulation too small')
    for samples,minimum in ((power,2),(frequency,1)):
        if (len(samples)<minimum or any(not start<=t<=end or len(v)!=evidence['gpu_count'] or
                any(not math.isfinite(x) or x<=0 for x in v) for t,v in samples) or
                any(b[0]<=a[0] for a,b in zip(samples,samples[1:]))):raise ValueError('invalid group sampler window')
    freq=statistics.fmean(v for _,values in frequency for v in values)
    if round(freq)!=p['freq_mhz']:raise ValueError('short window actual clock mismatch')
    return dict(ttft_seconds=statistics.median(ttfts),step_seconds=statistics.fmean(s['seconds'] for s in steps) if steps else None,
        effective_context=statistics.fmean(s['context'] for s in steps) if steps else None,
        observed_context_min=min((s['context'] for s in steps),default=None),observed_context_max=max((s['context'] for s in steps),default=None),
        mixed_power_w=statistics.fmean(sum(v) for _,v in power),mean_freq_mhz=freq,window_s=end-start,
        accumulated_decode_s=decode_s,requests=len(requests),segments=segments,
        decode_segments_have_samples=all(s['power_samples']>=2 and s['frequency_samples']>=1 for s in segments),
        pure_decode_power_qualified=False,experimental_sampling=True)


def fit(plan,rows):
    expected={point_key(p) for p in plan['training']}
    if set(rows)!=expected or any(len(r['repeats'])!=r['point']['repeats'] for r in rows.values()):
        raise ValueError('short fit requires complete training and no holdout')
    nodes={};prefill={};steps={}
    for f in FREQUENCIES:
        pres=sorted((r for r in rows.values() if r['point']['freq_mhz']==f and r['point']['role']=='prefill_timing'),key=lambda r:r['point']['input_tokens'])
        dec=sorted((r for r in rows.values() if r['point']['freq_mhz']==f and r['point']['role']=='short_mixed'),key=lambda r:r['point']['input_tokens'])
        prefill[str(f)]=[dict(input_tokens=r['point']['input_tokens'],seconds=statistics.fmean(x['summary']['ttft_seconds'] for x in r['repeats'])) for r in sorted(pres,key=lambda x:x['point']['input_tokens'])]
        # Fit one affine curve over actual mean decode contexts, not nominal prompts.
        x=[r['summary']['effective_context'] for row in dec for r in row['repeats']]
        y=[r['summary']['step_seconds'] for row in dec for r in row['repeats']]
        slope,intercept=np.polyfit(x,y,1)
        lower=min(r['summary']['observed_context_min'] for row in dec for r in row['repeats'])
        upper=max(r['summary']['observed_context_max'] for row in dec for r in row['repeats'])
        if min(intercept+slope*lower,intercept+slope*upper)<=0:raise ValueError('nonpositive short timing prediction')
        steps[str(f)]=dict(intercept=float(intercept),slope=float(slope),context=[lower,upper])
        nodes[str(f)]=[dict(input_tokens=r['point']['input_tokens'],power_w=statistics.fmean(x['summary']['mixed_power_w'] for x in r['repeats'])) for r in sorted(dec,key=lambda x:x['point']['input_tokens'])]
    return dict(schema=1,kind=KIND,**{k:plan[k] for k in IDENTITY},prefill_timing=prefill,decode_timing=steps,mixed_power=nodes,
        batch=1,mixed_output_tokens=64,holdout_used=False,experimental_sampling=True,pure_decode_power_qualified=False,
        formal_eligible=False,original_continuous_decode_power_protocol_passed=False)


def linear(nodes,x,key):
    if not nodes[0]['input_tokens']<=x<=nodes[-1]['input_tokens']:raise ValueError('short input outside measured nodes')
    for a,b in zip(nodes,nodes[1:]):
        if a['input_tokens']<=x<=b['input_tokens']:
            return a[key]+(x-a['input_tokens'])/(b['input_tokens']-a['input_tokens'])*(b[key]-a[key])
    raise ValueError('missing short node')


def audit(candidate,plan,rows):
    failures=[];timing=[];power=[];expected={point_key(p) for p in plan['holdout']}
    if set(rows)!=expected:failures.append(dict(kind='incomplete_holdout',missing=sorted(expected-set(rows))))
    for key,row in rows.items():
        if key not in expected:raise ValueError('foreign holdout row')
        if len(row['repeats'])!=row['point']['repeats']:
            failures.append(dict(kind='incomplete_repeats',key=key));continue
        p=row['point'];f=str(p['freq_mhz'])
        for rep in row['repeats']:
            value=rep['summary']
            if p['role']=='prefill_timing':
                predicted=linear(candidate['prefill_timing'][f],p['input_tokens'],'seconds');actual=value['ttft_seconds']
            else:
                shape=candidate['decode_timing'][f]
                if not shape['context'][0]<=value['observed_context_min']<=value['observed_context_max']<=shape['context'][1]:
                    failures.append(dict(kind='whole_context_window_outside_training',key=key));continue
                predicted=shape['intercept']+shape['slope']*value['effective_context'];actual=value['step_seconds']
                expected_power=linear(candidate['mixed_power'][f],p['input_tokens'],'power_w')
                power.append(abs(expected_power-value['mixed_power_w'])/value['mixed_power_w'])
            timing.append(abs(predicted-actual)/actual)
    t=max(timing,default=math.inf);pmax=max(power,default=math.inf);pmean=statistics.fmean(power) if power else math.inf
    if t>.10:failures.append(dict(kind='timing_max',value=t))
    if pmax>.15 or pmean>.10:failures.append(dict(kind='whole_request_power_error',maximum=pmax,mape=pmean))
    return dict(complete=set(rows)==expected and all(len(r['repeats'])==r['point']['repeats'] for r in rows.values()),experimental_components_passed=not failures,timing_max_relative=t,
        mixed_power_max_relative=pmax,mixed_power_mape=pmean,failures=failures,formal_eligible=False,
        original_continuous_decode_power_protocol_passed=False,pure_decode_power_qualified=False,
        missing_gates=['continuous_decode_power_window','sensor_lag_separation','native_cuda_timing_crosscheck','full_profile_quality_audit'])
