"""Build phase-separated measured tables from held real-KV profiling.

Only full observed decode batches enter the table. Instant phase integration
requires time-stamped sample support on every participating GPU; unsupported
windows use the enforced device limit. Legacy average-source handling remains
separate. Sample support does not establish the sensor update period or its
accuracy. Independent workload calibration remains required.
"""
import argparse
from bisect import bisect_left,bisect_right
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from ecopadg.measure.power import trapezoid_mean_power
from ecopadg.metrics import clip_power_window
from .profiles import ProfilePoint,validate_profile_observations
from .interconnect import InterconnectTopology
from .measurement import power_evidence


def quantile(values,q):
    return sorted(values)[min(len(values)-1,int(q*len(values)))]


PHASE_POWER_RESOLUTION_POLICY = dict(schema=1,
    instant_min_distinct_timestamps_per_gpu=2,
    instant_max_unobserved_fraction=.5,
    legacy_short_phase_s=.2,
    timestamp_semantics='NVML CPU timestamps; distinct readings are not independent sensor updates')


def instant_phase_coverage(timestamps,gpus,start,end):
    """Check sampling geometry without consulting workload or validation error.

    Each GPU needs two distinct in-window NVML timestamps, with no unsupported
    interval (including either phase boundary) longer than half the phase.
    The caller has already verified source identity and timestamp freshness.
    Constant power readings are valid; distinct values are not required.
    """
    duration=end-start
    if duration<=0: raise ValueError('positive phase duration required')
    per_gpu=[]
    for gpu in gpus:
        series=timestamps[gpu]
        interior=list(dict.fromkeys(series[bisect_right(series,start):bisect_left(series,end)]))
        supported=[start,*interior,end]
        gap=max(b-a for a,b in zip(supported,supported[1:]))
        per_gpu.append(dict(gpu=gpu,distinct_in_window_timestamps=len(interior),
            first_timestamp_s=interior[0] if interior else None,
            last_timestamp_s=interior[-1] if interior else None,
            maximum_unobserved_gap_s=gap))
    supported=bool(per_gpu) and all(
        row['distinct_in_window_timestamps']>=PHASE_POWER_RESOLUTION_POLICY['instant_min_distinct_timestamps_per_gpu']
        and row['maximum_unobserved_gap_s']<=duration*PHASE_POWER_RESOLUTION_POLICY['instant_max_unobserved_fraction']
        for row in per_gpu)
    return dict(sampling_supported=supported,duration_s=duration,per_gpu=per_gpu)


def settled_references(paths,operator):
    """Read independently measured idle power, without a modeling margin.

    A short operator prelude may still reflect an earlier clock or workload.
    Its power remains diagnostic; it is not a settled reference for a new
    frequency. The conservative residency margin is applied later by merge.
    """
    def identity(raw):
        result=set()
        for engine in raw.get('engine_provenance',()):
            source=next((value for name,value in engine.get('source_files_at_import',{}).items()
                         if name.endswith('/serving/engine.py')),None)
            key=(engine.get('image_id'),engine.get('model'),engine.get('engine_version'),source)
            if not all(key) or key[2]!='0.9.2':
                raise ValueError('settled reference requires complete matching engine provenance')
            result.add(key)
        if len(result)!=1:
            raise ValueError('settled reference requires one matching engine identity')
        return result
    references={};artifacts={}
    expected=identity(operator) if paths else None
    for path in paths:
        path=Path(path).resolve();data=path.read_bytes();raw=json.loads(data)
        if (raw.get('complete') is not True or raw.get('sampling_error')
                or raw.get('wakeup',{}).get('passed') is not True
                or identity(raw)!=expected):
            raise ValueError('incomplete or mismatched independently measured settled reference')
        digest=hashlib.sha256(data).hexdigest();artifacts[str(path)]=digest
        for index,item in enumerate(raw.get('residency',())):
            if item.get('parked') is True:continue
            tp=item.get('tp');frequency=item.get('frequency_mhz');watts=item.get('watts')
            if (item.get('parked') is not False or type(tp) is not int or tp not in (1,2,4,8)
                    or type(frequency) is not int or frequency<=0
                    or isinstance(watts,bool) or not isinstance(watts,(int,float))
                    or not math.isfinite(watts) or watts<0):
                raise ValueError('invalid measured settled residency point')
            key=(tp,frequency)
            if key not in references or watts>references[key]['watts']:
                references[key]=dict(tp=tp,frequency_mhz=frequency,watts=watts,
                    source_path=str(path),source_sha256=digest,sample_index=index)
    return references,dict(method='maximum independently measured settled power; no model margin',
                           artifacts=artifacts,points=[references[key] for key in sorted(references)])


def build(raw_path,interconnect=None,*,residency_paths=()):
    data=raw_path.read_bytes();raw=json.loads(data)
    digest=hashlib.sha256(data).hexdigest()
    if raw.get('schema')!=3 or not raw.get('complete') or raw.get('sampling_error'):
        raise ValueError('requires complete hardware profiling with valid power sampling')
    power_provenance=power_evidence(raw['power_samples'],raw.get('power_source'),raw.get('power_metadata'))
    if raw.get('power_source',{}).get('mode')=='instant' and not power_provenance['power_source_verified']:
        raise ValueError('instant profile requires verified per-sample power metadata')
    references,reference_evidence=settled_references(residency_paths,raw)
    if digest in reference_evidence['artifacts'].values():
        raise ValueError('settled residency must be an independent measurement')
    topology=raw['topology'];power=raw['power_samples'];groups=defaultdict(list)
    if not raw.get('frequency_samples'):
        raise ValueError('operator profiles require observed SM clock samples')
    power_times=[t for t,_ in power]
    instant_timestamps=({g:[row['nvml_timestamp_us'][g]/1e6 for row in raw['power_metadata']]
                         for g in range(8)} if power_provenance['power_source_verified'] else None)
    frequencies=raw['frequency_samples'];frequency_times=[t for t,_ in frequencies]
    transfers=defaultdict(list);phase_sources=[];idle_node=[]
    kv=[]
    for instance_id in {i['id'] for i in topology.values()}:
        path=Path(raw['runtime_dir'])/(instance_id+'.control.json.kv.0.jsonl')
        if path.exists():
            kv.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    by_request=defaultdict(list)
    for event in kv:
        for rid in event['request_ids']: by_request[(event['engine_id'],rid)].append(event)
    def watts(gpus,start,end):
        left=max(0,bisect_right(power_times,start)-1)
        right=min(len(power),bisect_left(power_times,end)+1)
        rows=clip_power_window(power[left:right],start,end,pad_s=0)
        return trapezoid_mean_power([(t,[ws[g] for g in gpus]) for t,ws in rows])
    def observed_clocks(gpus,start,end):
        rows=frequencies[bisect_left(frequency_times,start):bisect_right(frequency_times,end)]
        values=[clocks[g] for _,clocks in rows for g in gpus]
        return dict(samples=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values)) if values else None
    def phase_watts(gpus,start,end,idle,local_idle,reference):
        measured=watts(gpus,start,end)
        reasons=[]
        evidence=dict(started_s=start,finished_s=end,integrated_power_w=measured,residency_w=idle,
                      local_prelude_power_w=local_idle,residency_reference=reference,
                      signed_power_above_reference_w=measured-idle,
                      below_reference=measured<idle)
        if instant_timestamps is not None:
            coverage=instant_phase_coverage(instant_timestamps,gpus,start,end)
            evidence.update(coverage)
            if not coverage['sampling_supported']: reasons.append('insufficient_instant_sample_support')
        elif end-start<PHASE_POWER_RESOLUTION_POLICY['legacy_short_phase_s']:
            reasons.append('legacy_average_short_phase')
        # An independent maximum idle reading is not a physical lower bound.
        # Keep supported instantaneous observations unchanged, even below that
        # reference; only the later training model receives an idle floor.
        if measured<idle and (reference is None or instant_timestamps is None):
            reasons.append('integrated_power_below_residency')
        evidence['fallback_reasons']=reasons
        if reasons:
            return sum(raw['power_limit_w'][g] for g in gpus),'enforced_device_limit_upper_bound',evidence
        return measured,('integrated_nvml_instant' if instant_timestamps is not None else 'integrated_nvml'),evidence
    for run in raw['runs']:
        if run['skipped']: continue
        n,b,f,out=run['input_tokens'],run['batch'],run['frequency_mhz'],run['output_tokens']
        commanded=run.get('commanded_frequencies',{})
        if any(commanded.get(str(g),commanded.get(g))!=f for i in topology.values() for g in i['gpus']):
            raise ValueError('requested profile frequency differs from the clock command')
        idle_start,idle_end=run['idle_start_s'],run['idle_end_s']
        idle_node.append(watts(range(8),idle_start,idle_end))
        for role in (('mixed',) if run['layout']=='mixed' else ('prefill','decode')):
            instance=topology[role];gpus=instance['gpus']
            local_idle=watts(gpus,idle_start,idle_end)
            reference=references.get((instance['tp'],f))
            if residency_paths and reference is None:
                raise ValueError('missing independently measured settled TP/frequency reference')
            idle=reference['watts'] if reference else local_idle
            events=[e for e in run['events'] if e['instance']==instance['id'] and e['tokens']>0]
            prefills=[e for e in events if e['prefill']==1 and e['decode']==0]
            decodes=[e for e in events if e['decode']==b and e['prefill']==0]
            if len(prefills)!=b or (role!='prefill' and len(decodes)!=out-1):
                raise ValueError('missing observed prefill/import or full decode batch')
            prefill_durations=[];prefill_powers=[]
            for event in prefills:
                start,end=event['started_s'],event['finished_s']
                if role in ('prefill','decode'):
                    hits=by_request[(instance['id'],event['request_ids'][0])]
                    if len(hits)!=1: raise ValueError('missing exact request transfer interval')
                    transfer=hits[0]
                    direction='send' if role=='prefill' else 'receive'
                    # Transfer energy includes only each endpoint's dynamic
                    # power. Residency is counted once by the node objective.
                    xw=watts(gpus,transfer['started_s'],transfer['finished_s'])
                    transfers[(n,instance['tp'],direction)].append(dict(
                        seconds=transfer['finished_s']-transfer['started_s'],
                        joules=max(0,xw-idle)*(transfer['finished_s']-transfer['started_s'])))
                    if role=='prefill': end=transfer['started_s']
                if role!='decode':
                    prefill_durations.append(end-start)
                    w,source,evidence=phase_watts(gpus,start,end,idle,local_idle,reference)
                    prefill_powers.append(w)
                    phase_sources.append(dict(role=role,tp=instance['tp'],frequency_mhz=f,
                        input_tokens=n,batch=b,phase='prefill',source=source,power_evidence=evidence,
                        observed_sm_mhz=observed_clocks(gpus,start,end)))
            prefill_s=max(prefill_durations,default=0)
            pw=max(prefill_powers,default=idle)
            iteration=0.;dw=idle
            if role!='prefill':
                # Late-context inter-step spacing includes the engine adapter,
                # scheduler and periodic transport telemetry between kernels.
                tail=decodes[-min(65,len(decodes)):]
                spacings=[y['finished_s']-x['finished_s'] for x,y in zip(tail,tail[1:])]
                iteration=quantile(spacings,.95) if spacings else tail[0]['finished_s']-tail[0]['started_s']
                dw,source,evidence=phase_watts(gpus,tail[0]['started_s'],tail[-1]['finished_s'],idle,local_idle,reference)
                phase_sources.append(dict(role=role,tp=instance['tp'],frequency_mhz=f,
                    input_tokens=n,batch=b,phase='decode',source=source,power_evidence=evidence,
                    observed_sm_mhz=observed_clocks(gpus,tail[0]['started_s'],tail[-1]['finished_s'])))
            # P handles one real request at a time; its point must not claim
            # the held target decode batch was a source prefill batch.
            measured_batch=1 if role=='prefill' else b
            prefill_work=prefill_s*(b if role=='mixed' else 1)
            duration=prefill_work+max(out-1,0)*iteration
            mean_power=(pw*prefill_work+dw*max(out-1,0)*iteration)/duration if duration else idle
            # A P worker only materializes the prompt and first output token.
            # Target decode work must not move the source measurement into a
            # larger context bucket than the online prefill query.
            measured_context=n+1 if role=='prefill' else n+out
            groups[(role,instance['tp'],f,n,measured_context,measured_batch)].append(dict(
                prefill_s=prefill_s,iteration_s=iteration,power_w=mean_power,residency_w=idle,
                prefill_power_w=pw,decode_power_w=dw))
    points=[]
    for (role,tp,f,n,context,b),samples in sorted(groups.items()):
        means={k:statistics.mean(s[k] for s in samples) for k in samples[0]}
        error=max([.05]+[max(s[k] for s in samples)/means[k]-1
                         for k in ('prefill_s','iteration_s') if means[k]>0])
        energy_error=max([.05]+[max(s[k] for s in samples)/means[k]-1
                         for k in ('power_w','prefill_power_w','decode_power_w') if means[k]>0])
        points.append(ProfilePoint(role,tp,f,n,context,b,**means,error_fraction=error,
            samples=len(samples),source_sha256=digest,
            interference_s=means['prefill_s'] if role=='mixed' else 0,
            energy_error_fraction=energy_error))
    validate_profile_observations(points)
    links=[]
    if 'decode' in topology:
        for n in sorted({r['input_tokens'] for r in raw['runs'] if not r['skipped'] and r['layout']=='pd'}):
            source,target=topology['prefill']['tp'],topology['decode']['tp']
            send=transfers[(n,source,'send')];receive=transfers[(n,target,'receive')]
            if not send or not receive: raise ValueError('incomplete measured transfer direction')
            sg,dg=topology['prefill']['gpus'],topology['decode']['gpus']
            links.append(dict(source_tp=source,target_tp=target,max_input_tokens=n,
                seconds_upper=1.1*(max(x['seconds'] for x in send)+max(x['seconds'] for x in receive)),
                import_seconds_upper=1.1*max(x['seconds'] for x in receive),
                incremental_j=max(x['joules'] for x in send)+max(x['joules'] for x in receive),
                source_sha256=digest,validated=True,profile_batch=1,
                source_gpus=sg,target_gpus=dg,
                interconnect_class=interconnect.link_class(sg,dg) if interconnect else '',
                topology_sha256=interconnect.source_sha256 if interconnect else ''))
    return dict(schema=2,measurement='hardware',model='Qwen2.5-14B-Instruct',
        power_source_verified=power_provenance['power_source_verified'],
        power_source=raw.get('power_source',dict(mode='legacy_average',
            api='nvmlDeviceGetPowerUsage',averaging_window_s=1.,
            note='L20 legacy source; not instantaneous power')),
        purpose='development_operator_profiles',source_sha256=digest,
        node_residency_w=max(idle_node,default=0),
        gpu_count=8,unallocated_idle_measured=False,
        unallocated_idle_note='Operator topology lists measured roles, not all physically resident engines; use separate proven empty-GPU evidence',
        heldout_calibration_complete=False,mixed_interference_measured=False,
        frequency_commands_verified=True,frequency_samples_source_sha256=digest,
        phase_power_resolution_policy=dict(PHASE_POWER_RESOLUTION_POLICY),
        residency_reference=reference_evidence,
        phase_power_sources=phase_sources,points=[asdict(p) for p in points],
        profiling_method='sequential actual prefill/import, held real KV, full decode batch; late-context spacing'),links


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--profiles',type=Path,required=True)
    p.add_argument('--transfers',type=Path,required=True)
    p.add_argument('--interconnect',type=Path)
    p.add_argument('--residency',type=Path,nargs='+',default=(),
                   help='Independent settled residency raws; excludes the model safety margin')
    args=p.parse_args()
    if args.profiles.exists() or args.transfers.exists(): p.error('refusing to overwrite evidence')
    topology=InterconnectTopology.parse(args.interconnect.read_text()) if args.interconnect else None
    profiles,transfers=build(args.raw,topology,residency_paths=args.residency)
    args.profiles.write_text(json.dumps(profiles,indent=2))
    args.transfers.write_text(json.dumps(transfers,indent=2))
    print(json.dumps(dict(points=len(profiles['points']),links=len(transfers))))


if __name__=='__main__': main()
