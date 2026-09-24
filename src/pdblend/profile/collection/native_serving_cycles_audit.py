"""Raw request-cycle audit, with exact token/forward ownership and eight-board energy.

Sampled scheduler occupancy is explicitly distinct from decode batch and CUDA
duty. No averaged whole-cycle power is converted into a pure-role profile node.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

from pdblend.bench.comparison_metering import summarize_comparison
from pdblend.online.native_control import validate_state
from pdblend.results.journal import payload_receipt
from pdblend_runtime.probe import NativeSpec
from .native_timing_audit import need,finite
from .native_timing_plan import digest
from .native_timing_collect import WORKER


class CycleFrequencyQualificationError(ValueError):
    """All other evidence passed; complete observations reject this sample."""
    def __init__(self, audited, evidence):
        message = 'cycle fleet observed frequency differs from requested clock'
        super().__init__(message)
        self.audit = dict(audited, passed=False, errors=[message],
            error_kind='expected_measurement_qualification_gap',
            qualification_gap='observed_frequency_mismatch',
            non_frequency_checks_passed=True, frequency_data_complete=True,
            frequency_evidence=evidence)


def _continuation_protocol(raw, specs):
    """Require real initial clock/all-rank ACKs before allowing recovery."""
    start, end, tail = raw['service_started_s'], raw['service_end_s'], raw['tail_end_s']
    for iid, spec in specs.items():
        before, after = raw['before'][iid], raw['after'][iid]
        need(before.get('acknowledged') is True and before.get('drained') is True
             and after.get('acknowledged') is True and after.get('drained') is True
             and before['native_at_s'] <= start and end <= after['native_at_s'] <= tail,
             'cycle continuation lacks ordered initial/final drain ACK')
        ranks = raw['measurement_start'][iid].get('ranks', [])
        need(len(ranks) == spec.tp and {r.get('rank') for r in ranks} == set(range(spec.tp))
             and all(r.get('acknowledged') is True and r.get('system') == 'pdblend'
                     and r.get('scope') == 'runner' and r.get('tp') == spec.tp
                     and r.get('pp') == spec.pp for r in ranks),
             'cycle continuation lacks all-rank measurement arm ACK')
        clock = raw['clocks'][iid]; ack = clock['ack']; frequency = raw['point']['frequency_mhz']
        need(ack.get('acknowledged') is True and ack.get('success') is True
             and ack.get('requested_frequency_mhz') == frequency
             and sorted(g.get('gpu_uuid', '') for g in ack.get('gpus', []))
                 == sorted(raw['capabilities'][iid]['gpu_uuids']),
             'cycle continuation requested clock ACK differs')
        observations = clock['observations']
        need(observations and all(finite(r.get('at_s')) and before['native_at_s'] <= r['at_s'] <= start
             and len(r['frequencies_mhz']) == spec.tp
             and all(finite(f) and f > 0 for f in r['frequencies_mhz']) for r in observations),
             'cycle continuation initial clock observations missing or unordered')
        need(all(abs(f-frequency) <= 15 for f in observations[-1]['frequencies_mhz']),
             'cycle continuation requested clock was never initially observed')


def occupancy(rows,start,end):
    """Piecewise constant scheduler samples; >1s is missing, never idle zero."""
    ordered=sorted(rows,key=lambda r:r['state']['native_at_s'])
    stamps=[r['state']['native_at_s'] for r in ordered]
    need(len(stamps)>=2 and all(b>a for a,b in zip(stamps,stamps[1:])), 'nonmonotone native occupancy samples')
    covered=0.;histogram=defaultdict(float);gaps=[]
    for left,right in zip(ordered,ordered[1:]):
        t,u=left['state']['native_at_s'],right['state']['native_at_s'];a,b=max(t,start),min(u,end)
        if b<=a:continue
        gaps.append(u-t)
        if u-t>1.:continue
        count=len(left['state']['running']);histogram[count]+=b-a;covered+=b-a
    duration=end-start
    return dict(coverage_fraction=covered/duration,max_gap_s=max(gaps,default=None),
        time_by_running_requests_s={str(k):v for k,v in sorted(histogram.items())},
        mean_running_requests=sum(k*v for k,v in histogram.items())/covered if covered else None,
        averaging_denominator_s=covered,window_s=duration,
        semantics='sampled_native_scheduler_running_requests_not_decode_batch',missing_filled_with_zero=False)


def _audit(raw,plan,*,trace_builder=None):
    from .native_serving_cycles import cycle_trace
    point=raw['point'];need(point in plan['points'],'cycle point is absent from immutable design')
    need(raw.get('schema')=='pdblend-native-request-cycle-window/v1' and raw.get('status')=='measured'
         and raw.get('hardware_executed') is True and not raw.get('cleanup_errors')
         and raw.get('plan_sha256')==digest(plan),'request-cycle collection incomplete or plan differs')
    trace=(trace_builder or cycle_trace)(plan,point);need(raw['trace']==trace,'real request-cycle trace does not reproduce')
    start=raw['service_started_s'];end=start+point['duration_s'];tail=raw['tail_end_s']
    need(finite(start) and finite(tail) and end==raw['service_end_s'] and tail>=end,
         'real fixed service and drain-tail interval missing')
    if point['purpose']=='holdout':need(raw.get('candidate'),'held-out native workload lacks frozen candidate binding')
    else:need(raw.get('candidate') is None,'training was contaminated by a holdout candidate')
    uuids=raw['lease']['gpu_uuids'];gpu_ids=raw['lease']['gpu_ids']
    need(len(set(uuids))==len(uuids)==len(set(gpu_ids))==len(gpu_ids)==8 and
         raw['lease'].get('gpu_uuid_binding_verified') is True,'cycle power needs all eight physical UUIDs')
    launches=raw['actual_launch'];specs={r['spec']['instance_id']:NativeSpec(**r['spec']) for r in launches}
    need(len(launches)==len(specs)==plan['resident_instances'] and sorted(g for s in specs.values() for g in s.gpus)==sorted(gpu_ids),
         'cycle launch inventory must cover eight physical boards once')
    clients=raw['client_requests'];requests={r['req_id']:r for r in trace['requests']}
    need(len(clients)==len(requests) and {r['req_id'] for r in clients}==set(requests),
         'request-cycle cohort missing/duplicate outcomes')
    byid={r['request_id']:r for r in clients};need(len(byid)==len(clients),'duplicate native request IDs')
    for client in clients:
        request=requests[client['req_id']];events=client['events']
        need(client['instance_id'] in specs and client.get('terminal') is True and not client.get('error')
             and client['completion_tokens']==request['max_tokens'] and events
             and client['scheduled_s']==start+request['arrival_s']
             and client['scheduled_s']<=client['submitted_s']<=client['finished_s']<=tail,
             'native cycle request failed, timed out or has wrong token count/time')
        receipt=payload_receipt(events,journal_path='embedded:events',request_id=client['request_id'])
        need(all(client.get(k)==v for k,v in receipt.items()),'native cycle token receipt differs from complete journal')
        count=0;previous=client['submitted_s']
        for index,event in enumerate(events):
            count+=len(event['token_ids'])
            need(event.get('token_index')==count and previous<=event['received_s']<=client['finished_s']
                 and bool(event.get('finished'))==(index==len(events)-1),
                 'native token stream reordered, incomplete, or has early terminal')
            previous=event['received_s']
    # Independently replay actual least-load acquire/release decisions.
    counts={iid:0 for iid in specs};active={};acquired=set();released=set();stamp=start
    for route in raw['routes']:
        rid,iid=route['request_id'],route['instance_id'];need(rid in byid and iid==byid[rid]['instance_id']
            and stamp<=route['at_s']<=tail,'route ownership/time differs');stamp=route['at_s']
        if route['event']=='acquire':
            need(rid not in acquired and route['before']==counts and iid==min(counts,key=lambda k:(counts[k],k)),
                 'native request-cycle routing is not complete least-load')
            counts[iid]+=1;active[rid]=iid;acquired.add(rid)
        else:
            need(route['event']=='release' and active.pop(rid,None)==iid,'routing release missing its acquisition')
            counts[iid]-=1;need(route['after']==counts,'routing counter release differs');released.add(rid)
    need(acquired==released==set(byid) and not active and not any(counts.values()),'routing counts did not fully return')
    identities=[];occupancies={};cuda=[]
    for launch in launches:
        spec=specs[launch['spec']['instance_id']];iid=spec.instance_id;cap=raw['capabilities'][iid]
        need(spec.tp==plan['tp'] and spec.pp==1 and spec.max_num_seqs==32 and spec.max_model_len==8192
             and spec.kv_connector=='P2pNcclConnector' and WORKER in spec.extra_args
             and Path(spec.model).name==plan['model_id'] and launch['argv'][1:]==spec.command()[1:],
             'cycle actual native worker/topology/launch differs')
        need(cap.get('supported') is True and cap.get('model_id')==plan['model_id'] and cap.get('tp')==plan['tp']
             and cap.get('pp')==1 and cap.get('gpu_uuids')==[uuids[gpu_ids.index(g)] for g in spec.gpus],
             'cycle native capability physical/model identity differs')
        identity={k:cap.get(k) for k in ('model_id','tp','pp','model_hash','tokenizer_hash','image_digest','source_revision','engine_revision')}
        need(all(identity.values()),'native cycle identity incomplete');identities.append(identity)
        validate_state(raw['before'][iid],generation=spec.generation,tp=spec.tp,drained=True)
        validate_state(raw['after'][iid],generation=spec.generation,tp=spec.tp,drained=True,observed_after_s=end)
        resumed=raw['resume'][iid];validate_state(resumed['state'],generation=spec.generation,tp=spec.tp,drained=True)
        need(resumed['control'].get('acknowledged') is True and all(resumed['state'].get(k)==v for k,v in
             dict(role='mixed',mode='temporal',accepting=True,admit_prefill=True,admit_decode=True).items()),
             'cycle initial role/admission differs')
        need(raw['measurement_start'][iid].get('acknowledged') is True,'native cycle measurement start lacks ACK')
        stopped=raw['measurement_stop'][iid].get('ranks',[])
        need(len(stopped)==spec.tp and {r.get('rank') for r in stopped}==set(range(spec.tp))
             and all(r.get('acknowledged') is True for r in stopped),'native cycle stop missing a physical rank')
        observations=[r for r in raw['state_observations'] if r['instance_id']==iid]
        owned={r for r,c in byid.items() if c['instance_id']==iid}
        for row in observations:
            validate_state(row['state'],generation=spec.generation,tp=spec.tp)
            need(set(row['state']['all_queue'])<=owned and len(row['state']['running'])<=32,
                 'native cycle state has extraneous work or exceeds max_num_seqs')
        occupancies[iid]=occupancy(observations,start,end)
        need(occupancies[iid]['coverage_fraction']>=1.-1e-9,'sampled occupancy has an uncovered interval')
        ranks=sorted(raw['samples'][iid].get('ranks',[]),key=lambda r:r.get('rank',-1))
        need(len(ranks)==spec.tp and [r.get('rank') for r in ranks]==list(range(spec.tp)),
             'request-cycle native CUDA rank inventory incomplete')
        key=lambda r:tuple(tuple(r[k]) if isinstance(r[k],list) else r[k] for k in
            ('role','batch','request_ids','prompt_lengths','context_lengths','scheduled_lengths'))
        need(all([key(r) for r in rank['samples']]==[key(r) for r in ranks[0]['samples']] for rank in ranks),
             'native request-cycle per-rank shape sequence differs')
        progress={rid:0 for rid in owned}
        for rank in ranks:
            for event in rank['samples']:
                need(event.get('system')=='pdblend' and event.get('measurement_scope')=='runner'
                     and event.get('rank')==rank['rank'] and event.get('tp')==spec.tp and event.get('pp')==1
                     and not event.get('failed') and finite(event.get('gpu_elapsed_ms')) and event['gpu_elapsed_ms']>0
                     and start<=event['at_s']<=tail and event['role'] in ('prefill','decode','mixed'),
                     'native request-cycle CUDA event identity/time differs')
        for index,event in enumerate(ranks[0]['samples']):
            ids=event['request_ids'];batch=event['batch']
            need(type(batch)is int and 1<=batch<=32 and len(ids)==len(set(ids))==batch and set(ids)<=owned
                 and all(len(event[k])==batch for k in ('prompt_lengths','context_lengths','scheduled_lengths')),
                 'native cycle forward ownership/shape vectors incomplete')
            for rid,prompt,context,scheduled in zip(ids,event['prompt_lengths'],event['context_lengths'],event['scheduled_lengths']):
                expected=requests[byid[rid]['req_id']]
                need(prompt==len(expected['prompt']) and type(scheduled)is int and scheduled>0
                     and context==progress[rid]+scheduled and context<=8192,
                     'native request-cycle forward progression or real prompt differs')
                progress[rid]=context
            cuda.append(dict(instance_id=iid,at_s=event['at_s'],role=event['role'],batch=batch,
                context_lengths=event['context_lengths'],scheduled_lengths=event['scheduled_lengths'],
                max_rank_latency_ms=max(r['samples'][index]['gpu_elapsed_ms'] for r in ranks)))
        need(all(progress[rid]==len(requests[byid[rid]['req_id']]['prompt'])+requests[byid[rid]['req_id']]['max_tokens']-1
                 for rid in owned),'native CUDA journal does not cover the complete successful cohort')
    need(all(identity==identities[0] for identity in identities),'native replica model/source identity differs')
    power=raw['power'];measured=summarize_comparison(power,gpu_uuids=uuids,origin_s=start,duration_s=end-start,
        tail_end_s=tail,gpu_uuid_binding_verified=True)
    need(measured['energy_comparable'] is True,'complete eight-board service/tail energy is unavailable')
    from .native_power_audit import _frequency_evidence
    frequency = _frequency_evidence(power, dict(gpus=gpu_ids), point, start, tail)
    distribution={}
    for row in cuda:
        if not start<=row['at_s']<end:continue
        key=(row['instance_id'],row['role'],row['batch'])
        values=distribution.setdefault(key,dict(instance_id=key[0],role=key[1],scheduled_batch=key[2],
            events=0,max_rank_cuda_ms=0.,context_min=min(row['context_lengths']),context_max=max(row['context_lengths'])))
        values['events']+=1;values['max_rank_cuda_ms']+=row['max_rank_latency_ms']
        values['context_min']=min(values['context_min'],min(row['context_lengths']))
        values['context_max']=max(values['context_max'],max(row['context_lengths']))
    audited = dict(passed=True,identity=identities[0],requests=len(clients),native_cuda_events=len(cuda),
        measured=measured,occupancy_by_replica=occupancies,
        scheduled_cuda_event_distribution=list(distribution.values()),
        cuda_event_distribution_scope='actual_events_only_not_wall_clock_decode_batch_or_pure_role_power',
        scope='all_mixed_complete_request_cycle_calibration',power_scope=raw['power_scope'],
        pure_decode_nodes_created=False,prefill_kernel_power_qualified=False,
        formal_eligible=False,full_profile_qualified=False,parameters_fitted=False)
    # A typed rejection is possible only after every non-frequency check,
    # including the distribution replay above, and the recovery protocol pass.
    if frequency['mismatch_count']:
        _continuation_protocol(raw, specs)
        raise CycleFrequencyQualificationError(audited, frequency)
    return audited


def audit_cycle_window(raw,plan):
    try:return _audit(raw,plan)
    except CycleFrequencyQualificationError as exc:return exc.audit
    except (ValueError,RuntimeError,KeyError,TypeError,IndexError,OSError) as exc:
        return dict(passed=False,errors=[str(exc)],formal_eligible=False,full_profile_qualified=False)
