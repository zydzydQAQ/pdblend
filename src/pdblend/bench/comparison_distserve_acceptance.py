"""Independent native DistServe pair routing, KV and client-window acceptance."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .comparison_acceptance import _bound, _need, _finite, _equal, _state, _drained
from .comparison_native_acceptance import (native_topology, audit_native_startup,
    audit_native_reset, audit_native_metrics, audit_native_meter)
from .comparison_distserve_inputs import validate_distserve_inputs
from .comparison_meter_method import audit_isolated_meter_method
from .resident_session import digest, file_sha
from pdblend_baselines.distserve.policy import PrefillScheduler, Request


def binding(path):
    return dict(path=str(Path(path).resolve()),sha256=file_sha(path))


async def qualify_pairs(specs, out):
    """Outside service energy: same-model golden/cancel/recovery on each pair."""
    from pdblend_baselines.distserve.gpu_probe import run_resident
    refs=[];by_id={s.instance_id:s for s in specs}
    for pair in range(len(specs)//2):
        p,d=by_id[f'dist-{pair}-P'],by_id[f'dist-{pair}-D'];path=Path(out)/str(pair)
        result=await run_resident(p.base_url,d.base_url,p.zmq_address,d.zmq_address,p.tp,path)
        if result.get('status')!='passed' or result.get('cleanup_errors') or result.get('quarantined_roles'):
            raise RuntimeError('DistServe native pair mechanism probe failed: '+str(pair))
        refs.append(dict(replica=pair,result=binding(path/'runtime.json'),events=binding(path/'events.jsonl'),
                         trace=binding(path/'trace.json')))
    return refs


def _ids(events,*,terminal=True):
    tokens=[];stamps=[];ended=False
    for env in events:
        row=env.get('payload',env);ids=row.get('token_ids');stamp=row.get('received_s',env.get('at_s'))
        _need(not ended and isinstance(ids,list) and all(type(i) is int and i>=0 for i in ids),
              'native/client stream has invalid IDs or post-terminal output')
        _need(_finite(stamp) and (not stamps or stamp>=stamps[-1]), 'client/native timestamps regress')
        tokens.extend(ids);stamps.extend([stamp]*len(ids));ended=row.get('finished') is True
        _need(row.get('token_index')==len(tokens), 'stream token index is discontinuous')
    _need(not terminal or ended,'successful stream lacks terminal output')
    return tokens,stamps


def _kv(receipts,*,tp,generation,successful,single_token=False):
    """Replay actual rank transaction ACKs; failed requests may have a prefix."""
    steps=[r['step'] for r in receipts]
    required=['prefill','release'] if single_token else ['prefill','expect_load','transfer','load_ack','release']
    if successful:_need(steps==required, 'successful P/D transfer receipt sequence differs')
    _need(len(steps)==len(set(steps)) and all(s in required+['cancel_P','cancel_D'] for s in steps),
          'duplicate or unknown KV step')
    progress=[s for s in steps if not s.startswith('cancel_')]
    cancelled=[s for s in ('cancel_D','cancel_P') if s in steps]
    _need(progress==required[:len(progress)] and steps==progress+cancelled,
          'failed KV receipt order is not a valid execution prefix')
    identity=None;first=[]
    for row in receipts:
        step,receipt=row['step'],row['receipt']
        _need(receipt.get('acknowledged') is True, 'native KV control ACK missing')
        if step.startswith('cancel_'):
            _need(receipt.get('generation')==generation,'cancel epoch differs');continue
        ranks=receipt.get('ranks',[])
        _need(len(ranks)==tp and {r.get('rank') for r in ranks}==set(range(tp))
              and all(r.get('generation')==generation for r in ranks),'all-rank KV epoch ACK missing')
        if step=='prefill':
            first=[t for e in receipt.get('outputs',[]) for t in e.get('token_ids',[])]
            _need(len(first)==1 and receipt['outputs'][-1].get('finished') is True
                  and receipt.get('retained_handle') and all(receipt['retained_handle'] in r.get('held_requests',[]) for r in ranks),
                  'prefill did not retain KV and one real first token')
        elif step=='release':
            _need(receipt.get('released') is True and all(r.get('released') is True for r in ranks),
                  'source retained KV release incomplete')
        else:
            keys=('target_request_id','transaction_id','generation')
            current={k:ranks[0].get(k) for k in keys}
            _need(current['target_request_id'] and current['transaction_id'] and current['generation']==generation
                  and all(all(r.get(k)==v for k,v in current.items()) and r.get('acknowledged') is True for r in ranks),
                  'KV rank transaction or target identity differs')
            if identity is None:identity=current
            _need(identity==current,'KV transaction changed across transfer/load')
            if step=='transfer':_need(all(r.get('layers',0)>0 for r in ranks),'transferred layer count missing')
            if step=='load_ack':
                _need(receipt.get('generation')==generation and receipt.get('transaction_id')==identity['transaction_id']
                      and receipt.get('request_id')==identity['target_request_id']
                      and all(r.get('expected_layers',0)>0 and r.get('loaded_layers')==r['expected_layers'] for r in ranks),
                      'decode load does not acknowledge every layer')
    return first


def _probe_cases(trace,point,identity):
    from .gates import random_prompt
    cases=(('512','concurrent',512,16),('2048','concurrent',2048,16),
           ('7168','long',7168,16),('one','single_token',512,1),
           ('cancel','cancel',512,512),('recover','recovery',512,16))
    rows=trace.get('requests',[])
    _need(len(rows)==len(cases) and trace.get('seed')==701
          and trace.get('model_id')==point['model_id']
          and trace.get('tokenizer_hash')==identity['tokenizer_hash'],
          'pair probe must cover all six bound model-tokenized cases')
    expected={}
    for row,(name,phase,length,count) in zip(rows,cases):
        rid='distserve-701-'+name
        _need(row.get('request_id')==rid and row.get('phase')==phase
              and row.get('prompt')==random_prompt(length,701+length)
              and row.get('max_tokens')==count and row.get('seed')==701
              and row.get('temperature')==0 and row.get('ignore_eos') is True,
              'pair probe case identity or fixed work differs: '+rid)
        expected[rid]=row
    return expected


def _probe_goldens(result):
    references=result.get('references',{})
    ids={'distserve-701-'+name for name in ('512','2048','7168')}
    _need(set(references)==ids,'three independent ordinary golden references required')
    times={}
    for rid in (*sorted(ids),'repeat'):
        value=result.get('reference_repeat',{}) if rid=='repeat' else references[rid]
        events=value.get('events',[])
        tokens,stamps=_ids(events)
        request_id='distserve-reference-repeat' if rid=='repeat' else 'distserve-reference-'+rid
        _need(len(tokens)==16 and tokens==value.get('token_ids')
              and all(row.get('request_id')==request_id for row in events),
              'ordinary golden lacks complete independently identified raw stream')
        times[rid]=stamps
    _need(result['reference_repeat']['token_ids']==references['distserve-701-512']['token_ids']
          and times['repeat'][0]>=max(times[rid][-1] for rid in ids),
          'repeated same-model golden differs or precedes original observations')


def _probe_requests(events,expected,result):
    queued={};finished={};clients=defaultdict(list);receipts=defaultdict(list)
    _need(all(_finite(row.get('at_s')) for row in events)
          and all(a['at_s']<=b['at_s'] for a,b in zip(events,events[1:])),
          'probe journal timestamps missing or regressing')
    for event in events:
        kind=event.get('event');rid=event.get('request_id')
        if rid is not None:
            _need(rid in expected,'probe journal contains an unbound request')
        if kind=='distserve_request_queued':
            row=expected[rid]
            _need(rid not in queued and event.get('payload')==row
                  and event.get('input_tokens')==len(row['prompt'])
                  and event.get('output_tokens')==row['max_tokens'],
                  'probe queued payload or case coverage differs')
            queued[rid]=event
        elif rid is not None:
            _need(rid in queued and rid not in finished,'probe request event is outside its lifetime')
            if kind=='distserve_request_finished':
                _need(not event.get('cleanup_errors'),'probe request cleanup failed')
                finished[rid]=event
            elif kind=='distserve_client_sse':clients[rid].append(event)
            elif kind=='distserve_native_receipt':
                receipts[rid].append(dict(step=event['step'],receipt=event['receipt']))
    _need(set(queued)==set(finished)==set(expected),'all six probes must actually queue and finish')
    success=set(expected)-{'distserve-701-cancel'}
    _need(set(result.get('outcomes',{}))==success,'probe successful outcome coverage differs')
    for rid in success:
        out=result['outcomes'][rid]
        _need(all(_finite(out.get(key)) for key in ('submitted_s','finished_s'))
              and out['submitted_s']<=queued[rid]['at_s']<=finished[rid]['at_s']<=out['finished_s']
              and out.get('ok') is True and out.get('golden_match') is True and not out.get('error')
              and finished[rid].get('status')=='completed',
              'probe outcome differs from actual request lifetime/completion')
    concurrent=('distserve-701-512','distserve-701-2048')
    _need(result.get('concurrent_requests_observed') is True
          and max(queued[r]['at_s'] for r in concurrent)<min(finished[r]['at_s'] for r in concurrent),
          'concurrent probe requests did not actually overlap')
    recovered=[row for row in events if row.get('event')=='distserve_recovered']
    _need(len(recovered)==1 and finished['distserve-701-cancel']['at_s']<=recovered[0]['at_s']
          <=queued['distserve-701-recover']['at_s']
          and _equal(recovered[0].get('states'),result.get('recovery',{}).get('states')),
          'recovery lacks actual post-cancellation native observation before new work')
    return clients,receipts,finished


def audit_pair_probes(point,identity,startup,instances):
    refs=startup.get('distserve_pair_probes',[]);n=len(instances)//2
    _need(len(refs)==n and {r.get('replica') for r in refs}==set(range(n)), 'pair mechanism coverage incomplete')
    for ref in refs:
        result=_bound(ref['result']);trace=_bound(ref['trace']);events=_bound(ref['events'],journal=True)
        _need(result.get('system')=='distserve' and result.get('complete') is True and result.get('status')=='passed'
              and result.get('seed')==701
              and result.get('events_sha256')==ref['events']['sha256'] and result.get('trace_sha256')==ref['trace']['sha256']
              and not result.get('cleanup_errors') and not result.get('quarantined_roles'), 'mechanism raw binding/cleanup differs')
        expected=_probe_cases(trace,point,identity);_probe_goldens(result)
        clients,receipts,finished=_probe_requests(events,expected,result)
        caps=result['capabilities'];tp=instances[f'dist-{ref["replica"]}-P']['tp']
        _need(set(caps)=={'P','D'} and result.get('tp')==tp and result.get('pp')==1,
              'pair probe capability roles or topology incomplete')
        for role,cap in caps.items():
            spec=instances[f'dist-{ref["replica"]}-{role}']
            _need(cap['gpu_uuids']==spec['gpu_uuids'] and cap['tp']==tp and cap['pp']==1
                  and cap['model_id']==point['model_id'] and all(cap[k]==identity[k] for k in ('model_hash','tokenizer_hash','image_digest')),
                  'pair probe model/placement differs')
        starts=[row for row in events if row.get('event')=='distserve_runtime_started']
        _need(len(starts)==1 and _equal(starts[0].get('capabilities'),caps), 'pair probe native startup missing or duplicated')
        generations={starts[0]['states'][role]['generation'] for role in ('P','D')}
        _need(len(generations)==1,'mechanism peers lack common native epoch');generation=generations.pop()
        for row in trace['requests']:
            if row['phase']=='cancel':continue
            rid=row['request_id'];outcome=result['outcomes'][rid];native=outcome['result']
            events_out=outcome['events'];tokens=[t for e in events_out for t in e['token_ids']]
            reference=result['references'].get(rid,result['references']['distserve-701-512'])['token_ids'][:row['max_tokens']]
            _need(tokens==reference==native['token_ids'] and len(tokens)==row['max_tokens']
                  and events_out[-1].get('finished') is True and native['status']=='completed',
                  'pair pipeline differs from same-model ordinary golden')
            journal_tokens,_=_ids(clients[rid])
            _need(journal_tokens==tokens and _equal(events_out,[event['payload'] for event in clients[rid]])
                  and _equal(native.get('events'),events_out) and native.get('request_id')==rid
                  and native.get('tokens')==len(tokens)==finished[rid].get('generated')
                  and _equal(native['receipts'],receipts[rid]),'probe client journal/completion differs from golden output')
            first=_kv(native['receipts'],tp=tp,generation=generation,successful=True,single_token=row['max_tokens']==1)
            _need(first==tokens[:1],'probe first token differs from retained prefill output')
        cancelled=result['cancel'];before=cancelled['before'];target=cancelled['target_request_id']
        _need(target in before['all_queue'] and before['kv_allocations'].get(target)
              and cancelled['acknowledgement'].get('acknowledged') is True
              and cancelled['result']['status']=='cancelled' and 2<=cancelled['result']['tokens']<512
              and {'cancel_D','cancel_P'}<={r['step'] for r in cancelled['result']['receipts']}
              and result['recovery'].get('acknowledged') is True,'actual cancellation/recovery evidence incomplete')
        rid='distserve-701-cancel';tokens,_=_ids(clients[rid],terminal=False);native=cancelled['result']
        _need(tokens==native.get('token_ids') and len(tokens)==native.get('tokens')==finished[rid].get('generated')
              and finished[rid].get('status')=='cancelled' and native.get('request_id')==rid
              and cancelled['acknowledgement'].get('request_id')==rid
              and _equal(cancelled.get('events'),[event['payload'] for event in clients[rid]])
              and _equal(native.get('events'),cancelled.get('events'))
              and _equal(native['receipts'],receipts[rid]), 'cancellation lacks the bound partial client/native stream')
        _kv(native['receipts'],tp=tp,generation=generation,successful=False)
        _need(set(result['recovery'].get('states',{}))=={'P','D'},'recovery native states incomplete')
        for state in result['recovery']['states'].values():
            _need(_state(state,tp,state.get('response_at_s'))==generation,'recovery native epoch differs')
        _need(len(result['drain'])==2,'pair probe final drain missing')
        for state in result['drain']:
            _need(state.get('acknowledged') is True and state.get('drained') is True,'pair probe drain lacks ACK')
            _state(state,tp,state.get('response_at_s'))


def audit_protocol(trace,native,events,choice,instances,reset,origin):
    """Replay author prefill admission and global least-loaded pair routing."""
    requests={f'distserve-701-{i}':r for i,r in enumerate(trace['requests'])}
    outcomes={r['request_id']:r for r in native['outcomes']}
    _need(len(outcomes)==len(native['outcomes']) and set(outcomes)==set(requests),'terminal cohort incomplete or duplicated')
    n=choice['selected']['replicas'];tp=choice['selected']['tp'];schedulers={};queued={};finished={}
    admissions=set();bridged=set();clients=defaultdict(list);natives=defaultdict(list);receipts=defaultdict(list);closed=set()
    for event in events:
        kind=event.get('event',event.get('kind'));pair=event.get('replica')
        _need(type(pair) is int and 0<=pair<n and _finite(event.get('at_s')), 'event pair or time unbound')
        rid=event.get('request_id')
        if rid is not None and kind!='distserve_request_queued':
            _need(rid in queued and queued[rid][0]==pair, 'request pair differs from queued owner')
        if kind=='distserve_runtime_started':
            _need(pair not in schedulers and event['at_s']<=origin,'controller state reused or started late')
            caps=event['capabilities'];states=event['states']
            for role in ('P','D'):
                iid=f'dist-{pair}-{role}';state=states[role]
                _need(caps[role]['gpu_uuids']==instances[iid]['gpu_uuids']
                      and _state(state,tp,state.get('response_at_s'))==reset['generation'][iid]
                      and state.get('accepting') is True,'native pair startup differs from fresh reset')
            schedulers[pair]=PrefillScheduler(max_batch_size=32,max_tokens_per_batch=min(8192,states['P']['max_num_batched_tokens']),
                num_gpu_blocks=min(4096,states['P']['total_kv_tokens']//16),block_size=16)
        elif kind=='distserve_request_queued':
            rid=event['request_id'];req=requests[rid];payload=event['payload']
            _need(len(schedulers)==n and rid not in queued and pair==min(range(n),key=lambda i:(
                len(schedulers[i].waiting)+len(schedulers[i].processing),i)), 'global prefill routing differs')
            _need(payload.get('prompt')==req['prompt'] and payload.get('max_tokens')==req['max_tokens']
                  and payload.get('seed')==701 and payload.get('ignore_eos') is True and payload.get('temperature')==0
                  and origin+req['arrival_s']<=event['at_s'],'request content, seed or schedule differs')
            request=Request(rid,len(req['prompt']),req['max_tokens'],payload);queued[rid]=(pair,request)
            schedulers[pair].add(request)
        elif kind=='distserve_prefill_admission':
            batch=schedulers[pair].next_batch(free_gpu_blocks=event['free_kv_tokens']//16)
            ids=[r.request_id for r in batch]
            _need(ids==event['request_ids'] and not admissions.intersection(ids),'author FCFS admission differs')
            admissions.update(ids)
        elif kind=='distserve_bridge_ready':
            rid=event['request_id'];_need(rid in admissions and rid not in bridged,'bridge without unique prefill admission')
            schedulers[pair].complete((queued[rid][1],));bridged.add(rid)
        elif kind=='distserve_request_finished':
            rid=event['request_id'];_need(rid in queued and rid not in finished,'duplicate or unqueued completion')
            finished[rid]=event;schedulers[pair].processing.pop(rid,None);schedulers[pair].release(rid)
            schedulers[pair].cancel(rid)
        elif kind=='distserve_native_receipt':receipts[event['request_id']].append(dict(step=event['step'],receipt=event['receipt']))
        elif kind=='distserve_client_sse':clients[event['request_id']].append(event)
        elif kind=='distserve_native_sse':natives[event['request_id']].append(event)
        elif kind=='distserve_runtime_closed':
            _need(pair not in closed and event['at_s']>=origin+150 and not event.get('quarantined'), 'pair close missing or quarantined')
            for role in ('P','D'):_state(event['states'][role],tp,event['states'][role].get('response_at_s'))
            closed.add(pair)
        elif kind=='distserve_runtime_failure':raise ValueError('native controller failure needs investigation')
    _need(closed==set(range(n)) and not(set(clients)|set(natives)|set(receipts))-set(requests),'unbound request or unclosed controller')
    for rid,row in outcomes.items():
        req=requests[rid];failed=row.get('ok') is False or bool(row.get('error'))
        _need(row.get('arrival_s')==req['arrival_s'] and row.get('input_tokens')==len(req['prompt'])
              and row.get('output_tokens')==req['max_tokens'] and _finite(row.get('finished_s'))
              and row['finished_s']>=origin+req['arrival_s'],'outcome request identity/time differs')
        tokens,times=_ids(clients[rid],terminal=not failed)
        _need(row.get('completion_tokens')==len(tokens) and row.get('event_count')==len(clients[rid])
              and row.get('token_ids_sha256')==hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
              'client token receipt hash/count differs')
        if rid not in queued:
            _need(failed and not tokens and not receipts[rid] and not natives[rid], 'unadmitted request has unexplained work');continue
        pair=queued[rid][0];generation=reset['generation'][f'dist-{pair}-P']
        _need(row.get('replica')==pair and rid in finished and not finished[rid].get('cleanup_errors'), 'pair/final request receipt differs')
        single_token=req['max_tokens']==1
        first=_kv(receipts[rid],tp=tp,generation=generation,successful=not failed,single_token=single_token)
        if single_token:
            _need(not natives[rid], 'single-token prefill completion unexpectedly executed decode')
        native_tokens,native_times=_ids(natives[rid],terminal=not failed and not single_token)
        expected=first+native_tokens
        _need(tokens==expected[:len(tokens)] and (failed or len(tokens)==req['max_tokens']==len(expected)),
              'prefill carry or native/client output continuity differs')
        if not failed:
            _need(row['result']['status']=='completed' and row['result']['tokens']==len(tokens)
                  and _equal(row['result']['receipts'],receipts[rid]), 'native completion or KV receipt differs')
    return True


def audit_distserve_window(point,engine_identity,startup_qualification,reset,native_result,
                          canonical_metrics,metering,drain,raw_refs):
    failures={};checked=[];data={};reduced=None
    def gate(name,fn):
        try:value=fn()
        except (ValueError,TypeError,KeyError,OSError,IndexError,AttributeError,OverflowError) as exc:
            failures[name]=str(exc);return None
        checked.append(name);return value
    for key in ('trace','events','power','native_result','canonical_requests','metering','startup_qualification','reset','drain'):
        data[key]=gate('raw.'+key,lambda key=key:_bound(raw_refs.get(key),journal=key=='events',power=key=='power'))
    for key,supplied in [('native_result',native_result),('startup_qualification',startup_qualification),('reset',reset),('metering',metering),('drain',drain)]:
        gate('binding.'+key,lambda key=key,supplied=supplied:_need(_equal(data[key],supplied),'bound raw file differs'))
    gate('binding.trace',lambda:_need(raw_refs.get('trace')==point.get('trace'),'shared frozen trace differs'))
    preflight=validate_distserve_inputs(point,engine_identity,source_manifest=startup_qualification.get('source_manifest'),replay_search=True)
    failures.update({'inputs.'+k:v for k,v in preflight['gate_failures'].items()})
    instances=gate('dist.native_fleet',lambda:native_topology(point,engine_identity));origin=native_result.get('service_started_s')
    def native_identity():
        _need(native_result.get('system')=='distserve' and native_result.get('seed')==701 and _finite(origin)
              and native_result.get('trace_sha256')==point['trace']['sha256']
              and native_result.get('events_sha256')==raw_refs['events']['sha256']
              and native_result.get('selected')==preflight['choice']['selected']
              and native_result.get('plan_sha256')==hashlib.sha256(json.dumps(preflight['choice'],sort_keys=True).encode()).hexdigest()
              and _finite(native_result.get('service_finished_s')) and native_result['service_finished_s']>=origin+150
              and not native_result.get('cleanup_errors'),'native service identity or window boundary differs')
    gate('dist.native_identity',native_identity)
    if instances:
        gate('dist.startup',lambda:audit_native_startup(point,engine_identity,startup_qualification,instances,raw_refs))
        gate('dist.pair_goldens',lambda:audit_pair_probes(point,engine_identity,startup_qualification,instances))
        gate('dist.reset',lambda:audit_native_reset(reset,instances,origin))
        def final_drain():
            generations,times=_drained(drain,instances)
            _need(generations==reset['generation'] and min(times)>=native_result['service_finished_s']
                  and max(times)<=metering['tail_end_s'],'all-rank drain epoch or tail differs')
        gate('dist.drain',final_drain)
    gate('dist.author_pipeline',lambda:audit_protocol(data['trace'],native_result,data['events'],preflight['choice'],instances,reset,origin))
    reduced=gate('metrics.client_canonical',lambda:audit_native_metrics(point,data['trace'],native_result['outcomes'],data['events'],origin,
        data['canonical_requests'],canonical_metrics))
    gate('metering.raw_eight_gpu_window',lambda:audit_native_meter(engine_identity,data['power'],metering,origin))
    modes=(point.get('metering_execution', 'in_process'),engine_identity.get('metering_execution', 'in_process'))
    gate('metering.execution_identity',lambda:_need(modes[0]==modes[1]
        and modes[0] in ('in_process','isolated_process'),'unsupported or mismatched meter execution'))
    if 'isolated_process' in modes:
        gate('metering.isolated_process_method',lambda:audit_isolated_meter_method(
            point,engine_identity,startup_qualification,native_result,metering,raw_refs))
    valid=not failures
    return dict(schema='distserve-single-observation-acceptance-v1',evidence_valid=valid,formal_eligible=valid,
        scope='measured_symmetric_tp_pp1_author_queues_and_native_pipeline',complete_reproduction=False,
        slo_pass=reduced['slo_pass'] if reduced else False,missing_gates=list(failures),gate_failures=failures,
        checked_gates=checked,preflight=preflight,evidence_sha256=digest(raw_refs),optimality_established=False)
