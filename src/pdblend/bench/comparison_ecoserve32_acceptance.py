"""32B-specific four-TP2 EcoServe audit; frozen 7B/14B code stays unchanged.

Only topology/rank checks differ. Shared pure receipt, token, reset and isolated
meter checks are imported read-only from the reviewed single-observation audit.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from .comparison_acceptance import _bound, _need, _finite, _equal, _state, _drained
from .comparison_ecoserve32_inputs import validate_ecoserve_inputs, eco_topology, _source
from .comparison_ecoserve_acceptance import (_reset, _tokens, _observation_end, _layout_add,
    _shutdown_read_cancel, _isolated_meter_method, _observed_active_frequency, RAW_REFS)
from .comparison_metrics import canonical_outcomes, reduce_comparison
from .comparison_metering import summarize_comparison
from .resident_session import digest, engine_signature
from pdblend_baselines.ecoserve.policy import PrefillProfile
from pdblend_baselines.ecoserve.run_native import automatic_actions


def _flag(argv, flag):
    positions=[i for i,arg in enumerate(argv) if arg==flag or arg.startswith(flag+'=')]
    _need(len(positions)==1,'missing or overridden launch flag: '+flag)
    index=positions[0]
    if '=' in argv[index]:return argv[index].split('=',1)[1]
    _need(index+1<len(argv),'launch flag has no value: '+flag)
    return argv[index+1]


def _startup(point, identity, startup, instances, raw_refs, preflight):
    _need(startup.get('engine_signature')==engine_signature(identity)
          and startup.get('exclusive_gpu_uuids')==identity['fleet_gpu_uuids'],'startup engine/fleet binding differs')
    launch=startup.get('actual_launch_identity',{});source=preflight['source_continuity']['source_sha256']
    for key in ('image_digest','model_hash','tokenizer_hash','runtime_source_sha256','measurement_source_sha256'):
        observed=launch.get(key,startup.get(key))
        _need(observed==identity.get(key) and identity.get(key),'actual startup identity differs: '+key)
    _need(launch.get('source_revision')==source,'actual source revision differs')
    current=_source(startup['source_manifest'])
    fingerprints=startup.get('source_fingerprints',{})
    for name in ('runtime','measurement'):
        files=fingerprints.get(name+'_files',{})
        _need(files and all(current['files'].get(k)==v for k,v in files.items())
              and digest(files)==identity[name+'_source_sha256'],'source subset differs: '+name)
    rows=launch.get('instances',[])
    _need(len(rows)==len(instances) and {r.get('instance_id') for r in rows}==set(instances),'actual launch fleet incomplete')
    for row in rows:
        spec=instances[row['instance_id']];argv=row.get('argv',[]);env=row.get('environment',{})
        _need(row.get('gpu_uuids')==spec['gpu_uuids'] and identity.get('environment')
              and all(env.get(k)==v for k,v in identity['environment'].items()),'actual UUID/environment differs')
        visible=','.join(str(identity['fleet_gpu_uuids'].index(u)) for u in spec['gpu_uuids'])
        _need(env.get('CUDA_VISIBLE_DEVICES')==visible,'CUDA physical mapping differs')
        _need('pdblend_runtime.serve' in argv and Path(argv[argv.index('pdblend_runtime.serve')+1]).name==point['model_id'],
              'actual model/entrypoint differs')
        for flag,value in (('--tensor-parallel-size','2'),('--pipeline-parallel-size','1'),
                           ('--max-num-seqs','32'),('--max-model-len','8192'),
                           ('--max-num-batched-tokens','8192'),('--dtype','bfloat16')):
            _need(_flag(argv,flag)==value,'actual launch option differs: '+flag)
        connector=json.loads(_flag(argv,'--kv-transfer-config'))
        _need(connector.get('kv_connector')=='P2pNcclConnector' and connector.get('kv_role')=='kv_both',
              'EcoServe native KV connector differs')
        _need('--no-enable-prefix-caching' in argv and not any(a.startswith(('--enable-prefix-caching','--worker-cls','--scheduler-cls')) for a in argv),
              'prefix/native worker policy differs')
        from pdblend_runtime.probe import NativeSpec
        model_path=argv[argv.index('pdblend_runtime.serve')+1]
        actual_spec=NativeSpec(row['instance_id'],tuple(identity['fleet_gpu_uuids'].index(u) for u in spec['gpu_uuids']),
            int(_flag(argv,'--port')),model_path,tp=2,pp=1,**spec['launch_options'])
        _need(argv[1:]==actual_spec.command()[1:],'complete TP2 launch command differs from frozen inventory')
    caps=startup.get('capabilities',{})
    _need(set(caps)==set(instances),'startup capabilities do not cover full fleet')
    for iid,spec in instances.items():
        cap=caps[iid]
        _need(cap.get('supported') is True and cap.get('tp')==2 and cap.get('pp')==1
              and cap.get('gpu_uuids')==spec['gpu_uuids'] and cap.get('model_id')==point['model_id']
              and cap.get('source_revision')==source and cap.get('engine_revision')=='vllm-0.10.1.1'
              and all(cap.get(k)==identity[k] for k in ('model_hash','tokenizer_hash','image_digest')),
              'startup native capability identity differs')
        state=cap['state'];_state(state,2,cap.get('received_s',state.get('response_at_s')))
        _need(all(r.get('retained_kv_supported') is True for r in state['ranks']),'native KV support unproven')
    refs=startup.get('ordinary_reference',[])
    _need(len(refs)==len(instances) and {r.get('instance_id') for r in refs}==set(instances),'ordinary reference fleet incomplete')
    for ref in refs:
        pair=ref.get('responses',[])
        _need(len(pair)==2,'ordinary deterministic pair missing')
        for response in pair:
            events=response.get('events',[]);ids=[t for event in events for t in event.get('token_ids',[])]
            _need(events and events[-1].get('finished') is True and len(ids)==16
                  and all(type(t) is int and t>=0 for t in ids) and ids==response.get('token_ids'),
                  'ordinary reference token/terminal evidence incomplete')
        _need(pair[0]['token_ids']==pair[1]['token_ids'],'ordinary reference differs')
    _drained(startup.get('drain'),instances)
    concurrency=_bound(raw_refs.get('concurrency_environment',startup.get('concurrency_environment')))
    lease_ref=raw_refs.get('lease_manifest',startup.get('lease_manifest'));lease=_bound(lease_ref)
    fleet=identity['fleet_gpu_uuids']
    _need(concurrency.get('allocated_gpu_uuids')==fleet and concurrency.get('physical_gpu_uuids')==fleet
          and concurrency.get('lease_manifest_sha256')==lease_ref['sha256']
          and not concurrency.get('peer_jobs') and concurrency.get('peer_snapshots')
          and all(not r.get('peers') for r in concurrency['peer_snapshots']), 'exclusive physical lease unproven')
    _need(lease.get('gpu_uuids')==fleet and lease.get('payload',{}).get('exclusive') is True
          and lease['payload'].get('gpu_count')==8 and lease['payload'].get('reserve_host') is True,
          'exclusive eight-GPU lease manifest differs')


def _protocol(events, native, config, identity, instances, reset, power, observation_end_s):
    origin=native['service_started_s'];end=origin+150
    starts=[r for r in events if r.get('kind')=='eco_startup']
    _need(len(starts)==1,'fresh Eco controller startup missing/duplicated')
    start=starts[0];groups=[]
    for iid in list(instances)[:config['eco_initial_instances']]:groups=_layout_add(groups,iid)
    _need(start.get('groups')==groups and start.get('profile_sha256')==config['eco_profile_sha256']
          and start.get('prediction_formula')=='author_csv_ms_integer_truncation'
          and start.get('origin')=='official_core_with_paper_supplement' and start.get('frozen') is False
          and start['at_s']<=origin,'initial macro inventory/author lookup differs')
    states=native.get('initial_native_states',{})
    _need(set(states)==set(instances),'initial native inventory incomplete')
    for iid,state in states.items():
        generation=_state(state,2,state.get('response_at_s'))
        _need(generation>=reset['generation'][iid] and state['response_at_s']<=origin
              and state.get('role')=='mixed' and state.get('mode')=='temporal'
              and all(state.get(k) is True for k in ('accepting','admit_prefill','admit_decode')),
              'initial native role/admission/generation differs')
    observations=[r for r in events if r.get('kind')=='eco_service_window_start']
    _need(len(observations)==1 and observations[0].get('duration_s')==150
          and origin<=observations[0]['at_s']<=origin+1,'service origin receipt differs')
    closed=[r for r in events if r.get('kind')=='eco_closed']
    _need(len(closed)==1 and closed[0].get('failure') is None and not closed[0].get('quarantined')
          and closed[0]['at_s']>=end,'controller did not close cleanly after service')
    _need(not any(r.get('kind') in ('eco_controller_failure','eco_membership_rollback') for r in events),
          'controller failure/rollback needs separate investigation')
    clocks={iid:[] for iid in instances};startup_clock=set();startup_park=set()
    for row in events:
        if row.get('kind')!='eco_http_receipt':continue
        if row.get('error'):
            _need(row.get('instance_id') in instances and _shutdown_read_cancel(row,native,closed[0]['at_s']),
                  'native HTTP control failed')
            continue
        path=row.get('path');iid=row.get('instance_id');reply=row.get('response',{})
        if path in ('/baseline/clock','/baseline/park'):
            _need(iid in instances and reply.get('acknowledged') is True and reply.get('success') is True
                  and {r.get('gpu_uuid') for r in reply.get('gpus',[])}==set(instances[iid]['gpu_uuids']),
                  'physical clock/park receipt incomplete')
            if path=='/baseline/clock':
                _need(row.get('body',{}).get('frequency_mhz')==2520 and reply.get('requested_frequency_mhz')==2520,
                      'EcoServe changed active frequency')
            clocks[iid].append((row['at_s'],path=='/baseline/clock'))
            if row['at_s']<=start['at_s']:
                (startup_clock if path=='/baseline/clock' else startup_park).add(iid)
        elif path=='/baseline/control':
            _need(reply.get('acknowledged') is True,'native control ACK missing')
        if path=='/baseline/state':
            _need(iid in instances,'native state belongs to unknown instance')
            _state(reply,instances[iid]['tp'],row.get('at_s'),drained=False)
    scales=[r for r in events if r.get('kind')=='eco_scale_observation' and origin<=r.get('at_s',0)<end]
    _need(scales and all(r.get('origin')=='paper_supplement' and r.get('history_window_s')==60.
          and r.get('ttft_threshold_s')==config['slo_ttft_s'] for r in scales),
          'original-period scaling observations missing or changed')
    assigned={i for group in groups for i in group}
    _need(startup_clock==assigned and startup_park==set(instances)-assigned,
          'initial active/parked physical inventory differs')
    profile=PrefillProfile.load(config['eco_prefill_csv']);admissions={};version=0
    actions=automatic_actions(events)
    commits=[(index,row) for index,row in enumerate(events) if row.get('kind')=='eco_membership_commit']
    _need(len(actions)==len(commits),'membership action lacks automatic clock/park provenance')
    prepares={action['commit_index']:events[action['prepare_index']] for action in actions}
    for index,row in enumerate(events):
        kind=row.get('kind')
        if kind=='eco_membership_commit':
            pre=prepares[index];version+=1
            before=row.get('before');after=row.get('after');iid=row.get('instance_id')
            _need(before==groups and row.get('version')==version and row.get('origin')=='paper_supplement'
                  and row.get('worker_kv_preserved') is True and iid in instances,
                  'membership sequence/owner differs')
            old={i for g in before for i in g};new=[i for g in after for i in g]
            _need(len(new)==len(set(new)) and set(new)<=set(instances)
                  and all(2<=len(g)<=3 for g in after),'macro bounds/inventory differs')
            if row.get('operation')=='add':
                observations=[e for e in events[:index] if e.get('kind')=='eco_scale_observation'
                              and e.get('at_s',0)<=pre['at_s']]
                _need(observations,'automatic add has no measured TTFT trigger')
                obs=observations[-1]
                _need(_finite(obs.get('mean_ttft_s')) and obs['mean_ttft_s']>config['slo_ttft_s']
                      and obs.get('history_count',0)>0 and obs.get('ttft_exceeds_threshold') is True
                      and iid==min(obs.get('available_instances',[])), 'automatic add TTFT trigger differs')
                _need(row.get('trigger')=='mean_ttft' and iid not in old and set(new)==old|{iid}
                      and after==_layout_add(before,iid),'automatic add layout differs')
            else:
                observations=[e for e in events[:index] if e.get('kind')=='eco_scale_macro_observation'
                              and e.get('at_s',0)<=pre['at_s'] and iid in e.get('members',[])]
                _need(observations,'automatic remove has no saved-TPOT trigger')
                obs=observations[-1];members=obs['members']
                threshold=config['slo_ttft_s']*1000*(len(members)+1)/len(members)
                _need(_finite(obs.get('mean_saved_tpot_ms')) and obs['mean_saved_tpot_ms']>threshold
                      and obs.get('saved_tpot_threshold_ms')==threshold and obs.get('saved_tpot_count',0)>0
                      and obs.get('saved_tpot_exceeds_threshold') is True and obs.get('minimum_instances')==2
                      and obs.get('assigned_instances')==len(old) and iid==members[-1],
                      'automatic remove saved-TPOT trigger differs')
                _need(row.get('operation')=='remove' and row.get('trigger')=='saved_tpot'
                      and iid in old and set(new)==old-{iid},'automatic removal differs')
                drained=[r for r in events[:index] if r.get('kind')=='eco_member_drained'
                         and r.get('instance_id')==iid and r.get('at_s',0)>=pre['at_s']]
                _need(drained and drained[-1].get('active_engine_request') is False,'removed member was not drained')
                state=drained[-1]['observed_engine_state'];_state(state,2,state['response_at_s'])
            _need(set(pre.get('observed_engine_states',{}))==set(instances),'membership prepare omits native fleet')
            for member,observed in pre['observed_engine_states'].items():
                _state(observed,instances[member]['tp'],observed.get('response_at_s'),drained=False)
            groups=after
        elif kind=='eco_admission':
            rid=row.get('request_id');iid=row.get('instance_id')
            _need(rid not in admissions and iid in {i for g in groups for i in g}
                  and row.get('origin')=='official_core','admission owner/member differs')
            admissions[rid]=row
    return admissions,profile


def audit_ecoserve_window(point, engine_identity, startup_qualification, reset, native_result,
                          canonical_metrics, metering, drain, raw_refs):
    failures={};checked=[];data={};recomputed=None;boundary=None
    def gate(name,fn):
        try:value=fn()
        except (ValueError,TypeError,KeyError,OSError,IndexError,AttributeError,OverflowError) as exc:
            failures[name]=str(exc);return None
        checked.append(name);return value
    for name in RAW_REFS:
        value=gate('raw.'+name,lambda name=name:_bound(raw_refs.get(name),journal=name=='events',power=name=='power'))
        if value is not None:data[name]=value
    for name,supplied in [('native_result',native_result),('startup_qualification',startup_qualification),
                          ('reset',reset),('metering',metering),('drain',drain)]:
        gate('binding.'+name,lambda name=name,supplied=supplied:_need(name in data and _equal(data[name],supplied),
             'supplied receipt differs from bound raw file'))
    gate('binding.trace',lambda:_need(raw_refs.get('trace')==point.get('trace'),'point trace binding differs'))
    preflight=validate_ecoserve_inputs(point,engine_identity,source_manifest=startup_qualification.get('source_manifest'))
    for name,message in preflight['gate_failures'].items():failures['inputs.'+name]=message
    boundary=gate('eco.observation_boundary',lambda:_observation_end(native_result,data['events']))
    instances=gate('eco.fixed_fleet',lambda:eco_topology(point,engine_identity))
    if instances:
        gate('eco.startup',lambda:_startup(point,engine_identity,startup_qualification,instances,raw_refs,preflight))
        gate('eco.reset',lambda:_reset(reset,instances,native_result['service_started_s']))
        def drain_check():
            generations,times=_drained(drain,instances)
            _need(min(times)>=boundary['observation_end_s'] and max(times)<=metering['tail_end_s'],
                  'meter tail omits final drain')
            raw_drains=native_result.get('drain_receipts',{})
            _need(set(raw_drains)==set(instances) and not native_result.get('cleanup_errors'),
                  'runner native drain inventory incomplete')
            for iid,state in raw_drains.items():
                _need(state.get('acknowledged') is True and state.get('drained') is True,'runner drain lacks ACK')
                generation=_state(state,2,state.get('response_at_s'))
                _need(generation==generations[iid] and generation>=reset['generation'][iid],
                      'post-window native drain generation differs')
        gate('eco.all_rank_drain',drain_check)
    def metrics():
        origin=native_result['service_started_s'];config=preflight['config']
        _need(native_result.get('system')=='ecoserve' and native_result.get('model_id')==point['model_id']
              and native_result.get('seed')==701 and native_result.get('duration_s')==150
              and _finite(origin) and boundary is not None,'native service identity/time differs')
        _need(native_result.get('trace_sha256')==point['trace']['sha256']
              and native_result.get('events_sha256')==raw_refs['events']['sha256']
              and native_result.get('config_sha256')==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              'native trace/config/journal binding differs')
        outcomes=native_result.get('outcomes',[]);trace=data['trace'];requests=trace['requests']
        expected={f'ecoserve-701-{i}':r for i,r in enumerate(requests)}
        _need(len(outcomes)==len(expected) and {r.get('request_id') for r in outcomes}==set(expected),
              'offered outcome cohort incomplete/duplicate')
        _need(native_result.get('journal_rows')==len(data['events']), 'complete native journal row count differs')
        protocol=_protocol(data['events'],native_result,config,engine_identity,instances,reset,data['power'],boundary['observation_end_s'])
        admissions,profile=protocol;streams=defaultdict(list)
        for event in data['events']:
            if event.get('kind') in ('eco_client_sse','eco_native_sse'):
                _need(event.get('request_id') in expected,'unbound native/client request')
                streams[(event['request_id'],event['kind'])].append(event)
        outcome_events=defaultdict(list)
        for event in data['events']:
            if event.get('kind')=='eco_request_outcome':
                _need(event.get('request_id') in expected,'journal contains unbound outcome')
                outcome_events[event['request_id']].append(event)
        _need(set(admissions)<=set(expected),'unbound admission in journal')
        for row in outcomes:
            rid=row['request_id'];req=expected[rid]
            _need(row.get('arrival_s')==req['arrival_s'] and row.get('input_tokens')==len(req['prompt'])
                  and row.get('output_tokens')==req['max_tokens'],'native request shape differs')
            final=outcome_events[rid]
            _need(len(final)==1 and all(k in final[0] and _equal(v,final[0][k]) for k,v in row.items())
                  and _finite(final[0].get('at_s')) and row['finished_s']<=final[0]['at_s']<=boundary['controller_closed_s'],
                  'request terminal outcome missing or differs from complete journal')
            failed=bool(row.get('error') or row.get('ok') is False or row.get('correct') is False)
            client_events=streams[(rid,'eco_client_sse')];native_events=streams[(rid,'eco_native_sse')]
            client,client_times=_tokens(client_events,require_terminal=not failed)
            native,native_times=_tokens(native_events,require_terminal=not failed)
            expected_hash=hashlib.sha256(json.dumps(client,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            _need(row.get('completion_tokens')==len(client) and row.get('event_count')==len(client_events)
                  and row.get('token_ids_sha256')==expected_hash and row.get('terminal')==bool(
                      client_events and client_events[-1]['payload'].get('finished')),
                  'terminal request receipt count/hash/stream flag differs')
            _need(all(origin+req['arrival_s']<=stamp<=row['finished_s'] for stamp in client_times)
                  and all(origin+req['arrival_s']<=stamp<=boundary['controller_closed_s'] for stamp in native_times),
                  'token receipt lies outside its actual request observation')
            _need(client==native[:len(client)] and len(client)<=len(native)
                  and (failed or client==native) and all(c>=n for c,n in zip(client_times,native_times)),
                  'buffer lost/reordered or predated native output')
            if rid in admissions:
                _need(admissions[rid].get('predicted_prefill_ms')==profile.predict_ms(len(req['prompt']))
                      and admissions[rid].get('prompt_blocks')==(len(req['prompt'])+16)//16,
                      'author admission/prediction receipt differs')
            else:
                _need(failed and not client and not native and not client_events and not native_events,
                      'non-rejected request has no author admission')
        rows=canonical_outcomes('ecoserve',trace,outcomes,service_started_s=origin,journal=data['events'])
        result=reduce_comparison(trace,rows,service_started_s=origin,duration_s=150,
                                 slo=(point['slo']['ttft_s'],point['slo']['tpot_s']))
        _need(result['token_timing_complete'] and result['unresolved_requests']==0,'client timing/terminal evidence incomplete')
        _need(_equal(result['request_metrics'],data['canonical_requests']) and all(k in canonical_metrics and _equal(v,canonical_metrics[k])
              for k,v in result.items() if k!='request_metrics'),'canonical metrics differ from raw client reduction')
        return result
    recomputed=gate('eco.raw_protocol_and_canonical_metrics',metrics)
    gate('eco.observed_active_frequency',lambda:_observed_active_frequency(
        data['events'],native_result,engine_identity,instances,data['power'],boundary['observation_end_s']))
    def power():
        snapshot=data['power']
        _need(snapshot.get('gpu_uuids')==engine_identity['fleet_gpu_uuids'] and snapshot.get('gpu_uuid_binding_verified') is True,
              'raw physical UUID mapping unproven')
        result=summarize_comparison(snapshot,gpu_uuids=engine_identity['fleet_gpu_uuids'],
            origin_s=native_result['service_started_s'],tail_end_s=metering['tail_end_s'],duration_s=150,gpu_uuid_binding_verified=True)
        _need(result['energy_comparable'] and _equal(result,metering),'raw eight-card energy/window summary differs')
    gate('metering.raw_eight_gpu_window',power)
    if point.get('metering_execution') == 'isolated_process':
        gate('metering.isolated_process_method', lambda:_isolated_meter_method(
            point,engine_identity,startup_qualification,native_result,metering,raw_refs))
    valid=not failures
    return dict(schema='ecoserve32-single-observation-acceptance-v1',scope='author_lookup_protocol+single_observation',
        evidence_valid=valid,formal_eligible=valid,full_profile_qualified=False,
        slo_pass=recomputed['slo_pass'] if recomputed else False,missing_gates=list(failures),
        gate_failures=failures,checked_gates=checked,preflight=preflight,observation_boundary=boundary,
        inherited_artifact_flags_unchanged=True,optimality_established=False,evidence_sha256=digest(raw_refs))
