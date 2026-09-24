"""Read-only replay of PD native runtime journals and instant board samples.

This validates measured components, never promotes them to a full profile or
claims incremental transition energy. Native worker compatibility and a frozen
calibration/holdout composition remain separate gates.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

from pdblend.bench.comparison_metering import summarize_comparison
from pdblend.online.native_control import validate_state
from .native_frequency_domain import identity_frequencies,require_same_domain,with_domain
from .native_timing_plan import digest


def need(condition,message):
    if not condition:raise ValueError(message)


def bound(reference,*,lines=False):
    payload=Path(reference['path']).read_bytes()
    need(hashlib.sha256(payload).hexdigest()==reference['sha256'],'raw checksum differs')
    return [json.loads(line) for line in payload.splitlines() if line.strip()] if lines else json.loads(payload)


def power_slice(raw,start,end):
    timestamps=[row[0] for row in raw['samples']]
    lo=max(0,bisect_left(timestamps,start-1)-1)
    hi=min(len(timestamps),bisect_right(timestamps,end+1)+1)
    value={k:v for k,v in raw.items() if k not in ('samples','power_metadata','utilization_samples','utilization_readings','frequency_samples')}
    value['samples']=raw['samples'][lo:hi];value['power_metadata']=raw['power_metadata'][lo:hi]
    value['frequency_samples']=[r for r in raw.get('frequency_samples',[]) if start-1<=r[0]<=end+1]
    value['utilization_samples']=[r for r in raw.get('utilization_samples',[]) if start-1<=r[0]<=end+1]
    value['utilization_readings']=[r for r in raw.get('utilization_readings',[]) if start-1<=r['read_finished_s']<=end+1]
    return value


def validate_transfer_payload(result, *, input_tokens, prefill_instance, decode_instance):
    """Replay a non-streamed P first token plus streamed ordinary/D results.

    Completion has no generation field. Epoch proof is checked separately
    against native admission/drain receipts, never invented on an HTTP result.
    """
    from pdblend.engine.handoff_timing import measure_handoff
    need(prefill_instance!=decode_instance,'PD source and decoder must differ')
    ordinary=result.get('ordinary');pre=result.get('prefill');combined=result.get('combined')
    need(isinstance(ordinary,list) and len(ordinary)==2 and isinstance(pre,dict)
         and isinstance(combined,dict),'transfer raw request/reference absent')
    required={'error','usage_received','stream_done','prompt_tokens','completion_tokens','token_ids',
              'token_times_s','submitted_s','first_token_s','finished_s','instance_id','request_id'}
    for value in [*ordinary,pre,combined]:
        need(isinstance(value,dict) and required<=value.keys() and value['error'] is None
             and value['usage_received'] is True,'transfer request incomplete')
        count=1 if value is pre else 16
        need(type(value['prompt_tokens']) is int and value['prompt_tokens']==input_tokens
             and type(value['completion_tokens']) is int and value['completion_tokens']==count,
             'transfer token usage differs')
        ids,times=value['token_ids'],value['token_times_s']
        need(isinstance(ids,list) and len(ids)==count and all(type(n) is int and n>=0 for n in ids)
             and isinstance(times,list) and len(times)==count,'transfer token payload incomplete')
        arrivals=[value['submitted_s'],*times,value['finished_s']]
        need(all(type(t) in (int,float) and math.isfinite(t) for t in arrivals)
             and all(b>=a for a,b in zip(arrivals,arrivals[1:]))
             and value['first_token_s']==times[0],'transfer token arrivals incomplete or out of order')
        need(value['instance_id']==(prefill_instance if value is pre else decode_instance),
             'transfer endpoint identity differs')
        need(value['stream_done'] is (False if value is pre else True),
             'prefill is non-streamed; ordinary and combined require complete streams')
    need(pre['finished_s']==pre['first_token_s'] and pre['request_id']==combined['request_id']
         and pre['token_ids']==combined['token_ids'][:1]
         and pre['token_times_s']==combined['token_times_s'][:1]
         and pre['submitted_s']==combined['submitted_s'],'carried prefill token or request identity differs')
    need(ordinary[0]['token_ids']==ordinary[1]['token_ids']==combined['token_ids'],
         'PD golden differs from ordinary reference')
    need(combined.get('pd_protocol')=='carry_first_token' and combined.get('decode_completion_tokens')==15
         and combined.get('decode_first_token_s')==combined['token_times_s'][1]
         and type(combined.get('decode_submitted_s')) in (int,float)
         and pre['finished_s']<=combined['decode_submitted_s']<=combined['decode_first_token_s'],
         'D continuation does not follow the actual carried first token')
    return measure_handoff(SimpleNamespace(**ordinary[0]),SimpleNamespace(**pre),SimpleNamespace(**combined))


def validate_transfer_request(row, *, specs, training_seed=9701, holdout_seed=9702):
    """Full collector/replayer contract including actual native epochs."""
    epochs=row['native_epoch'];prefill=epochs['prefill_instance'];decode=epochs['decode_instance']
    need(prefill!=decode and {prefill,decode}==set(epochs['before'])==set(epochs['admission']),
         'transfer native peer inventory differs')
    need(specs[prefill]['generation']==specs[decode]['generation'],'PD native peer generations differ')
    need(type(training_seed) is int and type(holdout_seed) is int and training_seed!=holdout_seed
         and type(row.get('repeat')) is int and 0<=row['repeat']<4,'transfer split/seed declaration differs')
    need(len(row['prompt'])==row['input_tokens'] and row.get('seed')==(training_seed if row['repeat']<3 else holdout_seed),
         'transfer prompt or training/holdout seed differs')
    timing=validate_transfer_payload(row['result'],input_tokens=row['input_tokens'],
                                    prefill_instance=prefill,decode_instance=decode)
    first_submit=row['result']['ordinary'][0]['submitted_s']
    for iid in (prefill,decode):
        spec=specs[iid];before=epochs['before'][iid];admission=epochs['admission'][iid]
        for state in (before,admission['state']):
            validate_state(state,generation=spec['generation'],tp=spec['tp'],pp=spec['pp'],drained=True)
        need(before.get('accepting') is False and before.get('acknowledged') is True and before.get('drained') is True
             and admission['control'].get('acknowledged') is True
             and admission['control'].get('generation')==spec['generation']
             and admission['state'].get('accepting') is True
             and before['native_at_s']<=admission['state']['native_at_s']<=first_submit,
             'transfer native epoch/admission evidence differs')
    return timing


def replay_runtime(report):
    result=dict(schema='pdblend-native-runtime-audit-v1',status='failed',raw_components_complete=False,
                formal_eligible=False,energy_comparable=False,errors=[],phases=[],
                remaining_gates=['independent_runtime_holdout','worker_identity_compatibility','full_profile_composition'])
    try:
        from .native_runtime_collect import RUNTIME_PLAN,build_runtime_plan
        need(report.get('complete') is True and report.get('ready_for_timing') is True,
             'collection did not complete and restore')
        need(report.get('repeats')==3 and report.get('settle_s')==2 and report.get('measure_s')==5,
             'runtime collection protocol differs')
        domain_ref=report.get('frequency_domain_ref')
        plan=build_runtime_plan(domain_ref)
        if domain_ref is None:
            need(not any(k in report for k in ('frequency_domain','frequency_domain_sha256')),
                 'legacy runtime cannot carry unbound frequency identity')
        else:
            require_same_domain(report,plan)
            need(report.get('scope')==plan['scope'] and report.get('handoff_prediction_qualified') is False
                 and report.get('full_profile_qualified') is False,'new runtime scope falsely includes handoff/full profile')
        frequencies=tuple(plan['frequencies_mhz']);restore_frequency=max(frequencies)
        need(report.get('runtime_plan')==plan,'frozen runtime train/holdout plan differs')
        need(bound(report['runtime_plan_binding'])==plan,'runtime plan file differs')
        need(report['restoration'].get('passed') is True and not report['restoration'].get('errors'),
             'inventory restoration failed')
        specs={v['spec']['instance_id']:v['spec'] for v in report['actual_launch']}
        need(len(specs)==len(report['restoration']['instances']),'restoration inventory incomplete')
        rows=bound(report['journal'],lines=True);power=bound(report['power'])
        if domain_ref is not None:
            need(rows and all(r.get('frequency_domain_sha256')==plan['frequency_domain_sha256']
                 and r.get('runtime_plan_sha256')==digest(plan) for r in [*rows,power]),
                 'new runtime journal/power domain or plan binding differs')
            need(report.get('measured_components')==['capacity','static','clock','L1','off_wake']
                 and not any(r.get('kind') in ('transfer','transfer_request') for r in rows),
                 'new runtime cannot smuggle signed transfer observations into its scoped component')
            first=next(iter(report['initial_capabilities'].values()))
            keys=('model_id','model_hash','tokenizer_hash','engine_revision','source_revision','image_digest','tp','pp')
            need(all(first.get(k) for k in keys) and all(all(c.get(k)==first.get(k) for k in keys)
                 for c in report['initial_capabilities'].values()),'new runtime component model/source identity differs')
            expected_identity=with_domain(dict(system='pdblend',**{k:first[k] for k in keys}),plan['frequency_domain'])
            need(report.get('component_identity')==expected_identity,'new runtime component identity/domain differs')
            need(report['restoration'].get('clock_mhz')==restore_frequency,'new runtime restoration frequency identity differs')
        from .native_runtime_topology import validate_topology,validate_phase,validate_frequencies,with_off_start_proof,validate_clock
        physical=validate_topology(list(specs.values()),report['lease'],power=power,
                                   capabilities=report['initial_capabilities'])
        need([r['sequence'] for r in rows]==list(range(len(rows))),'journal sequence differs')
        capacities=[r for r in rows if r['kind']=='capacity']
        need({r['instance_id'] for r in capacities}==set(specs),'native capacity inventory incomplete')
        for row in capacities:
            spec=specs[row['instance_id']];state=row['state']
            validate_state(state,generation=spec['generation'],tp=spec['tp'],pp=spec['pp'],drained=True)
            need(state['max_num_seqs']==spec['max_num_seqs']==32 and state['total_kv_tokens']>0,
                 'capacity launch scope differs')
            cap=row['capability'];expected=report['initial_capabilities'][row['instance_id']]
            need(cap==expected and cap.get('supported') is True,'capability binding differs')
        for row in report['restoration']['instances']:
            spec=specs[row['instance_id']]
            validate_state(row['drain'],generation=spec['generation'],tp=spec['tp'],pp=spec['pp'],drained=True)
            need(row.get('process_alive') is True and row['resume']['state'].get('generation')==spec['generation'],
                 'restoration epoch/liveness differs')
            need(row['clock']['ack'].get('requested_frequency_mhz')==restore_frequency,'restored clock differs')
            if domain_ref is not None:validate_clock(row['clock'],spec,physical,restore_frequency)
            stopped=row['measurement_stop'].get('ranks',[])
            need(len(stopped)==spec['tp']*spec['pp'] and {r.get('rank') for r in stopped}==set(range(spec['tp']*spec['pp']))
                 and all(r.get('acknowledged') is True for r in stopped),'restored measurement scope lacks all-rank stop ACK')
        statics=[r for r in rows if r['kind']=='static']
        expected={(name,repeat) for name in tuple('active_idle@'+str(f) for f in frequencies)+('active_idle_reset','L1','off')+
                    tuple('clock_target@'+str(f) for f in frequencies) for repeat in range(4)}
        need(len(statics)==len(expected) and {(r['state'],r['repeat']) for r in statics}==expected,
             'missing or duplicate static runtime observations')
        operations=[r for r in rows if r['kind']=='operation']
        need(all(r.get('purpose')==('training' if r['repeat']<3 else 'holdout') for r in rows if 'repeat' in r),
             'runtime train/holdout classification differs')
        need(all(r.get('status')=='passed' for r in operations),'a runtime operation failed')
        for name in ('park','unpark','off','wake',f'clock_{frequencies[1]}_to_{frequencies[0]}',
                     f'clock_{frequencies[0]}_to_{frequencies[1]}'):
            relevant=[r for r in operations if r['operation']==name]
            need(len(relevant)==4 and {r['repeat'] for r in relevant}=={0,1,2,3},'missing repeated '+name)
        for row in operations:
            if row['operation']=='off':
                need(row['receipt']['off'].get('compute_processes_gone') is True,'off lacks compute-empty evidence')
                last=row['receipt']['off']['observations'][-1]
                need(all(not d['compute_pids'] for d in last['devices']),'off retained compute process')
            if row['operation']=='wake':
                if domain_ref is not None:
                    validate_clock(row['receipt']['clock'],specs[row['instance_id']],physical,restore_frequency)
                value=row['receipt']['ordinary']
                need(not value['error'] and value['stream_done'] and value['completion_tokens']==16,
                     'wake successful ordinary reuse missing')
        golden={}
        for row in rows:
            if row['kind']!='ordinary_golden':continue
            value=row['result'];iid=row['instance_id']
            need(not value.get('error') and value.get('stream_done') is True and value.get('usage_received') is True
                 and value.get('prompt_tokens')==512 and value.get('completion_tokens')==16
                 and len(value.get('token_ids',[]))==16,'ordinary golden usage/output incomplete')
            golden.setdefault(iid,[]).append(value['token_ids'])
        representative=statics[0]['instance_id']
        need(len(golden.get(representative,[]))>=6 and all(ids==golden[representative][0] for ids in golden[representative]),
             'two pre-off references plus four matching wake outputs required')
        transfers=[r for r in rows if r['kind']=='transfer']
        if 'transfer' in report['measured_components']:
            requests={}
            request_rows={}
            for row in rows:
                if row['kind']!='transfer_request':continue
                need(row['tag'] not in requests,'duplicate transfer request tag')
                requests[row['tag']]=validate_transfer_request(row,specs=specs)
                request_rows[row['tag']]=row
            need(len(transfers)==12 and {(r['input_tokens'],r['repeat']) for r in transfers}==
                 {(n,r) for n in (512,2048,7168) for r in range(4)},'PD transfer shape/repeat coverage missing')
            for row in transfers:
                need(row['prefill_instance']!=row['decode_instance'] and row['timings']
                     and all(t.get('protocol')=='carry_first_token_second_output_v1' for t in row['timings']),
                     'PD transfer timing protocol differs')
                expected_timings=[requests[f"transfer-{row['input_tokens']}-{row['repeat']}-measure-{i}"]
                                  for i in range(len(row['timings']))]
                need(expected_timings==row['timings'],'transfer timings do not replay from raw token arrivals')
                related=[r for r in request_rows.values() if r['input_tokens']==row['input_tokens'] and r['repeat']==row['repeat']]
                need(related and all(r['native_epoch']['before']==row['before'] for r in related),
                     'transfer requests are not bound to the phase native epoch')
                for iid,state in row['after'].items():
                    spec=specs[iid];validate_state(state,generation=spec['generation'],tp=spec['tp'],pp=spec['pp'],drained=True)
                    need(state['native_at_s']>=row['finished_s'],'transfer post-window drain is stale')
        uuid_to_index={g:i for i,g in enumerate(power['gpus'])}
        measurements=[r for r in rows if r['kind'] in ('static','operation','transfer')]
        need(all(b['started_s']>=a['finished_s'] for a,b in zip(measurements,measurements[1:])),
             'serial independent runtime windows overlap')
        comparison_groups={}
        for row in measurements:
            validate_phase(with_off_start_proof(row,measurements),specs,physical,restore_frequency=restore_frequency)
            start,end=row['started_s'],row['finished_s']
            if row['kind'] in ('static','transfer'):
                need(start-row['settle_started_s']>=2 and end-start>=5,'settle/measure interval is too short')
            need(end>start,'operation elapsed interval missing')
            value=power_slice(power,start,end)
            summary=summarize_comparison(value,gpu_uuids=report['lease']['gpu_uuids'],origin_s=start,
                duration_s=end-start,tail_end_s=end,gpu_uuid_binding_verified=report['lease']['gpu_uuid_binding_verified'])
            need(summary['energy_comparable'] is True,'raw instant eight-device power coverage incomplete')
            if row['kind']=='static':
                if row['state']=='L1':
                    need(all(f==405 for f in row['memory_before_mhz']+row['memory_after_mhz']),
                         'L1 memory clock floor not observed')
                target=210 if row['state']=='L1' else (int(row['state'].split('@')[1]) if '@' in row['state'] else None)
                if target is not None:
                    validate_frequencies(value,row['gpus'],start,end,target)
            result['phases'].append(dict(sequence=row['sequence'],kind=row['kind'],
                component=row.get('state',row.get('operation','transfer')),
                energy_j=summary['energy_service_j'],per_gpu=summary['service']['power']['per_gpu'],
                energy_scope='absolute_eight_board_observation_not_incremental_transition_energy',
                incremental_energy_j=None,max_gap_s=summary['service']['power']['max_gap_s']))
            if row['kind']=='static':
                physical=dict(zip(power['gpus'],report['lease']['gpu_uuids']))
                observed=sum(summary['service']['power']['per_gpu'][physical[g]]['integral'] for g in row['gpus'])/(end-start)
                metric='group_power_w';component=row['state']
            elif row['kind']=='operation':
                observed=end-start;metric='elapsed_s';component=row['operation']
            else:
                observed=statistics.median(t['overhead_s'] for t in row['timings'])
                metric='second_output_overhead_s';component='transfer_'+str(row['input_tokens'])
            comparison_groups.setdefault((component,metric),{}).setdefault(row['purpose'],[]).append(observed)
        result['holdout_comparisons']=[]
        for (component,metric),split in sorted(comparison_groups.items()):
            need(len(split.get('training',[]))==3 and len(split.get('holdout',[]))==1,
                 'runtime component lacks three training and one independent held-out value')
            prediction=statistics.median(split['training']);observed=split['holdout'][0]
            result['holdout_comparisons'].append(dict(component=component,metric=metric,
                training=split['training'],training_prediction=prediction,heldout=observed,
                absolute_error=abs(prediction-observed),
                relative_error=abs(prediction-observed)/abs(observed) if observed else None))
        errors=[r['relative_error'] for r in result['holdout_comparisons']]
        result['holdout_passed']=False
        if errors and all(e is not None and math.isfinite(e) for e in errors):
            ordered=sorted(errors)
            summary=dict(mean_relative_error=statistics.mean(errors),
                         p95_relative_error=ordered[max(0,math.ceil(.95*len(ordered))-1)],
                         max_relative_error=max(errors))
            result['holdout_error_summary']=summary
            result['holdout_passed']=all(summary[k]<=v for k,v in plan['holdout_limits'].items())
        result.update(status='passed',raw_components_complete=True,
                      independent_holdout_collected=True,component_qualified=False,
                      capacity={r['instance_id']:r['state']['total_kv_tokens'] for r in capacities})
        if domain_ref is not None:
            result.update(component_identity=report['component_identity'],runtime_plan_sha256=digest(plan),
                frequency_domain_ref=domain_ref,scope=plan['scope'],scoped_runtime_qualified=result['holdout_passed'],
                handoff_prediction_qualified=False,full_profile_qualified=False)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,IndexError,AttributeError) as exc:
        result['errors'].append(repr(exc))
    return result
