"""Independent raw replay of the model-owned capacity-aware timing collector."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

from .native_timing_audit import need,finite
from .native_timing_plan import binding,digest
from .native_frequency_domain import plan_frequencies,with_domain,require_same_domain,validate_collection_inputs
from .native_timing_plan_v2 import build_plan,MODEL_TP
from .native_timing_capacity import partition_windows,fit_measured_partition
from .native_timing_replay import (Resolver,mounts,_validate_attempt,_power,_audited_window,
    audit_interference_peers,audit_native_launch)
from . import native_timing_single_pass as single_pass

SCHEMA='pdblend-native-timing-replay-evidence/v2'


def _plan(inputs,resolver):
    plan=resolver.read(inputs['point_plan'])
    if single_pass.is_single_pass(plan):
        parent = _plan(dict(inputs, point_plan=plan['parent_point_plan']), resolver)
        expected = single_pass.from_parent(parent, plan['parent_point_plan'])
        need(digest(expected) == digest(plan), 'single-pass bound design does not reproduce')
        need(inputs.get('timing_first') is False and not any(inputs.get(key) for key in
             ('collect_runtime', 'power_pilot_plan', 'request_cycle_plan', 'layout_energy_plan')),
             'single-pass development cannot schedule legacy qualification or supplements')
        return plan
    refs=[plan['query_ledger'],plan['query_provenance'],*plan['corpus_refs'].values()]
    if plan.get('frequency_domain_ref'):refs.append(plan['frequency_domain_ref'])
    for ref in refs:resolver.read(ref)
    local=lambda ref:dict(ref,path=str(resolver.path(ref['path'])))
    expected=build_plan(local(plan['query_ledger']),local(plan['query_provenance']),
        {k:local(v) for k,v in plan['corpus_refs'].items()},model_id=plan['model_id'],
        frequency_domain_ref=local(plan['frequency_domain_ref']) if plan.get('frequency_domain_ref') else None)
    for key in ('query_ledger','query_provenance','corpus_refs'):expected[key]=plan[key]
    if plan.get('frequency_domain_ref'):expected['frequency_domain_ref']=plan['frequency_domain_ref']
    require_same_domain(plan,inputs)
    need(inputs.get('frequency_domain_ref')==plan.get('frequency_domain_ref'),'timing inputs frequency domain ref differs')
    for row in expected['corpus_bindings']:row['path']=plan['corpus_refs'][row['dataset']]['path']
    need(expected==plan,'v2 timing plan differs from model-owned immutable design')
    if 'frequency_domain' in plan:
        validate_collection_inputs(plan,inputs,bound_reader=resolver.read,collect_runtime=bool(inputs.get('collect_runtime')),
            power_pilot=bool(inputs.get('power_pilot_plan')),request_cycles=bool(inputs.get('request_cycle_plan')),
            layout_energy=bool(inputs.get('layout_energy_plan')))
    return plan


def _files(attempt,manifest,execution,resolver):
    root=attempt/'native-timing';completion=binding(root/'completion.json')
    need(execution.get('receipt_sha256',{}).get('native-timing/completion.json')==completion['sha256'],
         'worker did not bind v2 timing completion')
    report=resolver.read(completion)
    development = report.get('schema') == single_pass.COLLECTION_SCHEMA
    need(report.get('schema') in ('pdblend-native-timing-collection/v2', single_pass.COLLECTION_SCHEMA) and report.get('system')=='pdblend'
         and report.get('status')=='passed' and report.get('complete') is True
         and report.get('hardware_executed') is True and not report.get('error') and not report.get('cleanup_errors'),
         'v2 timing collection did not safely complete')
    inputs_ref=manifest['payload']['input_manifest'];inputs=resolver.read(inputs_ref)
    need(inputs.get('schema')==(single_pass.INPUT_SCHEMA if development else 'pdblend-native-timing-inputs/v2') and inputs.get('system')=='pdblend',
         'v2 timing input schema differs')
    plan=_plan(inputs,resolver);source=resolver.read(inputs['source_manifest'])
    need(single_pass.is_single_pass(plan) is development, 'timing schema cannot promote single-pass observations')
    need(report.get('point_plan')==inputs['point_plan'] and report.get('capacity_policy')==plan['capacity_policy']
         and inputs.get('model_id')==plan['model_id'],'v2 collection plan/capacity/model binding differs')
    need(digest(source['files'])==source['source_sha256']==inputs['source_sha256']==manifest['payload']['source_sha256']
         and inputs['image_digest']==manifest['payload']['image_digest'],'v2 timing source/image differs')
    source_root=resolver.path(inputs['source_manifest']['path']).parent
    for name,checksum in source['files'].items():
        path=(source_root/name).resolve()
        need(path.is_relative_to(source_root) and binding(path)['sha256']==checksum,'v2 timing source bytes differ: '+name)
    refs=dict(completion=completion,input_manifest=inputs_ref,point_plan=inputs['point_plan'],
        source_manifest=inputs['source_manifest'],model_verification=inputs['model_verification'],
        timing_component=report['timing_component'],measurement_qualification=report['measurement_qualification'],
        window_partition=report['window_partition'],query_ledger=plan['query_ledger'],query_provenance=plan['query_provenance'],
        **{'corpus_'+k:v for k,v in plan['corpus_refs'].items()})
    if development: refs['parent_point_plan'] = plan['parent_point_plan']
    for ref in refs.values():resolver.read(ref)
    raw_refs=report['raw_bindings'];paths=[resolver.path(r['path']) for r in raw_refs]
    expected={f'{digest(point)[:20]}-{repeat}.json' for point in plan['points'] for repeat in range(point['repeats'])}
    need(len(paths)==len(set(paths))==len(expected) and {p.name for p in paths}==expected
         and all(p.parent==root/'samples' for p in paths) and set((root/'samples').glob('*.json'))==set(paths),
         'v2 complete planned raw inventory differs')
    owners=report.get('window_owners',[])
    need(len(owners)==len(raw_refs) and [r.get('raw') for r in owners]==raw_refs,
         'v2 window ownership is not in complete bound raw order')
    for ref in raw_refs:resolver.read(ref)
    ids=tuple('pd-timing-'+str(i) for i in range(plan['resident_instances']))
    interference=[root/'interference'/f'{f}-{repeat}-{iid}-{phase}.json'
        for f in plan_frequencies(plan) for repeat in range(0 if development else 3) for iid in ids for phase in ('isolated','parallel')]
    need(set((root/'interference').glob('*.json'))==set(interference),'v2 interference raw inventory differs')
    return root,report,inputs,plan,refs,raw_refs,interference,ids


def capture_evidence(attempt,queue,out,*,path_map=()):
    attempt,out=Path(attempt).resolve(),Path(out).resolve()
    need(not out.exists(),'refusing to overwrite v2 timing evidence')
    manifest=json.loads((attempt/'manifest.json').read_text());execution=json.loads((attempt/'execution.json').read_text())
    job=json.loads(Path(queue).read_text())['jobs'].get(manifest['job_id'],{})
    _validate_attempt(manifest,execution,job)
    resolver=Resolver([*path_map,*mounts(execution['argv'])])
    _,_,_,plan,refs,raw_refs,interference,_=_files(attempt,manifest,execution,resolver)
    value=dict(schema=single_pass.EVIDENCE_SCHEMA if single_pass.is_single_pass(plan) else SCHEMA,created_s=time.time(),binding_scope='new_post_collection_snapshot_not_original_completion_binding',
        formal_eligible=False,full_profile_qualified=False,attempt_root=str(attempt),queue_job=job,queue_job_sha256=digest(job),
        attempt_manifest=binding(attempt/'manifest.json'),worker_execution=binding(attempt/'execution.json'),
        references={name:dict(ref,resolved_path=str(resolver.path(ref['path']))) for name,ref in refs.items()},
        samples=[dict(ref,resolved_path=str(resolver.path(ref['path']))) for ref in raw_refs],
        interference=[binding(p) for p in interference],path_map=[list(x) for x in path_map])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return binding(out)


def _identities(report,inputs,plan,manifest,resolver,ids):
    verified=resolver.read(inputs['model_verification'])
    model=next((v for v in verified.get('models',{}).values() if v.get('model_id')==plan['model_id']),None)
    need(verified.get('all_pass') is True and model and model.get('verified') is True,'v2 verified own model absent')
    identity=dict(model_id=plan['model_id'],tp=MODEL_TP[plan['model_id']],pp=1,engine_revision='vllm-0.10.1.1',
        source_revision=inputs['source_sha256'],image_digest=inputs['image_digest'])
    for kind,key in [('weight','model_hash'),('tokenizer','tokenizer_hash')]:
        inventory=[(r['path'],r['bytes'],r['sha256']) for r in model['files'] if r['kind']==kind]
        need(inventory,'v2 verified model inventory incomplete');identity[key]=digest(inventory)
    caps=report['capabilities'];launches=report['actual_launch'];tp=plan['tp']
    need(set(caps)==set(ids) and len(launches)==len(ids) and {r['spec']['instance_id'] for r in launches}==set(ids),
         'v2 native capability/launch inventory differs')
    identities={}
    for index,iid in enumerate(ids):
        expected=dict(identity,gpu_uuids=manifest['gpu_uuids'][index*tp:(index+1)*tp])
        need(all(caps[iid].get(k)==v for k,v in expected.items()),'v2 native model/TP/physical identity differs')
        launch=next(row for row in launches if row['spec']['instance_id']==iid)
        audit_native_launch(launch,caps[iid],instance_id=iid,gpus=range(index*tp,(index+1)*tp),model_id=plan['model_id'])
        identities[iid]=expected
    if 'frequency_domain' in plan:identity=with_domain(identity,plan['frequency_domain'])
    require_same_domain(plan,report)
    need(report.get('frequency_domain_ref')==plan.get('frequency_domain_ref'),'report frequency domain ref differs')
    return identity,identities


def _interference(root,raws,report,identities,uuids):
    from pdblend.bench.comparison_acceptance import _equal
    if report.get('schema') == single_pass.COLLECTION_SCHEMA:
        need(not raws and not report.get('interference_peer_states'),
             'single-pass development cannot relabel interference observations')
        return single_pass.development_qualification([], uuids), [], dict(
            status='not_measured_development', replayed=False, continuous_idle_clock_coverage=False)
    ids=tuple(identities);tp=next(iter(identities.values()))['tp'];checks=[];intervals=[]
    need('interference_peer_states' in report,'v2 all-peer preparation evidence is mandatory')
    frequencies=plan_frequencies(dict(report,model_id=next(iter(identities.values()))['model_id'],tp=tp,pp=1))
    peers=audit_interference_peers(report,raws,identities,frequencies=frequencies)
    for f in frequencies:
        for repeat in range(3):
            summaries={}
            for phase in ('isolated','parallel'):
                for index,iid in enumerate(ids):
                    raw=raws[root/'interference'/f'{f}-{repeat}-{iid}-{phase}.json']
                    need(raw['point']==dict(role='decode',batch=8,prompt_tokens=1024,output_tokens=64,
                        frequency_mhz=f,purpose='interference',seed=9701,repeat=repeat,
                        **({'frequency_domain_sha256':report['frequency_domain_sha256']} if 'frequency_domain' in report else {})),
                        'v2 interference shape changed')
                    rows=_audited_window(raw,identities[iid]);watts=_power(raw,tp,list(range(index*tp,(index+1)*tp)))
                    need(watts>0,'v2 interference power must be a positive observation')
                    summaries[(phase,iid)]=dict(latency_ms=statistics.median(r['latency_ms'] for r in rows),
                        power_w=watts,start_s=raw['start_s'],end_s=raw['end_s'])
                    intervals.append((raw['measurement_started_s'],raw['drain']['response_at_s'],phase,f,repeat,iid))
            overlap=min(summaries[('parallel',i)]['end_s'] for i in ids)-max(summaries[('parallel',i)]['start_s'] for i in ids)
            for iid in ids:
                old,new=summaries[('isolated',iid)],summaries[('parallel',iid)]
                errors={key:abs(new[key]/old[key]-1) for key in ('latency_ms','power_w')}
                checks.append(dict(instance_id=iid,frequency_mhz=f,repeat=repeat,relative_errors=errors,
                    common_window_s=overlap,passed=overlap>=5 and max(errors.values())<=.05))
    for left in intervals:
        if left[2]=='isolated':
            need(not any(left!=right and max(left[0],right[0])<min(left[1],right[1]) for right in intervals),
                 'v2 isolated interference windows overlapped')
    concurrent=all(row['passed'] for row in checks)
    qualification=dict(qualified=True,parallel_qualified=concurrent,mode='parallel' if concurrent else 'serial_resident_fallback',
        limit=.05,checks=checks,exclusive_fleet_gpu_uuids=uuids,energy_comparable=False)
    return qualification,intervals,peers


def _cleanup(report,uuids,ids,last_window,execution):
    from pdblend.bench.comparison_acceptance import _drained
    tp=8//len(ids)
    _,drains=_drained(report['final_drains'],{iid:dict(tp=tp) for iid in ids})
    need(min(drains)>=last_window and max(drains)<=execution['finished_s'],'v2 final drain order differs')
    cleanup=report.get('physical_cleanup',{});observations=cleanup.get('observations',[])
    need(cleanup.get('passed') is True and not cleanup.get('error') and observations
         and finite(cleanup.get('started_s')) and finite(cleanup.get('finished_s'))
         and max(drains)<=cleanup['started_s']<=cleanup['finished_s']<=execution['finished_s'],
         'v2 physical cleanup lifetime/acknowledgement differs')
    for row in observations:
        devices=row.get('devices',[])
        need(finite(row.get('at_s')) and cleanup['started_s']<=row['at_s']<=cleanup['finished_s']
             and len(devices)==8 and [d.get('gpu') for d in devices]==list(range(8))
             and [d.get('gpu_uuid') for d in devices]==uuids
             and all(isinstance(d.get('compute_pids'),list) and all(type(p)is int and p>0 for p in d['compute_pids']) for d in devices),
             'v2 physical cleanup GPU/process observations differ')
    need(all(a['at_s']<=b['at_s'] for a,b in zip(observations,observations[1:]))
         and all(not d['compute_pids'] for d in observations[-1]['devices']), 'v2 owned physical fleet was not empty')
    starts=report.get('actual_engine_starts',{})
    need(set(starts)==set(ids) and all(rows for rows in starts.values())
         and all(row.get('kind')=='start' and row.get('instance')==iid and type(row.get('pid'))is int
                 and row['pid']>0 and finite(row.get('t_s')) and row['t_s']<=cleanup['started_s']
                 for iid,rows in starts.items() for row in rows)
         and report.get('engine_loads')==sum(len(rows) for rows in starts.values()),
         'v2 actual engine start/load inventory differs')


def replay_evidence(evidence_ref,*,path_map=()):
    from pdblend.bench.comparison_acceptance import _equal
    resolver=Resolver(path_map);evidence=resolver.read(evidence_ref)
    need(evidence.get('schema') in (SCHEMA, single_pass.EVIDENCE_SCHEMA) and evidence.get('formal_eligible') is False,'unknown v2 timing replay evidence')
    manifest=resolver.read(evidence['attempt_manifest']);execution=resolver.read(evidence['worker_execution'])
    need(digest(evidence['queue_job'])==evidence['queue_job_sha256'],'captured v2 queue job differs')
    _validate_attempt(manifest,execution,evidence['queue_job'])
    translated=[(r['path'],str(resolver.path(r['resolved_path']))) for r in [*evidence['references'].values(),*evidence['samples']]]
    resolver=Resolver([*path_map,*translated,*evidence.get('path_map',[]),*mounts(execution['argv'])])
    root,report,inputs,plan,refs,raw_refs,interference,ids=_files(resolver.path(evidence['attempt_root']),manifest,execution,resolver)
    development = single_pass.is_single_pass(plan)
    need(evidence['schema'] == (single_pass.EVIDENCE_SCHEMA if development else SCHEMA),
         'single-pass evidence cannot claim original three-repeat qualification')
    need(set(evidence['references'])==set(refs) and all(evidence['references'][k]['sha256']==r['sha256']
         and resolver.path(evidence['references'][k]['path'])==resolver.path(r['path']) for k,r in refs.items())
         and [{k:r[k] for k in ('path','sha256')} for r in evidence['samples']]==raw_refs,
         'captured v2 reference inventory differs')
    need(len(evidence['interference'])==len(interference) and
         {resolver.path(r['path']) for r in evidence['interference']}==set(interference),'captured v2 interference inventory differs')
    raw_interference={resolver.path(ref['path']):resolver.read(ref) for ref in evidence['interference']}
    identity,identities=_identities(report,inputs,plan,manifest,resolver,ids)
    qualification,intervals,peers=_interference(root,raw_interference,report,identities,manifest['gpu_uuids'])
    need(_equal(resolver.read(refs['measurement_qualification']),qualification),'v2 interference qualification does not reproduce')
    owned_intervals=[]
    def windows():
        for owner in report['window_owners']:
            raw=resolver.read(owner['raw']);point=dict(raw['point']);repeat=point.pop('repeat')
            need(resolver.path(owner['raw']['path']).name==f'{digest(point)[:20]}-{repeat}.json',
                 'v2 raw point/repeat does not match bound filename')
            if raw.get('status')=='unsupported_capacity':
                owned_intervals.append((raw['drain_received_s'],raw['observed_s'],owner['instance_id']))
            else:owned_intervals.append((raw['measurement_started_s'],raw['drain']['response_at_s'],owner['instance_id']))
            yield dict(instance_id=owner['instance_id'],raw=raw)
    partition=partition_windows(plan,windows(),identities=identities)
    need(_equal(partition,resolver.read(refs['window_partition'])),'v2 measured/unsupported partition does not reproduce')
    need(report.get('measured_windows')==len(partition['measured'])
         and report.get('unsupported_windows')==len(partition['unsupported']), 'v2 measured/unsupported counts differ')
    for iid in ids:
        owned=sorted(r for r in owned_intervals if r[2]==iid)
        need(all(a[1]<=b[0] for a,b in zip(owned,owned[1:])), 'v2 one engine overlapped its timing/capacity windows')
    if not qualification['parallel_qualified'] and not development:
        ordered=sorted(owned_intervals)
        need(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])), 'v2 serial fallback overlapped windows')
    if intervals:
        need(min(r[0] for r in owned_intervals)>=max(r[1] for r in intervals),'v2 points preceded interference qualification')
    fitted=fit_measured_partition(partition,identity=dict(system='pdblend',**identity),raw_bindings=raw_refs,
        measurement_qualification=dict(qualification,receipt=refs['measurement_qualification']),limits=plan['holdout_limits'])
    need(_equal(fitted,resolver.read(refs['timing_component'])),'v2 measured-only fit/hull/holdout does not reproduce')
    need(report.get('component_qualified') is fitted['component_qualified'],'v2 completion component qualification differs')
    _cleanup(report,manifest['gpu_uuids'],ids,max(r[1] for r in owned_intervals),execution)
    result = dict(schema=single_pass.REPLAY_SCHEMA if development else 'pdblend-native-timing-replay/v2',component=deepcopy(fitted['component']),supported_fit=fitted,
        component_qualified=fitted['component_qualified'],identity=dict(system='pdblend',**identity),
        evidence=binding(resolver.path(evidence_ref['path'])),replayed_windows=len(raw_refs),
        replayed_interference_windows=len(interference),measured_windows=len(partition['measured']),
        unsupported_windows=len(partition['unsupported']),interference_peer_preparation=peers,
        formal_eligible=False,full_profile_qualified=False,auxiliary_power_qualifies_power_component=False)
    if development:
        result.update(qualification_level=single_pass.LEVEL, original_design_qualified=False,
                      parallel_qualified=False)
    return result
