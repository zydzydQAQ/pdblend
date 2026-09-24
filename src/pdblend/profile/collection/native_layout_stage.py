"""Independent timing replay while the same native Fleet is still resident.

A stage receipt proves completed sampling/holdout and a live, drained boundary.
It never fabricates queue success, worker completion or physical GPU release.
An explicit post-job replayer must supply final lifetime gates. Legacy v2
requires job success; the new terminal kind preserves attributable later failure.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import time

from .native_timing_audit import need,finite
from .native_timing_plan import binding,digest
from .native_frequency_domain import plan_frequencies
from .native_timing_replay import Resolver
from .native_runtime_collect import write_new

SCHEMA='pdblend-native-resident-timing-stage/v1'


def capture_resident_timing(report,*,input_manifest_ref,attempt_manifest_ref,specs,fleet,out):
    """Freeze actual pre-layout evidence after final_drains, before cleanup."""
    out=Path(out);need(not out.exists(),'resident timing stage already exists')
    need(not report.get('physical_cleanup'),'resident stage must precede physical cleanup')
    observed=[]
    for spec in specs:
        instance=fleet[spec.instance_id]
        need(instance.alive() and instance.process is not None,'resident stage lost an actual engine')
        need(asdict(instance.spec)==asdict(spec),'resident stage Fleet/spec epoch differs')
        observed.append(dict(instance_id=spec.instance_id,spec=asdict(spec),process_alive=instance.alive(),
            process_pid=instance.process.pid,checked_s=time.time(),
            starts=[dict(e) for e in instance.events if e.get('kind')=='start']))
    saved=write_new(out.with_name(out.stem+'-report.json'),deepcopy(report))
    root=Path(report['timing_component']['path']).resolve().parent
    value=dict(schema=SCHEMA,captured_s=time.time(),report=saved,input_manifest=input_manifest_ref,
        attempt_manifest=attempt_manifest_ref,timing_root=str(root),live_instances=observed,
        interference=[binding(p) for p in sorted((root/'interference').glob('*.json'))],
        resident_stage_only=True,physical_cleanup_verified=False,queue_terminal_verified=False,
        worker_execution_complete=False,formal_eligible=False,full_profile_qualified=False)
    reference=write_new(out,value)
    replay_resident_timing(reference)
    return reference


def _live_boundary(stage,report,ids,last_window):
    from pdblend.bench.comparison_acceptance import _drained
    from pdblend.online.native_control import validate_state
    tp=8//len(ids);_,drains=_drained(report['final_drains'],{iid:dict(tp=tp) for iid in ids})
    need(min(drains)>=last_window and max(drains)<=stage['captured_s'],'resident stage native drain boundary differs')
    rows=stage['live_instances'];need(len(rows)==len(ids) and {r['instance_id'] for r in rows}==set(ids),
                                    'resident stage live inventory incomplete')
    launches={r['spec']['instance_id']:r['spec'] for r in report['actual_launch']}
    for row in rows:
        iid=row['instance_id'];starts=row['starts'];spec=row['spec']
        need(spec==launches[iid] and row.get('process_alive') is True and type(row.get('process_pid')) is int
             and row['process_pid']>0 and finite(row.get('checked_s')) and max(drains)<=row['checked_s']<=stage['captured_s']
             and starts and all(r.get('kind')=='start' and r.get('instance')==iid and type(r.get('pid')) is int
                and r['pid']>0 and finite(r.get('t_s')) and r['t_s']<=row['checked_s'] for r in starts)
             and starts[-1]['pid']==row['process_pid'],'resident stage actual live PID/start/spec differs')
        drain=next(r for r in report['final_drains'] if r['instance_id']==iid)
        for key in ('drain','state'):
            validate_state(drain[key],generation=spec['generation'],tp=tp,pp=1,drained=True,observed_after_s=last_window)
    return dict(live_instances=len(ids),gpu_count=8,native_drained=True,physical_released=False)


def replay_resident_timing(reference,*,path_map=()):
    return _replay_resident_timing(reference,path_map=path_map,stage_schema=SCHEMA,layout_only=True)


def _replay_resident_timing(reference,*,path_map=(),stage_schema,layout_only):
    """Shared strict raw replay; callers retain their explicit lifetime scope."""
    from .native_timing_replay_v2 import _plan,_identities,_interference
    from .native_timing_capacity import partition_windows,fit_measured_partition
    from pdblend.bench.comparison_acceptance import _equal
    resolver=Resolver(path_map);stage=resolver.read(reference)
    need(stage.get('schema')==stage_schema and stage.get('resident_stage_only') is True
         and stage.get('physical_cleanup_verified') is False and stage.get('queue_terminal_verified') is False
         and stage.get('worker_execution_complete') is False and stage.get('formal_eligible') is False
         and finite(stage.get('captured_s')),'resident timing stage makes a false final-lifetime claim')
    manifest=resolver.read(stage['attempt_manifest']);payload=manifest.get('payload',{});uuids=manifest.get('gpu_uuids',[])
    need(manifest.get('immutable') is True and payload.get('system')=='pdblend'
         and payload.get('gpu_count')==8 and payload.get('exclusive') is True and payload.get('reserve_host') is True
         and 'pdblend.profile.collection.native_timing_collect' in payload.get('argv',[])
         and len(uuids)==len(set(uuids))==8 and all(isinstance(u,str) and u.startswith('GPU-') for u in uuids),
         'resident timing stage lacks the bound exclusive eight-GPU attempt')
    need(payload['input_manifest']==stage['input_manifest'],'resident timing attempt/input binding differs')
    inputs=resolver.read(stage['input_manifest']);report=resolver.read(stage['report'])
    need(inputs.get('schema')=='pdblend-native-timing-inputs/v2' and inputs.get('system')=='pdblend'
         and report.get('schema')=='pdblend-native-timing-collection/v2' and report.get('system')=='pdblend'
         and not report.get('error') and not report.get('cleanup_errors') and not report.get('physical_cleanup'),
         'resident timing stage collection is incomplete or already released')
    plan=_plan(inputs,resolver);source=resolver.read(inputs['source_manifest']);root=resolver.path(stage['timing_root'])
    if layout_only:
        need(plan['model_id']=='Qwen2.5-32B-Instruct' and plan['tp']==2 and plan['resident_instances']==4,
             'resident layout bridge is restricted to native32 TP2/M4')
    need(root==resolver.path(stage['attempt_manifest']['path']).parent/'native-timing'
         and report['point_plan']==inputs['point_plan'] and report['capacity_policy']==plan['capacity_policy'],
         'resident stage root/plan/capacity binding differs')
    need(digest(source['files'])==source['source_sha256']==inputs['source_sha256']==payload['source_sha256']
         and inputs['image_digest']==payload['image_digest'],'resident timing source/image identity differs')
    source_root=resolver.path(inputs['source_manifest']['path']).parent
    for name,checksum in source['files'].items():
        path=(source_root/name).resolve()
        need(path.is_relative_to(source_root) and binding(path)['sha256']==checksum,'resident source bytes differ: '+name)
    ids=tuple('pd-timing-'+str(i) for i in range(plan['resident_instances']))
    identity,identities=_identities(report,inputs,plan,manifest,resolver,ids)
    raw_refs=report['raw_bindings'];paths=[resolver.path(r['path']) for r in raw_refs]
    expected={f'{digest(p)[:20]}-{r}.json' for p in plan['points'] for r in range(p['repeats'])}
    need(len(paths)==len(set(paths))==len(expected) and {p.name for p in paths}==expected
         and all(p.parent==root/'samples' for p in paths) and set((root/'samples').glob('*.json'))==set(paths),
         'resident timing raw point inventory incomplete')
    owners=report['window_owners'];need(len(owners)==len(raw_refs) and [r['raw'] for r in owners]==raw_refs,
                                      'resident timing owner/raw binding order differs')
    interference=[root/'interference'/f'{f}-{r}-{iid}-{phase}.json' for f in plan_frequencies(plan)
                  for r in range(3) for iid in ids for phase in ('isolated','parallel')]
    need(set((root/'interference').glob('*.json'))==set(interference)
         and len(stage['interference'])==len(interference)
         and {resolver.path(r['path']) for r in stage['interference']}==set(interference),
         'resident interference raw inventory incomplete')
    raws={resolver.path(ref['path']):resolver.read(ref) for ref in stage['interference']}
    qualification,intervals,peers=_interference(root,raws,report,identities,uuids)
    need(_equal(qualification,resolver.read(report['measurement_qualification'])),
         'resident interference qualification does not reproduce')
    owned=[]
    def windows():
        for owner in owners:
            raw=resolver.read(owner['raw']);point=dict(raw['point']);repeat=point.pop('repeat')
            need(resolver.path(owner['raw']['path']).name==f'{digest(point)[:20]}-{repeat}.json',
                 'resident raw point/filename differs')
            if raw.get('status')=='unsupported_capacity':owned.append((raw['drain_received_s'],raw['observed_s'],owner['instance_id']))
            else:owned.append((raw['measurement_started_s'],raw['drain']['response_at_s'],owner['instance_id']))
            yield dict(instance_id=owner['instance_id'],raw=raw)
    partition=partition_windows(plan,windows(),identities=identities)
    need(_equal(partition,resolver.read(report['window_partition'])) and report['measured_windows']==len(partition['measured'])
         and report['unsupported_windows']==len(partition['unsupported']),'resident measured/capacity partition differs')
    for iid in ids:
        serial=sorted(r for r in owned if r[2]==iid)
        need(all(a[1]<=b[0] for a,b in zip(serial,serial[1:])),'resident instance timing windows overlap')
    if not qualification['parallel_qualified']:
        serial=sorted(owned);need(all(a[1]<=b[0] for a,b in zip(serial,serial[1:])), 'resident serial fallback windows overlap')
    need(min(r[0] for r in owned)>=max(r[1] for r in intervals),'resident timing preceded interference qualification')
    fitted=fit_measured_partition(partition,identity=dict(system='pdblend',**identity),raw_bindings=raw_refs,
        measurement_qualification=dict(qualification,receipt=report['measurement_qualification']),limits=plan['holdout_limits'])
    need(_equal(fitted,resolver.read(report['timing_component'])) and report.get('component_qualified') is fitted['component_qualified'],
         'resident measured-only timing fit/holdout differs')
    boundary=_live_boundary(stage,report,ids,max(r[1] for r in owned))
    return dict(schema='pdblend-native-resident-timing-replay/v1',component=deepcopy(fitted['component']),supported_fit=fitted,
        component_qualified=fitted['component_qualified'],identity=dict(system='pdblend',**identity),
        evidence=binding(resolver.path(reference['path'])),raw_bindings=raw_refs,timing_component=report['timing_component'],
        replayed_windows=len(raw_refs),replayed_interference_windows=len(interference),interference_peer_preparation=peers,
        timing_first_window_s=min(r[0] for r in owned),timing_last_window_s=max(r[1] for r in owned),
        collection_first_window_s=min(r[0] for r in intervals),
        resident_boundary=boundary,resident_stage_only=True,physical_cleanup_verified=False,queue_terminal_verified=False,
        formal_eligible=False,full_profile_qualified=False)


def verify_final_timing(stage_ref,final_ref,*,path_map=()):
    """Finalize through the explicitly selected strict terminal evidence kind."""
    evidence=Resolver(path_map).read(final_ref)
    if evidence.get('schema')=='pdblend-native-terminal-timing-stage-evidence/v1':
        from .native_timing_stage import replay_terminal_evidence as replay_evidence
    else:
        from .native_timing_replay_v2 import replay_evidence
    from pdblend.bench.comparison_acceptance import _equal
    resident=replay_resident_timing(stage_ref,path_map=path_map);final=replay_evidence(final_ref,path_map=path_map)
    need(_equal(resident['supported_fit'],final['supported_fit']) and resident['identity']==final['identity'],
         'final timing replay differs from the frozen pre-layout component/raw observations')
    return final
