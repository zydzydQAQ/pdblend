"""Explicit post-collection binding and independent replay of native timing.

The frozen collector does not hash its interference files. A NEW externally
bound evidence manifest captures them after the lease job has succeeded; it
does not pretend the old completion already bound those bytes.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

from .native_timing_audit import audit_window, fit_component, need, finite
from .native_timing_plan import binding, build_plan, digest

SCHEMA='pdblend-native-timing-replay-evidence/v1'
INSTANCES=tuple('pd-timing-'+str(i) for i in range(8))


def audit_native_launch(launch, capability, *, instance_id, gpus, model_id):
    """Reconstruct the exact native32 command; accept actual warmup epoch 1."""
    from dataclasses import asdict
    from pdblend_runtime.probe import NativeSpec
    spec=launch.get('spec',{});argv=launch.get('argv')
    need(type(spec.get('port')) is int and 0<spec['port']<65536
         and type(spec.get('generation')) is int and spec['generation']>0
         and Path(spec.get('model','')).name==model_id,'native launch identity differs')
    actual=NativeSpec(instance_id,tuple(gpus),spec['port'],spec['model'],tp=len(gpus),
        max_num_seqs=32,generation=spec['generation'],extra_args=('--enforce-eager','--worker-cls',
            'pdblend.profile.collection.native_timing_worker.PDNativeTimingWorker'))
    need(spec==json.loads(json.dumps(asdict(actual))) and isinstance(argv,list)
         and len(argv)>1 and isinstance(argv[0],str) and bool(argv[0])
         and argv[1:]==actual.command()[1:],'complete native launch spec/command differs')
    environment=dict(CUDA_VISIBLE_DEVICES=','.join(str(g) for g in actual.gpus),
        VLLM_USE_V1='1',NCCL_CUMEM_ENABLE='0',NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='0')
    need(launch.get('environment')==environment,'native launch environment differs')
    state=capability.get('state',{})
    need(capability.get('supported') is True and state.get('max_num_seqs')==32
         and state.get('max_model_len')==8192 and state.get('generation')==spec['generation'],
         'native capability limits/generation differ from actual launch')


class Resolver:
    """Explicit mount/prefix translation; never guesses by basename."""
    def __init__(self,mappings=()):
        self.mappings=sorted([(Path(a),Path(b).resolve()) for a,b in mappings],key=lambda r:len(r[0].parts),reverse=True)

    def path(self,value):
        path=Path(value)
        for old,new in self.mappings:
            if path.is_relative_to(old):
                # An explicitly nested relocation must remain idempotent for
                # references freshly bound to an already resolved local file.
                if new!=old and new.is_relative_to(old) and path.is_relative_to(new):return path.resolve()
                return (new/path.relative_to(old)).resolve()
        return path.resolve()

    def read(self,ref):
        path=self.path(ref['path'])
        need(binding(path)['sha256']==ref['sha256'],'native timing evidence checksum differs: '+str(path))
        return json.loads(path.read_text())


def mounts(argv):
    rows=[]
    for index,arg in enumerate(argv[:-1]):
        if arg=='-v':
            host,target,*_=argv[index+1].split(':')
            if host.startswith('/') and target.startswith('/'):rows.append((target,host))
    return rows


def _validate_attempt(manifest,execution,job):
    payload=manifest.get('payload',{})
    need(manifest.get('immutable') is True and manifest.get('job_id')==job.get('job_id')
         and manifest.get('attempt')==job.get('attempts') and payload==job.get('payload')
         and job.get('status')=='succeeded' and job.get('lease_id') is None,
         'timing attempt is not the completed immutable queue job')
    need(execution.get('status')=='passed' and execution.get('complete') is True
         and execution.get('returncode')==0 and not execution.get('error')
         and finite(execution.get('finished_s')),'timing worker execution did not complete')
    need(payload.get('system')=='pdblend' and payload.get('scope')=='native_cuda_timing_component_only'
         and payload.get('gpu_count')==8 and payload.get('exclusive') is True
         and payload.get('reserve_host') is True
         and 'pdblend.profile.collection.native_timing_collect' in payload.get('argv',[]),
         'timing collection is not the bound exclusive eight-GPU invocation')
    uuids=manifest.get('gpu_uuids',[])
    need(len(uuids)==len(set(uuids))==8 and all(isinstance(u,str) and u.startswith('GPU-') for u in uuids),
         'timing attempt physical UUID inventory incomplete')


def _bound_plan(inputs,resolver):
    plan=resolver.read(inputs['point_plan'])
    # Rebuild the fixed non-evaluation design using resolved paths, retaining
    # the recorded paths in the value comparison and final receipts.
    refs=[plan['query_ledger'],plan['query_bindings']]
    for ref in refs:resolver.read(ref)
    expected=build_plan(*(dict(ref,path=str(resolver.path(ref['path']))) for ref in refs),model_id=plan['model_id'])
    expected.update(query_ledger=plan['query_ledger'],query_bindings=plan['query_bindings'])
    need(plan==expected,'timing plan differs from fixed non-evaluation design')
    return plan


def _files(attempt,manifest,execution,resolver):
    root=attempt/'native-timing'
    completion_ref=binding(root/'completion.json')
    need(execution.get('receipt_sha256',{}).get('native-timing/completion.json')==completion_ref['sha256'],
         'worker did not bind timing completion')
    report=resolver.read(completion_ref)
    need(report.get('schema')=='pdblend-native-timing-collection-v1' and report.get('system')=='pdblend'
         and report.get('status')=='passed' and report.get('complete') is True and report.get('hardware_executed') is True
         and not report.get('cleanup_errors'),'native timing collection incomplete')
    inputs_ref=manifest['payload']['input_manifest'];inputs=resolver.read(inputs_ref)
    need(inputs.get('schema')=='pdblend-native-timing-inputs-v1' and inputs.get('system')=='pdblend', 'timing input schema differs')
    plan=_bound_plan(inputs,resolver)
    source=resolver.read(inputs['source_manifest'])
    need(digest(source['files'])==source['source_sha256']==inputs['source_sha256']==manifest['payload']['source_sha256']
         and inputs['image_digest']==manifest['payload']['image_digest'], 'timing source/image identity differs')
    source_root=resolver.path(inputs['source_manifest']['path']).parent
    for name,checksum in source['files'].items():
        path=(source_root/name).resolve()
        need(path.is_relative_to(source_root) and binding(path)['sha256']==checksum,'timing source file differs: '+name)
    component=resolver.read(report['timing_component'])
    qualification_ref=component['measurement_qualification']['receipt']
    resolver.read(qualification_ref);resolver.read(inputs['model_verification'])
    expected_samples={f'{digest(point)[:20]}-{repeat}.json':dict(point,repeat=repeat)
                      for point in plan['points'] for repeat in range(point['repeats'])}
    raw_refs=report['raw_bindings'];resolved=[resolver.path(r['path']) for r in raw_refs]
    need(len(set(resolved))==len(resolved)==len(expected_samples)
         and {p.name for p in resolved}==set(expected_samples)
         and all(p.parent==root/'samples' for p in resolved), 'timing raw sample coverage/path differs')
    need(component.get('raw_bindings')==raw_refs,'component sample bindings differ from completion')
    for ref in raw_refs:resolver.read(ref)
    interference=[root/'interference'/f'{f}-{repeat}-{iid}-{phase}.json'
        for f in (1500,2520) for repeat in range(3) for iid in INSTANCES for phase in ('isolated','parallel')]
    need(set((root/'interference').glob('*.json'))==set(interference),'timing interference inventory incomplete or unexpected')
    need(set((root/'samples').glob('*.json'))==set(resolved),'timing sample inventory differs')
    references=dict(completion=completion_ref,input_manifest=inputs_ref,point_plan=inputs['point_plan'],
        source_manifest=inputs['source_manifest'],model_verification=inputs['model_verification'],
        timing_component=report['timing_component'],measurement_qualification=qualification_ref,
        query_ledger=plan['query_ledger'],query_bindings=plan['query_bindings'])
    return root,report,inputs,plan,component,references,raw_refs,interference


def capture_evidence(attempt,queue,out,*,path_map=()):
    """Create a NEW manifest after worker AND queue success; never overwrite."""
    attempt,out=Path(attempt).resolve(),Path(out).resolve()
    if json.loads((attempt/'native-timing/completion.json').read_text()).get('schema')=='pdblend-native-timing-collection/v2':
        from .native_timing_replay_v2 import capture_evidence as capture_v2
        return capture_v2(attempt,queue,out,path_map=path_map)
    need(not out.exists(),'refusing to overwrite timing replay evidence')
    manifest=json.loads((attempt/'manifest.json').read_text());execution=json.loads((attempt/'execution.json').read_text())
    jobs=json.loads(Path(queue).read_text())['jobs'];job=jobs.get(manifest['job_id'],{})
    _validate_attempt(manifest,execution,job)
    resolver=Resolver([*path_map,*mounts(execution['argv'])])
    *_,refs,raw_refs,interference=_files(attempt,manifest,execution,resolver)
    value=dict(schema=SCHEMA,created_s=time.time(),binding_scope='new_post_collection_snapshot_not_original_completion_binding',
        formal_eligible=False,full_profile_qualified=False,attempt_root=str(attempt),queue_job=job,queue_job_sha256=digest(job),
        attempt_manifest=binding(attempt/'manifest.json'),worker_execution=binding(attempt/'execution.json'),
        references={name:dict(ref,resolved_path=str(resolver.path(ref['path']))) for name,ref in refs.items()},
        samples=[dict(ref,resolved_path=str(resolver.path(ref['path']))) for ref in raw_refs],
        interference=[binding(p) for p in interference],path_map=[list(x) for x in path_map])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return binding(out)


def _power(raw,tp,gpus):
    rows=raw.get('power_samples',[]);metadata=raw.get('power_metadata',[])
    need(len(rows)==len(metadata) and len(rows)>=2,'interference power sample metadata incomplete')
    expected=dict(mode='instant',source_id='nvml:field:186:scope:0:mW',field_id=186,scope_id=0,value_type=1,return_code=0)
    for (stamp,watts),meta in zip(rows,metadata):
        need(finite(stamp) and len(watts)==tp and all(finite(w) and w>=0 for w in watts)
             and meta.get('gpus')==gpus
             and all(meta.get(k)==[v]*tp for k,v in expected.items())
             and all(len(meta.get(k,[]))==tp and all(finite(t) for t in meta[k]) for k in ('read_started_s','read_finished_s'))
             and all(a<=b<=stamp for a,b in zip(meta['read_started_s'],meta['read_finished_s'])),
             'interference instantaneous power source/acquisition invalid')
    need(all(a[0]<b[0] and all(x<=y for x,y in zip(am['read_finished_s'],bm['read_started_s']))
             for a,b,am,bm in zip(rows,rows[1:],metadata,metadata[1:])),
         'interference power acquisition order differs')
    selected=[(t,v) for t,v in rows if raw['start_s']<=t<raw['end_s']]
    need(len(selected)>=2 and selected[0][0]<=raw['start_s']+1. and selected[-1][0]>=raw['end_s']-1.
         and all(0<b[0]-a[0]<=1. for a,b in zip(selected,selected[1:])), 'interference power coverage incomplete')
    return statistics.mean(sum(v) for _,v in selected)


def _audited_window(raw,identity):
    from pdblend.results.journal import payload_receipt
    need(finite(raw.get('measurement_started_s')) and finite(raw.get('end_s')),
         'timing window timestamps missing')
    for client in raw['client_requests']:
        need(finite(client.get('submitted_s')) and finite(client.get('finished_s'))
             and raw['measurement_started_s']<=client['submitted_s']<=client['finished_s']<=raw['end_s'],
             'timing client is outside its owned window')
        events=client.get('events',[]);count=0;ended=False;previous=client['submitted_s']
        for event in events:
            ids=event.get('token_ids');stamp=event.get('received_s')
            need(not ended and isinstance(ids,list) and all(type(i)is int and i>=0 for i in ids)
                 and finite(stamp) and previous<=stamp<=client['finished_s'], 'timing client stream/timestamps differ')
            count+=len(ids);previous=stamp;ended=event.get('finished') is True
            need(event.get('token_index')==count,'timing client token index differs')
        receipt=payload_receipt(events,journal_path='embedded:events',request_id=client['request_id'])
        need(all(client.get(k)==v for k,v in receipt.items()) and ended
             and count==raw['point']['output_tokens'],'timing raw client budget/hash differs')
    clocks=raw.get('frequency_samples',[])
    need(all(finite(t) for t,_ in clocks) and all(a[0]<b[0] for a,b in zip(clocks,clocks[1:])),
         'timing frequency timestamp order differs')
    rows=audit_window(raw,identity=identity)
    clients={row['request_id']:row for row in raw['client_requests']}
    need(all(all(clients[rid]['submitted_s']<=row['at_s']<=clients[rid]['finished_s']
                 for rid in row['request_ids']) for row in rows),
         'native CUDA sample is outside its bound client lifetime')
    need(raw['drain']['response_at_s']>=raw['end_s'],'timing drain predates the service window')
    return rows


def audit_interference_peers(report,raw_interference,identities,*,frequencies=(1500,2520)):
    """Replay an optional new collector receipt, without upgrading old files.

    Its NVML observations prove the whole fleet reached the requested clock
    before the recorded >=2s idle settle. They are not a continuous frequency
    trace throughout that idle interval.
    """
    from pdblend.bench.comparison_acceptance import _drained
    if 'interference_peer_states' not in report:
        return dict(status='not_recorded_legacy',replayed=False,continuous_idle_clock_coverage=False)
    peers=report['interference_peer_states']
    need(isinstance(peers,list) and len(peers)==len(frequencies) and [r.get('frequency_mhz') for r in peers]==list(frequencies),
         'complete ordered peer-frequency preparation required')
    previous_end=None;checked=[]
    for peer in peers:
        frequency=peer['frequency_mhz']
        _,drain_times=_drained(peer.get('drains'),identities)
        windows=[r for r in raw_interference.values() if r['point']['frequency_mhz']==frequency]
        need(windows,'peer preparation lacks its interference windows')
        clocks=peer.get('clocks',[])
        need(len(clocks)==len(identities) and {r.get('instance_id') for r in clocks}==set(identities),
             'peer clock ACK inventory incomplete')
        clock_times=[]
        for row in clocks:
            clock=row.get('clock',{});identity=identities[row['instance_id']]
            physical=clock.get('gpus',[])
            need(clock.get('acknowledged') is True and clock.get('success') is True
                 and clock.get('requested_frequency_mhz')==frequency and finite(clock.get('at_s'))
                 and len(physical)==identity['tp'] and [g.get('gpu_uuid') for g in physical]==identity['gpu_uuids']
                 and all(finite(g.get('frequency_mhz')) and g['frequency_mhz']>0 for g in physical),
                 'peer physical clock ACK differs')
            clock_times.append(clock['at_s'])
        observations=peer.get('observations',[])
        need(observations and all(finite(r.get('at_s')) and len(r.get('frequencies_mhz',[]))==8
             and all(finite(f) and f>0 for f in r['frequencies_mhz']) for r in observations)
             and all(a['at_s']<b['at_s'] for a,b in zip(observations,observations[1:]))
             and all(abs(f-frequency)<=15 for f in observations[-1]['frequencies_mhz']),
             'whole-fleet peer frequency did not reach the requested clock')
        start,end=peer.get('settle_started_s'),peer.get('observed_s')
        need(finite(start) and finite(end) and end-start>=2.
             and max(drain_times)<=min(clock_times) and all(a<=b for a,b in zip(clock_times,clock_times[1:]))
             and max(clock_times)<=observations[0]['at_s']<=observations[-1]['at_s']<=start
             and end<=min(r['measurement_started_s'] for r in windows)
             and (previous_end is None or previous_end<=min(drain_times)),
             'peer drain/clock/settle/interference order differs')
        previous_end=max(r['drain']['response_at_s'] for r in windows)
        checked.append(dict(frequency_mhz=frequency,instances=len(identities),physical_gpus=8,
                            settle_s=end-start,stable_observed_s=observations[-1]['at_s']))
    return dict(status='recorded_and_replayed',replayed=True,frequencies=checked,
                continuous_idle_clock_coverage=False)


def replay_evidence(evidence_ref,*,path_map=()):
    """Verify every byte binding, then recompute all raw audits and NNLS fits."""
    from pdblend.bench.comparison_acceptance import _equal,_drained
    resolver=Resolver(path_map);evidence=resolver.read(evidence_ref)
    if evidence.get('schema')=='pdblend-native-timing-replay-evidence/v2':
        from .native_timing_replay_v2 import replay_evidence as replay_v2
        return replay_v2(evidence_ref,path_map=path_map)
    need(evidence.get('schema')==SCHEMA and evidence.get('formal_eligible') is False,'unknown timing replay evidence')
    manifest=resolver.read(evidence['attempt_manifest']);execution=resolver.read(evidence['worker_execution'])
    need(digest(evidence['queue_job'])==evidence['queue_job_sha256'],'captured queue job identity differs')
    _validate_attempt(manifest,execution,evidence['queue_job'])
    translated=[(row['path'],str(resolver.path(row['resolved_path']))) for row in
                [*evidence['references'].values(),*evidence['samples']]]
    resolver=Resolver([*path_map,*translated,*evidence.get('path_map',[]),*mounts(execution['argv'])])
    attempt=resolver.path(evidence['attempt_root'])
    root,report,inputs,plan,component,references,raw_refs,interference=_files(attempt,manifest,execution,resolver)
    need(set(evidence['references'])==set(references) and all(
         evidence['references'][k]['sha256']==r['sha256'] and
         resolver.path(evidence['references'][k]['path'])==resolver.path(r['path']) for k,r in references.items())
         and [{k:r[k] for k in ('path','sha256')} for r in evidence['samples']]==raw_refs,
         'captured timing reference inventory differs')
    need(len(evidence['interference'])==len(interference) and
         {resolver.path(r['path']) for r in evidence['interference']}==set(interference), 'captured interference inventory differs')
    raw_interference={resolver.path(ref['path']):resolver.read(ref) for ref in evidence['interference']}
    caps=report['capabilities'];need(set(caps)==set(INSTANCES),'eight native timing capabilities required')
    verified=resolver.read(inputs['model_verification'])
    model=next((v for v in verified.get('models',{}).values() if v.get('model_id')==plan['model_id']),None)
    need(verified.get('all_pass') is True and model and model.get('verified') is True,'verified timing model absent')
    identity=dict(model_id=plan['model_id'],tp=1,pp=1,engine_revision='vllm-0.10.1.1',
        source_revision=inputs['source_sha256'],image_digest=inputs['image_digest'])
    for kind,key in [('weight','model_hash'),('tokenizer','tokenizer_hash')]:
        inventory=[(r['path'],r['bytes'],r['sha256']) for r in model['files'] if r['kind']==kind]
        need(inventory,'verified timing model inventory incomplete')
        identity[key]=digest(inventory)
    identities={}
    launches=report.get('actual_launch',[])
    need(len(launches)==8 and {r.get('spec',{}).get('instance_id') for r in launches}==set(INSTANCES),
         'timing actual engine launch inventory incomplete')
    for index,iid in enumerate(INSTANCES):
        expected=dict(identity,gpu_uuids=[manifest['gpu_uuids'][index]])
        need(all(caps[iid].get(k)==v for k,v in expected.items()),'timing native model/UUID differs')
        launch=next(r for r in launches if r['spec']['instance_id']==iid)
        audit_native_launch(launch,caps[iid],instance_id=iid,gpus=[index],model_id=plan['model_id'])
        identities[iid]=expected
    peer_replay=audit_interference_peers(report,raw_interference,identities)
    qualification=resolver.read(references['measurement_qualification']);checks=[]
    intervals=[]
    for f in (1500,2520):
        for repeat in range(3):
            summaries={}
            for phase in ('isolated','parallel'):
                for iid in INSTANCES:
                    raw=raw_interference[root/'interference'/f'{f}-{repeat}-{iid}-{phase}.json']
                    expected=dict(role='decode',batch=8,prompt_tokens=1024,output_tokens=64,
                        frequency_mhz=f,purpose='interference',seed=9701,repeat=repeat)
                    need(raw['point']==expected,'interference workload differs')
                    rows=_audited_window(raw,identities[iid]);watts=_power(raw,1,[INSTANCES.index(iid)])
                    summaries[(phase,iid)]=dict(latency_ms=statistics.median(r['latency_ms'] for r in rows),
                        power_w=watts,start_s=raw['start_s'],end_s=raw['end_s'])
                    intervals.append((raw['measurement_started_s'],raw['drain']['response_at_s'],phase,f,repeat,iid))
            overlap=min(summaries[('parallel',i)]['end_s'] for i in INSTANCES)-max(summaries[('parallel',i)]['start_s'] for i in INSTANCES)
            for iid in INSTANCES:
                old,new=summaries[('isolated',iid)],summaries[('parallel',iid)]
                errors={key:abs(new[key]/old[key]-1) for key in ('latency_ms','power_w')}
                checks.append(dict(instance_id=iid,frequency_mhz=f,repeat=repeat,relative_errors=errors,
                    common_window_s=overlap,passed=overlap>=5 and max(errors.values())<=.05))
    for left in intervals:
        if left[2]!='isolated':continue
        need(not any(left!=right and max(left[0],right[0])<min(left[1],right[1]) for right in intervals),
             'claimed isolated interference windows actually overlapped')
    concurrent=all(row['passed'] for row in checks)
    rebuilt=dict(qualified=True,parallel_qualified=concurrent,mode='parallel' if concurrent else 'serial_resident_fallback',
        limit=.05,checks=checks,exclusive_fleet_gpu_uuids=manifest['gpu_uuids'],energy_comparable=False)
    need(_equal(qualification,rebuilt),'raw interference replay differs from recorded qualification')
    training=[];holdout=[];windows=[];seen=set()
    point_owner={digest(point):INSTANCES[index%8] for index,point in enumerate(plan['points'])}
    for ref in raw_refs:
        raw=resolver.read(ref);point=dict(raw['point']);repeat=point.pop('repeat')
        need(digest(point) in point_owner and type(repeat)is int and 0<=repeat<3,'unplanned timing raw point')
        need(resolver.path(ref['path']).name==f'{digest(point)[:20]}-{repeat}.json',
             'timing raw point does not match its bound filename')
        key=(digest(point),repeat);need(key not in seen,'duplicate timing raw point/repeat');seen.add(key)
        iid=point_owner[digest(point)];rows=_audited_window(raw,identities[iid])
        (training if point['purpose']=='training' else holdout).extend(rows)
        windows.append((raw['measurement_started_s'],raw['drain']['response_at_s'],iid))
    for iid in INSTANCES:
        owned=sorted(w for w in windows if w[2]==iid)
        need(all(a[1]<=b[0] for a,b in zip(owned,owned[1:])), 'one timing engine sampled overlapping windows')
    if not concurrent:
        ordered=sorted(windows)
        need(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])), 'serial fallback sampled overlapping timing windows')
    need(min(w[0] for w in windows)>=max(w[1] for w in intervals),'training started before interference qualification finished')
    _,drain_times=_drained(report['final_drains'],{iid:dict(tp=1) for iid in INSTANCES})
    need(min(drain_times)>=max(w[1] for w in windows) and max(drain_times)<=execution['finished_s'],
         'final timing drain/worker completion order differs')
    rebuilt_component=fit_component(training,holdout,identity=dict(system='pdblend',**identity),raw_bindings=raw_refs,
        measurement_qualification=dict(rebuilt,receipt=references['measurement_qualification']),limits=plan['holdout_limits'])
    need(_equal(component,rebuilt_component),'native timing coefficients/hull/holdout do not reproduce')
    need(report.get('component_qualified') is rebuilt_component['component_qualified'],'completion timing qualification differs')
    return dict(schema='pdblend-native-timing-replay/v1',component=deepcopy(rebuilt_component),
        evidence=binding(resolver.path(evidence_ref['path'])),replayed_windows=len(raw_refs),
        replayed_interference_windows=len(interference),formal_eligible=False,full_profile_qualified=False,
        auxiliary_power_qualifies_power_component=False,interference_peer_preparation=peer_replay)
