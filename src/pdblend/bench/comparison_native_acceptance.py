"""Shared raw checks for native retained-KV comparison adapters.

Only transport, source, launch, reset and measurement are shared. Each owning
system must independently audit its deployment, request and policy receipts.
No existing frozen Mixed or EcoServe audit is changed by this module.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

from .comparison_acceptance import _bound, _need, _finite, _equal, _state, _drained
from .resident_session import digest, engine_signature
from .comparison_metrics import canonical_outcomes, reduce_comparison
from .comparison_metering import summarize_comparison


def native_topology(point, identity):
    engine_signature(identity)
    _need(point.get('system') in ('distserve','pdblend'), 'own native system required')
    _need(identity.get('dtype') == 'bfloat16' and identity.get('entrypoint') == 'pdblend_runtime.serve'
          and identity.get('worker_extension') == 'native_v1', 'native entrypoint/dtype/worker differs')
    fleet=identity.get('fleet_gpu_uuids',[])
    _need(len(fleet)==len(set(fleet))==8 and all(str(u).startswith('GPU-') for u in fleet),
          'exclusive eight-board identity required')
    instances=identity['instances']
    _need(all(r['pp']==1 and set(r['gpu_uuids'])<=set(fleet) and r['tp'] in (1,2,4,8) for r in instances),
          'native placement lies outside physical fleet')
    for row in instances:
        options=row['launch_options']
        _need(options.get('max_num_seqs')==32 and options.get('max_model_len')==8192
              and options.get('kv_connector')=='P2pNcclConnector', 'native batch/context/KV policy differs')
    _need(point.get('engine_identity')==identity, 'point engine signature differs')
    return {r['instance_id']:r for r in instances}


def audit_native_startup(point, identity, startup, instances, raw_refs):
    _need(startup.get('engine_signature') == engine_signature(identity), 'startup engine signature differs')
    _need(startup.get('exclusive_gpu_uuids') == identity['fleet_gpu_uuids'], 'startup physical fleet differs')
    source = _bound(startup.get('source_manifest'))
    files = source.get('files', {})
    _need(files and source.get('source_sha256') == digest(files), 'source inventory digest differs')
    root = Path(startup['source_manifest']['path']).parent
    for name, expected in files.items():
        path = (root/name).resolve()
        _need(path.is_relative_to(root.resolve()) and path.is_file()
              and hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'runtime source bytes differ: '+name)
    runtime = {k:v for k,v in files.items() if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))}
    measurement = {k:v for k,v in files.items() if k.startswith('pdblend/measure/') or k in (
        'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py',
        'pdblend/bench/client.py')}
    _need(runtime and measurement and identity['runtime_source_sha256'] == digest(runtime)
          and identity['measurement_source_sha256'] == digest(measurement), 'runtime/measurement source binding differs')
    launch = startup.get('actual_launch_identity', {})
    for key in ('image_digest', 'runtime_source_sha256', 'measurement_source_sha256'):
        _need(launch.get(key) == identity.get(key) and identity.get(key), 'actual launch differs: '+key)
    _need(all(startup.get(k) == identity[k] for k in ('model_hash','tokenizer_hash')),
          'verified startup model/tokenizer inventory differs')
    fingerprints = startup.get('source_fingerprints', {})
    _need(fingerprints.get('runtime_files') == runtime and fingerprints.get('measurement_files') == measurement,
          'startup source subsets differ from full manifest')
    _need(launch.get('source_revision') == source['source_sha256'], 'actual source revision differs')
    actual = launch.get('instances', [])
    _need(len(actual) == len(instances) and {r.get('instance_id') for r in actual} == set(instances),
          'actual launch fleet incomplete')
    for row in actual:
        frozen = instances[row['instance_id']]
        _need(row.get('gpu_uuids') == frozen['gpu_uuids'], 'actual instance UUID assignment differs')
        env = row.get('environment', {})
        _need(identity.get('environment') and all(env.get(k) == v for k,v in identity['environment'].items()),
              'actual launch environment differs')
        expected_devices = ','.join(str(identity['fleet_gpu_uuids'].index(u)) for u in frozen['gpu_uuids'])
        _need(env.get('CUDA_VISIBLE_DEVICES') == expected_devices, 'actual CUDA device order differs')
        argv = row.get('argv', [])
        _need(isinstance(argv, list) and all(isinstance(v, str) for v in argv)
              and 'pdblend_runtime.serve' in argv, 'actual process argv missing')
        for flag, value in (('--tensor-parallel-size', str(frozen['tp'])), ('--pipeline-parallel-size','1'),
                            ('--max-num-seqs','32'), ('--max-model-len','8192'), ('--dtype','bfloat16')):
            occurrences = [index for index,arg in enumerate(argv) if arg == flag or arg.startswith(flag+'=')]
            _need(len(occurrences) == 1, 'actual process argument missing or overridden: '+flag)
            index = occurrences[0]
            actual_value = argv[index].split('=',1)[1] if '=' in argv[index] else (argv[index+1] if index+1 < len(argv) else None)
            _need(actual_value == value, 'actual process argument differs: '+flag)
        options = frozen['launch_options']
        _need(options.get('kv_connector') == 'P2pNcclConnector', 'native retained KV connector required')
        from pdblend_runtime.probe import NativeSpec
        def argument(flag):
            found = [i for i,a in enumerate(argv) if a == flag or a.startswith(flag+'=')]
            _need(len(found) == 1, 'missing or overridden native argument: '+flag)
            index = found[0]
            return argv[index].split('=',1)[1] if '=' in argv[index] else argv[index+1]
        # Rebuild the complete engine command from the frozen inventory. Ports
        # are lease-owned addresses; every execution switch must still match.
        expected_spec = NativeSpec(frozen['instance_id'], tuple(identity['fleet_gpu_uuids'].index(u)
            for u in frozen['gpu_uuids']), int(argument('--port')), argv[argv.index('pdblend_runtime.serve')+1],
            tp=frozen['tp'], pp=frozen['pp'], **options)
        _need(argv[1:] == expected_spec.command()[1:], 'actual complete engine command differs')
        _need(row.get('launch_options') == options, 'recorded launch options differ')
        _need('--no-enable-prefix-caching' in argv and '--enable-prefix-caching' not in argv,
              'native prefix cache policy differs')
        _need(not any(a.startswith(('--worker-cls','--scheduler-cls')) for a in argv),
              'native worker/scheduler must use the frozen entrypoint defaults')
        module_position = argv.index('pdblend_runtime.serve')
        _need(module_position+1 < len(argv) and Path(argv[module_position+1]).name == point['model_id'],
              'actual launched model path differs')
    caps = startup.get('capabilities', {})
    _need(set(caps) == set(instances), 'capability fleet incomplete')
    for name, instance in instances.items():
        cap = caps[name]
        _need(cap.get('supported') is True and cap.get('tp') == instance['tp'] and cap.get('pp') == 1
              and cap.get('model_id') == point['model_id'] and cap.get('engine_revision') == 'vllm-0.10.1.1'
              and cap.get('gpu_uuids') == instance['gpu_uuids']
              and cap.get('source_revision') == source['source_sha256']
              and all(cap.get(k) == identity[k] for k in ('model_hash','tokenizer_hash','image_digest')),
              'native capability identity differs: '+name)
        _state(cap.get('state'), instance['tp'], cap.get('received_s', cap.get('state', {}).get('response_at_s')))
        _need(all(r.get('retained_kv_supported') is True for r in cap['state']['ranks']),
              'retained KV is unsupported by a native rank')
    refs = startup.get('ordinary_reference', [])
    _need(len(refs) == len(instances) and {r.get('instance_id') for r in refs} == set(instances),
          'ordinary reference fleet incomplete')
    for ref in refs:
        responses = ref.get('responses', [])
        _need(len(responses) == 2, 'two ordinary reference responses required')
        tokens = []
        for response in responses:
            events = response.get('events', [])
            ids = [t for e in events for t in e.get('token_ids', [])]
            _need(events and events[-1].get('finished') is True and len(ids) == 16
                  and all(type(t) is int and t >= 0 for t in ids) and ids == response.get('token_ids'),
                  'ordinary reference token/terminal evidence incomplete')
            tokens.append(ids)
        _need(tokens[0] == tokens[1], 'ordinary reference is not deterministic')
    _drained(startup.get('drain'), instances)
    concurrency = _bound(raw_refs.get('concurrency_environment', startup.get('concurrency_environment')))
    lease = _bound(raw_refs.get('lease_manifest', startup.get('lease_manifest')))
    lease_ref = raw_refs.get('lease_manifest', startup.get('lease_manifest'))
    fleet = identity['fleet_gpu_uuids']
    _need(concurrency.get('allocated_gpu_uuids') == fleet and concurrency.get('physical_gpu_uuids') == fleet
          and concurrency.get('lease_manifest_sha256') == lease_ref['sha256']
          and not concurrency.get('peer_jobs') and concurrency.get('peer_snapshots')
          and all(not row.get('peers') for row in concurrency['peer_snapshots']), 'exclusive physical host lease unproven')
    _need(lease.get('gpu_uuids') == fleet and lease.get('payload', {}).get('exclusive') is True
          and lease.get('payload', {}).get('gpu_count') == 8
          and lease.get('payload', {}).get('reserve_host') is True, 'exclusive eight-GPU lease manifest differs')


def audit_native_reset(reset, instances, origin):
    generations,times=_drained(reset.get('drain'),instances)
    _need(reset.get('initial_clock_mhz')==2520 and _finite(reset.get('reset_s'))
          and reset['reset_s']>=0 and max(times)<=origin, 'reset clock/time differs')
    states,acks=reset.get('reopen_state',{}),reset.get('reopen_ack',{})
    _need(set(states)==set(acks)==set(instances), 'reset native inventory incomplete')
    for iid,spec in instances.items():
        state=states[iid];stamp=state.get('response_at_s')
        generation=_state(state,spec['tp'],stamp)
        _need(max(times)<=stamp<=origin and generation>generations[iid]
              and acks[iid].get('acknowledged') is True and acks[iid].get('generation')==generation
              and reset.get('generation',{}).get(iid)==generation
              and state.get('role')=='mixed' and state.get('mode')=='temporal'
              and all(state.get(k) is True for k in ('accepting','admit_prefill','admit_decode')),
              'fresh generation, role or admission reset differs')
    _need(len(set(reset['generation'].values()))==1, 'communication peers need one reset generation')


def audit_native_meter(identity, snapshot, metering, origin):
    _need(snapshot.get('gpu_uuids')==identity['fleet_gpu_uuids']
          and snapshot.get('gpu_uuid_binding_verified') is True, 'physical NVML identity unproven')
    reduced=summarize_comparison(snapshot,gpu_uuids=identity['fleet_gpu_uuids'],origin_s=origin,
        duration_s=150,tail_end_s=metering['tail_end_s'],gpu_uuid_binding_verified=True)
    _need(reduced['energy_comparable'] and _equal(reduced,metering),
          'raw eight-board measurement differs or coverage is incomplete')
    return reduced


def audit_native_metrics(point, trace, outcomes, events, origin, canonical_rows, metrics):
    _need(point.get('seed')==trace.get('seed')==701 and point.get('duration_s')==trace.get('duration_s')==150
          and trace.get('selection_split')=='evaluation', 'fixed independent evaluation protocol differs')
    for key in ('model_id','dataset','rate_rps','slo'):
        _need(trace.get(key)==point.get(key), 'trace differs: '+key)
    requests=trace['requests']
    _need(requests and len(outcomes)==len(requests), 'complete terminal cohort required')
    _need(all(r.get('idx',i)==i and 0<=r['arrival_s']<150 and r['prompt']
              and 2<=r['max_tokens']<=512 and len(r['prompt'])+r['max_tokens']<=8192
              for i,r in enumerate(requests)), 'unsupported frozen request identity/domain')
    canonical=canonical_outcomes(point['system'],trace,outcomes,service_started_s=origin,journal=events)
    reduced=reduce_comparison(trace,canonical,service_started_s=origin,duration_s=150,
                              slo=(point['slo']['ttft_s'],point['slo']['tpot_s']))
    _need(reduced['token_timing_complete'] and reduced['unresolved_requests']==0,
          'client exact delivery or request terminal evidence incomplete')
    _need(_equal(reduced['request_metrics'],canonical_rows), 'canonical request rows differ')
    _need(all(k in metrics and _equal(v,metrics[k]) for k,v in reduced.items() if k!='request_metrics'),
          'canonical aggregate differs from raw reduction')
    return reduced
