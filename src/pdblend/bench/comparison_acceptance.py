"""Independent raw-evidence acceptance for one fixed-TP Mixed observation.

No profile is required by Mixed. Formal eligibility means an auditable single
observation under the frozen protocol, not SLO success, repeatability, or an
optimality claim. SLO failures remain valid observations when evidence is complete.

``raw_refs`` binds trace, requests, outcomes, events, power, native_result,
canonical_requests, metering, startup_qualification, reset and drain using
{path, sha256}. Startup additionally binds source_manifest,
concurrency_environment and lease_manifest (the latter two may be raw_refs).
These must be immutable snapshots, not a worker's changing live JSON file.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .comparison_metrics import canonical_outcomes, reduce_comparison
from .comparison_metering import summarize_comparison
from .resident_session import digest, engine_signature
from .comparison_journal import iter_comparison_journal as iter_journal
from pdblend.results.power_archive import read_power_archive

SCHEMA = 'mixed-single-observation-acceptance-v1'
REQUIRED_RAW_REFS = ('trace', 'requests', 'outcomes', 'events', 'power',
    'native_result', 'canonical_requests', 'metering', 'startup_qualification', 'reset', 'drain')
MODEL_TP = {'Qwen2.5-7B-Instruct': 1, 'Qwen2.5-14B-Instruct': 1, 'Qwen2.5-32B-Instruct': 2}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _equal(a, b):
    if _finite(a) and _finite(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def _bound(ref, *, journal=False, power=False):
    _need(isinstance(ref, dict) and isinstance(ref.get('path'), str)
          and isinstance(ref.get('sha256'), str), 'missing path/sha256 binding')
    path = Path(ref['path'])
    _need(path.is_file(), 'raw evidence file is missing: ' + str(path))
    _need(hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'],
          'raw evidence checksum differs: ' + str(path))
    if power:
        return read_power_archive(path)
    return list(iter_journal(path)) if journal else json.loads(path.read_text())


def _state(state, tp, received_s, *, drained=True):
    """Validate recorded receipt time, never the time of a later offline audit."""
    _need(isinstance(state, dict), 'native state missing')
    required = ('all_queue', 'running', 'waiting', 'kv_allocations', 'retained_kv_requests',
                'transfer_allocations', 'pending_transfers', 'free_kv_tokens', 'total_kv_tokens')
    _need(all(k in state for k in required), 'native request/KV inventory incomplete')
    generation = state.get('generation')
    _need(type(generation) is int and state.get('acknowledged_generation') == generation
          and state.get('tp') == tp and state.get('pp') == 1
          and state.get('block_size') == 16 and state.get('max_num_seqs') == 32
          and state.get('max_model_len') == 8192 and state.get('healthy') is True
          and state.get('native_evidence_complete') is True and state.get('transport_healthy') is True
          and not state.get('error') and not state.get('runtime_error'), 'native topology/health/generation differs')
    start, end, native = (state.get(k) for k in ('rank_observation_started_s',
                                               'rank_observation_finished_s', 'native_at_s'))
    _need(all(_finite(v) for v in (start, end, native, received_s))
          and start <= end <= received_s + .05 and -.05 <= received_s-native <= 1.,
          'native receipt freshness is unproven')
    ranks = state.get('ranks', [])
    _need(len(ranks) == tp and {r.get('rank') for r in ranks} == set(range(tp)), 'native ranks incomplete')
    for r in ranks:
        _need(r.get('generation') == generation and r.get('healthy') is True
              and r.get('native_evidence_complete') is True and _finite(r.get('at_s'))
              and start-.05 <= r['at_s'] <= end+.05
              and 'pending_transfers' in r and 'transfer_allocations' in r,
              'native rank identity/freshness/inventory differs')
        if drained:
            retained = r.get('retained')
            no_connector = retained is None and r.get('retained_kv_supported') is False
            retained_empty = (isinstance(retained, dict) and all(k in retained and not retained[k] for k in
                ('held_count', 'held_requests', 'held_bytes', 'receiving_transactions')))
            _need(not r['pending_transfers'] and not r['transfer_allocations']
                  and (no_connector or retained_empty), 'rank retained transfer/KV inventory is not empty')
    if drained:
        _need(not any(state[k] for k in required[:-2])
              and type(state['total_kv_tokens']) is int and state['total_kv_tokens'] > 0
              and state['free_kv_tokens'] == state['total_kv_tokens'], 'native drain is not complete')
    return generation


def _drained(rows, instances):
    if isinstance(rows, dict):
        rows = rows.get('states', rows.get('drain'))
    _need(isinstance(rows, list) and len(rows) == len(instances)
          and {r.get('instance_id') for r in rows} == set(instances), 'drain fleet is incomplete')
    generations, times = {}, []
    for row in rows:
        instance = instances[row['instance_id']]
        received = row.get('received_s')
        d, state = row.get('drain', {}), row.get('state', {})
        _need(d.get('acknowledged') is True and d.get('drained') is True, 'drain lacks acknowledgement')
        g = _state(d, instance['tp'], received)
        _need(_state(state, instance['tp'], received) == g, 'generation changed across drain')
        generations[row['instance_id']] = g
        times.append(received)
    return generations, times


def _topology(point, identity):
    tp = MODEL_TP.get(point.get('model_id'))
    _need(point.get('system') == 'mixed' and tp is not None, 'only the three-model Mixed protocol is qualified')
    _need(identity.get('dtype') == 'bfloat16' and identity.get('entrypoint') == 'pdblend_runtime.serve'
          and identity.get('worker_extension') == 'native_v1', 'frozen runtime entrypoint/dtype differs')
    engine_signature(identity)
    fleet = identity.get('fleet_gpu_uuids', [])
    _need(len(fleet) == len(set(fleet)) == 8 and all(str(u).startswith('GPU-') for u in fleet),
          'frozen physical fleet must contain eight unique GPU UUIDs')
    rows = identity['instances']
    _need(len(rows) == 8//tp and all(r['tp'] == tp and r['pp'] == 1 for r in rows), 'fixed TP fleet differs')
    _need([u for r in rows for u in r['gpu_uuids']] == fleet, 'instance fleet does not cover frozen GPU order')
    for row in rows:
        options = row['launch_options']
        _need(options.get('max_num_seqs') == 32 and options.get('max_model_len') == 8192
              and 'kv_connector' in options and options['kv_connector'] is None,
              'Mixed must use maxseq32/context8192 without a KV connector')
    if point.get('engine_identity') is not None:
        _need(_equal(point['engine_identity'], identity), 'point engine identity differs')
    return {r['instance_id']: r for r in rows}


def _startup(point, identity, startup, instances, raw_refs):
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
        _need('--no-enable-prefix-caching' in argv and not any(a.startswith('--kv-transfer-config') for a in argv)
              and not any(a.startswith('--enable-prefix-caching') for a in argv), 'actual KV/prefix-cache policy differs')
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


def _trace(point, trace, requests):
    _need(trace.get('seed') == 701 and trace.get('duration_s') == 150 and trace.get('requests'), 'trace seed/duration/requests differ')
    for key in ('model_id', 'dataset', 'slo', 'rate_rps'):
        _need(trace.get(key) == point.get(key) and point.get(key) is not None, 'trace identity differs: '+key)
    _need(requests.get('seed') == 701 and requests.get('duration_s') == 150, 'executed request seed/duration differs')
    expected, previous = [], -1.
    for idx, row in enumerate(trace['requests']):
        prompt, n, at = row.get('prompt'), row.get('max_tokens'), row.get('arrival_s')
        _need(isinstance(prompt, list) and prompt and all(type(t) is int and t >= 0 for t in prompt)
              and type(n) is int and 2 <= n <= 512 and len(prompt)+n <= 8192
              and _finite(at) and previous <= at < 150 and at >= 0 and row.get('idx', idx) == idx,
              'frozen trace request domain differs')
        expected.append(dict(idx=idx, arrival_s=at, prompt=prompt, max_tokens=n,
                             source=row.get('source', trace['dataset']))); previous = at
    _need(_equal(requests.get('requests'), expected), 'executed prompts/arrivals/token counts differ from trace')


def _routing(events, outcomes, instances):
    counts, active, routed, released = {name:0 for name in instances}, {}, {}, set()
    known = {row['request_id']:row for row in outcomes}
    _need(len(known) == len(outcomes), 'duplicate native request id')
    previous = -math.inf
    for row in events:
        event = row.get('event')
        if event not in ('route', 'release'):
            continue
        rid, stamp = row.get('request_id'), row.get('at_s')
        _need(rid in known and _finite(stamp) and stamp >= previous, 'route/release request or time differs')
        previous = stamp
        if event == 'route':
            candidates = [name for name in instances if counts[name] < 32]
            _need(candidates and rid not in routed, 'route duplicate or beyond fleet capacity')
            name = min(candidates, key=lambda name:(counts[name]/32, counts[name], name))
            _need(row.get('instance_id') == name and row.get('policy') == 'fixed_tp_least_load'
                  and row.get('tp') == instances[name]['tp'] and row.get('load') == counts[name]/32
                  and known[rid].get('instance_id') == name, 'independent least-load replay differs')
            active[rid] = routed[rid] = name; counts[name] += 1
        else:
            _need(rid in active and rid not in released, 'release lacks unique route')
            counts[active.pop(rid)] -= 1; released.add(rid)
            _need(row.get('active') == counts, 'released replica counts differ')
    _need(not active and not any(counts.values()) and set(routed) == released, 'routing counts were not fully released')
    _need(all(row['request_id'] in routed or row.get('error') for row in outcomes), 'successful outcome lacks route')
    return dict(routed_requests=len(routed), released_requests=len(released), peak_limit=32)


def audit_window(point, engine_identity, startup_qualification, reset, native_result,
                 canonical_metrics, metering, drain, raw_refs):
    """Return explicit missing gates; never trust caller-provided passed flags.

    Invalid or missing evidence is reported rather than raising. All supported
    Mixed method gates are needed for both evidence_valid and formal_eligible.
    SLO is independently recomputed and is deliberately a separate result.
    """
    failures, checked, data = {}, [], {}
    recomputed = None

    def gate(name, function):
        try:
            value = function()
        except (ValueError, TypeError, KeyError, OSError, IndexError, AttributeError, OverflowError) as exc:
            failures[name] = str(exc)
            return None
        checked.append(name)
        return value

    for key in REQUIRED_RAW_REFS:
        value = gate('raw.'+key, lambda key=key: _bound(raw_refs.get(key),
                     journal=key in ('events','outcomes'), power=key == 'power'))
        if value is not None:
            data[key] = value
    for key, supplied in (('startup_qualification',startup_qualification), ('reset',reset),
                          ('native_result',native_result), ('metering',metering), ('drain',drain)):
        gate('binding.'+key, lambda key=key,supplied=supplied:
             _need(key in data and _equal(data[key], supplied), 'supplied receipt differs from bound raw file'))
    gate('binding.trace', lambda: _need(raw_refs.get('trace') == point.get('trace'), 'point trace binding differs'))
    instances = gate('mixed.fixed_tp_fleet', lambda: _topology(point, engine_identity))
    if instances:
        gate('mixed.startup_identity', lambda: _startup(point,engine_identity,startup_qualification,instances,raw_refs))
        def check_reset():
            generations, times = _drained(reset.get('drain'), instances)
            _need(reset.get('initial_clock_mhz') == 2520 and _finite(reset.get('reset_s'))
                  and reset['reset_s'] >= 0 and max(times) <= native_result['started_s'], 'reset clock/time unproven')
            states, acks = reset.get('reopen_state', {}), reset.get('reopen_ack', {})
            _need(set(states) == set(acks) == set(instances), 'post-warmup reopen fleet incomplete')
            for name, instance in instances.items():
                state = states[name]
                received = state.get('response_at_s')
                generation = _state(state,instance['tp'],received)
                _need(max(times) <= received <= native_result['started_s'], 'reopen is outside the reset interval')
                _need(acks[name].get('acknowledged') is True and generation > generations[name]
                      and generation == reset.get('generation', {}).get(name)
                      and all(state.get(k) is True for k in ('accepting','admit_prefill','admit_decode'))
                      and state.get('role') == 'mixed' and state.get('mode') == 'temporal',
                      'post-warmup generation/admission reset differs')
        gate('mixed.reset', check_reset)
        gate('mixed.least_load_release', lambda: _routing(data['events'], data['outcomes'], instances))
        def check_drain():
            generations, times = _drained(drain, instances)
            _need(generations == reset.get('generation'), 'window native generation changed')
            _need(min(times) >= native_result['finished_s'] and max(times) <= metering['tail_end_s'],
                  'metering does not cover the final native drain')
        gate('mixed.full_native_drain', check_drain)
    if 'trace' in data and 'requests' in data:
        gate('trace.executed_request_identity', lambda: _trace(point,data['trace'],data['requests']))
    def reduce():
        _need(native_result.get('system') == 'mixed' and native_result.get('native_runner') is True
              and native_result.get('seed') == 701 and native_result.get('duration_s') == 150
              and native_result.get('routing_policy') == 'independent_least_load_fixed_tp'
              and native_result.get('counts_reclaimed') is True, 'native run identity differs')
        origin = native_result['started_s']
        _need(_finite(origin) and _finite(native_result.get('finished_s'))
              and native_result['finished_s'] >= origin+150, 'native service window is not complete')
        for key in ('events','outcomes'):
            _need(native_result.get(key+'_sha256') == raw_refs[key]['sha256'], 'native journal binding differs: '+key)
        rows = data['outcomes']; requests = data['trace']['requests']
        _need(len(rows) == len(requests) and {r.get('idx') for r in rows} == set(range(len(requests))),
              'terminal outcome cohort incomplete or duplicated')
        for row in rows:
            idx = row['idx']; req = requests[idx]
            _need(row.get('request_id') == f'mixed-701-{idx}' and row.get('arrival_s') == req['arrival_s']
                  and row.get('input_tokens') == len(req['prompt']) and row.get('max_tokens') == req['max_tokens']
                  and row.get('sampling_seed') == 701, 'native outcome request identity differs')
        canonical = canonical_outcomes('mixed',data['trace'],rows,service_started_s=origin,journal=data['events'])
        reduced = reduce_comparison(data['trace'],canonical,service_started_s=origin,duration_s=150,
                                    slo=(point['slo']['ttft_s'],point['slo']['tpot_s']))
        _need(reduced['token_timing_complete'] and reduced['unresolved_requests'] == 0,
              'client token delivery or terminal outcome evidence incomplete')
        _need(_equal(reduced['request_metrics'],data['canonical_requests']), 'canonical request rows differ from raw reduction')
        _need(all(k in canonical_metrics and _equal(v,canonical_metrics[k]) for k,v in reduced.items()
                  if k != 'request_metrics'), 'canonical metric aggregate differs from raw reduction')
        return reduced
    recomputed = gate('metrics.client_canonical', reduce)
    def check_metering():
        snapshot = data['power']
        _need(snapshot.get('gpu_uuids') == engine_identity['fleet_gpu_uuids']
              and snapshot.get('gpu_uuid_binding_verified') is True, 'raw NVML physical UUID provenance missing')
        reduced = summarize_comparison(snapshot,gpu_uuids=engine_identity['fleet_gpu_uuids'],
            origin_s=native_result['started_s'],tail_end_s=metering['tail_end_s'],duration_s=150,
            gpu_uuid_binding_verified=True)
        _need(reduced['energy_comparable'], 'eight-board instant power coverage/source is incomplete')
        _need(_equal(reduced,metering), 'reported metering differs from raw sample integration')
        return reduced
    gate('metering.raw_eight_gpu_window', check_metering)
    valid = not failures
    return dict(schema=SCHEMA, evidence_valid=valid, formal_eligible=valid,
                observation_scope='single_observation', slo_pass=recomputed['slo_pass'] if recomputed else False,
                missing_gates=list(failures), gate_failures=failures, checked_gates=checked,
                evidence_sha256=digest(raw_refs), optimality_established=False)
