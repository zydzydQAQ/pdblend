"""Prove zero output only for a traced request that never reached native forwarding."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
BASE = ROOT / 'common/uniform-rate-20260909-v2/metrics.py'
BASE_SHA = 'bf7e61f5c56dcae7b1e4b1a3c3a10e3f2502cc36ed05bfe86e011ee12cdf831d'
# In this exact runtime, forward_started_s is assigned synchronously after
# awaiting admission and before either prefill or decode native HTTP submission.
LEGACY_RUNTIME_SHA = '251490dff01ec10681bcf5ab25cf394f71ddb6df3477c518e259712ac29b3d1c'
DOMAIN_RUNTIME_SHA = '9833760bda5f1c94a85c044bc351b091f3d8de6b1b78c276afda44b84c141b9c'
# Closed set of actual qualified host/model combinations. The domain successor
# changes qualified frequency plumbing; request forwarding is identical.
RUNTIME_IDENTITIES = {
    ('iZwz9i5bte3xkpmcoes3t2Z', '32b'): LEGACY_RUNTIME_SHA,
    ('iZwz9gfq11hx1sbob59yrgZ', '7b'): LEGACY_RUNTIME_SHA,
    ('iZwz9i5bte3xkpmcoes3t2Z', '14b'): DOMAIN_RUNTIME_SHA,
    ('iZwz9274emxme9019d2sjgZ', '14b'): DOMAIN_RUNTIME_SHA,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def need(condition, why):
    if not condition:
        raise ValueError(why)


def checked(reference):
    data=Path(reference['path']).read_bytes()
    need(hashlib.sha256(data).hexdigest()==reference['sha256'],'evidence digest changed: '+reference['path'])
    return json.loads(data)


def audit(checkpoint_reference):
    cp=checked(checkpoint_reference)
    row=cp['row'];binding=checked(cp['binding'])
    expected_runtime = RUNTIME_IDENTITIES.get((binding['hostname'], row['model']))
    need(expected_runtime is not None and row['system'] in ('mixed','distserve','dynamollm'),
         'unrecognized physical host/model/runtime family')
    need(sha(BASE)==BASE_SHA,'base token auditor changed')
    spec=importlib.util.spec_from_file_location('_unforwarded_token_base',BASE)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    manifest=checked(cp['host_manifest'])
    runtime=Path(binding['host_release'])/'src/ecopadg/serving/runtime.py'
    need(sha(runtime)==expected_runtime and manifest['files']['src/ecopadg/serving/runtime.py']==expected_runtime,
         'unrecognized forward-before-native source')
    need(binding['files'][str(runtime)]==expected_runtime,'executed runtime is not pinned by binding')
    receipt=checked(cp['receipt'])
    need(receipt['measurement_valid'] is True and receipt['child_stopped'] is True
         and receipt['clock_restore_complete'] is True and not receipt['outer_cleanup_errors']
         and all(x['complete'] is True for x in receipt['restoration'].values()), 'unclean measured operation')
    directory=Path(cp['receipt']['path']).parents[2]/'cells'/row['cell_id']
    paths={name:directory/name for name in ('bench.csv','control.jsonl','runtime_config.json','summary.json')}
    for path in paths.values():
        need(cp['artifacts'][str(path)]==sha(path),'unpinned or changed request/journal evidence')
    config=json.loads(paths['runtime_config.json'].read_text())
    need(config['evaluation_protocol']=='evaluation-v3' and config['journal']==str(paths['control.jsonl']),
         'request timing instrumentation is not enabled')
    with paths['bench.csv'].open() as stream:
        requests=list(csv.DictReader(stream))
    trace=checked(dict(path=row['trace'],sha256=row['trace_sha256']))
    need(len(requests)==len(trace['requests'])==row['n_requests'], 'incomplete trace denominator')
    by_client={r['request_id']:r for r in requests}
    need(len(by_client)==len(requests) and set(by_client)=={str(r['idx']) for r in requests}, 'duplicate client request IDs')
    events=[json.loads(line) for line in paths['control.jsonl'].read_text().splitlines()]
    timings=[e for e in events if e.get('kind')=='request_timing']
    ends=[e for e in events if e.get('kind')=='request_end']
    need(len(timings)==len(ends)==len(requests), 'request journal is truncated or duplicated')
    timing_by_client={str(e['client_request_id']):e for e in timings}
    ending_by_id={e['request_id']:e for e in ends}
    need(set(timing_by_client)==set(by_client) and len(ending_by_id)==len(ends)
         and {e['request_id'] for e in timings}==set(ending_by_id), 'controller/client request mapping differs')
    details=[]
    for client,request in by_client.items():
        if base.received_output_count(request) is not None:
            continue
        if request.get('token_count_source')=='server_usage' and base.truth(request.get('token_ids_verified')):
            continue
        need(base.truth(request.get('request_timeout')) and request['error']=='request_hard_timeout'
             and not base.truth(request['success']) and request['token_evidence_schema']=='2', 'unknown output is not a traced deadline')
        need(not base.truth(request['token_stream_opened']) and not base.truth(request['token_stream_parse_error'])
             and base.truth(request['token_sequence_verified']) and int(request['n_text_chunks'])==0
             and int(request['generated_tokens'])==int(request['received_token_count'])==0
             and json.loads(request['received_token_ids'])==json.loads(request['received_token_events'])==[]
             and not request['first_token_s'] and not request['last_token_s'] and not request['http_status'],
             'unknown request has output or response evidence')
        timing=timing_by_client[client];rid=timing['request_id'];ending=ending_by_id[rid]
        own=[e for e in events if e.get('request_id')==rid]
        need({e['kind'] for e in own}=={'request_timing','request_end'} and len(own)==2,
             'request was admitted, forwarded, emitted, or not cleanly cancelled')
        need(not timing['completed'] and not ending['completed'] and 'queued_s' in timing
             and 'forward_started_s' not in timing and 'first_token_s' not in timing
             and 'stream_end_s' not in timing, 'request passed the native-forward boundary')
        for key,target in [('planned_arrival_s','planned_arrival_s'),('actual_dispatch_s','actual_dispatch_s'),
                           ('hard_deadline_s','request_deadline_s')]:
            need(math.isclose(float(timing[key]),float(request[target]),abs_tol=1e-5,rel_tol=0),
                 'controller timing belongs to another request')
        need(float(timing['actual_dispatch_s'])<=timing['handler_arrival_s']<=timing['queued_s']
             <=timing['hard_deadline_s']<=float(ending['at_s'])<=timing['cleanup_end_s'], 'deadline or cleanup order differs')
        need(math.isclose(timing['hard_deadline_s']-timing['planned_arrival_s'],120.,abs_tol=1e-5),
             'hard deadline changed')
        details.append(dict(request_id=client,controller_request_id=rid,actual_output_tokens=0,
            classification='hard_deadline_before_native_forwarding',request_timing=timing,request_end=ending))
    need(details,'no unknown request needs this independent diagnosis')
    return dict(schema='independent-unforwarded-deadline-zero-output-v2',passed=True,
        independently_recomputed=True,checkpoint=checkpoint_reference,receipt=cp['receipt'],
        binding=cp['binding'],host_manifest=cp['host_manifest'],runtime_source=ref(runtime),
        base_auditor=ref(BASE),auditor=ref(__file__),artifacts={str(p):sha(p) for p in paths.values()},
        zero_output_details=details,full_request_timing_count=len(timings),full_request_end_count=len(ends),
        every_other_request_has_exact_received_or_terminal_output=True,
        changes_historical_slo=False,changes_timeout_classification=False,
        reasoning='Pinned runtime writes forward_started_s before any native request; complete per-request timing has queued_s but no forward marker through the original hard deadline and acknowledged cleanup.')
