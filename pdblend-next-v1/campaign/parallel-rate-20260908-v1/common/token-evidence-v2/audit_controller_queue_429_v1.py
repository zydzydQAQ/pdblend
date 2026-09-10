"""Independently identify C's exact controller queue-full response before forwarding."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
METRICS=ROOT/'common/uniform-rate-20260909-v2/metrics_queue_deadlines_v4.py'
METRICS_SHA='343fff02a0646e976f1a45a880a9bb23866273a7b421d92eda47e97a93f7f593'
SOURCES={
    'src/ecopadg/serving/runtime.py':'908ed24a2f21e4a284c8e1ef1394246f51d65ff973a67f0232b66a500dec2ec7',
    'src/ecopadg/serving/admission.py':'1f43ebdfedb8d7f379b05af63d60864a5bf524dbb6d8139592edc30b229c6252',
    'benchmarks/scripts/bench_vllm.py':'ef124dfeccdab635ece210d99fbbb7bc1959e14d1a3113d06e55a55fd21a4b5d',
}
EXPECTED_BODY={'error':{'type':'admission_rejection','code':'admission_queue_full','message':'admission queue full'}}


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def checked(reference):
    raw=Path(reference['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest()==reference['sha256'],'changed pinned evidence: '+reference['path'])
    return json.loads(raw)


def audit(checkpoint_reference):
    cp=checked(checkpoint_reference);row=cp['row'];binding=checked(cp['binding']);manifest=checked(cp['host_manifest'])
    need(binding['hostname']=='iZwz9gfq11hx1sbob59yrgZ' and row['model']=='7b'
         and row['system'] in ('mixed','distserve','dynamollm') and cp['node']=='C',
         'unrecognized physical controller family')
    sources={}
    for name,digest in SOURCES.items():
        path=Path(binding['host_release'])/name
        need(sha(path)==manifest['files'][name]==binding['files'][str(path)]==digest,
             'unrecognized exact controller/collector source')
        sources[str(path)]=digest
    need(sha(METRICS)==METRICS_SHA,'independent metric auditor changed')
    spec=importlib.util.spec_from_file_location('_queue429_independent_metrics',METRICS)
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    raw_metrics=metrics.audit_checkpoint(checkpoint_reference['path'])
    receipt=checked(cp['receipt']);summary=receipt['summary']
    need(not summary['runtime_error'] and summary['drain_complete'] and not summary['incomplete_drain'],
         'runtime or drain engineering failure')
    directory=Path(cp['receipt']['path']).parents[2]/'cells'/row['cell_id']
    names=('bench.csv','control.jsonl','runtime_config.json','summary.json','arrival_window.json','cleanup.json')
    paths={name:directory/name for name in names}
    for path in paths.values():need(cp['artifacts'][str(path)]==sha(path),'changed request/control evidence')
    config=json.loads(paths['runtime_config.json'].read_text())
    need(config['evaluation_protocol']=='evaluation-v3' and config['journal']==str(paths['control.jsonl'])
         and config['comparison_system']==row['system'],'wrong controller timing domain')
    with paths['bench.csv'].open() as stream:requests=list(csv.DictReader(stream))
    rejected={};other={}
    for request in requests:
        rid=request['request_id']
        if request['admission_rejection']=='admission_queue_full':
            need(not metrics.truth(request['success']) and request['http_status']=='429'
                 and not metrics.truth(request['request_timeout']),'queue-full is not an explicit controller rejection')
            prefix='RuntimeError: HTTP 429: '
            need(request['error'].startswith(prefix) and json.loads(request['error'][len(prefix):])==EXPECTED_BODY,
                 'foreign or malformed 429 response')
            need(request['token_evidence_schema']=='2' and not metrics.truth(request['token_stream_opened'])
                 and not metrics.truth(request['token_stream_parse_error']) and metrics.truth(request['token_sequence_verified'])
                 and not metrics.truth(request['received_token_count_exact'])
                 and int(request['input_tokens'])==int(request['generated_tokens'])==int(request['received_token_count'])==int(request['n_text_chunks'])==0
                 and json.loads(request['received_token_ids'])==json.loads(request['received_token_events'])==[]
                 and not any(request[k] for k in ('first_token_s','last_token_s','stream_end_s','output_token_sha256')),
                 'rejected request has accepted stream or output evidence')
            rejected[rid]=request
        else:
            need(not request['admission_rejection'],'unknown admission rejection class')
            other[rid]=request
    need(rejected and len(rejected)+len(other)==len(requests)==row['n_requests'],
         'no rejection or duplicate request denominator')
    events=[json.loads(line) for line in paths['control.jsonl'].read_text().splitlines()]
    timings=[e for e in events if e.get('kind')=='request_timing'];ends=[e for e in events if e.get('kind')=='request_end']
    by_client={str(e['client_request_id']):e for e in timings};by_id={e['request_id']:e for e in ends}
    need(len(timings)==len(ends)==len(by_client)==len(by_id)==len(other)
         and set(by_client)==set(other) and {e['request_id'] for e in timings}==set(by_id),
         'accepted controller journal is incomplete or a rejected request was forwarded')
    for e in events:
        if 'client_request_id' in e:need(str(e['client_request_id']) not in rejected,'rejection has controller admission evidence')
    for client,timing in by_client.items():
        request=other[client];ending=by_id[timing['request_id']]
        for a,b in (('planned_arrival_s','planned_arrival_s'),('actual_dispatch_s','actual_dispatch_s'),('hard_deadline_s','request_deadline_s')):
            need(math.isclose(float(timing[a]),float(request[b]),abs_tol=1e-5,rel_tol=0),'foreign controller/client mapping')
        need(timing['completed']==ending['completed'] and float(ending['at_s'])<=timing['cleanup_end_s'],
             'accepted request cleanup is inconsistent')
        if metrics.truth(request['success']):
            need(timing['completed'] and 'forward_started_s' in timing and 'stream_end_s' in timing,
                 'completed request lacks native forwarding and terminal evidence')
        else:
            need(not timing['completed'] and metrics.truth(request['request_timeout'])
                 and request['error']=='request_hard_timeout','non-rejected failure is not an explicit deadline')
    need(summary['admission_rejections']==len(rejected),'rejection summary differs from all raw responses')
    return dict(schema='independent-controller-queue-full-429-v1',passed=True,independently_recomputed=True,
        checkpoint=checkpoint_reference,binding=cp['binding'],receipt=cp['receipt'],host_manifest=cp['host_manifest'],
        auditor=ref(__file__),metric_auditor=ref(METRICS),sources=sources,artifacts={str(p):sha(p) for p in paths.values()},
        refusal_details=[dict(request_id=rid,actual_output_tokens=0,classification='controller_admission_queue_full') for rid in rejected],
        n_expected=len(requests),controller_rejections=len(rejected),controller_accepted_requests=len(other),
        complete_controller_timing_count=len(timings),request_timeouts=raw_metrics['request_timeouts'],
        actual_output_tokens=raw_metrics['actual_output_tokens'],token_throughput_is_exact=raw_metrics['token_throughput_is_exact'],
        full_request_denominator_preserved=True,changes_historical_slo=False,changes_timeout_classification=False,
        no_PDB_capacity_boundary_claim=True,
        reasoning='Pinned runtime emits this exact 429 only at the two pending-queue-full branches before native forwarding. Exact collector response, empty output, complete accepted-request journal and clean drain agree; rejection is not a request timeout.')
