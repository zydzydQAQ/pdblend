"""CPU-only independent audit. Preserve every original checkpoint and partial flag."""
import collections
import copy
import csv
import json
from pathlib import Path
import statistics
import metrics_snapshot as m

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = HERE.parent / 'ascending-resume-20260909-v2'
UNIFORM = HERE.parent / 'uniform-rate-20260909-v1'
CELL = 'ascending-v1-B-32b-alpaca-r4.5-s701-w100-distserve-slo1-repeat'
CHECKPOINTS = [V2 / 'baseline-distserve-performance-001/results/checkpoints' / (CELL + '1.json'),
    UNIFORM / 'predecessor-distserve-001/results/checkpoints' / (CELL + '2.json')]
HISTORICAL = [ROOT.parent / 'B32B-baseline-main-first-sequence-v1/attempt-001/bindings/distserve/results/checkpoints' / ('32b-alpaca-r' + rate + '-s701-w100-distserve-slo1.json') for rate in ('2.5', '4')]
HISTORICAL += [ROOT / 'B' / directory / 'results/checkpoints' / ('parallel-rate-p4-completion-32b-alpaca-r5-s701-w100-distserve-slo1-repeat' + str(rep) + '.json')
    for directory, rep in [('baselines-reconciled-002', 1), ('baselines-reconciled-007', 2)]]


def raw_checkpoint(path):
    cp = m.read(path)
    for source, digest in cp['artifacts'].items():
        assert m.sha(source) == digest, source
    ref = cp['receipt']
    if isinstance(ref, str):
        ref = dict(path=ref, sha256=cp['receipt_sha256'])
    receipt = m.checked(ref)
    row = cp['row']
    audit = m.audit_receipt(ref['path'], row, receipt_reference=ref)
    assert audit['slo_attainment'] == receipt['summary']['slo_attainment']
    assert receipt['child_exitcode'] == 0 and not receipt.get('error')
    assert not receipt.get('sampling_error') and not receipt.get('integration_error')
    assert receipt['summary']['drain_complete'] and not receipt['summary']['runtime_error']
    directory = Path(ref['path']).parents[2] / 'cells' / row['cell_id']
    audit.update(checkpoint=m.ref(path), cell_id=row['cell_id'], rate_rps=row['rate_rps'],
        repeat=row.get('repeat'), runtime_config=m.ref(directory / 'runtime_config.json'),
        original_artifact_hashes_checked=len(cp['artifacts']))
    return cp, receipt, directory, audit


def identities(cp, receipt_dir):
    binding = m.checked(cp['binding'])
    result = []
    for side in ('before', 'after'):
        path = receipt_dir / ('identity.' + side + '.json')
        rows = m.read(path)
        assert len(rows) == len(binding['instances']) == 4
        for actual, expected in zip(rows, binding['instances']):
            container, runtime = actual['container'], actual['runtime']
            assert container['Id'] == expected['container']['id']
            assert container['Image'] == expected['container']['image']
            assert container['State']['StartedAt'] == expected['container']['StartedAt']
            assert container['State']['Running'] and not container['State'].get('OOMKilled')
            assert not runtime.get('error') and not runtime.get('runtime_error')
            assert all(not runtime.get(k) for k in ('running', 'waiting', 'active', 'kv_allocations', 'transfer_allocations'))
        result.append(m.ref(path))
    return result


def timeout_proof(request, events, engine_rows):
    assert request['error'] == 'request_hard_timeout' and m.truth(request['request_timeout'])
    assert not m.truth(request['success']) and request.get('http_status') in ('', '200', None)
    assert request.get('admission_rejection') in ('', '0', None, 'False', 'false')
    key = request['request_id']
    timing = [e for e in events if e['kind'] == 'request_timing' and e['client_request_id'] == key]
    assert len(timing) == 1
    timing = timing[0]
    rid = timing['request_id']
    own = {e['kind']: e for e in events if e['request_id'] == rid}
    assert set(own) == {'admission', 'prefill_complete', 'first_token', 'request_end', 'request_timing'}
    assert not timing['completed'] and not own['request_end']['completed']
    deadline = float(request['request_deadline_s'])
    assert abs(deadline - float(request['planned_arrival_s']) - 120) < 1e-6
    assert timing['hard_deadline_s'] == deadline
    assert abs(own['request_end']['at_s'] - deadline) < .02
    queued = timing['forward_started_s'] - timing['queued_s']
    backend = timing['first_token_s'] - timing['forward_started_s']
    assert queued > 90 and 0 < backend < .2
    assert max(timing['first_instance_ages_s'].values()) < .2
    assert 0 < deadline - float(request['last_token_s']) < .1
    assert float(request['last_token_s']) > float(request['first_token_s'])
    assert int(request['n_text_chunks']) > 100
    clocks = own['admission']['clock_outcomes']
    assert all(x['requested'] == x['commanded'] == 2520 and not x['conservative_fallback'] for x in clocks)
    decode = [e for e in engine_rows if e['role'] == 'decode' and any(rid in x for x in e['request_ids'])]
    assert len(decode) > 100
    assert abs(decode[-1]['finished_s'] - deadline) < .1
    assert all(not e.get('error') for e in decode)
    return dict(client_request_id=key, internal_request_id=rid, queue_to_forward_s=queued,
        forward_to_first_token_s=backend, recorded_ttft_s=float(request['ttft_s']),
        prescribed_output_tokens=int(request['output_len']), observed_text_chunks=int(request['n_text_chunks']),
        native_decode_events=len(decode), last_native_decode_s=decode[-1]['finished_s'],
        original_hard_deadline_s=deadline, timeout_endpoint_delta_s=own['request_end']['at_s'] - deadline,
        allocation_snapshot_age_max_s=max(timing['first_instance_ages_s'].values()),
        original_frequency_mhz=2520, frequency_fallback=False,
        exact_generated_tokens_for_partial_request_known=False)


def engine_proof(index_path):
    evidence = m.read(index_path)
    if isinstance(evidence, dict):
        evidence = evidence['instances']
    all_rows, proof = [], []
    for item in evidence:
        assert m.sha(item['window_path']) == item['window_sha256']
        rows = [json.loads(x) for x in Path(item['window_path']).read_text().splitlines()]
        assert len(rows) == item['records'] and rows
        assert all(not r.get('error') for r in rows)
        all_rows.extend(rows)
        roles = set(r['role'] for r in rows)
        durations = [r['finished_s'] - r['started_s'] for r in rows]
        gaps = [b['started_s'] - a['finished_s'] for a, b in zip(rows, rows[1:])]
        if roles == {'decode'}:
            assert sum(durations) > 190 and max(durations) < .1 and max(gaps) < .01
        proof.append(dict(raw=dict(path=item['window_path'], sha256=item['window_sha256']),
            records=len(rows), roles=sorted(roles), busy_event_duration_s=sum(durations),
            max_event_duration_s=max(durations), max_inter_event_gap_s=max(gaps)))
    assert len(proof) == 4
    return all_rows, proof


def main():
    history = [raw_checkpoint(path) for path in HISTORICAL]
    results = []
    for repeat, path in enumerate(CHECKPOINTS, 1):
        cp, receipt, directory, audit = raw_checkpoint(path)
        row = cp['row']
        assert row['repeat'] == repeat and row['request_hard_timeout_s'] == row['drain_after_arrival_window_s'] == 120
        assert row['trace_duration_s'] == 100 and row['seed'] == 701
        assert (row['slo_ttft_s'], row['slo_tpot_s'], row['slo_scale']) == (1, .1, 1)
        requests = list(csv.DictReader((directory / 'bench.csv').open()))
        failed = [r for r in requests if not m.truth(r['success']) or r['error'] or m.truth(r['request_timeout'])]
        assert len(failed) == audit['failed_requests'] == audit['request_timeouts'] == (2 if repeat == 1 else 1)
        events = [json.loads(x) for x in (directory / 'control.jsonl').read_text().splitlines()]
        counts = collections.Counter(e['kind'] for e in events)
        assert counts == dict.fromkeys(('admission', 'prefill_complete', 'first_token', 'request_end', 'request_timing'), 410)
        assert all(not e.get('error') for e in events)
        assert len({e['request_id'] for e in events}) == 410
        index = HERE / ('runtime-window-index.json' if repeat == 1 else 'repeat2-runtime-window-index.json')
        engine_rows, engines = engine_proof(index)
        proofs = [timeout_proof(r, events, engine_rows) for r in failed]
        cfg = m.read(directory / 'runtime_config.json')
        comparisons = []
        for oldcp, oldreceipt, olddir, oldaudit in history:
            original = m.read(olddir / 'runtime_config.json')
            a, b = copy.deepcopy(cfg), copy.deepcopy(original)
            a.pop('journal'); b.pop('journal')
            assert a == b, 'original fixed policy differs'
            comparisons.append(dict(reference=oldaudit['runtime_config'], identical_except=['journal']))
        audit.update(identity_evidence=identities(cp, Path(cp['receipt']['path']).parent),
            control_events=m.ref(directory / 'control.jsonl'), control_event_counts=dict(counts),
            engine_window_index=m.ref(index), engines=engines, timeout_proofs=proofs,
            historical_runtime_comparison=comparisons, measurement_valid=True, work_complete=False,
            failure_class='independently_diagnosed_capacity_deadline',
            diagnosis_scope='Observed capacity/deadline of the original fixed DistServe policy at this workload; not a hardware upper bound or proof against every latent software defect.',
            no_observed_runtime_transport_restart_oom_or_cleanup_fault=True,
            exact_output_tokens_for_partial_requests_unknown=True, original_checkpoint_preserved=True)
        results.append(audit)
    logs = m.read(HERE / 'service-log-index.json')
    for entry in logs['logs']:
        assert m.sha(entry['log_path']) == entry['sha256'] and entry['potential_error_line_count'] == 0
    report = dict(schema='B32B-distserve-independent-timeout-diagnosis-v1', passed=True, cpu_only=True,
        observations=results, historical_raw_audits=[x[3] for x in history],
        service_log_index=m.ref(HERE / 'service-log-index.json'),
        service_log_limit='No stdout/stderr in R1 measurement window; this absence is supporting evidence only. Native event streams supply positive execution evidence.',
        classification_contract=m.ref(ROOT / 'common/ascending-rate-execution-v2/contract.py'),
        no_original_result_modified=True, no_gpu_action=True, no_automatic_retry=True,
        successor_preparation_cancelled=True,
        successor_reason='Another user-owned task already measured DistServe R2 and owns the remaining B schedule; do not launch a competing successor.',
        future_incomplete_policy='Every future incomplete/unknown measurement needs its own diagnosis; this classification is specific to these checkpoint hashes.',
        second_repeat_status_note='The uniform predecessor status complete=true denotes processed observations; raw work_complete=false and 409/410 must remain visible.',
        audit_sources=[m.ref(Path(__file__)), m.ref(HERE / 'metrics_snapshot.py'), m.ref(ROOT / 'audit_cooperative_arrivals_v1.py')])
    return report


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
