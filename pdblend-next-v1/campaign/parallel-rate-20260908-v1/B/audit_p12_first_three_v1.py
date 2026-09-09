"""Saved-only inspection of the three observed B P12 cells; never runs a request."""
from pathlib import Path
import collections
import csv
import hashlib
import importlib.util
import json
import math
import time

R = Path(__file__).resolve().parent.parent
D = R / 'B/distributed-14b-v1'
P = D / 'pdb-p12-performance-001'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def ref(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


q = module(R / 'common/distributed14b-qualification-v3/verify.py', 'p12_saved_q')
nv = module(R / 'common/distributed14b-profile-validation-v1/validate.py', 'p12_saved_native')
common = module(R / 'common/execution-until-complete-v1/run.py', 'p12_saved_common')
host = ref(R / 'hosts/14b-capacity-p12/manifest.json')
power_functions = q.original_power_functions(host)
records = []
for cp_path in sorted((P / 'results/checkpoints').glob('*.json')):
    cp = read(cp_path)
    for path, digest in cp['artifacts'].items():
        assert q.sha(path) == digest, path
    receipt = q.checked(cp['receipt'])
    binding = q.checked(cp['binding'])
    summary = receipt['summary']
    operation = Path(cp['receipt']['path']).parent
    cell = P / 'results/cells' / cp['row']['cell_id']
    requests = list(csv.DictReader((cell / 'bench.csv').open()))
    controls = lines(cell / 'control.jsonl')
    clocks = lines(cell / 'control.clock-guard.jsonl')
    bad = [row for row in requests if row['success'] != '1']
    assert len(requests) == summary['n_expected']
    assert len(bad) == summary['failed_requests']
    assert summary['runtime_error'] is None
    assert not any(row.get('http_status') == '503' for row in requests)
    assert receipt['measurement_valid'] and receipt['clock_restore_complete']
    assert receipt['child_stopped'] and receipt['child_exitcode'] == 0
    assert receipt['outer_cleanup_errors'] == []
    outer_power, full_clocks = q.power_operation(operation, receipt, host)
    with (cell / 'power.csv').open() as stream:
        powers = [(float(row['t_s']), [float(row[f'gpu{g}_w']) for g in range(8)]) for row in csv.DictReader(stream)]
    energy = power_functions['trapezoid_energy'](power_functions['clip_power_window'](
        powers, summary['measurement_start_s'], summary['measurement_end_s'], pad_s=0))
    assert math.isclose(energy, summary['energy_j'], abs_tol=1e-5)
    reacquires = [e for e in clocks if e['kind'] == 'idle_domain_reacquisition']
    assert len(reacquires) == 1
    actual = reacquires[0]
    assert actual['confirmed'] is True and actual['physical_state_unknown'] is False
    assert actual['idle_domain_reacquire_timeout_s'] == 1.5
    assert actual['original_active_settle_timeout_s'] == .3
    assert actual['deadline_extended'] is False and actual['request_deadline_unchanged'] is True
    final_reads = actual['confirmation_observations'][-len(actual['gpus']):]
    assert {v['gpu'] for v in final_reads} == set(actual['gpus'])
    assert all(abs(v['observed_mhz'] - 2100) <= 15 for v in final_reads)
    assert all(v['finished_s'] <= actual['physical_confirmation_deadline_s'] for v in final_reads)
    assert not any(e.get('physical_state_unknown') is True for e in clocks)
    assert not any(e.get('error') and not e.get('safely_replannable') for e in clocks)
    for instance in binding['instances']:
        clean = receipt['restoration'][instance['id']]
        assert clean['complete'] is True and clean['errors'] == []
        common.barrier(clean['before'], clean['proof'], instance)
        resumed = clean['resumed']
        nv.native_saved(resumed['after'], instance['id'], 8192)
        assert resumed['control'] == dict(generation=resumed['before']['generation'] + 1,
            role='mixed', mode='continuous', admit_prefill=True, admit_decode=True,
            scheduler_budget=dict(schema_version=1, max_num_batched_tokens=8192, max_num_seqs=32))
        assert resumed['after']['generation'] == resumed['control']['generation']
    failures = []
    for row in bad:
        assert row['error'] == 'request_hard_timeout' and row['request_timeout'] == 'True'
        assert row['generated_tokens'] == '0' and row['token_ids_verified'] == '0'
        assert row['token_count_source'] == 'missing'
        timing = [e for e in controls if e['kind'] == 'request_timing' and e['client_request_id'] == row['request_id']]
        assert len(timing) == 1
        timing = timing[0]
        admission = [e for e in controls if e['kind'] == 'admission' and e['request_id'] == timing['request_id']]
        assert len(admission) == 1
        admission = admission[0]
        assert 'completion recovery' in admission['plan']['reason']
        assert timing['completed'] is False
        arrival = float(row['planned_arrival_s'])
        deadline = float(row['request_deadline_s'])
        assert math.isclose(deadline - arrival, 120, abs_tol=1e-5)
        assert timing['hard_deadline_s'] == deadline
        assert 0 <= float(row['actual_dispatch_s']) - arrival < .01
        assert 0 <= timing['handler_arrival_s'] - arrival < .01
        assert timing['queued_s'] < timing['forward_started_s'] < timing['first_token_s'] < deadline
        assert 0 <= float(row['finish_s']) - deadline < .1
        assert 0 <= timing['cleanup_end_s'] - deadline < .1
        queue_start, queue_end = timing['queued_s'], timing['forward_started_s']
        other_tokens = [e for e in controls if e['kind'] == 'first_token' and e['request_id'] != timing['request_id'] and queue_start <= e['at_s'] <= queue_end]
        other_ends = [e for e in controls if e['kind'] == 'request_end' and e['request_id'] != timing['request_id'] and e.get('completed') is True and queue_start <= e['at_s'] <= queue_end]
        assert len(other_tokens) > 50 and len(other_ends) > 50
        failures.append(dict(client_request_id=row['request_id'], request_id=timing['request_id'],
            requested_input_tokens=int(row['prompt_len']), requested_output_tokens=int(row['output_len']),
            verified_generated_tokens=None, partial_text_chunks=int(row['n_text_chunks']),
            partial_chunks_are_not_verified_output_tokens=True,
            actual_dispatch_lateness_s=float(row['actual_dispatch_s']) - arrival,
            handler_lateness_s=timing['handler_arrival_s'] - arrival,
            queued_to_forward_s=queue_end - queue_start,
            forward_to_first_token_s=timing['first_token_s'] - queue_end,
            other_first_tokens_while_waiting=len(other_tokens), other_complete_requests_while_waiting=len(other_ends),
            original_hard_timeout_s=120, preserved_request_timing=timing, preserved_admission=admission))
    if failures:
        assert len(failures) == 1 and summary['request_timeouts'] == 1
        assert summary['completed_work_requests'] == 163 and summary['n_expected'] == 164
    else:
        assert summary['work_complete'] is True and summary['request_timeouts'] == 0
    records.append(dict(checkpoint=ref(cp_path), receipt=cp['receipt'], binding=cp['binding'],
        release=cp['release'], source_manifest=host, rate_rps=cp['row']['rate_rps'],
        complete_work=summary['work_complete'], completed_work_requests=summary['completed_work_requests'],
        offered_requests=summary['n_expected'], failed_requests=summary['failed_requests'],
        request_timeouts=summary['request_timeouts'], http503=0, runtime_error=None,
        slo_attainment=summary['slo_attainment'], raw_energy_j=energy, full_operation_power=outer_power,
        energy_boundaries_overlap_do_not_add=True, raw_energy_measurement_valid=True,
        native_cleanup_complete=True, clock_restore_complete=True,
        actual_reacquire_confirmation_s=max(x['finished_s'] for x in final_reads) - actual['last_command_finished_s'],
        idle_reacquisition=actual, request_failures=failures, artifacts=cp['artifacts'],
        complete_slo_boundary_eligible=not bool(failures),
        classification='original_deadline_exceeded_after_capacity_planner_queue_wait' if failures else 'complete_work_with_confirmed_idle_recovery',
        limits=['Incomplete work does not establish the first COMPLETE SLO < 90% boundary.',
                'Client partial text chunks do not establish verified output token count.',
                'No source or request deadline was changed by this saved-data audit.']))
assert len(records) == 3
result = dict(schema='B-P12-first-three-saved-audit-v1', independently_recomputed=True, node='B',
    created_s=time.time(), hostname=binding['hostname'], records=records, auditor=ref(Path(__file__)),
    queue_status=ref(P / 'status.json'), new_gpu_observations=0)
target = D / 'p12-first-three-independent-audit-001.json'
with target.open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(ref(target)))
