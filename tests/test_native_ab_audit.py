import copy
import hashlib
import json

import pytest

from pdblend.bench.native_ab_audit import audit, classify

PLAN_IDENTITY = dict(tp=1, pp=1, pool_id='', generation=0, profile_key='independent-profile')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value)+'\n' for value in values))


def native(tp=1):
    return dict(generation=0, tp=tp, pp=1, native_evidence_complete=True,
                transport_healthy=True, native_at_s=500., acknowledged=True, drained=True,
                ranks=[dict(rank=i, generation=0, native_evidence_complete=True, healthy=True,
                            at_s=500., pending_transfers=0, transfer_allocations={}) for i in range(tp)],
                all_queue=[], running=[], waiting=[], retained_kv_requests=[], pending_transfers=0,
                transfer_allocations={}, kv_allocations={}, total_blocks=100, free_blocks=100,
                reserved_blocks=0)


@pytest.fixture
def campaign(tmp_path):
    trace = [dict(idx=i, prompt=[1]*length, arrival_s=i*20., max_tokens=128)
             for i, length in enumerate((512, 1024, 2048))]
    write(tmp_path/'trace.json', trace)
    digest = hashlib.sha256((tmp_path/'trace.json').read_bytes()).hexdigest()
    binding = dict(model_id='Qwen2.5-7B-Instruct', tp=1, pp=1, seed=701, duration_s=300.,
                   source_sha256='1'*64, image_digest='sha256:'+'2'*64, trace_sha256=digest,
                   inputs={name: dict(path=name+'.json', sha256=digest if name == 'trace' else '3'*64)
                           for name in ('profile', 'trace', 'tuning_trace', 'model_verification')},
                   gpu_uuids=['GPU-a', 'GPU-b'], cohort_id='cohort', initial_plan=PLAN_IDENTITY)
    write(tmp_path/'preflight.json', dict(bindings=binding))
    function = dict(functional_passed=True, checks={k: True for k in
        ('native_cancel_and_kv_cleanup', 'reservation_released', 'admission_reopened', 'successful_reuse')},
        cancel_receipts={'i0': dict(request_id='cancel-test', acknowledged=True, cancelled=True,
                                   instance_id='i0', generation=0, native_state=native())},
        reuse=dict(completion_tokens=128, token_times_s=[20.+i*.01 for i in range(128)]))
    write(tmp_path/'functional'/'completion.json', function)
    write(tmp_path/'qualification'/'comparison.json', dict(comparison_mode='parallel'))
    for mode in ('solo', 'parallel'):
        write(tmp_path/'qualification'/f'{mode}.json',
              dict(latency_s=[1., 1., 1.], mean_power_w=200., gpu_uuids=binding['gpu_uuids']))
    for arm in ('A', 'B'):
        write(tmp_path/arm/'summary.json', dict(arm=arm, bindings=binding, shared_lifecycle_pids=[10, 11],
              formal_eligible=False, energy_comparable=False, window_s=300.))
        outcomes = [dict(idx=i, arrival_s=row['arrival_s'], input_tokens=len(row['prompt']),
                        max_tokens=128, sampling_seed=701, completion_tokens=128, path='M',
                        submitted_s=100.+i*20, first_token_s=101.+i*20,
                        finished_s=111.+i*20, error=None) for i, row in enumerate(trace)]
        lines(tmp_path/arm/'outcomes.jsonl', outcomes)
        lines(tmp_path/arm/'routes.jsonl', [dict(request_id=f'r{i}', submitted_s=row['submitted_s']+.001,
            path='M', terminal_state='completed', prefill_instance='i0', decode_instance='i0',
            tp=1, pp=1, generation=0) for i, row in enumerate(outcomes)])
        write(tmp_path/arm/'native-cleanup.json', dict(i0=native(), i1=native()))
        logs = [dict(kind='plan', t=90., counts={'M': 2}, plan_identity=PLAN_IDENTITY)]
        planning = []
        if arm == 'B':
            logs += [dict(kind='forecast', t=t, decision_reason='planner_change') for t in (110., 120.)]
            logs += [dict(kind='transition_phase', t=121., started_s=120.1, finished_s=121.,
                          status='passed', operation='clock_set')]
            planning = [dict(origin='periodic_controller', t=t, duration_s=.004) for t in (110., 120.)]
        lines(tmp_path/arm/'controller.jsonl', logs)
        lines(tmp_path/arm/'planning-times.jsonl', planning)
        write(tmp_path/arm/'metering.json', dict(error=None))
    return tmp_path


def mutate(path, edit, jsonl=False):
    value = [json.loads(row) for row in path.read_text().splitlines()] if jsonl else json.loads(path.read_text())
    edit(value)
    (lines if jsonl else write)(path, value)


def test_full_native_audit_recomputes_metrics(campaign):
    result = audit(campaign)
    assert result['status'] == 'passed', result
    assert result['classification'] == '无改善'
    assert result['arms']['B']['metrics']['joint_slo_rate'] == 1
    assert result['arms']['B']['metrics']['throughput_token_s'] == 384/300
    assert result['arms']['B']['controller']['planning_seconds']['p95'] == .004
    assert not result['formal_eligible'] and not result['energy_comparable']


@pytest.mark.parametrize('change', ['identity', 'seed', 'token_count', 'missing_request', 'generation',
                                  'kv', 'rank', 'cancel', 'planner', 'lifecycle', 'trace'])
def test_missing_or_conflicting_evidence_fails_closed(campaign, change):
    if change == 'identity':
        mutate(campaign/'B/summary.json', lambda d: d['bindings'].update(source_sha256='9'*64))
    elif change == 'seed':
        mutate(campaign/'B/outcomes.jsonl', lambda d: d[0].update(sampling_seed=1701), True)
    elif change == 'token_count':
        mutate(campaign/'B/outcomes.jsonl', lambda d: d[0].update(completion_tokens=127), True)
    elif change == 'missing_request':
        mutate(campaign/'B/outcomes.jsonl', lambda d: d.pop(), True)
    elif change == 'generation':
        mutate(campaign/'B/routes.jsonl', lambda d: d[0].update(generation=1), True)
    elif change == 'kv':
        mutate(campaign/'B/native-cleanup.json', lambda d: d['i0'].update(kv_allocations={'r0': [1]}))
    elif change == 'rank':
        mutate(campaign/'B/native-cleanup.json', lambda d: d['i0'].update(ranks=[]))
    elif change == 'cancel':
        mutate(campaign/'functional/completion.json', lambda d: d['cancel_receipts']['i0'].update(acknowledged=False))
    elif change == 'planner':
        lines(campaign/'B/planning-times.jsonl', [])
    elif change == 'lifecycle':
        mutate(campaign/'B/summary.json', lambda d: d.update(shared_lifecycle_pids=[99]))
    elif change == 'trace':
        mutate(campaign/'trace.json', lambda d: d[0].update(max_tokens=129))
    result = audit(campaign)
    assert result['status'] == 'inconclusive', result
    assert result['errors']


def test_interference_only_marks_affected_model(campaign):
    mutate(campaign/'qualification/parallel.json', lambda d: d.update(mean_power_w=211.))
    result = audit(campaign)
    assert result['status'] == 'inconclusive'
    assert result['requires_serial_retest']
    assert result['affected_model'] == 'Qwen2.5-7B-Instruct'
    assert result['arms']['A']['metrics']['slo_passed']


def test_serial_fallback_requires_actual_complete_windows(campaign):
    mutate(campaign/'qualification/comparison.json', lambda d: d.update(comparison_mode='serial_after_interference'))
    assert audit(campaign)['status'] == 'inconclusive'
    write(campaign/'qualification/serial-execution.json', dict(exclusive_member=True,
          model_id='Qwen2.5-7B-Instruct', cohort_id='cohort', intervals={
              'A': dict(start_s=100., end_s=400.), 'B': dict(start_s=500., end_s=800.)}))
    result = audit(campaign)
    assert result['status'] == 'passed', result
    assert not result['requires_serial_retest']


def test_no_automatic_action_is_not_an_improvement(campaign):
    mutate(campaign/'B/controller.jsonl', lambda d: d.pop(), True)
    result = audit(campaign)
    assert result['status'] == 'passed'
    assert result['classification'] == '未触发'


def test_p99_gate_detects_tail_regression(campaign):
    mutate(campaign/'B/outcomes.jsonl', lambda d: d[0].update(first_token_s=106., finished_s=116.), True)
    result = audit(campaign)
    assert result['status'] == 'failed', result
    assert result['classification'] == '退化'
    assert result['arms']['B']['metrics']['ttft_s']['p99'] == 6.


def test_eight_gpu_cannot_pass_without_pd_golden_and_routes(campaign):
    mutate(campaign/'preflight.json', lambda d: d['bindings'].update(gpu_uuids=[f'GPU-{i}' for i in range(8)]))
    for arm in ('A', 'B'):
        write(campaign/arm/'native-cleanup.json', {f'i{i}': native() for i in range(8)})
    result = audit(campaign)
    assert result['status'] == 'inconclusive'
    assert result['errors']


def test_descriptive_classification_threshold():
    a = dict(slo_passed=True, goodput_token_s=10., ttft_s={'p95': 1.}, tpot_s={'p95': .1})
    b = copy.deepcopy(a)
    b['ttft_s']['p95'] = .8
    assert classify(a, b, triggered=True) == '改善'
    b['tpot_s']['p95'] = .12
    assert classify(a, b, triggered=True) == '退化'
    assert classify(a, a, triggered=False) == '未触发'


def test_actual_runner_probe_and_binding_shape(campaign):
    from pdblend.bench.native_ab_smoke import compare_probes
    mutate(campaign/'preflight.json', lambda d: d['bindings'].update(gpu_uuids='GPU-a,GPU-b'))
    probes = []
    for mode in ('solo', 'parallel'):
        probe = dict(latency_s=[1., 1., 1., 1.], median_latency_s=1., mean_power_w=200.,
                     gpus=[0, 1], started_s=90., ended_s=110., power_error=None,
                     measurement_start_s=92., measurement_end_s=108., power_samples=[[92., [100., 100.]],
                     [108., [100., 100.]]])
        write(campaign/'qualification'/f'{mode}.json', probe)
        probes.append(probe)
    comparison = compare_probes(*probes)
    comparison['members_unchanged'] = True
    write(campaign/'qualification/comparison.json', comparison)
    for arm in ('A', 'B'):
        mutate(campaign/arm/'summary.json', lambda d: d.update(comparison_mode='parallel_qualified',
               cohort_members_unchanged=True, resident_processes_unchanged=True))
    assert audit(campaign)['status'] == 'passed'


def eight_gpu_pd_fixture(campaign):
    mutate(campaign/'preflight.json', lambda d: d['bindings'].update(gpu_uuids=[f'GPU-{i}' for i in range(8)]))
    for mode in ('solo', 'parallel'):
        mutate(campaign/'qualification'/f'{mode}.json', lambda d: d.update(gpu_uuids=[f'GPU-{i}' for i in range(8)]))
    for arm in ('A', 'B'):
        write(campaign/arm/'native-cleanup.json', {f'i{i}': native() for i in range(8)})
    mutate(campaign/'B/outcomes.jsonl', lambda d: d[0].update(path='PD'), True)
    mutate(campaign/'B/routes.jsonl', lambda d: d[0].update(path='PD', decode_instance='i1'), True)
    mutate(campaign/'B/controller.jsonl', lambda d: d.append(dict(kind='plan', t=121.,
           counts={'P': 1, 'D': 1, 'M': 6}, query_results={}, plan_identity=PLAN_IDENTITY)), True)
    rows = []
    for length in (512, 2048, 7168):
        output = dict(token_ids=[10, 11], completion_tokens=2, prompt_tokens=length,
                      stream_done=True, usage_received=True, error=None)
        rows.append(dict(input_tokens=length, reference=[output, output], client_pd=output,
                    proxy_PD=dict(completion=output, record=dict(path='PD', prefill_instance='i0', decode_instance='i1'))))
    write(campaign/'pd-golden.json', dict(status='passed', complete=True, rows=rows))


def test_eight_gpu_requires_exact_pd_output_and_automatic_plan(campaign):
    eight_gpu_pd_fixture(campaign)
    assert audit(campaign)['status'] == 'passed'
    mutate(campaign/'pd-golden.json', lambda d: d['rows'][0]['client_pd'].update(token_ids=[10, 12]))
    result = audit(campaign)
    assert result['status'] == 'inconclusive'
    assert 'golden' in result['errors'][0]


def test_manual_pd_golden_never_replaces_automatic_routing(campaign):
    eight_gpu_pd_fixture(campaign)
    mutate(campaign/'B/controller.jsonl', lambda d: d[-1].update(query_results={'mechanism_forced_roles': True}), True)
    result = audit(campaign)
    assert result['status'] == 'inconclusive'
    assert 'automatic' in result['errors'][0]


@pytest.mark.parametrize('key', ['tp', 'profile_key', 'generation'])
def test_plan_identity_loss_retains_measurements_but_rejects_strategy_comparison(campaign, key):
    mutate(campaign/'B/controller.jsonl', lambda d: d[0]['plan_identity'].pop(key), True)
    result = audit(campaign)
    assert result['status'] == 'inconclusive'
    assert result['arms']['A']['metrics']['joint_slo_rate'] == 1
    assert result['arms']['B']['metrics']['joint_slo_rate'] == 1
    assert result['classification'] == 'inconclusive'
    assert 'Plan identity' in result['errors'][0]
