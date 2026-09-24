import json

from pdblend.results.catalog import collect, normalize, write_csv


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(tmp_path, *, attempt_id='attempt-0001-a', audit_status='passed'):
    root = tmp_path/'results'
    queue = root/'2026-09-22/three-model/queue.json'
    attempt = queue.parent/'queue-attempts/pdblend-quick-ab-32b-test'/attempt_id
    job_id = 'pdblend-quick-ab-32b-test'
    save(queue, dict(jobs={job_id: dict(status='passed', payload=dict(system='pdblend', model_id='Qwen2.5-32B-Instruct'))},
                     leases={'lease': dict(job_id=job_id, attempt_dir=str(attempt))}))
    binding = dict(model_id='Qwen2.5-32B-Instruct', tp=2, pp=1, seed=701, duration_s=300,
                   source_sha256='1'*64, trace_sha256='2'*64, image_digest='sha256:'+'3'*64,
                   gpu_uuids=['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d'],
                   inputs={'profile': dict(sha256='4'*64)}, initial_plan={'counts': {'M': 2}})
    save(attempt/'manifest.json', {})
    save(attempt/'preflight.json', dict(bindings=binding))
    save(attempt/'completion.json', dict(status='passed', complete=True, execution_complete=True,
          bindings=binding, functional_passed=True, comparison_mode='parallel_qualified'))
    save(attempt/'functional/completion.json', dict(functional_passed=True, started_s=10., finished_s=15.))
    arm_proofs = {}
    for arm in ('A', 'B'):
        save(attempt/arm/'summary.json', dict(arm=arm, bindings=binding, window_s=300.,
             formal_eligible=False, energy_comparable=False, comparison_mode='parallel_qualified',
             energy_j=1000., slo=dict(offered=30, succeeded=30, output_tokens=3840, joint_output_tokens=3840,
                                      joint_slo_rate=1., ttft_p95=.9)))
        arm_proofs[arm] = dict(metrics=dict(offered=30, succeeded=30, output_tokens=3840,
             joint_output_tokens=3840, success_rate=1., joint_slo_rate=1.,
             throughput_request_s=.1, throughput_token_s=12.8, goodput_request_s=.1, goodput_token_s=12.8,
             ttft_s={'p50': .3, 'p95': .5, 'p99': .8}, tpot_s={'p50': .01, 'p95': .02, 'p99': .03}),
             controller=dict(periodic_decisions=0 if arm == 'A' else 29,
                    periodic_planner_calls=0 if arm == 'A' else 29, actual_actions=0 if arm == 'A' else 2,
                    automatic_pd_requests=0, planning_seconds={'p50': .003, 'p95': .007, 'p99': .009}))
    identity = {key: binding[key] for key in ('model_id', 'tp', 'pp', 'seed', 'source_sha256', 'trace_sha256', 'image_digest')}
    identity['inputs'] = {'profile': '4'*64}
    save(attempt/'audit.json', dict(schema='pdblend.native-ab-audit/v1', identity=identity,
          status=audit_status, classification='无改善', arms=arm_proofs,
          functional=dict(passed=True), requires_serial_retest=False,
          interference=dict(passed=True, comparison_mode='parallel'), errors=[]))
    return root, attempt


def test_native_rows_separate_execution_function_and_arms(tmp_path):
    root, attempt = fixture(tmp_path)
    rows, errors = collect(root)
    assert not errors
    assert len(rows) == 4
    assert {row['record_kind'] for row in rows} == {'attempt_aggregate', 'functional_qualification', 'native_ab_arm'}
    aggregate = next(r for r in rows if r['record_kind'] == 'attempt_aggregate')
    assert aggregate['status'] == 'passed'
    assert aggregate['audit_status'] == 'passed'
    assert aggregate['execution_complete'] is True
    assert aggregate['duration_s'] == '' and aggregate['offered_requests'] == ''
    function = next(r for r in rows if r['record_kind'] == 'functional_qualification')
    assert function['trace_sha256'] == '' and function['duration_s'] == 5.
    assert function['functional_passed'] is True
    arm = next(r for r in rows if r['arm'] == 'B')
    assert arm['model_id'] == 'Qwen2.5-32B-Instruct' and arm['tp'] == 2
    assert arm['instance_count'] == 2 and len(json.loads(arm['tp_layout'])) == 2
    assert arm['tp_mode'] == 'fixed_tp'
    assert arm['ttft_p95_s'] == .5  # Audited value replaces reported .9.
    assert arm['planner_p95_s'] == .007 and arm['control_actions'] == 2
    assert arm['throughput_token_s'] == 12.8
    assert arm['latency_comparable'] is True
    assert arm['baseline_definition'] == 'fixed_initial_vs_periodic_same_source'
    assert arm['efficacy_classification'] == '无改善'
    assert all(r['formal_eligible'] is False for r in rows)
    assert arm['energy_total_j'] == '' and arm['energy_recorded_j'] == 1000.


def test_original_execution_pass_does_not_hide_audit_failure(tmp_path):
    root, attempt = fixture(tmp_path, audit_status='inconclusive')
    value = json.loads((attempt/'audit.json').read_text())
    value.update(classification='inconclusive', requires_serial_retest=True,
                 interference={'passed': False}, errors=['parallel > 5%'])
    save(attempt/'audit.json', value)
    rows, _ = collect(root)
    aggregate = next(r for r in rows if r['record_kind'] == 'attempt_aggregate')
    assert aggregate['status'] == 'passed' and aggregate['audit_status'] == 'inconclusive'
    assert aggregate['requires_serial_retest'] is True
    assert all(r['latency_comparable'] is False for r in rows)
    function = next(r for r in rows if r['record_kind'] == 'functional_qualification')
    assert function['audit_status'] == 'passed' and function['failure_reason'] == ''
    assert function['requires_serial_retest'] == ''


def test_identity_reaudit_preserves_original_receipt_and_rejects_strategy_claim(tmp_path):
    root, attempt = fixture(tmp_path)
    original = (attempt/'audit.json').read_bytes()
    revision = json.loads(original)
    revision.update(status='inconclusive', classification='inconclusive',
                    errors=['executed Plan lost its profile identity'])
    save(attempt/'audit-plan-identity.json', revision)
    rows, errors = collect(root)
    assert not errors
    assert (attempt/'audit.json').read_bytes() == original
    arm = next(row for row in rows if row['arm'] == 'B')
    assert arm['joint_slo_rate'] == 1
    assert arm['audit_status'] == 'inconclusive'
    assert arm['efficacy_classification'] == 'inconclusive'
    assert arm['latency_comparable'] is False
    assert arm['audit_path'].endswith('/audit-plan-identity.json')


def test_parallel_rejected_rows_never_borrow_final_serial_metrics(tmp_path):
    root, attempt = fixture(tmp_path)
    source = json.loads((attempt/'A/summary.json').read_text())
    source['slo']['ttft_p95'] = 4.1
    save(attempt/'parallel-unqualified/A/summary.json', source)
    rows, _ = collect(root)
    row = next(r for r in rows if r['evidence_status'] == 'parallel_unqualified')
    assert row['ttft_p95_s'] == 4.1
    assert row['planner_p95_s'] == ''
    assert row['audit_path'] == ''
    assert row['latency_comparable'] is False and row['requires_serial_retest'] is True
    assert row['efficacy_classification'] == 'inconclusive'


def test_foreign_audit_cannot_supply_metrics_or_efficacy(tmp_path):
    root, attempt = fixture(tmp_path)
    value = json.loads((attempt/'audit.json').read_text())
    value['identity']['trace_sha256'] = '9'*64
    save(attempt/'audit.json', value)
    rows, _ = collect(root)
    arm = next(r for r in rows if r['arm'] == 'B')
    assert arm['ttft_p95_s'] == .9
    assert arm['planner_p95_s'] == '' and arm['efficacy_classification'] == ''
    assert arm['latency_comparable'] is False


def test_live_partial_attempt_has_aggregate_and_no_invented_metrics(tmp_path):
    root, attempt = fixture(tmp_path)
    (attempt/'completion.json').unlink()
    (attempt/'audit.json').unlink()
    rows, _ = collect(root)
    assert len([r for r in rows if r['record_kind'] == 'attempt_aggregate']) == 1
    assert all(r['planner_p95_s'] == '' for r in rows)
    assert all(r['audit_status'] == '' for r in rows)


def test_new_columns_do_not_backfill_historical_rows(tmp_path):
    path = tmp_path/'summary.json'
    save(path, {})
    row = normalize(dict(model_id='old', goodput_token_s=44., planner_p95_s=.01),
                    path=path, root=tmp_path, historical=True)
    assert row['goodput_token_s'] == '' and row['planner_p95_s'] == ''
    assert row['arm'] == '' and row['audit_status'] == ''
    root = tmp_path/'results'
    write_csv(root/'runs.csv', [dict(run_id='old', evidence_status='raw_pruned', model_id='historical')])
    rows, errors = collect(root)
    assert not errors and len(rows) == 1
    assert rows[0]['evidence_status'] == 'raw_pruned'
    assert rows[0]['arm'] == ''


def test_retry_is_a_separate_attempt_even_with_same_trace(tmp_path):
    import shutil
    root, attempt = fixture(tmp_path)
    retry = attempt.with_name('attempt-0002-b')
    shutil.copytree(attempt, retry)
    queue = root/'2026-09-22/three-model/queue.json'
    data = json.loads(queue.read_text())
    data['leases']['retry'] = dict(data['leases']['lease'], attempt_dir=str(retry))
    save(queue, data)
    rows, errors = collect(root)
    assert not errors and len(rows) == 8
    assert len({r['run_id'] for r in rows}) == 8
    assert {r['attempt_id'] for r in rows} == {'attempt-0001-a', 'attempt-0002-b'}
