"""Actual gate evidence counterexamples; this does not assert a future900 pass."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

s = importlib.util.spec_from_file_location('p8_contract_cpu', Path(__file__).with_name('qualification_contract.py'))
c = importlib.util.module_from_spec(s)
s.loader.exec_module(c)


@pytest.fixture(scope='module')
def real_gate():
    old = list(sys.path)
    try:
        sys.path[:0] = [str(c.A), str(c.ROOT)]
        m = c.load(c.A / 'p8_qualification_audit_v3.py', 'p8_contract_actual_gate_auditor')
        audit = m.audit_gate()
        return m, audit
    finally:
        sys.path[:] = old


def test_actual_cold_low_slo_main_work_remains_qualified_gate(real_gate):
    m, audit = real_gate
    assert audit['slo_attainment'] < .9
    assert c.checked_stage_spec(audit, m)['arm'] == 'dynamic'
    assert c.main_work_evidence(audit, m)['main_native_requests_on_added'] > 0


@pytest.mark.parametrize('change', ['file_sha', 'capacity_path', 'capacity_sha', 'profile'])
def test_config_and_entire_source_pin_mismatch_rejected(real_gate, monkeypatch, change):
    m, audit = real_gate
    sp = c.checked(audit['spec'])
    config = c.checked(sp['config'])
    if change == 'file_sha':
        sp['files'][next(iter(sp['files']))] = '0' * 64
    elif change == 'capacity_path':
        config['capacity_binding_path'] += '.wrong'
    elif change == 'capacity_sha':
        config['capacity_binding_sha256'] = '0' * 64
    else:
        config['profiles'] += '.wrong'
    original = c.checked
    monkeypatch.setattr(c, 'checked', lambda r: sp if r == audit['spec'] else config if r == sp['config'] else original(r))
    with pytest.raises(RuntimeError):
        c.checked_stage_spec(audit, m)


@pytest.mark.parametrize('field,value', [('source', {}), ('full_work', False), ('failed_requests', 1), ('request_timeouts', 1)])
def test_stage_identity_and_full_work_never_waived(real_gate, field, value):
    m, actual = real_gate
    audit = copy.deepcopy(actual)
    audit[field] = value
    with pytest.raises(RuntimeError):
        c.checked_stage_spec(audit, m)


@pytest.mark.parametrize('change', ['probe_only', 'out_of_range', 'no_queued_demand', 'wrong_gpu'])
def test_added_correctness_probe_or_unmeasured_growth_is_not_main_work(real_gate, tmp_path, change):
    m, actual = real_gate
    out = Path(actual['output'])
    inv = json.loads((out / 'inventory.json').read_text())
    control = [json.loads(line) for line in (out / 'control.jsonl').open()]
    dispatch = [json.loads(line) for line in (out / 'engine-dispatch.jsonl').open()]
    if change in ('probe_only', 'out_of_range'):
        for event in control:
            if event['kind'] == 'admission':
                event['client_request_id'] = 'oracle-only' if change == 'probe_only' else str(actual['n_expected'])
    elif change == 'no_queued_demand':
        for event in inv['events']:
            if event['kind'] == 'capacity_decision':
                event['demand']['queued_requests'] = 0
    else:
        for key in set(inv['known_instances']) - set(inv['initial_ids']):
            inv['known_instances'][key]['gpus'] = [4]
    (tmp_path / 'inventory.json').write_text(json.dumps(inv))
    for name, rows in [('control.jsonl', control), ('engine-dispatch.jsonl', dispatch)]:
        (tmp_path / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    audit = dict(actual, output=str(tmp_path))
    with pytest.raises((RuntimeError, ValueError)):
        c.main_work_evidence(audit, m)


def test_changed_stored_qualification_cannot_override_real_reaudit(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'build_qualification', lambda: dict(passed=True, full_work=True))
    path = tmp_path / 'fake.json'
    path.write_text(json.dumps(dict(passed=True, full_work=False)))
    with pytest.raises(RuntimeError):
        c.validate_qualification(c.ref(path))


def test_every_actual_transition_retains_signed_prediction_error(real_gate):
    module, audit = real_gate
    errors = c.transition_prediction_errors(audit, module)
    assert len(errors) == audit['physical_commits'] == 2
    assert {e['operation'] for e in errors} == {'restore_cold', 'remove'}
    assert all(e['energy_error_j'] == e['measured_whole_node_energy_j'] - e['planned_energy_j'] for e in errors)
    assert all(e['bounds_are_not_guarantees'] and e['overlapping_energy_not_added_to_service'] for e in errors)

@pytest.mark.parametrize('change', ['pid','container_id','source','future','before_window','kv','transfer_age','ack','budget','accepting','owner','missing_sender'])
def test_saved_terminal_identity_and_native_proof_cannot_be_forged(real_gate, change):
    module, audit = real_gate
    status = c.checked(audit['status']); spec = c.checked(audit['spec'])
    instances = c.checked(spec['original_binding'])['instances']
    restoration = c.checked(status['retained_restoration'])
    rows = copy.deepcopy(restoration['after']); first = rows[0]; native = first['runtime']
    if change == 'pid': first['container']['State']['Pid'] += 1
    elif change == 'container_id': first['container']['Id'] = 'wrong'
    elif change == 'source': first['provenance']['model'] = 'wrong'
    elif change == 'future': native['timestamp'] = status['finished_s'] + 1
    elif change == 'before_window': native['timestamp'] = status['started_s'] - 1
    elif change == 'kv': native['kv_allocations'] = {'other': 1}
    elif change == 'transfer_age': native['transfer_observed_s'] = native['timestamp'] - 1.039
    elif change == 'ack': native['acknowledged_generation'] -= 1
    elif change == 'budget': native['scheduler_budget_effective']['max_num_batched_tokens'] = 2048
    elif change == 'accepting': native['accepting'] = False
    elif change == 'owner': native['id'] = 'another-run'
    elif change == 'missing_sender': native.pop('transfer_send_counters_observed')
    with pytest.raises((RuntimeError,ValueError)):
        module.saved_identity(rows, instances, status['started_s'], status['finished_s'], restored=True)


def test_historical_audit_does_not_read_current_docker_layout(real_gate, monkeypatch):
    module, audit = real_gate
    monkeypatch.setattr(module.prior.subprocess, 'check_output', lambda *a,**k: (_ for _ in ()).throw(AssertionError('live Docker forbidden in historical read-only audit')))
    assert module.audit_gate()['result'] == audit['result']


@pytest.mark.parametrize('key', ['measurement_adapter','measurement_hooks','measurement_driver','measurement_hardware_selftest'])
def test_wrong_actual_measurement_component_rejected(key):
    spec=c.checked(c.ref(c.A/'p8-qualification900-inputs-003/spec.json'))
    spec[key]=dict(spec[key],sha256='0'*64)
    with pytest.raises(RuntimeError):c.measurement_contract(spec)


def test_actual_sampler_contract_cpu_does_not_assert900_pass():
    spec=c.checked(c.ref(c.A/'p8-qualification900-inputs-003/spec.json'))
    value=c.measurement_contract(spec)
    assert value['measurement_adapter']==spec['measurement_adapter']
