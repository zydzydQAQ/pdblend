"""CPU-only rejection tests; synthetic restarts never issue a qualification."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fresh_eco_qualification_tested', HERE / 'qualify_restored.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def original():
    return q.pinned(dict(path=next(iter(q.ORIGINALS)), sha256=next(iter(q.ORIGINALS.values()))))


def fixture(tmp_path):
    old = original()
    fresh = copy.deepcopy(old)
    inventory = q.read(old['identity_file'])
    for instance, row in zip(fresh['instances'], inventory):
        instance['container']['StartedAt'] = '2026-09-08T07:00:00.000000000Z'
        row['State']['StartedAt'] = instance['container']['StartedAt']
        row['State']['Pid'] += 100
        instance['host_pid'] = row['State']['Pid']
    replies = [dict(instance_id=i['id'], prompt_length=n, request_id=i['id'] + str(n),
                    response=dict(token_ids=list(range(64)), usage=dict(prompt_tokens=n, completion_tokens=64)))
               for n in (128, 7168) for i in fresh['instances']]
    status = dict(complete=True, clock_restore_complete=True, errors=[],
        power_evidence=dict(power_source_verified=True), setup_and_correctness_energy_j=1000,
        started_s=100, measurement_start_s=101, measurement_end_s=110, finished_s=111,
        correctness=dict(passed=True, owned=[], replies=replies,
            restoration=[dict(complete=True) for _ in range(4)], started_s=102, finished_s=109))
    directory = tmp_path / 'restore'
    directory.mkdir()
    (directory / 'power').mkdir()
    (directory / 'power/power.csv').write_text('t_s,' + ','.join(f'gpu{i}_w' for i in range(8)) + '\n'
        + ''.join(str(t) + ',' + ','.join([str(1000 / 9 / 8)] * 8) + '\n' for t in (100, 105, 115)))
    for name, value in [('binding.json', fresh), ('containers.after.json', inventory), ('status.json', status)]:
        (directory / name).write_text(json.dumps(value))
    bootstrap = copy.deepcopy(fresh)
    for instance in bootstrap['instances']:
        instance.pop('host_pid')
    bootstrap['restored_priority'] = dict(binding=q.ref(directory / 'binding.json'),
        status=q.ref(directory / 'status.json'), inventory=q.ref(directory / 'containers.after.json'))
    _, _, _, _, power, _ = q.sources()
    binder = q.load('fresh_test_binder', q.BINDER / 'bind.py')
    # This stub isolates restoration contract failures. It cannot reach qualify().
    fake_power = SimpleNamespace(audit_raw=lambda directory, files: dict(power_source_verified=True))
    return old, fresh, bootstrap, inventory, status, directory, binder, fake_power


def check(f):
    old, fresh, bootstrap, inventory, status, directory, binder, power = f
    for name, value in [('binding.json', fresh), ('containers.after.json', inventory), ('status.json', status)]:
        (directory / name).write_text(json.dumps(value))
    bootstrap['restored_priority'] = dict(binding=q.ref(directory / 'binding.json'),
        status=q.ref(directory / 'status.json'), inventory=q.ref(directory / 'containers.after.json'))
    return q.restore_contract(bootstrap, old, q.ref(directory / 'containers.after.json'), binder, power)


def test_frozen_sources_and_original_policy():
    q.sources()
    old = original()
    fresh = copy.deepcopy(old['instances'])
    for instance in fresh:
        instance['container']['StartedAt'] = 'new observed start'
        instance['provenance']['pid'] += 1
        instance['host_pid'] = 123
    q.same_policy(old['instances'], fresh)


@pytest.mark.parametrize('change', ['route', 'source', 'model', 'role', 'container', 'gpu'])
def test_only_process_identity_may_change(change):
    old = original()['instances']
    fresh = copy.deepcopy(old)
    if change == 'route': fresh[0]['url'] += '/other'
    if change == 'source': fresh[0]['provenance']['source_files_at_import']['/fake'] = '0' * 64
    if change == 'model': fresh[0]['provenance']['dtype'] = 'float16'
    if change == 'role': fresh[0]['role'] = 'prefill'
    if change == 'container': fresh[0]['container']['id'] = 'another-container'
    if change == 'gpu': fresh[0]['gpus'] = [6, 7]
    with pytest.raises(RuntimeError, match='policy changed'):
        q.same_policy(old, fresh)


def test_restoration_identity_contract_accepts_recorded_restart(tmp_path):
    status, inventory, files = check(fixture(tmp_path))
    assert status['complete'] and len(inventory) == 4 and len(files) == 4


@pytest.mark.parametrize('change', ['host_pid', 'missing_restart', 'missing_reply', 'duplicate_request',
    'output_mismatch', 'truncated_output', 'cleanup', 'clock', 'sampling', 'time', 'energy', 'energy_mismatch', 'docker_config'])
def test_bad_restoration_is_not_eligible(tmp_path, change):
    f = fixture(tmp_path)
    old, raw, bootstrap, inventory, status, *_ = f
    if change == 'host_pid': raw['instances'][0]['host_pid'] += 1
    if change == 'missing_restart':
        stamp = old['instances'][0]['container']['StartedAt']
        raw['instances'][0]['container']['StartedAt'] = stamp
        bootstrap['instances'][0]['container']['StartedAt'] = stamp
        inventory[0]['State']['StartedAt'] = stamp
    if change == 'missing_reply': status['correctness']['replies'].pop()
    if change == 'duplicate_request': status['correctness']['replies'][1]['request_id'] = status['correctness']['replies'][0]['request_id']
    if change == 'output_mismatch': status['correctness']['replies'][0]['response']['token_ids'][0] = 999
    if change == 'truncated_output': status['correctness']['replies'][0]['response']['token_ids'].pop()
    if change == 'cleanup': status['correctness']['restoration'][0]['complete'] = False
    if change == 'clock': status['clock_restore_complete'] = False
    if change == 'sampling': status['sampling_error'] = 'lost sensor'
    if change == 'time': status['correctness']['finished_s'] = 200
    if change == 'energy': status['setup_and_correctness_energy_j'] = float('nan')
    if change == 'energy_mismatch': status['setup_and_correctness_energy_j'] += 500
    if change == 'docker_config': inventory[0]['Config']['Env'].append('UNDECLARED_OPTION=true')
    with pytest.raises(RuntimeError):
        check(f)


def test_sha_drift_rejected_before_read(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text('{}')
    reference = q.ref(path)
    path.write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match='SHA differs'):
        q.pinned(reference)


def test_existing_output_rejected_before_any_qualification(tmp_path):
    with pytest.raises(RuntimeError, match='no overwrite'):
        q.qualify({}, '', {}, {}, tmp_path)


def test_actual_frozen_27_gate_keeps_old_failure():
    binder, qualifier, shape, oracle, power, _ = q.sources()
    old = original()
    bootstrap = q.pinned(old['qualified_bootstrap'])
    registered = q.pinned(old['oracle'])
    observed = shape.inspect_fresh_gate(Path(old['correctness_evidence']), bootstrap,
        registered['reference_tokens_by_label'], power.load())
    assert observed['original_gate']['verified'] == dict(ordinary=True, pd=True, temporal=False)
    assert observed['temporal']['temporal_native_trajectory_exact'] is True
    assert observed['temporal']['legacy_single_vs_pair_exact'] is False
    assert observed['temporal']['full_temporal_outputs'] == 256
    assert observed['eligible_systems']['ecoserve'] is False


def test_historical_native_oracle_revalidation_is_unchanged():
    _, _, _, oracle, _, _ = q.sources()
    registered = q.pinned(original()['oracle'])
    assert oracle.verify(registered['inputs']) == registered
