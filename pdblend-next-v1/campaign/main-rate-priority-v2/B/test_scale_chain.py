"""CPU contracts for append-only suffix attribution and separate stage handoff."""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


h = module('tested_priority_scale_chain', ROOT / 'scale_chain.py')
tail = module('tested_priority_tail', ROOT / 'continue.py')
old_contract = h.contract()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return h.ref(path)


def pair(tmp_path):
    old = h.read(h.CAMPAIGN / 'B32B-main-to-scale-handoff-v2/attempt-001/scale-bindings/ecoserve/binding.json')
    new = copy.deepcopy(old)
    before = h.read(old['identity_file'])
    after = copy.deepcopy(before)
    for instance, row in zip(new['instances'], after):
        instance['container']['StartedAt'] = '2026-09-08T07:00:00.000000000Z'
        row['State']['StartedAt'] = instance['container']['StartedAt']
        row['State']['Pid'] += 100
        instance['host_pid'] = row['State']['Pid']
    new['output'] = str(tmp_path / 'fresh-results')
    for name, binding, rows in [('old', old, before), ('new', new, after)]:
        identity = tmp_path / name / 'identity.json'
        identity_ref = save(identity, rows)
        binding['files'][str(identity)] = identity_ref['sha256']
        binding['identity_file'] = str(identity)
    return old, new, before, after


def bindings(tmp_path, values):
    old, new, before, after = values
    for name, value, rows in [('old', old, before), ('new', new, after)]:
        identity_ref = save(tmp_path / name / 'identity.json', rows)
        value['files'][identity_ref['path']] = identity_ref['sha256']
    return save(tmp_path / 'old/binding.json', old), save(tmp_path / 'new/binding.json', new)


def policy_contract():
    def policy(old, new, datasets):
        for dataset in datasets:
            old_contract.policy_equal(h.read(old['configs'][dataset]), h.read(new['configs'][dataset]))
    return SimpleNamespace(source_policy=policy, identity_index=old_contract.identity_index,
                           core_instance=old_contract.core_instance)


def test_new_output_and_full_original_policy_with_fresh_host_identity(tmp_path):
    values = pair(tmp_path)
    references = bindings(tmp_path, values)
    old, new = h.compatibility(policy_contract(), *references)
    assert old['output'] != new['output']
    assert old['instances'][0]['container']['StartedAt'] != new['instances'][0]['container']['StartedAt']


@pytest.mark.parametrize('bad', ['same_output', 'same_pid', 'same_start', 'gpu', 'route', 'source', 'model'])
def test_restart_compatibility_fails_closed(tmp_path, bad):
    values = pair(tmp_path)
    old, new, before, after = values
    if bad == 'same_output': new['output'] = old['output']
    if bad == 'same_pid': after[0]['State']['Pid'] = before[0]['State']['Pid']
    if bad == 'same_start':
        new['instances'][0]['container']['StartedAt'] = old['instances'][0]['container']['StartedAt']
        after[0]['State']['StartedAt'] = before[0]['State']['StartedAt']
    if bad == 'gpu': new['instances'][0]['gpus'] = [6, 7]
    if bad == 'route': new['instances'][0]['port'] += 1
    if bad == 'source': new['instances'][0]['provenance']['source_files_at_import']['/unknown'] = '0' * 64
    if bad == 'model': new['model'] = '14b'
    with pytest.raises(RuntimeError):
        h.compatibility(policy_contract(), *bindings(tmp_path, values))


def prefix_fixture(tmp_path, monkeypatch):
    stopped = dict(pid=999999999, phase='stopped', complete=False, finished_s=10,
                   steps=[dict(complete=True, exitcode=0, verified_new_checkpoint=True)])
    scale_ref = save(tmp_path / 'old-scale.json', stopped)
    handoff_ref = save(tmp_path / 'old-handoff.json', stopped)
    groups = [dict(group=dict(system=system, scale_binding=dict(path='/historical/' + system, sha256='a' * 64)),
                   reused=[dict(cell_id=system + '-old')], pending=([dict(cell_id='eco-tail')] if system == 'ecoserve' else []))
              for system in ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve')]
    prefix = dict(schema='B-clean-stopped-scale-prefix-v2', complete_prefix=True, all_children_clean=True,
        whole_baseline192_complete=False, scale_status=scale_ref, handoff_status=handoff_ref,
        boundary_intent=save(tmp_path / 'intent.json', {}), original_spec=save(tmp_path / 'spec.json', {}),
        original_release=save(tmp_path / 'release.json', {}),
        groups=[dict(system=g['group']['system'], binding=g['group']['scale_binding'],
                     completed=g['reused'], pending_ids=[r['cell_id'] for r in g['pending']]) for g in groups])
    monkeypatch.setattr(h, 'contract', lambda: SimpleNamespace(check_spec=lambda *a: dict(groups=groups)))
    return prefix, groups


def test_clean_stopped_prefix_preserves_original_status_bytes(tmp_path, monkeypatch):
    prefix, groups = prefix_fixture(tmp_path, monkeypatch)
    reference = save(tmp_path / 'prefix.json', prefix)
    before = {prefix[key]['path']: Path(prefix[key]['path']).read_bytes() for key in ('scale_status', 'handoff_status')}
    _, actual, _, checked = h.check_prefix(reference)
    assert actual['whole_baseline192_complete'] is False and checked['groups'] == groups
    assert all(Path(path).read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize('bad', ['parent_live', 'parent_complete', 'failed_child', 'changed_prefix', 'other_pending'])
def test_prefix_cannot_relabel_or_extend_history(tmp_path, monkeypatch, bad):
    prefix, groups = prefix_fixture(tmp_path, monkeypatch)
    if bad in ('parent_live', 'parent_complete', 'failed_child'):
        state = h.fixed(prefix['scale_status'])
        if bad == 'parent_live': state['pid'] = os.getpid()
        if bad == 'parent_complete': state['complete'] = True
        if bad == 'failed_child': state['steps'][0]['exitcode'] = 1
        prefix['scale_status'] = save(tmp_path / 'old-scale.json', state)
    if bad == 'changed_prefix': prefix['groups'][0]['completed'] = []
    if bad == 'other_pending':
        groups[0]['pending'] = [dict(cell_id='unauthorized-pdb-tail')]
        prefix['groups'][0]['pending_ids'] = ['unauthorized-pdb-tail']
    with pytest.raises(RuntimeError):
        h.check_prefix(save(tmp_path / 'prefix.json', prefix))


def history_fixture(tmp_path, monkeypatch):
    output = tmp_path / 'new-results'
    fresh_ref = dict(path=str(tmp_path / 'fresh-binding.json'), sha256='b' * 64)
    group = dict(group=dict(system='ecoserve', scale_binding={'path': '/historical/eco', 'sha256': 'a' * 64}),
                 reused=[dict(cell_id='original-low-slo')], pending=[dict(cell_id='new-low-slo'), dict(cell_id='new-high-slo')])
    calls = []
    def verify(row, refs, source, required_binding):
        calls.append((row, refs, source, required_binding))
        return dict(cell_id=row['cell_id'], measurement_valid=True, slo_attainment=.01)
    contract = SimpleNamespace(verify_existing=verify)
    spec = dict(source=dict(sha256='c' * 64))
    monkeypatch.setattr(h, 'check_prefix', lambda _: (contract, {}, spec, dict(groups=[group])))
    monkeypatch.setattr(h, 'compatibility', lambda *args: ({}, dict(output=str(output))))
    qualifier = SimpleNamespace(audit_binding=lambda _: dict(passed=True))
    return output, fresh_ref, qualifier, calls


def test_new_checkpoint_attributed_only_to_new_binding_and_poor_slo_retained(tmp_path, monkeypatch):
    output, reference, qualifier, calls = history_fixture(tmp_path, monkeypatch)
    save(output / 'checkpoints/new-low-slo.json', {})
    result = h.reconcile({}, reference, qualifier)
    assert result['historical_ids'] == ['original-low-slo']
    assert result['completed'][0]['slo_attainment'] == .01
    assert result['pending'] == [dict(cell_id='new-high-slo')]
    assert calls == [(dict(cell_id='new-low-slo'), [reference], 'c' * 64, reference)]


@pytest.mark.parametrize('bad', ['historical_replay', 'undeclared', 'orphan_operation', 'orphan_cell', 'failed_qualification'])
def test_suffix_rejects_replay_or_silent_retry(tmp_path, monkeypatch, bad):
    output, reference, qualifier, _ = history_fixture(tmp_path, monkeypatch)
    if bad == 'historical_replay': save(output / 'checkpoints/original-low-slo.json', {})
    if bad == 'undeclared': save(output / 'checkpoints/unknown.json', {})
    if bad == 'orphan_operation': (output / 'operations/new-low-slo').mkdir(parents=True)
    if bad == 'orphan_cell': (output / 'cells/new-low-slo').mkdir(parents=True)
    if bad == 'failed_qualification': qualifier.audit_binding = lambda _: dict(passed=False)
    with pytest.raises(RuntimeError):
        h.reconcile({}, reference, qualifier)


def test_prepare_bootstrap_is_new_unqualified_and_keeps_real_fresh_identity(tmp_path):
    values = pair(tmp_path)
    old_ref, fresh_ref = bindings(tmp_path, values)
    fresh = h.fixed(fresh_ref)
    fresh['deadline_s'] = 1788868800
    fresh['fresh_ablation_binding'] = True
    fresh_ref = save(tmp_path / 'new/binding.json', fresh)
    ready = dict(original_baseline_binding=old_ref, fresh_restoration_binding=fresh_ref,
        fresh_inventory=h.ref(tmp_path / 'new/identity.json'), measured_restore_status=save(tmp_path / 'restore-status.json', {}))
    before = Path(fresh_ref['path']).read_bytes()
    result_ref = tail.prepare_bootstrap(ready, tmp_path / 'tail')
    result = h.fixed(result_ref)
    assert result['configs'] == {} and result['output_correctness_verified'] is False
    assert result['correctness_gate_required_before_performance'] is True
    assert result['deadline_s'] == tail.ORIGINAL_DEADLINE
    assert result['identity_file'] == ready['fresh_inventory']['path']
    assert result['instances'] == fresh['instances']
    assert 'mechanism_proof' not in result and 'correctness_evidence' not in result
    assert result['restored_priority']['binding'] == fresh_ref
    assert Path(fresh_ref['path']).read_bytes() == before


def test_tail_rejects_return_while_coordinator_is_live(tmp_path):
    ready = dict(schema='B-measured-baseline-return-ready-v2', ready=True,
                 fresh_qualification_still_required=True, coordinator_pid=os.getpid())
    reference = save(tmp_path / 'ready.json', ready)
    with pytest.raises(RuntimeError, match='release and exit'):
        tail.ready_evidence(reference['path'])


def aggregate_fixture(tmp_path, monkeypatch, change=None):
    main = [dict(row=dict(cell_id=f'main-{system}-{dataset}-{n}', system=system, dataset=dataset))
            for system in h.BASELINES for dataset in ('alpaca', 'sharegpt', 'longbench') for n in range(10)]
    scale = [dict(cell_id=f'scale-{system}-{dataset}-{s}-{n}', system=system, dataset=dataset, slo_scale=s)
             for system in h.BASELINES for dataset in ('alpaca', 'sharegpt', 'longbench')
             for s in (.5, 2.) for n in range(3)]
    if change == 'missing_main': main.pop()
    if change == 'wrong_scale': scale[0]['slo_scale'] = 1
    if change == 'wrong_dataset': scale[0]['dataset'] = 'unknown'
    release_ref = save(tmp_path / 'release.json', dict(models={'32b': dict(records=main)}))
    source_ref = save(tmp_path / 'source.json', dict(cells=scale))
    proofs = [dict(cell_id=row['cell_id']) for row in scale]
    if change == 'missing_scale': proofs.pop()
    if change == 'duplicate_scale': proofs[-1] = proofs[0]
    calls = []
    contract = SimpleNamespace(released=SimpleNamespace(verify_release=lambda *a, **kw: calls.append((a, kw))))
    history = dict(pending=[], prefix=dict(original_release=release_ref), contract=contract,
        groups=[dict(group=dict(system='mixed'), reused=proofs[:-1])], completed=proofs[-1:],
        spec=dict(source=source_ref))
    monkeypatch.setattr(h, 'reconcile', lambda *a: history)
    status_ref = save(tmp_path / 'suffix-status.json', dict(complete=True, failed=[], remaining=[], finished_s=50))
    return status_ref, calls


def test_complete192_requires_deep_original_release_and_full_distinct_domain(tmp_path, monkeypatch):
    status_ref, calls = aggregate_fixture(tmp_path, monkeypatch)
    result = h.aggregate192({}, {'path': 'fresh', 'sha256': 'b' * 64}, status_ref, None)
    assert result['total'] == 192 and result['verified_baseline_main'] == 120 and result['verified_baseline_scale'] == 72
    assert result['original_stopped_parents_unchanged'] is True
    assert result['poor_slo_does_not_block_observation_completion'] is True
    assert calls[0][1] == dict(expected_model='32b', deep=True)


@pytest.mark.parametrize('change', ['missing_main', 'missing_scale', 'duplicate_scale', 'wrong_scale', 'wrong_dataset'])
def test_partial_or_duplicate_baseline_domain_cannot_publish192(tmp_path, monkeypatch, change):
    status_ref, _ = aggregate_fixture(tmp_path, monkeypatch, change)
    with pytest.raises(RuntimeError):
        h.aggregate192({}, {}, status_ref, None)
