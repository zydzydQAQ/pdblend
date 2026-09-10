"""Frozen PDB-only scope and honest retirement of the user-stopped baseline queue."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / 'ascending-rate-v1'
V2 = HERE.parent / 'ascending-resume-20260909-v2'
UNIFORM = HERE.parent / 'uniform-rate-20260909-v1'
RUNNER = ROOT / 'C/uniform-rate-20260909-v1'
sys.path.insert(0, str(RUNNER))
import support as p
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v1'))
import contract
import metrics


def source_contract():
    manifest = p.read(HERE / 'manifest.json')
    assert manifest['schema'] == 'B32B-pdblend-only-source-v1'
    for path, digest in manifest['files'].items():
        assert p.sha(path) == digest, 'frozen PDB-only source/evidence changed: ' + path
    priority = p.checked(manifest['user_priority'])
    assert priority['effective_systems'] == ['pdblend']
    assert priority['future_pdb_progress_must_not_wait_for_baseline_completion'] is True
    assert (UNIFORM / 'STOP').exists(), 'retired baseline queue must remain disabled'
    assert not (HERE / 'STOP').exists(), 'PDB-only stop requested'
    return manifest


def retired_evidence(*, probe_processes=False):
    manifest = source_contract()
    inputs = p.checked(manifest['predecessor_inputs'])
    states = [p.checked(r) for r in inputs['statuses']]
    receipts = []
    owners = []
    for reference in inputs.get('interrupted_historical_statuses', []):
        state = p.checked(reference)
        owners.append({k: state[k] for k in ('pid', 'startticks') if k in state})
        for record in list(state.get('stages', {}).values()) + state.get('children', []):
            owners.append({k: record[k] for k in ('pid', 'startticks') if k in record})
    for state in states:
        assert state.get('finished_s') and not state.get('node_lease_held'), 'prior owner has not released measurement'
        owners.append({k: state[k] for k in ('pid', 'startticks') if k in state})
        for record in list(state.get('stages', {}).values()) + state.get('children', []):
            assert record.get('finished_s') and 'exitcode' in record
            owners.append({k: record[k] for k in ('pid', 'startticks') if k in record})
        for reference in state.get('observed_checkpoints', []):
            cp = p.checked(reference)
            receipt_ref = cp['receipt']
            if isinstance(receipt_ref, str):
                receipt_ref = dict(path=receipt_ref, sha256=cp['receipt_sha256'])
            receipt = p.checked(receipt_ref)
            # No baseline SLO/work-completion prerequisite: only real resource cleanup.
            assert receipt['child_stopped'] and receipt['clock_restore_complete']
            assert not receipt['outer_cleanup_errors']
            assert receipt['restoration'] and all(x.get('complete') and not x.get('errors') for x in receipt['restoration'].values())
            owners.append(dict(pid=receipt['child_pid']))
            receipts.append(receipt_ref)
    eco = p.checked(inputs['stopped_ecoserve'])
    assert eco['attempted'] == eco['completed'] == [inputs['completed_eco_cell']]
    assert len(eco['observed_checkpoints']) == 1 and not eco['failed']
    assert eco['complete'] is False and eco['error'] == 'AssertionError()'
    pipeline = p.checked(inputs['stopped_pipeline'])
    assert pipeline['error'] == "AssertionError('stop requested')"
    assert not pipeline.get('observations') and not pipeline.get('boundaries')
    assert not (UNIFORM / 'pdb-restoration-001').exists(), 'another task already started B PDB restoration'
    for relative in ('results/operations', 'results/checkpoints'):
        target = UNIFORM / 'predecessor-ecoserve-001' / relative / (inputs['unattempted_eco_cell'] + ('.json' if relative.endswith('checkpoints') else ''))
        assert not target.exists(), 'unexpected second EcoServe attempt'
    if probe_processes:
        for owner in owners:
            assert not p.active_owner(owner), 'retired owner still running: ' + str(owner)
    return dict(schema='B32B-user-stopped-baseline-predecessor-v1',
        source_statuses=inputs['statuses'], cleanup_receipts=receipts,
        user_priority=manifest['user_priority'], baseline_suite_complete=False,
        original_statuses_preserved=True, baseline_SLO_or_complete_work_required=False,
        hardware_owner_processes_checked=probe_processes, retired_owners=owners,
        skipped_baseline_cell=inputs['unattempted_eco_cell'],
        current_native_idle_and_identity_checked_by_original_restore=True)


def predecessor():
    proof = retired_evidence(probe_processes=True)
    path = HERE / 'predecessor-release-001.json'
    assert not path.exists(), 'new restore attempt already recorded'
    p.save(path, proof)
    return p.ref(path)


def pdb_reuse():
    inputs = p.checked(source_contract()['predecessor_inputs'])
    references = p.checked(inputs['pdb_reuse_index'])
    assert len(references) == 2
    old_state = p.checked(inputs['old_pdb_status'])
    assert len(old_state['observed_checkpoints']) == 2
    for reference, checkpoint in zip(references, old_state['observed_checkpoints']):
        observation = p.checked(reference)
        assert observation['checkpoint'] == checkpoint and observation['system'] == 'pdblend'
        recomputed = metrics.audit_checkpoint(checkpoint['path'])
        assert all(observation[k] == value for k, value in recomputed.items())
        assert observation['measurement_valid'] and observation['work_complete']
        cp = p.checked(checkpoint)
        verified = p.load(cp['qualification_validator'], 'B_pdb_only_old_qualification').verify(cp['qualification'])
        assert verified['passed'] and verified['independently_recomputed']
    group = contract.resolve_group(source_contract()['declaration'], '32b', 'alpaca', actual_host='B')
    contract.apply_audited_reuse(group, [p.checked(r) for r in references])
    return references


def validate():
    manifest = source_contract()
    proof = retired_evidence()
    reuse = pdb_reuse()
    starts = {}
    for dataset in contract.DATASETS:
        group = contract.resolve_group(manifest['declaration'], '32b', dataset, actual_host='B')
        if dataset == 'alpaca':
            group = contract.apply_audited_reuse(group, [p.checked(r) for r in reuse])
        selected = contract.select_group(group, [])
        starts[dataset] = dict(phase=selected['phase'], rate_rps=selected.get('rate_rps'),
            next_cell_ids=[t['cell_id'] for t in selected.get('next_tasks', [])])
    return dict(passed=True, cpu_only=True, systems=['pdblend'], historical_PDB_repeats_recomputed=2,
        baseline_cleanup_receipts=len(proof['cleanup_receipts']), retired_statuses=len(proof['source_statuses']),
        baseline_results_preserved=True, baseline_remaining_skipped=proof['skipped_baseline_cell'],
        first_dispatch_by_dataset=starts, live_process_and_native_checks_deferred_to_run=True)
