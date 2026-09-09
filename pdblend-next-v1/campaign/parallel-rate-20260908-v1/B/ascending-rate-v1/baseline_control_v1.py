"""One retained B32B baseline restart and original 27 native requests, after PDB4.5."""
import argparse
import asyncio
import copy
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cells_v1 as r
g, p, R = r.g, r.p, r.R
REPO = R.parents[1]
B = R / 'B'
C = REPO / 'campaign'
ORIGINAL = C / 'B32B-ecoserve-qualified-main-v1/binding.json'
RESTORE = HERE / 'baseline-restoration-001'
QUAL = HERE / 'baseline-qualification-001'
GATE = B / 'baseline-gate-until-complete-v1/validate.py'
ECOHOST = REPO / 'releases/five-system100-B32B-baseline-eco-drain-v1-runtime'
OLD_BINDINGS = {
    'mixed': B / 'baseline-qualification-after-external-001/mixed/binding.json',
    'distserve': B / 'baseline-qualification-after-external-001/distserve/binding.json',
    'dynamollm': B / 'cooperative-dynamo-qualification-001/dynamollm/binding.json',
    'ecoserve': B / 'eco-drain-qualification-001/ecoserve/binding.json',
}


def module(path, name):
    return g.load(path, name)


def terminal():
    release_ref = p.ref(HERE / 'pdb-release-002/release.json')
    release = p.checked(release_ref)
    state_ref = p.ref(HERE / 'pdb-performance-001/status.json')
    state = p.checked(state_ref)
    assert state['complete'] and not state['failed'] and not state['node_lease_held'] and state['finished_s']
    assert not r.alive(state['pid']) and state['release'] == release_ref
    assert len(state['observed_checkpoints']) == len(state['completed']) == 2
    observations = []
    for reference, row in zip(state['observed_checkpoints'], release['rows']):
        cp = p.checked(reference)
        assert cp['row'] == row and cp['release'] == release_ref and cp['arrival_qualification']['passed']
        for path, digest in cp['artifacts'].items():
            assert p.sha(path) == digest
        receipt = p.checked(cp['receipt'])
        assert receipt['measurement_valid'] and r.full(receipt['summary'], row)
        assert receipt['clock_restore_complete'] and all(v['complete'] for v in receipt['restoration'].values())
        observations.append(dict(receipt['summary'], cell_id=row['cell_id'], repeat=row['repeat']))
    contract = module(Path(release['declaration_contract']['path']), 'ascending_B_boundary')
    group = contract.resolve_group(release['declaration'], '32b', 'alpaca', actual_host='B')
    selected = contract.select_group(group, observations)
    assert selected['phase'] == 'baselines' and selected['cap_rate_rps'] in (4.5, 5.)
    new_rows = [t['row'] for t in selected['baseline_tasks'] if t['action'] == 'execute']
    assert len(new_rows) == 8 and {x['rate_rps'] for x in new_rows} == {4.5}
    return dict(schema='B32B-ascending-Alpaca-boundary-v1', node='B', model='32b', dataset='alpaca',
                declaration=release['declaration'], cap_rate_rps=selected['cap_rate_rps'], decision=selected,
                actual_new_pdb_checkpoints=state['observed_checkpoints'], pdb_terminal=state_ref,
                required_new_baseline_cell_ids=[x['cell_id'] for x in new_rows],
                pinned_old_4_and_5_used_in_numeric_order=True, no_old_result_reexecuted=True), release


async def restore():
    boundary, release = terminal()
    assert not (HERE / 'baseline-boundary-001.json').exists()
    p.save(HERE / 'baseline-boundary-001.json', boundary)
    original = p.read(ORIGINAL)
    assert p.sha(ORIGINAL) == '87dbf43fcf8076dc1b4bc588e14fad717a2d2b3eeb83cae1514f3fe2d078d119'
    helper = module(B / 'baseline-return-after-external-source-v1/execution.py', 'ascending_B_baseline_restore')
    common = helper.load_common(original['host_release'])
    target = copy.deepcopy(original)
    target.update(deadline_s=None, campaign_lifecycle='until_declared_complete_v1')
    previous = p.checked(release['binding'])
    from ecopadg.serving.campaign import node_lease
    assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
    with node_lease():
        binding = await helper.restore_core(common, target, previous, RESTORE)
    return dict(complete=True, binding=p.ref(RESTORE / 'binding.json'), status=p.ref(RESTORE / 'status.json'))


def make_bootstrap():
    status = p.read(RESTORE / 'status.json')
    assert status['complete'] and status['correctness']['passed'] and status['clock_restore_complete'] and not status['errors']
    boot = copy.deepcopy(p.read(RESTORE / 'binding.json'))
    assert not QUAL.exists()
    boot.update(host_release=str(ECOHOST), configs={}, output=str(QUAL / 'gate-work'),
                output_correctness_verified=False, correctness_gate_required_before_performance=True,
                old_correctness_is_historical_only=True, formal_eligible=False,
                identity_file=str(RESTORE / 'containers.after.json'),
                restored_priority=dict(binding=p.ref(RESTORE / 'binding.json'), status=p.ref(RESTORE / 'status.json'),
                                       inventory=p.ref(RESTORE / 'containers.after.json')))
    for key in ('mechanism_proof', 'correctness_evidence', 'fresh_ablation_binding', 'ablation', 'qualification', 'oracle'):
        boot.pop(key, None)
    manifest = p.read(ECOHOST / 'manifest.json')
    boot['files'].update({str(ECOHOST / f): h for f, h in manifest['files'].items()})
    for f in (ECOHOST / 'manifest.json', Path(__file__), HERE / 'baseline-boundary-001.json', GATE,
              RESTORE / 'binding.json', RESTORE / 'status.json', RESTORE / 'containers.after.json'):
        boot['files'][str(f)] = p.sha(f)
    p.save(QUAL / 'bootstrap.json', boot)
    return p.ref(QUAL / 'bootstrap.json')


def gate():
    bp = make_bootstrap()
    b = p.checked(bp)
    runtime = {p.read(i['engine_config'])['runtime_dir'] for i in b['instances']}
    assert len(runtime) == 1
    argv = [sys.executable, '-B', str(GATE), '--binding', bp['path'], '--runtime-dir', runtime.pop(),
            '--out', str(QUAL / 'original27'), '--run']
    started = time.time()
    stopped = False
    with (QUAL / 'original27.log').open('xb') as f:
        child = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT)
        def stop(*_):
            nonlocal stopped
            if not stopped:
                stopped = True
                child.terminate()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        exitcode = child.wait()
    p.save(QUAL / 'gate-invocation.json', dict(argv=argv, pid=child.pid, stop_requested=stopped, exitcode=exitcode, started_s=started,
        finished_s=time.time(), bootstrap=bp, host_manifest=p.ref(ECOHOST / 'manifest.json'), gate_source=p.ref(GATE)))
    s = p.read(QUAL / 'original27/status.json')
    assert not stopped and exitcode in (0, 1) and s['complete'] and s['measurement_valid']
    assert s['native_cleanup_complete'] and s['clock_restore_complete'] and not s['cleanup_errors']
    assert not s.get('sampling_error')
    return dict(native_gate_complete=True, legacy_passed=s['passed'], legacy_mechanism_gate=s['mechanism_gate'])


def derive_other(system):
    assert system in ('mixed', 'distserve', 'dynamollm')
    c = module(B / 'baseline_control_after_external_v2.py', 'ascending_B_original_binding')
    helper = module(B / 'baseline-return-after-external-source-v1/execution.py', 'ascending_B_binding_host')
    historical = p.read(OLD_BINDINGS[system])
    helper.load_common(historical['host_release'])
    boot_ref = p.ref(QUAL / 'bootstrap.json')
    original_ref = p.ref(c.POLICIES[system])
    out = QUAL / system
    binding = c.bind_other(system, original_ref, boot_ref, QUAL / 'original27', out)
    assert binding['configs'] == historical['configs']
    binding['host_release'] = historical['host_release']
    host = Path(binding['host_release'])
    for f, digest in p.read(host / 'manifest.json')['files'].items():
        binding['files'][str(host / f)] = digest
    for f in (host / 'manifest.json', OLD_BINDINGS[system], Path(__file__), QUAL / 'gate-invocation.json'):
        binding['files'][str(f)] = p.sha(f)
    binding['ascending_native_qualification'] = dict(original_selected_binding=p.ref(OLD_BINDINGS[system]),
        bootstrap=boot_ref, fresh_gate=p.ref(QUAL / 'original27/status.json'),
        native_engine_sources_unchanged=True, controller_source_and_policy_unchanged=True,
        gate_only_native_mechanisms_not_performance=True)
    return binding


def qualify():
    boot_ref = p.ref(QUAL / 'bootstrap.json')
    q = module(B / 'qualify_eco_drained_baseline_v2.py', 'ascending_B_original_Eco_qualifier')
    eco = q.qualify(boot_ref, QUAL / 'original27', p.ref(ORIGINAL), p.ref(RESTORE / 'containers.after.json'), QUAL / 'ecoserve')
    q.audit_binding(eco)
    refs = {'ecoserve': eco}
    for system in ('mixed', 'distserve', 'dynamollm'):
        binding = derive_other(system)
        path = QUAL / system / 'binding.json'
        assert not path.exists()
        p.save(path, binding)
        refs[system] = p.ref(path)
    p.save(QUAL / 'bindings.json', refs)
    return refs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('restore', 'gate', 'qualify'))
    args = ap.parse_args()
    async def controlled_restore():
        task = asyncio.current_task()
        stopped = False
        def stop():
            nonlocal stopped
            if not stopped:
                stopped = True
                task.cancel()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop)
        return await restore()
    print(json.dumps(asyncio.run(controlled_restore()) if args.action == 'restore' else gate() if args.action == 'gate' else qualify()))
