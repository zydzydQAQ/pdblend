"""Fresh B32B bindings for uniform rates, after the existing eight baseline runs."""
import argparse
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / 'ascending-rate-v1'
RESUME = HERE.parent / 'ascending-resume-20260909-v2'
sys.path.insert(0, str(OLD))
import qualify_pdb_v1 as g
p = g.p


def load(path, name):
    return g.load(Path(path), name)


def control():
    module = load(OLD / 'baseline_control_v2.py', 'uniform_B_baseline_control')
    module.HERE = HERE
    module.RESTORE = HERE / 'baseline-restoration-001'
    module.QUAL = HERE / 'baseline-qualification-001'
    return module


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def predecessor():
    state = p.read(RESUME / 'pipeline-001/status.json')
    assert state.get('finished_s') and not alive(state['pid']), 'existing B baseline pipeline is still active'
    assert not state.get('node_lease_held')
    if state.get('complete'):
        assert len(state['observed_checkpoints']) == 8 and len(state['completed_systems']) == 4 and not state.get('error')
        return p.ref(RESUME / 'pipeline-001/status.json')
    # Preserve the original stop. Only the independently diagnosed timeout
    # continuation can satisfy this predecessor; no success flag is forged.
    tail = load(HERE / 'finish_predecessor.py', 'uniform_B_predecessor_audit')
    diagnosis = p.ref(tail.initial_diagnosis())
    sources = [p.ref(RESUME / 'baseline-mixed-performance-001/status.json')]
    for system, count in (('distserve', 1), ('dynamollm', 2), ('ecoserve', 2)):
        reference = p.ref(HERE / ('predecessor-' + system + '-001/status.json'))
        result = p.checked(reference)
        assert result['complete'] and not result['failed'] and not result.get('error')
        assert result.get('finished_s') and not result['node_lease_held'] and not alive(result['pid'])
        assert len(result['completed']) == len(result['observed_checkpoints']) == count
        for checkpoint in result['observed_checkpoints']:
            assert p.sha(checkpoint['path']) == checkpoint['sha256']
            tail.diagnose(checkpoint['path'])
        sources.append(reference)
    value = dict(schema='uniform-B-original-baseline-completion-v1', complete=True,
        old_failure_preserved=True, diagnosis=diagnosis, source_statuses=sources,
        total_measurements=8, repeated_completed_measurements=0)
    path = HERE / 'predecessor-completed.json'
    if path.exists():
        assert p.read(path) == value
    else:
        p.save(path, value)
    return p.ref(path)


async def restore_pdb():
    previous_ref = predecessor()
    target = copy.deepcopy(p.read(OLD / 'pdb-release-002/binding.json'))
    original = p.read(ROOT / 'B/completion-release-p4-001/release.json')
    target['configs'] = {name: value['path'] for name, value in original['configs']['fixed2'].items()}
    for value in [*original['configs']['fixed2'].values(), *original['profile_refs'].values()]:
        p.checked(value)
        target['files'][value['path']] = value['sha256']
    target.update(deadline_s=None, campaign_lifecycle='until_declared_complete_v1',
                  experiment_scope='B32B all three datasets, unchanged P4 policy and equivalent P12 OFF source')
    previous = p.read(RESUME / 'baseline-ecoserve-release-001/binding.json')
    helper = load(ROOT / 'B/baseline-return-after-external-source-v1/execution.py', 'uniform_B_original_restore')
    common = helper.load_common(target['host_release'])
    from ecopadg.serving.campaign import node_lease
    assert 'PDBLEND_NODE_LOCK_FD' not in os.environ and socket.gethostname() == target['hostname']
    with node_lease():
        actual = await helper.restore_core(common, target, previous, HERE / 'pdb-restoration-001')
    state_path = HERE / 'pdb-restoration-001/status.json'
    state = p.read(state_path)
    state.update(pid=os.getpid(), node_lease_held=False, predecessor=previous_ref)
    p.save(state_path, state)
    return actual


def make_pdb_spec():
    restore = HERE / 'pdb-restoration-001'
    state = p.read(restore / 'status.json')
    assert state['complete'] and not state['node_lease_held'] and not alive(state['pid'])
    spec = copy.deepcopy(p.read(OLD / 'qualification-spec-001.json'))
    spec.update(restoration=p.ref(restore / 'status.json'), binding=p.ref(restore / 'binding.json'),
                ordinary=p.ref(restore / 'correctness/status.json'))
    for key in ('restoration', 'binding', 'ordinary'):
        spec['files'][spec[key]['path']] = spec[key]['sha256']
    spec['files'].update(p.read(restore / 'binding.json')['files'])
    path = HERE / 'pdb-qualification-spec.json'
    assert not path.exists()
    p.save(path, spec)
    g.validate(spec)
    return path


def freeze_pdb():
    out = HERE / 'pdb-qualification-001'
    status = p.read(out / 'status.json')
    assert status['passed'] and status['finished_s'] and not status['node_lease_held'] and not alive(status['pid'])
    spec = p.read(HERE / 'pdb-qualification-spec.json')
    files = dict(spec['files'])
    for root in (HERE / 'pdb-restoration-001', out):
        files.update({str(path): p.sha(path) for path in root.rglob('*') if path.is_file()})
    for path in (HERE / 'pdb-qualification-spec.json', OLD / 'verify_pdb_v1.py', OLD / 'qualify_pdb_v1.py', Path(__file__)):
        files[str(path)] = p.sha(path)
    contract = dict(schema='B32B-ascending-saved-qualification-v1', spec=p.ref(HERE / 'pdb-qualification-spec.json'),
                    status=p.ref(out / 'status.json'), files=files)
    path = HERE / 'pdb-qualified.json'
    assert not path.exists()
    p.save(path, contract)
    verifier = load(OLD / 'verify_pdb_v1.py', 'uniform_B_pdb_saved')
    verified = verifier.verify(p.ref(path))
    entry = dict(binding=verified['binding'], qualification=p.ref(path),
        qualification_validator=p.ref(OLD / 'verify_pdb_v1.py'), host_manifest=verified['host_manifest'],
        predecessors=[p.ref(out / 'status.json')])
    p.save(HERE / 'pdb-entry.json', entry)
    return entry


async def restore_baselines():
    # The caller writes this only after all three ordered PDB groups are capped.
    boundary = p.read(HERE / 'pdb-boundaries.json')
    assert boundary['complete'] and set(boundary['groups']) == {'alpaca', 'sharegpt', 'longbench'}
    state = p.read(boundary['pipeline_status'])
    assert not state.get('node_lease_held')
    c = control()
    p.save(HERE / 'baseline-boundary-001.json', boundary)
    target = copy.deepcopy(p.read(c.ORIGINAL))
    target.update(deadline_s=None, campaign_lifecycle='until_declared_complete_v1')
    previous = p.checked(p.read(HERE / 'pdb-entry.json')['binding'])
    helper = load(ROOT / 'B/baseline-return-after-external-source-v1/execution.py', 'uniform_B_baseline_restore')
    common = helper.load_common(target['host_release'])
    from ecopadg.serving.campaign import node_lease
    assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
    with node_lease():
        return await helper.restore_core(common, target, previous, c.RESTORE)


def qualify_baselines():
    c = control()
    q = load(RESUME / 'qualify_eco_drained_baseline_v2.py', 'uniform_B_Eco_qualifier')
    eco = q.qualify(p.ref(c.QUAL / 'bootstrap.json'), c.QUAL / 'original27', p.ref(c.ORIGINAL),
                    p.ref(c.RESTORE / 'containers.after.json'), c.QUAL / 'ecoserve')
    q.audit_binding(eco)
    references = {'ecoserve': eco}
    for system in ('mixed', 'distserve', 'dynamollm'):
        binding = c.derive_other(system)
        path = c.QUAL / system / 'binding.json'
        assert not path.exists()
        p.save(path, binding)
        references[system] = p.ref(path)
    p.save(c.QUAL / 'bindings.json', references)
    return references


def freeze_baselines():
    c = control()
    entries = {}
    for system, reference in p.read(c.QUAL / 'bindings.json').items():
        binding = p.checked(reference)
        out = HERE / ('baseline-' + system + '-qualified')
        out.mkdir(exist_ok=False)
        files = dict(binding['files'])
        files.update({str(f): p.sha(f) for f in c.QUAL.rglob('*') if f.is_file()})
        files.update({str(f): p.sha(f) for f in c.RESTORE.rglob('*') if f.is_file()})
        contract = dict(schema='B32B-ascending-baseline-saved-qualification-v1', system=system, files=files,
            original_selected_binding=p.ref(c.OLD_BINDINGS[system]), fresh_binding=reference,
            executed_binding=reference, execution_output=binding['output'], execution_files=binding['files'],
            host_manifest=p.ref(Path(binding['host_release']) / 'manifest.json'))
        p.save(out / 'qualification.json', contract)
        verifier = load(HERE / 'verify_baseline.py', 'uniform_B_baseline_saved')
        verified = verifier.verify(p.ref(out / 'qualification.json'))
        entries[system] = dict(binding=verified['binding'], qualification=p.ref(out / 'qualification.json'),
            qualification_validator=p.ref(HERE / 'verify_baseline.py'), host_manifest=verified['host_manifest'], predecessors=[])
    p.save(HERE / 'baseline-entries.json', entries)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pdb-restore', 'pdb-spec', 'pdb-freeze', 'baseline-restore', 'baseline-gate', 'baseline-qualify', 'baseline-freeze'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    assert args.run, 'phase execution requires --run'
    assert not (HERE / 'STOP').exists()
    async def controlled(fn):
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        return await fn()
    if args.phase == 'pdb-restore':
        value = asyncio.run(controlled(restore_pdb))
    elif args.phase == 'pdb-spec':
        value = str(make_pdb_spec())
    elif args.phase == 'pdb-freeze':
        value = freeze_pdb()
    elif args.phase == 'baseline-restore':
        value = asyncio.run(controlled(restore_baselines))
    elif args.phase == 'baseline-gate':
        value = control().gate()
    elif args.phase == 'baseline-qualify':
        value = qualify_baselines()
    else:
        value = freeze_baselines()
    print(json.dumps(value))


if __name__ == '__main__':
    main()
