"""Read-only baseline completion gate; never acquires hardware or a node lease.

Call again under the runner's fresh node lease immediately before deployment.
An old result snapshot is progress evidence, never permission to skip newer work.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

CAMPAIGN = Path('/root/workspace/pdblend-next-v1/campaign')
BASELINES = ('mixed', 'distserve', 'dynamollm', 'ecoserve')
DATASETS = ('alpaca', 'sharegpt', 'longbench')
SETTINGS = {
    '14b': dict(label='A14B', hostname='iZwz92bdfqihqp38tekqjyZ',
        release_sha='a87c7a683dddf9bacb07d7c864dae3be9c93d75118cabdcbd6c13aff7a38f3b6',
        pdb='A14B-five-system100-v1/pdblend/binding.json',
        groups=[('A14B-scale-stage-v1/resident-prepared-001/spec.json',
                 'A14B-scale-stage-v1/resident-attempt-001/status.json'),
                ('A14B-scale-stage-v1/heterogeneous-prepared/spec.json',
                 'A14B-scale-stage-v1/heterogeneous-attempt-001/status.json')]),
    '7b': dict(label='C7B', hostname='iZwz9gfq11hx1sbob59yrgZ',
        release_sha='91dadcea6d1b240245bede7854559e7a2d9e885c40720274eff3eb3cef0589ae',
        pdb='C7B-five-system100-v1/pdblend-r2/binding.json',
        groups=[('C7B-scale-bindings-v1/actual-001/scale-spec.json',
                 'C7B-scale-stage-v1/attempt-001/status.json')]),
    '32b': dict(label='B32B', hostname='iZwz9i5bte3xkpmcoes3t2Z',
        pdb='B32B-five-system100-v1/binding.pdblend.r2.json', groups=[]),
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def process_rows(proc=Path('/proc')):
    """Include the autonomous parent, not just its briefly lived cell child."""
    rows = []
    for path in proc.glob('[0-9]*/cmdline'):
        try:
            args = [x.decode(errors='replace') for x in path.read_bytes().split(b'\0') if x]
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Permission errors must fail closed: an incomplete process view is unsafe.
        if not args or not any('python' in Path(a).name for a in args[:1]):
            continue
        if any('\n' in a for a in args):
            continue
        scripts = ['/' + a.lstrip('/') for a in args[1:] if 'campaign/' in a and a.endswith('.py')]
        owns_queue = any(
            ('/scale-only-continuation-' in a and Path(a).name in ('supervise.py', 'scale_driver.py', 'child.py'))
            or ('/five-system-execution-' in a and Path(a).name in ('run.py', 'child.py'))
            or ('/five-system-orchestration-' in a and Path(a).name == 'bridge.py')
            or ('/deadline-phase-bridge-' in a and Path(a).name == 'bridge.py')
            or ('/C7B-cooperative-continuation-' in a and Path(a).name == 'watch.py')
            or ('/C7B-dynamo-cooperative-execution-' in a and Path(a).name == 'prepare.py')
            for a in scripts)
        if owns_queue:
            rows.append(dict(pid=int(path.parent.name), argv=args))
    return rows


def latest_snapshot(root, model):
    """Read only the latest completed v4 publication; do not trust its old liveness."""
    candidates = []
    for path in (root / 'five-system-results-v4').glob('actual-snapshot-*/manifest.json'):
        result = path.parent / 'results.json'
        if result.is_file():
            candidates.append((path.parent.name, path, result))
    if not candidates:
        return dict(available=False, note='Actual model release and scale raw checks are authoritative.')
    _, manifest_path, result_path = max(candidates)
    manifest = read(manifest_path)
    if manifest.get('files', {}).get('results.json') != sha(result_path):
        raise ValueError('latest completed snapshot results SHA differs')
    result = read(result_path)
    points = [p for p in result.get('points', []) if p.get('model') == model and p.get('system') in BASELINES]
    mapping = {}
    for p in points:
        source = p.get('executed_source') or {}
        if source.get('binding_path'):
            mapping[p['cell_id']] = dict(path=source['binding_path'], sha256=source['binding_sha256'])
    return dict(available=True, path=str(result_path), created_s=result.get('created_s'),
        manifest_sha256=sha(manifest_path), verified=sum(p.get('metrics_verified') is True and
        p.get('checkpoint_verified') is True for p in points), declared=len(points),
        source_mapping=mapping, liveness_authority=False)


def terminal_parent(spec_path, status_path, release_sha, proc_root=Path('/proc')):
    state = read(status_path)
    if state.get('spec_sha256') != sha(spec_path) or state.get('release_sha256') != release_sha:
        raise ValueError('supervisor spec/release binding differs: ' + str(status_path))
    if (state.get('complete') is not True or state.get('phase') != 'selected_scale_groups_finished'
            or state.get('finished_s') is None or state.get('error')):
        raise ValueError('baseline supervisor not clean terminal: ' + str(status_path))
    pid = state.get('pid')
    if type(pid) is not int or pid <= 0 or (proc_root / str(pid)).exists():
        raise ValueError('baseline supervisor process exit not established: ' + str(pid))
    for step in state.get('steps', []):
        if (step.get('complete') is not True or step.get('exitcode') != 0 or not step.get('finished_s')
                or step.get('unconfirmed_child') or not step.get('verified_new_checkpoint')):
            raise ValueError('baseline child or checkpoint not verified: ' + str(status_path))
        child = step.get('pid')
        if type(child) is not int or child <= 0 or (proc_root / str(child)).exists():
            raise ValueError('baseline child process exit not established: ' + str(child))
    return dict(path=str(status_path), sha256=sha(status_path), pid=pid,
        finished_s=state['finished_s'], steps=len(state.get('steps', [])))


def load_contract(root):
    path = root / 'scale-only-continuation-v3/contract.py'
    spec = importlib.util.spec_from_file_location('ablation_readiness_scale_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_actual(root, model, settings):
    """Revalidate actual source/CP/raw bytes even when the report snapshot lags."""
    contract = load_contract(root)
    release_path = root / 'model-main-release-v1' / ('actual-' + settings['label'] + '-release.json')
    contract.released.verify_release(release_path, settings['release_sha'], expected_model=model, deep=True)
    release = read(release_path)
    proof = release['models'][model]
    main = [r['row'] for r in proof['records'] if r['row']['system'] in BASELINES]
    all_rows = list(main)
    checked_groups = []
    for spec_name, _ in settings['groups']:
        spec_path = root / spec_name
        checked = contract.check_spec(read(spec_path), release_path, settings['release_sha'])
        if checked['pending']:
            raise ValueError('baseline scale checkpoints still pending: ' + spec_name)
        for group in checked['groups']:
            if group['group']['system'] not in BASELINES:
                continue
            all_rows.extend(group['rows'])
            checked_groups.append(dict(id=group['group']['id'], system=group['group']['system'],
                datasets=group['group']['datasets'], verified=len(group['reused']), pending=len(group['pending'])))
    validate_domain(all_rows, model)
    return dict(raw_verified=True, verified_baseline_main=120, verified_baseline_scale=72,
        total=192, groups=checked_groups, release_path=str(release_path), release_sha256=settings['release_sha'],
        work_and_slo_not_completion_gates=True)


def validate_domain(rows, model):
    ids = [r['cell_id'] for r in rows]
    if len(rows) != 192 or len(set(ids)) != 192 or any(r['model'] != model for r in rows):
        raise ValueError('all four baseline domains require exactly 192 unique verified cells')
    for system in BASELINES:
        for dataset in DATASETS:
            for phase, count, scales in (('main', 10, {1.0}), ('scale', 6, {.5, 2.0})):
                actual = [r for r in rows if (r['system'], r['dataset'], r['phase']) == (system, dataset, phase)]
                if len(actual) != count or {r['slo_scale'] for r in actual} != scales:
                    raise ValueError('baseline main/scale domain missing: ' + '/'.join((system, dataset, phase)))
    return True


def _inspect(model, *, root=CAMPAIGN, settings=None, hostname=None, proc_root=Path('/proc'),
             process_reader=process_rows, verifier=verify_actual, snapshot_reader=latest_snapshot):
    model = model.lower()
    result = dict(model=model, observed_s=time.time(), ready=False, reasons=[], evidence={},
        current_binding=None, requires_fresh_node_lease_recheck=True, hardware_actions=False)
    if model not in SETTINGS:
        result['reasons'].append('unsupported model'); return result
    settings = settings or SETTINGS[model]
    try:
        if (hostname or socket.gethostname()) != settings['hostname']:
            raise ValueError('readiness must be checked on the actual bound model host')
        binding_path = root / settings['pdb']
        if binding_path.is_file():
            binding = read(binding_path)
            result['current_binding'] = dict(path=str(binding_path), sha256=sha(binding_path),
                host_release=binding.get('host_release'), historical=True, rebind_after_restart_required=True)
        try:
            result['evidence']['snapshot'] = snapshot_reader(root, model)
        except (OSError, ValueError, KeyError) as exc:
            # A stale/absent report must not replace actual raw verification.
            result['evidence']['snapshot'] = dict(available=False, error=str(exc))
        if model == '32b':
            raise ValueError('32B baseline mechanism qualification and all main+scale completion release not installed; fail closed')
        parents = [terminal_parent(root / s, root / p, settings['release_sha'], proc_root)
                   for s, p in settings['groups']]
        result['evidence']['parents'] = parents
        live = process_reader(proc_root)
        if live:
            result['evidence']['live_queues'] = live
            raise ValueError('a baseline serving driver or automatic successor parent is still alive')
        completion = verifier(root, model, settings)
        if (completion.get('raw_verified') is not True or completion.get('verified_baseline_main') != 120
                or completion.get('verified_baseline_scale') != 72 or completion.get('total') != 192):
            raise ValueError('actual baseline raw verification is not complete')
        result['evidence']['actual_completion'] = completion
        # Check after expensive raw hashing as well; a free lock between cells is not a release.
        after = [terminal_parent(root / s, root / p, settings['release_sha'], proc_root)
                 for s, p in settings['groups']]
        if after != parents or process_reader(proc_root):
            raise ValueError('baseline parent state changed during raw verification')
        result['ready'] = True
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
        result['reasons'].append(str(exc))
    return result


def inspect_readiness(model):
    return _inspect(model)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=SETTINGS)
    print(json.dumps(inspect_readiness(parser.parse_args().model), indent=2))
