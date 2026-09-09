"""Wait for B's baseline, restore once, run priority repeats, then original core18.

The original watcher, restoration, lease and measurement code stay unchanged.
No hardware action occurs at import or without --run.
"""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'pdblend-ablation-20260908-v1'
ORIGINAL_ATTEMPT = ORIGINAL / 'attempts/32b-001'
OLD_WATCHER = ORIGINAL / 'watchers/32b/observer-001.json'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_modules():
    """Resolve the original bare imports without replacing unrelated modules."""
    names = ('execution', 'readiness', 'run', 'readiness_b32')
    saved = {name: sys.modules.get(name) for name in names}
    try:
        execution = load(ORIGINAL / 'execution.py', 'b_priority_original_execution')
        readiness = load(ORIGINAL / 'readiness.py', 'b_priority_original_readiness')
        sys.modules.update(execution=execution, readiness=readiness)
        runner = load(ORIGINAL / 'run.py', 'b_priority_original_runner')
        sys.modules['run'] = runner
        readiness_b32 = load(ORIGINAL / 'readiness_b32.py', 'b_priority_original_readiness_b32')
        sys.modules['readiness_b32'] = readiness_b32
        b_runner = load(ORIGINAL / 'run_b32.py', 'b_priority_original_b_runner')
        b_runner.install()
        return execution, runner, b_runner
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def verify_wrapper(manifest_path, expected_sha):
    require(sha(manifest_path) == expected_sha, 'priority wrapper manifest changed')
    manifest = read(manifest_path)
    require(manifest.get('kind') == 'B32B-priority-repeats-then-original18-v1', 'wrong wrapper manifest')
    mandatory = {'B_watch.py', 'repeat.py', 'declarations.json',
                 'B-watcher-stop-before.json', 'B-watcher-stop-after.json'}
    require(mandatory <= set(manifest['files']), 'wrapper source/authorization inventory incomplete')
    for name, digest in manifest['files'].items():
        require(sha(ROOT / name) == digest, 'priority wrapper source changed: ' + name)
    for name, digest in manifest['dependencies'].items():
        require(sha(name) == digest, 'frozen original dependency changed: ' + name)
    return manifest


def verify_stopped_waiter():
    before = read(ROOT / 'B-watcher-stop-before.json')
    after = read(ROOT / 'B-watcher-stop-after.json')
    require(after['target_pid'] == before['watcher']['pid'] == 526987
            and after['target_startticks'] == before['watcher']['startticks'], 'stopped watcher identity differs')
    require(after['watcher_exited'] and after['watcher_lock_released']
            and after['hardware_started'] is False and after['original_attempt_absent']
            and after['baseline_active'] and after['gpu_controls_issued'] is False,
            'old waiting-only stop was not verified')
    require(not Path('/proc/526987').exists(), 'original waiting watcher exists')
    require(sha(OLD_WATCHER) == after['old_journal_sha256'], 'old watcher journal changed')
    require(not ORIGINAL_ATTEMPT.exists(), 'original ablation attempt already exists')
    return after


def install_priority_hook(execution, runner, repeat, args):
    original_run_cells = execution.run_cells
    expected_cells = runner.selected_cells(ORIGINAL / 'declarations', '32b')
    require(len(expected_cells) == len({c['cell_id'] for c in expected_cells}) == 18,
            'original B core18 declaration differs')
    expected_repeats = [c for c in read(args.repeat_declarations)['cells'] if c['model'] == '32b']
    require(len(expected_repeats) == len({c['cell_id'] for c in expected_repeats}) == 16,
            'priority B16 repeat declaration differs')
    called = False

    async def priority_first(common, base, cells, out, stop_path, on_update):
        nonlocal called
        require(not called, 'priority wrapper cannot replay the new or original queue')
        called = True
        verify_wrapper(args.manifest, args.manifest_sha256)
        runner.verify_package()
        require(cells == expected_cells and Path(out).resolve() == ORIGINAL_ATTEMPT,
                'original18 order/output changed')
        require(Path(stop_path).resolve() == ORIGINAL / 'STOP', 'original STOP scope changed')
        require(base['model'] == '32b' and base['system'] == 'pdblend', 'wrong restored PDB binding')
        require(not args.repeat_out.exists(), 'priority repeat output already exists; no implicit retry')
        state = read(args.status)
        state.update(phase='priority_original_policy_repeats', priority_repeat_output=str(args.repeat_out),
                     priority_repeat_ids=[c['cell_id'] for c in expected_repeats],
                     original_core18_pending=[c['cell_id'] for c in expected_cells],
                     priority_order='current baseline terminal -> one original PDB restoration -> new16 -> original18',
                     replaced_waiter=str(ROOT / 'B-watcher-stop-after.json'))
        execution.write(args.status, state)
        result = await repeat.run_repeats(common, base, '32b', args.repeat_out, args.repeat_declarations)
        require(result.get('complete') is True, 'priority repeats stopped/failed; original18 not started')
        verify_wrapper(args.manifest, args.manifest_sha256)
        runner.verify_package()
        require(runner.selected_cells(ORIGINAL / 'declarations', '32b') == expected_cells,
                'original18 declaration changed during priority repeats')
        state = read(args.status)
        state.update(phase='core_ablation', priority_repeats_complete=True,
                     priority_repeat_result=result, original_core18_resumed_s=time.time())
        execution.write(args.status, state)
        return await original_run_cells(common, base, cells, out, stop_path, on_update)

    execution.run_cells = priority_first
    return original_run_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--repeat-out', type=Path, required=True)
    parser.add_argument('--repeat-declarations', type=Path, default=ROOT / 'declarations.json')
    parser.add_argument('--poll-s', type=float, default=30)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    for name in ('manifest', 'status', 'repeat_out', 'repeat_declarations'):
        setattr(args, name, getattr(args, name).resolve())
    require(args.repeat_declarations == ROOT / 'declarations.json', 'exact repeat declarations required')
    require(ROOT in args.status.parents and ROOT in args.repeat_out.parents,
            'new watcher and repeat evidence must stay in this task output')
    require(args.status != OLD_WATCHER and 5 <= args.poll_s <= 60, 'watch path/poll interval invalid')
    verify_wrapper(args.manifest, args.manifest_sha256)
    execution, runner, b_runner = original_modules()
    b_runner.verify_package()
    require(not args.status.exists() and not args.repeat_out.exists(), 'new watcher/repeat output required')
    if not args.run:
        print(json.dumps(dict(cpu_only=True, hardware_actions=False, priority_repeats=16,
                              subsequent_original_ablation_cells=18,
                              original_attempt=str(ORIGINAL_ATTEMPT))))
        return
    require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'original supervisor must acquire a fresh lease')
    repeat = load(ROOT / 'repeat.py', 'b_priority_repeats')
    install_priority_hook(execution, runner, repeat, args)
    original_args = argparse.Namespace(model='32b', declarations=ORIGINAL / 'declarations',
                                     status=args.status, attempt=ORIGINAL_ATTEMPT,
                                     watch=True, poll_s=args.poll_s, run=True)
    with (ORIGINAL / 'watcher-32b.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_stopped_waiter()
        asyncio.run(runner.supervise(original_args))


if __name__ == '__main__':
    main()
