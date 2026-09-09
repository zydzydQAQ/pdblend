"""Durable B tail: fresh legacy gate, Eco suffix, then original core18.

Every GPU stage is a separate process with a fresh original node lease.
The parent only watches immutable handoff evidence and child status.
"""
import argparse
import asyncio
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parents[1]
ABLATION = CAMPAIGN / 'pdblend-ablation-20260908-v1'
sys.path.insert(0, str(ROOT))
import scale_chain as history

read, ref, fixed, write, require, sha = history.read, history.ref, history.fixed, history.write, history.require, history.sha
ORIGINAL_DEADLINE = 1788872770.0400891


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def process_alive(pid):
    try:
        stat = Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()
        return stat[0] != 'Z'
    except OSError:
        return False


def verify_package(args):
    manifest = fixed(dict(path=str(args.manifest), sha256=args.manifest_sha256))
    require(manifest['kind'] == 'B-priority-tail-v2', 'wrong continuation package')
    for path, digest in manifest['files'].items():
        require(sha(path) == digest, 'continuation source changed: ' + path)


def ready_evidence(path):
    ready = read(path)
    require(ready['schema'] == 'B-measured-baseline-return-ready-v2' and ready['ready'] is True
            and ready['fresh_qualification_still_required'] is True, 'measured return handoff required')
    require(not process_alive(ready['coordinator_pid']), 'return coordinator must release and exit')
    for key in ('original_baseline_binding', 'fresh_restoration_binding', 'fresh_inventory',
                'measured_restore_status', 'prefix_proof', 'repeats_status'):
        fixed(ready[key])
    repeats = fixed(ready['repeats_status'])
    require(repeats['complete'] and len(repeats['completed']) == 16 and not repeats['failed'],
            'priority sixteen not completed')
    restore = fixed(ready['measured_restore_status'])
    require(restore['complete'] and restore['clock_restore_complete'] and not restore['errors']
            and not restore.get('sampling_error') and restore['power_evidence']['power_source_verified'],
            'measured baseline restore/cleanup invalid')
    return ready


def prepare_bootstrap(ready, out):
    original = fixed(ready['original_baseline_binding'])
    restored = fixed(ready['fresh_restoration_binding'])
    require(original['deadline_s'] == ORIGINAL_DEADLINE and restored['model'] == '32b'
            and restored['system'] == 'ecoserve', 'wrong original B baseline')
    value = copy.deepcopy(restored)
    value.update(deadline_s=original['deadline_s'], configs={}, output=str(out / 'future-gate-work'),
                 output_correctness_verified=False, correctness_gate_required_before_performance=True,
                 old_correctness_is_historical_only=True, formal_eligible=False,
                 identity_file=ready['fresh_inventory']['path'],
                 experiment_scope='measured original baseline restart; fresh original27 qualification required',
                 restored_priority=dict(binding=ready['fresh_restoration_binding'],
                                        status=ready['measured_restore_status'], inventory=ready['fresh_inventory']))
    for name in ('mechanism_proof', 'correctness_evidence', 'fresh_ablation_binding', 'ablation'):
        value.pop(name, None)
    for source in (ready['fresh_restoration_binding'], ready['measured_restore_status'], ready['fresh_inventory']):
        value['files'][source['path']] = source['sha256']
    for p in (ROOT / 'continue.py', ROOT / 'qualify_restored.py', ROOT / 'scale_chain.py'):
        value['files'][str(p)] = sha(p)
    path = out / 'fresh-bootstrap.json'
    write(path, value, True)
    return ref(path)


def source_common(binding):
    host = Path(binding['host_release'])
    sys.path[:0] = [str(host / 'src'), str(host), '/root/workspace/pdblend/.runtime-deps']
    os.environ['PYTHONPATH'] = ':'.join(sys.path[:3])
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    module = load(CAMPAIGN / 'five-system-execution-v3/run.py', 'priority_original_scale_executor')
    require(module.GLOBAL_DEADLINE == ORIGINAL_DEADLINE, 'original scale deadline changed')
    return module


def suffix_stage(args):
    verify_package(args)
    ready = ready_evidence(args.ready)
    qualifier = load(ROOT / 'qualify_restored.py', 'priority_fresh_eco_qualifier')
    binding_ref = read(args.out / 'qualified-binding-ref.json')
    binding = fixed(binding_ref)
    common = source_common(binding)
    from ecopadg.serving.campaign import node_lease
    require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'suffix needs its own fresh lease')
    with node_lease():
        result = asyncio.run(history.run_scale_suffix(common, binding_ref, ready['prefix_proof'],
                              args.out / 'scale-suffix', qualifier))
        require(result['complete'], 'suffix incomplete')
        proof = history.aggregate192(ready['prefix_proof'], binding_ref,
                    ref(args.out / 'scale-suffix/status.json'), qualifier)
        write(args.out / 'baseline192-proof.json', proof, True)


def ablation_stage(args):
    verify_package(args)
    ready = ready_evidence(args.ready)
    qualifier = load(ROOT / 'qualify_restored.py', 'priority_ablation_eco_qualifier')
    fresh_ref = read(args.out / 'qualified-binding-ref.json')
    wrapper = load(CAMPAIGN / 'main-rate-rerun-v1/B_watch.py', 'priority_original_b_ablation_modules')
    execution, runner, b_runner = wrapper.original_modules()
    b_runner.verify_package()
    cells = runner.selected_cells(ABLATION / 'declarations', '32b')
    require(len(cells) == 18, 'original eighteen declaration changed')
    for cell in cells:
        execution.check_cell(cell)
    parent = read(cells[0]['source_binding']['path'])
    common = execution.load_common(parent['host_release'])
    from ecopadg.serving.campaign import node_lease
    require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'ablation needs its own fresh lease')
    attempt = args.out / 'core18'
    require(not attempt.exists(), 'new original18 attempt required')

    async def perform():
        task = asyncio.current_task()
        interrupted = False
        def stop():
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                task.cancel()  # Frozen restoration/run_one executes bounded cleanup.
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, stop)
        proof = history.aggregate192(ready['prefix_proof'], fresh_ref,
                    ref(args.out / 'scale-suffix/status.json'), qualifier)
        require(not process_alive(read(args.out / 'scale-suffix/status.json')['pid']),
                'scale suffix process still exists')
        require(not (ROOT / 'STOP').exists() and not (ABLATION / 'STOP').exists(), 'STOP blocks original18')
        attempt.mkdir(parents=True)
        write(attempt / 'baseline-readiness.json', proof, True)
        write(attempt / 'declaration-order.json', cells, True)
        previous = fixed(fresh_ref)
        restored = await execution.restore_core(common, parent, previous, attempt / 'restoration')
        def update(state):
            write(attempt / 'status.json', dict(state, model='32b', phase='core_ablation',
                  pid=os.getpid(), append_only_baseline192_verified=True))
        result = await execution.run_cells(common, restored, cells, attempt, ABLATION / 'STOP', update)
        write(attempt / 'status.json', dict(result, model='32b', pid=os.getpid(),
              phase='core_complete' if result['complete'] else 'core_stopped', finished_s=time.time()))
        require(result['complete'], 'original18 stopped or failed; all attempts retained')

    with (ABLATION / 'watcher-32b.lock').open('a') as watcher:
        fcntl.flock(watcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with node_lease():
            asyncio.run(perform())


def supervise(args):
    require(socket.gethostname() == 'iZwz9i5bte3xkpmcoes3t2Z', 'actual B host required')
    require(not args.out.exists(), 'new tail output required; no automatic resume/retry')
    args.out.mkdir(parents=True)
    state = dict(schema='B-priority-tail-status-v2', pid=os.getpid(), started_s=time.time(),
                 phase='waiting_baseline_return', complete=False, child_steps=[], hardware_actions_in_parent=False)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    def update(**value):
        state.update(value, updated_s=time.time())
        write(args.out / 'status.json', state)
    def boundary():
        require(not stopped and not (ROOT / 'STOP').exists(), 'tail STOP before next GPU stage')
        require(time.time() + 520 < ORIGINAL_DEADLINE, 'original deadline reserve exhausted')
        verify_package(args)
    def command(phase, argv, max_seconds, permit_gate_exact_failure=False):
        boundary()
        entry = dict(phase=phase, argv=argv, started_s=time.time(), complete=False)
        state['child_steps'].append(entry)
        update(phase=phase)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with (args.out / (phase + '.log')).open('xb') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True, close_fds=True, env=env)
            entry['pid'] = child.pid
            update()
            while child.poll() is None:
                # GPU children own bounded work/cleanup. Parent never interrupts
                # a normal measurement simply because its queue was paused.
                require(time.time() - entry['started_s'] < max_seconds,
                        'child exit unconfirmed; stop all successors and inspect its lease')
                time.sleep(2)
        entry.update(exitcode=child.returncode, finished_s=time.time(), complete=True)
        update()
        if not permit_gate_exact_failure:
            require(child.returncode == 0, phase + ' failed; no automatic retry')
        return child.returncode
    update()
    try:
        while not args.ready.exists():
            boundary()
            predecessor = args.ready.parent / 'status.json'
            if predecessor.exists():
                prior = read(predecessor)
                require(prior.get('phase') != 'failed', 'priority stage failed before baseline return')
                state['priority_stage'] = {k:prior.get(k) for k in ('pid', 'phase', 'updated_s')}
            update()
            time.sleep(10)
        candidate = read(args.ready)
        while process_alive(candidate['coordinator_pid']):
            boundary()
            time.sleep(1)
        ready = ready_evidence(args.ready)
        write(args.out / 'handoff.json', dict(reference=ref(args.ready), evidence=ready), True)
        bootstrap = prepare_bootstrap(ready, args.out)
        validator = CAMPAIGN / 'B32B-legacy-baseline-correctness-v1/validate.py'
        gate = args.out / 'original27-gate'
        argv = [sys.executable, '-B', '-u', str(validator), '--binding', bootstrap['path'],
                '--runtime-dir', ready['runtime_dir'], '--out', str(gate), '--run']
        code = command('fresh_original27_gate', argv, 650, permit_gate_exact_failure=True)
        gate_status = read(gate / 'status.json')
        require(gate_status['complete'] and gate_status['measurement_valid']
                and gate_status['native_cleanup_complete'] and gate_status['clock_restore_complete']
                and not gate_status['cleanup_errors'], 'fresh gate measurement/cleanup invalid')
        update(fresh_gate_exitcode=code, fresh_gate_raw_passed=gate_status['passed'])
        boundary()
        qualifier = load(ROOT / 'qualify_restored.py', 'priority_tail_eco_qualifier')
        binding_ref = qualifier.qualify(bootstrap, gate, ready['original_baseline_binding'],
                                       ready['fresh_inventory'], args.out / 'qualification')
        require(qualifier.audit_binding(binding_ref)['passed'] is True, 'fresh raw native qualification failed')
        write(args.out / 'qualified-binding-ref.json', binding_ref, True)
        stage_argv = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--manifest', str(args.manifest),
                      '--manifest-sha256', args.manifest_sha256, '--ready', str(args.ready), '--out', str(args.out), '--run']
        command('scale_suffix', stage_argv + ['--stage', 'suffix'], max(600, ORIGINAL_DEADLINE - time.time() + 30))
        command('original_core18', stage_argv + ['--stage', 'ablation'], max(600, ORIGINAL_DEADLINE - time.time() + 30))
        update(phase='complete', complete=True)
    except BaseException as exc:
        update(phase='needs_attention', error=repr(exc))
        raise
    finally:
        update(finished_s=time.time())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--ready', type=Path, default=ROOT / 'attempt-001/baseline-ready.json')
    p.add_argument('--out', type=Path, default=ROOT / 'continuation-001')
    p.add_argument('--stage', choices=('watch', 'suffix', 'ablation'), default='watch')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    for name in ('manifest', 'ready', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    verify_package(args)
    require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'tail does not inherit a node lease')
    if not args.run:
        print(json.dumps(dict(cpu_only=True, hardware_actions=False, priority='new16 then old scale suffix then core18')))
        return
    if args.stage == 'suffix':
        suffix_stage(args)
    elif args.stage == 'ablation':
        ablation_stage(args)
    else:
        with (ROOT / 'tail-watch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            supervise(args)


if __name__ == '__main__':
    main()
