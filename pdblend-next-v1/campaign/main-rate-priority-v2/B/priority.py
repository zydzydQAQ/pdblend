"""B cell-boundary pause, original-policy repeats, and measured baseline return.

The running one-cell measurement is never signalled. Original source, queue
status and observations remain immutable to this coordinator. A separate
versioned continuation consumes baseline-ready.json after this process exits.
"""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
C = ROOT.parents[1]
V1 = C / 'main-rate-rerun-v1'
ABL = C / 'pdblend-ablation-20260908-v1'
SCALE = C / 'B32B-main-to-scale-handoff-v2/attempt-001/scale'
HANDOFF = SCALE.parent
SPEC = HANDOFF / 'scale-bindings/spec.json'
RELEASE = HANDOFF / 'B32B-model-release.json'
SPEC_SHA = '69507c8e3631af5dea3129a9dde223c4c2968af1676b3b6040097cb628da9975'
RELEASE_SHA = '3e99003eaf4c4074f00bc4e88055f6b4a4c6316e5f52f03582d0341b0363599e'
IDENTITIES = {667430: (12021456, V1 / 'B_watch.py'),
              584282: (11546581, C / 'scale-only-continuation-B32B-v1/supervise.py'),
              523564: (11328850, C / 'B32B-main-to-scale-handoff-v2/handoff.py')}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as handle:
            handle.write(text)
    else:
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(text)
        tmp.replace(path)


def process(pid):
    try:
        p = Path('/proc', str(pid))
        stat = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        args = [x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
        return dict(pid=pid, startticks=int(stat[19]), argv=args, state=stat[0], alive=stat[0] != 'Z')
    except (OSError, ValueError):
        return None


def exact(pid):
    current = process(pid)
    ticks, source = IDENTITIES[pid]
    require(current and current['alive'] and current['startticks'] == ticks
            and str(source) in current['argv'], 'process identity changed: ' + str(pid))
    return current


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(manifest, expected):
    require(sha(manifest) == expected, 'priority package manifest changed')
    value = read(manifest)
    require(value['kind'] == 'B-original-main-repeats-priority-v2', 'wrong package')
    for path, digest in value['files'].items():
        require(sha(path) == digest, 'priority dependency changed: ' + path)
    require(sha(SPEC) == SPEC_SHA and sha(RELEASE) == RELEASE_SHA, 'original scale spec/release changed')
    return value


def no_children(pid):
    for item in Path('/proc', str(pid), 'task').glob('*/children'):
        require(not item.read_text().strip(), 'waiting watcher acquired a child')


def initial_check():
    require(socket.gethostname() == 'iZwz9i5bte3xkpmcoes3t2Z', 'actual B host required')
    waiting = read(V1 / 'B-watch-status.json')
    require(waiting['pid'] == 667430 and waiting['phase'] == 'waiting_for_baselines'
            and waiting.get('hardware_started') is False, 'old waiter no longer waiting before GPU')
    require(not (V1 / 'B-repeat-001').exists() and not (ABL / 'attempts/32b-001').exists(),
            'old waiter already started its work')
    require(not (ABL / 'B32B-baseline-completion.gate.json').exists(), 'baseline publication already exists')
    pids = {pid: exact(pid) for pid in IDENTITIES}
    no_children(667430)
    scale = read(SCALE / 'status.json')
    handoff = read(HANDOFF / 'status.json')
    require(scale['pid'] == 584282 and scale['phase'] == 'scale_only' and not scale['complete'], 'scale already terminal')
    require(handoff['pid'] == 523564 and handoff['phase'] == 'running_scale' and handoff['scale_pid'] == 584282,
            'handoff ownership differs')
    require(scale['steps'] and not scale['steps'][-1]['complete'], 'need an identifiable running one-cell driver')
    current = scale['steps'][-1]
    require('--max-cells' in current['argv'] and current['argv'][current['argv'].index('--max-cells') + 1] == '1',
            'driver is not one-cell bounded')
    child = process(current['pid'])
    require(child and child['alive'] and child['argv'] == current['argv'], 'current child identity differs')
    return dict(observed_s=time.time(), processes=pids, old_waiter=waiting, scale_status=scale,
                handoff_status=handoff, current_driver=child, source_spec=ref(SPEC), source_release=ref(RELEASE),
                gpu_actions=False, direct_child_signals=False)


def stop_waiter(out):
    before = initial_check()
    write(out / 'pause-before.json', before, True)
    initial_check()
    os.kill(667430, signal.SIGTERM)
    until = time.monotonic() + 15
    while process(667430) and process(667430)['alive']:
        require(time.monotonic() < until, 'waiting watcher exit not confirmed')
        time.sleep(.1)
    old = read(V1 / 'B-watch-status.json')
    require(old.get('hardware_started') is False and not (V1 / 'B-repeat-001').exists(),
            'old watcher crossed into GPU work')
    write(out / 'waiting-watcher-stopped.json', dict(observed_s=time.time(), old_status=ref(V1 / 'B-watch-status.json'),
          process_exited=True, hardware_started=False, before=ref(out / 'pause-before.json')), True)


def boundary_pause(out, update):
    # Stop the actual supervisor first, eliminating its next-cell launch path.
    # Both handlers set boundary flags; neither forwards a signal to its cell.
    supervisor = exact(584282)
    handoff = exact(523564)
    issued = dict(issued_s=time.time(), supervisor=supervisor, handoff=handoff,
                  current_step=read(SCALE / 'status.json')['steps'][-1], direct_child_signals=False)
    write(out / 'boundary-stop-intent.json', issued, True)
    os.kill(584282, signal.SIGTERM)
    os.kill(523564, signal.SIGTERM)
    until = time.monotonic() + 550
    update(phase='waiting_current_cell_boundary', boundary_stop=ref(out / 'boundary-stop-intent.json'))
    while any(process(pid) and process(pid)['alive'] for pid in (584282, 523564)):
        require(time.monotonic() < until, 'boundary parent exit not confirmed; no serving successor')
        time.sleep(.5)
    state = read(SCALE / 'status.json')
    parent = read(HANDOFF / 'status.json')
    require(state['phase'] == 'stopped' and state.get('error') == "RuntimeError('scale STOP: no successor')",
            'scale was not a clean intentional boundary stop')
    require(parent['phase'] == 'stopped' and parent.get('scale_exitcode') != 0,
            'handoff stop attribution differs')
    for step in state['steps']:
        require(step.get('complete') and step.get('exitcode') == 0 and step.get('verified_new_checkpoint')
                and not step.get('unconfirmed_child') and not step.get('deadline_interrupt_s'),
                'scale prefix has an unverified/failed child')
        require(not (process(step['pid']) or {}).get('alive'), 'scale child is still alive')
    wrapper = load(V1 / 'B_watch.py', 'priority_old_b_wrapper')
    execution, runner, _ = wrapper.original_modules()
    contract = load(C / 'scale-only-continuation-B32B-v1/contract.py', 'priority_B_scale_contract')
    checked = contract.check_spec(read(SPEC), RELEASE, RELEASE_SHA)
    groups = checked['groups']
    require(all(g['group']['system'] == 'ecoserve' or not g['pending'] for g in groups),
            'unexpected non-Eco pending work')
    eco = next(g for g in groups if g['group']['system'] == 'ecoserve')
    proof = dict(schema='B-clean-stopped-scale-prefix-v2', observed_s=time.time(),
        complete_prefix=True, whole_baseline192_complete=False, original_spec=ref(SPEC), original_release=ref(RELEASE),
        scale_status=ref(SCALE / 'status.json'), handoff_status=ref(HANDOFF / 'status.json'),
        boundary_intent=ref(out / 'boundary-stop-intent.json'), all_children_clean=True,
        groups=[dict(system=g['group']['system'], binding=g['group'].get('scale_binding'),
                     completed=g['reused'], pending_ids=[r['cell_id'] for r in g['pending']]) for g in groups],
        previous_baseline_binding=eco['group']['scale_binding'])
    write(out / 'prefix-proof.json', proof, True)
    return execution, runner, eco['current'], proof


async def experiments(execution, runner, previous, proof, out, update, check_sources):
    current_task = asyncio.current_task()
    interrupted = False
    def stop():
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            current_task.cancel()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    repeat = load(V1 / 'repeat.py', 'priority_frozen_repeats')
    cells = repeat.verify_cells(V1 / 'declarations.json', '32b')
    original = read(cells[0]['source_binding']['path'])
    common = execution.load_common(original['host_release'])
    from ecopadg.serving.campaign import node_lease
    with node_lease():
        require(not any((process(p) or {}).get('alive') for p in IDENTITIES), 'old queue process returned')
        check_sources()
        update(phase='restoring_original_pdblend', hardware_started=True)
        fresh_pdb = await execution.restore_core(common, original, previous, out / 'restore-pdb')
        update(phase='priority_original_policy_repeats', restored_binding=ref(out / 'restore-pdb/binding.json'),
               repeat_output=str(out / 'repeats'))
        result = await repeat.run_repeats(common, fresh_pdb, '32b', out / 'repeats', V1 / 'declarations.json',
                                         on_update=lambda s: update(repeats=s))
        require(result['complete'], 'new16 not complete; no automatic restart or baseline successor')
        check_sources()
        update(phase='restoring_original_baseline', repeats_complete=True)
        fresh_baseline = await execution.restore_core(common, previous, fresh_pdb, out / 'restore-baseline')
        check_sources()
        runtime_dirs = {read(i['engine_config'])['runtime_dir'] for i in previous['instances']}
        require(len(runtime_dirs) == 1, 'baseline engine runtime directories differ')
        ready = dict(schema='B-measured-baseline-return-ready-v2', ready=True, observed_s=time.time(),
            original_baseline_binding=proof['previous_baseline_binding'],
            fresh_restoration_binding=ref(out / 'restore-baseline/binding.json'),
            fresh_inventory=ref(out / 'restore-baseline/containers.after.json'),
            measured_restore_status=ref(out / 'restore-baseline/status.json'), prefix_proof=ref(out / 'prefix-proof.json'),
            repeats_status=ref(out / 'repeats/status.json'), coordinator_pid=os.getpid(),
            runtime_dir=next(iter(runtime_dirs)),
            fresh_qualification_still_required=True, original_baseline192_complete=False,
            remaining_ids=next(g['pending_ids'] for g in proof['groups'] if g['system'] == 'ecoserve'))
        write(out / 'baseline-ready.json', ready, True)
        update(phase='baseline_return_ready', hardware_complete=True, baseline_ready=ref(out / 'baseline-ready.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    check = lambda: verify(args.manifest, args.manifest_sha256)
    check()
    require(not os.environ.get('PDBLEND_NODE_LOCK_FD'), 'fresh node lease required')
    initial_check()
    if not args.run:
        print(json.dumps(dict(cpu_only=True, ready_for_boundary_pause=True, hardware_actions=False)))
        return
    require(not args.out.exists(), 'new attempt required; no retry or overwrite')
    args.out.mkdir(parents=True)
    state = dict(schema='B-priority-stage-v2', pid=os.getpid(), started_s=time.time(), phase='starting',
                 hardware_started=False, complete=False, automatic_retries=False,
                 sequence='current cell -> new16 -> original baseline remainder -> original18')
    def update(**items):
        state.update(items, updated_s=time.time())
        write(args.out / 'status.json', state)
    update()
    try:
        stop_waiter(args.out)
        with (ABL / 'watcher-32b.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execution, runner, previous, proof = boundary_pause(args.out, update)
            check()
            asyncio.run(experiments(execution, runner, previous, proof, args.out, update, check))
        update(phase='handoff_ready', complete=True, continuation_complete=False)
    except BaseException as exc:
        update(phase='failed', error=repr(exc), needs_attention=True)
        raise
    finally:
        update(finished_s=time.time())


if __name__ == '__main__':
    main()
