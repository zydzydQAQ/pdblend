"""Observe actual B scale parents and publish baseline192 only after deep proof.

This process never owns a GPU lease and writes only beneath this package. It
stops publishing at 18:00 Beijing; elapsed time can never satisfy the gate.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True
import readiness_b32 as gate

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
CUTOFF = 1788861600
SCRIPT = 'scale-only-continuation-B32B-v1/supervise.py'
MANIFEST = 'publisher-manifest-b32.json'


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def inside(path, root):
    path = Path(path).resolve()
    gate.require(root.resolve() in path.parents, 'path must be inside ' + str(root))
    return path


def immutable(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        gate.require(path.read_bytes() == data, 'immutable evidence differs: ' + str(path))


def verify_package(package=ROOT):
    value = gate.shared.read(package / MANIFEST)
    gate.require(value['kind'] == 'B32B-readonly-gate-publisher-v1', 'wrong publisher manifest')
    required = {'publish_b32_gate.py', 'readiness_b32.py', 'readiness.py', 'execution-manifest-b32.json'}
    gate.require(required <= set(value['files']), 'publisher source coverage incomplete')
    for name, digest in value['files'].items():
        gate.require(gate.shared.sha(package / name) == digest, 'publisher source changed: ' + name)
    return value


def parse_parent(args, *, pid, cwd, campaign=CAMPAIGN):
    """Only the exact qualified B supervisor with --run is discoverable."""
    if not args or 'python' not in Path(args[0]).name or any('\n' in a for a in args):
        return None
    script = (campaign / SCRIPT).resolve()
    positions = [i for i, arg in enumerate(args[1:], 1)
                 if arg.endswith('.py') and (Path(arg) if Path(arg).is_absolute()
                    else Path(cwd) / arg).resolve() == script]
    if len(positions) != 1:
        return None
    tail = args[positions[0] + 1:]
    if '--run' not in tail:
        return None
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    for flag in ('spec', 'spec-sha256', 'release', 'release-sha256', 'out'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--group', action='append', default=[])
    parser.add_argument('--run', action='store_true')
    for flag in ('--spec', '--spec-sha256', '--release', '--release-sha256', '--out', '--run'):
        gate.require(sum(a == flag or a.startswith(flag + '=') for a in tail) == 1,
                     'duplicate/missing parent argument: ' + flag)
    try:
        parsed = parser.parse_args(tail)
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ValueError('invalid actual B parent argv') from exc
    def absolute(value):
        path = Path(value)
        return inside(path if path.is_absolute() else Path(cwd) / path, campaign)
    spec = dict(path=str(absolute(parsed.spec)), sha256=parsed.spec_sha256)
    release = dict(path=str(absolute(parsed.release)), sha256=parsed.release_sha256)
    value = gate.read_ref(spec, campaign)
    gate.read_ref(release, campaign)
    gate.require(value.get('model') == '32b' and value.get('release') == release,
                 'actual B spec model/release differs')
    gate.require(type(pid) is int and pid > 0, 'actual process pid required')
    return dict(schema=1, pid=pid, spec=spec, release=release,
                out=str(absolute(parsed.out)), groups=parsed.group, argv=args)


def discover(*, package=ROOT, campaign=CAMPAIGN, proc_root=Path('/proc')):
    discovered = []
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            args = [a.decode(errors='replace') for a in path.read_bytes().split(b'\0') if a]
            # Resolve cwd only for an otherwise plausible exact supervisor.
            if not any(Path(a).name == 'supervise.py' for a in args):
                continue
            cwd = (path.parent / 'cwd').resolve(strict=True)
            parent = parse_parent(args, pid=int(path.parent.name), cwd=cwd, campaign=campaign)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if parent is None:
            continue
        data = encoded(parent)
        key = hashlib.sha256(data).hexdigest()
        immutable(package / 'discovery' / ('parent-' + key + '.json'), data)
        discovered.append(parent)
    return discovered


def candidate_from_discovery(*, package=ROOT, campaign=CAMPAIGN, proc_root=Path('/proc')):
    paths = sorted((package / 'discovery').glob('parent-*.json'))
    gate.require(paths, 'waiting for an actual qualified B scale --run parent; no completion inferred')
    proofs, representatives, release = [], {}, None
    for path in paths:
        parent = gate.shared.read(path)
        gate.require(path.name == 'parent-' + hashlib.sha256(encoded(parent)).hexdigest() + '.json',
                     'discovery record changed: ' + str(path))
        gate.read_ref(parent['spec'], campaign)
        gate.read_ref(parent['release'], campaign)
        gate.require(release is None or release == parent['release'], 'discovered parents use different releases')
        release = parent['release']
        status = inside(Path(parent['out']) / 'status.json', campaign)
        state = gate.shared.read(status)
        gate.require(state.get('pid') == parent['pid'], 'parent status does not belong to observed process')
        proof = gate.shared.terminal_parent(Path(parent['spec']['path']), status, release['sha256'], proc_root)
        proofs.append(dict(discovery_path=str(path), discovery_sha256=gate.shared.sha(path), terminal=proof))
        # All parents above must terminate, even when groups share one full spec.
        key = (parent['spec']['path'], parent['spec']['sha256'])
        representatives.setdefault(key, dict(spec=parent['spec'], status=dict(path=str(status), sha256=proof['sha256'])))
    gate.require(not gate.baseline_processes(proc_root), 'baseline driver/automatic successor remains alive')
    value = dict(schema=1, kind=gate.KIND, model='32b', hostname=gate.HOSTNAME,
                 baseline_systems=list(gate.shared.BASELINES), release=release,
                 scale_parents=list(representatives.values()))
    return value, proofs


def publish_once(*, package=ROOT, campaign=CAMPAIGN, proc_root=Path('/proc'),
                 checker=gate.inspect_readiness, now=time.time):
    result = dict(observed_s=now(), published=False, hardware_actions=False)
    gate.require(now() < CUTOFF, '18:00 cutoff reached; publication stopped')
    target = package / gate.PUBLICATION
    if target.exists():
        checked = checker(package=package, campaign=campaign, proc_root=proc_root)
        return dict(result, already_published=True, ready=checked['ready'], gate=checked)
    discover(package=package, campaign=campaign, proc_root=proc_root)
    value, proofs = candidate_from_discovery(package=package, campaign=campaign, proc_root=proc_root)
    data = encoded(value)
    manifest = (package / 'execution-manifest-b32.json').read_bytes()
    key = hashlib.sha256(data + manifest).hexdigest()
    candidate = inside(package / 'gate-candidates' / key, package)
    immutable(candidate / gate.PUBLICATION, data)
    immutable(candidate / 'execution-manifest-b32.json', manifest)
    checked = checker(package=candidate, campaign=campaign, proc_root=proc_root)
    result.update(ready=checked['ready'], gate=checked, candidate=str(candidate), all_parent_proofs=proofs)
    if not checked['ready']:
        return result
    # Recheck every observed parent, not merely the representative per spec.
    discover(package=package, campaign=campaign, proc_root=proc_root)
    after, after_proofs = candidate_from_discovery(package=package, campaign=campaign, proc_root=proc_root)
    gate.require(after == value and after_proofs == proofs, 'parent/discovery evidence changed during deep verification')
    gate.require(now() < CUTOFF, '18:00 cutoff reached during deep verification')
    immutable(candidate / 'publisher-proof.json', encoded(result))
    try:
        os.link(candidate / gate.PUBLICATION, target)
    except FileExistsError:
        gate.require(target.read_bytes() == data, 'another immutable gate already exists')
    result.update(published=True, publication_path=str(target), publication_sha256=gate.shared.sha(target))
    return result


def save_status(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-s', type=float, default=15)
    parser.add_argument('--status', type=Path, default=ROOT / 'publisher-status.json')
    args = parser.parse_args()
    gate.require(socket.gethostname() == gate.HOSTNAME, 'publisher must observe the actual B host')
    gate.require(5 <= args.poll_s <= 60, 'poll interval must be 5..60 seconds')
    status = inside(args.status, ROOT)
    verify_package()
    with (ROOT / 'publisher-b32.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                value = publish_once()
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, AssertionError, ImportError) as exc:
                value = dict(observed_s=time.time(), published=False, ready=False,
                             hardware_actions=False, reasons=[str(exc)])
            save_status(status, value)
            if value.get('published') or value.get('already_published') or not args.watch or time.time() >= CUTOFF:
                break
            time.sleep(min(args.poll_s, max(0, CUTOFF - time.time())))


if __name__ == '__main__':
    main()
