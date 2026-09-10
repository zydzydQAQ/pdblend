"""Low-priority evidence mirror and persistent nine-group reporting monitor.

Only status.json snapshots are mutable. Every other downloaded byte is named by
an existing SHA-256 reference and installed without replacing a conflicting file.
The monitor never starts, stops, or modifies an experiment.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time

import contract as c
import report
import setup_energy_ledger as energy_ledger

HERE = Path(__file__).resolve().parent
WORKSPACE = Path('/root/workspace')
HOSTS = {'C': '47.106.163.29', 'A': '120.79.123.62'}
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o',
       'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
INDEX_SCRIPT = r'''
import base64,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); result=[]
for path in root.glob('uniform-*/**/status.json'):
    try:
        raw=path.read_bytes(); value=json.loads(raw)
        # Parsing is the only prerequisite for copying a mutable status snapshot.
        # Immutable references are emitted by the producer only after completion.
        extras=[]
        if (value.get('finished_s') and value.get('complete') is True and
                'setup_and_correctness_energy_j' in value and value.get('measurement_end_s')):
            for name in ('power.csv','power_source.json','power_metadata.jsonl'):
                source=path.parent/'power'/name
                if source.is_file():
                    extras.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        result.append(dict(path=str(path),mtime_ns=path.stat().st_mtime_ns,
                           data=base64.b64encode(raw).decode(),extra_refs=extras))
    except (OSError,ValueError): pass
print(json.dumps(result))
'''
TRANSFER_SCRIPT = r'''
import hashlib,json,os,pathlib,sys,time
requests=json.load(sys.stdin); out=sys.stdout.buffer
limit=int(sys.argv[1]); started=time.monotonic(); sent=0
for ref in requests:
    try:
        path=pathlib.Path(ref['path']); assert path.is_relative_to('/root/workspace')
        assert path.is_file() and path.stat().st_size <= 1024**3
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''): digest.update(block)
        assert digest.hexdigest()==ref['sha256'], 'remote digest mismatch'
        size=path.stat().st_size
    except Exception as exc:
        out.write((json.dumps(dict(path=ref['path'],error=repr(exc)))+'\n').encode());out.flush();continue
    out.write((json.dumps(dict(path=str(path),sha256=ref['sha256'],size=size))+'\n').encode());out.flush()
    with path.open('rb') as stream:
        remaining=size
        while remaining:
            block=stream.read(min(256*1024,remaining))
            if not block: raise RuntimeError('file shortened during transfer')
            out.write(block);out.flush();remaining-=len(block);sent+=len(block)
            delay=sent/limit-(time.monotonic()-started)
            if delay>0: time.sleep(delay)
'''


def remote_argv(host, script, *args):
    words = ['nice', '-n', '15', 'python3', '-B', '-c', script, *map(str, args)]
    return SSH + ['root@' + host, shlex.join(words)]


def safe_path(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(WORKSPACE):
        raise ValueError('reference outside evidence workspace: ' + str(path))
    # A local symlink must not redirect evidence writes outside the workspace.
    if not path.resolve().is_relative_to(WORKSPACE):
        raise ValueError('reference redirects outside evidence workspace: ' + str(path))
    return path


def stat_key(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def references(value):
    """Follow evidence links, excluding mutable code and source inventories."""
    if isinstance(value, list):
        for item in value:
            yield from references(item)
    elif isinstance(value, dict):
        # A declaration is an immutable scheduling snapshot, not a mandate to
        # mirror every obsolete experiment referenced by its provenance chain.
        # Completed checkpoints below carry their own exact trace and artifacts.
        if 'positions' in value and 'cells' in value and 'reused_observations' in value:
            return
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            path = value['path']
            if len(value['sha256']) == 64 and Path(path).suffix not in ('.py', '.sh') and Path(path).name != 'STOP':
                yield dict(path=path, sha256=value['sha256'])
            return
        if isinstance(value.get('artifacts'), dict):
            for path, digest in value['artifacts'].items():
                yield dict(path=path, sha256=digest)
        if isinstance(value.get('receipt'), str) and value.get('receipt_sha256'):
            yield dict(path=value['receipt'], sha256=value['receipt_sha256'])
        if isinstance(value.get('trace'), str) and value.get('trace_sha256'):
            yield dict(path=value['trace'], sha256=value['trace_sha256'])
        for key, item in value.items():
            if key not in ('artifacts', 'files', 'source_files', 'frozen_files',
                           'source_300s_trace', 'executed_source', 'source_manifests'):
                yield from references(item)


class Mirror:
    def __init__(self, cache, *, bandwidth=2*1024**2):
        self.cache_path = Path(cache)
        self.cache = c.read(cache) if self.cache_path.exists() else {}
        self.bandwidth = bandwidth
        self.errors = []
        self.downloaded = 0
        self.bytes = 0
        self.visited = set()

    def issue(self, node, path, error):
        self.errors.append(dict(node=node, path=str(path), error=str(error)))

    def present(self, ref):
        path = safe_path(ref['path'])
        if not path.exists():
            return False
        key = stat_key(path)
        saved = self.cache.get(str(path))
        if saved == dict(sha256=ref['sha256'], stat=key):
            return True
        if c.sha(path) != ref['sha256']:
            raise ValueError('local immutable evidence conflict; existing bytes preserved')
        self.cache[str(path)] = dict(sha256=ref['sha256'], stat=key)
        return True

    def install(self, ref, temporary):
        path = safe_path(ref['path'])
        if c.sha(temporary) != ref['sha256']:
            raise ValueError('downloaded bytes do not match immutable reference')
        if path.exists():
            self.present(ref)
            Path(temporary).unlink()
            return
        # link is exclusive: a concurrent local producer cannot be overwritten.
        try:
            os.link(temporary, path)
        except FileExistsError:
            self.present(ref)
        else:
            self.downloaded += 1
            self.bytes += path.stat().st_size
        Path(temporary).unlink()
        self.cache[str(path)] = dict(sha256=ref['sha256'], stat=stat_key(path))

    def fetch(self, node, refs):
        host = HOSTS[node]
        process = subprocess.Popen(remote_argv(host, TRANSFER_SCRIPT, self.bandwidth),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timer = threading.Timer(max(180, len(refs) * 10), process.kill)
        timer.daemon = True
        timer.start()
        process.stdin.write(json.dumps(refs).encode())
        process.stdin.close()
        try:
            for ref in refs:
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError('remote evidence stream ended early')
                header = json.loads(line)
                if header['path'] != ref['path']:
                    raise ValueError('remote evidence stream identity mismatch')
                if header.get('error'):
                    self.issue(node, ref['path'], header['error'])
                    continue
                remaining = header['size']
                if not isinstance(remaining, int) or not 0 <= remaining <= 1024**3:
                    raise ValueError('invalid transfer size')
                path = safe_path(ref['path'])
                path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(prefix='.evidence-', dir=path.parent)
                try:
                    with os.fdopen(descriptor, 'wb') as stream:
                        while remaining:
                            block = process.stdout.read(min(256*1024, remaining))
                            if not block:
                                raise RuntimeError('remote evidence stream truncated')
                            stream.write(block)
                            remaining -= len(block)
                    self.install(ref, temporary)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            code = process.wait(timeout=15)
            if code:
                raise RuntimeError('remote evidence transfer failed: ' + str(code))
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
            process.stderr.close()

    def hydrate(self, node, initial):
        frontier = list(initial)
        while frontier:
            unique = {}
            for ref in frontier:
                key = (node, ref['path'], ref['sha256'])
                if key not in self.visited:
                    self.visited.add(key)
                    unique[(ref['path'], ref['sha256'])] = ref
            if not unique:
                break
            refs = list(unique.values())
            missing = []
            good = []
            for ref in refs:
                try:
                    (good if self.present(ref) else missing).append(ref)
                except Exception as exc:
                    self.issue(node, ref['path'], exc)
            for start in range(0, len(missing), 48):
                batch = missing[start:start+48]
                try:
                    self.fetch(node, batch)
                except Exception as exc:
                    self.issue(node, 'transfer_batch', exc)
                for ref in batch:
                    try:
                        if self.present(ref):
                            good.append(ref)
                    except Exception as exc:
                        self.issue(node, ref['path'], exc)
            frontier = []
            report.save(self.cache_path, self.cache)
            for ref in good:
                if (Path(ref['path']).suffix == '.json'
                        and not Path(ref['path']).name.endswith(('manifest.json', 'spec.json'))):
                    try:
                        frontier.extend(references(c.read(ref['path'])))
                    except Exception as exc:
                        self.issue(node, ref['path'], exc)
        report.save(self.cache_path, self.cache)


def remote_statuses(node):
    directory = c.ROOT / node
    result = subprocess.run(remote_argv(HOSTS[node], INDEX_SCRIPT, directory),
        capture_output=True, timeout=40, check=True)
    return json.loads(result.stdout)


def mirror_statuses(node, snapshots):
    seeds = []
    for snapshot in snapshots:
        path = safe_path(snapshot['path'])
        expected = c.ROOT / node
        relative = path.relative_to(expected)
        if path.name != 'status.json' or not relative.parts[0].startswith('uniform-'):
            raise ValueError('unexpected mutable status path')
        raw = base64.b64decode(snapshot['data'], validate=True)
        value = json.loads(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != raw:
            descriptor, temporary = tempfile.mkstemp(prefix='.status-', dir=path.parent)
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(raw)
                os.replace(temporary, path)
                os.utime(path, ns=(snapshot['mtime_ns'], snapshot['mtime_ns']))
            finally:
                Path(temporary).unlink(missing_ok=True)
        if snapshot.get('extra_refs'):
            # Legacy cold restoration lacks a receipt reference. Pin its exact
            # terminal status bytes plus independently discovered raw-file SHA.
            digest = hashlib.sha256(raw).hexdigest()
            inventory = dict(schema='uniform-terminal-setup-evidence-v1', node=node,
                status=dict(path=str(path),sha256=digest), artifacts=snapshot['extra_refs'])
            inventory_path = HERE / 'terminal-evidence' / node / (digest + '.json')
            if inventory_path.exists():
                c.need(c.read(inventory_path) == inventory, 'terminal raw evidence changed after freezing')
            else:
                report.save(inventory_path, inventory)
            seeds.extend(snapshot['extra_refs'])
        # These refs are only installed by the runner after the corresponding
        # measurement, audit, or declaration is finalized. Active logs are absent.
        for key in ('declaration', 'plan', 'observations', 'observed_checkpoints',
                    'dynamic_reuse_observations', 'extensions', 'supervisor_terminal',
                    'pipeline_at_pdb_completion', 'last_measurement'):
            seeds.extend(references(value.get(key, [])))
        if value.get('finished_s') and not value.get('node_lease_held'):
            for key in ('measurement', 'setup_measurement'):
                seeds.extend(references(value.get(key, [])))
    return seeds


def signature():
    values = []
    for node in ('A', 'B', 'C'):
        for path in (c.ROOT / node).glob('uniform-*/**/status.json'):
            try:
                state = c.read(path)
                # Heartbeat timestamps alone do not trigger a full evidence scan.
                relevant = {k: state.get(k) for k in ('phase', 'complete', 'pdb_complete', 'scope',
                    'error', 'declaration', 'plan', 'observations', 'observed_checkpoints',
                    'dynamic_reuse_observations', 'finished_s')}
                values.append((str(path), relevant))
            except (OSError, ValueError):
                continue
    return hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest()


def completion_state(result, scope, *, states=None, guards=None, hydration_active=False):
    """Scope completion does not change the scientific five-system result flag."""
    if states is None:
        states = report.latest_states()
    if guards is None:
        guards = []
        for node in ('A', 'B', 'C'):
            for path in (c.ROOT / node).glob('uniform-*/**/status.json'):
                try:
                    state = c.read(path)
                    if state.get('schema') == 'uniform-pdblend-only-scope-completion-v1':
                        if state.get('complete'):
                            c.checked(state['supervisor_terminal'])
                        guards.append(state)
                except (OSError, ValueError, KeyError):
                    continue
    raw_complete = (result['complete'] if scope == 'five_systems' else
        len(result['groups']) == 9 and all(g.get('pdb_boundary_complete') for g in result['groups']))
    terminal = []
    for model in c.MODELS:
        path, state = states.get(model, (None, {}))
        matching = [g for g in guards if path is not None and g.get('pipeline') == str(Path(path).parent)]
        guard_complete = bool(matching) and all(g.get('complete') is True and g.get('pdb_complete') is True
            and g.get('finished_s') and not g.get('error') for g in matching)
        clean_exit = state.get('complete') is True and not state.get('error')
        # C's frozen five-system supervisor is intentionally stopped at its
        # baseline handoff; its separately pinned scope guard certifies that exit.
        valid_terminal = bool(state.get('finished_s') and not state.get('node_lease_held')
            and (clean_exit or scope == 'pdblend' and guard_complete)
            and (not matching or guard_complete))
        terminal.append(dict(model=model, path=str(path) if path else None,
                             pipeline_terminal=valid_terminal, matching_guards=len(matching)))
    return dict(completion_scope=scope, raw_scope_complete=bool(raw_complete),
        scope_complete=bool(raw_complete and all(v['pipeline_terminal'] for v in terminal) and not hydration_active),
        five_system_complete=bool(result['complete']), hydration_active=hydration_active,
        supervisor_checks=terminal)


def run(args):
    os.nice(10)
    try:
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], capture_output=True, check=False)
    except FileNotFoundError:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (HERE / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mirror = Mirror(HERE / 'mirror-cache.json', bandwidth=args.bandwidth)
    declaration = c.ref(args.declaration)
    base = c.load_declaration(declaration)
    history = {'C': list(references([v for v in base['reused_observations'] if v['measurement_host'] == 'C'])), 'A': []}
    previous = None
    last_plot = 0
    iteration = 0
    hydration = ThreadPoolExecutor(max_workers=1)
    future_mirror = None
    pending_seeds = {node: [] for node in HOSTS}
    last_history = 0
    reported_downloads = -1
    def hydrate_pending(seeds, revisit):
        if revisit:
            mirror.visited.clear()
        for node in HOSTS:
            mirror.hydrate(node, seeds[node])
    while True:
        started = time.time()
        iteration += 1
        before = mirror.downloaded
        snapshots = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = {node: executor.submit(remote_statuses, node) for node in HOSTS}
            for node, future in pending.items():
                try:
                    snapshots[node] = future.result()
                except Exception as exc:
                    mirror.issue(node, 'remote_statuses', exc)
        for node in HOSTS:
            try:
                pending_seeds[node].extend(mirror_statuses(node, snapshots.get(node, [])))
            except Exception as exc:
                mirror.issue(node, 'status_mirror', exc)
        # Hydration runs independently, so a large initial archive cannot block
        # the 30-second status heartbeat. Atomic installs make concurrent reads safe.
        if future_mirror is None or future_mirror.done():
            if future_mirror is not None:
                try:
                    future_mirror.result()
                except Exception as exc:
                    mirror.issue('all', 'hydrate', exc)
            revisit = started - last_history >= 300
            if revisit:
                for node in HOSTS:
                    pending_seeds[node].extend(history[node])
                last_history = started
            batch = {node: list({(r['path'],r['sha256']):r for r in refs
                if revisit or (node,r['path'],r['sha256']) not in mirror.visited}.values())
                for node, refs in pending_seeds.items()}
            pending_seeds = {node: [] for node in HOSTS}
            future_mirror = hydration.submit(hydrate_pending, batch, revisit) if any(batch.values()) else None
        current = signature()
        published = False
        due_plot = started - last_plot >= args.plot_interval
        if current != previous or mirror.downloaded != reported_downloads or due_plot:
            published_downloads = mirror.downloaded
            result = report.collect(declaration, args.out)
            rows = report.export(result, args.out)
            setup_result = energy_ledger.collect()
            energy_ledger.export(setup_result, args.out)
            failures = report.collect_engineering_failures(args.out / 'audit-cache', primary_result=result)
            report.export_engineering_failures(failures, args.out)
            reported_downloads = published_downloads
            published = True
            if due_plot or result['complete']:
                report.plot(result, rows, args.out)
                last_plot = time.time()
            previous = current
        else:
            result = c.read(args.out / 'results.json')
        finishing = completion_state(result, args.completion_scope,
            hydration_active=future_mirror is not None and not future_mirror.done())
        if (result.get('completion_scope') != args.completion_scope or
                result.get('scope_complete') != finishing['scope_complete']):
            result.update({k: finishing[k] for k in ('completion_scope', 'scope_complete', 'five_system_complete')})
            rows = report.export(result, args.out)
        if finishing['scope_complete']:
            rows = report.export(result, args.out)
            report.plot(result, rows, args.out)
            last_plot = time.time()
        state = dict(schema='uniform-rate-evidence-monitor-v1', pid=os.getpid(),
            started_s=getattr(run, 'started_s', started), updated_s=time.time(), iteration=iteration,
            complete=result['complete'], groups_complete=sum(g['complete'] for g in result['groups']),
            pdb_boundaries_complete=sum(g.get('pdb_boundary_complete', False) for g in result['groups']),
            observations=len(result['observations']), metric_audit_errors=len(result['metric_audit_errors']),
            downloaded_files=mirror.downloaded, downloaded_bytes=mirror.bytes,
            remote_status_counts={node: len(values) for node, values in snapshots.items()},
            mirror_errors=list(mirror.errors[-200:]), mirror_error_count=len(mirror.errors),
            hydration_active=future_mirror is not None and not future_mirror.done(),
            report_updated=published, last_plot_s=last_plot,
            engineering_failure_statuses=len(failures['entries']),
            setup_energy_entries=len(setup_result['entries']),
            setup_energy_pending=len(setup_result['pending_or_invalid_evidence']),
            out=str(args.out), remote_hosts=HOSTS,
            completion_scope=args.completion_scope, scope_complete=finishing['scope_complete'],
            five_system_complete=result['complete'], completion_checks=finishing,
            stop_rule='nine groups in requested scope pass raw audit; supervisors/guards terminal; hydration complete')
        if finishing['scope_complete'] or args.once:
            state['finished_s'] = time.time()
        run.started_s = state['started_s']
        report.save(HERE / 'monitor-status.json', state)
        print(json.dumps({k: state[k] for k in ('updated_s', 'iteration', 'complete', 'groups_complete',
            'observations', 'metric_audit_errors', 'downloaded_files', 'remote_status_counts')}), flush=True)
        if args.once and future_mirror is not None:
            future_mirror.result()
            result = report.collect(declaration, args.out)
            report.export(result, args.out)
            energy_ledger.export(energy_ledger.collect(), args.out)
        if finishing['scope_complete'] or args.once:
            hydration.shutdown(wait=True)
            break
        time.sleep(max(1, args.interval - (time.time() - started)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declaration', type=Path, default=HERE / 'release-001/declaration.json')
    parser.add_argument('--out', type=Path, default=HERE / 'reports/current')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--plot-interval', type=float, default=300)
    parser.add_argument('--bandwidth', type=int, default=2*1024**2)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--completion-scope', choices=('five_systems', 'pdblend'), default='five_systems')
    arguments = parser.parse_args()
    if arguments.interval <= 0 or arguments.plot_interval <= 0 or arguments.bandwidth <= 0:
        parser.error('intervals and bandwidth must be positive')
    run(arguments)
