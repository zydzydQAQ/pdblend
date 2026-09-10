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
import setup_energy_ledger_v2 as energy_ledger

HERE = Path(__file__).resolve().parent
WORKSPACE = Path('/root/workspace')
HOSTS = {'C': '47.106.163.29', 'A': '120.79.123.62'}
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o',
       'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
INDEX_SCRIPT = r'''
import base64,hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]); cached=json.load(sys.stdin); result=[]
def identity(s):
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,ino=s.st_ino)
for path in list(root.glob('uniform-rate-20260909-v2/**/status.json')) + list(root.glob('uniform-rate-20260909-v2/**/metadata.json')):
    try:
        first=path.stat(); current=identity(first)
        if cached.get(str(path))==current:
            result.append(dict(path=str(path),remote_stat=current,unchanged=True))
            continue
        with path.open('rb') as stream:
            before=os.fstat(stream.fileno()); raw=stream.read(); after=os.fstat(stream.fileno())
            named=path.stat()
            if (identity(before)!=identity(after) or identity(after)!=identity(named)
                    or before.st_dev!=after.st_dev or after.st_dev!=named.st_dev):
                raise ValueError('status changed during snapshot read')
            # Filesystem timestamps can coalesce equal-size in-place writes.
            # Only full snapshots pay for this second pass; deltas remain stat-only.
            stream.seek(0); digest=hashlib.sha256()
            for block in iter(lambda:stream.read(1024**2),b''): digest.update(block)
            final=os.fstat(stream.fileno()); named=path.stat()
            if (digest.hexdigest()!=hashlib.sha256(raw).hexdigest()
                    or identity(after)!=identity(final) or identity(final)!=identity(named)
                    or after.st_dev!=final.st_dev or final.st_dev!=named.st_dev):
                raise ValueError('status changed during snapshot verification')
        value=json.loads(raw); extras=[]
        completed_setup = (value.get('complete') is True and
            ('setup_and_correctness_energy_j' in value or 'full_operation_energy_j' in value))
        failed_clean_window = ('full_operation_energy_j' in value and value.get('measurement_valid') is True
            and value.get('node_lease_held') is False and value.get('clock_restore_complete') is True
            and not value.get('cleanup_errors') and not value.get('sampling_error'))
        # These are energy-only references; a failed qualification stays failed.
        if (value.get('finished_s') and value.get('measurement_end_s')
                and (completed_setup or failed_clean_window)):
            for name in ('power.csv','power_source.json','power_metadata.jsonl'):
                source=path.parent/'power'/name
                if source.is_file():
                    extras.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        result.append(dict(path=str(path),mtime_ns=after.st_mtime_ns,remote_stat=identity(after),
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
        # Global report/audit inventories include several physical hosts. Their
        # individual checkpoint seeds are routed explicitly below.
        if value.get('schema') == 'uniform-v2-reuse-raw-audit' or str(value.get('schema', '')).startswith('uniform-rate-results-'):
            return
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
        for field in ('artifacts', 'dynamic_artifacts'):
            if isinstance(value.get(field), dict):
                for path, digest in value[field].items():
                    yield dict(path=path, sha256=digest)
        if isinstance(value.get('receipt'), str) and value.get('receipt_sha256'):
            yield dict(path=value['receipt'], sha256=value['receipt_sha256'])
        if isinstance(value.get('trace'), str) and value.get('trace_sha256'):
            yield dict(path=value['trace'], sha256=value['trace_sha256'])
        for key, item in value.items():
            if key not in ('artifacts', 'dynamic_artifacts', 'files', 'source_files', 'frozen_files',
                           'source_300s_trace', 'executed_source', 'source_manifests'):
                yield from references(item)


def scientific_references(value):
    """Fast lane follows finalized cell evidence, without old qualification trees."""
    if not isinstance(value, dict):
        return
    if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
        yield from references(value)
        return
    for field in ('artifacts', 'dynamic_artifacts'):
        if isinstance(value.get(field), dict):
            for path, digest in value[field].items():
                yield dict(path=path, sha256=digest)
    for key in ('checkpoint', 'receipt', 'binding', 'raw_requests', 'raw_power', 'summary', 'trace_reference',
                'diagnosis_reference', 'metric_reconstruction'):
        reference = value.get(key)
        if isinstance(reference, dict) and 'path' in reference and 'sha256' in reference:
            yield reference
        elif isinstance(reference, str) and value.get(key + '_sha256'):
            yield dict(path=reference, sha256=value[key + '_sha256'])
    if isinstance(value.get('trace'), str) and value.get('trace_sha256'):
        yield dict(path=value['trace'], sha256=value['trace_sha256'])


class Mirror:
    def __init__(self, cache, *, bandwidth=2*1024**2):
        self.cache_path = Path(cache)
        self.cache = c.read(cache) if self.cache_path.exists() else {}
        self.bandwidth = bandwidth
        self.errors = []
        self.downloaded = 0
        self.bytes = 0
        self.visited = set()
        self.retry_refs = {}

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

    def hydrate(self, node, initial, *, walker=references):
        frontier = list(initial) + [ref for key, ref in self.retry_refs.items() if key[0] == node]
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
            # A transient network failure must be retried when the finalized
            # reference appears in the next status snapshot.
            good_keys = {(ref['path'], ref['sha256']) for ref in good}
            for ref in refs:
                key = (node, ref['path'], ref['sha256'])
                if (ref['path'], ref['sha256']) not in good_keys:
                    self.visited.discard(key)
                    self.retry_refs[key] = ref
                else:
                    self.retry_refs.pop(key, None)
            frontier = []
            report.save(self.cache_path, self.cache)
            for ref in good:
                if (Path(ref['path']).suffix == '.json'
                        and not Path(ref['path']).name.endswith(('manifest.json', 'spec.json'))):
                    try:
                        frontier.extend(walker(c.read(ref['path'])))
                    except Exception as exc:
                        self.issue(node, ref['path'], exc)
        report.save(self.cache_path, self.cache)


def status_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns, ino=value.st_ino)


def status_path(node, snapshot):
    path = safe_path(snapshot['path'])
    relative = path.relative_to(c.ROOT / node)
    if path.name not in ('status.json', 'metadata.json') or not relative.parts[0].startswith('uniform-'):
        raise ValueError('unexpected mutable status path')
    return path


def locally_current(path, saved):
    try:
        return bool(saved and saved.get('local_stat') == status_stat(path))
    except OSError:
        return False


def valid_remote_stat(value):
    return (isinstance(value, dict) and set(value) == {'size', 'mtime_ns', 'ctime_ns', 'ino'}
        and all(type(v) is int and v >= 0 for v in value.values()))


def mirrored_stat(path, raw):
    # A concurrent local snapshot writer must not bind old seeds to newer bytes.
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
        named = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if not (identity(before) == identity(after) == identity(named)
            and digest.hexdigest() == hashlib.sha256(raw).hexdigest()):
        raise ValueError('local mutable snapshot changed during mirror; refetch required')
    return dict(size=after.st_size, mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns, ino=after.st_ino)


def remote_statuses(node, status_cache=None):
    directory = c.ROOT / node
    known = {}
    for name, saved in (status_cache or {}).items():
        path = status_path(node, dict(path=name))
        if locally_current(path, saved) and valid_remote_stat(saved.get('remote_stat')):
            known[name] = saved['remote_stat']
    result = subprocess.run(remote_argv(HOSTS[node], INDEX_SCRIPT, directory),
        input=json.dumps(known).encode(), capture_output=True, timeout=40, check=True)
    return json.loads(result.stdout)


def mirror_full_statuses(node, snapshots):
    seeds = []
    for snapshot in snapshots:
        path = safe_path(snapshot['path'])
        expected = c.ROOT / node
        relative = path.relative_to(expected)
        if path.name not in ('status.json', 'metadata.json') or not relative.parts[0].startswith('uniform-'):
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
        for key in ('declaration', 'observations', 'observed_checkpoints',
                    'dynamic_reuse_observations', 'extensions', 'supervisor_terminal',
                    'pipeline_at_pdb_completion', 'last_measurement',
                    'completed', 'full_operation_measurement', 'setup_energy_index',
                    'evidence_closures'):
            seeds.extend(references(value.get(key, [])))
        if value.get('finished_s') and not value.get('node_lease_held'):
            for key in ('measurement', 'setup_measurement'):
                seeds.extend(references(value.get(key, [])))
    return seeds


def mirror_statuses(node, snapshots, status_cache=None):
    cache = status_cache if status_cache is not None else {}
    # A status may change locally after its stat was sent over SSH. Refetch it
    # immediately, while allowing all other unchanged snapshots to remain small.
    for retry in range(2):
        invalid = []
        for snapshot in snapshots:
            path = status_path(node, snapshot)
            if not valid_remote_stat(snapshot.get('remote_stat')):
                raise ValueError('missing or malformed remote status identity')
            if snapshot.get('unchanged') is True:
                saved = cache.get(str(path))
                if not (locally_current(path, saved) and saved['remote_stat'] == snapshot['remote_stat']):
                    invalid.append(str(path))
        if not invalid:
            break
        for name in invalid:
            cache.pop(name, None)
        if retry:
            raise ValueError('remote unchanged status lacks matching local snapshot')
        snapshots = remote_statuses(node, cache)
    seeds = []
    for snapshot in snapshots:
        path = status_path(node, snapshot)
        if snapshot.get('unchanged') is True:
            saved = cache[str(path)]
            if not locally_current(path, saved):
                cache.pop(str(path), None)
                raise ValueError('local status changed after delta validation; refetch required')
            seeds.extend(saved['seeds'])
            continue
        raw = base64.b64decode(snapshot['data'], validate=True)
        if len(raw) != snapshot['remote_stat']['size'] or snapshot['mtime_ns'] != snapshot['remote_stat']['mtime_ns']:
            raise ValueError('remote snapshot bytes/stat differ')
        extracted = mirror_full_statuses(node, [snapshot])
        cache[str(path)] = dict(remote_stat=snapshot['remote_stat'],
            local_stat=mirrored_stat(path, raw), seeds=extracted)
        seeds.extend(extracted)
    return seeds


def signature():
    values = []
    for node in ('A', 'B', 'C'):
        for path in (c.ROOT / node / 'uniform-rate-20260909-v2').glob('**/status.json'):
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
    """Only full five-system scope can finish the successor monitor."""
    if states is None:
        states = report.latest_states()
    terminal = []
    for group in result['groups']:
        key = (group['model'], group['dataset'], group.get('node', c.host(group['model'], group['dataset'])))
        path, state = states.get(key, (None, {}))
        # Entirely reused groups need no new GPU supervisor (32B LongBench).
        reuse_only = bool(group.get('complete') and path is None)
        valid = reuse_only or bool(state.get('complete') is True and state.get('finished_s')
            and not state.get('node_lease_held') and not state.get('error'))
        terminal.append(dict(model=key[0], dataset=key[1], node=key[2], path=str(path) if path else None,
                             pipeline_terminal=valid, reuse_only=reuse_only))
    complete = bool(scope == 'five_systems' and result['complete'])
    return dict(completion_scope='five_systems', raw_scope_complete=complete,
        scope_complete=bool(complete and all(v['pipeline_terminal'] for v in terminal) and not hydration_active),
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
    history = {node: list(references([{key: v[key] for key in ('checkpoint', 'receipt', 'raw_requests', 'raw_power', 'summary') if key in v}
        for v in base['reused_observations'] if v['measurement_host'] == ('Anew20260909' if node == 'A' else node)])) for node in HOSTS}
    status_caches = {node: {} for node in HOSTS}
    previous = None
    last_plot = 0
    iteration = 0
    hydration = ThreadPoolExecutor(max_workers=1)
    scientific_pool = ThreadPoolExecutor(max_workers=2)
    scientific_mirrors = {node: Mirror(HERE / ('cell-mirror-' + node + '.json'), bandwidth=args.bandwidth) for node in HOSTS}
    scientific_futures = {node: None for node in HOSTS}
    scientific_pending = {node: [] for node in HOSTS}
    reported_scientific_downloads = -1
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
            pending = {node: executor.submit(remote_statuses, node, status_caches[node]) for node in HOSTS}
            for node, future in pending.items():
                try:
                    snapshots[node] = future.result()
                except Exception as exc:
                    mirror.issue(node, 'remote_statuses', exc)
        for node in HOSTS:
            try:
                seeds = mirror_statuses(node, snapshots.get(node, []), status_caches[node])
                pending_seeds[node].extend(seeds)
                scientific_pending[node].extend(ref for ref in seeds
                    if '/audits/' in ref['path'] or '/checkpoints/' in ref['path'])
            except Exception as exc:
                mirror.issue(node, 'status_mirror', exc)
        for node in HOSTS:
            future = scientific_futures[node]
            if future is None or future.done():
                if future is not None:
                    try:
                        future.result()
                    except Exception as exc:
                        mirror.issue(node, 'cell_hydration', exc)
                batch = list({(r['path'], r['sha256']): r for r in scientific_pending[node]}.values())
                scientific_pending[node] = []
                scientific_futures[node] = (scientific_pool.submit(scientific_mirrors[node].hydrate,
                    node, batch, walker=scientific_references) if batch else None)
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
        scientific_downloads = sum(m.downloaded for m in scientific_mirrors.values())
        if current != previous or mirror.downloaded != reported_downloads or scientific_downloads != reported_scientific_downloads or due_plot:
            published_downloads = mirror.downloaded
            result = report.collect(declaration, args.out)
            rows = report.export(result, args.out)
            setup_result = energy_ledger.collect()
            energy_ledger.export(setup_result, args.out)
            failures = report.collect_engineering_failures(args.out / 'audit-cache', primary_result=result)
            report.export_engineering_failures(failures, args.out)
            reported_downloads = published_downloads
            reported_scientific_downloads = scientific_downloads
            published = True
            if due_plot or result['complete']:
                report.plot(result, rows, args.out)
                last_plot = time.time()
            previous = current
        else:
            result = c.read(args.out / 'results.json')
        hydration_active = ((future_mirror is not None and not future_mirror.done())
            or any(f is not None and not f.done() for f in scientific_futures.values()))
        finishing = completion_state(result, args.completion_scope, hydration_active=hydration_active)
        if (result.get('completion_scope') != args.completion_scope or
                result.get('scope_complete') != finishing['scope_complete']):
            result.update({k: finishing[k] for k in ('completion_scope', 'scope_complete', 'five_system_complete')})
            rows = report.export(result, args.out)
        if finishing['scope_complete']:
            rows = report.export(result, args.out)
            report.plot(result, rows, args.out)
            last_plot = time.time()
        state = dict(schema='uniform-rate-evidence-monitor-v2', pid=os.getpid(),
            started_s=getattr(run, 'started_s', started), updated_s=time.time(), iteration=iteration,
            complete=result['complete'], groups_complete=sum(g['complete'] for g in result['groups']),
            pdb_boundaries_complete=sum(g.get('pdb_boundary_complete', False) for g in result['groups']),
            observations=len(result['observations']), metric_audit_errors=len(result['metric_audit_errors']),
            downloaded_files=mirror.downloaded, downloaded_bytes=mirror.bytes,
            scientific_downloaded_files=scientific_downloads,
            scientific_mirror_errors={node: mirror.errors[-20:] for node, mirror in scientific_mirrors.items()},
            remote_status_counts={node: len(values) for node, values in snapshots.items()},
            mirror_errors=list(mirror.errors[-200:]), mirror_error_count=len(mirror.errors),
            hydration_active=hydration_active,
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
            for future in scientific_futures.values():
                if future is not None:
                    future.result()
            result = report.collect(declaration, args.out)
            report.export(result, args.out)
            energy_ledger.export(energy_ledger.collect(), args.out)
        if finishing['scope_complete'] or args.once:
            hydration.shutdown(wait=True)
            scientific_pool.shutdown(wait=True)
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
    parser.add_argument('--completion-scope', choices=('five_systems',), default='five_systems')
    arguments = parser.parse_args()
    if arguments.interval <= 0 or arguments.plot_interval <= 0 or arguments.bandwidth <= 0:
        parser.error('intervals and bandwidth must be positive')
    run(arguments)
