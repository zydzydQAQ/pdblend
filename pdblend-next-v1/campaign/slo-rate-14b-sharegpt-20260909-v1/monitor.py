"""Read-only SSH evidence monitor for this campaign; never starts GPU work.

Immutable transfer helpers copied from the established uniform monitor, pinned
below. Remote operations are read-only snapshots and SHA-checked byte streams.
Only local status/observations snapshots are replaced. Conflicting immutable
local files are preserved and reported. Run --once for one CPU collection pass.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import report
c = report
HERE = Path(__file__).resolve().parent
_crosscheck_spec = importlib.util.spec_from_file_location('slo_monitor_crosscheck', HERE / 'reports/crosscheck.py')
crosscheck = importlib.util.module_from_spec(_crosscheck_spec)
_crosscheck_spec.loader.exec_module(crosscheck)
WORKSPACE = Path('/root/workspace')
HOSTS = {'A': '120.79.123.62', 'C': '47.106.163.29'}
HOSTNAMES = {'A': 'iZwz9274emxme9019d2sjgZ', 'C': 'iZwz9gfq11hx1sbob59yrgZ'}
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o',
       'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
TRANSFER_HELPER_SOURCE = {'path': '/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/common/uniform-rate-20260909-v2/monitor.py', 'sha256': '3676ffa41a6ae6e873ad270f4bb61c74e67580d8c8bcd34223e7f630d5e51f28'}

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


INDEX_SCRIPT = r'''
import base64,hashlib,json,pathlib,socket,sys
root=pathlib.Path(sys.argv[1]); node=sys.argv[2]; snapshots=[]; metadata_references={}
paths=list((root/node).glob('*/status.json'))
observation_path=root/node/'observations.json'
if observation_path.is_file(): paths.append(observation_path)
for path in paths:
    try:
        raw=path.read_bytes(); value=json.loads(raw)
        if path.name!='observations.json' and value.get('schema')!='slo-rate-node-status-v1': continue
        if len(raw)>32*1024**2: raise ValueError('snapshot too large')
        if raw!=path.read_bytes(): raise ValueError('snapshot changed while reading')
        snapshots.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            data=base64.b64encode(raw).decode()))
        if path==observation_path:
            for observation in value:
                manifest=observation.get('materialization_manifest',{})
                candidate=pathlib.Path(manifest.get('path','/absent'))
                if candidate.parent.parent != root/'workloads' or candidate.name!='manifest.json': continue
                workload=candidate.parent/'workload.json'
                if workload.is_file() and workload.stat().st_size <= 512*1024:
                    data=workload.read_bytes()
                    metadata_references[str(workload)]=dict(path=str(workload),sha256=hashlib.sha256(data).hexdigest())
    except (OSError,ValueError): pass
print(json.dumps(dict(hostname=socket.gethostname(),snapshots=snapshots,metadata_references=list(metadata_references.values()))))
'''


def snapshot(node):
    result = subprocess.run(remote_argv(HOSTS[node], INDEX_SCRIPT, HERE, node),
        capture_output=True, timeout=45, check=True)
    value = json.loads(result.stdout)
    if value['hostname'] != HOSTNAMES[node]:
        raise ValueError('SSH endpoint has unexpected physical hostname')
    return value


def minimal_references(value):
    """Serving-time mirror: bounded terminal metric files, never qualification trees."""
    if not isinstance(value, dict):
        return
    for key in ('audit_reference', 'checkpoint', 'receipt', 'summary', 'binding',
                'raw_requests', 'raw_power', 'trace_reference', 'materialization_manifest'):
        reference = value.get(key)
        if isinstance(reference, dict) and isinstance(reference.get('path'), str) and reference.get('sha256'):
            yield dict(path=reference['path'], sha256=reference['sha256'])
        elif isinstance(reference, str) and value.get(key + '_sha256'):
            yield dict(path=reference, sha256=value[key + '_sha256'])
    if isinstance(value.get('trace'), str) and value.get('trace_sha256'):
        yield dict(path=value['trace'], sha256=value['trace_sha256'])
    for path, digest in value.get('artifacts', {}).items():
        if Path(path).name in ('bench.csv', 'power.csv', 'summary.json', 'runtime_config.json'):
            yield dict(path=path, sha256=digest)
    if isinstance(value.get('row'), dict):
        yield from minimal_references(value['row'])


def install_snapshot(node, record):
    path = safe_path(record['path'])
    if not path.is_relative_to(HERE / node):
        raise ValueError('snapshot outside this node campaign')
    raw = base64.b64decode(record['data'], validate=True)
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('snapshot hash mismatch')
    value = json.loads(raw)
    if path.name == 'observations.json':
        if path != HERE / node / 'observations.json' or not isinstance(value, list):
            raise ValueError('invalid observation snapshot')
    elif path.name != 'status.json' or value.get('schema') != 'slo-rate-node-status-v1':
        raise ValueError('invalid supervisor snapshot')
    if not path.exists() or path.read_bytes() != raw:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.mirror.tmp')
        temporary.write_bytes(raw)
        temporary.replace(path)
    return value


def collect_once(*, plots=True, bandwidth=2 * 1024**2):
    previous = report.read(HERE / 'monitor-status.json') if (HERE / 'monitor-status.json').exists() else {}
    state = dict(schema='slo-rate-readonly-monitor-v1', pid=os.getpid(), updated_s=time.time(),
        remote_mutations=False, GPU_operations=False, nodes={}, errors=[], helper_source=TRANSFER_HELPER_SOURCE,
        mirror_scope='terminal metric files only; qualification and full checkpoint inventories deferred',
        deferred_qualification_references=[])
    mirror = Mirror(HERE / 'monitor-hash-cache.json', bandwidth=bandwidth)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {node: pool.submit(snapshot, node) for node in HOSTS}
        for node, future in futures.items():
            try:
                snapshot_result = future.result()
                records = snapshot_result['snapshots']
                values = [install_snapshot(node, record) for record in records]
                supervisors = [v for v in values if isinstance(v, dict)]
                latest = max(supervisors, key=lambda v: v.get('started_s', 0), default={})
                observations = next((v for v in values if isinstance(v, list)), [])
                state['nodes'][node] = dict(phase=latest.get('phase', 'awaiting_supervisor'),
                    complete=latest.get('complete') is True, current_cell=latest.get('current_cell'),
                    observation_count=len(observations), error=latest.get('error'),
                    node_status_path=next((r['path'] for r in records if r['path'].endswith('/status.json')
                        and json.loads(base64.b64decode(r['data'])).get('started_s') == latest.get('started_s')), None))
                # Only producer-published terminal observations seed evidence transfer.
                # The running cell status and in-flight raw files are never copied.
                initial = []
                for observation in observations:
                    initial.extend(minimal_references(observation))
                    if isinstance(observation.get('qualification'), dict):
                        state['deferred_qualification_references'].append(observation['qualification'])
                initial.extend(snapshot_result.get('metadata_references', []))
                mirror.hydrate(node, initial, walker=minimal_references)
            except Exception as exc:
                state['errors'].append(dict(node=node, error=repr(exc)))
                state['nodes'][node] = dict(phase='collection_error', complete=False, error=repr(exc))
    state['errors'].extend(mirror.errors)
    state.update(downloaded_files=mirror.downloaded, downloaded_bytes=mirror.bytes,
                 pending_references=len(mirror.retry_refs))
    observation_inputs = [report.ref(HERE / node / 'observations.json') for node in HOSTS
                          if (HERE / node / 'observations.json').exists()]
    observation_inputs.append(report.ref(report.__file__))
    observation_inputs.append(report.ref(crosscheck.__file__))
    for node in ('A', 'B', 'C'):
        for name in ('setup-energy-ledger.json', 'failure-ledger.json', 'energy-ledger.json'):
            ledger_path = HERE / node / name
            if ledger_path.exists(): observation_inputs.append(report.ref(ledger_path))
    for node in HOSTS:
        _, terminal_ref = report.latest_supervisor(HERE, node)
        if terminal_ref:
            observation_inputs.append(terminal_ref)
    reference = HERE / 'B/reference.json'
    if reference.exists(): observation_inputs.append(report.ref(reference))
    digest = hashlib.sha256(json.dumps(observation_inputs, sort_keys=True).encode()).hexdigest()
    if digest != previous.get('observations_digest') or mirror.downloaded or not (HERE / 'reports/current/results.json').exists():
        try:
            result = report.build(HERE, plots=plots)
            state['reported_observations'] = len(result['observations'])
            checks = result['local_cpu_crosscheck']
            state['local_cpu_crosscheck'] = {key: checks[key] for key in ('passed_count', 'failed_count', 'pending_count')}
            state['errors'].extend(dict(component='local_cpu_crosscheck', **issue) for issue in checks['issues'])
            state['measurement_report_complete'] = result['complete']
            state['observations_digest'] = digest
        except Exception as exc:
            state['errors'].append(dict(component='report', error=repr(exc)))
    else:
        state['observations_digest'] = digest
        state['reported_observations'] = previous.get('reported_observations')
        state['local_cpu_crosscheck'] = previous.get('local_cpu_crosscheck')
        state['measurement_report_complete'] = previous.get('measurement_report_complete', False)
        state['errors'].extend(error for error in previous.get('errors', [])
                               if error.get('component') == 'local_cpu_crosscheck')
    state['complete'] = bool(state['nodes']) and all(v.get('complete') for v in state['nodes'].values()) and not state['errors'] and state.get('measurement_report_complete') is True
    report.save(HERE / 'monitor-status.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--bandwidth-mib', type=float, default=2)
    args = parser.parse_args()
    if args.interval < 10 or args.bandwidth_mib <= 0:
        raise ValueError('interval >=10 seconds and positive bandwidth required')
    with (HERE / 'monitor.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            state = collect_once(plots=not args.no_plots, bandwidth=int(args.bandwidth_mib * 1024**2))
            signature = json.dumps(dict(nodes=state['nodes'], errors=state['errors']), sort_keys=True)
            if signature != previous:
                print(signature, flush=True)
                previous = signature
            if args.once or state['complete'] or (HERE / 'MONITOR_STOP').exists():
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
