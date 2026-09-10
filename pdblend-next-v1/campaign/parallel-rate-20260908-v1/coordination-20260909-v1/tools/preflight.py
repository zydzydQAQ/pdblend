#!/usr/bin/env python3
"""Build a pinned A/C recovery inventory; connect only with explicit --connect."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED_HOSTS = {'A': 'iZwz92bdfqihqp38tekqjyZ', 'C': 'iZwz9gfq11hx1sbob59yrgZ'}
MODELS = {'A': '14b', 'C': '7b'}
DECLARATION_SHA = 'a566e7f203a77abe64aa2f122e66c47b87c44ab0b91f60c3453235ca69c4c32f'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_host(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]*', value):
        raise argparse.ArgumentTypeError('host must be an IP address or DNS name, without SSH options')
    return value


def safe_user(value):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', value):
        raise argparse.ArgumentTypeError('invalid SSH user')
    return value


def inventory(node, extra_refs=None):
    release = ROOT / 'common/ascending-rate-execution-v2/release-001'
    declaration_path = release / 'declaration.json'
    if sha(declaration_path) != DECLARATION_SHA:
        raise ValueError('frozen execution declaration changed; review inventory before probing')
    d = read(declaration_path)
    targets = {}

    def add(path, digest=None, kind='execution_package'):
        path = str(Path(path))
        if not Path(path).is_absolute() or not Path(path).is_relative_to('/root/workspace'):
            raise ValueError('inventory path outside experiment workspace: ' + path)
        if digest is not None and not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('invalid SHA256 for ' + path)
        item = targets.setdefault(path, dict(path=path, expected_sha256=digest, kinds=[]))
        if digest and item['expected_sha256'] not in (None, digest):
            raise ValueError('conflicting pinned hashes: ' + path)
        if digest:
            item['expected_sha256'] = digest
        if kind not in item['kinds']:
            item['kinds'].append(kind)

    add(declaration_path, DECLARATION_SHA)
    add(release / 'manifest.json', sha(release / 'manifest.json'))
    model = MODELS[node]
    model_label = 'Qwen2.5-' + model.upper() + '-Instruct'
    expected = set()
    for group_name in ('reused_observations', 'historical_A_observations', 'stopped_P12_history'):
        for obs in d[group_name]:
            if obs.get('measurement_host') != node:
                continue
            expected.add(obs['actual_hostname'])
            cp = obs['checkpoint']
            add(cp['path'], cp['sha256'], 'checkpoint' if group_name == 'reused_observations' else 'historical_checkpoint')
    if expected != {EXPECTED_HOSTS[node]}:
        raise ValueError('pinned historical hostname set differs: ' + repr(expected))
    for row in d['cells']:
        if row['node'] != node:
            continue
        add(row['trace_reference']['path'], row['trace_reference']['sha256'], 'trace')
        for field in ('source_300s_trace', 'source_spec'):
            ref = row.get(field)
            if ref:
                add(ref['path'], ref['sha256'], 'trace_source')
    manifest = read(release / 'manifest.json')
    for name, digest in manifest['files'].items():
        if name in ('node-' + node + '.json', 'cpu-tests.json', 'independent-verification.json'):
            add(release / name, digest)
    for group in d['groups']:
        if group['node'] == node:
            ref = group['pdb_source_contract']['target_manifest']
            add(ref['path'], ref['sha256'], 'controller_manifest')
    for path, digest in d['sources'].items():
        if (f'/{node}/final-p9-' in path or f'/{node}/p9-nonalpaca-' in path
                or (f'/{node}/' in path and 'qualification' in path and path.endswith('.json'))):
            add(path, digest, 'qualification' if ('qualification' in path or 'p9-nonalpaca' in path) else 'execution_package')
    if node == 'A':
        for name in ('A/final-p9-release-001/release.json', 'A/p9-nonalpaca-capacity-noaction-001/validation.json'):
            add(ROOT / name, d['sources'].get(str(ROOT / name)), 'qualification')
    else:
        for name in ('C/fixed-release-p4-strict/release.json', 'C/p4-completion-release/release.json',
                     'C/eco-drain37-v1/qualified/binding.json', 'C/eco-drain37-final-closure-001/closure.json'):
            path = ROOT / name
            add(path, d['sources'].get(str(path)) or (sha(path) if path.is_file() else None))
        profile = ROOT.parents[1] / 'campaign/main-slo-improvement-v1/C/profile-unified-v5-001'
        for name in ('profiles.development.json', 'qualification.json'):
            path = profile / name
            add(path, d['sources'].get(str(path)) or (sha(path) if path.is_file() else None), 'qualification')
    for path in (ROOT / 'common/ascending-rate-execution-v2/contract.py',
                 ROOT / 'common/execution-until-complete-v1/run.py',
                 ROOT / 'common/execution-until-complete-v1/child.py'):
        add(path, d['sources'].get(str(path)) or (sha(path) if path.is_file() else None))
    for reference in extra_refs or []:
        add(reference['path'], reference['sha256'], reference.get('kind', 'explicit_dependency'))
    for item in targets.values():
        path = Path(item['path'])
        try:
            if not path.is_file():
                item['local_status'] = 'missing'
            else:
                actual = sha(path)
                item['local_sha256'] = actual
                item['local_status'] = ('match' if actual == item['expected_sha256'] else 'mismatch') if item['expected_sha256'] else 'present_unpinned'
        except OSError as exc:
            item.update(local_status='read_error', local_error=str(exc))
    return dict(node=node, expected_hostname=EXPECTED_HOSTS[node],
                model_paths=['/models/' + model_label, '/root/workspace/models/' + model_label],
                lock_path='/root/workspace/pdblend/new-results/campaigns/node-experiment.lock',
                files=sorted(targets.values(), key=lambda x: x['path']))


def ssh_argv(args, source):
    argv = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
            '-oUpdateHostKeys=no', '-oConnectTimeout=8', '-oServerAliveInterval=10',
            '-oServerAliveCountMax=2', '-oPasswordAuthentication=no',
            '-oKbdInteractiveAuthentication=no', '-oForwardAgent=no', '-oForwardX11=no',
            '-oPermitLocalCommand=no', '-oControlMaster=no', '-oControlPath=none']
    if args.identity_file:
        argv += ['-oIdentitiesOnly=yes', '-i', str(args.identity_file.expanduser().resolve())]
    argv += ['-l', args.user, args.host, 'python3 -B -c ' + shlex.quote(source)]
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', required=True, choices=('A', 'C'))
    parser.add_argument('--host', type=safe_host)
    parser.add_argument('--user', type=safe_user)
    parser.add_argument('--identity-file', type=Path, help='SSH private key path; contents are never read by this tool')
    parser.add_argument('--out', type=Path, help='new local JSON snapshot file; existing files are never overwritten')
    parser.add_argument('--extra-refs', type=Path, help='local JSON list of additional explicit {path, sha256} experiment dependencies')
    parser.add_argument('--connect', action='store_true', help='perform the read-only SSH probe; default is local inventory only')
    args = parser.parse_args()
    if args.connect and not (args.host and args.user and args.out):
        parser.error('--connect requires --host, --user and --out')
    if args.out and args.out.exists():
        parser.error('--out already exists; choose a new snapshot path')
    request = inventory(args.node, read(args.extra_refs) if args.extra_refs else None)
    snapshot = dict(schema='pdblend-preflight-controller-v1',
                    captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    mode='ssh_read_only' if args.connect else 'local_only', node=args.node,
                    ssh_host=args.host, ssh_user=args.user, gpu_work_started=False,
                    remote_files_written=False, remote_connection_attempted=False,
                    physical_qualification_granted=False,
                    tool_sha256=sha(__file__), probe_sha256=sha(HERE / 'remote_readonly_probe.py'),
                    inventory=request,
                    local_counts=dict(Counter(x['local_status'] for x in request['files'])),
                    local_missing_retained_checkpoints=sum('checkpoint' in x['kinds'] and x['local_status'] == 'missing' for x in request['files']))
    if args.extra_refs:
        snapshot['extra_refs'] = dict(path=str(args.extra_refs.resolve()), sha256=sha(args.extra_refs))
    exitcode = 0
    if args.connect:
        source = (HERE / 'remote_readonly_probe.py').read_text()
        snapshot['remote_connection_attempted'] = True
        try:
            result = subprocess.run(ssh_argv(args, source), input=json.dumps(request),
                                    capture_output=True, text=True, timeout=180, check=False)
            snapshot['ssh_returncode'] = result.returncode
            if result.returncode:
                snapshot['connection_error'] = result.stderr[-4000:]
                exitcode = 2
            else:
                snapshot['remote'] = json.loads(result.stdout)
                if not snapshot['remote']['identity']['hostname_matches']:
                    exitcode = 3
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            snapshot['connection_error'] = str(exc)
            exitcode = 2
    snapshot['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(snapshot, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    summary = {key: snapshot[key] for key in ('mode', 'node', 'remote_connection_attempted',
               'gpu_work_started', 'local_counts', 'local_missing_retained_checkpoints')}
    summary['snapshot'] = str(args.out.resolve()) if args.out else None
    if 'connection_error' in snapshot:
        summary['connection_error'] = snapshot['connection_error']
    if 'remote' in snapshot:
        remote = snapshot['remote']
        summary.update(identity=remote['identity'], remote_file_counts=remote['file_counts'],
                       recoverable_local_missing_files=len(remote['recoverable_local_missing_files']),
                       experiment_lock=remote['experiment_lock'])
    print(json.dumps(summary, ensure_ascii=False))
    return exitcode


if __name__ == '__main__':
    sys.exit(main())
