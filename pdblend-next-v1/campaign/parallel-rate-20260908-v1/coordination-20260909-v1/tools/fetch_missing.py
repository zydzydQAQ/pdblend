#!/usr/bin/env python3
"""Copy only missing local experiment files from an audited read-only preflight.

Both ends verify SHA256. No remote writes and no overwrite of local files.
"""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import preflight

REMOTE_SOURCE = r'''
import base64, hashlib, json, socket, sys
from pathlib import Path
request=json.load(sys.stdin)
assert socket.gethostname()==request['expected_hostname'], 'remote hostname changed'
total=0
for item in request['files']:
    path=Path(item['path'])
    assert path.resolve().is_relative_to('/root/workspace'), 'file escaped workspace'
    assert path.is_file() and path.stat().st_size<=64*1024*1024, 'missing or oversized input'
    data=path.read_bytes(); total+=len(data)
    assert len(data)<=64*1024*1024 and total<=128*1024*1024, 'transfer size limit'
    assert hashlib.sha256(data).hexdigest()==item['sha256'], 'source hash changed: '+str(path)
    print(json.dumps(dict(path=str(path),sha256=item['sha256'],bytes=len(data),data=base64.b64encode(data).decode())),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='new local transfer audit JSON')
    parser.add_argument('--identity-file', type=Path)
    parser.add_argument('--include-observed-unpinned', action='store_true',
                        help='also fetch unpinned files using their exact observed snapshot SHA; this does not certify historical provenance')
    parser.add_argument('--metadata-only', action='store_true', help='skip CSV, JSONL and log payloads; retain references for later audited recovery')
    parser.add_argument('--fetch', action='store_true', help='perform transfer; default only validates the list')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out already exists')
    snapshot = preflight.read(args.snapshot)
    remote = snapshot.get('remote', {})
    if not remote.get('identity', {}).get('hostname_matches'):
        parser.error('snapshot must contain a successful expected-hostname match')
    files = []
    for row in remote['files']:
        if row['local_status'] != 'missing':
            continue
        origin = 'historical_pinned_reference'
        if row['status'] != 'match':
            if not args.include_observed_unpinned or row['status'] != 'present_unpinned':
                continue
            origin = 'observed_preflight_only'
        path = Path(row['path'])
        if args.metadata_only and path.suffix in ('.csv', '.jsonl', '.log'):
            continue
        if not path.is_absolute() or not path.resolve().is_relative_to('/root/workspace'):
            parser.error('destination escaped workspace')
        digest = row['actual_sha256']
        if path.exists():
            if not path.is_file() or preflight.sha(path) != digest:
                parser.error('existing local file differs; never overwrite: ' + str(path))
            continue
        files.append(dict(path=str(path), sha256=digest, bytes=row['bytes'], sha256_origin=origin))
    if any(row['bytes'] > 64 * 1024 * 1024 for row in files) or sum(row['bytes'] for row in files) > 128 * 1024 * 1024:
        parser.error('bounded transfer limit exceeded; make a smaller audited inventory')
    audit = dict(schema='pdblend-missing-file-transfer-v1',
                 captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 snapshot=dict(path=str(args.snapshot.resolve()), sha256=preflight.sha(args.snapshot)),
                 remote_hostname=remote['identity']['hostname'],
                 metadata_only=args.metadata_only,
                 remote_files_written=False, gpu_work_started=False, overwritten_files=[],
                 requested=files, created=[], transfer_started=False, complete=False)
    error = None
    if args.fetch and files:
        ssh_args = argparse.Namespace(host=preflight.safe_host(snapshot['ssh_host']),
                                      user=preflight.safe_user(snapshot['ssh_user']),
                                      identity_file=args.identity_file)
        request = dict(expected_hostname=remote['identity']['expected_hostname'], files=files)
        audit['transfer_started'] = True
        try:
            result = subprocess.run(preflight.ssh_argv(ssh_args, REMOTE_SOURCE), input=json.dumps(request),
                                    capture_output=True, text=True, timeout=180, check=False)
            if result.returncode:
                raise RuntimeError(result.stderr[-4000:])
            received = [json.loads(line) for line in result.stdout.splitlines() if line]
            if [x['path'] for x in received] != [x['path'] for x in files]:
                raise ValueError('transfer list differs from the explicit request')
            # Validate every byte before creating any destination.
            decoded = []
            for row, expected in zip(received, files):
                data = base64.b64decode(row['data'], validate=True)
                if len(data) != row['bytes'] or row['sha256'] != expected['sha256'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                    raise ValueError('download SHA256/size mismatch: ' + expected['path'])
                decoded.append(data)
            for data, expected in zip(decoded, files):
                path = Path(expected['path'])
                if path.exists():
                    if not path.is_file() or preflight.sha(path) != expected['sha256']:
                        raise ValueError('destination changed after preflight: ' + str(path))
                    continue
                if not path.resolve().is_relative_to('/root/workspace'):
                    raise ValueError('destination escaped workspace before write')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(data)
                audit['created'].append(dict(expected, verified_sha256=preflight.sha(path)))
            audit['complete'] = True
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            error = str(exc)
            audit['error'] = error
    elif args.fetch:
        audit['complete'] = True
    audit['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(audit=str(args.out.resolve()), requested=len(files),
                          planned_bytes=sum(x['bytes'] for x in files), created=len(audit['created']),
                          transfer_started=audit['transfer_started'], complete=audit['complete'], error=error)))
    return 2 if error else 0


if __name__ == '__main__':
    sys.exit(main())
