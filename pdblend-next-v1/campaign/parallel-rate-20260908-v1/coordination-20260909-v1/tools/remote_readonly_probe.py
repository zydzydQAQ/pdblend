"""Standard-library remote inventory. Read-only; receive a JSON request on stdin.

This source is sent to `python3 -B -c` over SSH, never installed remotely.
Do not import campaign modules: imports there can perform deployment actions.
"""
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

MAX_HASH_BYTES = 64 * 1024 * 1024
WORKSPACE = Path('/root/workspace')


def command(argv, timeout=15):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                check=False, stdin=subprocess.DEVNULL)
        return dict(returncode=result.returncode, stdout=result.stdout[:100000],
                    stderr=result.stderr[:2000])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(returncode=None, error=str(exc))


def file_check(target):
    path = Path(target['path'])
    result = dict(target)
    try:
        # Only explicit experiment paths may be inspected; never traverse symlinks
        # outside the experiment workspace or read arbitrary remote files.
        resolved = path.resolve()
        if not resolved.is_relative_to(WORKSPACE):
            return dict(result, status='outside_workspace')
        if not path.is_file():
            return dict(result, status='missing' if not path.exists() else 'not_a_file')
        size = path.stat().st_size
        result['bytes'] = size
        if size > MAX_HASH_BYTES:
            return dict(result, status='too_large_to_hash')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        result['actual_sha256'] = digest.hexdigest()
        expected = target.get('expected_sha256')
        result['status'] = ('match' if result['actual_sha256'] == expected else 'mismatch') if expected else 'present_unpinned'
        if 'qualification' in target['kinds'] and size <= 1024 * 1024:
            try:
                value = json.loads(path.read_text())
                if isinstance(value, dict):
                    result['qualification_flags'] = {
                        key: value[key] for key in ('schema', 'passed', 'complete', 'gpu_executed', 'serving_validation_pending')
                        if key in value and isinstance(value[key], (str, bool, int, float, type(None)))
                    }
            except (ValueError, UnicodeError):
                result['qualification_json_parse_failed'] = True
        return result
    except OSError as exc:
        return dict(result, status='read_error', error=str(exc))


def lock_observation(path):
    result = dict(path=path, lock_acquired_by_probe=False)
    try:
        p = Path(path)
        if not p.exists():
            return dict(result, exists=False, state='absent')
        stat = p.stat()
        device_inode = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
        owners = []
        for line in Path('/proc/locks').read_text().splitlines():
            fields = line.replace(' -> ', ' ').split()
            if len(fields) < 6:
                continue
            try:
                major, minor, inode = fields[5].split(':')
                observed = (int(major, 16), int(minor, 16), int(inode))
            except (ValueError, IndexError):
                continue
            if observed == device_inode:
                owners.append(dict(kind=fields[1], mode=fields[3], pid=int(fields[4])))
        return dict(result, exists=True, state='held' if owners else 'no_visible_owner',
                    owners=owners, observation_only_not_a_lock_guarantee=True)
    except (OSError, ValueError) as exc:
        return dict(result, state='unknown', error=str(exc))


def campaign_processes():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            argv = (path / 'cmdline').read_bytes().split(b'\0')
            # Return script paths and PID only; command arguments can contain secrets.
            scripts = [x.decode(errors='replace') for x in argv if b'/campaign/' in x
                       and x.endswith(b'.py') and b' ' not in x]
            if scripts:
                found.append(dict(pid=int(path.name), script_paths=scripts[:3]))
        except OSError:
            continue
    return found


def probe(request):
    if request.get('node') not in ('A', 'C') or not isinstance(request.get('files'), list):
        raise ValueError('explicit A/C inventory required')
    started = time.time()
    gpu = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                   '--format=csv,noheader,nounits'])
    gpu_rows = []
    if gpu.get('returncode') == 0:
        for row in csv.reader(io.StringIO(gpu['stdout'])):
            if len(row) == 6:
                gpu_rows.append(dict(zip(('index', 'uuid', 'name', 'memory_total_mib',
                                          'memory_used_mib', 'utilization_percent'), (x.strip() for x in row))))
    compute = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                       '--format=csv,noheader,nounits'])
    # No inspect, environment, command arguments, mount details, or container logs.
    containers = command(['docker', 'ps', '-a', '--no-trunc', '--format',
                          '{{json .ID}}\t{{json .Names}}\t{{json .Image}}\t{{json .State}}'])
    hostname = socket.gethostname()
    files = [file_check(target) for target in request['files']]
    counts = {}
    for item in files:
        counts[item['status']] = counts.get(item['status'], 0) + 1
    recoverable = [dict(path=x['path'], sha256=x['actual_sha256'], kinds=x['kinds'])
                   for x in files if x['status'] == 'match' and x.get('local_status') == 'missing']
    return dict(schema='pdblend-readonly-preflight-v1', node=request['node'],
                captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                started_s=started, finished_s=time.time(), remote_read_only=True,
                gpu_work_started=False, remote_files_written=False,
                physical_qualification_granted=False,
                identity=dict(hostname=hostname, expected_hostname=request['expected_hostname'],
                              hostname_matches=hostname == request['expected_hostname'],
                              gpu_count=len(gpu_rows), expected_gpu_count=8,
                              all_eight_L20=len(gpu_rows) == 8 and all('L20' in x['name'] for x in gpu_rows),
                              historical_gpu_uuid_comparison='not_pinned; hostname match alone is not full hardware qualification'),
                gpus=gpu_rows, gpu_query=gpu, gpu_processes=compute, containers=containers,
                model_paths=[dict(path=x, exists=Path(x).is_dir()) for x in request['model_paths']],
                experiment_lock=lock_observation(request['lock_path']),
                visible_campaign_processes=campaign_processes(), files=files,
                file_counts=counts, recoverable_local_missing_files=recoverable,
                recovery_action_performed=False)


if __name__ == '__main__':
    print(json.dumps(probe(json.load(sys.stdin)), ensure_ascii=False, allow_nan=False))
