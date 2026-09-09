"""Wait for the first genuinely idle A/B/C host, then launch one campaign.

The --wait-only observer never mutates a running engine. A separate host
executor must independently acquire the existing exclusive lease and recheck
idleness before any serving action. Missing READY keeps the queue waiting.
"""
from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tarfile
import time
import traceback

import dispatcher
import runner
import trace_source

HERE = Path(__file__).resolve().parent
OUT = HERE / 'supervision'


def ssh(label):
    host = dispatcher.HOSTS[label]
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o',
            'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=' + host['alias'], host['address']]


def remote_python(label, source):
    if dispatcher.HOSTS[label]['hostname'] == socket.gethostname():
        command = [sys.executable, '-']
    else:
        command = ssh(label) + ['python3 -']
    result = subprocess.run(command, input=source, text=True, capture_output=True, timeout=40)
    if result.returncode:
        raise RuntimeError('host observation/launch failed: ' + result.stderr[-1500:])
    return json.loads(result.stdout)


def ready():
    ready_path, package_path = HERE / 'READY.json', HERE / 'package.json'
    if not ready_path.exists() or not package_path.exists():
        return False
    item = runner.read(ready_path)
    return item.get('cpu_validation_passed') is True and item.get('package_sha256') == trace_source.file_sha(package_path)


def install(label):
    package = runner.read(HERE / 'package.json')
    files = dict(package['files'])
    files[str(HERE / 'package.json')] = trace_source.file_sha(HERE / 'package.json')
    files[str(HERE / 'READY.json')] = trace_source.file_sha(HERE / 'READY.json')
    for name, digest in files.items():
        if trace_source.file_sha(name) != digest:
            raise ValueError('package changed before host staging: ' + name)
    if dispatcher.HOSTS[label]['hostname'] == socket.gethostname():
        return dict(host=label, verified=True, files=len(files), copied=0)
    bundle = OUT / ('package-' + label + '.tar')
    with tarfile.open(bundle, 'w') as archive:
        for name in files:
            archive.add(name, arcname=name.lstrip('/'), recursive=False)
    receiver = '''import hashlib,json,sys,tarfile,tempfile
from pathlib import Path
with tempfile.TemporaryFile() as stream:
 while True:
  data=sys.stdin.buffer.read(4*1024*1024)
  if not data:break
  stream.write(data)
 stream.seek(0)
 with tarfile.open(fileobj=stream,mode='r:') as archive:
  members=archive.getmembers()
  for member in members:
   assert member.isfile() and '..' not in Path(member.name).parts and not member.name.startswith('/'),member.name
   target=Path('/')/member.name
   assert str(target).startswith('/root/workspace/'),str(target)
   if target.exists():
    assert target.is_file() and hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(archive.extractfile(member).read()).digest(),'existing different file; preserved: '+str(target)
  copied=0
  for member in members:
   target=Path('/')/member.name
   if target.exists():continue
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as out:out.write(archive.extractfile(member).read())
   copied+=1
print(json.dumps({'verified':True,'files':len(members),'copied':copied}))
'''
    with bundle.open('rb') as stream:
        result = subprocess.run(ssh(label) + [shlex.join(['python3', '-c', receiver])],
                                stdin=stream, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError('safe package staging failed: ' + result.stderr.decode(errors='replace')[-2000:])
    return dict(json.loads(result.stdout), host=label)


def launch(label, claim_id):
    command = ['python3', '-u', str(HERE / 'host_executor.py'), '--run', '--claim-id', claim_id]
    code = '''from pathlib import Path
import json,subprocess
root=Path(ROOT)
root.mkdir(parents=True,exist_ok=True)
with (root/'host.log').open('xb') as log:
 child=subprocess.Popen(COMMAND,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps({'pid':child.pid}))
'''.replace('ROOT', repr(str(HERE / 'claims' / claim_id))).replace('COMMAND', repr(command))
    return remote_python(label, code)


def status_of(label, claim_id):
    path = HERE / 'claims' / claim_id / 'status.json'
    return remote_python(label, 'import json\nfrom pathlib import Path\np=Path(' + repr(str(path)) +
                         ')\nprint(p.read_text() if p.exists() else json.dumps({"status":"starting"}))\n')


def mirror(label):
    if dispatcher.HOSTS[label]['hostname'] == socket.gethostname():
        return
    destination = HERE / 'mirrors' / label
    destination.mkdir(parents=True, exist_ok=True)
    # Only this campaign's output, never old source/results or serving paths.
    host = dispatcher.HOSTS[label]
    for directory in ('execution', 'claims'):
        result = subprocess.run(['rsync', '-a', '--ignore-missing-args', '-e', shlex.join(ssh(label)[:-1]),
             host['address'] + ':' + str(HERE / directory) + '/', str(destination / directory) + '/'],
             capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError('result mirror failed: ' + result.stderr.decode(errors='replace')[-1000:])


def run(interval=20):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'observer.lock').open('a+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(pid=os.getpid(), started_s=time.time(), status='waiting',
                     protocol_id='a14b-sharegpt-slo90-v1', host=None, claim_id=None)
        previous = OUT / 'status.json'
        if previous.exists():
            old = runner.read(previous)
            if old.get('claim_id'):
                state.update(host=old['host'], claim_id=old['claim_id'])
        observations = dispatcher.new_state()
        while True:
            try:
                if state['claim_id']:
                    actual = status_of(state['host'], state['claim_id'])
                    state.update(status=actual['status'], execution=actual)
                    if actual['status'] == 'declined':
                        state.setdefault('declined_claims', []).append(dict(host=state['host'], claim_id=state['claim_id']))
                        state.update(host=None, claim_id=None, status='waiting')
                        observations = dispatcher.new_state()
                    else:
                        mirror(state['host'])
                    if actual['status'] in ('complete', 'failed'):
                        state['finished_s'] = time.time()
                        runner.save(previous, state)
                        return 0 if actual['status'] == 'complete' else 2
                else:
                    observations, snapshot = dispatcher.poll_once(observations)
                    state.update(status='waiting_for_idle_host' if ready() else 'waiting_for_validated_package',
                                 idle_observation=snapshot, ready=ready())
                    candidate = snapshot.get('candidate')
                    if candidate and ready():
                        state.update(status='staging_candidate', host=candidate)
                        runner.save(previous, state)
                        state['staging'] = install(candidate)
                        claim_id = str(time.time_ns())
                        state.update(claim_id=claim_id, status='claiming')
                        runner.save(previous, state)  # durable before dispatch
                        state['launch'] = launch(candidate, claim_id)
                state.update(updated_s=time.time(), last_error=None)
                runner.save(previous, state)
            except Exception as exc:
                state.update(last_error=str(exc), traceback=traceback.format_exc(), updated_s=time.time())
                runner.save(previous, state)
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-only', action='store_true', required=True)
    parser.add_argument('--interval', type=int, default=20)
    args = parser.parse_args()
    if args.interval < 10:
        raise ValueError('read-only polling interval must be at least ten seconds')
    raise SystemExit(run(args.interval))
