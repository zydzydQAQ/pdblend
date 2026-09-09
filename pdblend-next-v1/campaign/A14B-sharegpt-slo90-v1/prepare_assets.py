"""Prepare model and image on a candidate host without touching GPU ownership."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL = Path('/root/workspace/models/Qwen2.5-14B-Instruct')
HOSTS = {
    'B': ('172.16.50.102', '39.108.209.97', 'iZwz9i5bte3xkpmcoes3t2Z'),
    'C': ('172.16.50.105', '47.106.163.29', 'iZwz9gfq11hx1sbob59yrgZ'),
}
IMAGE = 'pdblend-next:io-v3'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=HOSTS, required=True)
    args = parser.parse_args()
    address, alias, hostname = HOSTS[args.host]
    out = ROOT / 'asset-staging' / args.host
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        write(out / ('status-history-' + str(time.time_ns()) + '.json'),
              json.loads((out / 'status.json').read_text()))
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o',
           'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=' + alias, address]
    state = dict(pid=os.getpid(), started_s=time.time(), target_host=hostname,
                 source_host=socket.gethostname(), phase='verify_target', complete=False,
                 gpu_actions=False, interrupts_existing_experiments=False)

    def update(**values):
        state.update(values, updated_s=time.time())
        write(out / 'status.json', state)

    def remote(script):
        result = subprocess.run(ssh + ['python3 -'], input=script, text=True,
                                capture_output=True, timeout=1200)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])
        return result.stdout

    try:
        update()
        observed = remote('import socket;print(socket.gethostname())\n').strip()
        if observed != hostname:
            raise RuntimeError('candidate host identity differs')
        update(phase='freeze_model_source')
        files = {}
        for path in sorted(MODEL.iterdir()):
            if not path.is_file():
                continue
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('model source changed while hashing')
            files[path.name] = dict(size=after.st_size, sha256=digest.hexdigest())
        write(out / 'model-source.json', dict(path=str(MODEL), files=files))
        temporary = str(MODEL.parent / '.A14B-sharegpt-slo90-v1-model')
        remote('from pathlib import Path\nPath(' + repr(temporary) + ').mkdir(parents=True,exist_ok=True)\n')
        update(phase='copy_model', bytes=sum(v['size'] for v in files.values()))
        import shlex
        with (out / 'model-copy.log').open('ab') as log:
            result = subprocess.run(['rsync', '-a', '--partial', '--exclude=*/',
                                     '-e', shlex.join(ssh[:-1]), str(MODEL) + '/',
                                     address + ':' + temporary + '/'], stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError('model copy failed; partial files retained')
        update(phase='verify_model')
        script = '''import hashlib,json,os
from pathlib import Path
files=FILES
target=Path(TARGET)
source=target if target.exists() else Path(TEMP)
for name, ref in files.items():
 p=source/name; h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 assert p.stat().st_size==ref['size'] and h.hexdigest()==ref['sha256'],name
reused=source==target
if not reused:
 assert not target.exists(),'target appeared during verification; refuse replacement'
 source.rename(target)
print(json.dumps({'verified':True,'target':str(target),'files':len(files),'reused_existing_after_full_sha256':reused}))
'''.replace('FILES', repr(files)).replace('TEMP', repr(temporary)).replace('TARGET', repr(str(MODEL)))
        write(out / 'model-target-verification.json', json.loads(remote(script)))
        update(phase='copy_image')
        source_image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE], text=True))[0]['Id']
        existing = subprocess.run(ssh + ['docker image inspect --format={{.Id}} ' + IMAGE],
                                  text=True, capture_output=True)
        if existing.returncode == 0 and existing.stdout.strip() == source_image:
            update(phase='complete', complete=True, image_id=source_image,
                   existing_image_verified=True, finished_s=time.time())
            return
        with (out / 'image-copy.log').open('ab') as log:
            producer = subprocess.Popen(['docker', 'image', 'save', IMAGE], stdout=subprocess.PIPE, stderr=log)
            consumer = subprocess.Popen(ssh + ['docker image load'], stdin=producer.stdout, stdout=log, stderr=log)
            producer.stdout.close()
            consumer_code = consumer.wait()
            producer_code = producer.wait()
        if consumer_code or producer_code:
            raise RuntimeError('image transport failed')
        target_image = subprocess.check_output(ssh + ['docker image inspect --format={{.Id}} ' + IMAGE], text=True).strip()
        if target_image != source_image:
            raise RuntimeError('image identity differs after transport')
        update(phase='complete', complete=True, image_id=source_image, finished_s=time.time())
    except BaseException as exc:
        update(phase='failed', error=repr(exc), finished_s=time.time())
        raise


if __name__ == '__main__':
    main()
