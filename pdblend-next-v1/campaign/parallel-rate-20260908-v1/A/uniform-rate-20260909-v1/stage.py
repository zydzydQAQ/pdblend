"""Copy only absent pinned experiment dependencies to new A, never overwrite."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SSH = ['ssh', '-o', 'StrictHostKeyChecking=yes', '-o', 'BatchMode=yes', 'root@120.79.123.62']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(files):
    expected = {str(path): sha(path) for path in files}
    check = '''import json,pathlib,hashlib,sys
expected=json.load(sys.stdin);missing=[]
for name,digest in expected.items():
 p=pathlib.Path(name)
 if p.exists():assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest, 'changed existing file: '+name
 else:missing.append(name)
print(json.dumps(missing))'''
    result = subprocess.run(SSH + ['python3 -c ' + shlex.quote(check)], input=json.dumps(expected).encode(),
                            capture_output=True, check=True)
    missing = json.loads(result.stdout)
    subset = {name: expected[name] for name in missing}
    if not missing:
        return dict(checked=len(files), copied=0)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as archive:
        for path in missing:
            archive.add(path, arcname=path.lstrip('/'))
    receive = '''import sys,tarfile,pathlib,hashlib,json
expected=EXPECTED
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|*') as archive:
 for item in archive:
  path=pathlib.Path('/'+item.name)
  assert item.isfile() and str(path) in expected
  data=archive.extractfile(item).read()
  assert hashlib.sha256(data).hexdigest()==expected[str(path)]
  path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():assert path.read_bytes()==data
  else:
   with path.open('xb') as stream:stream.write(data)
print(json.dumps({'copied':len(expected)}))'''.replace('EXPECTED', repr(subset))
    subprocess.run(SSH + ['python3 -c ' + shlex.quote(receive)], input=buf.getvalue(), check=True)
    return dict(checked=len(files), copied=len(missing), transport_bytes=len(buf.getvalue()))


def dependencies():
    common = ROOT / 'common/uniform-rate-20260909-v1'
    d = json.loads((common / 'release-001/declaration.json').read_text())
    paths = {p for p in common.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and (p.is_relative_to(common / 'release-001') or p.parent == common and p.suffix == '.py')}
    paths.update((ROOT / 'C/uniform-rate-20260909-v1').glob('*.py'))
    for row in d['cells']:
        if row['model'] == '14b':
            paths.add(Path(row['trace']))
            paths.add(Path(row['source_300s_trace']['path']))
    for item in d['reconstructed_missing_sources']:
        source, original = item['exact_local_copy'], item['original_reference']
        assert sha(source['path']) == source['sha256'] == original['sha256']
        path = Path(original['path'])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(Path(source['path']).read_bytes())
        assert sha(path) == original['sha256']
        paths.add(path)
    return sorted(paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies', action='store_true')
    parser.add_argument('files', nargs='*', type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(dependencies() if args.dependencies else args.files)))
