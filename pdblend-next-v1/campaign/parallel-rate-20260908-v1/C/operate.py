"""Owned C deployment and append-only result capture, with exact file hashes."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
SSH = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=8', '-oStrictHostKeyChecking=yes',
       '-oHostKeyAlias=47.106.163.29', '172.16.50.105']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def remote(code, data=None):
    r = subprocess.run(SSH + ['python3 -B -c ' + shlex.quote(code)], input=data,
                       capture_output=True, timeout=180)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors='replace')[-5000:])
    return r.stdout

def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, 'refuse to overwrite: ' + str(path)
    else:
        with path.open('xb') as f: f.write(data)

def stage(release_path):
    r = json.loads(release_path.read_text())
    base = json.loads(Path(r['binding']['path']).read_text())
    files = dict(base['files']); files.update(r['files'])
    for ref in [r[k] for k in ['binding', 'declaration', 'qualification']]: files[ref['path']] = ref['sha256']
    files[str(release_path)] = sha(release_path)
    d = json.loads(Path(r['declaration']['path']).read_text())
    for c in d['cells']: files[c['trace']['path']] = c['trace']['sha256']
    assert all(sha(p) == h for p, h in files.items())
    inspect = '''import sys,json,pathlib,hashlib
files=json.load(sys.stdin);out=dict(missing=[],different=[],same=0)
for p,h in files.items():
 f=pathlib.Path(p)
 if not f.exists():out['missing'].append(p)
 elif hashlib.sha256(f.read_bytes()).hexdigest()!=h:out['different'].append(p)
 else:out['same']+=1
print(json.dumps(out))'''
    audit = json.loads(remote(inspect, json.dumps(files).encode()))
    assert not audit['different'], audit
    missing = {p: files[p] for p in audit['missing']}
    out = HERE / ('stage-' + str(time.time_ns()) + '.json')
    write_new(out, json.dumps(dict(captured_s=time.time(), release=str(release_path), files=files,
                                 inspection=audit), indent=2).encode())
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as tar:
        for p in missing:
            assert p.startswith('/root/workspace/pdblend-next-v1/')
            tar.add(p, arcname=p.lstrip('/'), recursive=False)
    extract = '''import pathlib,sys,tarfile,hashlib
expected=''' + repr(missing) + '''
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for member in tar:
  dest=pathlib.Path('/')/member.name
  assert member.isfile() and str(dest) in expected
  data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==expected[str(dest)]
  dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as f:f.write(data)
print('staged')'''
    remote(extract, archive.getvalue())
    verify = json.loads(remote(inspect, json.dumps(files).encode()))
    assert not verify['different'] and not verify['missing'], verify
    print(json.dumps(dict(staged=len(missing), files_verified=len(files), audit=str(out))))

def launch(release_path, output):
    code = '''import subprocess,json,pathlib,time,os
release=pathlib.Path(RELEASE);out=pathlib.Path(OUTPUT);root=pathlib.Path(ROOT)
assert not out.exists()
cmd=['python3','-B',str(root/'runner.py'),'--release',str(release),'--out',str(out),'--stage','screen_fixed2']
check=subprocess.run(cmd,text=True,capture_output=True);assert check.returncode==0,check.stderr
with (root/(out.name+'.log')).open('xb') as log:
 p=subprocess.Popen(cmd+['--run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/root/workspace/pdblend-next-v1')
v=dict(pid=p.pid,argv=cmd+['--run'],started_s=time.time(),defaultcheck=check.stdout)
with (root/(out.name+'.launch.json')).open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))'''
    head = 'RELEASE=' + repr(str(release_path)) + '\nOUTPUT=' + repr(str(output)) + '\nROOT=' + repr(str(HERE)) + '\n'
    data = remote(head + code)
    write_new(HERE / (output.name + '.launch.json'), data)
    print(data.decode())

def observe(output):
    code = '''import pathlib,json,time
out=pathlib.Path(OUTPUT);f=out/'status.json';d=json.loads(f.read_text()) if f.exists() else {'phase':'not_created'}
print(json.dumps(dict(captured_s=time.time(),status=d,process_alive=pathlib.Path('/proc/'+str(d.get('pid'))).exists())))'''
    data = remote('OUTPUT=' + repr(str(output)) + '\n' + code)
    v=json.loads(data);s=v['status'];print(json.dumps({**{k:s.get(k) for k in ['pid','phase','complete','current_cell','error','engineering_gate_failed']},
                                                   'completed':len(s.get('completed',[])), 'alive':v['process_alive'],'captured_s':v['captured_s']}))
    write_new(HERE / 'observations' / (str(time.time_ns()) + '.json'), data)

def sync(output):
    known={str(f):sha(f) for f in output.rglob('*') if f.is_file() and f.name!='status.json'}
    code='''import json,sys,pathlib,hashlib,base64,time
out=pathlib.Path(OUTPUT);known=json.load(sys.stdin);files=set()
for cp in (out/'results/checkpoints').glob('*.json'):
 c=json.loads(cp.read_text());files.update([cp,pathlib.Path(c['binding'])])
 for p,h in c['artifacts'].items():
  f=pathlib.Path(p);assert hashlib.sha256(f.read_bytes()).hexdigest()==h;files.add(f)
for dirname in ['setup','results/engineering-gates']:
 files.update(f for f in (out/dirname).rglob('*') if f.is_file())
for name in ['release-reference.json','declaration-order.json']:
 f=out/name
 if f.exists():files.add(f)
rows=[]
for f in sorted(files):
 data=f.read_bytes();h=hashlib.sha256(data).hexdigest();assert out in f.parents
 if str(f) in known:assert known[str(f)]==h
 rows.append(dict(path=str(f),sha256=h,data=base64.b64encode(data).decode() if known.get(str(f))!=h else None))
print(json.dumps(dict(captured_s=time.time(),status=(out/'status.json').read_text(),rows=rows)))'''
    data=remote('OUTPUT='+repr(str(output))+'\n'+code,json.dumps(known).encode());v=json.loads(data);count=0
    for r in v['rows']:
        if r['data'] is not None:
            b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256'];write_new(Path(r['path']),b);count+=1
    write_new(HERE/'observations'/(str(time.time_ns())+'-status.json'),v['status'].encode())
    write_new(HERE/'captures'/(str(time.time_ns())+'.json'),json.dumps(dict(captured_s=v['captured_s'],files={r['path']:r['sha256'] for r in v['rows']}),indent=2).encode())
    print(json.dumps(dict(new_files=count,status=json.loads(v['status']).get('phase'))))

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['stage','launch','observe','sync']);a.add_argument('--release',type=Path,default=HERE/'fixed-release-p1/release.json');a.add_argument('--out',type=Path,default=HERE/'screen-p1');v=a.parse_args()
    {'stage':lambda:stage(v.release),'launch':lambda:launch(v.release,v.out),'observe':lambda:observe(v.out),'sync':lambda:sync(v.out)}[v.action]()

if __name__=='__main__':main()
