"""Stage an explicit frozen spec and its files; create missing paths only."""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import preflight as p

REMOTE = r'''
import base64,hashlib,json,socket,sys
from pathlib import Path
d=json.load(sys.stdin)
assert socket.gethostname()==d['hostname'], 'wrong physical hostname'
allowed=('/root/workspace/pdblend-next-v1','/root/workspace/pdblend/new-results')
rows=[]
for x in d['files']:
    path=Path(x['path'])
    assert path.is_absolute() and any(path.resolve().is_relative_to(root) for root in allowed), 'unsafe destination'
    if path.exists():
        assert path.is_file(), 'destination is not a file'
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual==x['sha256'], 'existing remote bytes differ: '+str(path)
        rows.append(dict(path=str(path),sha256=actual,status='same'))
    else:
        rows.append(dict(path=str(path),sha256=x['sha256'],status='missing'))
if d['action']=='create':
    decoded=[]
    for x,row in zip(d['files'],rows):
        if row['status']=='same':continue
        data=base64.b64decode(x['data'],validate=True)
        assert len(data)==x['bytes'] and hashlib.sha256(data).hexdigest()==x['sha256'], 'payload differs'
        decoded.append((Path(x['path']),data,row))
    for path,data,row in decoded:
        assert any(path.resolve().is_relative_to(root) for root in allowed), 'destination changed'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        row['status']='created'
else:
    assert d['action']=='inspect'
print(json.dumps(dict(hostname=socket.gethostname(),files=rows,gpu_work_started=False,existing_files_overwritten=False)))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True,type=Path)
    parser.add_argument('--host',required=True,type=p.safe_host)
    parser.add_argument('--user',required=True,type=p.safe_user)
    parser.add_argument('--identity-file',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--stage',action='store_true')
    args=parser.parse_args()
    if args.out.exists():parser.error('new audit path required')
    spec=p.read(args.spec);files=dict(spec['files']);files[str(args.spec.resolve())]=p.sha(args.spec)
    for path,digest in files.items():
        assert any(Path(path).resolve().is_relative_to(root) for root in ('/root/workspace/pdblend-next-v1','/root/workspace/pdblend/new-results')), path
        assert p.sha(path)==digest, path
    request=dict(hostname=spec['hostname'],action='inspect',files=[dict(path=path,sha256=h) for path,h in sorted(files.items())])
    audit=dict(schema='pdblend-create-missing-stage-v1',started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               spec=dict(path=str(args.spec.resolve()),sha256=p.sha(args.spec)),host=args.host,user=args.user,
               local_files_verified=len(files),gpu_work_started=False,existing_remote_files_overwritten=False,
               remote_stage_started=False,complete=False)
    def remote(payload):
        r=subprocess.run(p.ssh_argv(args,REMOTE),input=json.dumps(payload),capture_output=True,text=True,timeout=180,check=False)
        assert r.returncode==0,r.stderr[-5000:]
        return json.loads(r.stdout)
    error=None
    try:
        if args.stage:
            before=remote(request);audit['before']=before
            missing=[x for x in before['files'] if x['status']=='missing']
            payload=[]
            total=0
            for x in missing:
                data=Path(x['path']).read_bytes();total+=len(data)
                assert hashlib.sha256(data).hexdigest()==x['sha256']
                payload.append(dict(path=x['path'],sha256=x['sha256'],bytes=len(data),data=base64.b64encode(data).decode()))
            assert total<=128*1024*1024, 'bounded staging size exceeded'
            if payload:
                audit['remote_stage_started']=True
                audit['created']=remote(dict(hostname=spec['hostname'],action='create',files=payload))
            final=remote(request);audit['final']=final
            assert all(x['status']=='same' for x in final['files'])
            audit['complete']=True
    except (OSError,ValueError,AssertionError,subprocess.TimeoutExpired) as exc:
        error=str(exc);audit['error']=error
    audit['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(audit,f,indent=2);f.write('\n')
    print(json.dumps(dict(audit=str(args.out.resolve()),complete=audit['complete'],
                         local_files_verified=len(files),created=len(audit.get('created',{}).get('files',[])),error=error)))
    return 2 if error else 0


if __name__=='__main__':sys.exit(main())
