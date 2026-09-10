"""Read C phase status; expose terminal evidence hashes for bounded recovery."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import preflight as p

REMOTE=r'''
import hashlib,json,socket,sys
from pathlib import Path
d=json.load(sys.stdin)
assert socket.gethostname()==d['hostname']
path=Path(d['status'])
assert any(path.resolve().is_relative_to('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/C/'+name) for name in ('ascending-resume-20260909-v1','ascending-resume-20260909-v2'))
assert path.name=='status.json'
state=None;digest=None;alive=None;files=[]
if path.is_file():
    data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();state=json.loads(data)
    proc=Path('/proc')/str(state.get('pid'))/'stat'
    alive=proc.exists() and proc.read_text().rsplit(') ',1)[1].split()[0]!='Z'
    if state.get('finished_s') and not alive:
        for f in sorted(path.parent.rglob('*')):
            if not f.is_file() or not f.resolve().is_relative_to(path.parent.resolve()):continue
            size=f.stat().st_size
            if size>64*1024*1024:continue
            data=f.read_bytes();h=hashlib.sha256(data).hexdigest()
            files.append(dict(path=str(f),bytes=len(data),sha256=h))
print(json.dumps(dict(hostname=socket.gethostname(),status=state,status_sha256=digest,process_alive=alive,files=files)))
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host',default='47.106.163.29',type=p.safe_host)
    ap.add_argument('--user',default='root',type=p.safe_user)
    ap.add_argument('--identity-file',type=Path)
    ap.add_argument('--status',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists()
    request=dict(hostname=p.EXPECTED_HOSTS['C'],status=str(a.status))
    run=subprocess.run(p.ssh_argv(a,REMOTE),input=json.dumps(request),capture_output=True,text=True,timeout=45,check=False)
    assert run.returncode==0,run.stderr[-4000:]
    observed=json.loads(run.stdout);files=[]
    for item in observed.pop('files'):
        local=Path(item['path']);h=item['sha256']
        status=('match' if p.sha(local)==h else 'mismatch') if local.is_file() else 'missing'
        files.append(dict(path=item['path'],bytes=item['bytes'],actual_sha256=h,expected_sha256=h,
                          status='match',local_status=status,kinds=['new_phase_evidence']))
    snapshot=dict(schema='pdblend-C-phase-observation-v1',ssh_host=a.host,ssh_user=a.user,
                  captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_work_started=False,
                  remote_read_only=True,remote_files_written=False,
                  remote=dict(identity=dict(hostname=observed['hostname'],expected_hostname=p.EXPECTED_HOSTS['C'],
                                            hostname_matches=observed['hostname']==p.EXPECTED_HOSTS['C']),
                              files=files,**observed))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(snapshot,f,indent=2);f.write('\n')
    state=observed['status'] or {}
    print(json.dumps(dict(snapshot=str(a.out.resolve()),process_alive=observed['process_alive'],
                         complete=state.get('complete'),passed=state.get('passed'),phase=state.get('phase'),
                         error=state.get('error'),terminal_files=len(files))))


if __name__=='__main__':main()
