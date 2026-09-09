"""Read-only handoff gate: predecessor cleanup, current identity and a free lease."""
import json
from pathlib import Path
import shlex
import subprocess
import time
from operate import SSH

HERE=Path(__file__).resolve().parent
CODE='''import pathlib,json,time,hashlib,sys,importlib.util,asyncio,os,fcntl
root=pathlib.Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/C')
screen=root/'explore-p4-alpaca-r15';s=json.loads((screen/'status.json').read_text())
assert s['pid']==871452 and s['phase']=='complete' and s['node_lease_held'] is False
assert not pathlib.Path('/proc/871452').exists() and not s['failed']
assert len(s['completed'])==1
others=[]
for f in pathlib.Path('/proc').iterdir():
 if not f.name.isdigit() or int(f.name)==os.getpid():continue
 try:a=(f/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 except:continue
 if 'python' in a and '/campaign/' in a and '-c ' not in a and 'ecopadg.serving.engine' not in a:others.append(dict(pid=f.name,argv=a))
assert not others,'other campaign processes '+str(others)
read=lambda f:json.loads(pathlib.Path(f).read_text())
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
records=[];cps=sorted((screen/'results/checkpoints').glob('*.json'))
assert len(cps)==len(s['completed']) and {f.stem for f in cps}==set(s['completed'])
assert {f.name for f in (screen/'results/operations').iterdir() if f.is_dir()}=={f.stem for f in cps}
for f in cps:
 c=read(f);r=read(c['receipt']);assert sha(c['receipt'])==c['receipt_sha256']
 assert c['measurement_valid'] and r['measurement_valid'] and r['child_stopped'] and r['clock_restore_complete']
 assert not r['outer_cleanup_errors'] and r['summary']['post_measurement_cleanup']['cleanup_complete']
 assert all(sha(p)==h for p,h in c['artifacts'].items())
 records.append(dict(checkpoint=str(f),sha256=sha(f),receipt=c['receipt'],receipt_sha256=c['receipt_sha256']))
host='/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/hosts/7b-fixed-p4'
sys.path[:0]=[host+'/src',host,'/root/workspace/pdblend/.runtime-deps'];import aiohttp
commonpath='/root/workspace/pdblend-next-v1/campaign/five-system-execution-v3/run.py'
spec=importlib.util.spec_from_file_location('common',commonpath);common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
b=read('/root/workspace/pdblend-next-v1/campaign/main-rate-rerun-v1/C-attempt-001/binding.json')
async def observe():
 async with aiohttp.ClientSession(trust_env=False) as session:return await common.identity(session,b)
identity=asyncio.run(observe())
fd=os.open('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock',os.O_RDONLY)
try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(fd,fcntl.LOCK_UN)
finally:os.close(fd)
print(json.dumps(dict(captured_s=time.time(),passed=True,predecessor_status=s,verified_checkpoints=records,
                     identity=identity,node_lease_free=True,competing_campaign_processes=others)))
'''

if __name__=='__main__':
    p=subprocess.run(SSH+['python3 -B -c '+shlex.quote(CODE)],capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stderr
    v=json.loads(p.stdout);path=HERE/('explore-alp15-terminal-verified-'+str(time.time_ns())+'.json')
    with path.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(path),passed=v['passed'],checkpoints=len(v['verified_checkpoints']))))
