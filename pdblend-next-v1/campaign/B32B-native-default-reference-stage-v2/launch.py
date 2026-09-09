import hashlib,json,os,subprocess,sys,time
from pathlib import Path
BASE=Path('/root/workspace/pdblend-next-v1');STAGE=BASE/'campaign/B32B-native-default-reference-stage-v2'
PACKAGE=BASE/'campaign/B32B-native-default-reference-execution-v2';ATTEMPT=BASE/'campaign/B32B-native-default-reference-attempt-002'
MODE=sys.argv[1];assert MODE in ('prepare','run')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(PACKAGE/'manifest.json')=='b4e299bc703e2146582d3180d01d41eddb60dac2b91394033f5d0a1b66e7d4ad'
assert not os.environ.get('PDBLEND_NODE_LOCK_FD')
HOST=BASE/'releases/five-system100-B32B-v1-runtime';env=dict(os.environ,PYTHONPATH=f'{HOST}/src:{HOST}:/root/workspace/pdblend/.runtime-deps')
argv=['python3','-u',str(PACKAGE/'run.py')]
if MODE=='prepare':argv+=['--prepare','--out',str(ATTEMPT)]
else:
 assert json.loads((STAGE/'prepare-terminal.json').read_text())['exit_code']==0
 argv+=['--run','--spec',str(ATTEMPT/'spec.json'),'--spec-sha256',sha(ATTEMPT/'spec.json')]
launch=STAGE/(MODE+'-launch.json');assert not launch.exists()
with (STAGE/(MODE+'.log')).open('xb') as log:
 p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,close_fds=True,start_new_session=True)
 rec=dict(pid=p.pid,monitor_pid=os.getpid(),argv=argv,started_s=time.time(),mode=MODE)
 launch.write_text(json.dumps(rec,indent=2)+'\n');code=p.wait()
 (STAGE/(MODE+'-terminal.json')).write_text(json.dumps(dict(rec,exit_code=code,finished_s=time.time()),indent=2)+'\n')
