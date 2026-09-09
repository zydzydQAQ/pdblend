import hashlib,json,os,subprocess,time
from pathlib import Path
C=Path('/root/workspace/pdblend-next-v1/campaign');O=C/'B32B-temporal-matched-shape-fresh-gate-launch-v1';G=C/'B32B-legacy-baseline-correctness-v1';B=C/'B32B-temporal-matched-shape-bootstrap-v1/binding.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(B)=='8afbe0d60ed5e86bff55ab27f29b2508bfc87d2be98e8cdc15c7680b1b73d014'
m=json.loads((G/'manifest.json').read_text())
for p,h in m['files'].items():assert sha(G/p)==h
assert not os.environ.get('PDBLEND_NODE_LOCK_FD')
HOST=C.parent/'releases/five-system100-B32B-v1-runtime';env=dict(os.environ,PYTHONPATH=f'{HOST}/src:{HOST}:/root/workspace/pdblend/.runtime-deps')
argv=['python3','-u',str(G/'validate.py'),'--binding',str(B),'--runtime-dir',str(C/'B32B-five-system100-baseline-deployment-v1/runtime'),'--out',str(C/'B32B-temporal-matched-shape-fresh-gate-v1')]
pre=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60)
(O/'default-check.json').write_text(json.dumps(dict(argv=argv,exit_code=pre.returncode,stdout=pre.stdout,stderr=pre.stderr),indent=2)+'\n');assert pre.returncode==0
assert not (O/'launch.json').exists();assert not Path(argv[-1]).exists()
argv+=['--run']
with (O/'run.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
 rec=dict(pid=p.pid,monitor_pid=os.getpid(),argv=argv,started_s=time.time(),binding_sha256=sha(B),gate_manifest_sha256=sha(G/'manifest.json'))
 (O/'launch.json').write_text(json.dumps(rec,indent=2)+'\n');code=p.wait()
 (O/'terminal.json').write_text(json.dumps(dict(rec,exit_code=code,finished_s=time.time(),original_exact_gate_not_rewritten=True),indent=2)+'\n')
