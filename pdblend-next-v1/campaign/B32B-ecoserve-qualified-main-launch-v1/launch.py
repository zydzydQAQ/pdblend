"""One original Eco main invocation. Only the actual-host index is published."""
import hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
C=Path('/root/workspace/pdblend-next-v1/campaign');S=C/'B32B-ecoserve-qualified-main-launch-v1'
B=C/'B32B-ecoserve-qualified-main-v1/binding.json';SOURCE=C/'five-system-fixed-window-v1/sources/B32B/manifest.json';RUN=C/'five-system-execution-v3/run.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
b=json.loads(B.read_text());assert b['system']=='ecoserve' and b['correctness_protocol_id']=='legacy-temporal-default-trajectory-exact-v2' and b['output_correctness_verified'] is True and b['legacy_single_vs_pair_exact'] is False
assert not Path(b['output']).exists() and not (S/'launch.json').exists() and not os.environ.get('PDBLEND_NODE_LOCK_FD')
assert not any(Path('/proc',str(pid)).exists() for pid in [495118,495120,488364,488365,383819,420140])
pubpath=C/'AC-baseline-sequence-v1/publish.py';assert sha(pubpath)=='46564c454c0a62b7da0b86c218f25789253ccc9a91500aacc07b0644ca6fe41f'
sp=importlib.util.spec_from_file_location('eco_actual_index_publish',pubpath);pub=importlib.util.module_from_spec(sp);sp.loader.exec_module(pub)
H=Path(b['host_release']);env=dict(os.environ,PYTHONPATH=f'{H}/src:{H}:/root/workspace/pdblend/.runtime-deps')
argv=['python3','-u',str(RUN),'--manifest',str(SOURCE),'--binding',str(B),'--system','ecoserve','--phase','main','--max-cells','30']
pre=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60);write(S/'default-check.json',dict(argv=argv,exit_code=pre.returncode,stdout=pre.stdout,stderr=pre.stderr));assert pre.returncode==0
argv+=['--run']
def publish(pid,exitcode=None):
 r=pub.update(B,SOURCE,pid,'main',list(b['configs']),S/'status.json',exitcode)
 p=C/'current-experiment.json';v=json.loads(p.read_text());v.update(correctness_protocol_id=b['correctness_protocol_id'],qualification=b['qualification'],legacy_output_correctness_verified=False,legacy_single_vs_pair_exact=False)
 tmp=p.with_suffix('.tmp');write(tmp,v);tmp.replace(p)
 with (C/'CURRENT_EXPERIMENT.md').open('a') as f:f.write('\nCorrectness protocol: '+b['correctness_protocol_id']+'. Original cross-shape exact failure retained; this qualification uses an independently generated native reference with the same prescribed scheduling trajectory.\n')
 r['sha256']=sha(p);write(S/('index-running.json' if exitcode is None else 'index-terminal.json'),r)
with (S/'run.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
 rec=dict(pid=p.pid,monitor_pid=os.getpid(),argv=argv,started_s=time.time(),binding_sha256=sha(B),source_sha256=sha(SOURCE),qualification=b['qualification'])
 write(S/'launch.json',rec);write(S/'status.json',dict(rec,phase='main',running=True))
 try:publish(p.pid)
 except Exception as exc:write(S/'index-error.json',dict(error=repr(exc)))
 code=p.wait();write(S/'terminal.json',dict(rec,exit_code=code,finished_s=time.time()));write(S/'status.json',dict(rec,phase='main',running=False,exit_code=code,finished_s=time.time()))
 try:publish(p.pid,code)
 except Exception as exc:write(S/'index-terminal-error.json',dict(error=repr(exc)))
