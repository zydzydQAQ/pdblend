import hashlib,json,subprocess,time,urllib.request
from pathlib import Path
C=Path('/root/workspace/pdblend-next-v1/campaign');N=C/'B32B-native-default-reference-attempt-002';O=C/'B32B-native-default-reference-audit-v2';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((N/'results/status.json').read_text());assert all(s[k] is True for k in ['complete','observation_completed','measurement_valid','all_original_restored','clock_restore_complete'])
b=json.loads((N/'results/restored-bootstrap.binding.json').read_text());names=[i['container']['name'] for i in b['instances']]
containers=json.loads(subprocess.check_output(['docker','inspect',*names,'pdb-v2-nativeref100b2']))
for i,x in zip(b['instances'],containers):
 assert x['Id']==i['container']['id'] and x['Image']==i['container']['image'] and x['State']['StartedAt']==i['container']['StartedAt'] and x['State']['Running'] and x['State']['Pid']>0
assert containers[-1]['State']['Running'] is False and containers[-1]['State']['Pid']==0
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));runtimes={}
for i in b['instances']:
 with opener.open(i['url']+'/runtime',timeout=5) as r:v=json.load(r)
 assert v['accepting'] is True and v['active']==0 and v['running']==0 and v['waiting']==0 and v['mode']=='continuous' and v['role']=='mixed'
 runtimes[i['id']]=v
pids=[488230,488229,488365,488364,490460]
live={str(p):Path('/proc',str(p)).exists() for p in pids};assert not any(live.values())
proof=dict(observed_s=time.time(),processes_live=live,containers=containers,runtimes=runtimes,bootstrap_sha256=sha(N/'results/restored-bootstrap.binding.json'))
(O/'actual-terminal.json').write_text(json.dumps(proof,indent=2)+'\n')
g=json.loads((C/'B32B-native-default-reference-stage-v2/guards-before.json').read_text());actual={p:sha(p) for p in g};changes={p:[h,actual[p]] for p,h in g.items() if h!=actual[p]}
(C/'B32B-native-default-reference-stage-v2/guards-after-operation.json').write_text(json.dumps(dict(observed_s=time.time(),count=len(g),changed=changes,sha256=actual),indent=2)+'\n');assert not changes
print(json.dumps(dict(actual_terminal_sha256=sha(O/'actual-terminal.json'),all_processes_exited=True,original_host_pids=[x['State']['Pid'] for x in containers[:4]],original_idle_accepting=True,guards=len(g),unchanged=True)))
