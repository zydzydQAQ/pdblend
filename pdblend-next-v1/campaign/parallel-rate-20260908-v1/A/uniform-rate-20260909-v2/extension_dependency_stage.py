"""Deploy absent exact Builder read dependencies without altering its frozen code."""
import hashlib, importlib.util, json, os, pathlib, shlex, subprocess, sys
U = pathlib.Path(__file__).resolve().parent
R = U.parents[1]
C = R / 'common/uniform-rate-20260909-v2'
sys.path.insert(0, str(C))
import generate
opened = set()
def hook(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(args[0]).resolve()
        mode = args[1]
        if str(path).startswith('/root/workspace/') and isinstance(mode, str) and ('r' in mode or mode == ''):
            opened.add(path)
sys.addaudithook(hook)
out = U / 'extension-builder-local-preflight-001'
b = generate.Builder(out)
snapshot = C.parent / 'uniform-rate-20260909-v1/reports/current/results.json'
snapshot_bytes = (out / 'previous-report-snapshot.json').read_bytes()
# Original snapshot's JSON serialization can differ; use the precise frozen copy
# as the remote read-only report input, with an explicit path mapping receipt.
opened.discard(snapshot)
files = sorted(p for p in opened if p.is_file() and not p.is_relative_to(out))
spec = importlib.util.spec_from_file_location('newA_deploy', R/'A/uniform-rate-20260909-v1/stage.py')
stage = importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
receipt = stage.stage(files)
payload = dict(path=str(snapshot), data=snapshot_bytes.decode(), sha256=hashlib.sha256(snapshot_bytes).hexdigest())
remote = '''import json,sys,pathlib,hashlib
v=json.load(sys.stdin);p=pathlib.Path(v['path']);data=v['data'].encode()
assert hashlib.sha256(data).hexdigest()==v['sha256']
if p.exists():
 old=p.read_bytes(); assert json.loads(old)==json.loads(data), 'existing snapshot has different content'
else:
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))'''
result = subprocess.run(stage.SSH + ['python3 -c '+shlex.quote(remote)], input=json.dumps(payload).encode(), capture_output=True, check=True)
receipt.update(snapshot_source=stage.sha(out/'previous-report-snapshot.json'), snapshot_target=json.loads(result.stdout), files={str(p):stage.sha(p) for p in files})
(out/'deployment-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
