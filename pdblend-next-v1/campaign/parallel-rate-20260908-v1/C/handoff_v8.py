"""Request the existing v8 queue's declared cell-boundary stop; never signal a child."""
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
SSH=['ssh','-oBatchMode=yes','-oConnectTimeout=8','-oStrictHostKeyChecking=yes',
     '-oHostKeyAlias=47.106.163.29','172.16.50.105']
CODE='''import ast,pathlib,json,time,hashlib,os
root=pathlib.Path('/root/workspace/pdblend-next-v1/campaign/main-slo-improvement-v8')
owned=pathlib.Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/C')
status=root/'C/fixed-screen-001/status.json';s=json.loads(status.read_text());pid=s['pid']
assert pid==787021 and s['phase']=='running' and not s['complete']
proc=pathlib.Path('/proc')/str(pid);argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(bytes([0])) if v]
assert argv==['python3',str(root/'runner.py'),'--release',str(root/'C/fixed-release-001/release.json'),'--out',str(root/'C/fixed-screen-001'),'--stage','screen_fixed2','--run']
start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
source=(root/'runner.py').read_text();tree=ast.parse(source)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
handler=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='stop')
assert len(handler.body)==1 and isinstance(handler.body[0],ast.Assign)
assert ast.unparse(handler.body[0])=='args.stop_requested = True'
execute=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='execute')
loop=next(n for n in ast.walk(execute) if isinstance(n,ast.For) and ast.unparse(n.target)=='cell')
guard=loop.body[0];assert isinstance(guard,ast.If)
assert "(p.ROOT / 'STOP').exists()" in ast.unparse(guard.test)
assert any(isinstance(n,ast.Break) for n in guard.body)
assert 'receipt = await executor.run_one' in source and "state['completed'].append(cell['cell_id'])" in source
stop=root/'STOP';assert not stop.exists(),'existing unrelated STOP must not be overwritten'
text='User priority update 2026-09-08: finish the current v8 cell and its checkpoint/native cleanup, then stop before the next cell. Transfer C ownership to parallel-rate-20260908-v1 unified p1; no child interruption. Deadline 21:06:10 CST.\\n'
with stop.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
receipt=dict(written_s=time.time(),authorization='Current user explicitly authorized unified two-fix p1 and cell-boundary handoff',
 pid=pid,start_ticks=start_ticks,argv=argv,current_cell=s['current_cell'],completed=len(s['completed']),
 stop_path=str(stop),stop_sha256=hashlib.sha256(stop.read_bytes()).hexdigest(),
 runner_path=str(root/'runner.py'),runner_sha256=hashlib.sha256(source.encode()).hexdigest(),
 protocol_sha256=hashlib.sha256((root/'protocol.py').read_bytes()).hexdigest(),
 ast_verified_boundary_only=True,child_signalled=False,frozen_source_changed=False)
with (owned/'v8-boundary-stop-receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''

if __name__=='__main__':
    p=subprocess.run(SSH+['python3 -B -c '+shlex.quote(CODE)],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    result=json.loads(p.stdout)
    with (HERE/'v8-boundary-stop-receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
