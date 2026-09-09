"""Apply the user's reduced scope through the running runner's STOP boundary."""
import json
from pathlib import Path
import time
from operate import remote,write_new

HERE=Path(__file__).resolve().parent
while time.time()<1788872770:
    code='''import json,pathlib,time,hashlib
root=pathlib.Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/C')
s=json.loads((root/'screen-p4/status.json').read_text());result=dict(captured_s=time.time(),phase=s['phase'],completed=len(s['completed']),current_cell=s.get('current_cell'),stopped=False)
if s['phase']=='running' and s.get('current_cell')=='parallel-rate-p4-fixed2-7b-longbench-r3-s701-w100-pdblend-slo1-repeat1':
 assert s['pid']==829075 and s['node_lease_held'] is True and len(s['completed'])==5
 runner=root/'workflow-p4-strict/runner.py';release=json.loads((root/'fixed-release-p4-strict/release.json').read_text())
 assert hashlib.sha256(runner.read_bytes()).hexdigest()==release['files'][str(runner)]
 assert str(runner).encode() in pathlib.Path('/proc/829075/cmdline').read_bytes().split(bytes([0]))
 stop=root/'workflow-p4-strict/STOP'
 text='User reduced scope: finish current first-repeat LongBench r3 full measurement and cleanup, then stop before repeat2. No higher-rate search after first complete SLO<0.90. New-rate interpolation and baseline repeat queues remain unexecuted. No signal or child interruption. Issued '+time.strftime('%Y-%m-%d %H:%M:%S CST')+'\\n'
 with stop.open('x') as f:f.write(text)
 result.update(stopped=True,stop=str(stop),stop_sha256=hashlib.sha256(stop.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),text=text)
print(json.dumps(result))'''
    raw=remote(code);state=json.loads(raw)
    if state['stopped'] or state['phase'] not in ['starting','running']:
        write_new(HERE/'reduced-scope-p4-boundary-receipt.json',raw)
        print(raw.decode(),flush=True);break
    time.sleep(5)
