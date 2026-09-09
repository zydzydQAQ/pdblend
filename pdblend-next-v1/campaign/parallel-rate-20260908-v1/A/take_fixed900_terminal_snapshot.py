"""Fresh read-only post-termination snapshot; never writes the old measurement."""
import asyncio,fcntl,hashlib,importlib.util,json,os,socket,subprocess,time
from pathlib import Path
A=Path(__file__).resolve().parent
ROOT=A.parent
OUT=A/'p6-fixed900-negative-diagnosis-001'
COMMON=ROOT/'common/execution-until-complete-v1/run.py'
def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def require(v,m):
 if not v:raise RuntimeError(m)
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False
async def main():
 import aiohttp
 spec=read(A/'p6-qualification900-inputs-002/fixed2/spec.json')
 base=read(spec['original_binding']['path']);old=A/'p6-qualification900-fixed2-002'
 status=read(old/'status.json');require(not alive(status['pid']) and status['cleanup_complete'] and not status['cleanup_errors'],'actual old owner must have finished cleanup')
 before=read(old/'identity.before.json')
 lock=Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')
 fd=os.open(lock,os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.time();s=importlib.util.spec_from_file_location('fixed900_terminal_common',COMMON);c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
 try:
  async with aiohttp.ClientSession(trust_env=False) as session:after=await c.identity(session,base)
  require(len(after)==len(before)==2,'exact two initial instances')
  byid={r['container']['Id']:r for r in before}
  for item in after:
   cur=item['container'];prior=byid[cur['Id']]['container']
   require(all(cur[k]==prior[k] for k in ['Id','Image']) and all(cur['State'][k]==prior['State'][k] for k in ['Pid','StartedAt']),'actual retained container/process changed')
   require(item['provenance']==byid[cur['Id']]['provenance'],'exact full provenance changed')
  clocks=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,clocks.current.sm,clocks.applications.graphics,pstate,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
  require(len(clocks.strip().splitlines())==8,'all eight current clocks must be observed')
  result=dict(schema='A-fixed900-fresh-readonly-terminal-snapshot-v1',passed=True,hostname=socket.gethostname(),pid=os.getpid(),started_s=started,finished_s=time.time(),fresh_post_termination_evidence=True,not_part_of_original_measurement=True,old_identity_after_not_created=True,read_only=True,node_lease_held_during_snapshot=True,lock_path=str(lock),lock_device=os.fstat(fd).st_dev,lock_inode=os.fstat(fd).st_ino,old_owner_exited=True,original_status=ref(old/'status.json'),original_before=ref(old/'identity.before.json'),original_binding=spec['original_binding'],actual_identity=after,all_eight_clock_snapshot_csv=clocks,files={r['path']:r['sha256'] for r in [ref(__file__),ref(COMMON),ref(old/'status.json'),ref(old/'identity.before.json'),spec['original_binding']]})
 finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
 result['node_lease_released']=True
 OUT.mkdir(exist_ok=False)
 p=OUT/'fresh-terminal-snapshot.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(ref(p)))
asyncio.run(main())
