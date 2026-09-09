"""Close an appended terminal stdout record without changing prior restore raw."""
import asyncio,fcntl,hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
C=Path(__file__).resolve().parent;RESTORE=C/'baseline-after-gamma-restore-003';OUT=C/'restore003-closure-001';HOST=Path('/root/workspace/pdblend-next-v1/releases/five-system100-C7B-baseline-v1-runtime')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def check_log(raw,expected):
 assert expected==hashlib.sha256(b'').hexdigest(),'only the actually frozen empty stdout prefix is authorized'
 assert raw==b'{"restored": true, "measurement_valid": true, "fresh_27_required": true}\n','unexpected log tail: diagnostic required'
 return dict(original_prefix_bytes=0,original_prefix_sha256=expected,terminal_json=json.loads(raw),current_sha256=hashlib.sha256(raw).hexdigest(),exact_unique_terminal_record=True)
async def main():
 import aiohttp
 assert not OUT.exists();r=read(RESTORE/'deployment-receipt.json');sp=read(RESTORE/'deployment.json');pid=read(RESTORE/'launch.json')['pid'];assert not Path('/proc',str(pid)).exists();assert r['complete'] and r['measurement_valid'] and not r['errors']
 port=read(RESTORE/'startup-port-reservation.json');assert port['complete'] and port['restored'] and port['before']==port['after']
 log=str(RESTORE/'run.log');proof=check_log(Path(log).read_bytes(),r['artifacts'][log]);assert all(sha(p)==h for p,h in r['artifacts'].items() if p!=log)
 assert all(sha(p)==h for p,h in sp['files'].items())
 sys.path[:0]=[str(HOST/'src'),str(HOST),'/root/workspace/pdblend/.runtime-deps'];commonpath=C.parent/'common/execution-until-complete-v1/run.py';s=importlib.util.spec_from_file_location('restore_closure_common',commonpath);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 fd=os.open('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock',os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  assert Path('/proc/sys/net/ipv4/ip_local_reserved_ports').read_text().strip()==port['before'];names=(await m.command('docker','ps','--format','{{.Names}}')).split();assert set(names)==set(sp['expected_containers'])
  actual=json.loads(await m.command('docker','inspect',*names));before={x['Name']:x for x in read(RESTORE/'containers.after.json')};native={}
  async with aiohttp.ClientSession(trust_env=False) as session:
   for i in sp['instances']:
    x=next(x for x in actual if x['Name']=='/'+i['container_name']);y=before[x['Name']];assert x['Id']==y['Id'] and x['Image']==y['Image'] and x['State']['StartedAt']==y['State']['StartedAt'] and x['State']['Pid']==y['State']['Pid'] and x['State']['Running']
    p=await m.http(session,i,'/provenance');assert p==r['new_provenance'][i['id']];raw=await m.wait_idle(session,i);assert raw['accepting'] is True and raw['role']=='mixed' and raw['mode']=='continuous';native[i['id']]=dict(provenance=p,runtime=raw)
  OUT.mkdir();(OUT/'actual-identity-native.json').write_text(json.dumps(dict(containers=actual,native=native),indent=2)+'\n')
  result=dict(schema='C-restore-terminal-stdout-closure-v1',passed=True,created_s=time.time(),restore_receipt=ref(RESTORE/'deployment-receipt.json'),restore_spec=ref(RESTORE/'deployment.json'),startup_ports=ref(RESTORE/'startup-port-reservation.json'),identity=ref(OUT/'actual-identity-native.json'),actual_current_reserved_ports=port['before'],fresh_unique_lease_verified=True,old_restore_owner_exited=True,all_original_8_idle_same_PID_StartedAt=True,log=ref(Path(log)),stdout_append_proof=proof,other_artifacts_all_exact=True,original_artifacts={p:h for p,h in r['artifacts'].items() if p!=log},source=ref(Path(__file__).resolve()),common_source=ref(commonpath),old_receipt_and_raw_unchanged=True)
 finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
 result['node_lease_released']=True;(OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(ref(OUT/'receipt.json')))
if __name__=='__main__':asyncio.run(main())
