"""CPU-only real parent FLOCK inheritance and authority negative controls."""
import copy,fcntl,importlib.util,json,os,subprocess,sys,tempfile
from pathlib import Path
F=Path(__file__).resolve().parent;sys.path.insert(0,str(F))
import dynamic_measurement as m

def main():
 with tempfile.TemporaryDirectory(prefix='uniform-authority-test-') as directory:
  d=Path(directory);lock=d/'test.lock';handle=lock.open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  st=lock.stat();lease=dict(fd=handle.fileno(),holder_pid=os.getpid(),holder_start_ticks=m.ownership.start_ticks(os.getpid()),device=st.st_dev,inode=st.st_ino)
  old=m.ownership.LOCK;m.ownership.LOCK=lock
  try:
   cfg=d/'template.json';cfg.write_text(json.dumps(dict(strategy='pdblend-v1',slo_ttft_s=1.,max_service_frequency_mhz=2100,capacity_inventory_path=str(d/'inventory.json'))))
   binding=dict(configs=dict(alpaca=str(cfg)),instances=[dict(id='actual-native-placeholder')]);row=dict(cell_id='cpu-control',dataset='alpaca');operation=d/'operation';operation.mkdir()
   actual=m.bind_parent_authority(binding,row,operation,str(cfg),d/'inventory.json',lease)
   value=json.loads(Path(actual).read_text());base=json.loads(cfg.read_text())
   assert {k:v for k,v in value.items() if k not in ('capacity_job_path','capacity_lease_authority')}==base
   assert json.loads((operation/'invocation.json').read_text())['binding']==binding
   module=F.parents[2]/'hosts/14b-capacity-p12/capacity_executor.py'
   code='''import importlib.util,json,sys
spec=importlib.util.spec_from_file_location("cpu_real_lease",sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=json.load(open(sys.argv[2]));kw=dict(authority=c['capacity_lease_authority'],expected_inventory=c['capacity_inventory_path'],expected_job_path=c['capacity_job_path'])
m.check_lease(sys.argv[3],**kw)
for key in ('expected_inventory','expected_job_path'):
 bad=dict(kw);bad[key]+='-foreign'
 try:m.check_lease(sys.argv[3],**bad)
 except (RuntimeError,ValueError):pass
 else:raise AssertionError('foreign invocation accepted')
ref=dict(kw['authority']);ref['sha256']='0'*64
try:m.check_lease(sys.argv[3],**dict(kw,authority=ref))
except (RuntimeError,ValueError):pass
else:raise AssertionError('tampered authority accepted')
print('real parent inherited lease; wrong inventory/job/hash rejected')
'''
   child=subprocess.run([sys.executable,'-B','-c',code,str(module),actual,str(lock)],pass_fds=(handle.fileno(),),text=True,capture_output=True)
   assert child.returncode==0,child.stderr
   print(child.stdout.strip())
  finally:m.ownership.LOCK=old;handle.close()
if __name__=='__main__':main()
