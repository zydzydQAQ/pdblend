"""PowerSampler-compatible transport; raw collection stays in its own process."""
import hashlib,json,os,subprocess,sys,threading,time,uuid
from pathlib import Path

_CONTEXT=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def checked(ref):
 if sha(ref['path'])!=ref['sha256']:raise RuntimeError('isolated measurement reference changed')
 return json.loads(Path(ref['path']).read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def install(artifact_root,host_manifest_ref,adapter_manifest_ref):
 global _CONTEXT
 manifest=checked(adapter_manifest_ref)
 if not all(sha(p)==h for p,h in manifest['files'].items()):raise RuntimeError('isolated sampler source changed')
 checked(host_manifest_ref)
 new=(Path(artifact_root),host_manifest_ref,adapter_manifest_ref)
 if _CONTEXT is not None and _CONTEXT!=new:raise RuntimeError('sampler already bound to a different invocation')
 _CONTEXT=new
 import ecopadg.measure.power as module
 if module.PowerSampler is not IsolatedPowerSampler:
  expected=Path(host_manifest_ref['path']).parent/'src/ecopadg/measure/power.py'
  if Path(module.__file__).resolve()!=expected.resolve():raise RuntimeError('wrong actual measurement source')
  module.PowerSampler=IsolatedPowerSampler
 return True

class _Liveness:
 def __init__(self,owner):self.owner=owner
 def is_alive(self):
  p=self.owner._process;t=self.owner._reader
  return bool((p is not None and p.poll() is None) or (t is not None and t.is_alive()))

class IsolatedPowerSampler:
 def __init__(self,gpus,interval=.02,backend=None,clock=time.time,sample_clocks=False):
  if _CONTEXT is None:raise RuntimeError('explicit isolated sampler binding required')
  if list(gpus)!=list(range(8)) or interval!=.02 or clock is not time.time:raise RuntimeError('original all8 50Hz wall-clock measurement required')
  if backend is None or backend.power_source.get('mode')!='instant':raise RuntimeError('original explicit instantaneous backend required')
  self.gpus=list(gpus);self.interval=interval;self.backend=backend;self.sample_clocks=sample_clocks
  self.power_source=dict(backend.power_source);self.samples=[];self.utilization_samples=[];self.frequency_samples=[];self.power_metadata=[]
  self.error=None;self._process=None;self._reader=None;self._thread=_Liveness(self);self._terminal=None;self._header=False;self._digest=hashlib.sha256();self._directory=None;self._stopped=False
 def _failure(self,message):self.error=(self.error+'; ' if self.error else '')+str(message)
 def start(self):
  if self._process is not None:raise RuntimeError('isolated sampler object is single-use')
  root,host,adapter=_CONTEXT;self._directory=root/('sampler-'+uuid.uuid4().hex);self._directory.mkdir(parents=True,exist_ok=False)
  spec=dict(host_manifest=host,adapter_manifest=adapter,gpus=self.gpus,interval=self.interval,sample_clocks=self.sample_clocks,read_only=True)
  write(self._directory/'spec.json',spec)
  stderr=(self._directory/'stderr.log').open('xb')
  self._process=subprocess.Popen([sys.executable,'-I',str(Path(__file__).with_name('sampler_worker.py')),'--spec',str(self._directory/'spec.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr)
  stderr.close();write(self._directory/'launch.json',dict(pid=self._process.pid,parent_pid=os.getpid(),started_s=time.time(),read_only=True,affinity=sorted(os.sched_getaffinity(0)),independent_interpreter=True))
  self._reader=threading.Thread(target=self._consume,daemon=True);self._reader.start()
 def _accept(self,line):
  row=json.loads(line);kind=row['kind']
  if self._terminal is not None:raise RuntimeError('IPC contains rows after terminal')
  if kind=='ready':
   if self._header or self.samples or row['pid']!=self._process.pid or row['source']!=self.power_source or row['host_manifest']!=_CONTEXT[1] or row['read_only'] is not True:raise RuntimeError('isolated worker identity/source differs')
   self._header=True
  elif kind=='row':
   if not self._header or row['index']!=len(self.samples):raise RuntimeError('missing, duplicated or reordered power row')
   sample=row['power'];metadata=row['metadata']
   if len(sample)!=2 or len(sample[1])!=8 or metadata.get('gpus')!=list(range(8)) or metadata.get('t_s')!=sample[0]:raise RuntimeError('power row/GPU/metadata mismatch')
   self.power_metadata.append(metadata)
   if row['utilization'] is not None:self.utilization_samples.append(row['utilization'])
   if row['frequency'] is not None:self.frequency_samples.append(row['frequency'])
   self.samples.append(sample)
  elif kind=='terminal':
   counts=dict(samples=len(self.samples),metadata=len(self.power_metadata),utilization=len(self.utilization_samples),frequency=len(self.frequency_samples))
   if not self._header or row['pid']!=self._process.pid or row['counts']!=counts or row['emitted']!=len(self.samples) or row['stream_sha256']!=self._digest.hexdigest():raise RuntimeError('isolated terminal snapshot is incomplete or changed')
   self._terminal=row
   if row['sampler_thread_stopped'] is not True or row['error']:self._failure('worker sampling failed: '+str(row['error']))
   return
  else:raise RuntimeError('unknown isolated IPC row')
  self._digest.update(line)
 def _consume(self):
  try:
   for line in self._process.stdout:
    if not line.endswith(b'\n'):raise RuntimeError('partial IPC row')
    self._accept(line)
   if self._terminal is None:self._failure('missing isolated terminal snapshot')
  except BaseException as exc:self._failure(repr(exc))
 def wait_ready(self,timeout=3):
  until=time.monotonic()+timeout
  while len(self.samples)<2:
   if self.error or self._process.poll() is not None or time.monotonic()>=until:raise RuntimeError(self.error or 'isolated sampler did not become ready')
   time.sleep(.005)
  return True
 def stop(self):
  if self._process is None or self._stopped:return
  if self._process.poll() is None:
   try:self._process.stdin.write(b'STOP\n');self._process.stdin.flush();self._process.stdin.close()
   except (BrokenPipeError,OSError):pass
  try:self._process.wait(timeout=10)
  except subprocess.TimeoutExpired:
   self._failure('sampler child did not stop within10s');self._process.terminate()
   try:self._process.wait(timeout=2)
   except subprocess.TimeoutExpired:self._process.kill();self._process.wait(timeout=2)
  self._reader.join(timeout=2)
  if self._thread.is_alive():self._failure('sampler child or IPC reader remained alive')
  if self._process.returncode!=0:self._failure('sampler child exit '+str(self._process.returncode))
  if self._terminal is None:self._failure('no complete terminal snapshot')
  self._stopped=True
  write(self._directory/'receipt.json',dict(pid=self._process.pid,finished_s=time.time(),child_exited=self._process.poll() is not None,reader_stopped=not self._reader.is_alive(),returncode=self._process.returncode,error=self.error,terminal=self._terminal,raw_sample_count=len(self.samples),read_only=True,complete=not self.error))
 def mean_power_w(self):
  from ecopadg.measure.power import trapezoid_mean_power
  return trapezoid_mean_power(self.samples)
 def total_energy_j(self):
  from ecopadg.measure.power import trapezoid_energy
  return trapezoid_energy(self.samples)
 def temperature_c(self):return float(self.backend.temperature_c(self.gpus[0]))
