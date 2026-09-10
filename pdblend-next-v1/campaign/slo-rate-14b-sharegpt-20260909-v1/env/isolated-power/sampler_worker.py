"""Read-only worker: run the exact frozen sampler in an independent interpreter."""
import argparse,gc,hashlib,json,os,sys,threading,time
from pathlib import Path


def checked(ref):
 p=Path(ref['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'];return json.loads(p.read_text())


def execute(spec):
 host=Path(spec['host_manifest']['path']).parent;manifest=checked(spec['host_manifest'])
 adapter=checked(spec['adapter_manifest'])
 for p,h in adapter['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 for name in ('src/ecopadg/measure/power.py','src/ecopadg/measure/backends.py'):
  assert hashlib.sha256((host/name).read_bytes()).hexdigest()==manifest['files'][name]
 sys.path[:0]=[str(host/'src'),'/root/workspace/pdblend/.runtime-deps']
 from ecopadg.measure.backends import PynvmlBackend
 from ecopadg.measure.power import PowerSampler
 assert Path(sys.modules['ecopadg.measure.power'].__file__).resolve()==(host/'src/ecopadg/measure/power.py').resolve()
 backend=PynvmlBackend(power_mode='instant')
 sampler=PowerSampler(spec['gpus'],interval=spec['interval'],backend=backend,sample_clocks=spec['sample_clocks'])
 # The dedicated process owns no cyclic serving tasks; its entire heap is
 # freed on exit. Avoid collector pauses in the measurement-only process.
 gc.collect();gc.disable()
 stopping=threading.Event();hashsum=hashlib.sha256();emitted=0
 def publish(value,hashed=True):
  line=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode()
  if hashed:hashsum.update(line)
  sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
 def stop_reader():
  sys.stdin.buffer.readline();stopping.set()
 threading.Thread(target=stop_reader,daemon=True).start()
 sampler.start();thread=sampler._thread
 publish(dict(kind='ready',pid=os.getpid(),source=sampler.power_source,host_manifest=spec['host_manifest'],
              original_sampler_source=manifest['files']['src/ecopadg/measure/power.py'],
              read_only=True,garbage_collection_disabled_only_in_sampler=True))
 try:
  while True:
   # Publication may block on its pipe. Sampling continues in the original
   # sampler thread and retains every row in its original unbounded lists.
   if stopping.is_set() or sampler.error:break
   available=min(len(sampler.samples),len(sampler.utilization_samples),len(sampler.frequency_samples) if spec['sample_clocks'] else len(sampler.samples))
   while emitted<available:
    publish(dict(kind='row',index=emitted,power=sampler.samples[emitted],metadata=sampler.power_metadata[emitted],
      utilization=sampler.utilization_samples[emitted] if emitted<len(sampler.utilization_samples) else None,
      frequency=sampler.frequency_samples[emitted] if emitted<len(sampler.frequency_samples) else None))
    emitted+=1
   if stopping.is_set() or sampler.error:break
   stopping.wait(.005)
 finally:
  sampler.stop()
  while emitted<len(sampler.samples):
   publish(dict(kind='row',index=emitted,power=sampler.samples[emitted],metadata=sampler.power_metadata[emitted],
    utilization=sampler.utilization_samples[emitted] if emitted<len(sampler.utilization_samples) else None,
    frequency=sampler.frequency_samples[emitted] if emitted<len(sampler.frequency_samples) else None));emitted+=1
  publish(dict(kind='terminal',pid=os.getpid(),sampler_thread_stopped=not thread.is_alive(),error=sampler.error,
    counts=dict(samples=len(sampler.samples),metadata=len(sampler.power_metadata),utilization=len(sampler.utilization_samples),frequency=len(sampler.frequency_samples)),
    emitted=emitted,stream_sha256=hashsum.hexdigest()),hashed=False)
  if thread.is_alive() or sampler.error:raise RuntimeError('isolated sampler failed or remained alive')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();execute(json.loads(Path(a.spec).read_text()))
