"""Finite target-host probes under one original lease; no serving-policy edits."""
import argparse,asyncio,importlib.util,json,os,signal,socket,sys,time
from pathlib import Path
from types import SimpleNamespace
import validate as v
HERE=Path(__file__).resolve().parent
DEPLOY=HERE.parent/'distributed14b-deployment-v1'
sys.path.insert(0,str(DEPLOY));import deploy
STREAM=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/A/frequency2400-short16-code-001/stream.py')
s=importlib.util.spec_from_file_location('distributed14b_frozen_stream',STREAM);sm=importlib.util.module_from_spec(s);s.loader.exec_module(sm)

def mutable_write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

class Probe(sm.NaturalStream):
    def __init__(self,session,instance,out):
        self.session=session;self.args=SimpleNamespace(port=instance['port'],seed=0);self.issued=set();self.tasks=[]
        self.stream_journal=(out/'streams.jsonl').open('x',buffering=1);self.result_journal=(out/'requests.jsonl').open('x',buffering=1)
    def close(self):self.stream_journal.close();self.result_journal.close()

async def point(common,session,binding,p,out,clocks):
    out.mkdir();instance=next(i for i in binding['instances'] if i['id']==p['instance_id']);probe=Probe(session,instance,out)
    raw=dict(point=p,started_s=None,finished_s=None,requests=[],error=None,cleanup=dict(complete=False))
    cfg=v.read(instance['engine_config']); event_path=Path(cfg['runtime_dir'])/(instance['id']+'.control.events.jsonl');offset=None
    try:
        raw['budget_transition']=await common.resume(session,instance,2048)
        await clocks.set(instance['gpus'],p['frequency_mhz'],verify_rise=False)
        raw['warmup']=await probe.request(128,32)
        v.require(raw['warmup'].get('success') and len(raw['warmup']['output_token_ids'])==32 and raw['warmup'].get('usage',{}).get('completion_tokens')==32,'explicit warmup incomplete')
        raw['runtime_before']=await common.wait_idle(session,instance)
        offset=event_path.stat().st_size;raw['native_event_source']=dict(path=str(event_path),start_byte=offset,synchronous_event_before_snapshot=True)
        gate=asyncio.Event();release={}
        probe.tasks=[asyncio.create_task(probe.request(p['input_tokens'],p['output_tokens'],gate,release)) for _ in range(p['batch'])]
        await asyncio.sleep(0);raw['started_s']=release['epoch_s']=time.time();release['monotonic_s']=time.monotonic();gate.set()
        mutable_write(out/'raw.json',raw)
        raw['requests']=await asyncio.gather(*probe.tasks)
        raw['runtime_after_requests']=await common.wait_idle(session,instance)
        v.require(all(r.get('success') and len(r['output_token_ids'])==p['output_tokens'] for r in raw['requests']),'point work incomplete')
    except BaseException as exc:
        raw['error']=repr(exc)
        for rid in probe.issued:
            try:await common.http(session,instance,'/cancel',dict(request_id=rid))
            except BaseException as error:raw.setdefault('cancel_errors',[]).append(repr(error))
        if probe.tasks:
            rows=await asyncio.gather(*probe.tasks,return_exceptions=True);raw['requests']=[r if isinstance(r,dict) else dict(success=False,error=repr(r)) for r in rows]
    finally:
        try:raw['cleanup']=await asyncio.wait_for(common.restore(session,instance),120)
        except BaseException as exc:raw['cleanup']=dict(complete=False,error=repr(exc))
        raw['finished_s']=time.time()
        if offset is not None:
            with event_path.open('rb') as f:f.seek(offset);data=f.read()
            (out/'events.jsonl').write_bytes(data);raw['events']=v.ref(out/'events.jsonl')
        probe.close();mutable_write(out/'raw.json',raw)
    return raw

async def execute(spec,out,lease):
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.backend import ClockOwner
    binding=v.source_check(spec);deploy.require_lease(lease)
    v.require(socket.gethostname()==spec['hostname'] and not out.exists(),'fresh actual target output required')
    out.mkdir(parents=True);deploy.write(out/'spec.json',spec)
    common=deploy.adapter.load_runtime(binding['host_release'],Path(binding['executor']).parent)
    hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
    status=dict(schema='distributed14b-target-profile-result-v1',pid=os.getpid(),started_s=time.time(),passed=False,complete=False,errors=[],points=[],profile_reference=spec['profile_reference'],profile_publication_allowed=False,not_formal_performance=True,binding=spec['binding'],spec=v.ref(out/'spec.json'))
    meter=deploy.Measurement(out,hardware,status);clocks=None
    def update():mutable_write(out/'status.json',status)
    stop=False
    def request_stop():
        nonlocal stop;stop=True
    for sig in (signal.SIGTERM,signal.SIGINT):asyncio.get_running_loop().add_signal_handler(sig,request_stop)
    async with aiohttp.ClientSession(trust_env=False) as session:
        try:
            common.validate_binding(binding);deploy.write(out/'identity.before.json',await common.identity(session,binding))
            await meter.start();clocks=await asyncio.to_thread(ClockOwner,hardware,tuple(range(8)))
            for p in spec['points']:
                v.require(not stop and not (out.parent/'STOP_PROFILE').exists(),'STOP at completed point boundary')
                v.source_check(spec);status['current_point']=p;update()
                path=out/p['point_id'];raw=await point(common,session,binding,p,path,clocks)
                await asyncio.sleep(.08)
                evidence=v.validate_point(raw,[json.loads(x) for x in (path/'events.jsonl').read_text().splitlines() if x],meter.sampler.frequency_samples)
                evidence['reference_comparison']=v.prediction_error(evidence,p,v.fixed(spec['profile_reference']))
                deploy.write(path/'validation.json',evidence);status['points'].append(dict(point_id=p['point_id'],raw=v.ref(path/'raw.json'),validation=v.ref(path/'validation.json'),passed=True));update()
            status['complete']=len(status['points'])==len(spec['points'])
        except BaseException as exc:status['errors'].append(repr(exc))
        finally:
            restores=await asyncio.gather(*(common.restore(session,i) for i in binding['instances']),return_exceptions=True)
            status['native_cleanup']=[dict(complete=False,error=repr(x)) if isinstance(x,BaseException) else x for x in restores]
            if not all(x.get('complete') for x in status['native_cleanup']):status['errors'].append('native cleanup incomplete')
            if clocks is not None:
                try:await clocks.close();status['clock_restore_complete']=True
                except BaseException as exc:status['errors'].append('clock cleanup '+repr(exc))
            try:deploy.write(out/'identity.after.json',await common.identity(session,binding))
            except BaseException as exc:status['errors'].append('identity cleanup '+repr(exc))
            await meter.finish();status['finished_s']=time.time();status['passed']=bool(status['complete'] and status['measurement_valid'] and not status['errors']);update()
            files={str(p):v.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
            status['files']={p:h for p,h in files.items() if p!=str(out/'status.json')};update()
            files[str(out/'status.json')]=v.sha(out/'status.json')
            deploy.write(out/'evidence-manifest.json',dict(schema='distributed14b-target-profile-evidence-v1',files=files,passed=status['passed'],spec=v.ref(out/'spec.json'),source=v.ref(HERE/'manifest.json')))
    return status

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    spec=v.read(a.spec);binding=v.source_check(spec)
    deploy.adapter.load_runtime(binding['host_release'],Path(binding['executor']).parent)
    if not a.run:print(json.dumps(dict(cpu_only=True,points=len(spec['points']))));return
    v.require('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh original node owner required')
    from ecopadg.serving.campaign import node_lease
    with node_lease() as lease:result=asyncio.run(execute(spec,a.out,lease))
    print(json.dumps({k:result[k] for k in ('passed','complete','measurement_valid','errors')}));raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
