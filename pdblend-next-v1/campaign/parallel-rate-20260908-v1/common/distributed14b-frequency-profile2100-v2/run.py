"""Finite target-host probes under one original lease; no serving-policy edits."""
import argparse,asyncio,importlib.util,json,os,signal,socket,sys,time
from pathlib import Path
from types import SimpleNamespace
import validate as v
HERE=Path(__file__).resolve().parent
DEPLOY=HERE.parent/'distributed14b-deployment-v1'
sys.path.insert(0,str(DEPLOY));import deploy
STREAM=HERE/'stream.py'
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
        raw['idle_residency']=dict(start_s=time.time(),native=await common.wait_idle(session,instance))
        await asyncio.sleep(2)
        raw['idle_residency']['end_s']=time.time()
        raw['warmup']=await probe.request(128,32)
        v.require(raw['warmup'].get('success') and len(raw['warmup']['output_token_ids'])==32 and raw['warmup'].get('usage',{}).get('completion_tokens')==32,'explicit warmup incomplete')
        raw['runtime_before']=await common.wait_idle(session,instance)
        offset=event_path.stat().st_size;raw['native_event_source']=dict(path=str(event_path),start_byte=offset,synchronous_event_before_snapshot=True)
        gate=asyncio.Event();release={}
        probe.tasks=[asyncio.create_task(probe.request(p['input_tokens'],p['output_tokens'],gate,release,arrival_offset_s=offset)) for offset in p['arrival_offsets_s']]
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
    status=dict(schema='distributed14b-actual2100-profile-result-v1',pid=os.getpid(),started_s=time.time(),passed=False,complete=False,errors=[],points=[],profile_reference=spec['profile_reference'],profile_publication_allowed=False,not_formal_performance=True,binding=spec['binding'],spec=v.ref(out/'spec.json'))
    hooks=v.load(Path(spec['measurement_hooks']['path']),'actual2100_isolated_hooks')
    hooks.install(out/'isolated-samplers',spec['host_manifest'],spec['measurement_adapter'])
    meter=deploy.Measurement(out,hardware,status);clocks=None
    order=v.load(HERE/'source_order.py','actual2100_source_order')
    class SourceGate:
        async def command(self,*argv,timeout=12):return await deploy.command(list(argv),status.setdefault('source_commands',[]),timeout=timeout)
    def update():mutable_write(out/'status.json',status)
    stop=False
    def request_stop():
        nonlocal stop;stop=True
    for sig in (signal.SIGTERM,signal.SIGINT):asyncio.get_running_loop().add_signal_handler(sig,request_stop)
    async with aiohttp.ClientSession(trust_env=False) as session:
        try:
            common.validate_binding(binding);before=await common.identity(session,binding);deploy.write(out/'identity.before.json',before)
            source_before=await order.capture(SourceGate(),dict(instances=before),HERE/'source-order-contract.json')
            deploy.write(out/'source-order.before.json',source_before)
            await meter.start();clocks=await asyncio.to_thread(ClockOwner,hardware,tuple(range(8)))
            for p in spec['points']:
                v.require(not stop and not (out.parent/'STOP_PROFILE').exists(),'STOP at completed point boundary')
                v.source_check(spec);status['current_point']=p;update()
                path=out/p['point_id'];raw=await point(common,session,binding,p,path,clocks)
                await asyncio.sleep(.08)
                evidence=v.validate_profile_point(raw,[json.loads(x) for x in (path/'events.jsonl').read_text().splitlines() if x],meter.sampler.samples,meter.sampler.frequency_samples,source_order_verified=True)
                deploy.write(path/'validation.json',evidence);status['points'].append(dict(point_id=p['point_id'],raw=v.ref(path/'raw.json'),validation=v.ref(path/'validation.json'),passed=True));update()
            switcher=v.load(HERE/'frequency_switch.py','actual2100_frequency_switch')
            status['switches']=[]
            for gpu in spec['transition_gpus']:
                v.require(not stop,'STOP before new transition batch')
                instance=next(i for i in binding['instances'] if i['gpus']==[gpu])
                path=out/f'transitions-gpu{gpu}';path.mkdir();probe=Probe(session,instance,path)
                try:raw=await switcher.observe(common,session,instance,probe,clocks,hardware,path)
                finally:probe.close()
                await asyncio.sleep(.08)
                result=switcher.derive(raw,meter.sampler.samples,meter.sampler.frequency_samples)
                deploy.write(path/'validation.json',result);status['switches'].append(dict(gpu=gpu,raw=v.ref(path/'raw.json'),validation=v.ref(path/'validation.json')));update()
            status['complete']=len(status['points'])==28 and len(status['switches'])==2
        except BaseException as exc:status['errors'].append(repr(exc))
        finally:
            restores=await asyncio.gather(*(common.restore(session,i) for i in binding['instances']),return_exceptions=True)
            status['native_cleanup']=[dict(complete=False,error=repr(x)) if isinstance(x,BaseException) else x for x in restores]
            if not all(x.get('complete') for x in status['native_cleanup']):status['errors'].append('native cleanup incomplete')
            if clocks is not None:
                try:await clocks.close();status['clock_restore_complete']=True
                except BaseException as exc:status['errors'].append('clock cleanup '+repr(exc))
            try:
                after=await common.identity(session,binding);deploy.write(out/'identity.after.json',after)
                deploy.write(out/'source-order.after.json',await order.capture(SourceGate(),dict(instances=after),HERE/'source-order-contract.json'))
            except BaseException as exc:status['errors'].append('identity cleanup '+repr(exc))
            await meter.finish()
            try:
                roots=hooks.directories(out/'isolated-samplers');status['isolated_samplers']=hooks.sampler_references(roots)
                hooks.completed_artifacts(roots,spec['host_manifest'],spec['measurement_adapter'])
                status['measurement_adapter']=spec['measurement_adapter'];status['measurement_hooks']=spec['measurement_hooks']
            except BaseException as exc:status['errors'].append('isolated measurement '+repr(exc))
            status['finished_s']=time.time();status['passed']=bool(status['complete'] and status['measurement_valid'] and not status['errors']);update()
            files={str(p):v.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
            status['files']={p:h for p,h in files.items() if p!=str(out/'status.json')};update()
            files[str(out/'status.json')]=v.sha(out/'status.json')
            deploy.write(out/'evidence-manifest.json',dict(schema='distributed14b-actual2100-profile-evidence-v1',files=files,passed=status['passed'],spec=v.ref(out/'spec.json'),source=v.ref(HERE/'manifest.json')))
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
