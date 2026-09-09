"""Bounded legacy correctness gate: default checks CPU hashes only."""
import argparse,asyncio,csv,hashlib,importlib.util,json,os
from pathlib import Path
import signal,sys,time
from checks import Checks,require,write
ROOT=Path(__file__).resolve().parent
COMMON=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/common/execution-until-complete-v1/run.py')
COMMON_SHA='77bcbbb68e20419e5bc469a838c71e1abfa789dc167d901501715fda0ff4a8a9'

def common():
    require(hashlib.sha256(COMMON.read_bytes()).hexdigest()==COMMON_SHA,'common execution bytes changed')
    spec=importlib.util.spec_from_file_location('baseline_common_execution',COMMON);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def mechanism_gates(checks,measurement_valid,temporal_owner_proven):
    return dict(ordinary=bool(measurement_valid and checks.get('ordinary_cross_replica_exact')),
        pd=bool(measurement_valid and checks.get('pd_exact_all_declared_pairs') and checks.get('cancel_all_tp_ranks')),
        temporal=bool(measurement_valid and checks.get('temporal_exact') and temporal_owner_proven))

def validate_scope(binding):
    require(binding.get('model')=='32b','this gate is the B32B four-TP2 scope')
    require(len(binding.get('instances',[]))==4,'four resident TP2 replicas required')
    require([i['gpus'] for i in binding['instances']]==[[0,1],[2,3],[4,5],[6,7]],'fixed original B GPU pairs required')
    for i in binding['instances']:
        require(i['tp']==2 and i['native_kind']=='legacy_sync_put' and not i.get('scheduler_cache_observed'),'legacy owner ACK must not invent v3 cache telemetry')
        require(i['container']['image']=='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b','original baseline image changed')
        require('service_budget_tokens' not in i and 'restore_budget_tokens' not in i,'legacy does not accept v3 budget commands')

async def execute(args,binding):
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.measure.power import PowerSampler,trapezoid_energy
    from ecopadg.metrics import clip_power_window
    from ecopadg.serving.measurement import save_raw,power_evidence
    from ecopadg.serving.backend import ClockOwner
    m=common();m.validate_binding(binding);validate_scope(binding)
    require(not args.out.exists(),'original correctness output must remain untouched');args.out.mkdir(parents=True)
    status=dict(started_s=time.time(),complete=False,passed=False,measurement_valid=False,work_timeout_s=390,cleanup_timeout_s=90,baseline_performance_executed=False)
    def save():write(args.out/'status.json',status)
    save();sampler=None;checker=None;clocks=None;verified=False;started=None;failure=None;offsets={}
    loop=asyncio.get_running_loop();task=asyncio.current_task();interrupted=False
    def stop():
        nonlocal interrupted
        if not interrupted:interrupted=True;task.cancel()
    for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,stop)
    async with aiohttp.ClientSession(trust_env=False) as session:
        try:
            identity=await m.identity(session,binding);write(args.out/'identity.before.json',identity)
            # Static budget is an engine-start argument on this legacy image.
            for instance, actual_identity in zip(binding['instances'],identity):
                path=Path(instance['engine_config']);require(str(path) in binding['files'],'engine config not frozen')
                cfg=m.read(path);require(cfg['max_num_batched_tokens']==8192 and cfg['max_num_seqs']==32 and cfg['max_model_len']==8192 and cfg['tp']==2,'static startup budget/TP/model work limit differs')
                actual=actual_identity['provenance'];expected_model='/models/Qwen2.5-32B-Instruct'
                require(cfg['model']==instance['provenance'].get('model')==actual.get('model')==expected_model and actual.get('max_model_len')==8192,'actual model/work limit differs')
                sources=instance['provenance'].get('source_files_at_import');require(isinstance(sources,dict) and sources and actual.get('source_files_at_import')==sources,'actual source mapping not bound')
                require(all(binding['files'].get(f)==digest for f,digest in sources.items()),'imported engine source outside frozen inputs')
                event=args.runtime_dir/(instance['id']+'.control.events.jsonl');require(event.is_file(),'real owner event stream missing');offsets[str(event)]=event.stat().st_size
            verified=True
            hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant');sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True);sampler.start()
            until=time.monotonic()+5
            while len(sampler.samples)<2:
                require(not sampler.error and time.monotonic()<until,'all8 power readiness failed');await asyncio.sleep(.02)
            require(power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)['power_source_verified'],'instant source not proven')
            started=time.time();clocks=await asyncio.to_thread(ClockOwner,hardware,tuple(range(8)))
            await asyncio.wait_for(clocks.set(range(8),2520,verify_rise=False),15)
            checker=Checks(session,binding,args.out/'checks');status['phase']='ordinary_pd_temporal';save()
            try:await asyncio.wait_for(checker.run(),390)
            except BaseException as exc:failure=exc;status['error']=repr(exc)
        except BaseException as exc:failure=exc;status['error']=repr(exc)
        finally:
            cleanup_end=time.monotonic()+90;errors=status['cleanup_errors']=[]
            def remaining(limit):return max(.001,min(limit,cleanup_end-time.monotonic()))
            if verified and checker is not None:
                try:status['native_cleanup_complete']=await asyncio.wait_for(checker.cleanup(),remaining(75))
                except BaseException as exc:errors.append('native cleanup '+repr(exc));status['native_cleanup_complete']=False
            elif verified:
                try:
                    # Checks never acquired a request; still restore the verified deployment.
                    result=await asyncio.wait_for(asyncio.gather(*(m.restore(session,i) for i in binding['instances']),return_exceptions=True),remaining(65))
                    status['native_cleanup_complete']=all(isinstance(v,dict) and v.get('complete') for v in result)
                    write(args.out/'setup-failure-native-cleanup.json',[dict(error=repr(v)) if isinstance(v,BaseException) else v for v in result])
                except BaseException as exc:errors.append('setup cleanup '+repr(exc));status['native_cleanup_complete']=False
            if clocks is not None:
                try:await asyncio.wait_for(clocks.close(),remaining(10));status['clock_restore_complete']=True
                except BaseException as exc:errors.append('clock release '+repr(exc));status['clock_restore_complete']=False
            else:status['clock_restore_complete']=True
            ended=time.time();status.update(measurement_start_s=started,measurement_end_s=ended)
            for path,offset in offsets.items():
                try:
                    with Path(path).open('rb') as f:f.seek(offset);data=f.read()
                    destination=args.out/(Path(path).name);destination.write_bytes(data)
                    require(not data or data.endswith(b'\n'),'owner event incomplete tail')
                    events=[json.loads(x) for x in data.splitlines()];temporal=[e for e in events if e.get('mode')=='temporal' and e.get('tokens',0)>0]
                    if temporal:
                        proof=dict(count=len(temporal),prefill_steps=sum(bool(e['prefill']) for e in temporal),decode_steps=sum(bool(e['decode']) for e in temporal),mixed_steps=sum(bool(e['prefill'] and e['decode']) for e in temporal))
                        status.setdefault('temporal_owner_proof',{})[path]=proof;require(proof['mixed_steps']==0,'real temporal prefill/decode overlap')
                    status.setdefault('events',{})[path]=dict(offset_start=offset,offset_end=offset+len(data),sha256=hashlib.sha256(data).hexdigest())
                except BaseException as exc:errors.append('owner evidence '+repr(exc))
            if sampler is not None:
                try:await asyncio.sleep(.12);await asyncio.to_thread(sampler.stop)
                except BaseException as exc:errors.append('sampler stop '+repr(exc))
                try:
                    power=args.out/'power';power.mkdir();save_raw(power,[],sampler.samples,sampler.utilization_samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata)
                    with (power/'clocks.csv').open('x',newline='') as f:
                        w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)]);w.writerows([t,*freq] for t,freq in sampler.frequency_samples)
                    status['power_evidence']=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata);status['sampling_error']=sampler.error
                    status['full_operation_energy_j']=trapezoid_energy(clip_power_window(sampler.samples,started,ended,pad_s=0)) if started else None
                except BaseException as exc:errors.append('power save/integration '+repr(exc))
            if verified:
                try:write(args.out/'identity.after.json',await asyncio.wait_for(m.identity(session,binding),remaining(12)));m.validate_binding(binding)
                except BaseException as exc:errors.append('after identity '+repr(exc))
            temporal=status.get('temporal_owner_proof',{});temporal_proven=bool(temporal) and all(x['prefill_steps'] and x['decode_steps'] and not x['mixed_steps'] for x in temporal.values())
            status['measurement_valid']=bool(started is not None and not errors and status.get('native_cleanup_complete') and status.get('clock_restore_complete') and not status.get('sampling_error') and status.get('power_evidence',{}).get('power_source_verified') and status.get('full_operation_energy_j') is not None)
            status['mechanism_gate']=mechanism_gates(checker.state.get('checks',{}) if checker else {},status['measurement_valid'],temporal_proven)
            status['system_eligibility']={'mixed':status['mechanism_gate']['ordinary'],'dynamollm-resident':status['mechanism_gate']['ordinary'],'distserve':status['mechanism_gate']['ordinary'] and status['mechanism_gate']['pd'],'ecoserve':status['mechanism_gate']['ordinary'] and status['mechanism_gate']['temporal']}
            status.update(complete=True,passed=bool(failure is None and all(status['mechanism_gate'].values())),finished_s=time.time())
            if checker:
                checker.state.update(complete=True,passed=status['passed'],finished_s=time.time());checker.save();checker.close()
            save()
    require(status['passed'],'legacy correctness failure retained; no baseline performance permission inferred')
    return status

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binding',type=Path,required=True);p.add_argument('--runtime-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    b=json.loads(a.binding.read_text());h=Path(b['host_release']);sys.path[:0]=[str(h/'src'),str(h),'/root/workspace/pdblend/.runtime-deps'];validate_scope(b);common().validate_binding(b)
    if a.run:
        from ecopadg.serving.campaign import node_lease
        with node_lease():asyncio.run(execute(a,b))
    else:print(json.dumps(dict(cpu_only=True,hardware_actions=False,binding_checked=True,scope='four TP2 legacy correctness, no baseline performance')))
if __name__=='__main__':main()
