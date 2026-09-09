"""Use existing real role/transport/budget validators, bound to the B deployment."""
import asyncio
import csv
import importlib.util
import json
import time
from types import SimpleNamespace
import aiohttp

from run import ROOT, RELEASE, IDS, NAMES, PORTS, require, read, write, http, idle, drain_proof, verify_live, command


def load_budget():
    spec=importlib.util.spec_from_file_location('b32_frozen_gpu_budget_validation',ROOT/'budget_validation.frozen.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


async def restore(session,port):
    before=await http(session,port,'/runtime')
    barrier=await http(session,port,'/drain',dict(expected_generation=before['generation']))
    drain_proof(before,barrier,tp=2)
    target=barrier['generation']+1
    await http(session,port,'/control',dict(generation=target,role='mixed',mode='continuous',
        admit_prefill=True,admit_decode=True,scheduler_budget=dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32)))
    raw=await http(session,port,'/runtime');idle(raw,accepting=True)
    load_budget().verify_ack(raw,target,8192,32)
    return dict(drain=barrier,restored=raw)


def cross_replica_outputs(result):
    rows=list(result['instances'].values())
    require(len(rows)==2 and all(r.get('passed') for r in rows),'both real TP2 replicas must pass')
    for tokens in (8192,1024,2048):
        key=f'budget_{tokens}'
        require(rows[0]['checks'][key]['outputs']==rows[1]['checks'][key]['outputs'],
                'same-model cross-instance output token mismatch at '+str(tokens))
    return {'budgets':[8192,1024,2048],'prompt_lengths':[128,7168],
            'output_tokens_each':64,'all_equal':True}


async def validate_all():
    from ecopadg.serving import runtime_validation
    from ecopadg.serving.measurement import save_raw,power_evidence
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.measure.power import PowerSampler,trapezoid_energy
    from ecopadg.metrics import clip_power_window
    require(not (ROOT/'validation-status.json').exists(),'validation already attempted; keep original evidence')
    image=read(ROOT/'image-receipt.json')['image']
    raw=dict(complete=False,passed=False,phase='sampling_preflight',started_s=time.time(),
        purpose='PDB TP2 correctness only; no performance or capacity certification',
        whole_node_gpu_count=8,automatic_baseline_execution=False)
    write('validation-status.json',raw)
    sampler=None;start=None
    async with aiohttp.ClientSession(trust_env=False) as session:
        try:
            hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
            sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True);sampler.start()
            deadline=time.monotonic()+5
            while len(sampler.samples)<2:
                require(not sampler.error and time.monotonic()<deadline,'eight-GPU power readiness failed')
                await asyncio.sleep(.02)
            require(power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)['power_source_verified'],'instant power source not verified')
            start=time.time();raw['measurement_start_s']=start
            for i in range(2):await verify_live(session,i,image)
            # Existing profiler's pdb-v2-<id> convention is honored by the new B names.
            # No monkeypatch and no alteration to the frozen source.
            raw['phase']='tp2_runtime_and_pd_validation';write('validation-status.json',raw)
            await runtime_validation.validate(SimpleNamespace(topology=ROOT/'topology.json',
                runtime_dir=ROOT/'runtime',out=ROOT/'runtime-validation'))
            rv=read(ROOT/'runtime-validation/raw.json')
            require(rv.get('passed') is True,'existing TP2 runtime validator failed')
            checks=rv.get('checks',{})
            for key in ('pd_output_tokens_equal','cancel_drained_all_ranks','idempotent_acknowledgement',
                        'client_disconnect_drained','next_request_unchanged','same_resident_processes'):
                require(checks.get(key) is True,'missing real transport/runtime evidence: '+key)
            require(checks.get('out_of_order_version',{}).get('status') in (400,409),'out-of-order request was not rejected')
            raw['runtime_checks']=checks
            # Budget validation is a separate controller experiment after role/PD validation.
            for p in PORTS:await restore(session,p)
            raw['phase']='scheduler_budget_validation';write('validation-status.json',raw)
            module=load_budget()
            validator=module.Validation(SimpleNamespace(ports=PORTS,runtime_dir=ROOT/'runtime',out=ROOT/'budget-validation'))
            require(await validator.run(),'real scheduler budget validation failed')
            result=read(ROOT/'budget-validation/result.json')
            raw['cross_replica_outputs']=cross_replica_outputs(result)
            for index in range(2):await verify_live(session,index,image)
            raw.update(phase='checks_passed',passed=True)
        except BaseException as exc:
            raw.update(phase='failed',error=repr(exc),passed=False)
            raise
        finally:
            raw['final_cleanup']={}
            if start is not None:
                for index,port in enumerate(PORTS):
                    try:raw['final_cleanup'][str(port)]=await asyncio.wait_for(restore(session,port),90)
                    except BaseException as exc:
                        raw.update(passed=False,incomplete_drain=True)
                        raw['final_cleanup'][str(port)]={'error':repr(exc)}
                        # This isolated deployment alone is owned; retain its logs and stopped container.
                        try:await command('docker','stop','--time','30',NAMES[index],timeout=120)
                        except BaseException as cleanup:raw['final_cleanup'][str(port)]['stop_error']=repr(cleanup)
                end=time.time();raw['measurement_end_s']=end
            if sampler is not None:
                await asyncio.sleep(.15)
                await asyncio.to_thread(sampler.stop)
                dest=ROOT/'validation-power';dest.mkdir(exist_ok=True)
                save_raw(dest,[],sampler.samples,sampler.utilization_samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata)
                with (dest/'clocks.csv').open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)])
                    w.writerows([t]+list(v) for t,v in sampler.frequency_samples)
                evidence=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)
                raw.update(power_evidence=evidence,sampling_error=sampler.error)
                if start is not None:
                    try:raw['total_node_energy_j']=trapezoid_energy(clip_power_window(sampler.samples,start,end,pad_s=0))
                    except BaseException as exc:raw.update(passed=False,measurement_window_error=repr(exc))
                raw['measurement_valid']=bool(start is not None and evidence['power_source_verified'] and not sampler.error
                    and not raw.get('incomplete_drain') and not raw.get('measurement_window_error'))
            raw.update(complete=True,finished_s=time.time())
            write('validation-status.json',raw)
    require(raw['passed'] and raw.get('measurement_valid'),'B TP2 validation/cleanup failed; retain evidence')
