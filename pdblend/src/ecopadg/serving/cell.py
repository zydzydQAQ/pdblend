"""Run an explicitly configured runtime cell on already prepared instances.

The historical shell entry delegates here when PDBLEND_RUNTIME_CONFIG is set.
Formal cells require mechanism gates and a typed, unchanged artifact freeze.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

from aiohttp import web
from benchmarks.scripts.bench_vllm import run_trace,bench_rows
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler
from .controller import Controller,STRATEGIES
from .evidence import baseline_gaps,formal_freeze_gaps,sha256
from .measurement import save_raw,summarize_cell
from .campaign import node_lease
from .provenance import verify_live_engines
from .frequency import verify_frozen_costs


def verify_trace_identity(args,trace):
    if any(trace.get(key)!=getattr(args,key) for key in ('dataset','load','seed','split')):
        raise ValueError('trace identity differs from the declared evaluation cell')
    if (not trace.get('requests') or (args.dataset!='dynamic' and len(trace['requests'])<500)
            or (args.dataset=='dynamic' and trace.get('duration_s')!=3600)):
        raise ValueError('formal trace does not meet the frozen workload size or duration')


def verify_formal_config(args,config,freeze):
    frozen={freeze['files'][p] for p in freeze['groups']['protocol']}
    if sha256(args.config) not in frozen:
        raise ValueError('runtime configuration is not in the frozen protocol')
    if config.get('power_mode') != 'instant':
        raise ValueError('formal frozen configuration must explicitly require instant power')
    if args.strategy and args.strategy!=config['strategy']:
        raise ValueError('formal strategy override differs from frozen configuration')
    for field in ('slo_ttft_s','slo_tpot_s'):
        value=getattr(args,field,None)
        if value is not None and value!=config[field]:
            raise ValueError('formal SLO override differs from frozen configuration')
    if config.get('allow_unprofiled_fallback',False) or not config.get('manage_clocks',True):
        raise ValueError('formal run requires profiled decisions and the hardware clock owner')
    path=Path(config.get('profiles','')).resolve()
    if str(path) not in freeze['groups']['profiles']:
        raise ValueError('formal profile table is not frozen')
    profiles=json.loads(path.read_text())
    if not all(profiles.get(key) is True for key in ('frequency_commands_verified',
            'heldout_calibration_complete','mixed_interference_measured')):
        raise ValueError('formal profiles require clock evidence, held-out calibration and mixed interference')
    if not all(profiles.get(key) is True for key in ('instant_prefill_calibration_complete',
            'instant_heldout_calibration_complete')):
        raise ValueError('formal profiles require verified instant prefill and held-out calibration')
    if config.get('park_idle',True) and not profiles.get('resident_idle_measured'):
        raise ValueError('formal parking requires measured resident idle power')
    sources=profiles.get('certification_artifacts',{})
    if not sources or any(p not in freeze['groups']['profiles'] or freeze['files'].get(p)!=digest
                          for p,digest in sources.items()):
        raise ValueError('profile certification evidence is not frozen')
    if config.get('transfers'):
        path=Path(config.get('transfer_evidence','')).resolve()
        if str(path) not in freeze['groups']['profiles']:
            raise ValueError('formal transfer evidence is not frozen')
        proof=json.loads(path.read_text())
        sources=proof.get('certification_artifacts',{})
        if (not proof.get('certified') or proof.get('instant_power_costs_verified') is not True
                or proof.get('receiver_transfer_energy_included') is not True
                or proof.get('links')!=config['transfers'] or not sources
                or proof.get('engine_image')!=freeze.get('identities',{}).get('engine_image')
                or any(p not in freeze['groups']['profiles'] or freeze['files'].get(p)!=digest
                       for p,digest in sources.items())):
            raise ValueError('transfer costs differ from their frozen hardware evidence')
    if (config['strategy']=='mixed_dvfs' or config['strategy'].startswith(('pdblend','dynamollm'))) and config.get('dvfs',True):
        verify_frozen_costs(config,profiles,freeze)


async def run_cell(args):
    config=json.loads(args.config.read_text())
    if config.get('power_mode', 'instant') != 'instant':
        raise ValueError('new serving cells require explicit NVML instant power')
    freeze=json.loads(args.freeze.read_text()) if args.freeze else {}
    if args.split=='formal':
        verify_formal_config(args,config,freeze)
    config['strategy']=args.strategy or config['strategy']
    config['power_mode']='instant'
    for field in ('slo_ttft_s','slo_tpot_s'):
        value=getattr(args,field,None)
        if value is not None:
            if value<=0: raise ValueError('SLO must be positive')
            config[field]=value
    args.out.mkdir(parents=True,exist_ok=False)
    config['journal']=str(args.out/'control.jsonl')
    trace=json.loads(args.trace.read_text())
    mechanisms=json.loads(args.mechanisms.read_text()) if args.mechanisms else {}
    if args.split=='formal':
        verify_trace_identity(args,trace)
        gaps=baseline_gaps(mechanisms)
        changed=formal_freeze_gaps(freeze)
        if gaps or changed:
            raise ValueError('formal gate closed: '+json.dumps(dict(mechanisms=gaps,freeze=changed)))
        if sha256(args.trace) not in {freeze['files'][p] for p in freeze['groups']['traces']}:
            raise ValueError('trace not in frozen evaluation set')
        if config.get('topology'):
            # Every in-run replacement must use the same immutable image.
            config['topology']['image']=freeze['identities']['engine_image']
    (args.out/'runtime_config.json').write_text(json.dumps(config,indent=2))
    controller=Controller(config)
    runner=web.AppRunner(controller.application())
    sampler=None
    started=time.time()
    try:
        await runner.setup()
        await web.TCPSite(runner,'127.0.0.1',config.get('port',18080)).start()
        if args.split=='formal':
            provenance=await verify_live_engines(controller.backend,freeze)
            await asyncio.to_thread((args.out/'live_engines.before.json').write_text,json.dumps(provenance,indent=2))
        ready=time.time()
        # L20's legacy power-usage API averages the previous second. Every new
        # serving comparison uses the explicit instantaneous field instead.
        hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
        sampler=PowerSampler(range(8),interval=.02,backend=hardware)
        sampler.start()
        # This is only the pre-boundary power sample, not model warm-up.
        await asyncio.sleep(.1)
        if sampler.error or len(sampler.samples) < 2:
            raise RuntimeError('instant power preflight failed: '+str(sampler.error or 'insufficient samples'))
        outputs,_=await run_trace(trace,f"http://127.0.0.1:{config.get('port',18080)}",
            config.get('model_name','Qwen2.5-14B-Instruct'),timeout_s=args.timeout)
        reconfiguration_end=await controller.quiesce_controls()
        rows=bench_rows(trace,outputs,config['slo_ttft_s'],config['slo_tpot_s'])
        await asyncio.sleep(.1)
        await asyncio.to_thread(sampler.stop)
        try: await controller.journal.flush()
        except Exception as exc: controller.failure='evidence journal: '+repr(exc)
        await asyncio.to_thread(save_raw,args.out,rows,sampler.samples,sampler.utilization_samples,
            power_source=sampler.power_source,power_metadata=sampler.power_metadata)
        summary=await asyncio.to_thread(summarize_cell,trace,rows,sampler.samples,sampler.utilization_samples,
            (config['slo_ttft_s'],config['slo_tpot_s']),sampling_error=sampler.error,
            reconfiguration_end_s=reconfiguration_end,power_source=sampler.power_source,
            power_metadata=sampler.power_metadata,require_power_mode='instant')
        provenance_error=None
        if args.split=='formal':
            try:
                provenance=await verify_live_engines(controller.backend,freeze)
                await asyncio.to_thread((args.out/'live_engines.after.json').write_text,json.dumps(provenance,indent=2))
            except Exception as exc:
                provenance_error=repr(exc)
        unchanged=not formal_freeze_gaps(freeze)
        summary.update(system='pdblend' if args.split=='formal' and config['strategy'].startswith('pdblend-')
                       else config['strategy'],variant=config['strategy'],
            dataset=args.dataset,load=args.load,seed=args.seed,
            split=args.split,trace_sha256=sha256(args.trace),
            formal_eligible=args.split=='formal' and unchanged and provenance_error is None
                and summary['power_source_verified'],
            startup_seconds=ready-started,
            first_arrival_wait_seconds=summary['measurement_start_s']-ready,
            warmup_seconds=config.get('reported_warmup_seconds'),
            trace_duration_s=trace.get('duration_s'),**freeze.get('identities',{}))
        summary['runtime_error']=controller.failure
        summary['admission_planning']=controller.planning_stats.summary()
        if controller.failure:
            summary['validity']='invalid_runtime'
            summary['capacity_observation_valid']=False
            summary['formal_eligible']=False
        if args.split=='formal' and (not unchanged or provenance_error):
            summary['validity']='invalid_provenance'
            summary['provenance_error']=provenance_error or 'frozen artifact changed during the run'
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps(summary,allow_nan=False),flush=True)
        return summary
    finally:
        if sampler:
            await asyncio.to_thread(sampler.stop)
        await runner.cleanup()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--strategy',choices=STRATEGIES)
    p.add_argument('--split',choices=('development','calibration','formal'),default='development')
    p.add_argument('--dataset',default='diagnostic')
    p.add_argument('--load',default='diagnostic')
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--freeze',type=Path)
    p.add_argument('--mechanisms',type=Path)
    p.add_argument('--timeout',type=float,default=900)
    p.add_argument('--slo-ttft-s',type=float)
    p.add_argument('--slo-tpot-s',type=float)
    args=p.parse_args()
    with node_lease():
        asyncio.run(run_cell(args))


if __name__=='__main__':
    main()
