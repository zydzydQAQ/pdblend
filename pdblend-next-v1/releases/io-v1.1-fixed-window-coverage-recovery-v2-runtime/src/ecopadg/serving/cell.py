"""Run an explicitly configured runtime cell on already prepared instances.

The historical shell entry delegates here when PDBLEND_RUNTIME_CONFIG is set.
Formal cells require mechanism gates and a typed, unchanged artifact freeze.
"""
import argparse
import asyncio
import json
import math
from pathlib import Path
import time

from aiohttp import web
from benchmarks.scripts.bench_vllm import run_trace,bench_rows,EVALUATION_V3,REQUEST_TIMEOUT_S
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler
from .controller import Controller,STRATEGIES
from .evidence import baseline_gaps,formal_freeze_gaps,sha256
from .measurement import save_raw,summarize_cell
from .campaign import node_lease
from .provenance import verify_live_engines
from .frequency import verify_frozen_costs


FIXED_WINDOW_PROTOCOL='per-dataset-slo-fixed-window-v2'
DATASET_SLOS={'alpaca':(1.,.1),'sharegpt':(5.,.15),'longbench':(15.,.2)}


def configure_fixed_window(args, config, trace):
    """Validate the opt-in development protocol before constructing a controller."""
    protocol=config.get('measurement_window_protocol')
    if protocol is None:
        if trace.get('protocol_id') == FIXED_WINDOW_PROTOCOL or getattr(args,'slo_scale',None) is not None:
            raise ValueError('fixed-window trace or SLO scale requires an explicit fixed-window config')
        return None
    if (protocol != FIXED_WINDOW_PROTOCOL or config.get('evaluation_protocol') != EVALUATION_V3
            or args.split != 'development' or not config['strategy'].startswith('pdblend')
            or args.dataset not in DATASET_SLOS):
        raise ValueError('fixed-window protocol requires PDB-only evaluation-v3 development')
    if (config.get('arrival_window_s') != 300 or trace.get('arrival_window_s') != 300
            or trace.get('protocol_id') != protocol or trace.get('measurement_schema') != 3
            or trace.get('duration_s') != 300 or trace.get('split') != 'development'
            or trace.get('dataset') != args.dataset or trace.get('seed') != args.seed
            or args.seed not in (701,1701)):
        raise ValueError('fixed-window trace/config identity or 300-second window differs')
    requests=trace.get('requests',[])
    if (not requests or len(requests) != len(trace.get('prompts',[]))
            or trace.get('n_requests') != len(requests)):
        raise ValueError('fixed-window trace must contain every declared request and prompt')
    offsets=[r.get('arrival_s') for r in requests]
    if (any(type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x < 300 for x in offsets)
            or offsets[0] != 0 or offsets != sorted(offsets)):
        raise ValueError('fixed-window arrivals must start at zero and lie in [0,300)')
    scale=getattr(args,'slo_scale',None)
    if scale is None: scale=config.get('slo_scale',1.)
    if type(scale) not in (int,float) or scale not in (.5,1.,2.):
        raise ValueError('SLO scale must be one of 0.5, 1, 2')
    base=DATASET_SLOS[args.dataset]
    for field,number in zip(('slo_ttft_s','slo_tpot_s'),base):
        override=getattr(args,field,None)
        if override is not None and override != number*scale:
            raise ValueError('explicit SLO differs from dataset SLO times the declared scale')
        config[field]=number*scale
    config.update(slo_scale=scale,slo_protocol='per-dataset-slo-v1',slo_attainment_target=.9)
    return dict(protocol_id=protocol,arrival_window_s=300.,slo_scale=scale,
        base_slo_s=dict(ttft=base[0],tpot=base[1]),
        effective_slo_s=dict(ttft=config['slo_ttft_s'],tpot=config['slo_tpot_s']),
        request_hard_timeout_s=REQUEST_TIMEOUT_S,drain_after_arrival_window_s=REQUEST_TIMEOUT_S,
        n_expected=len(requests),planned_arrival_span_s=offsets[-1],
        no_minimum_request_requirement=True,sparse_screen=len(requests)<30,
        request_count_warning='fewer than 30 requests; descriptive screen only' if len(requests)<30 else None,
        formal_eligible=False)


async def retain_fixed_window(trace, outputs, bench_elapsed_s, window):
    """Keep the live controller and real power sampler running through idle suffixes.

    The benchmark owns both the actual wall epoch and monotonic duration. No
    fabricated request, latency, power sample or padded dispatch span is used.
    """
    if len(outputs) != len(trace['requests']):
        raise ValueError('cannot establish a fixed window from missing benchmark records')
    epoch=outputs[0]['planned_arrival_s']
    if (not math.isfinite(epoch) or not math.isfinite(bench_elapsed_s) or bench_elapsed_s < 0
            or any(not math.isclose(row['planned_arrival_s'],epoch+req['arrival_s'],abs_tol=1e-5,rel_tol=0)
                   for row,req in zip(outputs,trace['requests']))):
        raise ValueError('benchmark epoch or per-request planned arrival differs from the trace')
    end=epoch+window['arrival_window_s']
    before=time.time()
    # The monotonic duration also prevents an early stop on a forward wall-clock
    # adjustment. Wall-clock inconsistency is retained and invalidates the cell.
    await asyncio.sleep(max(0.,end-before,window['arrival_window_s']-bench_elapsed_s))
    after=time.time()
    return dict(**window,arrival_epoch_s=epoch,arrival_window_end_s=end,
        hold_started_s=before,hold_ended_s=after,idle_hold_observed_s=after-before,
        bench_elapsed_s=bench_elapsed_s,
        clock_consistent=abs((before-epoch)-bench_elapsed_s)<1.,
        window_observed_complete=after>=end,
        native_drain_deadline_s=end+REQUEST_TIMEOUT_S)


async def finish_v3_measurement(controller, outputs, *, deadline_s=None):
    """Charge already-started work and controls up to one absolute drain bound."""
    deadline=(max(row['planned_arrival_s'] for row in outputs)+REQUEST_TIMEOUT_S
              if deadline_s is None else deadline_s)
    started=time.time()
    try:
        result=await asyncio.wait_for(controller.finish_measurement(deadline),
            timeout=max(.001,deadline-time.time()))
        if not isinstance(result,dict):
            raise TypeError('finish_measurement must return a drain evidence dictionary')
        result=dict(result)
    except Exception as exc:
        result=dict(drain_complete=False,drain_end_s=None,controls_end_s=None,
                    residual=None,error=repr(exc))
    result.update(drain_started_s=started,deadline_s=deadline,observed_end_s=time.time())
    return result


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
    v3=config.get('evaluation_protocol')==EVALUATION_V3
    selected_strategy=args.strategy or config['strategy']
    evaluation_system='pdblend' if selected_strategy.startswith('pdblend') else selected_strategy
    evaluation_identity={}
    if v3:
        # q is fixed in the new protocol; historical configurations are unchanged.
        if config.get('slo_attainment_target',.9)!=.9:
            raise ValueError('evaluation-v3 requires SLO attainment target 0.90')
        if args.split=='formal':
            from .evaluation_v3 import verify_cell
            if not getattr(args,'evaluation_plan',None) or not getattr(args,'evaluation_cell_id',None):
                raise ValueError('evaluation-v3 formal cells require a plan and cell id')
            evaluation_identity=verify_cell(args.evaluation_plan,args.evaluation_cell_id,
                evaluation_system,args.config,args.trace)
    if config.get('power_mode', 'instant') != 'instant':
        raise ValueError('new serving cells require explicit NVML instant power')
    freeze=json.loads(args.freeze.read_text()) if args.freeze else {}
    if args.split=='formal' and not v3:
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
    window=configure_fixed_window(args,config,trace)
    mechanisms=json.loads(args.mechanisms.read_text()) if args.mechanisms else {}
    if args.split=='formal' and not v3:
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
    rows=[]; raw_saved=False; summary=None
    cleanup=dict(evaluation_protocol=EVALUATION_V3) if v3 else None
    started=time.time()
    try:
        await runner.setup()
        await web.TCPSite(runner,'127.0.0.1',config.get('port',18080)).start()
        if args.split=='formal' and not v3:
            provenance=await verify_live_engines(controller.backend,freeze)
            await asyncio.to_thread((args.out/'live_engines.before.json').write_text,json.dumps(provenance,indent=2))
        elif args.split=='formal' and v3:
            from .provenance_v3 import verify_live_engines_v3
            provenance=await verify_live_engines_v3(controller.backend,args.evaluation_plan,
                evaluation_identity['model'],config)
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
        outputs,bench_elapsed_s=await run_trace(trace,f"http://127.0.0.1:{config.get('port',18080)}",
            config.get('model_name','Qwen2.5-14B-Instruct'),timeout_s=args.timeout,
            **({'evaluation_protocol':EVALUATION_V3} if v3 else {}))
        rows=bench_rows(trace,outputs,config['slo_ttft_s'],config['slo_tpot_s'])
        window_result=await retain_fixed_window(trace,outputs,bench_elapsed_s,window) if window else None
        if window_result:
            (args.out/'arrival_window.json').write_text(json.dumps(window_result,indent=2,allow_nan=False))
        drain_result=await finish_v3_measurement(controller,outputs,
            **(dict(deadline_s=window_result['native_drain_deadline_s']) if window_result else {})) if v3 else None
        reconfiguration_end=(drain_result.get('controls_end_s') if v3
                             else await controller.quiesce_controls())
        await asyncio.sleep(.1)
        await asyncio.to_thread(sampler.stop)
        try: await controller.journal.flush()
        except Exception as exc: controller.failure='evidence journal: '+repr(exc)
        await asyncio.to_thread(save_raw,args.out,rows,sampler.samples,sampler.utilization_samples,
            power_source=sampler.power_source,power_metadata=sampler.power_metadata)
        raw_saved=True
        summary=await asyncio.to_thread(summarize_cell,trace,rows,sampler.samples,sampler.utilization_samples,
            (config['slo_ttft_s'],config['slo_tpot_s']),sampling_error=sampler.error,
            reconfiguration_end_s=reconfiguration_end,power_source=sampler.power_source,
            power_metadata=sampler.power_metadata,require_power_mode='instant',
            **(dict(evaluation_protocol=EVALUATION_V3,drain_result=drain_result,
                    slo_attainment_target=.9,
                    arrival_lateness_limit_s=config.get('arrival_lateness_limit_s')) if v3 else {}))
        provenance_error=None
        if args.split=='formal' and not v3:
            try:
                provenance=await verify_live_engines(controller.backend,freeze)
                await asyncio.to_thread((args.out/'live_engines.after.json').write_text,json.dumps(provenance,indent=2))
            except Exception as exc:
                provenance_error=repr(exc)
        unchanged=not formal_freeze_gaps(freeze) if not v3 else True
        if v3 and args.split=='formal':
            try:
                final_identity=verify_cell(args.evaluation_plan,args.evaluation_cell_id,
                    evaluation_system,args.config,args.trace)
                unchanged=final_identity==evaluation_identity
                provenance=await verify_live_engines_v3(controller.backend,args.evaluation_plan,
                    evaluation_identity['model'],config)
                await asyncio.to_thread((args.out/'live_engines.after.json').write_text,json.dumps(provenance,indent=2))
            except Exception as exc:
                unchanged=False; provenance_error=repr(exc)
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
        if v3:
            summary.update(evaluation_identity)
            summary['formal_eligible']=bool(summary['formal_eligible'] and summary['measurement_valid'])
        if window_result:
            valid=(window_result['window_observed_complete'] and window_result['clock_consistent']
                and summary['measurement_end_s']>=window_result['arrival_window_end_s']
                and abs(summary['measurement_start_s']-window_result['arrival_epoch_s'])<1e-5)
            summary.update(measurement_window_protocol=FIXED_WINDOW_PROTOCOL,
                fixed_window=window_result,fixed_window_valid=valid,
                slo_protocol=config['slo_protocol'],slo_scale=config['slo_scale'],
                goodput_fixed_arrival_window_rps=summary['good_requests']/window['arrival_window_s'],
                formal_eligible=False)
            if not valid:
                summary.update(measurement_valid=False,slo_feasible=False,validity='invalid_fixed_window')
        summary['runtime_error']=controller.failure
        summary['admission_planning']=controller.planning_stats.summary()
        if controller.failure:
            summary['validity']='invalid_runtime'
            summary['capacity_observation_valid']=False
            summary['formal_eligible']=False
            if v3: summary.update(measurement_valid=False,slo_feasible=False)
        if args.split=='formal' and (not unchanged or provenance_error):
            summary['validity']='invalid_provenance'
            summary['provenance_error']=provenance_error or 'frozen artifact changed during the run'
            if v3: summary.update(measurement_valid=False,slo_feasible=False,formal_eligible=False)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps(summary,allow_nan=False),flush=True)
        return summary
    finally:
        if sampler:
            await asyncio.to_thread(sampler.stop)
            if v3 and not raw_saved:
                await asyncio.to_thread(save_raw,args.out,rows,sampler.samples,sampler.utilization_samples,
                    power_source=sampler.power_source,power_metadata=sampler.power_metadata)
        if v3:
            cleanup['cleanup_started_s']=time.time()
            cleanup['measurement_stopped_s']=time.time()
            try:
                await asyncio.wait_for(runner.cleanup(),timeout=5)
                cleanup['cleanup_complete']=True
            except Exception as exc:
                cleanup.update(cleanup_complete=False,error=repr(exc))
            cleanup['cleanup_end_s']=time.time()
            cleanup['cleanup_duration_s']=cleanup['cleanup_end_s']-cleanup['cleanup_started_s']
            (args.out/'cleanup.json').write_text(json.dumps(cleanup,indent=2,allow_nan=False))
            if summary is not None:
                summary['post_measurement_cleanup']=cleanup
                (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        else:
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
    p.add_argument('--evaluation-plan',type=Path)
    p.add_argument('--evaluation-cell-id')
    p.add_argument('--timeout',type=float,default=900)
    p.add_argument('--slo-ttft-s',type=float)
    p.add_argument('--slo-tpot-s',type=float)
    p.add_argument('--slo-scale',type=float,choices=(.5,1.,2.))
    args=p.parse_args()
    with node_lease():
        asyncio.run(run_cell(args))


if __name__=='__main__':
    main()
