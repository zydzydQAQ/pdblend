"""Prepare and audit a real 31.5-minute DynamoLLM hierarchy validation.

``generate`` performs only CPU/file work. ``run`` is an explicitly queued GPU
stage. Period coverage and actual adaptive actions are reported separately.
"""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import traceback

from .budget import read_budget
from .baselines import DynamoLLMPolicy
from .calibration_setup import (validated_inputs, read, write, positive, spec,
    verify_artifacts, dynamo_pools, current_engine_sources)
from .evidence import sha256, freeze_files, validate_freeze
from .frequency import verify_frozen_costs
from .profiles import ProfileStore, OutputPredictor
from .topology import validate_layout


DURATION = 1890.
PERIODS = dict(DynamoLLMPolicy.PERIODS)
PURPOSE = 'real-clock DynamoLLM mechanism validation; not an energy comparison or capacity calibration'


def trace(seed=77, rates=(.04, .08, .12, .08, .06, .04), *,
          arrival_process='poisson', phase_shapes=None):
    if len(rates) != 6 or any(not math.isfinite(r) or r <= 0 for r in rates):
        raise ValueError('six positive, finite diagnostic phase rates required')
    if arrival_process not in ('poisson','periodic'):
        raise ValueError('diagnostic arrivals must be poisson or periodic')
    shapes = [(n, out) for n in (128, 512, 2048) for out in (64, 192, 384)]
    phase_shapes = [shapes]*6 if phase_shapes is None else [s if s is not None else shapes for s in phase_shapes]
    if (len(phase_shapes)!=6 or any(not choices or any(len(pair)!=2 or any(
            type(v) is not int or v<=0 for v in pair) for pair in choices) for choices in phase_shapes)):
        raise ValueError('six nonempty sets of positive integer input/output shapes required')
    rng = random.Random(seed); arrivals = [(0.,0)]
    for phase, rate in enumerate(rates):
        start = at = phase * 300.; number = 0
        end = (phase + 1) * 300. if phase < 5 else DURATION
        while True:
            number += 1
            at = at+rng.expovariate(rate) if arrival_process=='poisson' else start+number/rate
            if at >= end: break
            arrivals.append((at,phase))
    arrivals.append((DURATION,5))  # The real client keeps every hierarchy alive.
    requests = []; prompts = []
    for index, (at,phase) in enumerate(sorted(arrivals)):
        choices=phase_shapes[phase];n, out = choices[(index + seed) % len(choices)]
        requests.append(dict(arrival_s=at, prompt_len=n, output_len=out))
        prompts.append(([9707, 1879, 13] * (n // 3 + 1))[:n])
    return dict(schema=1, seed=seed, duration_s=DURATION, requests=requests, prompts=prompts,
        purpose=PURPOSE, phase_rates=list(rates),arrival_process=arrival_process,phase_shapes=phase_shapes,
        visibility='arrival schedule and prescribed output work go only to the client; controller sees requests causally')


def predict_scale_shard(profiles, costs, instances, assignments, workload, prior, ttft_s, tpot_s,
                        *, assume_first_completed_by_s=None):
    """CPU prerequisite, not an observed action or a performance guarantee.

    The unconditional case uses only the historical prior. An explicit offline
    assumption may additionally show what happens after the first request has
    completed successfully by a deadline. This is a conditional prediction,
    never a measured completion or history injected into the live controller.
    """
    from .dynamo import DynamoScheduler
    from .dynamo_topology import DynamoTopologyPlanner, TopologyCost
    from .types import RequestBudget, InstanceState, RuntimeSnapshot
    scheduler=DynamoScheduler(profiles,assignments)
    predictor=OutputPredictor(prior)
    prefix=sorted((r for r in workload['requests'] if r['arrival_s']<=300),key=lambda r:r['arrival_s'])
    if assume_first_completed_by_s is not None and (not prefix or
            not prefix[0]['arrival_s'] < assume_first_completed_by_s < 300):
        raise ValueError('conditional completion deadline must follow arrival and precede the real 300 second epoch')
    assumed=False
    for index,r in enumerate(prefix):
        if assume_first_completed_by_s is not None and not assumed and r['arrival_s']>assume_first_completed_by_s:
            predictor.observe_completed(prefix[0]['prompt_len'],prefix[0]['output_len'])
            assumed=True
        scheduler.arrival(RequestBudget(str(index),r['arrival_s'],r['prompt_len'],
            predictor.predict(r['prompt_len']),ttft_s,tpot_s,output_limit=r['output_len']))
    snapshot=RuntimeSnapshot(0,300,tuple(InstanceState(s.instance_id,'mixed',s.tp,
        tuple(s.gpus),300,0,2520,0,0,0) for s in instances))
    planner=DynamoTopologyPlanner(profiles,[TopologyCost(**c) for c in costs])
    forecast=scheduler.forecast(300)
    proposal=planner.choose(snapshot,assignments,forecast,'ScaleShard',cached_weights=True)
    return dict(status='conditional_cpu_prediction_not_executed' if assumed else 'cpu_prediction_not_executed',
        epoch_s=300,arrived_requests=len(prefix),
        prediction='historical prior, then completed history only; max_tokens never caps the prediction',
        assumed_completion=(dict(request_index=0,completed_by_s=assume_first_completed_by_s,
            output_tokens=prefix[0]['output_len'],measured=False,
            condition='first request successfully completes its prescribed output within the validation client timeout') if assumed else None),
        forecast=forecast,
        proposal=({**proposal,'source_cost':asdict(proposal['source_cost'])} if proposal else None))


def generate(manifest, out):
    out = Path(out).resolve()
    if out.exists(): raise ValueError('refusing to overwrite Dynamo validation preparation')
    profiles, transfers, search, template, freq, freq_raw, costs, artifacts = validated_inputs(manifest)
    if manifest.get('power_mode','instant')!='instant':
        raise ValueError('current Dynamo validation requires the explicit instantaneous power source')
    desired = [spec(s) for s in manifest['instances']]
    initial = [spec(s) for s in manifest['initial_instances']]
    validate_layout(desired, range(8)); validate_layout(initial, range(8))
    if len(desired) < 3 or not {1, 2} <= {s.tp for s in desired}:
        raise ValueError('at least three instances including TP1 and TP2 required for independent serving during changes')
    reachable = {tp for c in costs for tp in c['target_tps']} | {s.tp for s in desired}
    if not reachable <= {s.tp for s in desired}:
        raise ValueError('every reachable TP needs an ordinary output-reference instance in the starting layout')
    dataset = manifest.get('initialization_dataset', 'sharegpt')
    corpus_path = Path(manifest['corpus']) / (dataset + '.json')
    records = read(corpus_path)['calibration']
    if len(records) < 128: raise ValueError('independent calibration history is required for initial pools/predictor')
    assignments, initialization = dynamo_pools(records, [s.instance_id for s in desired])
    outputs = sorted(r['output_tokens'] for r in records)
    prior = outputs[min(len(outputs)-1, math.ceil(.9*len(outputs))-1)]
    workload = trace(manifest.get('seed',77), manifest.get('phase_rates',(.04,.08,.12,.08,.06,.04)),
        arrival_process=manifest.get('arrival_process','poisson'),phase_shapes=manifest.get('phase_shapes'))
    if any(r['prompt_len']+r['output_len']>template['max_model_len'] for r in workload['requests']):
        raise ValueError('diagnostic request exceeds the real engine context limit')
    store = ProfileStore.load(manifest['profiles'])
    for tp in reachable:
        for request in workload['requests']:
            if store.lookup('mixed',tp,2520,request['prompt_len'],
                    request['prompt_len']+request['output_len'],1) is None:
                raise ValueError('diagnostic shape is outside a reachable TP profile')
    shard_prediction=predict_scale_shard(store,costs,desired,assignments,workload,prior,
        manifest.get('slo_ttft_s',5),manifest.get('slo_tpot_s',.1),
        assume_first_completed_by_s=manifest.get('assume_first_completed_by_s'))
    if manifest.get('assume_first_completed_by_s') is not None:
        if manifest['assume_first_completed_by_s'] != 180:
            raise ValueError('conditional completion must use the actual 180 second validation timeout')
        shard_prediction['unconditional_cold_start']=predict_scale_shard(store,costs,desired,assignments,
            workload,prior,manifest.get('slo_ttft_s',5),manifest.get('slo_tpot_s',.1))
    if manifest.get('require_scale_shard_prediction'):
        proposal=shard_prediction['proposal']
        if (not proposal or tuple(proposal['source_cost']['source_tps'])!=(2,) or
                tuple(proposal['add_tps'])!=(1,1)):
            raise ValueError('current measured inputs do not predict the diagnostic TP2 to two TP1 ScaleShard')
    root = Path(manifest['campaign_root']).resolve(); budget = read_budget(root)
    preparation_limit = positive(manifest.get('preparation_limit_s',600),'preparation limit')
    run_limit = positive(manifest.get('run_limit_s',3000),'run limit')
    if run_limit < DURATION+180 or preparation_limit+run_limit+60 > budget['remaining_s']:
        raise ValueError('complete real-clock validation does not fit the effective authorized campaign budget')
    template = dict(template, operation_timeout_s=45, transfer_buffer_bytes=4*1024**3,
        verify_transport=False, validated_tp_pairs=sorted({(t['source_tp'],t['target_tp']) for t in transfers['links']}))
    template_path = out/'engine-template.json'
    config = dict(strategy='dynamollm',port=manifest.get('port',18080),model_name='Qwen2.5-14B-Instruct',
        profiles=str(Path(manifest['profiles']).resolve()),instances=[s.endpoint() for s in desired],
        journal=str(out/'run/control.jsonl'),slo_ttft_s=manifest.get('slo_ttft_s',5),
        slo_tpot_s=manifest.get('slo_tpot_s',.1),output_prior=prior,max_pending=256,
        manage_clocks=True,node_gpus=list(range(8)),park_idle=True,prepare_peers=True,
        power_mode='instant',
        allow_unprofiled_fallback=False,dynamo_assignments=assignments,dynamo_input_cuts=[255,1023],
        dynamo_output_cuts=[99,349],frequency_costs=freq,frequency_evidence=freq_raw,
        topology_costs=costs,retained_weights=manifest['retained_weights'],
        topology=dict(runtime_dir=str(out/'runtime'),image=manifest['image'],engine_template=str(template_path)))
    verify_frozen_costs(config,profiles,dict(files=artifacts,groups=dict(profiles=list(artifacts)),
                                          identities=dict(engine_image=manifest['image'])))
    restoration = dict(instances=[asdict(s) for s in desired],initial_instances=[asdict(s) for s in initial+desired],
        image=manifest['image'],engine_template=str(template_path),retained_weights=manifest['retained_weights'],
        ownership_root=str(out))
    generated = {out/'runtime-config.json':config,out/'trace.json':workload,
        template_path:template,out/'restoration.json':restoration,
        out/'scale-shard-prediction.json':shard_prediction}
    out.mkdir(parents=True)
    for p, value in generated.items(): write(p,value)
    artifacts.update({str(p):sha256(p) for p in generated})
    artifacts[str(corpus_path.resolve())] = sha256(corpus_path)
    write(out/'input-evidence.json',dict(artifacts=artifacts,purpose=PURPOSE))
    stages = [dict(name='dynamo-revalidation-prepare',gpu=True,limit_s=preparation_limit,
        argv=['python3','-m','ecopadg.serving.calibration_setup','restore','--manifest',str(out/'restoration.json'),
              '--out',str(out/'preparation')]),
        dict(name='dynamo-revalidation-real-clock',gpu=True,limit_s=run_limit,requires=['dynamo-revalidation-prepare'],
        argv=['python3','-m','ecopadg.serving.dynamo_validation_setup','run','--setup',str(out),'--out',str(out/'run')])]
    write(out/'campaign.json',dict(output=str(root),budget_s=budget['limit_s'],stages=stages))
    result = dict(status='prepared_not_executed',purpose=PURPOSE,periods_s=PERIODS,
        initialization=dict(dataset=dataset,split='calibration',source_sha256=sha256(corpus_path),**initialization),
        expected_minimum_epochs=dict(ScaleInst=1,ScaleShard=6,ScaleFreq=378),
        output_reference_tps=sorted(reachable),original_deadline_s=budget['original_deadline_s'], effective_deadline_s=budget['deadline_s'],
        budget_revision_seq=budget['revision_seq'], authorization_sha256=budget['authorization_sha256'],
        limitations=['adaptive ROI may produce no ScaleInst or ScaleShard action; those action gates stay incomplete',
            'all nine logical types exist; sparse demand may spill to larger resident pools',
            'startup, ordinary references and final layout restoration are separate from the measured workload',
            '31.5 minutes verifies the control period, not a full 30-minute post-switch energy amortization'])
    write(out/'setup.json',result)
    return result


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def audit(raw, events, rows, summary, power=()):
    epochs = {name:[e for e in events if e.get('kind')=='dynamo_control_epoch' and e.get('operation')==name]
              for name in PERIODS}
    begins = {e['transaction']:e for e in events if e.get('kind')=='topology_begin'}
    commits = {e['transaction']:e for e in events if e.get('kind')=='topology_commit'}
    cycles = {}; actions = {}; transactions = []
    for operation, period in PERIODS.items():
        stamps = []
        for event in epochs[operation]:
            transaction = event.get('result',{}).get('transaction')
            stamps.append(begins.get(transaction,event)['at_s'])
        origin = raw['period_origin_s'][operation]
        expected = int(DURATION//period)
        grid = [int(round((t-origin)/period)) for t in stamps]
        cycles[operation] = dict(count=len(stamps),expected_minimum=expected,
            declared_periods_correct=all(e.get('period_s')==period for e in epochs[operation]),
            cycle_indices=grid,passed=set(range(1,expected+1))<=set(grid) and len(set(grid))==len(grid)
                and all(index>=1 for index in grid)
                and all(e.get('period_s')==period and not e.get('error') for e in epochs[operation]))
        actions[operation] = 0
        if operation=='ScaleFreq': continue
        for event in epochs[operation]:
            if not event.get('executed'): continue
            tx = event.get('result',{}).get('transaction')
            if tx not in begins or tx not in commits: continue
            before, after = begins[tx], commits[tx]
            source = [s['tp'] for s in before['before']]; target = [s['tp'] for s in before['after']]
            if (operation=='ScaleShard' and (sum(source)!=sum(target) or sorted(source)==sorted(target))): continue
            old = {s['instance_id'] for s in before['before']}
            remaining = [rid for rid,owner in raw['routes'].items() if owner not in old and any(
                e.get('kind')=='request_end' and e.get('request_id')==rid and e.get('completed')
                and before['at_s']<=e['at_s']<=after['at_s'] for e in events)]
            actions[operation] += 1
            from ecopadg.measure.power import trapezoid_energy
            from ecopadg.metrics import clip_power_window
            energy = (trapezoid_energy(clip_power_window(power,before['at_s'],after['at_s'],pad_s=0))
                      if power else None)
            transactions.append(dict(operation=operation,transaction=tx,source_tps=source,target_tps=target,
                duration_s=after['at_s']-before['at_s'],unaffected_requests_completed=remaining,
                total_eight_gpu_energy_j=energy,capacity_recovery=before.get('capacity_recovery'),
                begin_s=before['at_s'],end_s=after['at_s']))
    clock_events = [e for e in events if e.get('kind')=='dynamo_clock_commit']
    actions['ScaleFreq'] = sum(sum(e['before'].get(i)!=frequency for i,frequency in e['after'].items()
        if i in e['before']) for e in clock_events)
    correct = len(rows)==len(raw['workload']['requests'])
    checks = []
    for index,row in enumerate(rows):
        key = raw['reference_keys'].get(str(index))
        passed = bool(index<len(raw['workload']['requests']) and key and str(row['request_id'])==str(index) and int(row['success'])==1
            and int(row['generated_tokens'])==raw['workload']['requests'][index]['output_len']
            and row['output_token_sha256']==token_hash(raw['reference'][key]))
        checks.append(passed); correct &= passed
    measurement = (summary.get('validity')=='ok' and summary.get('measurement_schema')==2
        and summary.get('gpu_count')==8 and summary.get('energy_j',0)>0
        and summary.get('power_mode')=='instant' and summary.get('power_source_verified') is True
        and summary['measurement_end_s']-summary['measurement_start_s']>=DURATION)
    failures = [e for e in events if e.get('kind') in ('topology_failure','topology_recovery_failed')]
    cycles_passed = all(v['passed'] for v in cycles.values()) and correct and measurement and not failures
    staggered = bool(transactions) and any(t['unaffected_requests_completed'] for t in transactions)
    complete = cycles_passed and all(actions.values()) and staggered and not raw.get('errors')
    fields = dict(length_prediction=False,nine_logical_pools=False,
        fragmentation=bool([e for e in events if e.get('kind')=='dynamo_pool_fragmentation']),
        scale_inst_1800s=cycles['ScaleInst']['passed'] and actions['ScaleInst']>0,
        scale_shard_300s=cycles['ScaleShard']['passed'] and actions['ScaleShard']>0,
        scale_freq_5s=cycles['ScaleFreq']['passed'] and actions['ScaleFreq']>0,
        measured_reconfiguration=bool(transactions) and all(t['total_eight_gpu_energy_j'] is not None for t in transactions),staggered_switch=staggered,
        output_correctness=correct,independent_calibration=False)
    complete = complete and fields['measured_reconfiguration']
    return dict(purpose=PURPOSE,passed=complete,status='mechanisms_covered' if complete else
        'cycles_only_or_missing_actions' if cycles_passed else 'invalid_or_incomplete',
        cycles_passed=cycles_passed,cycles=cycles,actual_action_counts=actions,
        frequency_nonempty_plan_epochs=sum(bool(e.get('plan',{}).get('frequencies')) for e in epochs['ScaleFreq']),
        frequency_action_definition='changed driver-confirmed command, not the number of plans; SM samples reported separately',
        transactions=transactions,output_matches=checks,measurement_valid=measurement,
        proposed_mechanism_fields=fields,uncompleted=[k for k,v in fields.items() if v is not True])


def mechanism_proposal(path):
    result=read(path);valid=result['cycles_passed'] and result['status']!='invalid_or_incomplete'
    return {'dynamollm':{name:dict(passed=valid and passed is True,
        artifact=str(Path(path).resolve()),sha256=sha256(path),
        reason='period and actual action evidence checked separately; independent calibration and causal-classification proof remain separate')
        for name,passed in result['proposed_mechanism_fields'].items()}}


async def run(setup, out):
    """Reuse the real controller and measurement boundary; observe, never force ROI."""
    import aiohttp
    from aiohttp import web
    from benchmarks.scripts.bench_vllm import run_trace, bench_rows
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.measure.power import PowerSampler
    from .backend import HttpEngineBackend
    from .controller import Controller
    from .profiling import HardwareProfiler
    from .measurement import save_raw, summarize_cell
    from .calibration_setup import restore_layout
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    config=read(setup/'runtime-config.json');workload=read(setup/'trace.json')
    config['journal']=str(out/'control.jsonl')
    raw=dict(purpose=PURPOSE,errors=[],reference={},reference_keys={},routes={},workload=workload)
    runner=controller=sampler=None;failure=None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),trust_env=False) as session:
        try:
            verify_artifacts(read(setup/'input-evidence.json')['artifacts'])
            package=Path(__file__).parent.parent
            raw['source_files']=await asyncio.to_thread(freeze_files,
                [*Path(__file__).parent.glob('*.py'),*package.joinpath('measure').glob('*.py'),
                 package/'metrics.py',package/'types.py',Path(run_trace.__code__.co_filename)])
            desired=config['instances'];topology={i['id']:i for i in desired}
            profiler=HardwareProfiler(session,topology,setup/'runtime')
            raw['provenance_before']=await profiler.provenance()
            expected_source=current_engine_sources()
            if any(e['image_id']!=config['topology']['image'] or e['source_files_at_import']!=expected_source
                   for e in raw['provenance_before']): raise RuntimeError('prepared engine provenance changed')
            # One ordinary reference for every public workload/TP combination.
            # Every future TP is present in this starting layout, checked above.
            shapes={(q['prompt_len'],q['output_len']) for q in workload['requests']}
            representatives={i['tp']:i for i in desired}
            raw['reference_started_s']=time.time()
            async def references(tp,instance):
                for n,count in sorted(shapes):
                    response=await profiler.call(instance,'/v1/completions',dict(
                        prompt=([9707,1879,13]*(n//3+1))[:n],max_tokens=count,
                        temperature=0,top_p=1.,seed=0,ignore_eos=True,stream=False))
                    if len(response['token_ids'])!=count or response['usage']['completion_tokens']!=count:
                        raise RuntimeError('ordinary reference output work mismatch')
                    raw['reference'][f'{tp}:{n}:{count}']=response['token_ids']
            tasks=[asyncio.create_task(references(tp,i)) for tp,i in representatives.items()]
            try: await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
            raw['reference_finished_s']=time.time()
            controller=Controller(config);runner=web.AppRunner(controller.application(),shutdown_timeout=2)
            started=time.time();await runner.setup()
            await web.TCPSite(runner,'127.0.0.1',config['port']).start()
            raw['startup_seconds']=time.time()-started
            raw['period_origin_s']=dict(controller.dynamo_scheduler.hierarchy.last)
            original_execute=controller.backend.execute
            async def observed_execute(plan):
                before=dict(controller.backend.frequency)
                await original_execute(plan)
                if plan.reason=='DynamoLLM ScaleFreq five-second epoch':
                    await controller.journal.emit(dict(kind='dynamo_clock_commit',at_s=time.time(),
                        before=before,after=dict(controller.backend.frequency),
                        outcomes=controller.backend.frequency_outcomes[-len(plan.frequencies):] if plan.frequencies else []))
            controller.backend.execute=observed_execute
            hardware=await asyncio.to_thread(PynvmlBackend,power_mode=config['power_mode'])
            sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True)
            sampler.start();await asyncio.sleep(.1)
            outputs,_=await run_trace(workload,f'http://127.0.0.1:{config["port"]}',
                config['model_name'],timeout_s=180)
            reconfiguration_end=await controller.quiesce_controls()
            await asyncio.sleep(.1);await asyncio.to_thread(sampler.stop);await controller.journal.flush()
            rows=bench_rows(workload,outputs,config['slo_ttft_s'],config['slo_tpot_s'])
            await asyncio.to_thread(save_raw,out,rows,sampler.samples,sampler.utilization_samples,
                power_source=sampler.power_source,power_metadata=sampler.power_metadata)
            summary=summarize_cell(workload,rows,sampler.samples,sampler.utilization_samples,
                (config['slo_ttft_s'],config['slo_tpot_s']),sampling_error=sampler.error,
                reconfiguration_end_s=reconfiguration_end,power_source=sampler.power_source,
                power_metadata=sampler.power_metadata,require_power_mode='instant')
            events=[json.loads(line) for line in (out/'control.jsonl').read_text().splitlines() if line]
            degrees={i['id']:i['tp'] for i in desired}
            for e in events:
                if e.get('kind')=='topology_begin':
                    degrees.update({s['instance_id']:s['tp'] for s in e['after']})
                if e.get('kind')=='admission':
                    route=e['plan']['routes'][0];raw['routes'][e['request_id']]=route['decode_id']
                    q=workload['requests'][int(e['client_request_id'])]
                    raw['reference_keys'][e['client_request_id']]=f'{degrees[route["decode_id"]]}:{q["prompt_len"]}:{q["output_len"]}'
            final=[s.endpoint() for s in controller.topology_manager.specs.values()]
            raw['final_instances']=final
            raw['provenance_after']=await HardwareProfiler(session,{i['id']:i for i in final},setup/'runtime').provenance()
            if (controller.failure or validate_freeze(raw['source_files']) or any(
                    e['image_id']!=config['topology']['image'] or e['source_files_at_import']!=expected_source
                    for e in raw['provenance_after'])):
                raise RuntimeError('runtime failed or implementation/engine provenance changed')
            await asyncio.to_thread(verify_artifacts,read(setup/'input-evidence.json')['artifacts'])
            raw['outputs']=outputs;raw['frequency_samples']=sampler.frequency_samples
            raw['summary']=summary;raw['audit']=audit(raw,events,rows,summary,sampler.samples)
            write(out/'summary.json',summary);write(out/'mechanisms.json',raw['audit'])
        except BaseException as exc:
            failure=exc;raw['errors'].append(traceback.format_exc())
        finally:
            if sampler:
                await asyncio.to_thread(sampler.stop)
                raw.setdefault('frequency_samples',sampler.frequency_samples)
                raw['power_source']=sampler.power_source
                if failure is not None: raw['partial_power_samples']=sampler.samples
            # Preserve interruption evidence before a potentially slow rollback
            # or between-run restoration can be cut short by the outer budget.
            write(out/'raw.json',raw)
            if runner:
                try:await runner.cleanup()
                except Exception:raw['errors'].append(traceback.format_exc())
            try:await restore_layout(read(setup/'restoration.json'),out.with_name(out.name+'.restoration'))
            except Exception:raw['errors'].append(traceback.format_exc())
            if raw['errors'] and 'audit' in raw:
                raw['audit'].update(passed=False,status='invalid_or_incomplete')
            write(out/'raw.json',raw)
            if 'audit' in raw:
                raw['audit']['artifacts']={str((out/name).resolve()):sha256(out/name)
                    for name in ('raw.json','control.jsonl','bench.csv','power.csv','power_source.json',
                                 'power_metadata.jsonl','summary.json') if (out/name).exists()}
                write(out/'mechanisms.json',raw['audit'])
                write(out/'mechanism-proposal.json',mechanism_proposal(out/'mechanisms.json'))
    if failure is not None or raw['errors']:
        raise RuntimeError('Dynamo mechanism validation failed; inspect raw.json') from failure
    return raw['audit']


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('generate');prepare.add_argument('--manifest',type=Path,required=True)
    prepare.add_argument('--out',type=Path,required=True)
    execute=sub.add_parser('run');execute.add_argument('--setup',type=Path,required=True)
    execute.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='generate':result=generate(read(args.manifest),args.out)
    else:
        from .campaign import node_lease
        async def cancellable():
            loop=asyncio.get_running_loop();loop.add_signal_handler(signal.SIGTERM,asyncio.current_task().cancel)
            try:return await run(args.setup,args.out)
            finally:loop.remove_signal_handler(signal.SIGTERM)
        with node_lease():result=asyncio.run(cancellable())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
