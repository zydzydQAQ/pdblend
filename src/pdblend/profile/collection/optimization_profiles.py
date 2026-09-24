"""Bounded resident collection for low B and measured common-window Mixed power.

Use a fresh SamplingEpochs cohort after existing jobs. Nothing is enqueued by
this module. One engine is reused for training and frozen independent holdout.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import os
import time
from pathlib import Path

from pdblend.profile.calibration import optimization_profiles as cal
from pdblend.profile.calibration.power_calibration import write_immutable
from pdblend.profile.collection import sampling_guard as guard
from pdblend.profile.collection.wave import atomic_json
from pdblend.profile.collection.window_sampling import summarize_window
from pdblend.source_inventory import implementation_hashes, verify_implementation


def prepare(*, base_candidate, identity_raw, out, frequencies=(1500,), **options):
    out = Path(out)
    if out.exists():
        raise FileExistsError('fresh immutable optimization package required')
    source = json.loads(Path(identity_raw).read_text())
    from pdblend.profile.query.model import PerfModel
    base = PerfModel.load(base_candidate)
    if (base.system, Path(base.model).name, base.tp, base.pp) != tuple(source[k] for k in ('system','model_id','tp','pp')):
        raise ValueError('optimization package base/model/topology differ')
    if any(f not in base.freqs for f in frequencies):
        raise ValueError('optimization frequencies must belong to the selected base')
    plan = cal.make_plan(source,frequencies=frequencies,**options)
    write_immutable(out/'plan.json',plan)
    manifest = dict(kind=cal.KIND, **{k:source[k] for k in cal.IDENTITY},
        serving_entrypoint=cal.SERVING_ENTRYPOINT,native_control=True,native_timing_crosscheck_passed=False,
        historical_base_timing_serving_variant=source.get('serving_entrypoint','vllm serve (historical)'),
        plan_sha256=cal.digest(out/'plan.json'),
        implementation_sha256=implementation_hashes(packages=('pdblend','pdblend_runtime')),
        environment=source['environment'], inputs={name:dict(path=str(Path(path).resolve()),sha256=cal.digest(path))
            for name,path in (('base_candidate',base_candidate),('identity_raw',identity_raw))},
        required_sampling='new_explicit_epoch_cohort', formal_eligible=False,energy_comparable=False)
    write_immutable(out/'manifest.json',manifest)
    return manifest


def load_package(package):
    package = Path(package)
    manifest = json.loads((package/'manifest.json').read_text())
    plan = json.loads((package/'plan.json').read_text())
    if manifest['kind'] != cal.KIND or cal.digest(package/'plan.json') != manifest['plan_sha256']:
        raise ValueError('optimization package/plan binding changed')
    verify_implementation(manifest['implementation_sha256'],packages=('pdblend','pdblend_runtime'))
    for binding in manifest['inputs'].values():
        if cal.digest(binding['path']) != binding['sha256']:
            raise ValueError('optimization frozen input changed')
    return manifest, plan


async def native_events(client,after_seq):
    import aiohttp
    async with client.session.get(client.base_url+'/baseline/events',params=dict(after_seq=after_seq),
                                  timeout=aiohttp.ClientTimeout(total=30)) as response:
        response.raise_for_status()
        value=await response.json()
    if not isinstance(value,dict):
        raise ValueError('native scheduler events returned no object')
    return value


async def measure_repeat(profiler, client, gpus, point, repeat, *, _events=None):
    """Power and timing share the full measured interval; no decode relabeling."""
    from pdblend.bench.gates import random_prompt
    tag = 'optimization-'+cal.key(point).replace('/','-')+f'-{point["phase"]}-{repeat}'
    probes, warm_probes = [], []
    rate = point['prefill_rate_rps']
    events=_events or (lambda after_seq:native_events(client,after_seq))

    async def probe_cycle(seconds, rows, *, start):
        if not rate:
            await asyncio.sleep(seconds)
            return
        count = max(1, round(seconds*rate))
        for i in range(count):
            target = start+i/rate
            await asyncio.sleep(max(0,target-time.time()))
            submitted = time.time()
            # A delayed injector changes the offered rate and cannot become a
            # nominal-rate training point silently.
            if submitted-target > min(.1,.1/rate):
                raise RuntimeError('mixed prefill injector missed measured cadence')
            result = await client.complete(random_prompt(point['chunk_tokens'],9701+repeat*100+i),1,
                tag+f'-probe-{len(rows)}-{int(start)}')
            if result.error or result.first_token_s is None or not result.stream_done:
                raise RuntimeError('mixed prefill probe did not terminate normally')
            rows.append(dict(submitted_s=result.submitted_s,first_token_s=result.first_token_s,
                             finished_s=result.finished_s,token_times_s=list(result.token_times_s)))
        await asyncio.sleep(max(0,start+seconds-time.time()))

    async with profiler._background(client,point['batch'],point['context_tokens'],tag) as (live,tasks):
        settle_start = time.time()
        await probe_cycle(point['settle_s'],warm_probes,start=settle_start)
        profiler._require_running(tasks)
        before=await events(getattr(profiler,'_optimization_native_cursor',0))
        before_received=time.time()
        # Public vLLM may decorate X-Request-Id. Bind the actual native IDs
        # observed after all warm probes completed; do not guess that encoding.
        steady=[row for row in before.get('events',[]) if row.get('kind')=='schedule' and
                row.get('prefill_mode') is False and len(row.get('schedule_queue',[]))==point['batch']]
        if not steady:
            raise ValueError('native scheduler has no observed steady background batch')
        background_ids=list(steady[-1]['schedule_queue'])
        sampler = profiler.meter.sampler(gpus)
        sampler.start()
        start = time.time()
        try:
            await probe_cycle(point['measure_s'],probes,start=start)
            end = time.time()
            profiler._require_running(tasks)
        finally:
            sampler.stop()
        if sampler.error:
            raise RuntimeError('optimization power sampler failed: '+str(sampler.error))
        after_requested=time.time()
        after=await events(before['next_seq'])
        profiler._optimization_native_cursor=after['next_seq']
        token_times = [list(r.token_times_s) for r in live]
        power = [x for x in sampler.samples if start <= x[0] <= end]
        clocks = [x for x in sampler.frequency_samples if start <= x[0] <= end]
        summary = summarize_window(token_times=token_times,context=point['context_tokens'],
            start_s=start,end_s=end,power=power,frequency=clocks,gpu_count=len(gpus),
            settle_s=start-settle_start,measurement_s=point['measure_s'])
        summary['raw_window_mean_power_w'] = summary['power_w']
        summary['energy_j'] = cal.integrate_power(power,start,end,len(gpus))
        summary['power_w'] = summary['energy_j']/(end-start)
        if abs(summary['mean_freq_mhz']/point['freq_mhz']-1) > .05:
            raise RuntimeError('optimization clock differs from configured tier')
        if point['family'] == 'mixed':
            if len(probes) < 2 or any(p['finished_s'] > end for p in probes):
                raise RuntimeError('common-window mixed prefill evidence incomplete')
            summary.update(prefill_requests=len(probes),offered_prefill_rate_rps=rate,
                observed_prefill_rate_rps=len(probes)/(end-start),
                power_scope='common_wall_clock_window_with_decode_and_periodic_prefill')
        else:
            summary['power_scope'] = 'continuous_decode_only'
        evidence=dict(point=point,repeat=repeat,start_s=start,end_s=end,settle_start_s=settle_start,
            serving_entrypoint=profiler.raw['serving_entrypoint'],
            token_times_s=token_times,probes=probes,warmup_probes=warm_probes,power=power,frequency=clocks,
            gpu_count=len(gpus),measured_gpu_ids=list(gpus),summary=summary,
            power_sensor_lag_qualified=False,native_timing_qualified=False,
            native_schedule=dict(before=before,after=after,before_received_s=before_received,
                after_requested_s=after_requested,background_request_ids=background_ids,
                submitted_background_request_ids=[tag+f'-{i}' for i in range(point['batch'])],
                native_id_binding='observed_steady_schedule_after_warm_probes_completed'))
        try:
            evidence['native_batch_observation']=cal.native_batch_observation(evidence)
            if not evidence['native_batch_observation']['passed']:
                raise ValueError('native actual batch distribution does not qualify offered low batch')
        except ValueError as exc:
            exc.evidence=evidence
            raise
        return evidence


async def run_existing(*, package, profiler, client, gpus, out, window_boundary, qualification_guard):
    manifest, plan = load_package(package)
    out = Path(out)
    if (any(profiler.raw.get(k) != manifest[k] for k in cal.IDENTITY) or
            len(gpus) != manifest['tp'] or len(set(gpus)) != len(gpus)):
        raise ValueError('optimization resident identity differs')
    if profiler.raw.get('serving_entrypoint') != manifest.get('serving_entrypoint') or manifest.get('serving_entrypoint') != cal.SERVING_ENTRYPOINT:
        raise ValueError('optimization requires the explicitly measured native serving variant')
    for name in ('image_digest','vllm','torch','cuda','hardware_id'):
        if not manifest['environment'].get(name) or profiler.raw['environment'].get(name) != manifest['environment'][name]:
            raise ValueError('optimization runtime differs: '+name)
    before = deepcopy(profiler.raw)
    binding = dict(package_sha256=cal.digest(Path(package)/'manifest.json'),plan_sha256=manifest['plan_sha256'],
        serving_entrypoint=profiler.raw['serving_entrypoint'],
        serving_configuration=deepcopy(profiler.raw['serving_configuration']),
        identity={k:manifest[k] for k in cal.IDENTITY},environment=profiler.raw['environment'])
    raw = dict(kind=cal.KIND,binding=binding,training={},holdout={})
    if (out/'raw.json').exists():
        raise FileExistsError('new optimization output required; incomplete samples remain archived')
    write_immutable(out/'plan.json',plan)
    write_immutable(out/'package-manifest.json',manifest)
    phases = []
    result = dict(status='failed',complete=False,formal_eligible=False,energy_comparable=False)
    started = time.monotonic()
    try:
        training = []
        candidate = None
        for phase in ('training','holdout'):
            if phase == 'holdout':
                candidate = cal.fit_component(plan,training,manifest['inputs']['base_candidate']['sha256'])
                write_immutable(out/'candidate.json',candidate)
            for point in plan[phase]:
                row = dict(point=point,repeats=[])
                raw[phase][cal.key(point)] = row
                for index in range(3):
                    wait_start = time.monotonic()
                    await guard.call(window_boundary,point,index,phase)
                    phases.append(dict(phase='synchronization',seconds=time.monotonic()-wait_start))
                    stamp = guard.snapshot(qualification_guard)
                    saved = guard.save_binding(out,stamp)
                    profiler._lock(point['freq_mhz'],gpus)
                    measure_start = time.monotonic()
                    try:
                        evidence = await asyncio.wait_for(measure_repeat(profiler,client,gpus,point,index),timeout=600)
                    except BaseException as exc:
                        rejected=getattr(exc,'evidence',None)
                        if rejected is not None:
                            rejected.update(epoch_binding=stamp,plan_sha256=manifest['plan_sha256'])
                            path=out/'rejected'/f'{phase}-{cal.key(point).replace("/","-")}-{index}.json'
                            write_immutable(path,rejected)
                            raw.setdefault('rejected_windows',[]).append(dict(samples_file=str(path.relative_to(out)),
                                samples_sha256=cal.digest(path),qualification=saved,error=str(exc)))
                        raise
                    elapsed = time.monotonic()-measure_start
                    guard.unchanged(qualification_guard,stamp)
                    evidence.update(epoch_binding=stamp,plan_sha256=manifest['plan_sha256'],
                        frozen_candidate_sha256=cal.digest(out/'candidate.json') if candidate else None)
                    sample = out/'samples'/f'{phase}-{cal.key(point).replace("/","-")}-{index}.json'
                    write_immutable(sample,evidence)
                    row['repeats'].append(dict(samples_file=str(sample.relative_to(out)),samples_sha256=cal.digest(sample),
                        qualification=saved))
                    measured_s = evidence['end_s']-evidence['start_s']
                    phases.extend([dict(phase='measurement',seconds=measured_s),
                        dict(phase='prefill_settle_cleanup',seconds=max(0,elapsed-measured_s))])
                    atomic_json(out/'raw.json',raw)
                if phase == 'training':
                    training.append(dict(point=point,repeats=[json.loads((out/r['samples_file']).read_text())['summary']
                        for r in row['repeats']]))
                print(f'optimization {phase} {len(raw[phase])}/{len(plan[phase])}: {cal.key(point)}',flush=True)
        measured = cal.observations(raw,out,plan)
        if candidate != cal.fit_component(plan,measured['training'],manifest['inputs']['base_candidate']['sha256']):
            raise ValueError('frozen training candidate changed during holdout')
        audit = cal.audit_component(candidate,plan,measured['holdout'])
        atomic_json(out/'audit.json',audit)
        result.update(status='passed',complete=True,components_passed=audit['passed'],
            raw_sha256=cal.digest(out/'raw.json'),candidate_sha256=cal.digest(out/'candidate.json'),
            audit_sha256=cal.digest(out/'audit.json'),base_profile_sha256=candidate['base_profile_sha256'],
            queue_receipt_semantics='measurement_complete_only; independent_qualification_is_separate')
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        occupied = time.monotonic()-started
        totals = {name:sum(p['seconds'] for p in phases if p['phase'] == name)*len(gpus)
                  for name in ('synchronization','measurement','prefill_settle_cleanup')}
        accepted = totals['measurement'] if result.get('components_passed') else 0.0
        result['gpu_phase_metrics'] = dict(occupied_gpu_s=occupied*len(gpus),
            accepted_measurement_gpu_s=accepted,effective_fraction=accepted/max(occupied*len(gpus),1e-9),
            phases_gpu_s=totals,unqualified_measurement_gpu_s=totals['measurement']-accepted,
            scope='resident_collection; engine_loading_reported_by_job')
        atomic_json(out/'raw.json',raw)
        atomic_json(out/'completion.json',result)
        if profiler.raw != before:
            raise ValueError('optimization collection mutated parent profile archive')
    return result


def native_specs(specs):
    """Opt-in only for this new panel; historical profilers keep their launchers."""
    selected=[replace(spec,native_control=True) for spec in specs]
    if any(spec.command()[1:3] != ['-m',cal.SERVING_ENTRYPOINT] for spec in selected):
        raise ValueError('optimization native launcher command differs')
    return selected


def run(*,package,model,gpus,base_port,out,epochs_root,member):
    from pdblend.profile.collection.profiler import Profiler, _load_flock
    from pdblend.profile.collection.sampling_epochs import SamplingEpochs
    from pdblend.engine.launcher import Fleet
    from pdblend.engine.client import EngineClient
    manifest, _ = load_package(package)
    if len(gpus) != manifest['tp']:
        raise ValueError('optimization job requires one exact TP group')
    out = Path(out)
    profiler = Profiler(model,gpus,tp=manifest['tp'],pp=1,system='pdblend',out_dir=out/'profiler',
        hardware_id=manifest['environment']['hardware_id'],base_port=base_port,kv_connector='P2pNcclConnector')
    receipt=os.environ.get('PDBLEND_MODEL_VERIFICATION_RECEIPT')
    if not receipt or not Path(receipt).is_file():
        raise ValueError('optimization native serving requires a verified model receipt')
    profiler.specs=native_specs(profiler.specs)
    profiler.raw.update(serving_entrypoint=cal.SERVING_ENTRYPOINT,serving_configuration=dict(
        native_control=True,model_verification_sha256=cal.digest(receipt),
        native_timing_crosscheck_passed=False,historical_base_timing_is_native_qualified=False))
    epoch = SamplingEpochs(Path(epochs_root),member,profiler)
    started = time.monotonic()
    result = dict(status='failed',complete=False,formal_eligible=False,energy_comparable=False)
    try:
        with Fleet(profiler.specs,out/'logs') as fleet:
            loading = time.monotonic()
            with _load_flock():
                fleet.start_all()
            loading_s = time.monotonic()-loading
            instance = fleet[profiler.specs[0].instance_id]
            profiler.raw['kv_capacity_tokens'] = profiler._kv_capacity(instance)
            async def collect():
                await epoch.ready()
                async with EngineClient(instance.spec.instance_id,instance.spec.base_url) as client:
                    value = await run_existing(package=package,profiler=profiler,client=client,gpus=gpus,
                        out=out/'components',window_boundary=epoch.window_boundary,
                        qualification_guard=epoch.qualification_guard)
                await epoch.retire()
                return value
            try:
                result = asyncio.run(collect())
            except BaseException as exc:
                epoch.fail(exc)
                raise
        profiler.meter.reset_all()
        epoch.released()
        result['engine_load_gpu_s'] = loading_s*len(gpus)
    except BaseException as exc:
        epoch.fail(exc)
        result.update(status='failed',complete=False,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        profiler.meter.reset_all()
        result['total_job_gpu_s'] = (time.monotonic()-started)*len(gpus)
        accepted = result.get('gpu_phase_metrics',{}).get('accepted_measurement_gpu_s',0)
        result['effective_job_gpu_fraction'] = accepted/max(result['total_job_gpu_s'],1e-9)
        atomic_json(out/'completion.json',result)
    return result


def preflight(*,package,model,gpus,epochs_root,member,base_port=8100,**unused):
    """Validate source/model/runtime inputs without constructing GPU objects."""
    from importlib.metadata import version
    from pdblend.model_registry import ModelRegistry
    manifest,plan = load_package(package)
    if len(gpus) != manifest['tp'] or len(set(gpus)) != len(gpus):
        raise ValueError('optimization preflight requires one nonoverlapping TP group')
    cohort = json.loads((Path(epochs_root)/'cohort.json').read_text())
    if (member not in cohort.get('members',[]) or not cohort.get('cohort_id') or
            len(set(cohort['members'])) != len(cohort['members'])):
        raise ValueError('optimization preflight cohort membership invalid')
    receipt=os.environ.get('PDBLEND_MODEL_VERIFICATION_RECEIPT')
    if not receipt or not Path(receipt).is_file():
        raise ValueError('optimization native serving requires a verified model receipt')
    registry = ModelRegistry(Path(model).parent,verification_receipt=receipt)
    spec = registry.get(Path(model).name)
    spec.validate_config(); spec.validate_topology(manifest['tp'],1)
    if any(getattr(spec,k) != manifest[k] for k in ('model_id','model_hash','tokenizer_hash')):
        raise ValueError('optimization preflight model/tokenizer changed')
    for package_name in ('vllm','torch'):
        if version(package_name) != manifest['environment'][package_name]:
            raise ValueError('optimization pinned runtime differs: '+package_name)
    from pdblend.engine.launcher import make_specs
    specs=native_specs(make_specs(model,gpus,tp=manifest['tp'],pp=1,base_port=base_port))
    return dict(status='cpu_preflight_passed',hardware_executed=False,formal_eligible=False,
        serving_entrypoint=cal.SERVING_ENTRYPOINT,native_control=True,
        launcher_commands=[spec.command() for spec in specs],model_verification_sha256=cal.digest(receipt),
        package_sha256=cal.digest(Path(package)/'manifest.json'),plan_sha256=manifest['plan_sha256'],
        cohort_id=cohort['cohort_id'],member=member,training_points=len(plan['training']),
        holdout_points=len(plan['holdout']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='command',required=True)
    prep = modes.add_parser('prepare')
    prep.add_argument('--base-candidate',required=True,type=Path)
    prep.add_argument('--identity-raw',required=True,type=Path)
    prep.add_argument('--out',required=True,type=Path)
    prep.add_argument('--frequencies',nargs='+',type=int,default=[1500])
    merge = modes.add_parser('merge')
    merge.add_argument('--components',nargs='+',type=Path,required=True)
    merge.add_argument('--out',type=Path,required=True)
    collect = modes.add_parser('run')
    for name in ('package','out','epochs-root'):
        collect.add_argument('--'+name,type=Path,required=True)
    for name in ('model','member'):
        collect.add_argument('--'+name,required=True)
    collect.add_argument('--gpus',nargs='+',type=int,required=True)
    collect.add_argument('--base-port',type=int,required=True)
    collect.add_argument('--preflight-only',action='store_true')
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'prepare':
        result = prepare(**args)
    elif command == 'merge':
        result = cal.merge_components(args['components'],args['out'])
    else:
        only = args.pop('preflight_only')
        result = preflight(**args)
        if only:
            atomic_json(args['out']/'preflight.json',result)
        else:
            result = run(**args)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
