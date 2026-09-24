"""PD-owned runtime measurements on a caller's existing native fleet.

The caller owns the lease and continuously running eight-GPU PowerSampler.
This module never starts/stops that sampler, changes a queue, or grants formal
profile qualification. Off/restart is a measured mechanism, not service time.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import aiohttp

from pdblend.online.native_control import validate_state
from pdblend_runtime.probe import call
from .native_frequency_domain import validate_domain,domain_fields,with_domain
from .native_timing_plan import read_bound,digest


SCHEMA = 'pdblend-native-runtime-collection-v1'
FREQUENCIES = (1500, 2520)
REPEATS, SETTLE_S, MEASURE_S = 3, 2., 5.
HOLDOUT_REPEATS = 1
TOTAL_REPEATS = REPEATS+HOLDOUT_REPEATS
RUNTIME_PLAN = dict(training_repeats=REPEATS,holdout_repeats=HOLDOUT_REPEATS,
    training_repeat_indices=list(range(REPEATS)),holdout_repeat_indices=[REPEATS],
    training_seed=9701,holdout_seed=9702,selection_split='calibration_and_independent_holdout',
    evaluation_used_for_selection=False,settle_s=SETTLE_S,measure_s=MEASURE_S,
    frequencies_mhz=list(FREQUENCIES),transfer_input_tokens=[512,2048,7168],
    holdout_limits=dict(mean_relative_error=.10,p95_relative_error=.20,max_relative_error=.25))


def build_runtime_plan(frequency_domain_ref=None):
    """Explicit static/clock/lifecycle scope; no legacy signed transfer model."""
    if frequency_domain_ref is None:return deepcopy(RUNTIME_PLAN)
    domain=validate_domain(read_bound(frequency_domain_ref))
    plan=deepcopy(RUNTIME_PLAN)
    plan.update(schema='pdblend-native-runtime-domain-plan/v1',system='pdblend',
        model_id=domain['model_id'],tp=domain['tp'],pp=domain['pp'],
        frequency_domain_ref=deepcopy(frequency_domain_ref),**domain_fields(domain))
    plan.update(frequencies_mhz=list(domain['frequencies_mhz']),transfer_input_tokens=[],
        include_transfer=False,scope='capacity_static_clock_L1_off_wake_only',
        capacity_scope='every_replica',static_clock_lifecycle_scope='first_replica_only_others_resident',
        energy_scope='target_replica_group_watts_and_absolute_eight_board_observations_not_whole_layout_model',
        static_observed_tolerance_mhz=15,clock_ack_tolerance_mhz=15,
        maximum_observation_gap_s=1.,restore_frequency_mhz=max(domain['frequencies_mhz']),
        wake_scope='owned_engine_restart_clock_restore_then_golden_reuse_and_drain',
        handoff_prediction_qualified=False,full_profile_qualified=False,formal_eligible=False)
    return plan


def need(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_inventory(specs, fleet, meter, sampler, gpu_uuids):
    """Fail before any mutation if this is not the exact resident lease."""
    gpus = list(meter.gpus)
    need(len(gpus) == len(set(gpus)) == len(gpu_uuids) == len(set(gpu_uuids)) == 8,
         'eight distinct physical devices required')
    need(all(isinstance(u, str) and u.startswith('GPU-') for u in gpu_uuids), 'GPU UUIDs missing')
    need(sampler.backend is meter.backend and list(sampler.gpus) == gpus,
         'caller sampler must observe the same eight-device backend/order')
    need(sampler.sample_clocks is True and sampler.error is None,
         'live frequency-enabled sampler required')
    need(getattr(sampler,'_thread',None) is not None and sampler._thread.is_alive(),
         'caller power sampler thread is not running')
    source = sampler.power_source
    need(source.get('mode') == 'instant' and source.get('field_id') == 186
         and source.get('scope_id') == 0 and source.get('source_id') == 'nvml:field:186:scope:0:mW',
         'instant field186 power required; legacy average is not reusable')
    need(set(fleet.instances) == {s.instance_id for s in specs}, 'fleet inventory differs')
    need(sorted(g for s in specs for g in s.gpus) == sorted(gpus), 'fleet must cover lease once')
    need(len({(s.model, s.tp, s.pp, s.generation) for s in specs}) == 1,
         'one model/topology/generation required')
    from .native_runtime_topology import validate_topology
    validate_topology([asdict(s) for s in specs],dict(gpu_ids=gpus,gpu_uuids=list(gpu_uuids)))
    for spec in specs:
        need(spec.max_num_seqs == 32 and spec.pp == 1 and spec.kv_connector == 'P2pNcclConnector',
             'runtime collection requires native32 symmetric PD launch')
        instance = fleet[spec.instance_id]
        need(asdict(instance.spec) == asdict(spec) and instance.alive(), 'resident process/spec differs')
    observed = [meter.backend.gpu_uuid(g) for g in gpus]
    need(observed == list(gpu_uuids), 'physical UUIDs disagree with lease')
    return dict(gpu_ids=gpus, gpu_uuids=observed, gpu_uuid_binding_verified=True,
                checked_s=time.time(), power_source=dict(source))


def snapshot_sampler(sampler, gpu_uuids):
    # Power metadata is appended immediately before samples, frequency after.
    # Freeze a complete power prefix without changing the live sampler.
    count = min(len(sampler.samples), len(sampler.power_metadata))
    return dict(gpus=list(sampler.gpus), gpu_uuids=list(gpu_uuids),
                samples=list(sampler.samples[:count]), power_metadata=list(sampler.power_metadata[:count]),
                frequency_samples=list(sampler.frequency_samples),
                utilization_samples=list(sampler.utilization_samples),
                utilization_readings=list(sampler.utilization_readings),
                utilization_errors=list(sampler.utilization_errors),
                utilization_source=dict(sampler.utilization_source),
                power_source=dict(sampler.power_source), error=sampler.error,
                error_at_s=sampler.error_at_s, interval=sampler.interval)


class NativeRuntimeCollector:
    def __init__(self, specs, fleet, meter, sampler, out, *, gpu_uuids, frequency_domain_ref=None):
        self.specs = list(specs)
        self.runtime_plan=build_runtime_plan(frequency_domain_ref)
        self.domain=self.runtime_plan.get('frequency_domain')
        self.frequencies=tuple(self.runtime_plan['frequencies_mhz'])
        self.restore_frequency=max(self.frequencies)
        if self.domain:
            need(specs and all(Path(s.model).name==self.domain['model_id'] and (s.tp,s.pp)==(self.domain['tp'],self.domain['pp']) for s in specs),
                 'runtime frequency domain differs from actual resident model/topology')
        self.fleet, self.meter, self.sampler = fleet, meter, sampler
        self.out = Path(out)
        self.uuids = list(gpu_uuids)
        self.rows = []
        self.session = None
        self.capabilities = {}
        self.golden_ids = {}
        self.initial_specs = [asdict(s) for s in specs]

    def journal(self, row):
        row = json.loads(json.dumps(row, allow_nan=False))
        if 'repeat' in row:
            row['purpose']='training' if row['repeat']<REPEATS else 'holdout'
        if self.domain:row.update(frequency_domain_sha256=self.runtime_plan['frequency_domain_sha256'],
                                  runtime_plan_sha256=digest(self.runtime_plan))
        row['sequence'] = len(self.rows)
        self.rows.append(row)
        with (self.out/'journal.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        return row

    async def request(self, spec, endpoint, payload=None):
        return await call(self.session, spec.base_url, '/baseline/'+endpoint, payload)

    async def state(self, spec, *, drained=False):
        started = time.time()
        state = await self.request(spec, 'state')
        validate_state(state, generation=spec.generation, tp=spec.tp, pp=spec.pp,
                       drained=drained, observed_after_s=started)
        return state

    async def drain(self, spec):
        started = time.time()
        result = await self.request(spec, 'drain', dict(timeout_s=30))
        validate_state(result, generation=spec.generation, tp=spec.tp, pp=spec.pp,
                       drained=True, observed_after_s=started)
        need(result.get('acknowledged') is True and result.get('drained') is True,
             'native drain ACK missing')
        return result

    async def resume(self, spec):
        wanted = dict(generation=spec.generation, role='mixed', mode='temporal',
                      accepting=True, admit_prefill=True, admit_decode=True)
        ack = await self.request(spec, 'control', wanted)
        state = await self.state(spec, drained=True)
        need(ack.get('acknowledged') is True and ack.get('generation') == spec.generation
             and all(state.get(k) == v for k, v in wanted.items()), 'native epoch/resume differs')
        return dict(control=ack, state=state)

    async def stop_measurement(self,spec):
        receipt=await self.request(spec,'measurement/stop',{})
        ranks=receipt.get('ranks',[])
        need(len(ranks)==spec.tp*spec.pp and {r.get('rank') for r in ranks}==set(range(spec.tp*spec.pp))
             and all(r.get('acknowledged') is True for r in ranks),'measurement stop lacks all-rank ACK')
        return receipt

    async def capability(self, spec):
        cap = await self.request(spec, 'capability')
        physical = dict(zip(self.meter.gpus, self.uuids))
        need(cap.get('supported') is True and cap.get('model_id') == Path(spec.model).name
             and (cap.get('tp'), cap.get('pp')) == (spec.tp, spec.pp)
             and sorted(cap.get('gpu_uuids', [])) == sorted(physical[g] for g in spec.gpus),
             'native capability model/topology/UUID mismatch')
        for key, env in (('source_revision', 'PDBLEND_SOURCE_SHA256'), ('image_digest', 'PDBLEND_IMAGE_ID')):
            need(bool(os.environ.get(env)) and cap.get(key) == os.environ[env], 'native '+key+' differs')
        need(all(cap.get(k) for k in ('model_hash', 'tokenizer_hash', 'engine_revision')),
             'native implementation/model identity missing')
        if spec.instance_id in self.capabilities:
            previous = self.capabilities[spec.instance_id]
            keys = ('model_id','model_hash','tokenizer_hash','engine_revision','image_digest','source_revision','tp','pp','gpu_uuids')
            need(all(previous[k] == cap[k] for k in keys), 'restarted native identity changed')
        return cap

    async def operation(self, name, spec, repeat, function):
        row = dict(kind='operation', operation=name, instance_id=spec.instance_id,
                   gpus=list(spec.gpus), generation=spec.generation, repeat=repeat,
                   started_s=time.time(), status='failed')
        try:
            row['receipt'] = await function()
            row['status'] = 'passed'
            return row['receipt']
        except BaseException as exc:
            row['error'] = repr(exc)
            raise
        finally:
            row['finished_s'] = time.time()
            self.journal(row)

    async def clock(self, spec, frequency):
        ack = await self.request(spec, 'clock', dict(frequency_mhz=frequency))
        need(ack.get('acknowledged') is True and ack.get('success') is True
             and ack.get('requested_frequency_mhz') == frequency, 'clock command ACK differs')
        physical = dict(zip(self.meter.gpus,self.uuids))
        need(sorted(r.get('gpu_uuid') for r in ack.get('gpus',[]))==sorted(physical[g] for g in spec.gpus),
             'clock ACK physical devices differ')
        observations = []
        deadline = time.monotonic()+5
        while True:
            row = dict(at_s=time.time(), frequencies_mhz=[self.meter.current_freq(g) for g in spec.gpus])
            observations.append(row)
            if all(abs(value-frequency) <= 15 for value in row['frequencies_mhz']):
                return dict(ack=ack, observations=observations)
            if time.monotonic() >= deadline:
                raise RuntimeError('actual clock did not reach requested value: '+json.dumps(observations))
            await asyncio.sleep(.05)

    async def static_window(self, spec, state, repeat):
        before = None if state == 'off' else await self.state(spec, drained=True)
        off_before = await self.compute_empty(spec,timeout_s=1) if state == 'off' else None
        memory_settle_before = [self.meter.mem_freq(g) for g in spec.gpus]
        settle_start = time.time()
        await asyncio.sleep(SETTLE_S)
        # Audit the actual 5-second window, not the preceding settling period.
        memory_before = [self.meter.mem_freq(g) for g in spec.gpus]
        start = time.time()
        await asyncio.sleep(MEASURE_S)
        end = time.time()
        after = None if state == 'off' else await self.state(spec, drained=True)
        memory_after = [self.meter.mem_freq(g) for g in spec.gpus]
        # Off is checked at both ends, not inferred from low power.
        off = await self.compute_empty(spec, timeout_s=1) if state == 'off' else None
        self.journal(dict(kind='static', state=state, repeat=repeat, instance_id=spec.instance_id,
                          gpus=list(spec.gpus), generation=spec.generation,
                          settle_started_s=settle_start, started_s=start, finished_s=end,
                          before=before, after=after, memory_before_mhz=memory_before,
                          memory_after_mhz=memory_after,memory_settle_before_mhz=memory_settle_before,
                          off_before=off_before,off_evidence=off, status='measured'))

    async def compute_empty(self, spec, *, timeout_s=60):
        backend = self.meter.backend
        instance = self.fleet[spec.instance_id]
        need(not instance.alive() and instance.state == 'off', 'off instance process remains alive')
        lifecycle = [e for e in instance.events if e.get('kind') in ('start','stop')]
        need(lifecycle and lifecycle[-1]['kind'] == 'stop', 'owned process-stop evidence missing')
        physical = dict(zip(self.meter.gpus, self.uuids))
        started = time.time()
        deadline = time.monotonic()+timeout_s
        observations = []
        while True:
            rows = []
            for gpu in spec.gpus:
                uuid = backend.gpu_uuid(gpu)
                need(uuid == physical[gpu], 'off physical UUID changed')
                pids = [int(p.pid) for p in backend._nvml.nvmlDeviceGetComputeRunningProcesses(backend._handle(gpu))]
                rows.append(dict(gpu=gpu, uuid=uuid, compute_pids=pids))
            observations.append(dict(at_s=time.time(), devices=rows))
            if all(not row['compute_pids'] for row in rows):
                return dict(started_s=started, finished_s=time.time(), observations=observations,
                            owned_stop=dict(lifecycle[-1]), compute_processes_gone=True)
            if time.monotonic() >= deadline:
                raise RuntimeError('off retains compute processes: '+json.dumps(observations))
            await asyncio.sleep(.1)

    async def restart(self, spec):
        from pdblend_baselines.resident_campaign import model_load_lock
        instance = self.fleet[spec.instance_id]
        def load():
            with model_load_lock():
                instance.start()
                return instance.wait_ready(timeout_s=900)
        ready_s = await asyncio.to_thread(load)
        cap = await self.capability(spec)
        restored_clock=await self.clock(spec,self.restore_frequency) if self.domain else None
        resume = await self.resume(spec)
        ordinary = await self.ordinary_probe(spec)
        drain = await self.drain(spec)
        result=dict(ready_s=ready_s, capability=cap, resume=resume, ordinary=ordinary, drain=drain,
                    process_pid=instance.process.pid,
                    actual_starts=[dict(e) for e in instance.events if e.get('kind') == 'start'])
        if self.domain:result['clock']=restored_clock
        return result

    async def ordinary_probe(self, spec):
        from pdblend.bench.gates import random_prompt
        from pdblend.engine.client import EngineClient
        prompt = random_prompt(512,9701)
        async with EngineClient(spec.instance_id,spec.base_url) as client:
            result = await client.complete(prompt,16,'runtime-reuse-'+str(time.time_ns()),
                                           token_diagnostics=True,seed=9701)
        raw = asdict(result)
        self.journal(dict(kind='ordinary_golden',instance_id=spec.instance_id,prompt=prompt,result=raw))
        need(not result.error and result.stream_done and result.usage_received
             and result.prompt_tokens==512 and result.completion_tokens==16
             and isinstance(result.token_ids,list) and len(result.token_ids)==16,
             'post-wake ordinary exact-output evidence incomplete')
        if spec.instance_id in self.golden_ids:
            need(result.token_ids==self.golden_ids[spec.instance_id],'post-wake ordinary golden differs')
        else:
            self.golden_ids[spec.instance_id]=list(result.token_ids)
        return raw

    async def static_phases(self, spec):
        for repeat in range(TOTAL_REPEATS):
            for frequency in self.frequencies:
                await self.operation('clock_to_'+str(frequency), spec, repeat,
                                     lambda: self.clock(spec, frequency))
                await self.static_window(spec, 'active_idle@'+str(frequency), repeat)
            async def reset_clock():
                for gpu in spec.gpus:
                    self.meter.unpark(gpu)
                    self.meter.reset_clock(gpu)
                return dict(observed_mhz=[self.meter.current_freq(g) for g in spec.gpus])
            await self.operation('clock_reset', spec, repeat, reset_clock)
            await self.static_window(spec, 'active_idle_reset', repeat)
            async def park():
                drained = await self.drain(spec)
                for gpu in spec.gpus:
                    self.meter.park(gpu)
                return dict(drain=drained, memory_mhz=[self.meter.mem_freq(g) for g in spec.gpus],
                            observed_mhz=[self.meter.current_freq(g) for g in spec.gpus])
            await self.operation('park', spec, repeat, park)
            await self.static_window(spec, 'L1', repeat)
            async def unpark():
                for gpu in spec.gpus:
                    self.meter.unpark(gpu)
                clock = await self.clock(spec, self.restore_frequency)
                return dict(clock=clock, resume=await self.resume(spec), drain=await self.drain(spec))
            await self.operation('unpark', spec, repeat, unpark)
            # Explicit both directions, with real settle/measurement windows.
            for source, target in ((self.frequencies[1],self.frequencies[0]),(self.frequencies[0],self.frequencies[1])):
                await self.clock(spec, source)
                await asyncio.sleep(SETTLE_S)
                await self.operation('clock_'+str(source)+'_to_'+str(target), spec, repeat,
                                     lambda: self.clock(spec, target))
                await self.static_window(spec, 'clock_target@'+str(target), repeat)
            async def stop():
                drained = await self.drain(spec)
                await asyncio.to_thread(self.fleet[spec.instance_id].stop)
                return dict(drain=drained, off=await self.compute_empty(spec))
            await self.operation('off', spec, repeat, stop)
            await self.static_window(spec, 'off', repeat)
            await self.operation('wake', spec, repeat, lambda: self.restart(spec))

    async def transfers(self, prefill, decode):
        need(self.domain is None,
             'new frequency runtime has no qualified handoff; legacy signed contrast is diagnostic only')
        from pdblend.bench.gates import random_prompt
        from pdblend.engine.client import EngineClient, PDTransfer, pd_complete
        from .native_runtime_audit import validate_transfer_request
        transfer = PDTransfer(prefill.kv_connector, {s.instance_id:s.zmq_address for s in self.specs})
        async with EngineClient(prefill.instance_id, prefill.base_url) as pc, EngineClient(decode.instance_id, decode.base_url) as dc:
            for length in (512,2048,7168):
                for repeat in range(TOTAL_REPEATS):
                    before = {s.instance_id:await self.drain(s) for s in (prefill,decode)}
                    admission={}
                    for spec in (prefill,decode):
                        admission[spec.instance_id]=await self.resume(spec)
                        await self.clock(spec,2520)
                    seed=9701 if repeat<REPEATS else 9702
                    prompt = random_prompt(length, seed)
                    async def pair(tag):
                        ordinary = [await dc.complete(prompt, 16, tag+'-M'+str(r),token_diagnostics=True,seed=seed) for r in range(2)]
                        pre, combined = await pd_complete(transfer,pc,dc,prompt,16,tag+'-PD',token_diagnostics=True,seed=seed)
                        receipt = dict(ordinary=[asdict(x) for x in ordinary],prefill=asdict(pre),
                                       combined=None if combined is None else asdict(combined))
                        # Persist failed requests before validating them.
                        row=self.journal(dict(kind='transfer_request', tag=tag, input_tokens=length,repeat=repeat,
                            prompt=prompt,seed=seed,result=receipt,native_epoch=dict(
                                prefill_instance=prefill.instance_id,decode_instance=decode.instance_id,
                                before=before,admission=admission)))
                        return validate_transfer_request(row,specs={s.instance_id:asdict(s) for s in (prefill,decode)})
                    settle_started = time.time(); index = 0
                    while time.time()-settle_started < SETTLE_S:
                        await pair(f'transfer-{length}-{repeat}-settle-{index}');index += 1
                    start = time.time(); timings=[]
                    while time.time()-start < MEASURE_S:
                        timings.append(await pair(f'transfer-{length}-{repeat}-measure-{len(timings)}'))
                    end = time.time()
                    after = {s.instance_id:await self.drain(s) for s in (prefill,decode)}
                    self.journal(dict(kind='transfer',input_tokens=length,repeat=repeat,
                                      prefill_instance=prefill.instance_id,decode_instance=decode.instance_id,
                                      gpus=list(prefill.gpus)+list(decode.gpus),settle_started_s=settle_started,
                                      started_s=start,finished_s=end,before=before,after=after,
                                      timings=timings,status='measured',physical_copy_time=False))

    async def restore(self):
        errors, rows = [], []
        for spec in self.specs:
            row = dict(instance_id=spec.instance_id)
            try:
                need(asdict(self.fleet[spec.instance_id].spec) == asdict(spec), 'restore spec changed')
                for gpu in spec.gpus:
                    self.meter.unpark(gpu)
                    self.meter.set_clock(gpu,self.restore_frequency)
                if not self.fleet[spec.instance_id].alive():
                    row['restart'] = await self.restart(spec)
                # Failure while streaming must be cancelled before epoch restore.
                state = await self.request(spec,'state')
                pending=set(state['all_queue'])|set(state.get('retained_kv_requests',[]))|set(state.get('kv_allocations',{}))
                row['cancelled'] = [await self.request(spec,'cancel',dict(request_id=rid)) for rid in sorted(pending)]
                row['pre_drain'] = await self.drain(spec)
                row['measurement_stop'] = await self.stop_measurement(spec)
                row['resume'] = await self.resume(spec)
                row['clock'] = await self.clock(spec,self.restore_frequency)
                row['drain'] = await self.drain(spec)
                row['capability'] = await self.capability(spec)
                row['process_alive'] = self.fleet[spec.instance_id].alive()
                need(row['process_alive'], 'restored process is not alive')
            except BaseException as exc:
                row['error'] = repr(exc); errors.append(spec.instance_id+': '+repr(exc))
            rows.append(row)
        return dict(passed=not errors, errors=errors, instances=rows,
                    final_generation=self.specs[0].generation, clock_mhz=self.restore_frequency,
                    accepting=False, measurement_stopped=True if not errors else None,
                    actual_starts={s.instance_id:[dict(e) for e in self.fleet[s.instance_id].events if e.get('kind')=='start'] for s in self.specs})


async def collect_runtime(specs, fleet, meter, sampler, out, *, gpu_uuids, include_transfer=True, frequency_domain_ref=None):
    """Return a raw receipt; caller MUST require ready_for_timing before continuing.

    Capacity covers every replica. Static/off/clock use the first homogeneous
    instance; PD transfer uses the first two. Others stay loaded and measured.
    Three training plus one held-out off/wake cause four real reloads of the
    representative only; the other replicas remain resident throughout.
    """
    need(frequency_domain_ref is None or include_transfer is False,
         'new frequency runtime requires explicit include_transfer=False; handoff model remains blocked')
    runner = NativeRuntimeCollector(specs,fleet,meter,sampler,out,gpu_uuids=gpu_uuids,frequency_domain_ref=frequency_domain_ref)
    binding = validate_inventory(runner.specs,fleet,meter,sampler,gpu_uuids)
    runner.out.mkdir(parents=True,exist_ok=False)
    (runner.out/'journal.jsonl').touch(exist_ok=False)
    report = dict(schema=SCHEMA,status='failed',complete=False,formal_eligible=False,
                  energy_comparable=False,component_qualified=False,hardware_executed=True,lease=binding,
                  actual_launch=[dict(spec=asdict(s),argv=s.command()) for s in specs],
                  scope='representative_native_runtime_not_full_profile',repeats=REPEATS,
                  settle_s=SETTLE_S,measure_s=MEASURE_S,ready_for_timing=False,
                  runtime_plan=runner.runtime_plan,safe_restore_passed=False)
    report['runtime_plan_binding']=write_new(runner.out/'runtime-plan.json',runner.runtime_plan)
    if runner.domain:
        report.update({k:runner.runtime_plan[k] for k in ('model_id','tp','pp','frequency_domain_ref','frequency_domain','frequency_domain_sha256')})
        report.update(scope=runner.runtime_plan['scope'],handoff_prediction_qualified=False,full_profile_qualified=False)
    report['initial_starts']={s.instance_id:[dict(e) for e in fleet[s.instance_id].events if e.get('kind')=='start'] for s in specs}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        runner.session=session
        try:
            for spec in specs:
                cap=await runner.capability(spec);runner.capabilities[spec.instance_id]=cap
                drain=await runner.drain(spec)
                need(drain.get('max_num_seqs')==32 and type(drain.get('total_kv_tokens')) is int
                     and drain['total_kv_tokens']>0,'actual native32 capacity missing')
                stopped=await runner.stop_measurement(spec)
                runner.journal(dict(kind='capacity',instance_id=spec.instance_id,capability=cap,
                                    state=drain,measurement_stop=stopped))
            report['initial_capabilities']=runner.capabilities
            if runner.domain:
                keys=('model_id','model_hash','tokenizer_hash','engine_revision','source_revision','image_digest','tp','pp')
                first=runner.capabilities[specs[0].instance_id]
                need(all(all(cap.get(k)==first.get(k) for k in keys) for cap in runner.capabilities.values()),
                     'new domain runtime model/source identity differs across replicas')
                report['component_identity']=with_domain(dict(system='pdblend',**{k:first[k] for k in keys}),runner.domain)
            await runner.resume(specs[0])
            await runner.ordinary_probe(specs[0])
            await runner.ordinary_probe(specs[0])
            await runner.drain(specs[0])
            # Existing caller sampler must have acquired a real bracketing row.
            deadline=time.monotonic()+5
            while not sampler.samples or not sampler.frequency_samples:
                need(sampler.error is None and time.monotonic()<deadline,'caller sampler has no live samples')
                await asyncio.sleep(.05)
            await runner.static_phases(specs[0])
            if include_transfer:
                need(len(specs)>=2,'PD transfer requires two resident native replicas')
                await runner.transfers(specs[0],specs[1])
            report.update(status='measured',complete=True)
        except BaseException as exc:
            report['error']=repr(exc)
        finally:
            report['restoration']=await runner.restore()
            report['safe_restore_passed']=report['restoration']['passed']
            report['ready_for_timing']=report['safe_restore_passed'] and report['complete']
            if not report['restoration']['passed']:
                report.update(status='failed',complete=False)
            # Obtain a sample after the final operation; never stop/reset caller.
            end=time.time();deadline=time.monotonic()+5
            while sampler.error is None and time.monotonic()<deadline:
                if sampler.power_metadata and min(sampler.power_metadata[-1]['read_finished_s'])>=end:break
                await asyncio.sleep(.05)
            if sampler.error is not None:
                report.update(status='failed',complete=False,ready_for_timing=False,safe_restore_passed=False,sampler_error=sampler.error)
            elif (getattr(sampler,'_thread',None) is None or not sampler._thread.is_alive()
                    or not sampler.power_metadata or min(sampler.power_metadata[-1]['read_finished_s'])<end):
                report.update(status='failed',complete=False,ready_for_timing=False,safe_restore_passed=False,
                              sampler_error='caller sampler stopped or has no post-restoration power bracket')
            power=snapshot_sampler(sampler,gpu_uuids)
            if runner.domain:power.update(frequency_domain_sha256=runner.runtime_plan['frequency_domain_sha256'],
                                         runtime_plan_sha256=digest(runner.runtime_plan))
            report['power']=write_new(runner.out/'power.json',power)
            report['journal']=dict(path=str((runner.out/'journal.jsonl').resolve()),
                                   sha256=hashlib.sha256((runner.out/'journal.jsonl').read_bytes()).hexdigest())
            report['measured_components']=['capacity','static','clock','L1','off_wake']+(['transfer'] if include_transfer else [])
            report['remaining_gates']=['runtime_holdout_calibration_from_frozen_split','worker_identity_compatibility','full_profile_composition']
            report['necessary_mechanism_loads']=sum(len(v) for v in report['restoration']['actual_starts'].values())-sum(len(v) for v in report['initial_starts'].values())
            from .native_runtime_audit import replay_runtime
            audit=replay_runtime(report)
            report['audit']=write_new(runner.out/'audit.json',audit)
            report['runtime_measurements_valid']=audit['raw_components_complete']
            report['runtime_holdout_passed']=audit.get('holdout_passed',False)
            if not audit['raw_components_complete']:
                if report['complete']:
                    report['status']='measured_with_evidence_gaps'
            else:
                report['status']='passed'
            write_new(runner.out/'completion.json',report)
    return report
