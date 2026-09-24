"""Native resident adapter for immutable comparison groups on an eight-GPU lease."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import fields, replace
import json
import math
import os
import signal
from pathlib import Path
import time

import aiohttp

from .comparison_campaign import binding, load_bound, PROTOCOL
from .comparison_metrics import canonical_outcomes, reduce_comparison
from .comparison_metering import ComparisonMeteringSession, summarize_comparison
from .comparison_journal import iter_comparison_journal
from .independent_dispatch import Resources, execute as dispatch, request_rows, validate
from .metering import Gpus
from .resident_session import ResidentGroupSession, file_sha, write_new, digest
from pdblend.engine.launcher import Fleet
from pdblend.model_registry import ModelRegistry
from pdblend.results.journal import iter_journal
from pdblend.results.power_archive import write_power_archive
from pdblend_runtime.probe import NativeSpec, call, generate
from pdblend_runtime.cleanup import cleanup_owned
from pdblend_baselines.resident_campaign import verify_endpoints, drain_endpoints, warmup_endpoints, model_load_lock


def pdblend_window_resources(point, specs):
    """Load a qualified model and an explicitly bound homogeneous offline plan.

    The current dispatcher takes a single Plan. Heterogeneous pools require a
    separate per-pool deployment receipt and are refused here, not flattened.
    The native generation is a runtime epoch; all other selected plan fields
    must agree with the immutable deployment and profile.
    """
    from pdblend.profile.query.versions import load_profile
    from pdblend.profile.query.runtime import require_planner_components
    from pdblend.planner.pool import Plan, ACTIVE, PARKED

    from .comparison_pdblend_observation import observation_requested, validate_observation_inputs
    observation = observation_requested(point)
    if observation:
        validate_observation_inputs(point, point['inputs'])
    if point.get('system') != 'pdblend' or not specs:
        raise ValueError('PDblend requires its own nonempty resident deployment')
    topologies = {(s.tp, s.pp) for s in specs}
    generations = {s.generation for s in specs}
    if len(topologies) != 1 or len(generations) != 1:
        raise ValueError('unsupported PDblend heterogeneous plan or inconsistent native generation')
    tp, pp = next(iter(topologies))
    inputs = point['inputs']
    cfg = load_bound(inputs['system_config'])
    choice = load_bound(inputs['offline_choice'])
    for document in (cfg, choice):
        if (document.get('system') != 'pdblend' or document.get('model_id') != point['model_id']):
            raise ValueError('PDblend configuration/offline choice identity differs')
    if (choice.get('selection_split') not in ('calibration', 'tuning')
            or choice.get('evaluation_used_for_selection') is not False):
        raise ValueError('offline choice must use calibration/tuning only')
    profile_ref = cfg.get('profile')
    if isinstance(profile_ref, dict):
        load_bound(profile_ref)
        profile_path = Path(profile_ref['path']).resolve()
    elif isinstance(profile_ref, str):
        profile_path = (Path(inputs['system_config']['path']).parent / profile_ref).resolve()
    else:
        raise ValueError('PDblend configuration lacks an explicit profile selection')
    matches = [ref for ref in inputs.get('profiles', []) if Path(ref['path']).resolve() == profile_path]
    if len(matches) != 1:
        raise ValueError('configured PDblend profile is not uniquely hash-bound')
    load_bound(matches[0])
    if choice.get('profile_sha256') != matches[0]['sha256']:
        raise ValueError('offline choice is not bound to the selected profile')
    values = choice.get('plan')
    if not isinstance(values, dict) or not {'tp', 'pp'} <= values.keys():
        raise ValueError('offline choice requires an explicit Plan with TP/PP identity')
    if set(values) - {f.name for f in fields(Plan)}:
        raise ValueError('unknown offline Plan fields')
    if (values['tp'], values['pp']) != (tp, pp):
        raise ValueError('offline Plan topology differs from resident engines')
    try:
        plan = Plan(**values)
    except TypeError as exc:
        raise ValueError('incomplete offline Plan schema') from exc
    if (not isinstance(plan.counts, dict) or set(plan.counts) - set(ACTIVE + PARKED)
            or any(type(v) is not int or v < 0 for v in plan.counts.values())
            or sum(plan.counts.values()) != len(specs) or plan.active() <= 0
            or type(plan.tau) is not int or plan.tau < 0):
        raise ValueError('offline Plan does not cover the actual resident inventory')
    for name in ('power_w', 'ttft_s', 'tpot_s'):
        value = getattr(plan, name)
        if type(value) not in (int, float) or math.isnan(value) or value < 0 or (not observation and not math.isfinite(value)):
            raise ValueError('invalid offline Plan estimate: ' + name)
    loaded = load_profile(profile_path, system='pdblend', model_id=point['model_id'],
                          tp=tp, pp=pp, usage='development' if observation else 'formal')
    for role in ACTIVE:
        frequency = getattr(plan, 'f_' + role)
        if type(frequency) is not int or frequency not in loaded.model.freqs:
            raise ValueError('offline Plan frequency lies outside selected profile')
    require_planner_components(loaded.model, allow_pd=True, allow_dvfs=True)
    profile_key = json.dumps(loaded.profile_key, sort_keys=True, separators=(',', ':'))
    if plan.profile_key and plan.profile_key != profile_key:
        raise ValueError('offline Plan profile key differs from selected profile')
    plan = replace(plan, generation=next(iter(generations)), profile_key=profile_key)
    return loaded, plan


def read_native_measurement(system, out, raw):
    """Find native outcome formats and the actual service clock, never startup."""
    out = Path(out)
    journal = next((out / name for name in ('events.jsonl.gz', 'events.jsonl') if (out / name).exists()), None)
    raw_outcomes = next((out / name for name in ('outcomes.jsonl.gz', 'outcomes.jsonl', 'outcomes.json')
                         if (out / name).exists()), None)
    if raw_outcomes is None:
        outcomes = raw.get('outcomes', [])
    elif raw_outcomes.suffix == '.json':
        outcomes = json.loads(raw_outcomes.read_text())
        if isinstance(outcomes, dict):
            outcomes = outcomes['outcomes']
    else:
        outcomes = list(iter_journal(raw_outcomes))
    if not isinstance(outcomes, list):
        raise ValueError('native outcomes must be an explicit list')
    origins = ('service_started_s', 'service_window_started_s', 'service_start_s')
    if system == 'mixed':
        origins += ('started_s',)
    started = next((raw[k] for k in origins if raw.get(k) is not None), None)
    if started is None and system == 'dynamollm' and journal is not None:
        starts = [row['at_s'] for row in iter_journal(journal)
                  if row.get('event', row.get('kind')) == 'dynamo_service_window_start']
        if len(starts) == 1:
            started = starts[0]
    if type(started) not in (int, float) or not math.isfinite(started):
        raise ValueError('native runner did not bind a unique client service origin')
    return started, outcomes, journal


class NativeResidentAdapter:
    def __init__(self, out, *, base_port):
        self.out, self.base_port = Path(out), base_port
        self.fleet = self.gpus = self.monitor = None
        self.dynamo_session = None
        self.started_s = None
        self.load_count = 0
        self.ecoserve_inputs = {}
        self.distserve_inputs = {}
        self.isolated_metering = False
        self.metering_cleanup_errors = []

    @staticmethod
    def isolated_metering_requested(group):
        identity_mode = group['engine_identity'].get('metering_execution')
        modes = [p.get('metering_execution') for p in group['points']]
        if 'isolated_process' not in [identity_mode, *modes]:
            return False
        if (identity_mode != 'isolated_process' or not modes
                or len({p['system'] for p in group['points']}) != 1
                or any(p['system'] not in ('ecoserve','distserve')
                       or p.get('qualification_mode') != p['system']+'_native_bootstrap'
                       or p.get('metering_execution') != 'isolated_process'
                       for p in group['points'])):
            raise ValueError('isolated metering requires a matching explicit qualified single-system point/group identity')
        return True

    async def start(self, group):
        self.group = group
        self.identity = group['engine_identity']
        self.isolated_metering = self.isolated_metering_requested(group)
        expected = self.identity['fleet_gpu_uuids']
        actual = os.environ.get('PDBLEND_GPU_UUIDS', '').split(',')
        if actual != expected or len(set(actual)) != 8:
            raise ValueError('exclusive lease UUIDs differ from frozen group')
        for point in group['points']:
            trace = load_bound(point['trace'])
            if (trace['model_id'] != point['model_id'] or trace['dataset'] != point['dataset']
                    or trace['slo'] != point['slo'] or trace['seed'] != 701
                    or trace['rate_rps'] != point['rate_rps'] or trace['duration_s'] != 150
                    or any(not 0 <= r['arrival_s'] < 150 or len(r['prompt']) + r['max_tokens'] > 8192
                           for r in trace['requests'])):
                raise ValueError('point trace identity or context domain differs')
            if point.get('qualification_mode') == 'distserve_native_bootstrap':
                from .comparison_distserve_inputs import validate_distserve_inputs
                qualified = validate_distserve_inputs(point, self.identity, source_manifest=binding(
                    Path(os.environ['PDBLEND_SOURCE_MANIFEST'])), replay_search=True)
                if not qualified['preflight_ready']:
                    raise ValueError('DistServe input qualification failed: ' + json.dumps(qualified.get('gate_failures', qualified)))
                self.distserve_inputs[point['name']] = qualified
            elif point.get('qualification_mode') == 'ecoserve_native_bootstrap':
                if point['model_id'] == 'Qwen2.5-32B-Instruct':
                    from .comparison_ecoserve32_inputs import validate_ecoserve_inputs
                else:
                    from .comparison_ecoserve_inputs import validate_ecoserve_inputs
                qualified = validate_ecoserve_inputs(point, self.identity, source_manifest=binding(
                    Path(os.environ['PDBLEND_SOURCE_MANIFEST'])))
                if not qualified['preflight_ready']:
                    raise ValueError('EcoServe input qualification failed: ' + json.dumps(qualified['gate_failures']))
                self.ecoserve_inputs[point['name']] = qualified
            elif point.get('observation_scope') == 'baseline_profile_unqualified_evaluation/v1':
                from .comparison_baseline_observation import validate_observation_inputs
                if point['system'] != 'distserve':
                    raise ValueError('Dynamo observations require their independent resident adapter')
                validate_observation_inputs(point, self.identity)
            elif point.get('observation_scope') == 'pdblend_profile_unqualified_evaluation/v1':
                from .comparison_pdblend_observation import validate_observation_inputs
                validate_observation_inputs(point, point['inputs'])
            elif point.get('qualification_mode') != 'mixed_native_bootstrap':
                if not validate(point, point['inputs'])['formal_eligible']:
                    raise ValueError('independent qualification gates remain closed')
            elif point['system'] != 'mixed':
                raise ValueError('Mixed bootstrap cannot qualify another policy')
        source_path = Path(os.environ['PDBLEND_SOURCE_MANIFEST'])
        source = json.loads(source_path.read_text())
        for rel, sha in source['files'].items():
            if file_sha(source_path.parent / rel) != sha:
                raise ValueError('immutable execution source changed: ' + rel)
        registry = ModelRegistry(os.environ['PDBLEND_MODELS_DIR'],
                                 verification_receipt=os.environ['PDBLEND_MODEL_VERIFICATION_RECEIPT'])
        model = registry.get(group['model_id']); model.validate_config()
        if model.model_hash != self.identity['model_hash'] or model.tokenizer_hash != self.identity['tokenizer_hash']:
            raise ValueError('model/tokenizer identity differs')
        if os.environ['PDBLEND_IMAGE_ID'] != self.identity['image_digest']:
            raise ValueError('engine image differs')
        self.specs = []
        for index, row in enumerate(self.identity['instances']):
            options = dict(row['launch_options'])
            options['extra_args'] = tuple(options.get('extra_args', []))
            self.specs.append(NativeSpec(row['instance_id'], tuple(actual.index(x) for x in row['gpu_uuids']),
                self.base_port + index * 4, str(Path(os.environ['PDBLEND_MODELS_DIR']) / group['model_id']),
                tp=row['tp'], pp=row['pp'], **options))
        for key, expected_value in self.identity['environment'].items():
            if os.environ.get(key) != expected_value:
                raise ValueError('launch environment differs: ' + key)
        runtime_files = {k: v for k, v in source['files'].items()
                         if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))}
        measurement_files = {k: v for k, v in source['files'].items()
            if k.startswith('pdblend/measure/') or k in (
                'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py',
                'pdblend/bench/client.py')}
        if (digest(runtime_files) != self.identity['runtime_source_sha256']
                or digest(measurement_files) != self.identity['measurement_source_sha256']):
            raise ValueError('public runtime/measurement source fingerprint differs')
        actual_launch = dict(image_digest=os.environ['PDBLEND_IMAGE_ID'],
            model_hash=model.model_hash, tokenizer_hash=model.tokenizer_hash,
            source_revision=os.environ['PDBLEND_SOURCE_SHA256'],
            runtime_source_sha256=digest(runtime_files), measurement_source_sha256=digest(measurement_files),
            instances=[dict(instance_id=s.instance_id, argv=s.command(),
                environment={k:s.environment().get(k) for k in
                             (*self.identity['environment'], 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                              'VLLM_SERVER_DEV_MODE')},
                gpu_uuids=[actual[i] for i in s.gpus], tp=s.tp, pp=s.pp,
                launch_options=self.identity['instances'][index]['launch_options'])
                for index,s in enumerate(self.specs)])
        lease_path = Path(os.environ['PDBLEND_CONCURRENCY_ENVIRONMENT'])
        for name, path in [('concurrency-environment', lease_path),
                           ('lease-manifest', lease_path.parent/'manifest.json')]:
            with (self.out/(name+'.json')).open('xb') as stream:
                stream.write(path.read_bytes())
        self.gpus = Gpus(range(8), power_mode='instant')
        if self.isolated_metering:
            from .isolated_comparison_meter import IsolatedComparisonMeter
            self.monitor = IsolatedComparisonMeter(range(8), actual)
            self.monitor.start()
            # Exercise the actual child/NVML path only under this new lease,
            # before loading any model and outside all evaluation guards.
            await asyncio.sleep(2.3)
            startup_snapshot = self.monitor.snapshot()
            startup_method = self.monitor.method_receipt()
            write_power_archive(self.out/'metering-startup-power.json', startup_snapshot)
            write_new(self.out/'metering-method-startup.json', startup_method)
            from .comparison_meter_preflight import qualify_startup_snapshot
            preflight = qualify_startup_snapshot(startup_snapshot, startup_method, actual)
            preflight.update(raw_power=binding(self.out/'metering-startup-power.json'),
                method=binding(self.out/'metering-method-startup.json'))
            write_new(self.out/'metering-startup-preflight.json', preflight)
        else:
            self.monitor = ComparisonMeteringSession(range(8), actual).start()
        self.started_s = time.time()
        self.fleet = Fleet(self.specs, self.out / 'logs')
        started = time.time()
        with model_load_lock():
            acquired = time.time()
            for spec in self.specs:
                self.fleet[spec.instance_id].start()
                self.load_count += 1
                self.fleet[spec.instance_id].wait_ready(timeout_s=900)
        finished = time.time()
        caps = await verify_endpoints(self.specs)
        # Deterministic ordinary generation on every rank group, outside all
        # evaluation windows, also proves the exact output budget is preserved.
        reference = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            for spec in self.specs:
                pair = []
                for repeat in range(2):
                    pair.append(await generate(session, spec.base_url, dict(
                        request_id=f'qualification-{spec.instance_id}-{repeat}',
                        prompt=list(range(100, 228)), max_tokens=16, seed=9701,
                        temperature=0, ignore_eos=True)))
                if len(pair[0]['token_ids']) != 16 or pair[0]['token_ids'] != pair[1]['token_ids']:
                    raise RuntimeError('same-engine deterministic ordinary output differs')
                reference.append(dict(instance_id=spec.instance_id, responses=pair))
        distserve_pair_probes = None
        if ({p['system'] for p in group['points']} == {'distserve'}
                and not all(p.get('observation_scope') == 'baseline_profile_unqualified_evaluation/v1'
                            for p in group['points'])):
            from .comparison_distserve_acceptance import qualify_pairs
            distserve_pair_probes = await qualify_pairs(self.specs, self.out/'distserve-pair-probes')
        drain = await drain_endpoints(self.specs)
        self.qualification = dict(scope='native_resident_startup', source_manifest=binding(source_path),
            model_hash=model.model_hash, tokenizer_hash=model.tokenizer_hash, capabilities=caps,
            ordinary_reference=reference, drain=drain, exclusive_gpu_uuids=actual,
            engine_signature=group['engine_signature'], actual_launch_identity=actual_launch,
            source_fingerprints=dict(runtime_files=runtime_files, measurement_files=measurement_files),
            concurrency_environment=binding(self.out/'concurrency-environment.json'),
            lease_manifest=binding(self.out/'lease-manifest.json'))
        if distserve_pair_probes is not None:
            self.qualification['distserve_pair_probes'] = distserve_pair_probes
        if self.isolated_metering:
            self.qualification.update(metering_execution='isolated_process',
                metering_method_startup=binding(self.out/'metering-method-startup.json'),
                metering_startup_preflight=binding(self.out/'metering-startup-preflight.json'))
        write_new(self.out / 'qualification.json', self.qualification)
        return dict(engine_loads=self.load_count, engine_load_cycles=1,
                    load_lock_wait_s=acquired-started, engine_load_s=finished-acquired,
                    qualification=binding(self.out / 'qualification.json'))

    async def reset(self, point):
        started = time.time()
        pd_inventory = None
        if point['system'] == 'pdblend':
            if not hasattr(self, 'pdblend_boundary'):
                from .comparison_pdblend_lifecycle import PDblendResidentBoundary
                self.pdblend_boundary = PDblendResidentBoundary(self)
            pd_inventory = await self.pdblend_boundary.restore()
        await drain_endpoints(self.specs)
        # All native generations agree before a new P/D communication epoch.
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            states = [await call(session, s.base_url, '/baseline/state') for s in self.specs]
            generation = max(s['generation'] for s in states) + 1
            for spec in self.specs:
                for gpu in spec.gpus:
                    self.gpus.unpark(gpu); self.gpus.set_clock(gpu, 2520)
                reply = await call(session, spec.base_url, '/baseline/control', dict(
                    generation=generation, role='mixed', mode='temporal', accepting=True,
                    admit_prefill=True, admit_decode=True))
                if reply.get('acknowledged') is not True or reply.get('generation') != generation:
                    raise RuntimeError('initial admission/generation reset failed')
        self.specs = [replace(s, generation=generation) for s in self.specs]
        for spec in self.specs:
            self.fleet[spec.instance_id].spec = spec
        warmup_started = time.time()
        warmup = await warmup_endpoints(self.specs, point['name'])
        drained = await drain_endpoints(self.specs)
        warmup_s = time.time() - warmup_started
        # warmup advances generation once; pass the actual value to PDblend.
        by_id = {row['instance_id']: row['state']['generation'] for row in drained}
        reopen_ack, reopen_state = {}, {}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            next_generation = max(by_id.values()) + 1
            for spec in self.specs:
                ack = await call(session, spec.base_url, '/baseline/control', dict(
                    generation=next_generation, role='mixed', mode='temporal', accepting=True,
                    admit_prefill=True, admit_decode=True))
                state = await call(session, spec.base_url, '/baseline/state')
                if (ack.get('acknowledged') is not True or state.get('accepting') is not True
                        or state.get('generation') != next_generation):
                    raise RuntimeError('post-warmup admission reopen failed')
                reopen_ack[spec.instance_id] = ack
                reopen_state[spec.instance_id] = state
                by_id[spec.instance_id] = next_generation
        self.specs = [replace(s, generation=by_id[s.instance_id]) for s in self.specs]
        for spec in self.specs:
            self.fleet[spec.instance_id].spec = spec
        self.reset_receipt = dict(passed=True, warmup=warmup, warmup_s=warmup_s,
                    drain=drained, reset_s=time.time()-started,
                    initial_clock_mhz=2520, generation=by_id,
                    reopen_ack=reopen_ack, reopen_state=reopen_state)
        if pd_inventory is not None:
            self.reset_receipt['pdblend_inventory_reset'] = pd_inventory
        return self.reset_receipt

    async def execute(self, point, out):
        if not getattr(self, 'isolated_metering', False):
            return await self._execute_window(point, out)
        if point['system'] != 'ecoserve' or point.get('metering_execution') != 'isolated_process':
            raise ValueError('window metering method differs from explicit Eco session identity')
        self.monitor.begin_window()
        try:
            return await self._execute_window(point, out)
        except BaseException:
            # Preserve the native error; reap this child's sampler before any
            # session reuse. No RPC occurs until the service/tail guard closes.
            if self.monitor.method_receipt()['window_guard_active']:
                self.monitor.end_window()
            try:
                self.monitor.stop(after_s=time.time())
            except BaseException as exc:
                self.metering_cleanup_errors.append('failed-window meter stop: '+repr(exc))
            try:
                write_new(Path(out)/'metering-method-failure.json', self.monitor.method_receipt())
            except BaseException as exc:
                self.metering_cleanup_errors.append('failed-window method evidence: '+repr(exc))
            raise

    async def _execute_window(self, point, out):
        trace = load_bound(point['trace'])
        out = Path(out)
        if point.get('observation_scope') == 'baseline_profile_unqualified_evaluation/v1':
            from .comparison_baseline_observation import execute_native_observation
            if point['system'] != 'distserve':
                raise ValueError('Dynamo observations require their independent resident adapter')
            raw = await execute_native_observation(point, self.identity, out=out, specs=self.specs)
        elif point.get('qualification_mode') == 'mixed_native_bootstrap':
            from pdblend_baselines.mixed.run_native import execute
            if any(s.kv_connector is not None or s.max_num_seqs != 32 for s in self.specs):
                raise ValueError('independent Mixed engine parameters substituted')
            raw = await execute(self.specs, request_rows(trace), out, duration_s=150.,
                                slo=(point['slo']['ttft_s'], point['slo']['tpot_s']), seed=701)
        elif point.get('qualification_mode') == 'distserve_native_bootstrap':
            from pdblend_baselines.distserve.deployment import execute_on_resident
            raw = await execute_on_resident(self.distserve_inputs[point['name']]['choice'],
                self.specs, Path(point['trace']['path']), out, 150., request_timeout=240.)
        elif point.get('qualification_mode') == 'ecoserve_native_bootstrap':
            from pdblend_baselines.ecoserve.run_native import execute
            raw = await execute(self.ecoserve_inputs[point['name']]['config'],
                                {s.instance_id: s.base_url for s in self.specs},
                                Path(point['trace']['path']), out, 150.)
        else:
            resources = Resources(self.specs, self.fleet, self.gpus, proxy_port=self.base_port+80,
                                  comparison_record_tokens=True, dynamo_session=self.dynamo_session)
            if point['system'] == 'pdblend':
                loaded, resources.pd_plan = pdblend_window_resources(point, self.specs)
                resources.pd_model = loaded.model
                write_new(out.parent / 'pdblend-profile-selection.json', loaded.manifest_fields())
                self.pdblend_boundary.begin_window()
            raw = await dispatch(point, point['inputs'], resources, out)
        if point['system'] == 'pdblend':
            self.last_drain = await self.pdblend_boundary.finish_window(raw)
            tail_end = self.last_drain['tail_end_s']
        else:
            drained = await drain_endpoints(self.specs)
            tail_end = time.time()
            self.last_drain = dict(passed=True, states=drained, tail_end_s=tail_end)
        if getattr(self, 'isolated_metering', False):
            self.monitor.end_window()
        if point.get('observation_scope') == 'baseline_profile_unqualified_evaluation/v1':
            from .comparison_baseline_observation import finalize_observation
            await asyncio.sleep(.3)
            return finalize_observation(point, self.identity, out=out, native_result=raw,
                snapshot=self.monitor.snapshot(), tail_end_s=tail_end,
                startup=self.qualification, reset=self.reset_receipt, drain=self.last_drain)
        write_new(out / 'native-result.json', raw)
        if point['system'] == 'ecoserve':
            # Audit the frozen JSON representation, including integer GPU keys.
            raw = json.loads((out / 'native-result.json').read_text())
        started, outcomes, journal = read_native_measurement(point['system'], out, raw)
        rows = canonical_outcomes(point['system'], trace, outcomes, service_started_s=started,
                                  journal=iter_comparison_journal(journal) if journal is not None else None)
        metrics = reduce_comparison(trace, rows, service_started_s=started, duration_s=150.,
                                     slo=(point['slo']['ttft_s'], point['slo']['tpot_s']))
        if tail_end < started + 150.:
            raise RuntimeError('runner returned before frozen service window ended')
        # Sampling continues through reset and all other windows. Wait only for
        # a bracketing read, then bind this window's immutable raw snapshot.
        await asyncio.sleep(.3)
        power_snapshot = self.monitor.snapshot()
        metering = summarize_comparison(power_snapshot, gpu_uuids=self.identity['fleet_gpu_uuids'],
                                       origin_s=started, tail_end_s=tail_end, duration_s=150.)
        write_new(out / 'comparison-metering.json', metering)
        write_power_archive(out / 'power.json', power_snapshot)
        if getattr(self, 'isolated_metering', False):
            write_new(out/'metering-method.json', self.monitor.method_receipt())
        write_new(out / 'comparison-requests.json', metrics['request_metrics'])
        write_new(out / 'native-drain.json', self.last_drain)
        if point.get('qualification_mode') == 'distserve_native_bootstrap':
            from .comparison_distserve_acceptance import audit_distserve_window as audit_window
        elif point.get('qualification_mode') == 'ecoserve_native_bootstrap':
            if point['model_id'] == 'Qwen2.5-32B-Instruct':
                from .comparison_ecoserve32_acceptance import audit_ecoserve_window as audit_window
            else:
                from .comparison_ecoserve_acceptance import audit_ecoserve_window as audit_window
        elif point.get('observation_scope') == 'pdblend_profile_unqualified_evaluation/v1':
            from .comparison_pdblend_observation import audit_observation_window as audit_window
        elif point['system'] == 'pdblend':
            from .comparison_pdblend_acceptance import audit_pdblend_window as audit_window
        else:
            from .comparison_acceptance import audit_window
        raw_outcomes = next((out/name for name in ('outcomes.jsonl.gz', 'outcomes.jsonl', 'outcomes.json')
                             if (out/name).is_file()), None)
        raw_paths = dict(trace=Path(point['trace']['path']), requests=out/'requests.json',
            outcomes=raw_outcomes, events=journal, power=out/'power.json',
            native_result=out/'native-result.json', canonical_requests=out/'comparison-requests.json',
            metering=out/'comparison-metering.json', startup_qualification=self.out/'qualification.json',
            reset=out.parent/'reset.json', drain=out/'native-drain.json',
            concurrency_environment=self.out/'concurrency-environment.json',
            lease_manifest=self.out/'lease-manifest.json')
        if point['system'] == 'pdblend':
            raw_paths.update(controller=out/'controller.jsonl', routes=out/'routes.jsonl',
                native_cleanup=out/'native-cleanup.json', transition_measurements=out/'transition-measurements.json',
                frequencies=out/'freq.jsonl')
        if getattr(self, 'isolated_metering', False):
            raw_paths['metering_method'] = out/'metering-method.json'
        raw_refs = {name:binding(path) for name,path in raw_paths.items() if path and path.is_file()}
        raw_refs['trace'] = point['trace']
        observation = point.get('observation_scope') == 'pdblend_profile_unqualified_evaluation/v1'
        if observation:
            # Startup separately binds these session-level artifacts.
            raw_refs = {k:v for k,v in raw_refs.items() if k not in ('concurrency_environment','lease_manifest')}
        acceptance = audit_window(point, self.identity, self.qualification, self.reset_receipt,
                                  raw, metrics, metering, self.last_drain, raw_refs)
        if not observation:
            write_new(out / 'acceptance.json', acceptance)
        metrics.pop('request_metrics', None)
        metrics.update({k: v for k, v in metering.items() if not isinstance(v, (list, dict))})
        metrics['tail_s'] = tail_end-started-150.
        metrics['gpu_util_coverage_fraction'] = metering['util_coverage_fraction']
        metrics['measurement_protocol_version'] = PROTOCOL
        for index, uuid in enumerate(self.identity['fleet_gpu_uuids']):
            gpu = metering['service']['utilization']['per_gpu'][uuid]
            metrics[f'gpu{index}_util_mean_pct'] = gpu.get('mean_pct')
            metrics[f'gpu{index}_util_peak_pct'] = gpu.get('peak_pct')
        for index, gpu in enumerate(self.identity['fleet_gpu_uuids']):
            metrics[f'gpu{index}_uuid'] = gpu
        extras = {}
        if observation:
            acceptance['metrics_sha256'] = digest(metrics)
            write_new(out / 'observation-acceptance.json', acceptance)
            write_new(out / 'acceptance.json', acceptance)
            extras = dict(observation_scope=point['observation_scope'],
                observation_acceptance=acceptance, profile_qualified=False,
                measurement_evidence_valid=acceptance['measurement_evidence_valid'])
        verified = acceptance['evidence_valid']
        return dict(**extras,evidence_valid=verified, formal_eligible=acceptance['formal_eligible'],
            missing_gates=acceptance['missing_gates'], acceptance=acceptance,
            qualification=binding(self.out/'qualification.json'),
            metrics=metrics, identity=dict(model_hash=self.identity['model_hash'],
                tokenizer_hash=self.identity['tokenizer_hash'], image_digest=self.identity['image_digest'],
                runtime_source_sha256=self.identity['runtime_source_sha256'],
                measurement_source_sha256=self.identity['measurement_source_sha256'],
                source_sha256=os.environ['PDBLEND_SOURCE_SHA256'], gpu_uuids=self.identity['fleet_gpu_uuids'],
                measurement_protocol_version=PROTOCOL))

    async def drain(self, point):
        return self.last_drain

    async def close(self):
        errors = list(getattr(self, 'metering_cleanup_errors', []))
        metering_cleanup = None
        if hasattr(self, 'pdblend_boundary'):
            try:
                accounting = self.pdblend_boundary.refresh_load_count()
                write_new(self.out/'pdblend-engine-loads.json', accounting)
            except Exception as exc:
                errors.append('PDblend actual start accounting: '+str(exc))
        if self.fleet is not None:
            for instance in self.fleet.instances.values():
                process_group = instance.process.pid if instance.process is not None else None
                try:
                    instance.stop()
                except Exception as exc:
                    errors.append(str(exc))
                finally:
                    if process_group is not None:
                        try:
                            os.killpg(process_group, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except Exception as exc:
                            errors.append(str(exc))
        if self.gpus is not None:
            for gpu in range(8):
                try:
                    self.gpus.unpark(gpu)
                    self.gpus.reset_clock(gpu)
                except Exception as exc:
                    errors.append(str(exc))
        if self.monitor is not None:
            if getattr(self, 'isolated_metering', False):
                try:
                    if self.monitor.method_receipt()['window_guard_active']:
                        self.monitor.end_window()
                        errors.append('meter guard remained active at session cleanup')
                    self.monitor.stop(after_s=time.time())
                    write_power_archive(self.out/'session-power.json', self.monitor.snapshot())
                except BaseException as exc:
                    errors.append('isolated meter cleanup: '+repr(exc))
                finally:
                    method = self.monitor.method_receipt()
                    write_new(self.out/'metering-method-cleanup.json', method)
                    metering_cleanup = binding(self.out/'metering-method-cleanup.json')
                    if method['child_alive'] or method['status'] != 'stopped' or method.get('child_exitcode') != 0:
                        errors.append('isolated meter child lacks clean exit evidence')
            else:
                self.monitor.stop(after_s=time.time())
                write_power_archive(self.out / 'session-power.json', self.monitor.snapshot())
        processes_gone = all(not i.alive() for i in self.fleet.instances.values()) if self.fleet else True
        if self.gpus is not None:
            try:
                deadline = time.monotonic() + 30.
                while True:
                    remaining = {str(g): [p.pid for p in self.gpus.backend._nvml.nvmlDeviceGetComputeRunningProcesses(
                        self.gpus.backend._handle(g))] for g in range(8)}
                    if not any(remaining.values()) or time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(.25)
                processes_gone = processes_gone and not any(remaining.values())
            except Exception as exc:
                errors.append('physical process check: ' + str(exc))
                processes_gone = False
        result = dict(passed=not errors and processes_gone, errors=errors, engine_loads=self.load_count,
                      process_cleanup_verified=processes_gone)
        if metering_cleanup is not None:
            result.update(metering_execution='isolated_process', metering_method_cleanup=metering_cleanup)
        return result


def make_resident_adapter(group, out, *, base_port):
    """Select an explicit baseline observation wrapper without changing defaults."""
    from .comparison_dynamo_runtime import DynamoResidentAdapter, is_dynamo_group
    dynamo = is_dynamo_group(group)
    scope = 'baseline_profile_unqualified_evaluation/v1'
    points = group.get('points', [])
    observational = [p for p in points if p.get('observation_scope') == scope]
    if observational:
        systems = {p.get('system') for p in points}
        if (len(observational) != len(points) or systems not in ({'distserve'}, {'dynamollm'})
                or any(p.get('qualification_mode') != scope
                       or p.get('result_policy') != 'all_recorded_windows/v1' for p in points)):
            raise ValueError('baseline observation group requires one explicit policy and result scope')
        if dynamo:
            from .comparison_baseline_observation import make_dynamo_adapter
            return make_dynamo_adapter(out, base_port=base_port)
    adapter_class = DynamoResidentAdapter if dynamo else NativeResidentAdapter
    return adapter_class(out, base_port=base_port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base-port', type=int, required=True)
    parser.add_argument('--previous', type=Path, action='append', default=[])
    args = parser.parse_args()
    group = json.loads(args.group.read_text())
    adapter = make_resident_adapter(group, args.out, base_port=args.base_port)
    session = ResidentGroupSession(group, adapter, args.out, previous=args.previous)
    result = asyncio.run(session.run())
    print(json.dumps(dict(status=result['status'], windows=len(result['windows']))))
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
