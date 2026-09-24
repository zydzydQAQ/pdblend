"""Independent Dynamo sessions: retain weights, recreate every window controller.

No policy/profile is borrowed from another system. A session only reuses the
unchanged processes and topology it started. Dynamic topology changes are still
allowed by the native controller, but require a separate launch for the next
point until restoring that topology has its own hardware qualification.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from .deployment import SubprocessLifecycle, save
from .policy import PERIODS
from .profiles import PaperProfiles
from .runtime import DynamoController
from .telemetry import GroupTelemetry
from .transport import V1Transport
from .validation import MODELS, preflight


class ResidentReuseError(RuntimeError):
    """No resident window was run; a separate qualified launch is required."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _config(config, mode):
    value = deepcopy(config)
    value['mode'] = mode
    value['legal_tp'] = list(MODELS.get(value.get('model_id'), ()))
    value['tokenizer'] = value.get('model_path')
    value['dynamo_require_full_mechanisms'] = mode in ('full', 'comparison')
    value.setdefault('base_port', 16000)
    value.setdefault('dynamo_reference_tp', 4)
    if value.get('functional_profile_stage', {}).get('enabled'):
        raise ResidentReuseError('resident windows require an already prepared independent profile')
    return value


def _engine_identity(config):
    defaults = dict(max_model_len=8192, max_num_seqs=16, max_num_batched_tokens=8192,
                    gpu_memory_utilization=.85, base_port=16000)
    result = {key:config.get(key, default) for key,default in defaults.items()}
    for key in ('model_id', 'model_path', 'node_gpus', 'legal_tp'):
        result[key] = config.get(key)
    result['target_port'] = config.get('target_port', result['base_port']+32)
    result['store_port'] = config.get('store_port', result['base_port']+80)
    result['instances'] = [_instance_identity(row) for row in config['instances']]
    result['environment'] = {key:os.environ.get(key) for key in (
        'PDBLEND_SOURCE_SHA256', 'PDBLEND_IMAGE_ID', 'PDBLEND_GPU_UUIDS',
        'PDBLEND_MODEL_VERIFICATION_RECEIPT', 'PDBLEND_CLOCK_LOCK_DIR')}
    return result


def _instance_identity(row):
    return dict(instance_id=row.get('instance_id', row.get('id')), gpus=list(row['gpus']),
                tp=row['tp'], pp=row.get('pp', 1), generation=row.get('generation', 0),
                port=row['port'], url=row.get('url', 'http://127.0.0.1:'+str(row['port'])))


def _idle(state, spec):
    """Require fresh scheduler and every rank, including retained/transfer KV."""
    raw = state.get('native_raw', state)
    now = time.time()
    generation = spec.get('generation', 0)
    required = ('all_queue', 'running', 'waiting', 'kv_allocations', 'retained_kv_requests',
                'transfer_allocations', 'pending_transfers', 'free_kv_tokens', 'total_kv_tokens')
    if (any(key not in raw for key in required)
            or any(raw[key] for key in required[:-2])
            or raw['free_kv_tokens'] != raw['total_kv_tokens'] or raw['total_kv_tokens'] <= 0
            or raw.get('tp') != spec['tp'] or raw.get('pp') != 1
            or raw.get('generation') != generation or raw.get('acknowledged_generation') != generation
            or raw.get('native_evidence_complete') is not True or raw.get('transport_healthy') is not True
            or raw.get('healthy') is not True or raw.get('dynamo_weights_ready') is not True):
        raise ResidentReuseError('Dynamo resident scheduler/KV/identity is not clean')
    stamp = raw.get('native_at_s')
    began, ended = raw.get('rank_observation_started_s'), raw.get('rank_observation_finished_s')
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in (stamp, began, ended))
            or not -.05 <= now-stamp <= 1 or not began <= ended <= now+.05 or now-began > 1):
        raise ResidentReuseError('Dynamo resident scheduler/rank snapshot is stale')
    ranks = raw.get('ranks', [])
    if (len(ranks) != spec['tp'] or {r.get('rank') for r in ranks} != set(range(spec['tp']))
            or any(r.get('generation') != generation or r.get('healthy') is not True
                   or r.get('native_evidence_complete') is not True
                   or not began-.05 <= r.get('at_s', -1) <= ended+.05
                   or 'pending_transfers' not in r or r['pending_transfers']
                   or 'transfer_allocations' not in r or r['transfer_allocations']
                   or any(r.get('retained', {}).get(key) for key in (
                       'held_count', 'held_requests', 'held_bytes', 'receiving_transactions')) for r in ranks)):
        raise ResidentReuseError('Dynamo resident all-rank drain evidence is incomplete')
    return raw


class _Journal:
    def __init__(self, journal):
        self.session, self.window = journal, None

    def __call__(self, event, **fields):
        self.session(event, **fields)
        if self.window is not None:
            self.window(event, **fields)


class ResidentSession:
    """One lease, one engine identity, sequential independent request windows.

    ``start`` performs a fresh asset preflight before allocating NVML or engines.
    ``execute_window`` repeats it with the window config/trace. ``close`` owns
    only the process groups created here. Failed or changed sessions cannot be
    reused; the caller may start a new session or use run_v1.execute.
    """

    @classmethod
    async def start(cls, config, output, *, mode='functional', duration_s=150):
        from .run_v1 import Journal
        if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('positive finite resident window duration required')
        value = _config(config, mode)
        checked = preflight(value, mode=mode, duration_s=duration_s, seed=701)
        if not checked['ready']:
            raise ResidentReuseError('Dynamo resident preflight failed: '+json.dumps(checked['missing_evidence']))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError('refusing to overwrite Dynamo resident session')
        self = cls()
        self.config, self.output, self.preflight = value, output, checked
        self.identity = _engine_identity(value)
        self.identity_sha256 = _digest(self.identity)
        self.journal = _Journal(Journal(output/'events.jsonl'))
        self.telemetry = self.transport = self.lifecycle = None
        self.closed, self.quarantined = False, False
        self.windows = 0
        self.lock = asyncio.Lock()
        self.capabilities, self.initial_processes = {}, {}
        save(output/'config.json', value)
        save(output/'preflight.json', checked)
        try:
            if not all(self.identity['environment'].get(k) for k in (
                    'PDBLEND_SOURCE_SHA256', 'PDBLEND_IMAGE_ID', 'PDBLEND_GPU_UUIDS')):
                raise ResidentReuseError('resident engines require source/image/physical GPU lease identity')
            self.telemetry = GroupTelemetry(value['node_gpus'], self.journal)
            self.telemetry.start()
            self.transport = V1Transport(value['instances'], self.telemetry.clock, self.journal)
            await self.transport.start()
            self.lifecycle = SubprocessLifecycle(value, self.transport, self.journal, output/'engines')
            for row in value['instances']:
                await self.lifecycle.start(row)
            self.initial_processes = self._process_identity()
            await self._boundary(value, 'session_start', first=True)
            save(output/'started.json', dict(status='passed', engine_identity=self.identity,
                engine_identity_sha256=self.identity_sha256, capabilities=self.capabilities,
                processes=self.initial_processes, formal_eligible=False, energy_comparable=False))
            return self
        except BaseException as exc:
            self.quarantined = True
            try:
                await self._close()
            except BaseException as cleanup:
                raise ResidentReuseError(f'resident startup failed: {exc!r}; cleanup failed: {cleanup!r}') from exc
            raise

    def _process_identity(self):
        return {iid:dict(_instance_identity(row), pid=self.lifecycle.processes[iid].pid)
                for iid,row in self.lifecycle.instances.items()}

    async def _boundary(self, config, label, *, first=False):
        if (self._process_identity() != self.initial_processes
                or set(self.transport.instances) != set(self.initial_processes)
                or any(p.returncode is not None for p in self.lifecycle.processes.values())
                or (label == 'window_end' and self.journal.window is not None and any(
                    row['event'] in ('dynamo_instance_start', 'dynamo_instance_stop')
                    for row in self.journal.window.rows))):
            raise ResidentReuseError('physical layout/process changed; independent launch required')
        profiles = PaperProfiles.load(config['profiles'])
        records = {}
        for original in config['instances']:
            spec = _instance_identity(original)
            iid = spec['instance_id']
            cap = await self.transport.json(iid, '/baseline/capability', method='GET')
            identity = {key:cap.get(key) for key in ('model_id', 'model_hash', 'tokenizer_hash',
                'engine_revision', 'source_revision', 'image_digest', 'gpu_uuids', 'tp', 'pp')}
            if (cap.get('supported') is not True or identity['model_id'] != config['model_id']
                    or not identity['model_hash'] or not identity['tokenizer_hash']
                    or identity['engine_revision'] != 'vllm-0.10.1.1'
                    or identity['source_revision'] != self.identity['environment']['PDBLEND_SOURCE_SHA256']
                    or identity['image_digest'] != self.identity['environment']['PDBLEND_IMAGE_ID']
                    or identity['tp'] != spec['tp'] or identity['pp'] != 1
                    or identity['gpu_uuids'] != [self.telemetry.uuids[g] for g in spec['gpus']]
                    or (not first and identity != self.capabilities[iid])):
                raise ResidentReuseError('Dynamo resident native capability identity changed')
            if first:
                self.capabilities[iid] = identity
            quiesce = await self.transport.json(iid, '/baseline/dynamollm/quiesce', {})
            if quiesce.get('accepting') is not False:
                raise ResidentReuseError('Dynamo resident quiesce ACK missing')
            ack = await self.transport.json(iid, '/baseline/dynamollm/drain', {})
            if ack.get('drained') is not True or ack.get('owner_ack') is not True:
                raise ResidentReuseError('Dynamo resident drain ACK missing')
            ranks = ack.get('ranks', [])
            if (len(ranks) != spec['tp'] or {r.get('rank') for r in ranks} != set(range(spec['tp']))
                    or any(r.get('ok') is not True or r.get('drained') is not True
                           or r.get('generation') != spec['generation']
                           or r.get('cuda_synchronized') is not True
                           or r.get('active_weight_sessions') != 0 for r in ranks)):
                raise ResidentReuseError('Dynamo resident drain rank ACK missing')
            state = _idle(await self.transport.state(iid), spec)
            frequency = original.get('frequency_mhz', max(profiles.frequencies(spec['tp'])))
            await self.transport.clock(spec['gpus'], frequency)
            resume = await self.transport.json(iid, '/baseline/dynamollm/resume', {})
            if resume.get('accepting') is not True:
                raise ResidentReuseError('Dynamo resident resume ACK missing')
            ready = _idle(await self.transport.state(iid), spec)
            if any(ready.get(k) != v for k,v in dict(role='mixed', mode='temporal',
                    accepting=True, admit_prefill=True, admit_decode=True).items()):
                raise ResidentReuseError('Dynamo resident initial admission state not restored')
            records[iid] = dict(drain=ack, drained_state=state, ready_state=ready,
                                frequency_mhz=frequency)
        self.journal('dynamo_resident_boundary', label=label, instances=records,
                     engine_identity_sha256=self.identity_sha256)
        return records

    async def boundary(self, *, config=None, label='external_boundary'):
        """Fresh native drain/clock/admission receipt for an outer coordinator.

        This does not create or retain a controller. Each execute_window still
        owns its fresh predictor, controller, warmup and wall-clock epoch.
        """
        if self.closed or self.quarantined or self.lock.locked():
            raise ResidentReuseError('resident boundary requires an idle healthy session')
        value = _config(config or self.config, self.config['mode'])
        if _engine_identity(value) != self.identity:
            raise ResidentReuseError('engine configuration changed; independent launch required')
        async with self.lock:
            try:
                return await self._boundary(value, label)
            except BaseException:
                self.quarantined = True
                await self._close()
                raise

    async def execute_window(self, trace, *, output, duration_s=150, mode='comparison',
                             config=None, receipt=None):
        from .run_v1 import Journal, load_trace, qualify
        from .relay import StagedGpuTopologyHooks
        if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError('positive finite resident window duration required')
        if self.closed or self.quarantined or self.lock.locked():
            raise ResidentReuseError('resident session is closed, quarantined or already executing')
        value = _config(config or self.config, mode)
        if _engine_identity(value) != self.identity:
            raise ResidentReuseError('engine configuration changed; independent launch required')
        if mode not in ('functional', 'comparison'):
            raise ResidentReuseError('full/primitive validation retains its independently owned lifecycle')
        checked = preflight(value, mode=mode, duration_s=duration_s, seed=701)
        if not checked['ready'] or receipt is not None and receipt.get('ready') is not True:
            raise ResidentReuseError('window preflight failed: '+json.dumps(checked['missing_evidence']))
        if checked['evidence'].get('model_identity') != self.preflight['evidence'].get('model_identity'):
            raise ResidentReuseError('verified model identity changed within resident session')
        if trace != load_trace(value['trace'], duration_s):
            raise ResidentReuseError('resident requests differ from the bound window trace')
        from .deployment import sha
        checked['evidence'].update(trace_sha256=sha(value['trace']), config_sha256=_digest(value))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError('refusing to overwrite Dynamo resident window')
        async with self.lock:
            window = Journal(output/'events.jsonl')
            self.journal.window = window
            self.windows += 1
            controller, tasks, outcomes = None, [], []
            failure, cleanup_errors = None, []
            boundaries, started_s, requests_done_s = {}, None, None
            save(output/'config.json', value)
            save(output/'preflight.json', checked)
            save(output/'trace.json', dict(seed=701, requests=trace, duration_s=duration_s))
            try:
                boundaries['before'] = await self._boundary(value, 'window_start')
                self.transport.dynamo_topology = None
                if mode == 'comparison':
                    self.transport.dynamo_topology = StagedGpuTopologyHooks(
                        self.transport, self.lifecycle, self.journal,
                        goldens={int(tp):v for tp,v in value['goldens'].items()},
                        store_port=value.get('store_port', value['base_port']+80))
                rng = random.Random(9701)
                prompt = [rng.randint(1000,60000) for _ in range(128)]
                for iid in self.transport.instances:
                    ids = []
                    async for event in self.transport.stream(iid, dict(prompt=prompt,
                            max_tokens=16, seed=9701, temperature=0, ignore_eos=True,
                            request_id=f'dynamo-resident-warmup-{self.windows}-{iid}')):
                        ids.extend(event.get('token_ids', []))
                    if len(ids) != 16:
                        raise RuntimeError('Dynamo resident warmup token accounting failed')
                controller = DynamoController(value, self.transport, self.journal)
                await controller.startup()
                origin = time.monotonic()
                started_s = time.time()
                self.journal('dynamo_service_window_start', service_started_s=started_s,
                    duration_s=duration_s, service_deadline_s=started_s+duration_s,
                    gpu_uuids=self.telemetry.uuids, mode=mode,
                    profile_sha256=controller.profiles.fingerprint)

                async def request(row):
                    await asyncio.sleep(max(0., origin+row['arrival_s']-time.monotonic()))
                    item = dict(request_id=row['request_id'], arrival_s=row['arrival_s'],
                        scheduled_s=started_s+row['arrival_s'], submitted_s=time.time(),
                        input_tokens=len(row['prompt']), max_tokens=row['max_tokens'],
                        sampling_seed=701, completion_tokens=0, terminal=False, ok=False)
                    ids, stream = [], None
                    try:
                        stream = controller.handle(dict(prompt=row['prompt'], max_tokens=row['max_tokens'],
                            temperature=0, seed=701, ignore_eos=True), row['request_id'])
                        async for event in stream:
                            received = time.time()
                            self.journal('dynamo_sse', request_id=row['request_id'],
                                         received_s=received, payload=dict(event, received_s=received))
                            if event.get('token_ids'):
                                item.setdefault('first_token_s', received)
                                item['last_token_s'] = received
                                ids.extend(event['token_ids'])
                            item['terminal'] |= event.get('finished') is True or any(
                                c.get('finish_reason') is not None for c in event.get('choices', []))
                        item['ok'] = item['terminal'] and len(ids) == row['max_tokens']
                    except asyncio.CancelledError:
                        item['error'] = 'cancelled'
                        raise
                    except Exception as exc:
                        item['error'] = repr(exc)
                    finally:
                        if stream is not None:
                            try:
                                await stream.aclose()
                            except Exception as exc:
                                item.update(ok=False, error=repr(exc))
                        item.update(finished_s=time.time(), completion_tokens=len(ids),
                                    journal_path=window.path.name, token_ids_sha256=_digest(ids))
                        outcomes.append(item)
                        self.journal('dynamo_outcome', **item)

                tasks = [asyncio.create_task(request(row)) for row in trace]
                await asyncio.sleep(duration_s)
                await asyncio.wait_for(asyncio.gather(*tasks), value.get('request_drain_timeout_s',600))
                requests_done_s = time.time()
                self.journal('dynamo_service_window_end', service_deadline_s=started_s+duration_s,
                             requests_done_s=requests_done_s, elapsed_s=time.monotonic()-origin)
                if len(outcomes) != len(trace) or not all(row['ok'] for row in outcomes):
                    raise RuntimeError('resident window lost or failed an offered request')
            except BaseException as exc:
                failure = repr(exc)
                self.journal('dynamo_run_failure', error=failure)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                observed = {row['request_id'] for row in outcomes}
                for row in trace:
                    if row['request_id'] not in observed:
                        # Preserve the planned denominator when cancellation happened
                        # before a request task reached its native submission.
                        item = dict(request_id=row['request_id'], arrival_s=row['arrival_s'],
                            scheduled_s=None if started_s is None else started_s+row['arrival_s'],
                            submitted_s=None, input_tokens=len(row['prompt']),
                            max_tokens=row['max_tokens'], completion_tokens=0, terminal=False,
                            sampling_seed=701, ok=False, error='window_failed_before_submission',
                            finished_s=time.time())
                        outcomes.append(item)
                        self.journal('dynamo_outcome', **item)
                if controller is not None:
                    try:
                        await asyncio.wait_for(controller.close(), value.get('controller_close_timeout_s',30))
                    except BaseException as exc:
                        cleanup_errors.append(dict(component='controller', error=repr(exc)))
                if not cleanup_errors:
                    try:
                        boundaries['after'] = await self._boundary(value, 'window_end')
                    except BaseException as exc:
                        cleanup_errors.append(dict(component='resident_boundary', error=repr(exc)))
                self.transport.dynamo_topology = None
                result = qualify(window.rows, outcomes, mode=mode, duration_s=duration_s)
                if window.power_samples < 2 or window.power_errors:
                    result['failures'].append('resident_power_evidence_incomplete')
                    result.update(status='inconclusive', complete=False)
                if failure or cleanup_errors:
                    result.update(status='failed', complete=False, error=failure)
                self.quarantined = result['status'] != 'passed'
                result.update(system='dynamollm', model_id=value['model_id'], duration_s=duration_s,
                    service_started_s=started_s, service_deadline_s=None if started_s is None else started_s+duration_s,
                    requests_done_s=requests_done_s, offered_requests=len(trace), cleanup_errors=cleanup_errors,
                    provenance=checked['evidence'], trace_sha256=checked['evidence']['trace_sha256'],
                    periods_s=dict(PERIODS),
                    session_path=str(self.output.resolve()), window_index=self.windows,
                    engine_identity_sha256=self.identity_sha256, resident_reusable=not self.quarantined,
                    independent_launch_required=self.quarantined, resident_boundaries=boundaries,
                    gpu_uuids=self.telemetry.uuids, group_power_samples=window.power_samples,
                    group_power_errors=window.power_errors, stationary_weight_bytes=0,
                    original_weight_retention_implemented=False, journal_path=window.path.name,
                    raw_schema='pdblend-journal-v1', own_cleanup_complete=not cleanup_errors)
                self.journal.window = None
                window.close()
                from .deployment import sha
                result['events_sha256'] = sha(window.path)
                save(output/'outcomes.json', outcomes)
                if self.quarantined:
                    try:
                        await self._close()
                    except BaseException as exc:
                        result['cleanup_errors'].append(dict(component='session', error=repr(exc)))
                        result['own_cleanup_complete'] = False
                save(output/'completion.json', result)
            return result

    async def close(self):
        if self.lock.locked():
            raise ResidentReuseError('cannot close a resident session during its active window')
        await self._close()

    async def _close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        for name, resource in (('lifecycle',self.lifecycle), ('telemetry',self.telemetry),
                               ('transport',self.transport)):
            if resource is not None:
                try:
                    await resource.close()
                except BaseException as exc:
                    errors.append(dict(component=name, error=repr(exc)))
        self.journal('dynamo_resident_closed', cleanup_errors=errors, windows=self.windows)
        self.journal.session.close()
        from .deployment import sha
        save(self.output/'completion.json', dict(status='failed' if errors else 'passed',
            complete=not errors, windows=self.windows, quarantined=self.quarantined,
            cleanup_errors=errors, formal_eligible=False, energy_comparable=False,
            scope='resident_lifecycle', events_sha256=sha(self.journal.session.path)))
        if errors:
            raise ResidentReuseError('Dynamo session owned cleanup incomplete: '+repr(errors))
