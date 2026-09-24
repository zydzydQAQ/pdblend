"""Independent Dynamo resident lifecycle under the common eight-board meter.

No NativeSpec/Fleet instance is allocated by this adapter. Only the native
Dynamo runner can own its worker extension, controller and topology hooks.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import time

from .comparison_campaign import PROTOCOL, binding, load_bound
from .comparison_metering import ComparisonMeteringSession
from .comparison_metrics import canonical_outcomes, reduce_comparison
from .independent_dispatch import validate
from .resident_session import digest, engine_signature, file_sha, write_new
from pdblend.model_registry import ModelRegistry
from pdblend.results.journal import iter_journal
from pdblend.results.power_archive import write_power_archive
from pdblend_baselines.resident_campaign import model_load_lock
from pdblend_baselines.dynamollm.resident import ResidentSession, _config, _engine_identity
from pdblend_baselines.dynamollm.run_v1 import execute_on_resident, load_trace
from pdblend_baselines.dynamollm.validation import preflight


ENTRYPOINT = 'pdblend_baselines.dynamollm.run_v1'
WORKER_EXTENSION = 'pdblend_baselines.dynamollm.gpu_weights.DynamoWorkerExtension'


def is_dynamo_group(group):
    entrypoint = group.get('engine_identity', {}).get('entrypoint', '')
    selected = entrypoint in (ENTRYPOINT, ENTRYPOINT+'.execute',
                             'pdblend_baselines.dynamollm.resident')
    systems = {p.get('system') for p in group.get('points', [])}
    if (selected and systems != {'dynamollm'}) or ('dynamollm' in systems and not selected):
        raise ValueError('Dynamo group must bind its independent native entrypoint')
    return selected


def dynamo_launch_options(config):
    """Canonical switches actually passed by Dynamo's SubprocessLifecycle."""
    return dict(max_model_len=config.get('max_model_len', 8192),
        max_num_seqs=config.get('max_num_seqs', 16),
        max_num_batched_tokens=config.get('max_num_batched_tokens', 8192),
        gpu_memory_utilization=config.get('gpu_memory_utilization', .85),
        kv_connector=None, enforce_eager=True, enable_prefix_caching=False,
        enable_chunked_prefill=True)


def lease_config(config, base_port):
    """Relocate only native endpoints into the scheduler's owned port range."""
    value = deepcopy(config)
    previous = value.get('base_port', 16000)
    if type(base_port) is not int or not 1024 <= base_port <= 65436:
        raise ValueError('Dynamo requires an owned 100-port lease')
    for row in value['instances']:
        if type(row.get('port')) is not int:
            raise ValueError('Dynamo instance requires an integer owned port')
        offset = row['port']-previous
        if not 0 <= offset < 80 or row.get('url', f'http://127.0.0.1:{row["port"]}') != f'http://127.0.0.1:{row["port"]}':
            raise ValueError('Dynamo configuration has an unowned native endpoint')
        row['port'] = base_port+offset
        row['url'] = f'http://127.0.0.1:{row["port"]}'
    for key, default, lower, upper in (('target_port',32,0,80), ('store_port',80,80,93)):
        offset = value.get(key,previous+default)-previous
        if not lower <= offset < upper:
            raise ValueError('Dynamo configuration has an unowned '+key)
        value[key] = base_port+offset
    value['base_port'] = base_port
    ports = [row['port'] for row in value['instances']]
    if len(set(ports)) != len(ports) or value['target_port'] in ports:
        raise ValueError('Dynamo initial or dynamic endpoints collide')
    return value


def bind_dynamo_request_indices(trace, outcomes, journal):
    """Native Dynamo IDs encode list position; frozen idx need not be contiguous."""
    def bound(row):
        match = re.fullmatch(r'dynamo-701-(\d+)',str(row.get('request_id','')))
        if not match or int(match[1]) >= len(trace['requests']):
            raise ValueError('Dynamo client request ID is outside the frozen request schedule')
        position = int(match[1])
        return dict(row,idx=trace['requests'][position].get('idx',position))
    records = (iter_journal(journal) if isinstance(journal,(str,Path)) else journal) or ()
    return [bound(row) for row in outcomes], [bound(row) for row in records
        if row.get('event',row.get('kind')) == 'dynamo_sse']


class DynamoResidentAdapter:
    """ResidentGroupSession adapter; formal preflights precede all hardware."""
    def __init__(self, out, *, base_port):
        self.out, self.base_port = Path(out), base_port
        self.dynamo_session = self.monitor = None
        self.load_count = 0
        self.points = {}
        self.close_receipt = None

    def _prepare(self, point):
        if point.get('duration_s') != 150 or point.get('system') != 'dynamollm':
            raise ValueError('Dynamo comparison adapter requires a native 150-second point')
        if point['trace'] != point['inputs']['trace']:
            raise ValueError('Dynamo point and input trace bindings differ')
        audit = validate(point, point['inputs'])
        if not audit['formal_eligible']:
            raise ValueError('independent Dynamo qualification gates remain closed')
        config = audit['config']
        if file_sha(config['trace']) != point['trace']['sha256']:
            raise ValueError('Dynamo configured trace differs from shared frozen trace')
        if (set(config['node_gpus']) != set(range(8)) or len(config['node_gpus']) != 8
                or any(len(r['prompt'])+r['max_tokens'] > 8192 for r in audit['trace']['requests'])):
            raise ValueError('Dynamo requires the full eight-GPU inventory and supported contexts')
        profile = Path(config['profiles']).resolve()
        selected = [r for r in point['inputs']['profiles'] if Path(r['path']).resolve() == profile]
        if len(selected) != 1:
            raise ValueError('Dynamo selected profile must be uniquely hash-bound')
        load_bound(selected[0])
        value = _config(lease_config(config, self.base_port), 'comparison')
        checked = preflight(value, mode='comparison', duration_s=150, seed=701)
        if not checked['ready']:
            raise ValueError('Dynamo native preflight failed: '+json.dumps(checked['missing_evidence']))
        return value, checked

    def _check_inventory(self, value):
        identity = self.identity
        if identity['worker_extension'] != WORKER_EXTENSION or identity['dtype'] != 'bfloat16':
            raise ValueError('Dynamo worker extension/dtype differs from bound engine identity')
        by_id = {r.get('instance_id',r.get('id')):r for r in value['instances']}
        if set(by_id) != {r['instance_id'] for r in identity['instances']}:
            raise ValueError('Dynamo initial inventory differs from frozen group')
        for row in identity['instances']:
            native = by_id[row['instance_id']]
            if (native['tp'] != row['tp'] or native.get('pp',1) != row['pp']
                    or [identity['fleet_gpu_uuids'][g] for g in native['gpus']] != row['gpu_uuids']
                    or row['launch_options'] != dynamo_launch_options(value)):
                raise ValueError('Dynamo launch switches or placement differ from frozen group')
        for key, expected in identity['environment'].items():
            if os.environ.get(key) != expected:
                raise ValueError('Dynamo launch environment differs: '+key)

    async def _start_monitor(self, actual):
        """Observation adapters may isolate the unchanged public sampler."""
        return ComparisonMeteringSession(range(8), actual).start()

    async def start(self, group):
        if not is_dynamo_group(group):
            raise ValueError('independent Dynamo entrypoint required')
        self.group, self.identity = group, group['engine_identity']
        if engine_signature(self.identity) != group['engine_signature']:
            raise ValueError('Dynamo group signature differs')
        actual = os.environ.get('PDBLEND_GPU_UUIDS','').split(',')
        if actual != self.identity['fleet_gpu_uuids'] or len(set(actual)) != 8:
            raise ValueError('exclusive lease UUIDs differ from frozen Dynamo group')
        initial = None
        for point in group['points']:
            value, checked = self._prepare(point)
            self._check_inventory(value)
            if value['model_id'] != group['model_id']:
                raise ValueError('Dynamo group/model identity differs')
            current = _engine_identity(value)
            if initial is not None and current != initial:
                raise ValueError('Dynamo points require different resident engine identities')
            initial = current
            if point['name'] in self.points:
                raise ValueError('duplicate Dynamo point name')
            self.points[point['name']] = dict(point_sha256=digest(point), config=value, preflight=checked)
        source_path = Path(os.environ['PDBLEND_SOURCE_MANIFEST'])
        source = json.loads(source_path.read_text())
        for name, expected in source['files'].items():
            if file_sha(source_path.parent/name) != expected:
                raise ValueError('immutable execution source changed: '+name)
        model = ModelRegistry(os.environ['PDBLEND_MODELS_DIR'],
            verification_receipt=os.environ['PDBLEND_MODEL_VERIFICATION_RECEIPT']).get(group['model_id'])
        model.validate_config()
        if (model.model_hash != self.identity['model_hash'] or model.tokenizer_hash != self.identity['tokenizer_hash']
                or os.environ['PDBLEND_IMAGE_ID'] != self.identity['image_digest']):
            raise ValueError('Dynamo model/tokenizer/image identity differs')
        expected_path = (Path(os.environ['PDBLEND_MODELS_DIR'])/group['model_id']).resolve()
        if any(Path(p['config']['model_path']).resolve() != expected_path for p in self.points.values()):
            raise ValueError('Dynamo configuration selects a different verified model directory')
        first = next(iter(self.points.values()))['config']
        self.monitor = await self._start_monitor(actual)
        started = time.time()
        with model_load_lock():
            acquired = time.time()
            self.dynamo_session = await ResidentSession.start(first, self.out/'dynamo-session',
                                                             mode='comparison', duration_s=150)
        finished = time.time()
        self.load_count = len(first['instances'])
        for cap in self.dynamo_session.capabilities.values():
            if cap['model_hash'] != model.model_hash or cap['tokenizer_hash'] != model.tokenizer_hash:
                raise ValueError('Dynamo live engine model/tokenizer capability differs')
        reference = []
        for row in first['instances']:
            iid = row.get('instance_id',row.get('id'))
            outputs = []
            for repeat in range(2):
                ids, terminal = [], False
                async for event in self.dynamo_session.transport.stream(iid, dict(
                        request_id=f'dynamo-qualification-{iid}-{repeat}', prompt=list(range(100,228)),
                        max_tokens=16, seed=9701, temperature=0, ignore_eos=True)):
                    ids.extend(event.get('token_ids',[]))
                    terminal |= event.get('finished') is True
                if not terminal or len(ids) != 16:
                    raise RuntimeError('Dynamo ordinary output lacks exact terminal budget')
                outputs.append(ids)
            if outputs[0] != outputs[1]:
                raise RuntimeError('Dynamo ordinary deterministic output differs')
            reference.append(dict(instance_id=iid, seed=9701, token_ids=outputs[0]))
        boundary = await self.dynamo_session.boundary(config=first, label='comparison_startup')
        qualification = dict(scope='independent_dynamo_resident_startup',
            source_manifest=binding(source_path), engine_signature=group['engine_signature'],
            model_hash=model.model_hash, tokenizer_hash=model.tokenizer_hash,
            capabilities=self.dynamo_session.capabilities, exclusive_gpu_uuids=actual,
            ordinary_reference=reference, drain=boundary, native_preflights={
                name:row['preflight'] for name,row in self.points.items()})
        write_new(self.out/'qualification.json',qualification)
        return dict(engine_loads=self.load_count, engine_load_cycles=1,
                    load_lock_wait_s=acquired-started, engine_load_s=finished-acquired,
                    qualification=binding(self.out/'qualification.json'))

    def _point(self, point):
        if self.points.get(point['name'],{}).get('point_sha256') != digest(point):
            raise ValueError('Dynamo point changed after resident startup')
        # Revalidate frozen files and formal qualifications before every window.
        config, checked = self._prepare(point)
        if config != self.points[point['name']]['config']:
            raise ValueError('Dynamo effective configuration changed after startup')
        return config, checked

    async def reset(self, point):
        config, _ = self._point(point)
        states = await self.dynamo_session.boundary(config=config, label='comparison_reset')
        return dict(passed=True, states=states, controller_state='new_controller_per_execute_window',
                    engine_identity_sha256=self.dynamo_session.identity_sha256)

    async def execute(self, point, out):
        # Import only the pure reader; common NativeSpec/Fleet lifecycle is unused.
        from .comparison_runtime import read_native_measurement
        config, checked = self._point(point)
        trace = load_bound(point['trace'])
        out = Path(out)
        raw = await execute_on_resident(config, load_trace(point['trace']['path'],150),
            session=self.dynamo_session, output=out, duration_s=150, mode='comparison', receipt=checked)
        write_new(out/'native-result.json',raw)
        started, outcomes, journal = read_native_measurement('dynamollm',out,raw)
        outcomes, journal = bind_dynamo_request_indices(trace,outcomes,journal)
        rows = canonical_outcomes('dynamollm',trace,outcomes,service_started_s=started,journal=journal)
        drained = raw.get('resident_boundaries',{}).get('after',{})
        tail_end = max(time.time(),started+150.)
        metrics = reduce_comparison(trace,rows,service_started_s=started,duration_s=150.,
            slo=(point['slo']['ttft_s'],point['slo']['tpot_s']),observed_until_s=tail_end)
        await asyncio.sleep(.3)
        metering = self.monitor.summarize(origin_s=started,tail_end_s=tail_end,duration_s=150.)
        write_new(out/'comparison-metering.json',metering)
        write_power_archive(out/'power.json',self.monitor.snapshot())
        write_new(out/'comparison-requests.json',metrics.pop('request_metrics',[]))
        metrics.update({k:v for k,v in metering.items() if not isinstance(v,(list,dict))})
        metrics.update(tail_s=tail_end-started-150.,
            gpu_util_coverage_fraction=metering['util_coverage_fraction'],measurement_protocol_version=PROTOCOL)
        for index, uuid in enumerate(self.identity['fleet_gpu_uuids']):
            gpu = metering['service']['utilization']['per_gpu'][uuid]
            metrics.update({f'gpu{index}_uuid':uuid,f'gpu{index}_util_mean_pct':gpu.get('mean_pct'),
                            f'gpu{index}_util_peak_pct':gpu.get('peak_pct')})
        verified = (metering['energy_comparable'] and bool(drained) and raw.get('complete') is True
                    and raw.get('own_cleanup_complete') is True and raw.get('resident_reusable') is True
                    and metrics['token_timing_complete'])
        return dict(evidence_valid=bool(verified),formal_eligible=False,
            missing_gates=['independent_window_acceptance'],
            qualification=binding(self.out/'qualification.json'),metrics=metrics,
            identity=dict(model_hash=self.identity['model_hash'],tokenizer_hash=self.identity['tokenizer_hash'],
                image_digest=self.identity['image_digest'],runtime_source_sha256=self.identity['runtime_source_sha256'],
                measurement_source_sha256=self.identity['measurement_source_sha256'],
                source_sha256=os.environ['PDBLEND_SOURCE_SHA256'],gpu_uuids=self.identity['fleet_gpu_uuids'],
                measurement_protocol_version=PROTOCOL))

    async def drain(self, point):
        config, _ = self._point(point)
        states = await self.dynamo_session.boundary(config=config,label='comparison_drain')
        return dict(passed=True,states=states)

    async def close(self):
        if self.close_receipt is not None:
            return self.close_receipt
        errors = []
        if self.dynamo_session is not None:
            try:
                await self.dynamo_session.close()
            except Exception as exc:
                errors.append(repr(exc))
        if self.monitor is not None:
            try:
                self.monitor.stop(after_s=time.time())
                write_power_archive(self.out/'session-power.json',self.monitor.snapshot())
            except Exception as exc:
                errors.append(repr(exc))
        self.close_receipt = dict(passed=not errors,errors=errors,engine_loads=self.load_count,
                                 process_cleanup_verified=not errors)
        return self.close_receipt
