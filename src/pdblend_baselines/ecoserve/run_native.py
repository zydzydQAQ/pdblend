"""Execute EcoServe's own controller on identity-checked resident native engines.

The runner owns requests, control and clock cleanup, never engine lifetime.
Functional completion and automatically triggered membership changes have
separate receipts; a quiet observation cycle is not a scaling action.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .runtime import EcoServeRuntime, validate_native_state
from .mechanism_four import ObservedTransport, tokens
from ..native_profile import audit as audit_profile
from pdblend.results.journal import CompactJournal, payload_receipt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trace(path: Path, duration: float):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('positive finite service duration required')
    value = json.loads(path.read_text())
    if value.get('seed') != 701 or not isinstance(value.get('requests'), list) or not value['requests']:
        raise ValueError('nonempty shared seed-701 trace required')
    previous = -1.0
    rows = []
    for index, row in enumerate(value['requests']):
        arrival = row.get('arrival_s', row.get('at_s'))
        prompt = row.get('prompt')
        count = row.get('max_tokens', row.get('output_tokens'))
        if (type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival < previous
                or not 0 <= arrival < duration or not isinstance(prompt, list) or not prompt
                or any(type(t) is not int or t < 0 for t in prompt)
                or type(count) is not int or not 1 <= count <= 512 or len(prompt)+count > 8192):
            raise ValueError(f'invalid shared trace request {index}')
        rows.append((arrival, prompt, count))
        previous = arrival
    return value, rows


def validate_state(state, tp, *, drained=False):
    validate_native_state(state)
    generation = state['generation']
    ranks = state.get('ranks', [])
    now = time.time()
    if (type(generation) is not int or state.get('tp') != tp or state.get('pp') != 1
            or state.get('transport_healthy') is not True or state.get('healthy') is not True
            or len(ranks) != tp or {r.get('rank') for r in ranks} != set(range(tp))
            or not -.05 <= now-state['native_at_s'] <= 1
            or any(r.get('generation') != generation or r.get('healthy') is not True
                   or r.get('native_evidence_complete') is not True for r in ranks)):
        raise RuntimeError('native EcoServe all-rank generation/health/topology/freshness evidence differs')
    required = ('all_queue', 'running', 'waiting', 'kv_allocations', 'retained_kv_requests',
                'transfer_allocations', 'pending_transfers', 'free_kv_tokens', 'total_kv_tokens')
    if any(key not in state for key in required):
        raise RuntimeError('native EcoServe request/KV inventory incomplete')
    observed_start = state.get('rank_observation_started_s', float('nan'))
    observed_end = state.get('rank_observation_finished_s', float('nan'))
    if (not all(math.isfinite(value) for value in (observed_start, observed_end))
            or not observed_start <= observed_end <= now+.05
            or any(not observed_start-.05 <= r.get('at_s', -1) <= observed_end+.05
                   or 'pending_transfers' not in r or 'transfer_allocations' not in r for r in ranks)):
        raise RuntimeError('native EcoServe rank receipts are stale or incomplete')
    if drained and (any(state[key] for key in required[:-2])
                    or state['free_kv_tokens'] != state['total_kv_tokens']
                    or any(r.get('pending_transfers') or r.get('transfer_allocations')
                           or any(r.get('retained', {}).get(key) for key in
                                  ('held_count', 'held_requests', 'held_bytes', 'receiving_transactions'))
                           for r in ranks)):
        raise RuntimeError('native EcoServe drain retained requests, transfers, or KV blocks')


def expected_identity(config):
    source = os.environ.get('PDBLEND_SOURCE_SHA256')
    image = os.environ.get('PDBLEND_IMAGE_ID')
    receipt_path = config.get('model_verification_receipt') or os.environ.get('PDBLEND_MODEL_VERIFICATION_RECEIPT')
    if not source or not image or not receipt_path:
        raise ValueError('explicit source/image and verified model inventory are required')
    receipt = json.loads(Path(receipt_path).read_text())
    model = next((v for v in receipt.get('models', {}).values() if v.get('model_id') == config.get('model_id')), None)
    if receipt.get('all_pass') is not True or not model or model.get('verified') is not True:
        raise ValueError('EcoServe model is absent from the verified model inventory')
    expected = dict(model_id=config['model_id'], engine_revision='vllm-0.10.1.1',
                    source_revision=source, image_digest=image, verification_receipt_sha256=sha(receipt_path))
    for kind, key in (('weight', 'model_hash'), ('tokenizer', 'tokenizer_hash')):
        inventory = [(r['path'], r['bytes'], r['sha256']) for r in model['files'] if r['kind'] == kind]
        if not inventory:
            raise ValueError('EcoServe verified model inventory lacks '+kind)
        expected[key] = hashlib.sha256(json.dumps(inventory, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
    return expected


def validate_profile(config, expected, tp):
    path = Path(config['eco_prefill_csv'])
    report = audit_profile(path)
    manifest = Path(str(path)+'.manifest.json')
    raw = json.loads(manifest.read_text())
    meta = report['metadata']
    if report['system'] != 'ecoserve' or raw.get('model') != expected['model_id']:
        raise ValueError('EcoServe requires its own same-model native prefill CSV')
    for key in ('model_hash', 'tokenizer_hash', 'image_digest'):
        if meta.get(key) != expected[key]:
            raise ValueError('EcoServe profile provenance differs: '+key)
    if (meta.get('engine_version') != expected['engine_revision'] or meta.get('tp') != tp
            or meta.get('pp') != 1 or meta.get('frequency_mhz') != config.get('eco_active_frequency_mhz', 2520)):
        raise ValueError('EcoServe profile topology/engine/frequency differs')
    points = list(csv.DictReader(path.read_text().splitlines()))
    measured = {r['input_tokens']:r['minimum_ms'] for r in raw['rows']}
    if len(points) != len(measured) or {int(r['Length']):float(r['Prefill Time']) for r in points} != measured:
        raise ValueError('EcoServe CSV differs from its independently audited forward samples')
    return dict(csv_sha256=sha(path), manifest_sha256=sha(manifest), metadata=meta,
                raw_audit=report, formal_eligible=False)


class VerifiedTransport(ObservedTransport):
    """Audit the actual HTTP control receipts and physical clock ownership."""
    def __init__(self, endpoints, journal, specs, uuid_map):
        super().__init__(endpoints, journal)
        self.specs, self.uuid_map = specs, uuid_map

    async def state(self, identifier):
        result = await super().state(identifier)
        validate_state(result, self.specs[identifier]['tp'])
        return result

    async def _instance_call(self, identifier, method, path, body=None):
        lifecycle = getattr(self, 'comparison_lifecycle', None)
        if lifecycle is None:
            result = await super()._instance_call(identifier, method, path, body)
        else:
            with lifecycle.http_scope(identifier, method, path, body):
                result = await super()._instance_call(identifier, method, path, body)
        if path in ('/baseline/clock', '/baseline/park'):
            uuids = {self.uuid_map[g] for g in self.specs[identifier]['gpus']}
            rows = result.get('gpus', [])
            if (result.get('acknowledged') is not True or result.get('success') is not True
                    or len(rows) != len(uuids) or {r.get('gpu_uuid') for r in rows} != uuids):
                raise RuntimeError('EcoServe native physical clock receipt incomplete')
        if path == '/baseline/control' and result.get('acknowledged') is not True:
            raise RuntimeError('EcoServe native control was not acknowledged')
        return result


def automatic_actions(journal):
    result = []
    for index, row in enumerate(journal):
        if (row['kind'] != 'eco_membership_commit' or row.get('origin') != 'paper_supplement'
                or row.get('trigger') not in ('mean_ttft', 'saved_tpot') or row.get('before') == row.get('after')):
            continue
        prepares = [i for i, old in enumerate(journal[:index]) if old['kind'] == 'eco_membership_prepare'
                    and all(old.get(k) == row.get(k) for k in ('operation', 'instance_id', 'trigger', 'before'))]
        if not prepares:
            continue
        required_path = '/baseline/clock' if row['operation'] == 'add' else '/baseline/park'
        receipts = [i for i in range(prepares[-1]+1, index) if journal[i]['kind'] == 'eco_http_receipt'
                    and journal[i].get('instance_id') == row['instance_id']
                    and journal[i].get('path') == required_path
                    and journal[i].get('response', {}).get('acknowledged') is True]
        if receipts:
            result.append(dict(commit_index=index, prepare_index=prepares[-1], receipt_indices=receipts,
                               operation=row['operation'], instance_id=row['instance_id'], trigger=row['trigger']))
    return result


async def execute(config, endpoints, trace_path, out, duration=100.0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('completion.json', 'events.jsonl', 'events.jsonl.gz')):
        raise FileExistsError('refusing to overwrite EcoServe execution evidence')
    artifact = dict(schema='ecoserve-native-run-v2', system='ecoserve', model_id=config.get('model_id'),
        seed=701, duration_s=duration, status='failed', complete=False, formal_eligible=False,
        energy_comparable=False, functional_status='failed', automatic_policy_status='inconclusive',
        scope='functional', complete_reproduction=False,
        endpoints=endpoints, outcomes=[], actions=[], automatic_policy_triggered=False,
        trace_sha256=sha(trace_path), config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest())
    journal, tasks, verified = [], [], []
    runtime = transport = lifecycle = None
    raw = CompactJournal(out/'events.jsonl.gz')
    def emit(kind, **fields):
        if lifecycle is not None and kind == 'eco_http_receipt':
            fields = lifecycle.decorate_http(fields)
        fields.setdefault('at_s', time.time())
        row = dict(kind=kind, **fields)
        raw.write(row)
        if isinstance(row.get('payload'), dict):
            row = dict(row, payload={key:value for key,value in row['payload'].items()
                                    if key not in ('text', 'choices')})
        journal.append(row)
    try:
        if 'eco_comparison_lifecycle' in config:
            from .comparison_lifecycle import ComparisonLifecycle
            lifecycle = ComparisonLifecycle(config['eco_comparison_lifecycle'], emit)
        _, rows = load_trace(Path(trace_path), duration)
        specs = {row['id']:dict(row) for row in config['instances']}
        if len(specs) != len(config['instances']) or set(specs) != set(endpoints):
            raise ValueError('distinct EcoServe specs must exactly match endpoints')
        physical = os.environ.get('PDBLEND_GPU_UUIDS', '').split(',')
        groups = [gpu for spec in specs.values() for gpu in spec['gpus']]
        if (not groups or len(groups) > 8 or len(groups) != len(set(groups))
                or any(type(g) is not int or not 0 <= g < len(physical) or not physical[g].startswith('GPU-') for g in groups)
                or len({physical[g] for g in groups}) != len(groups)):
            raise ValueError('EcoServe requires disjoint physical UUID groups within its lease')
        for spec in specs.values():
            spec.setdefault('tp', len(spec['gpus']))
            if spec['tp'] != len(spec['gpus']) or spec.get('pp', 1) != 1:
                raise ValueError('EcoServe fixed TP/PP1 group differs from its GPU allocation')
        if len({spec['tp'] for spec in specs.values()}) != 1:
            raise ValueError('EcoServe requires one fixed TP across its resident instances')
        expected = expected_identity(config)
        artifact['identity'] = expected
        artifact['profile'] = validate_profile(config, expected, next(iter(specs.values()))['tp'])
        artifact['gpu_uuids'] = {g:physical[g] for g in groups}
        transport = VerifiedTransport(endpoints, emit, specs, artifact['gpu_uuids'])
        if lifecycle is not None:
            transport.comparison_lifecycle = lifecycle
        runtime = EcoServeRuntime(config, transport, emit)
        artifact['capabilities'] = {}
        for iid, spec in specs.items():
            cap = await transport._instance_call(iid, 'GET', '/baseline/capability')
            if (any(cap.get(k) != v for k, v in expected.items()) or cap.get('supported') is not True
                    or cap.get('tp') != spec['tp'] or cap.get('pp') != 1
                    or len(cap.get('gpu_uuids', [])) != spec['tp']
                    or set(cap.get('gpu_uuids', [])) != {physical[g] for g in spec['gpus']}):
                raise RuntimeError('EcoServe native endpoint model/source/image/TP/GPU identity differs: '+iid)
            validate_state(cap['state'], spec['tp'], drained=True)
            artifact['capabilities'][iid] = cap
            verified.append(iid)
        await runtime.start()
        artifact['initial_native_states'] = {iid:await transport.state(iid) for iid in endpoints}
        started = time.monotonic()
        artifact['service_started_s'] = time.time()
        emit('eco_service_window_start', duration_s=duration)
        async def request(index, arrival, prompt, count):
            await asyncio.sleep(max(0., started+arrival-time.monotonic()))
            rid = f'ecoserve-701-{index}'
            row = dict(request_id=rid, input_tokens=len(prompt), output_tokens=count,
                       arrival_s=arrival, submitted_s=time.time(), events=[], token_ids=[], ok=False)
            try:
                async with aclosing(runtime.handle(dict(prompt=prompt, max_tokens=count, seed=701,
                                                        temperature=0, ignore_eos=True), rid)) as stream:
                    async for event in stream:
                        row['events'].append({key:value for key,value in event.items()
                                              if key not in ('text', 'choices')})
                        emit('eco_client_sse', request_id=rid, payload=event)
                        if event.get('token_ids'):
                            row.setdefault('first_token_s', time.time())
                row['token_ids'] = tokens(row['events'])
                native = transport.engine_outputs.get(rid, [])
                row['native_token_ids'] = tokens(native)
                row['ok'] = (len(row['token_ids']) == count and row['token_ids'] == row['native_token_ids']
                             and bool(row['events'] and row['events'][-1].get('finished'))
                             and bool(native and native[-1].get('finished'))
                             and any(r['kind'] == 'eco_admission' and r.get('request_id') == rid for r in journal))
            except Exception as exc:
                row['error'] = repr(exc)
            finally:
                row['finished_s'] = time.time()
                row.update(payload_receipt(row.pop('events'), journal_path='events.jsonl.gz', request_id=rid))
                row.pop('token_ids', None)
                row.pop('native_token_ids', None)
                artifact['outcomes'].append(row)
                emit('eco_request_outcome', **row)
        tasks = [asyncio.create_task(request(i, *row)) if lifecycle is None else
                 lifecycle.create_request_task(request(i, *row), f'ecoserve-701-{i}')
                 for i, row in enumerate(rows)]
        await asyncio.sleep(duration)
        if lifecycle is None:
            await asyncio.wait_for(asyncio.gather(*tasks), float(config.get('request_timeout_s', 120)))
        else:
            await lifecycle.wait_cohort(tasks, float(config.get('request_timeout_s', 120)))
        artifact['service_finished_s'] = time.time()
        if runtime.controller.failure or runtime.controller.quarantined:
            raise RuntimeError('EcoServe background controller failed or quarantined an instance')
        artifact['functional_status'] = ('passed' if len(artifact['outcomes']) == len(rows)
                                         and all(row['ok'] for row in artifact['outcomes']) else 'failed')
    except BaseException as exc:
        artifact['error'] = repr(exc)
    finally:
        if lifecycle is None:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await lifecycle.cancel_cohort(tasks, error=artifact.get('error'))
        cleanup = []
        if runtime is not None:
            try:
                await asyncio.wait_for(runtime.close() if lifecycle is None else lifecycle.close(runtime), 30)
            except BaseException as exc:
                cleanup.append(dict(component='controller_close', error=repr(exc)))
            if runtime.controller.failure or runtime.controller.quarantined:
                cleanup.append(dict(component='controller_state', error=runtime.controller.failure,
                                    quarantined=sorted(runtime.controller.quarantined)))
        artifact['drain_receipts'] = {}
        if transport is not None:
            for iid in verified:
                try:
                    state = await transport.json(iid, '/drain', dict(timeout_s=min(float(config.get('eco_drain_timeout_s', 30)), 25)))
                    if state.get('acknowledged') is not True or state.get('drained') is not True:
                        raise RuntimeError('native drain ACK missing')
                    validate_state(state, transport.specs[iid]['tp'], drained=True)
                    artifact['drain_receipts'][iid] = state
                except BaseException as exc:
                    cleanup.append(dict(component='native_drain', instance_id=iid, error=repr(exc)))
                try:
                    await transport.park(list(transport.specs[iid]['gpus']))
                except BaseException as exc:
                    cleanup.append(dict(component='clock_reset', instance_id=iid, error=repr(exc)))
        artifact['cleanup_errors'] = cleanup
        if lifecycle is not None:
            artifact['comparison_lifecycle'] = lifecycle.summary()
        artifact['drain_kv_released'] = bool(verified) and set(artifact['drain_receipts']) == set(endpoints)
        artifact['journal_path'] = 'events.jsonl.gz'
        artifact['journal_rows'] = len(journal)
        artifact['actions'] = sorted({row['kind'] for row in journal})
        artifact['automatic_actions'] = automatic_actions(journal)
        artifact['automatic_policy_triggered'] = bool(artifact['automatic_actions'])
        functional = artifact['functional_status'] == 'passed' and not artifact.get('error') and not cleanup and artifact['drain_kv_released']
        artifact['functional_status'] = 'passed' if functional else 'failed'
        artifact['automatic_policy_qualified'] = functional and artifact['automatic_policy_triggered']
        artifact['automatic_policy_status'] = 'passed' if artifact['automatic_policy_qualified'] else 'inconclusive'
        # Functional campaign completion does not promote unobserved automatic
        # membership mechanisms; that gate remains explicitly independent.
        artifact['status'] = 'passed' if functional else 'failed'
        artifact['complete'] = artifact['status'] == 'passed'
        raw.close()
        artifact['events_sha256'] = sha(out/'events.jsonl.gz')
        (out/'completion.json').write_text(json.dumps(artifact, indent=2, allow_nan=False)+'\n')
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--endpoint', action='append', required=True)
    parser.add_argument('--duration', type=float, default=100.)
    args = parser.parse_args(argv)
    endpoints = dict(item.split('=', 1) for item in args.endpoint)
    if len(endpoints) != len(args.endpoint):
        parser.error('duplicate endpoint identifiers')
    result = asyncio.run(execute(json.loads(args.config.read_text()), endpoints, args.trace, args.out, args.duration))
    print(json.dumps({k:result[k] for k in ('status', 'complete', 'functional_status', 'automatic_policy_status')}))
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
