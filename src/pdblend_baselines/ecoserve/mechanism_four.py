"""Observed four-member EcoServe primitive; never automatic-policy qualification.

Three requests must retain live KV through an idle fourth member's add/remove.
A separate in-flight request is truly cancelled. Policy rotation and actual
output buffering are required; insufficient workload remains inconclusive.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import random
import time
from .runtime import EcoServeRuntime, MappedEcoServeTransport, validate_native_state


class ObservedTransport(MappedEcoServeTransport):
    def __init__(self, endpoints, journal):
        super().__init__(endpoints)
        self.journal, self.engine_outputs = journal, {}

    async def _instance_call(self, identifier, method, path, body=None):
        started = time.time()
        try:
            result = await super()._instance_call(identifier, method, path, body)
        except BaseException as exc:
            self.journal('eco_http_receipt', instance_id=identifier, method=method, path=path,
                         body=body, started_s=started, error=repr(exc))
            raise
        self.journal('eco_http_receipt', instance_id=identifier, method=method, path=path,
                     body=body, started_s=started, response=result)
        return result

    async def stream(self, identifier, payload):
        rid = payload['request_id']
        async for event in super().stream(identifier, payload):
            self.engine_outputs.setdefault(rid, []).append({key:value for key,value in event.items()
                                                          if key not in ('text', 'choices')})
            self.journal('eco_native_sse', instance_id=identifier, request_id=rid, payload=event)
            yield event


def fresh(state):
    validate_native_state(state)
    timestamp = state['native_at_s']
    if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
            or not -.05 <= time.time()-timestamp <= 1.0 or state.get('transport_healthy') is False
            or not isinstance(state.get('all_queue'), list) or not isinstance(state.get('kv_allocations'), dict)):
        raise RuntimeError('native state is stale or lacks actual request/KV inventory')


async def observe_states(transport, endpoints):
    """Validate each native reply on arrival, before waiting for other GPUs.

    This is a set of timestamped observations, not an atomic fleet snapshot.
    A long prefill on one member must not make an already received fresh idle
    member reply fail the one-second stale-response check retroactively.
    """
    async def observe(identifier):
        state = await transport.state(identifier)
        fresh(state)
        return identifier, state, time.time()
    rows = await asyncio.gather(*(observe(iid) for iid in endpoints))
    return ({iid: state for iid, state, _ in rows},
            {iid: received for iid, _, received in rows})


def tokens(events):
    result = []
    for event in events:
        values = event.get('token_ids')
        if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
            raise RuntimeError('actual token ID SSE evidence missing')
        result.extend(values)
        if event.get('token_index') != len(result):
            raise RuntimeError('SSE token index is discontinuous')
    return result


def kv_blocks(state, rid):
    blocks = state['kv_allocations'].get(rid)
    if (not isinstance(blocks, list) or not blocks or any(not isinstance(group, list) or not group
            or any(type(block) is not int or block < 1 for block in group) for group in blocks)):
        raise RuntimeError('live request lacks actual allocated KV block IDs: '+rid)
    return blocks


def observed_live_tokens(state, rid, events):
    """Pair native outputs only with the state timestamp that can cover them."""
    blocks = state['kv_allocations'].get(rid)
    if rid not in state['all_queue'] or not blocks or any(not group for group in blocks):
        # A newly admitted waiting request legitimately has no allocated KV.
        return None
    snapshot_at = state.get('scheduler_at_s', state['native_at_s'])
    covered = [event for event in events if event.get('at_s', float('inf')) <= snapshot_at]
    if not covered or covered[-1].get('finished'):
        return None
    return len(tokens(covered))


def preserves(before, after):
    return (before['instance_id'] == after['instance_id'] and before['generation'] == after['generation']
            and len(before['kv_blocks']) == len(after['kv_blocks'])
            and all(new[:len(old)] == old for old, new in zip(before['kv_blocks'], after['kv_blocks'])))


async def run(config, endpoints, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_suffix('.events.jsonl')
    if out.exists() or raw_path.exists():
        raise FileExistsError('refusing to overwrite EcoServe mechanism evidence')
    raw = raw_path.open('x')
    rows, errors, outcomes, snapshots = [], [], {}, {}
    tasks, runtime, transport = {}, None, None
    held_instances, hold_controls, held_packets, held_requests = {}, set(), [], set()
    cancellation, suspended = None, False
    pressure_ids = []
    pressure = dict(requested_count=config.get('eco_probe_pressure_requests', 0), executed=False,
                    request_ids=pressure_ids, input_tokens=7168, output_tokens=512, reason='not_reached')

    def journal(kind, **fields):
        row = dict(kind=kind, at_s=time.time(), **fields) if 'at_s' not in fields else dict(kind=kind, **fields)
        rows.append(row)
        raw.write(json.dumps(row, allow_nan=False)+'\n'); raw.flush()
        if kind == 'eco_admission':
            hold_controls.update(control['instance_id'] for control in fields.get('controls', [])
                                 if control['send_output'] is False)
        # A pending packet outside receive() proves real buffering. A later
        # policy rotation may already have enabled output; pending packets are
        # still held until the next engine output actually triggers the flush.
        if kind in ('eco_engine_output', 'eco_admission', 'eco_probe_snapshot') and runtime is not None:
            for iid in hold_controls:
                buffer = runtime.controller.buffers[iid]
                packets = []
                for rid, event in buffer.pending:
                    if rid not in held_requests:
                        held_requests.add(rid)
                        packets.append(dict(request_id=rid, token_index=event.get('token_index'),
                                            token_ids=list(event.get('token_ids', []))))
                if packets:
                    observed_s = time.time()
                    held_instances.setdefault(iid, observed_s)
                    held_packets.extend(dict(packet, instance_id=iid, observed_s=observed_s) for packet in packets)
                    journal('eco_observed_output_hold', instance_id=iid,
                            request_ids=sorted({rid for rid, _ in buffer.pending}),
                            pending_packets=len(buffer.pending), packets=packets, send_output=buffer.send_output)

    def policy_checks():
        return dict(policy_rotation=any(row['kind'] == 'eco_admission' and row.get('controls') for row in rows),
            actual_hold=bool(held_instances),
            held_output_flush=any(row['kind'] == 'eco_client_sse' and row.get('request_id') == packet['request_id']
                and row['at_s'] >= packet['observed_s'] and row['payload'].get('token_index') == packet['token_index']
                and row['payload'].get('token_ids') == packet['token_ids'] for row in rows for packet in held_packets))

    def complete_output(rid, expected):
        client = outcomes[rid]['events']; native = transport.engine_outputs.get(rid, [])
        if (not outcomes[rid]['finished'] or not native or not native[-1].get('finished')
                or len(tokens(client)) != expected or tokens(client) != tokens(native)):
            raise RuntimeError('request lacks continuous complete native/client output: '+rid)
        outcomes[rid]['native_events'] = native
        outcomes[rid]['continuous_complete_output'] = True

    async def snapshot(label):
        states, received = await observe_states(transport, endpoints)
        journal('eco_probe_snapshot', stage=label, states=states,
                received_s_by_instance=received, atomic_fleet_snapshot=False)
        return states

    async def wait_live(rids, label, previous=None, minimum=1):
        deadline = time.monotonic()+phase_timeout
        while True:
            states = await snapshot(label); observations = {}
            for rid in rids:
                if tasks[rid].done() or outcomes[rid].get('finished'):
                    raise RuntimeError('request finished before live mechanism observation: '+rid)
                entry = runtime.controller.active.get(rid)
                if not entry or entry['engine_done']: continue
                iid = entry['instance_id']; state = states[iid]
                events = transport.engine_outputs.get(rid, [])
                count = observed_live_tokens(state, rid, events)
                if count is None or count < minimum: continue
                observation = dict(instance_id=iid, generation=state['generation'],
                    kv_blocks=kv_blocks(state, rid), native_tokens=count, observed_s=state['native_at_s'])
                if previous and not preserves(previous[rid], observation):
                    raise RuntimeError('live KV identity or block prefix changed: '+rid)
                if previous and count <= previous[rid]['native_tokens']: continue
                observations[rid] = observation
            if len(observations) == len(rids):
                snapshots[label] = observations
                journal('eco_live_kv_receipt', stage=label, requests=observations)
                return observations
            if time.monotonic() >= deadline:
                raise TimeoutError('real live request/KV evidence not observed: '+label)
            await asyncio.sleep(poll)

    def submit(rid, length, maximum):
        rng = random.Random(701+length)
        payload = dict(prompt=[rng.randint(1000, 60000) for _ in range(length)],
                       max_tokens=maximum, seed=701, temperature=0, ignore_eos=True)
        outcomes[rid] = dict(request_id=rid, payload=payload, events=[], finished=False, cancelled=False)
        journal('eco_probe_request', request_id=rid, payload=payload)
        async def consume():
            try:
                async for event in runtime.controller.handle(payload, rid):
                    outcomes[rid]['events'].append(event)
                    journal('eco_client_sse', request_id=rid, payload=event)
                events = outcomes[rid]['events']
                outcomes[rid]['finished'] = bool(events and events[-1].get('finished'))
            except asyncio.CancelledError:
                outcomes[rid]['cancelled'] = True
                raise
            except Exception as exc:
                outcomes[rid]['error'] = repr(exc)
                raise
        tasks[rid] = asyncio.create_task(consume())

    main_ids = ['eco-four-701-main-'+str(index) for index in range(3)]
    cancel_id = 'eco-four-701-cancel'
    phase_timeout, poll = 30., .02
    try:
        specs = config.get('instances', [])
        if len(specs) != 4 or len({row.get('id') for row in specs}) != 4 or set(endpoints) != {row['id'] for row in specs}:
            raise ValueError('exactly four distinct mapped instances required')
        if config.get('eco_initial_instances') != 3 or config.get('eco_macro_lower') != 2 or config.get('eco_macro_upper') != 3:
            raise ValueError('explicit initial=3 and macro bounds lower=2 upper=3 required')
        phase_timeout = float(config.get('eco_probe_phase_timeout_s', 30))
        poll = float(config.get('eco_state_poll_s', .02))
        maximum = config.get('eco_probe_output_tokens', 512)
        lengths = config.get('eco_probe_input_lengths', [128, 512, 2048])
        pressure_length = config.get('eco_probe_cancel_input_tokens', 7168)
        pressure_count = pressure['requested_count']
        if (not math.isfinite(phase_timeout) or not 0 < phase_timeout <= 120 or not math.isfinite(poll) or poll <= 0
                or type(maximum) is not int or not 16 <= maximum <= 512
                or type(pressure_count) is not int or not 0 <= pressure_count <= 12
                or not isinstance(lengths, list) or len(lengths) != 3
                or any(type(n) is not int or not 1 <= n <= 7168 for n in [*lengths, pressure_length])
                or max(*lengths, pressure_length)+maximum > 8192):
            raise ValueError('bounded fixed-work probe geometry and deadlines required')
        transport = ObservedTransport(endpoints, journal)
        runtime = EcoServeRuntime(config, transport, journal)
        await runtime.start()
        # Only the explicit membership primitive runs here. Policy and original
        # periods stay unchanged; the automatic resize task is visibly paused.
        scaling = [task for task in runtime.controller.tasks if task.get_coro().__name__ == '_scale_loop']
        if len(scaling) != 1: raise RuntimeError('cannot identify the manual probe background scaler')
        scaling[0].cancel(); await asyncio.gather(scaling[0], return_exceptions=True)
        suspended = True
        journal('eco_manual_probe_scaler_suspended', configured_period_s=runtime.controller.period,
                automatic_policy_triggered=False)
        for rid, length in zip(main_ids, lengths): submit(rid, length, maximum)
        await wait_live(main_ids, 'initial_live')
        # Real prefill pressure, with no forced route/control or invented state.
        submit(cancel_id, pressure_length, maximum)
        cancel_before = (await wait_live([cancel_id], 'cancel_inflight', minimum=2))[cancel_id]
        tasks[cancel_id].cancel()
        await asyncio.gather(tasks[cancel_id], return_exceptions=True)
        cancel_receipts = [row for row in rows if row['kind'] == 'eco_http_receipt'
            and row.get('path') == '/baseline/cancel' and row.get('body', {}).get('request_id') == cancel_id
            and row.get('response', {}).get('acknowledged') is True]
        after_cancel = await snapshot('cancel_released'); state = after_cancel[cancel_before['instance_id']]
        if (not cancel_receipts or not outcomes[cancel_id]['cancelled']
                or cancel_id in state['all_queue'] or cancel_id in state['kv_allocations']
                or state.get('pending_transfers') or state.get('transfer_allocations')
                or state['generation'] != cancel_before['generation']):
            raise RuntimeError('in-flight cancellation lacks released KV and generation ACK')
        cancellation = dict(request_id=cancel_id, before=cancel_before, after=state,
                            receipts=cancel_receipts, released=True)
        journal('eco_cancel_release', **cancellation)
        before = await wait_live(main_ids, 'before_split')
        fourth = specs[3]['id']
        if not await runtime.controller.add_member(fourth, trigger='manual_layout_action'):
            raise RuntimeError('idle fourth member add was not acknowledged')
        after_split = await wait_live(main_ids, 'after_split', previous=before)
        if not await runtime.controller.remove_member(fourth, trigger='manual_layout_action'):
            raise RuntimeError('fourth member removal was not acknowledged')
        await wait_live(main_ids, 'after_merge', previous=after_split)
        await asyncio.wait_for(asyncio.gather(*(tasks[rid] for rid in main_ids)), runtime.controller.request_timeout)
        for rid in main_ids:
            complete_output(rid, maximum)
        observed = policy_checks()
        pressure['before_checks'] = observed
        if all(observed.values()):
            pressure['reason'] = 'required_policy_evidence_already_observed'
        elif not pressure_count:
            pressure['reason'] = 'disabled'
        else:
            pressure.update(executed=True, reason='missing_policy_evidence')
            pressure_ids.extend('eco-four-701-pressure-'+str(index) for index in range(pressure_count))
            journal('eco_pressure_start', **pressure)
            for rid in pressure_ids: submit(rid, 7168, 512)
            await asyncio.wait_for(asyncio.gather(*(tasks[rid] for rid in pressure_ids)),
                                   runtime.controller.request_timeout)
            for rid in pressure_ids: complete_output(rid, 512)
            pressure['after_checks'] = policy_checks()
            journal('eco_pressure_completed', **pressure)
        if not pressure['executed']:
            journal('eco_pressure_skipped', **pressure)
        await runtime.controller.refresh()
        drained = await snapshot('final_drained')
        if any(state['all_queue'] or state['kv_allocations'] or state.get('pending_transfers')
               or state.get('transfer_allocations') or state['free_kv_tokens'] != state['total_kv_tokens']
               for state in drained.values()):
            raise RuntimeError('complete requests did not return all native KV blocks')
        if runtime.controller.failure: raise RuntimeError(runtime.controller.failure)
    except BaseException as exc:
        errors.append(repr(exc)); journal('eco_probe_error', error=repr(exc))
    finally:
        for task in tasks.values():
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        if runtime:
            try: await runtime.close()
            except Exception as exc:
                errors.append(repr(exc)); journal('eco_probe_cleanup_error', error=repr(exc))

    commits = [row for row in rows if row['kind'] == 'eco_membership_commit']
    checks = dict(startup=any(row['kind'] == 'eco_startup' for row in rows),
        **policy_checks(),
        split_commit=any(row.get('operation') == 'add' and row.get('split') is True for row in commits),
        merge_commit=any(row.get('operation') == 'remove' and row.get('merge') is True for row in commits),
        live_kv_continuity=all(stage in snapshots for stage in ('before_split', 'after_split', 'after_merge')),
        complete_output=all(outcomes.get(rid, {}).get('continuous_complete_output') is True for rid in main_ids),
        pressure_complete_output=all(outcomes.get(rid, {}).get('continuous_complete_output') is True for rid in pressure_ids),
        cancel_release=bool(cancellation), parking=any(row['kind'] == 'eco_park' for row in rows))
    missing = [name for name, passed in checks.items() if not passed]
    status = 'passed' if not missing and not errors else 'inconclusive'
    journal('eco_probe_completion', status=status, checks=checks, errors=errors); raw.close()
    artifact = dict(schema='ecoserve-four-instance-mechanism-v4', status=status, complete=status == 'passed',
        mechanism_validated=status == 'passed', formal_eligible=False, energy_comparable=False,
        complete_reproduction=False, forced_primitive=True, manualprimitive=True,
        manual_layout_action=True, automatic_policy_triggered=False,
        background_scaling_suspended_for_manual_probe=suspended,
        original_scale_period_s=config.get('eco_scale_period_s', 5), seed=701,
        missing_required_actions=missing, checks=checks, errors=errors, journal=rows,
        live_kv_snapshots=snapshots, requests=outcomes, cancellation=cancellation, pressure=pressure,
        endpoints=endpoints, profile_sha256=runtime.controller.profile.source_sha256 if runtime else None,
        raw_receipts_path=str(raw_path), raw_receipts_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        at_s=time.time())
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--endpoint', action='append', required=True, help='instance_id=http://host:port')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(run(json.loads(args.config.read_text()), dict(item.split('=', 1) for item in args.endpoint), args.out))
    print(json.dumps(dict(status=result['status'], missing_required_actions=result['missing_required_actions'], errors=result['errors'])))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__': raise SystemExit(main())
