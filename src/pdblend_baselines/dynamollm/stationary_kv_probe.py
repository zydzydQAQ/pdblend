"""Prepare/run a source-KV functional probe on an explicitly owned resident.

The supplied resident must run stationary_service, using the same immutable
source/image/UUID identity in identity_binding. probe_on_resident never loads a
model; the optional owned harness loads one source and one independent peer.
Neither path starts a sampler, changes a clock, changes a TP group, or grants
formal Dynamo qualification. The Fleet owner performs final process cleanup.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .stationary_ipc import need
from .stationary_tensors import tensor_plan


PROBE_PLAN = dict(schema='dynamo-source-kv-resident-probe-plan/v1', source_only=True,
    prompt_token_ids=list(range(100, 228)), output_tokens=16, seed=9701,
    ordinary_before_repeats=2, ordinary_after_repeats=1, same_tp=True,
    peer_requests_while_source_KV_released=2,
    public_mutation_rejection_during_KV_absence=True,
    measurements=['original_parameter_identity', 'source_allocator_KV_blocks', 'actual_driver_free_bytes',
                  'native_closed_admission_and_full_rank_drain', 'source_exact_token_goldens_after_rebuild'],
    does_not_qualify=['CUDA_IPC_consumer_binding', 'target_TP_group', 'missing_fragment_transport',
                     'peak_target_memory', 'TP_conversion', '1800_300_5_period_mechanism', 'formal_150s_comparison'])
IDENTITY = ('model_id', 'model_hash', 'tokenizer_hash', 'engine_revision', 'image_digest',
            'source_revision', 'gpu_uuids', 'tp', 'pp')
PREFIX = '/baseline/dynamollm/stationary/'


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def read_bound(ref):
    data = Path(ref['path']).read_bytes()
    need(hashlib.sha256(data).hexdigest() == ref['sha256'], 'resident identity binding bytes differ')
    return json.loads(data)


def preflight(config_path):
    config_path = Path(config_path).resolve()
    raw = config_path.read_bytes(); config = json.loads(raw)
    need(config['schema'] == 'dynamo-source-kv-owned-probe-config/v1' and config['plan'] == PROBE_PLAN,
         'source KV probe configuration differs')
    need(config['tp'] == config['pp'] == 1 and config['gpu_count'] == 2
         and config['model_id'] == 'Qwen2.5-7B-Instruct' and config['max_num_seqs'] == 32,
         'initial source-KV functional probe is one 7B TP1 native32 owner and one isolated peer')
    source = Path(config['source_snapshot']).resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    from .stationary_ipc import digest
    need(digest(manifest['files']) == manifest['source_sha256'] == config['source_sha256'], 'source hash differs')
    need(all(hashlib.sha256((source/name).read_bytes()).hexdigest() == sha for name,sha in manifest['files'].items()),
         'frozen source bytes differ')
    need(Path(__file__).resolve() == source/'pdblend_baselines/dynamollm/stationary_kv_probe.py',
         'probe imported outside its frozen source')
    need(os.environ.get('PDBLEND_IMAGE_ID') == config['image_digest']
         and os.environ.get('PDBLEND_SOURCE_SHA256') == config['source_sha256']
         and os.environ.get('PDBLEND_KV_PROBE_CONFIG_SHA256') == hashlib.sha256(raw).hexdigest(),
         'launch image/source/config binding differs')
    verified=read_bound(config['model_verification'])
    item=verified.get('models',{}).get('7b',{})
    need(verified.get('all_pass') is True and item.get('verified') is True
         and item.get('model_id')==config['model_id'] and item.get('model_path')==config['model_path']
         and all(any(f.get('kind')==kind and f.get('bytes',0)>0 and len(f.get('sha256',''))==64
             for f in item.get('files',[])) for kind in ('weight','tokenizer')),
         'bound full model verification does not cover the actual source/peer model')
    return config


async def run_owned(config_path, out, base_port):
    """Optional one-load Fleet harness; existing owners can call probe_on_resident."""
    from dataclasses import asdict
    import asyncio
    import signal
    import aiohttp
    from pdblend.engine.launcher import Fleet
    from pdblend_runtime.probe import NativeSpec, call
    from .stationary_probe import require_compute_empty
    config = preflight(config_path)
    uuids = os.environ.get('PDBLEND_GPU_UUIDS', '').split(',')
    need(len(uuids) == 2 and len(set(uuids)) == 2 and all(u.startswith('GPU-') for u in uuids),
         'two distinct real lease UUIDs required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1', 'owned probe expects two UUID-restricted container GPUs')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    # The shared launcher enables sleep by default. This private development
    # source-only profile explicitly disables it; frozen launches stay intact.
    class StationarySpec(NativeSpec):
        def command(self):
            args = super().command()
            args[args.index('pdblend_runtime.serve')] = 'pdblend_baselines.dynamollm.stationary_service'
            return [arg for arg in args if arg != '--enable-sleep-mode']
        def environment(self):
            value = super().environment()
            value['CUDA_VISIBLE_DEVICES'] = uuids[self.gpus[0]]
            return value
    specs = [StationarySpec('dynamo-stationary-'+name, (rank,), base_port+rank, config['model_path'],
        tp=1, pp=1, generation=0, max_num_seqs=32, kv_connector=None,
        extra_args=('--enforce-eager', '--worker-extension-cls',
            'pdblend_baselines.dynamollm.stationary_kv_worker.DynamoStationaryKvWorkerExtension'))
        for rank,name in enumerate(('source','peer'))]
    spec = specs[0]
    fleet = Fleet(specs, out/'logs')
    result = dict(schema='dynamo-source-kv-owned-probe/v1', status='failed', hardware_executed=False,
        source_sha256=config['source_sha256'], gpu_uuids=uuids,
        actual_launch=[dict(spec=asdict(s),argv=s.command()) for s in specs],
        model_loads=0, cleanup_errors=[], qualified=False, formal_eligible=False, full_tp_switch_qualified=False,
        original_dynamo_mechanism_qualified=False, energy_comparable=False, clocks_modified=False)
    process_groups = []
    try:
        result['gpu_before'] = [require_compute_empty(u, timeout=1) for u in uuids]
        write_new(out/'gpu-before.json', result['gpu_before'])
        need(all(r['passed'] for r in result['gpu_before']), 'leased GPU already has compute processes')
        for s in specs:
            fleet[s.instance_id].start()
            process_groups.append(fleet[s.instance_id].process.pid)
            result['model_loads'] += 1
        result['owned_process_groups'] = process_groups
        readiness = await asyncio.gather(*(asyncio.to_thread(fleet[s.instance_id].wait_ready, 900)
            for s in specs), return_exceptions=True)
        need(all(not isinstance(value, BaseException) for value in readiness),
             'private resident startup failed: '+repr(readiness))
        result['hardware_executed'] = True
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            bindings = []
            for rank,s in enumerate(specs):
                cap = await call(session, s.base_url, '/baseline/capability')
                need(cap['gpu_uuids'] == [uuids[rank]] and cap['model_id'] == config['model_id']
                     and cap['source_revision'] == config['source_sha256']
                     and cap['image_digest'] == config['image_digest'] and cap['tp'] == cap['pp'] == 1,
                     'actual private resident differs from immutable launch')
                identity = {k:cap[k] for k in IDENTITY}
                identity_path = out/f'resident-identity-{rank}.json'; write_new(identity_path, identity)
                binding = dict(path=str(identity_path.resolve()), sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest())
                write_new(out/f'resident-identity-binding-{rank}.json', binding);bindings.append(binding)
            result['probe'] = await probe_on_resident(session, spec.base_url, out/'probe', identity_binding=bindings[0],
                peer=dict(url=specs[1].base_url, identity_binding=bindings[1]))
            need(result['probe']['ready_for_next'], 'source KV probe did not safely restore and match ordinary output')
            need(result['probe']['peer_isolation_exercised'], 'independent peer did not serve while source KV was absent')
            from .native_state import validate_state
            started = time.time()
            result['peer_final_drain'] = await call(session, specs[1].base_url, '/baseline/drain', dict(timeout_s=30))
            validate_state(result['peer_final_drain'], generation=0, tp=1, pp=1,
                           drained=True, observed_after_s=started)
            need(result['peer_final_drain'].get('acknowledged') is True
                 and result['peer_final_drain'].get('drained') is True, 'peer final drain ACK incomplete')
            result['status'] = 'passed'
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        for s in specs:
            try:
                await asyncio.to_thread(fleet[s.instance_id].stop)
            except BaseException as error:
                result['cleanup_errors'].append(repr(error))
        # Server-parent death alone cannot prove its worker descendants gone.
        for process_group in process_groups:
            try:
                os.killpg(process_group, signal.SIGTERM)
                await asyncio.sleep(.2)
                try: os.killpg(process_group, signal.SIGKILL)
                except ProcessLookupError: pass
            except ProcessLookupError:
                pass
            except BaseException as error:
                result['cleanup_errors'].append('owned process-group cleanup: '+repr(error))
        try:
            result['gpu_after'] = [require_compute_empty(u, timeout=10) for u in uuids]
            need(all(r['passed'] for r in result['gpu_after']), 'owned lease still has compute processes after cleanup')
        except BaseException as error:
            result['cleanup_errors'].append('compute-empty verification: '+repr(error))
            result.setdefault('gpu_after', dict(passed=False, error=repr(error)))
        write_new(out/'gpu-after.json', result['gpu_after'])
        result['engine_events'] = fleet.events()
        result['actual_engine_starts'] = {s.instance_id: [e for e in fleet[s.instance_id].events
            if e['kind'] == 'start'] for s in specs}
        result['actual_model_ready_count'] = sum(e['kind'] == 'ready' for e in result['engine_events'])
        if result['cleanup_errors']: result['status'] = 'failed'
        result['complete'] = result['status'] == 'passed' and not result['cleanup_errors']
        write_new(out/'completion.json', result)
    return result


async def verify_public_fence(session, url, transaction):
    # A valid, harmless closed-admission control would return 200 without the
    # gateway. The private gateway must reject it before native mutation.
    started = time.time()
    async with session.post(url+'/baseline/control', json=dict(accepting=False)) as response:
        status, text = response.status, await response.text()
    body = json.loads(text)
    need(status == 409 and body.get('phase') == 'released'
         and body.get('transaction_id') == transaction,
         'source gateway did not fence a real public mutation while KV was absent')
    return dict(start_s=started,end_s=time.time(),http_status=status,body=body,
                endpoint='/baseline/control',payload=dict(accepting=False))


async def probe_on_resident(session, url, out, *, identity_binding, plan=None, peer=None):
    from pdblend_runtime.probe import call, generate
    from .native_state import validate_state
    plan = deepcopy(PROBE_PLAN if plan is None else plan)
    need(plan == PROBE_PLAN, 'resident KV probe differs from fixed non-evaluation design')
    expected = read_bound(identity_binding)
    need(all(k in expected for k in IDENTITY), 'complete frozen resident identity required')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='dynamo-source-kv-resident-probe/v1', status='failed', plan=plan,
        expected_identity=identity_binding, hardware_executed=False, observations=[],
        safe_restore_passed=False, ready_for_next=False, full_tp_switch_qualified=False,
        peer_isolation_exercised=False,
        peer_slo_qualified=False, peer_scope='output_and_epoch_continuity_during_source_KV_absence_only',
        original_dynamo_mechanism_qualified=False, formal_eligible=False, qualification='raw_evidence_only')
    transaction = 'source-kv-' + uuid.uuid4().hex
    common = None
    async def rpc(op, **extra):
        value = await call(session, url, PREFIX + op, dict(common or {}, **extra))
        write_new(out / f'{len(report["observations"]):02d}-{op}.json', value)
        report['observations'].append(dict(operation=op, value=value))
        return value
    async def ordinary(stage, repeat):
        value = await generate(session, url, dict(request_id=f'{transaction}-{stage}-{repeat}',
            prompt=plan['prompt_token_ids'], max_tokens=plan['output_tokens'], seed=plan['seed'], ignore_eos=True))
        need(len(value['token_ids']) == plan['output_tokens'], 'ordinary native token sequence is incomplete')
        write_new(out / f'ordinary-{stage}-{repeat}.json', value)
        return value['token_ids']
    async def finish_known_source_state():
        status = await rpc('status')
        if status['phase'] == 'pinned': status = await rpc('abort_pin')
        if status['phase'] == 'released': status = await rpc('restore_kv')
        if status['phase'] == 'restored': status = await rpc('close_kv_workspace')
        if status['phase'] == 'closed':
            for rank in range(expected['tp']):
                if rank not in status['released_owner_ranks']:
                    status = await rpc('release', source_rank=rank, consumer_processes_gone=[])
        if status['phase'] == 'owners_released': status = await rpc('resume')
        need(status['phase'] == 'idle', 'private resident remains quarantined or uncertain; isolate its owned processes')
        return status
    try:
        cap = await call(session, url, '/baseline/capability')
        need(all(cap.get(k) == expected[k] for k in IDENTITY) and cap.get('supported') is True,
             'actual resident identity differs from frozen Fleet binding')
        report['actual_capability'] = cap
        report['hardware_executed'] = True
        common = dict(transaction_id=transaction, expected_generation=cap['state']['generation'],
                      expected_gpu_uuids=cap['gpu_uuids'])
        description = await rpc('describe')
        rows = description['ranks']
        need(all(row['shapes'] == rows[0]['shapes'] and row['geometry'] == rows[0]['geometry'] for row in rows),
             'resident source rank parameter metadata differs')
        tensor = tensor_plan(source_gpus=cap['gpu_uuids'], target_gpus=cap['gpu_uuids'],
            source_shapes=rows[0]['shapes'], target_shapes=rows[0]['shapes'], geometry=rows[0]['geometry'])
        need(tensor['planned_transfer_bytes'] == 0, 'source KV probe cannot exercise a weight transfer')
        report['tensor_plan'] = tensor
        before = [await ordinary('before', repeat) for repeat in range(plan['ordinary_before_repeats'])]
        need(all(value == before[0] for value in before), 'ordinary source is not deterministic at the fixed seed')
        if peer is not None:
            peer_expected = read_bound(peer['identity_binding'])
            peer_cap = await call(session, peer['url'], '/baseline/capability')
            need(peer['url'] != url and all(peer_cap.get(k) == peer_expected[k] for k in IDENTITY)
                 and peer_cap.get('supported') is True and peer_cap['state']['accepting'] is True
                 and set(peer_cap['gpu_uuids']).isdisjoint(cap['gpu_uuids'])
                 and all(peer_cap[k] == cap[k] for k in IDENTITY if k not in ('gpu_uuids','tp','pp')),
                 'peer must be an independent actual same-model/source resident on disjoint physical GPUs')
            async def peer_ordinary(stage):
                value = await generate(session, peer['url'], dict(request_id=transaction+'-peer-'+stage,
                    prompt=plan['prompt_token_ids'], max_tokens=plan['output_tokens'], seed=plan['seed'], ignore_eos=True))
                need(len(value['token_ids']) == plan['output_tokens'], 'peer ordinary token sequence incomplete')
                write_new(out/('peer-'+stage+'.json'), value)
                return value['token_ids']
            peer_before = await peer_ordinary('before')
            report['peer_capability'] = peer_cap
        await rpc('pin', plan=tensor)
        released = await rpc('release_kv')
        need(released['phase'] == 'released', 'source KV release did not complete all ranks')
        report['public_mutation_rejection'] = await verify_public_fence(session, url, transaction)
        write_new(out/'public-mutation-rejection.json', report['public_mutation_rejection'])
        if peer is not None:
            peer_start = time.time()
            for repeat in range(plan['peer_requests_while_source_KV_released']):
                need(await peer_ordinary('source-KV-released-'+str(repeat)) == peer_before,
                     'peer ordinary tokens changed during source KV detachment')
            peer_state = await call(session, peer['url'], '/baseline/state')
            need(peer_state.get('accepting') is True and peer_state['generation'] == peer_cap['state']['generation'],
                 'source KV transaction changed peer admission or epoch')
            report.update(peer_isolation_exercised=True, peer_interval=dict(start_s=peer_start,end_s=time.time(),
                source_KV_phase='released', purpose='intentional_peer_functional_check_not_transition_cost_fit'),
                peer_native_state=peer_state)
        await finish_known_source_state()
        after = [await ordinary('after', repeat) for repeat in range(plan['ordinary_after_repeats'])]
        need(all(value == before[0] for value in after), 'source token sequence changed after exact KV rebuild')
        report.update(status='passed', exact_ordinary_tokens_match=True, source_kv_lifecycle_exercised=True)
    except BaseException as error:
        report['error'] = repr(error)
    finally:
        try:
            # Only known, fully ACKed phases permit rollback. A partial rank
            # failure is quarantined by the service and is never guessed here.
            if common is not None:
                report['restoration'] = await finish_known_source_state()
                started = time.time()
                final = await call(session, url, '/baseline/drain', dict(timeout_s=30))
                restored_tx = report['restoration'].get('transaction')
                generation = common['expected_generation'] + int(bool(restored_tx)
                    and restored_tx['transaction_id'] == common['transaction_id'])
                validate_state(final, generation=generation, tp=expected['tp'], pp=expected['pp'],
                               drained=True, observed_after_s=started)
                need(final.get('acknowledged') is True and final.get('accepting') is False,
                     'final source drain does not leave admission safely closed')
                report['final_native_drain'] = final
                report['safe_restore_passed'] = True
        except BaseException as error:
            report['restore_error'] = repr(error)
        report['ready_for_next'] = report['status'] == 'passed' and report['safe_restore_passed']
        report['next_stage_must_explicitly_resume_admission'] = True
        write_new(out/'completion.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resident-url')
    parser.add_argument('--peer-url')
    parser.add_argument('--peer-identity-binding', type=Path)
    parser.add_argument('--identity-binding', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--base-port', type=int)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        if args.config: preflight(args.config)
        print(json.dumps(dict(plan=PROBE_PLAN, imports_ready=True, hardware_executed=False,
            requires_existing_owned_private_resident=True, formal_eligible=False)))
        return
    if args.config:
        import asyncio
        need(args.out and args.base_port, 'owned probe output and leased base port required')
        need(asyncio.run(run_owned(args.config, args.out, args.base_port))['complete'], 'owned source KV probe failed')
        return
    need(args.resident_url and args.identity_binding and args.out, 'owned resident URL, identity binding and output required')
    import asyncio
    import aiohttp
    async def execute():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            return await probe_on_resident(session, args.resident_url, args.out,
                identity_binding=json.loads(args.identity_binding.read_text()),
                peer=dict(url=args.peer_url, identity_binding=json.loads(args.peer_identity_binding.read_text()))
                    if args.peer_url and args.peer_identity_binding else None)
    report = asyncio.run(execute())
    need(report['ready_for_next'] is True, 'source KV functional probe failed or did not safely restore')


if __name__ == '__main__':
    main()
