"""Real, lease-scoped Dynamo weight-transition primitive on vLLM V1.

This explicitly triggered probe needs neither a performance table nor an output
predictor. It verifies the execution mechanism, not the controller's policy or
its original 1800/300/5-second hierarchy. A disk-loaded target first supplies its
own same-TP golden; that process is retired before the dummy target is created.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from .deployment import SubprocessLifecycle, gpu_devices, save, sha
from .gpu_topology import GpuTopologyHooks
from .policy import PERIODS
from .predictor import model_identity
from .profile_v1 import FREQUENCIES, collect as collect_profile, collect_golden, integers
from .reconfiguration import Reconfiguration, Transition
from .run_v1 import Journal
from .telemetry import GroupTelemetry
from .transport import V1Transport, validate_rank_ack
from .validation import MODELS


def placement(model_id, gpus, base_port):
    if model_id not in MODELS:
        raise ValueError('explicit supported Qwen2.5 model required')
    source_tp, target_tp = (2, 4) if model_id == 'Qwen2.5-32B-Instruct' else (1, 2)
    if (len(gpus) != source_tp + target_tp or len(set(gpus)) != len(gpus)
            or any(type(gpu) is not int or not 0 <= gpu < 8 for gpu in gpus)):
        raise ValueError(f'transition probe needs exactly {source_tp + target_tp} unique lease-local GPUs')
    if type(base_port) is not int or not 1024 <= base_port <= 65000:
        raise ValueError('base port must leave room for target and NCCL store ports')

    def instance(iid, group, port):
        return dict(id=iid, instance_id=iid, gpus=list(group), tp=len(group), pp=1,
                    port=port, url='http://127.0.0.1:' + str(port), generation=0, role='mixed')
    return (instance('dynamo-probe-source', gpus[:source_tp], base_port),
            instance('dynamo-probe-golden', gpus[source_tp:], base_port + 16))


def check_capability(value, *, model_id, tp, uuids, reference=None):
    if (value.get('model_id') != model_id or value.get('engine_revision') != 'vllm-0.10.1.1'
            or value.get('tp') != tp or value.get('pp') != 1
            or value.get('supported') is not True or value.get('native_evidence_complete') is not True
            or value.get('gpu_uuids') != uuids):
        raise RuntimeError('model/topology/UUID native capability mismatch')
    for key in ('model_hash', 'tokenizer_hash', 'verification_receipt_sha256'):
        if not isinstance(value.get(key), str) or len(value[key]) != 64:
            raise RuntimeError('missing native model provenance: ' + key)
    if not value.get('image_digest') or not value.get('source_revision'):
        raise RuntimeError('native image/source identity missing')
    if reference:
        for key in ('model_hash', 'tokenizer_hash', 'image_digest', 'source_revision',
                    'verification_receipt_sha256'):
            if value[key] != reference[key]:
                raise RuntimeError('source/target execution identity differs: ' + key)


async def check_output(transport, iid, golden, journal, *, stage):
    payload = dict(prompt=golden['prompt'], max_tokens=len(golden['token_ids']), seed=701,
                   request_id='dynamo-probe-' + stage + '-' + str(time.time_ns()),
                   temperature=0, ignore_eos=True)
    tokens, terminal = [], False
    async for row in transport.stream(iid, payload):
        journal('dynamo_probe_sse', instance_id=iid, stage=stage, payload=row)
        tokens.extend(row.get('token_ids', []))
        terminal = bool(row.get('finished'))
    outcome = dict(instance_id=iid, stage=stage, request_id=payload['request_id'], seed=701,
                   token_ids=tokens, finished=terminal, golden_source_sha256=golden['source_sha256'],
                   ok=terminal and tokens == golden['token_ids'])
    journal('dynamo_probe_output', **outcome)
    if not outcome['ok']:
        raise RuntimeError('same-target-TP output differs at ' + stage)
    return outcome


def audit_receipts(rows, *, transaction, source, target, golden):
    """Require actual native ACKs, not merely a CPU coordinator's completion."""
    tx = transaction.transaction_id
    errors = []
    receipts = [row for row in rows if row['event'] == 'dynamo_native_receipt'
                and row.get('transaction_id') == tx]
    phases = ('freeze_s', 'drain_s', 'release_s', 'prepare_s', 'verify_s', 'activate_s', 'retire_s')
    done = [row for row in rows if row['event'] == 'dynamo_transition'
            and row.get('transition', {}).get('transaction_id') == tx and row.get('phase') == 'complete']
    if len(done) != 1 or not set(phases) <= set(done[0].get('phases', {})):
        errors.append('complete_native_transition_phases_missing')

    def response(iid, operation):
        values = [row['response'] for row in receipts if row['instance_id'] == iid
                  and row['path'] == '/baseline/dynamollm/' + operation]
        if len(values) != 1:
            raise RuntimeError('one native receipt required for ' + iid + ':' + operation)
        return values[0]

    sent = received = 0
    try:
        for spec in (source, target):
            for operation in ('open', 'transfer', 'close'):
                validate_rank_ack(response(spec['id'], operation), spec, transaction_id=tx)
        sent_rows = response(source['id'], 'transfer')['ranks']
        received_rows = response(target['id'], 'transfer')['ranks']
        sent = sum(row['sent_bytes'] for row in sent_rows)
        received = sum(row['received_bytes'] for row in received_rows)
        if sent <= 0 or sent != received or not all(row.get('target_complete') is True
                and row.get('received_bytes', 0) > 0 and row.get('parameter_count', 0) > 0
                and row.get('operation_id') == tx for row in received_rows):
            raise RuntimeError('positive conserved complete target weight transfer missing')
        verified = response(target['id'], 'verify')
        if (verified.get('verified') is not True or verified.get('generation') != target['generation']
                or verified.get('transaction_id') != tx or verified.get('token_ids') != golden['token_ids']
                or verified.get('golden_source_sha256') != golden['source_sha256']):
            raise RuntimeError('same-TP private golden/generation receipt missing')
        activated = response(target['id'], 'activate')
        validate_rank_ack(activated, target, transaction_id=tx)
        if (activated.get('activated') is not True or activated.get('generation') != target['generation']
                or not all(row.get('weights_ready') is True for row in activated['ranks'])):
            raise RuntimeError('target activation ready acknowledgement missing')
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(str(exc))
    if not any(row['event'] == 'dynamo_gpu_instance_absent' and row.get('instance_id') == source['id']
               and row.get('ack', {}).get('absent') is True for row in rows):
        errors.append('source_retirement_absence_unconfirmed')
    for spec in (source, target):
        if not any(row['event'] == 'dynamo_gpu_drained' and row.get('instance_id') == spec['id']
                   and row.get('ack', {}).get('owner_ack') is True for row in rows):
            errors.append('native_empty_drain_missing:' + spec['id'])
    return dict(passed=not errors, failures=errors, sent_bytes=sent, received_bytes=received,
                transaction_id=tx, all_rank_acknowledgements=not errors)


async def profile_target(args, target, journal):
    """Own sparse profile of an activated resident target; no lifecycle calls."""
    output = args.out / 'target-profile'
    options = SimpleNamespace(model=args.model, gpus=list(target['gpus']), tp=target['tp'],
        base_port=target['port'], existing_url=target['url'], instance_id=target['id'],
        out=output, freqs=list(FREQUENCIES), inputs=[512], outputs=[64], batches=[1],
        settle=2., measure=5., resume=True, label_corpus_root=None, label_samples=32)
    result = dict(requested=True, status='inconclusive', complete=False,
        instance_id=target['id'], generation=target['generation'], tp=target['tp'], pp=1,
        gpus=list(target['gpus']), existing_url=target['url'], resident_reused=True,
        formal_eligible=False, energy_comparable=False, hardware_qualified=False,
        envelope=dict(frequencies=list(FREQUENCIES), input_tokens=512, output_tokens=64,
                      batch=1, settle_s=2., measure_s=5., repeats=3, independent_holdout=True),
        artifacts={}, holdout=[], errors=[])
    journal('dynamo_target_profile_started', **result)
    journal.checkpoint()
    try:
        profile = await collect_profile(options)
        completion = json.loads((output/'completion.json').read_text())
        result.update(collector_completion=completion, coverage=profile['coverage'])
        for cell_path in sorted((output/'cells').glob('*.json')):
            cell = json.loads(cell_path.read_text())
            result['holdout'].append(dict(path=str(cell_path), sha256=sha(cell_path),
                point=cell['point'], passed=cell['fit']['holdout_passed'],
                errors=cell['fit']['holdout_errors']))
        points = profile['points']
        covered = {(point['frequency_mhz'], point['input_tokens'], point['context_tokens'],
                    point['batch'], point['tp'], point['pp'], point['role']) for point in points}
        expected = {(frequency, 512, 576, 1, target['tp'], 1, 'mixed') for frequency in FREQUENCIES}
        qualified = (completion.get('status') == 'passed' and completion.get('complete') is True
            and profile.get('system') == 'dynamollm' and profile.get('measurement') == 'hardware'
            and profile.get('independent_profile') is True and covered == expected
            and len(result['holdout']) == 6 and all(row['passed'] for row in result['holdout']))
        result.update(status='passed' if qualified else 'inconclusive', complete=qualified)
        if not qualified:
            result['errors'].append('independent six-frequency coverage/holdout incomplete')
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        result['errors'].append(repr(exc))
    finally:
        for name in ('profile.json', 'completion.json', 'failures.json'):
            path = output/name
            if path.is_file():
                result['artifacts'][name] = dict(path=str(path), sha256=sha(path))
                if name == 'completion.json' and 'collector_completion' not in result:
                    try:
                        result['collector_completion'] = json.loads(path.read_text())
                    except (ValueError, OSError) as exc:
                        result['errors'].append('collector completion unreadable: '+repr(exc))
        save(args.out/'target-profile-summary.json', result)
        journal('dynamo_target_profile_completed', **result)
    return result


async def execute(args):
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ('completion.json','events.jsonl','events.jsonl.gz')):
        raise FileExistsError('refusing to overwrite Dynamo transition artifacts')
    journal = Journal(output / 'events.jsonl')
    telemetry = transport = lifecycle = transitions = None
    source = target = identity = golden = loaded = None
    audit, outcomes, cleanup, failure = {}, [], [], None
    result = dict(system='dynamollm', seed=701, scope='explicit_weight_transition_probe',
                  formal_eligible=False, energy_comparable=False, hardware_qualified=False,
                  complete_reproduction=False, controller_hierarchy_qualified=False,
                  periods_s=dict(PERIODS), pp=1, stationary_weight_bytes=0,
                  original_weight_retention_implemented=False,
                  independent_continuer_tested=False, live_kv_reshard_tested=False,
                  profile_target_requested=bool(getattr(args, 'profile_target', False)),
                  target_profile=dict(requested=bool(getattr(args, 'profile_target', False)),
                                      status='not_run', complete=False))
    started = time.time()
    try:
        identity = model_identity(args.model)
        source, reference = placement(identity['model'], args.gpus, args.base_port)
        gpu_devices(args.gpus)
        result.update(model_id=identity['model'], source_tp=source['tp'], target_tp=reference['tp'])
        config = dict(model_id=identity['model'], model_path=str(args.model), legal_tp=list(MODELS[identity['model']]),
                      node_gpus=args.gpus, base_port=args.base_port, target_port=args.base_port + 32,
                      startup_timeout_s=args.startup_timeout, max_model_len=8192)
        save(output / 'config.json', dict(config, source=source, golden_instance=reference,
                                         model_identity=identity, periods_s=dict(PERIODS)))
        telemetry = GroupTelemetry(args.gpus, journal)
        telemetry.start()
        transport = V1Transport([], telemetry.clock, journal)
        await transport.start()
        lifecycle = SubprocessLifecycle(config, transport, journal, output / 'engines')
        # Loads are deliberately sequential. The complete source remains alive
        # while the normal target is replaced with the actual dummy recipient.
        await lifecycle.start(source)
        source_cap = await transport.json(source['id'], '/baseline/capability', method='GET')
        check_capability(source_cap, model_id=identity['model'], tp=source['tp'],
                         uuids=[telemetry.uuids[gpu] for gpu in source['gpus']])
        await lifecycle.start(reference)
        target_cap = await transport.json(reference['id'], '/baseline/capability', method='GET')
        check_capability(target_cap, model_id=identity['model'], tp=reference['tp'],
                         uuids=[telemetry.uuids[gpu] for gpu in reference['gpus']], reference=source_cap)
        save(output / 'capabilities.json', dict(source=source_cap, normal_target=target_cap))
        golden = await collect_golden(transport, reference['id'], output=output,
                                      capability=target_cap, tp=reference['tp'])
        # A second ordinary request rejects an unstable reference before any
        # transferred weights can be judged against it.
        outcomes.append(await check_output(transport, reference['id'], golden, journal, stage='ordinary_repeat'))
        hooks = GpuTopologyHooks(transport, lifecycle, journal, goldens={reference['tp']: golden},
                                 store_port=args.base_port + 80, timeout_s=args.transition_timeout)
        await hooks._gate(reference['id'], False)
        await hooks._drain(reference['id'])
        await hooks._stop_confirmed(reference['id'])
        journal('dynamo_probe_reference_retired', instance_id=reference['id'],
                golden_source_sha256=golden['source_sha256'])
        transitions = Reconfiguration(hooks, journal)
        transition = Transition('probe-701-' + str(time.time_ns()), (source['id'],), (tuple(source['gpus']),),
                                (tuple(reference['gpus']),), overlap_memory_qualified=False,
                                timeout_s=args.transition_timeout, target_shapes=('LL',))
        if getattr(args, 'drain_batch', 0):
            from .loaded_drain import LoadedDrain
            result['loaded_drain_clock'] = telemetry.clock(args.gpus,2520)
            loaded = LoadedDrain(transport,source['id'],journal,input_tokens=args.drain_input,
                                 output_tokens=args.drain_output,batch=args.drain_batch)
            await loaded.start()

        async def commit(prepared):
            nonlocal target
            if len(prepared['instances']) != 1:
                raise RuntimeError('primitive requires exactly one acknowledged target')
            target = dict(prepared['instances'][0])
            expected = tuple(reference['gpus'])
            if (tuple(target['gpus']) != expected or target['tp'] != reference['tp']
                    or target['generation'] != source['generation'] + 1):
                raise RuntimeError('committed target layout/generation differs')
            journal('dynamo_probe_commit', transaction_id=transition.transaction_id, target=target)

        receipt = await transitions.execute(transition, commit=commit)
        save(output / 'transition.json', receipt)
        if loaded:
            result['loaded_drain'] = await loaded.finish()
            save(output/'loaded-drain.json',result['loaded_drain'])
        if source['id'] in lifecycle.instances or source['id'] in transport.instances:
            raise RuntimeError('retired source remains in live placement')
        transferred_cap = await transport.json(target['id'], '/baseline/capability', method='GET')
        check_capability(transferred_cap, model_id=identity['model'], tp=target['tp'],
                         uuids=[telemetry.uuids[gpu] for gpu in target['gpus']], reference=target_cap)
        save(output / 'transferred-capability.json', transferred_cap)
        outcomes.append(await check_output(transport, target['id'], golden, journal, stage='after_activate'))
        await hooks._gate(target['id'], False)
        await hooks._drain(target['id'])
        audit = audit_receipts(journal.rows, transaction=transition, source=source, target=target, golden=golden)
        save(output / 'receipt-audit.json', audit)
        if not audit['passed']:
            raise RuntimeError('native transition receipt audit failed: ' + repr(audit['failures']))
        if result['profile_target_requested']:
            # NCCL is already closed, source retired and transaction audited.
            # Only the existing target's public admission is reopened. Profile
            # qualification is separate from the proved transition primitive.
            try:
                await hooks._gate(target['id'], True)
                journal.checkpoint()
                result['target_profile'] = await profile_target(args, target, journal)
                profile_cleanup = result['target_profile'].get('collector_completion', {}).get('cleanup_errors', [])
                if profile_cleanup:
                    raise RuntimeError('resident target profiler cleanup failed: '+repr(profile_cleanup))
            finally:
                await hooks._gate(target['id'], False)
                await hooks._drain(target['id'])
                result['target_profile_final_drain'] = True
    except BaseException as exc:
        failure = repr(exc)
        journal('dynamo_probe_failure', error=failure)
    finally:
        if loaded is not None:
            try:await loaded.close()
            except BaseException as exc:cleanup.append(dict(component='loaded_drain',error=repr(exc)))
        for name, resource in [('lifecycle', lifecycle), ('telemetry', telemetry), ('transport', transport)]:
            if resource is not None:
                try:
                    await resource.close()
                except BaseException as exc:
                    cleanup.append(dict(component=name, error=repr(exc)))
        counts = Counter(row['gpu'] for row in telemetry.readings if 'error' not in row) if telemetry else Counter()
        power_valid = bool(telemetry) and all(counts[gpu] >= 2 for gpu in args.gpus)
        passed = not failure and not cleanup and audit.get('passed') is True and power_valid
        result.update(status='passed' if passed else 'failed' if failure or cleanup else 'inconclusive',
                      complete=passed, error=failure, cleanup_errors=cleanup,
                      own_cleanup_complete=not cleanup, started_s=started, finished_s=time.time(),
                      receipt_audit=audit, successful_output_checks=sum(row['ok'] for row in outcomes),
                      transition_complete=audit.get('passed') is True and sum(row['ok'] for row in outcomes) == 2,
                      gpu_uuids=telemetry.uuids if telemetry else {},
                      group_power_samples=journal.power_samples, group_power_errors=journal.power_errors,
                      per_gpu_power_samples=dict(counts), group_power_covered=power_valid,
                      source=source, target=target, model_identity=identity,
                      source_sha256=os.environ.get('PDBLEND_SOURCE_SHA256'),
                      image_digest=os.environ.get('PDBLEND_IMAGE_ID'),
                      quarantined_gpus=sorted(transitions.quarantined) if transitions else [],
                      golden_source_sha256=golden['source_sha256'] if golden else None)
        journal.close()
        save(output / 'outcomes.json', outcomes)
        result['journal_path'] = journal.path.name
        result['artifacts'] = {name: sha(output / name) for name in (journal.path.name, 'outcomes.json',
            'config.json', 'capabilities.json', 'transferred-capability.json', 'golden-raw.json',
            'goldens.json', 'transition.json', 'receipt-audit.json', 'target-profile-summary.json',
            'loaded-drain.json') if (output / name).is_file()}
        save(output / 'completion.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpus', type=integers, required=True, help='lease-local GPU indices, comma separated')
    parser.add_argument('--base-port', type=int, default=19000)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--startup-timeout', type=float, default=900)
    parser.add_argument('--transition-timeout', type=float, default=900)
    parser.add_argument('--profile-target', action='store_true',
                        help='reuse activated target for independent six-frequency 512/64/B1 profiling')
    parser.add_argument('--drain-batch',type=int,default=0,
                        help='real source work before quiesce; zero preserves empty-KV primitive')
    parser.add_argument('--drain-input',type=int,default=2048)
    parser.add_argument('--drain-output',type=int,default=512)
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value <= 0 for value in (args.startup_timeout, args.transition_timeout)):
        parser.error('positive finite stage deadlines required')
    if args.drain_batch:
        from .loaded_drain import validate
        validate(args.drain_input,args.drain_output,args.drain_batch)
    result = asyncio.run(execute(args))
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
