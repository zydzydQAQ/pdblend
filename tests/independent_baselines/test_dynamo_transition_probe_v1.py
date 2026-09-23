import json
import time
from types import SimpleNamespace

import pytest

from pdblend_baselines.dynamollm import transition_probe_v1 as probe
from pdblend_baselines.dynamollm.policy import PERIODS


@pytest.mark.parametrize('model,gpus,source_tp,target_tp', [
    ('Qwen2.5-7B-Instruct', [2, 3, 4], 1, 2),
    ('Qwen2.5-14B-Instruct', [0, 2, 5], 1, 2),
    ('Qwen2.5-32B-Instruct', [0, 1, 2, 3, 4, 5], 2, 4),
])
def test_probe_fits_minimal_disjoint_eight_gpu_placement(model, gpus, source_tp, target_tp):
    source, target = probe.placement(model, gpus, 19000)
    assert source['tp'] == source_tp and target['tp'] == target_tp
    assert not set(source['gpus']) & set(target['gpus'])
    assert source['gpus'] + target['gpus'] == gpus
    with pytest.raises(ValueError, match='exactly'):
        probe.placement(model, gpus[:-1], 19000)
    with pytest.raises(ValueError, match='port'):
        probe.placement(model, gpus, 65500)


def mock_hardware(monkeypatch, *, model='Qwen2.5-7B-Instruct', mismatch=None, transfer_failure=False):
    calls = []
    monkeypatch.setattr(probe, 'model_identity', lambda path: dict(model=model, manifest_sha256='a'*64))
    monkeypatch.delenv('PDBLEND_GPU_UUIDS', raising=False)
    monkeypatch.setenv('PDBLEND_IMAGE_ID', 'sha256:'+'b'*64)
    monkeypatch.setenv('PDBLEND_SOURCE_SHA256', 'c'*64)

    class Telemetry:
        def __init__(self, gpus, journal):
            self.uuids = {gpu: 'GPU-'+str(gpu) for gpu in gpus}
            self.journal = journal
            self.readings = [dict(gpu=gpu, gpu_uuid=self.uuids[gpu], power_w=100)
                             for gpu in gpus for _ in range(2)]
        def start(self):
            for row in self.readings:
                self.journal('dynamo_power', **row)
        def clock(self, *args):
            raise AssertionError('manual primitive must not pretend to execute ScaleFreq')
        async def close(self):
            calls.append(('telemetry-close',))

    class Transport:
        def __init__(self, instances, clock, journal):
            self.instances = {row['id']: row for row in instances}
            self.journal, self.gates = journal, {}
        async def start(self):
            pass
        async def close(self):
            calls.append(('transport-close',))
        async def state(self, iid):
            spec = self.instances[iid]
            return dict(timestamp=time.time(), generation=spec['generation'],
                acknowledged_generation=spec['generation'], active=0, running=0, waiting=0,
                kv_allocations={}, transfer_allocations={}, free_kv_tokens=1024,
                total_kv_tokens=1024, transport_healthy=True, evidence_complete=True,
                accepting=self.gates[iid])
        async def json(self, iid, path, payload=None, method='POST'):
            spec = self.instances[iid]
            payload = payload or {}
            tx = payload.get('transaction_id')
            op = path.rsplit('/', 1)[-1]
            calls.append((op, iid))
            ranks = [dict(rank=rank, generation=spec['generation'], ok=True,
                          transaction_id=tx) for rank in range(spec['tp'])]
            if op == 'capability':
                result = dict(model_id=model, engine_revision='vllm-0.10.1.1', tp=spec['tp'], pp=1,
                    supported=True, native_evidence_complete=True,
                    gpu_uuids=['GPU-'+str(gpu) for gpu in spec['gpus']], model_hash='d'*64,
                    tokenizer_hash='e'*64, verification_receipt_sha256='f'*64,
                    image_digest='sha256:'+'b'*64, source_revision='c'*64)
            elif op in ('quiesce', 'resume'):
                self.gates[iid] = op == 'resume'
                result = dict(accepting=self.gates[iid], generation=spec['generation'])
            elif op == 'drain':
                assert self.gates[iid] is False
                result = dict(drained=True, owner_ack=True,
                              ranks=[dict(row, drained=True) for row in ranks])
            elif op == 'describe':
                result = dict(ranks=[dict(row, parameters={'model.weight': [100]}, geometry={}) for row in ranks])
            elif op in ('open', 'close'):
                result = dict(ranks=ranks)
            elif op == 'transfer':
                if transfer_failure:
                    raise RuntimeError('injected NCCL failure')
                sending = iid == 'dynamo-probe-source'
                result = dict(ranks=[dict(row, sent_bytes=1024//spec['tp'] if sending else 0,
                    received_bytes=0 if sending else 1024//spec['tp'], source=sending,
                    target_complete=not sending, parameter_count=10, operation_id=tx) for row in ranks])
            elif op == 'verify':
                result = dict(verified=True, generation=spec['generation'], transaction_id=tx,
                    token_ids=list(range(16)), golden_source_sha256=spec['_golden']['source_sha256'])
            elif op == 'activate':
                self.gates[iid] = True
                result = dict(activated=True, generation=spec['generation'], transaction_id=tx,
                              ranks=[dict(row, weights_ready=True) for row in ranks])
            else:
                raise AssertionError(path)
            self.journal('dynamo_native_receipt', instance_id=iid, path=path,
                         transaction_id=tx, response=result)
            return result
        async def stream(self, iid, payload):
            assert self.gates[iid]
            assert payload['seed'] == 701 and payload['ignore_eos'] is True
            tokens = list(range(16))
            if mismatch and mismatch in payload['request_id']:
                tokens[0] += 100
            yield dict(token_ids=tokens, finished=True, request_id=payload['request_id'])

    class Lifecycle:
        def __init__(self, config, transport, journal, output):
            self.transport, self.journal = transport, journal
            self.instances = {}
            self.target_port = config['target_port']
        async def start(self, spec, dummy=False, golden=None):
            assert not set(spec['gpus']) & {gpu for row in self.instances.values() for gpu in row['gpus']}
            if dummy:
                assert golden and 'dynamo-probe-golden' not in self.instances
            calls.append(('start', spec['id'], dummy))
            row = dict(spec, _golden=golden)
            self.instances[spec['id']] = row
            self.transport.instances[spec['id']] = row
            self.transport.gates[spec['id']] = not dummy
            return dict(ready=True)
        async def stop(self, iid):
            calls.append(('stop', iid))
            self.instances.pop(iid, None)
            self.transport.instances.pop(iid, None)
            return dict(absent=True)
        async def close(self):
            calls.append(('lifecycle-close',))
            for iid in list(self.instances):
                await self.stop(iid)

    monkeypatch.setattr(probe, 'GroupTelemetry', Telemetry)
    monkeypatch.setattr(probe, 'V1Transport', Transport)
    monkeypatch.setattr(probe, 'SubprocessLifecycle', Lifecycle)
    return calls


def args(tmp_path, gpus=None):
    return SimpleNamespace(model=tmp_path/'model', gpus=gpus or [0, 1, 2], out=tmp_path/'result',
                           base_port=19000, startup_timeout=900., transition_timeout=900.)


@pytest.mark.asyncio
@pytest.mark.parametrize('model,gpus', [('Qwen2.5-7B-Instruct', [0, 1, 2]),
                                     ('Qwen2.5-32B-Instruct', [0, 1, 2, 3, 4, 5])])
async def test_probe_uses_real_coordinator_and_hooks_then_audits_all_native_receipts(tmp_path, monkeypatch, model, gpus):
    calls = mock_hardware(monkeypatch, model=model)
    result = await probe.execute(args(tmp_path, gpus))
    assert result['status'] == 'passed', result
    assert result['complete'] and result['own_cleanup_complete']
    assert result['transition_complete'] and not result['profile_target_requested']
    assert result['target_profile']['status'] == 'not_run'
    assert result['receipt_audit']['sent_bytes'] == result['receipt_audit']['received_bytes'] == 1024
    assert result['successful_output_checks'] == 2
    assert result['target']['generation'] == 1 and result['source']['generation'] == 0
    starts = [row for row in calls if row[0] == 'start']
    assert [row[2] for row in starts] == [False, False, True]
    assert calls.index(('stop', 'dynamo-probe-golden')) < calls.index(starts[-1])
    assert calls.index(('activate', result['target']['id'])) < calls.index(('stop', 'dynamo-probe-source'))
    assert calls.index(('stop', 'dynamo-probe-source')) < calls.index(('quiesce', result['target']['id']))
    for field in ('formal_eligible', 'energy_comparable', 'hardware_qualified', 'complete_reproduction',
                  'controller_hierarchy_qualified', 'original_weight_retention_implemented',
                  'independent_continuer_tested', 'live_kv_reshard_tested'):
        assert result[field] is False
    assert result['periods_s'] == dict(PERIODS) == {'ScaleInst': 1800., 'ScaleShard': 300., 'ScaleFreq': 5.}
    assert json.loads((tmp_path/'result'/'completion.json').read_text())['status'] == 'passed'
    with pytest.raises(FileExistsError):
        await probe.execute(args(tmp_path, gpus))


@pytest.mark.asyncio
async def test_unstable_same_tp_reference_aborts_before_any_dummy_or_transfer(tmp_path, monkeypatch):
    calls = mock_hardware(monkeypatch, mismatch='ordinary_repeat')
    result = await probe.execute(args(tmp_path))
    assert result['status'] == 'failed' and result['own_cleanup_complete']
    assert 'ordinary_repeat' in result['error']
    assert not any(row[0] == 'transfer' or row[0] == 'start' and row[2] for row in calls)
    assert ('stop', 'dynamo-probe-source') in calls and ('stop', 'dynamo-probe-golden') in calls


@pytest.mark.asyncio
async def test_nccl_failure_retires_partial_target_and_rebuilds_uncertain_source(tmp_path, monkeypatch):
    calls = mock_hardware(monkeypatch, transfer_failure=True)
    result = await probe.execute(args(tmp_path))
    assert result['status'] == 'failed' and result['own_cleanup_complete']
    assert 'NCCL' in result['error']
    assert not any(row[0] == 'activate' for row in calls)
    source_starts = [row for row in calls if row[:2] == ('start', 'dynamo-probe-source')]
    assert len(source_starts) == 2 and all(row[2] is False for row in source_starts)
    events = [json.loads(line) for line in (tmp_path/'result'/'events.jsonl').read_text().splitlines()]
    assert any(row['event'] == 'dynamo_gpu_recovery' for row in events)
    assert any(row['event'] == 'dynamo_transition' and row.get('source_restored') is True for row in events)


@pytest.mark.asyncio
async def test_receipt_audit_rejects_missing_rank_even_after_coordinator_complete(tmp_path, monkeypatch):
    mock_hardware(monkeypatch)
    result = await probe.execute(args(tmp_path))
    events = [json.loads(line) for line in (tmp_path/'result'/'events.jsonl').read_text().splitlines()]
    transition = probe.Transition(**json.loads((tmp_path/'result'/'transition.json').read_text())['transition'])
    golden = json.loads((tmp_path/'result'/'goldens.json').read_text())['2']
    for row in events:
        if row['event'] == 'dynamo_native_receipt' and row.get('path') == '/baseline/dynamollm/activate':
            row['response']['ranks'].pop()
    audit = probe.audit_receipts(events, transaction=transition, source=result['source'],
                                 target=result['target'], golden=golden)
    assert not audit['passed'] and 'rank/generation' in audit['failures'][0]


def collector_result(options, *, passed=True, missing_frequency=False):
    points = []
    for frequency in probe.FREQUENCIES[:-1] if missing_frequency else probe.FREQUENCIES:
        point = dict(frequency_mhz=frequency, input_tokens=512, output_tokens=64, batch=1)
        probe.save(options.out/'cells'/f'{frequency}.json', dict(point=point,
            fit=dict(holdout_passed=passed, holdout_errors=dict(ttft=.01 if passed else .2))))
        points.append(dict(role='mixed', pp=1, tp=options.tp, frequency_mhz=frequency,
                           input_tokens=512, context_tokens=576, batch=1))
    value = dict(system='dynamollm', measurement='hardware', independent_profile=True,
                 points=points, coverage=dict(all_six_frequencies=not missing_frequency))
    probe.save(options.out/'profile.json', value)
    probe.save(options.out/'completion.json', dict(status='passed' if passed else 'inconclusive',
        complete=passed, formal_eligible=False, energy_comparable=False,
        missing_points=[] if passed else [dict(reason='holdout_error')]))
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize('model,gpus,target_tp', [('Qwen2.5-7B-Instruct', [0, 1, 2], 2),
    ('Qwen2.5-14B-Instruct', [0, 1, 2], 2), ('Qwen2.5-32B-Instruct', list(range(6)), 4)])
async def test_optional_profile_reuses_activated_target_only_after_retire_and_audit(tmp_path, monkeypatch, model, gpus, target_tp):
    calls = mock_hardware(monkeypatch, model=model)
    work = args(tmp_path, gpus); work.profile_target = True
    async def collect(options):
        iid = options.instance_id
        assert options.existing_url == 'http://127.0.0.1:19032'
        assert options.tp == target_tp and options.gpus == gpus[-target_tp:]
        assert options.model == work.model and options.base_port == 19032
        assert options.freqs == list(probe.FREQUENCIES)
        assert (options.inputs, options.outputs, options.batches) == ([512], [64], [1])
        assert options.settle == 2 and options.measure == 5 and options.resume
        assert options.label_corpus_root is None
        assert ('stop', 'dynamo-probe-source') in calls and ('close', iid) in calls
        assert calls[-1] == ('resume', iid)
        assert json.loads((work.out/'receipt-audit.json').read_text())['passed']
        events = [json.loads(line) for line in (work.out/'events.jsonl').read_text().splitlines()]
        assert any(row['event'] == 'dynamo_probe_output' and row['stage'] == 'after_activate'
                   and row['ok'] for row in events)
        calls.append(('collect', iid))
        return collector_result(options)
    monkeypatch.setattr(probe, 'collect_profile', collect)
    result = await probe.execute(work)
    assert result['status'] == 'passed' and result['transition_complete']
    profile = result['target_profile']
    assert profile['status'] == 'passed' and profile['complete'] and profile['generation'] == 1
    assert len(profile['holdout']) == 6 and all(row['passed'] for row in profile['holdout'])
    assert not profile['formal_eligible'] and not profile['energy_comparable']
    assert result['target_profile_final_drain']
    assert 'target-profile-summary.json' in result['artifacts']
    iid = result['target']['id']
    tail = calls[calls.index(('collect', iid))+1:]
    assert tail[:2] == [('quiesce', iid), ('drain', iid)]
    assert not any(row[0] in ('start', 'open', 'transfer', 'close') for row in tail)
    assert len([row for row in calls if row[0] == 'start']) == 3
    assert result['periods_s'] == dict(PERIODS)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['holdout', 'missing_frequency', 'exception'])
async def test_optional_profile_failure_preserves_verified_transition_and_still_drains(tmp_path, monkeypatch, failure):
    calls = mock_hardware(monkeypatch)
    work = args(tmp_path); work.profile_target = True
    async def collect(options):
        calls.append(('collect', options.instance_id))
        if failure == 'exception':
            probe.save(options.out/'completion.json', dict(status='inconclusive', complete=False,
                                                          missing_points=['measurement_failed']))
            raise RuntimeError('independent target measurement failed')
        return collector_result(options, passed=failure != 'holdout', missing_frequency=failure == 'missing_frequency')
    monkeypatch.setattr(probe, 'collect_profile', collect)
    result = await probe.execute(work)
    assert result['status'] == 'passed' and result['complete'] and result['transition_complete']
    assert result['receipt_audit']['passed'] and result['successful_output_checks'] == 2
    assert not result['target_profile']['complete'] and result['target_profile']['errors']
    assert result['target_profile']['status'] in ('failed', 'inconclusive')
    assert result['target_profile_final_drain'] and result['own_cleanup_complete']
    assert ('drain', result['target']['id']) in calls[calls.index(('collect', result['target']['id'])):]
    if failure == 'exception':
        assert not result['target_profile']['collector_completion']['complete']


@pytest.mark.asyncio
async def test_profile_final_native_drain_failure_still_fails_job_without_erasing_transition(tmp_path, monkeypatch):
    calls = mock_hardware(monkeypatch)
    work = args(tmp_path); work.profile_target = True
    collected = []
    original_drain = probe.GpuTopologyHooks._drain
    async def collect(options):
        collected.append(options.instance_id)
        return collector_result(options)
    async def drain(self, iid):
        if iid in collected:
            raise RuntimeError('native target still has pending KV')
        return await original_drain(self, iid)
    monkeypatch.setattr(probe, 'collect_profile', collect)
    monkeypatch.setattr(probe.GpuTopologyHooks, '_drain', drain)
    result = await probe.execute(work)
    assert result['status'] == 'failed' and not result['complete']
    assert result['transition_complete'] and result['receipt_audit']['passed']
    assert 'pending KV' in result['error'] and result['own_cleanup_complete']
    assert ('stop', result['target']['id']) in calls


@pytest.mark.asyncio
async def test_profile_clock_cleanup_error_is_fatal_even_when_final_native_drain_succeeds(tmp_path, monkeypatch):
    mock_hardware(monkeypatch)
    work = args(tmp_path); work.profile_target = True
    async def collect(options):
        value = collector_result(options)
        probe.save(options.out/'completion.json', dict(status='inconclusive', complete=False,
            cleanup_errors=[dict(component='telemetry', error='owned clock reset failed')]))
        return value
    monkeypatch.setattr(probe, 'collect_profile', collect)
    result = await probe.execute(work)
    assert result['status'] == 'failed' and not result['complete']
    assert result['transition_complete'] and result['target_profile_final_drain']
    assert 'clock reset failed' in result['error']
