from copy import deepcopy
import asyncio
import json

import pytest

from pdblend.profile.collection.native_handoff import (
    collection_plan, endpoint_intervals, proxy_intervals, training_candidate,
    collect_endpoint_pair,
)
from pdblend.engine.client import Completion
from pdblend.profile.collection.native_runtime_collect import write_new
from pdblend.profile.collection.native_timing_plan import binding
from pdblend.profile.collection import native_handoff as handoff


def payload(n=2):
    return dict(prefill=dict(request_id='r', instance_id='p', completion_tokens=1,
        error=None, usage_received=True, stream_done=False, submitted_s=10.,
        first_token_s=11., finished_s=11., token_times_s=[11.], token_ids=[101]),
        combined=dict(request_id='r', instance_id='d', completion_tokens=n,
            decode_completion_tokens=n-1, pd_protocol='carry_first_token', error=None,
            usage_received=True, stream_done=True, submitted_s=10., first_token_s=11.,
            decode_submitted_s=11.02, decode_first_token_s=11.12,
            finished_s=11.12+(n-2)*.01,
            token_times_s=[11., *[11.12+i*.01 for i in range(n-1)]],
            token_ids=list(range(101, 101+n))))


def test_direct_endpoint_gap_is_nonnegative_without_ordinary_subtraction():
    value = payload()
    value['ordinary'] = [{'signed_residual': -1000}]
    result = endpoint_intervals(value)
    assert result['first_to_second_output_s'] == pytest.approx(.12)
    assert result['client_dispatch_s'] == pytest.approx(.02)
    assert result['decode_submit_to_first_s'] == pytest.approx(.1)
    assert not result['physical_copy_time']
    assert result['includes_http_scheduling_kv_and_first_decode']


@pytest.mark.parametrize('field,value', [
    ('decode_submitted_s', 10.9), ('decode_first_token_s', 11.01),
    ('finished_s', 11.11), ('decode_submitted_s', None), ('decode_first_token_s', float('nan')),
    ('decode_completion_tokens', 2), ('token_times_s', [11., 11.2]),
    ('token_ids', [1, 2]), ('request_id', 'foreign'), ('error', 'timeout'),
    ('stream_done', False), ('instance_id', 'p'),
])
def test_partial_reversed_or_misaligned_payload_is_not_clipped_into_a_measurement(field, value):
    row = payload()
    row['combined'][field] = value
    with pytest.raises(ValueError):
        endpoint_intervals(row)


def test_proxy_convenience_zero_cannot_hide_reversed_raw_clock():
    record = dict(path='PD', first_token_s=11., pd_handoff_started_s=11.,
                  first_decode_token_s=10.9, route_estimate={'observed_handoff_s': 0.})
    with pytest.raises(ValueError, match='reversed'):
        proxy_intervals(record)
    record['first_decode_token_s'] = 11.12
    assert proxy_intervals(record)['first_to_second_output_s'] == pytest.approx(.12)
    assert not proxy_intervals(record)['usable_for_independent_training']


def training():
    return [dict(input_tokens=512, repeat=repeat, purpose='training',
                 observations=[endpoint_intervals(payload(16))]) for repeat in range(3)]


def test_fit_is_training_only_and_does_not_generalize_to_short_output():
    windows = training()
    candidate = training_candidate(windows, identity={'model_id': 'test'}, source={})
    node = candidate['nodes'][0]
    assert node['predictions']['first_to_second_output_s'] == pytest.approx(.12)
    assert node['output_tokens'] == 16
    assert not node['p99_risk_qualified'] and not node['short_output_generalization_qualified']
    assert not node['actual_clock_domain_qualified']
    assert 'f_P_mhz' not in node and node['requested_f_P_mhz'] == 2520
    windows.append(dict(windows[0], purpose='holdout', repeat=3))
    with pytest.raises(ValueError, match='training only'):
        training_candidate(windows, identity={}, source={})
    windows[-1]['purpose'] = 'evaluation'
    with pytest.raises(ValueError, match='training only'):
        training_candidate(windows, identity={}, source={})


def test_missing_or_duplicate_repeat_cannot_supply_training_candidate():
    for rows in (training()[:2], training()[:2] + [deepcopy(training()[0])]):
        with pytest.raises(ValueError, match='three original'):
            training_candidate(rows, identity={}, source={})


def ledger(tmp_path, *, evaluation=False, split='tuning', lengths=(2024, 7168)):
    path = tmp_path/'ledger.json'
    path.write_text(json.dumps(dict(evaluation_read=evaluation, ledgers=[dict(
        model_id='test', selection_split=split, frequency_scope=[1500, 2520],
        queries=[dict(method='transfer_seconds', args=[n], finite_arguments=True) for n in lengths])])))
    return binding(path)


def test_plan_uses_only_exact_tuning_query_lengths_and_fresh_separate_seeds(tmp_path):
    plan = collection_plan(ledger(tmp_path), 'test')
    assert len(plan['points']) == 32
    assert {p['input_tokens'] for p in plan['points']} == {2024, 7168}
    assert {p['output_tokens'] for p in plan['points']} == {2, 4, 8, 16}
    assert plan['training_seed'] != plan['holdout_seed']
    assert plan['min_requests_per_node_per_split'] >= 100
    assert plan['prospective_order'] == [
        'training_only', 'freeze_bound_training_candidate', 'independent_holdout_only']
    assert not plan['collector_ready'] and not plan['physical_copy_time']


@pytest.mark.parametrize('kwargs', [dict(evaluation=True), dict(split='evaluation'), dict(lengths=())])
def test_evaluation_or_absent_pd_domain_cannot_select_nodes(tmp_path, kwargs):
    with pytest.raises(ValueError):
        collection_plan(ledger(tmp_path, **kwargs), 'test')


def test_plan_rejects_stale_query_evidence(tmp_path):
    reference = ledger(tmp_path)
    (tmp_path/'ledger.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        collection_plan(reference, 'test')


@pytest.mark.parametrize('failure', [None, 'bad_timestamp', 'transport_exception'])
def test_minimal_pair_collector_persists_real_raw_before_accepting_or_rejecting(tmp_path, monkeypatch, failure):
    raw = payload()
    raw['prefill']['prompt_tokens'] = raw['combined']['prompt_tokens'] = 2
    if failure == 'bad_timestamp':
        raw['combined']['decode_first_token_s'] = 10.

    class Client:
        async def complete(self, *args, **kwargs):
            if failure == 'transport_exception':
                raise RuntimeError('connection failed')
            return Completion('ordinary', 'd', 10., prompt_tokens=2, completion_tokens=2,
                stream_done=True, usage_received=True, token_ids=[101, 102])

    async def pd(*args, **kwargs):
        return Completion(**raw['prefill']), Completion(**raw['combined'])

    monkeypatch.setattr('pdblend.engine.client.pd_complete', pd)
    path = tmp_path/'raw.json'
    invoke = collect_endpoint_pair(None, Client(), Client(), [20, 21], output_tokens=2,
        tag='calibration', purpose='training', seed=9911, persist=lambda row: write_new(path, row))
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            asyncio.run(invoke)
    else:
        result = asyncio.run(invoke)
        assert result['observations']['first_to_second_output_s'] == pytest.approx(.12)
        assert not result['fleet_clock_cleanup_qualified']
    assert path.exists()
    saved = json.loads(path.read_text())
    assert not saved['physical_copy_time'] and not saved['component_qualified']
    assert saved['purpose'] == 'training' and saved['prompt'] == [20, 21]


def test_clock_failure_preserves_raw_endpoint_diagnostics_without_assigning_actual_clock_domain(monkeypatch):
    caps = {key: 'identity' for key in handoff.IDENTITY}
    caps['source_revision'] = 'source'
    report = dict(runtime_plan=dict(selection_split='calibration_and_independent_holdout',
        evaluation_used_for_selection=False), initial_capabilities={'p': caps},
        journal={'path': 'journal'}, power={'path': 'power'}, lease={})
    rows = []
    for length in (512, 2048, 7168):
        for repeat in range(4):
            purpose = 'training' if repeat < 3 else 'holdout'
            tag = f'transfer-{length}-{repeat}-measure-0'
            rows.append(dict(kind='transfer_request', tag=tag, purpose=purpose, result=payload(16)))
            rows.append(dict(kind='transfer', input_tokens=length, repeat=repeat, purpose=purpose,
                gpus=[0, 1], prefill_instance='p', decode_instance='d',
                started_s=10., finished_s=12., timings=[{}]))
    monkeypatch.setattr(handoff, '_source_files', lambda *a: {'source_sha256': 'source'})
    monkeypatch.setattr(handoff, 'read_bound', lambda *a: report)
    monkeypatch.setattr(handoff, '_translated', lambda report, *a: report)
    monkeypatch.setattr(handoff, 'replay_runtime', lambda report: dict(raw_components_complete=True,
                                                                    holdout_passed=False, errors=[]))
    monkeypatch.setattr(handoff, 'bound', lambda ref, **kw: rows if ref['path'] == 'journal' else {})

    def bad_clock(*args):
        raise ValueError('physical frequency coverage failed')

    monkeypatch.setattr(handoff, 'validate_frequencies', bad_clock)
    result = handoff.extract_runtime({}, {})
    assert len(result['windows']) == 12
    assert not result['physical_frequency_coverage_qualified'] and not result['component_qualified']
    assert all(not node['actual_clock_domain_qualified'] for node in result['candidate']['nodes'])
    assert result['windows'][0]['observations'][0]['first_to_second_output_s'] == pytest.approx(.12)
    assert all('physical frequency coverage' in row['physical_clock_error'] for row in result['windows'])
