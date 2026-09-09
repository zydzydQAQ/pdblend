import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import late_context as lc

ROOT = Path(__file__).resolve().parent


def fixture(empty=True):
    ids = [f'pdb-profile-cpu-{i}' for i in range(16)]
    rows = [dict(request_id=rid, success=True, done_marker=True, prompt_token_ids=[1]*512,
                 requested_output_tokens=256, output_token_ids=[2]*256,
                 token_received_s=[3+i*.04 for i in range(256)],
                 usage=dict(prompt_tokens=512, completion_tokens=256)) for rid in ids]
    spec = dict(batch_size=16, input_lengths=[512]*16, output_lengths=[256]*16, tp=2,
                target_gpus=[0, 1], budget_tokens=8192, max_num_seqs=32, clock_command_mhz=1500,
                seed=0, temperature=0, arrival_offsets_s=[0.]*16)
    raw = dict(requests=rows, spec=spec, generation=7, measurement_start_s=1., measurement_end_s=20.)
    def event(pf, dc, tokens, start, end, names):
        return dict(prefill=pf, decode=dc, tokens=tokens, started_s=start, finished_s=end,
                    request_ids=names, generation=7, role='mixed', mode='continuous')
    events = [event(16, 0, 8192, 2., 2.9, ids)]
    for index in range(255):
        t = 3+index*.04+(1 if empty and index >= 220 else 0)
        if empty and index == 220:
            events.append(event(0, 0, 0, t-.98, t-.01, []))
        events.append(event(0, 16, 16, t, t+.03, ids))
    return raw, events


def encode(events):return b''.join((json.dumps(e)+'\n').encode() for e in events)


def samples():
    power = [(i/50, [100.+g for g in range(8)]) for i in range(1051)]
    clocks = [(i/50, [1500.]*8) for i in range(1051)]
    return power, clocks


def test_full_per_request_tail_and_empty_step_keep_original_ordinal_and_time():
    raw, events = fixture()
    result, original = lc.reconstruct(raw, encode(events), source_order_verified=True)
    assert original == events
    assert result['full_owner_event_count'] == 257
    assert result['nonempty_owner_event_count'] == 256
    assert result['complete_decode_token_sum'] == 4080
    assert all(p['total_observed_decode_steps'] == 255 and p['attention_after_max'] == 767
               and p['declared_bucket_edge'] == 768 for p in result['per_request'].values())
    assert result['late_window']['finish_spacing_max_s'] == pytest.approx(1.04)
    assert len(result['late_window']['empty_steps_inside_window']) == 1
    assert result['late_window']['owner_event_indices'][-1] == 256
    assert result['late_window']['logical_context_by_step'][-1]['owner_event_index'] == 256
    empty = result['empty_owner_steps'][0]
    line = encode(events).splitlines(keepends=True)[empty['original_owner_event_index']]
    assert lc.hashlib.sha256(line).hexdigest() == empty['original_line_sha256']
    power, clocks = samples()
    lc.attach_tp2_window(result['late_window'], power, clocks, 1500)
    window = result['late_window']
    assert window['target_gpus_energy_j'] == pytest.approx(201*window['duration_s'])
    assert window['all_eight_gpu_energy_j'] == pytest.approx(828*window['duration_s'])
    full = lc.full_decode_windows(raw, original, power, clocks)
    assert full[0]['nonempty_decode_steps'] == 255 and full[0]['empty_steps_between'] == 1
    assert full[0]['finish_spacing_max_s'] == pytest.approx(1.04)


@pytest.mark.parametrize('mutation', ['missing_tail', 'duplicate_tail', 'generation', 'time_regression',
                                      'zero_with_id', 'zero_with_phase', 'zero_bool', 'foreign_id'])
def test_complete_owner_suffix_rejects_corruption(mutation):
    raw, events = fixture()
    if mutation == 'missing_tail':
        events[-1]['request_ids'].pop(); events[-1]['decode'] -= 1; events[-1]['tokens'] -= 1
    elif mutation == 'duplicate_tail':
        extra = copy.deepcopy(events[-1]); extra.update(started_s=19., finished_s=19.1); events.append(extra)
    elif mutation == 'generation': events[-1]['generation'] = 8
    elif mutation == 'time_regression': events[-1]['started_s'] = 1.
    elif mutation == 'zero_with_id': events[221]['request_ids'] = ['pdb-profile-cpu-0']
    elif mutation == 'zero_with_phase': events[221]['decode'] = 1
    elif mutation == 'zero_bool': events[221]['tokens'] = False
    elif mutation == 'foreign_id': events[-1]['request_ids'][0] = 'foreign'
    with pytest.raises(ValueError):lc.reconstruct(raw, encode(events), source_order_verified=True)


def test_no_truncated_65_line_tail_or_mixed_input_shape():
    raw, events = fixture()
    with pytest.raises(ValueError):lc.reconstruct(raw, encode(events[-65:]), source_order_verified=True)
    raw['spec']['input_lengths'][0] = 2048
    with pytest.raises(ValueError):lc.reconstruct(raw, encode(events), source_order_verified=True)


def test_no_source_proof_no_context_even_with_complete_work():
    raw, events = fixture()
    with pytest.raises(ValueError):lc.reconstruct(raw, encode(events), source_order_verified=False)


def source_fixture():
    actual = lc.read(ROOT/'expected-identity.json')
    prov = dict(instance_id='nextv3b0', tp=2, model=actual['model'],
                source_files_at_import=actual['source_files_at_import'], pid=1234)
    before = lc.source_record(actual, prov, actual, observed_s=0.)
    after = lc.source_record(actual, prov, actual, observed_s=21.)
    raw, _ = fixture(); raw.update(identity_before=actual, identity_after=copy.deepcopy(actual))
    return actual, prov, before, after, raw


def test_actual_b_source_pair_matches_full_identity_and_engine_pid():
    expected, _, before, after, raw = source_fixture()
    assert lc.validate_source_pair(before, after, raw, expected=expected)['verified'] is True


@pytest.mark.parametrize('mutation', ['missing_after', 'engine_pid', 'container_pid', 'source', 'logger', 'time', 'image', 'tp'])
def test_source_pair_missing_changed_or_unbracketed_rejected(mutation):
    expected, _, before, after, raw = source_fixture()
    if mutation == 'missing_after': after = {}
    elif mutation == 'engine_pid': after['engine_pid'] += 1
    elif mutation == 'container_pid': after['container_pid'] += 1
    elif mutation == 'source': after['files'].pop(next(iter(after['files'])))
    elif mutation == 'logger': after['engine_logger_sha256'] = '0'*64
    elif mutation == 'time': after['observed_s'] = 19.
    elif mutation == 'image': after['engine_image'] = 'sha256:'+'0'*64
    elif mutation == 'tp': after['tp'] = 1
    with pytest.raises(ValueError):lc.validate_source_pair(before, after, raw, expected=expected)


def test_current_default_source_receipt_is_not_future_before_after():
    expected, prov, _, _, _ = source_fixture()
    actual = copy.deepcopy(expected)
    default = '/usr/local/lib/python3.10/dist-packages/vllm/sampling_params.py'
    actual['live_vllm_and_serving_source_sha256'].pop(default)
    with pytest.raises(ValueError):lc.source_record(actual, prov, expected, observed_s=0.)


def test_clock_gpu1_failure_preserves_measured_energy():
    power, clocks = samples(); window = dict(start_s=5., end_s=10.)
    clocks[300][1][1] = 900.
    with pytest.raises(ValueError, match='both actual'):lc.attach_tp2_window(window, power, clocks, 1500)
    assert window['target_gpus_energy_j'] == pytest.approx(1005.)
    assert window['actual_target_clock_min_mhz'] == [1500., 900.]
    assert 'actual_both_target_clocks_valid' not in window


def test_missing_gpu_power_is_not_zero_fill():
    power, clocks = samples(); power[300][1][7] = None
    with pytest.raises(ValueError, match='power'):lc.attach_tp2_window(dict(start_s=5., end_s=10.), power, clocks, 1500)


def test_audit_preserves_whole_energy_if_events_or_source_are_missing(tmp_path):
    raw, _ = fixture(); lc.write_new(tmp_path/'raw.json', raw)
    power, clocks = samples()
    with (tmp_path/'power.csv').open('w') as f:
        f.write('t_s,'+','.join(f'gpu{i}_w' for i in range(8))+'\n')
        for t, values in power:f.write(','.join(map(str, [t]+values))+'\n')
    result = lc.audit_point(tmp_path, {}, {}, expected={}, original_evidence=None, tp2_validator=None)
    assert result['valid'] is False and result['errors']
    assert result['whole_batch_energy']['all_eight_gpu_energy_j'] == pytest.approx(19*828)


def test_six_original_warmups_belong_to_outer_not_primary(tmp_path):
    for index in range(6):
        path = tmp_path/'results'/f'point-{index:04d}'; path.mkdir(parents=True)
        raw = dict(measurement_start_s=4., measurement_end_s=10., warmup=dict(
            dispatch_s=2., stream_end_s=3., success=True, done_marker=True, prompt_token_ids=[1]*128,
            output_token_ids=[2]*64, token_received_s=[2.5]*64, usage=dict(prompt_tokens=128, completion_tokens=64)))
        lc.write_new(path/'raw.json', raw)
    report = lc.warmup_boundaries(tmp_path, 1., 11.)
    assert report['valid'] is True and report['do_not_add_primary_and_outer'] is True
    assert lc.warmup_boundaries(tmp_path, 2.1, 11.)['valid'] is False


def test_original_request_and_profiler_bytes_retained():
    parent = lc.read(ROOT/'parent-v1-manifest.json')
    for name in ('observation.frozen.py', 'capacity.py'):
        assert lc.sha(ROOT/name) == parent['files'][name]
    for name, value in parent['helper_files'].items():
        assert lc.sha(ROOT.parent/'budget-profiling-v2-candidate'/name) == value
