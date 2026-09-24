"""Synthetic prepared corpus/anchor; no GPU execution or evaluation outcomes."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from pdblend.bench import capacity_workloads as workloads
from pdblend.bench.client import load_split, poisson_trace


SEEDS = (8801, 8802, 8803)
MODEL = 'Qwen2.5-7B-Instruct'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return workloads.binding(path)


def inputs(root):
    corpus = root/'corpus'
    data = dict(dataset='alpaca', model_name=MODEL,
        calibration=[dict(prompt=[1, 2], output_tokens=4), dict(prompt=[3], output_tokens=2)],
        tuning=[dict(prompt=[4, 5], output_tokens=4), dict(prompt=[6, 7], output_tokens=16)],
        evaluation=[dict(prompt=[888], output_tokens=19)])
    data_ref = put(corpus/'alpaca.json', data)
    manifest = dict(complete=True, model_name=MODEL, dataset_sha256={'alpaca': data_ref['sha256']},
                    tokenizer_sha256='corpus-tokenizer', model_config_sha256='model-config')
    manifest_ref = put(corpus/'manifest.json', manifest)
    records = load_split(corpus, 'alpaca', 'tuning')
    request_ref = put(root/'anchor/alpaca-tuning-0/requests.json', dict(seed=9702, duration_s=120.,
        requests=[asdict(request) for request in poisson_trace(records, 1., 120., 9702, 'alpaca')]))
    confirmation = dict(dataset='alpaca', split='tuning', seed=9702, duration_s=120., rate_rps=1.,
                        metrics={'passed': True}, trace_sha256=request_ref['sha256'])
    confirmation_ref = put(root/'anchor/alpaca-tuning-0/completion.json', confirmation)
    anchor = dict(status='passed', complete=True, hardware_executed=True, model_id=MODEL,
        model_hash='model-hash', tokenizer_hash='model-tokenizer', evaluation_used_for_selection=False,
        selection_splits=['calibration', 'tuning'], cleanup_errors=[], calibration_seed=9701, tuning_seed=9702,
        corpus_manifest_sha256=manifest_ref['sha256'], corpus_tokenizer_sha256='corpus-tokenizer',
        anchors={'alpaca': dict(model_id=MODEL, corpus_sha256=data_ref['sha256'], base_rate_rps=1.,
            confirmation_path='/output/anchor/alpaca-tuning-0/completion.json',
            confirmation_sha256=confirmation_ref['sha256'], scope='highest_tested_and_confirmed_passing_rate')})
    return corpus, put(root/'anchor/completion.json', anchor)


def family(root, *, split='tuning', **kwargs):
    corpus, anchor = inputs(root)
    return workloads.prepare_family(root/'family', corpus=corpus, dataset='alpaca', split=split,
        anchor=anchor, seeds=SEEDS, minimum_scale=.25, **kwargs)


def test_family_extends_one_common_window_and_fixes_independent_identity(tmp_path):
    value = family(tmp_path)['family']
    identity = value['identity']
    assert identity['duration_s'] > 150
    assert identity['seeds'] == list(SEEDS)
    assert identity['minimum_scale'] == .25 and identity['minimum_requests'] == 100
    assert min(value['duration_selection']['steps'][-1]['request_counts']) >= 100
    assert value['duration_selection']['uses_performance'] is False
    assert identity['anchor']['confirmation_requests_replayed']
    assert value['family_id'] == 'capacity-'+workloads.digest(identity)
    assert not value['execution_ready'] and value['execution_wiring_pending']
    assert not value['capacity_claimed'] and not value['formal_eligible']


def test_same_family_uses_same_duration_at_all_rates_and_same_cohort_for_every_system(tmp_path):
    prepared = family(tmp_path)
    first = workloads.prepare_rate(prepared['manifest'], tmp_path/'low', rate_scale=.25)['workloads']
    second = workloads.prepare_rate(prepared['manifest'], tmp_path/'high', rate_scale=1.)['workloads']
    assert first['duration_s'] == second['duration_s'] == prepared['family']['identity']['duration_s']
    assert len(first['traces']) == 3 and len(first['workload_assignments']) == 15
    for row in first['traces']:
        trace = workloads.read_bound(row['trace'])
        assert len(trace['requests']) >= 100
        points = [point for point in first['workload_assignments'] if point['seed'] == row['seed']]
        assert len(points) == 5 and all(point['trace'] == row['trace'] for point in points)
        assert all(point['inputs']['trace'] == row['trace'] for point in points)
        assert all(point['capacity_workload_family'] == point['inputs']['capacity_workload_family'] == prepared['manifest'] for point in points)
        assert all(point['output_workload'] == trace['output_workload'] for point in points)
        assert workloads.validate_trace(row['trace'])['trace'] == trace
        assert all(point['selection_split'] == trace['selection_split'] == 'tuning' for point in points)
        assert trace['requests_sha256'] == workloads.digest(trace['requests'])
        assert all(request['prompt'] in ([4, 5], [6, 7]) for request in trace['requests'])
        assert trace['output_tokens_total'] == sum(request['max_tokens'] for request in trace['requests'])
        assert trace['output_workload'] == prepared['family']['identity']['output_workload']
    # RNG content stream has a shared prefix across rates; the larger rate
    # contains more arrivals, without shortening anyone's service window.
    low = workloads.read_bound(first['traces'][0]['trace'])['requests']
    high = workloads.read_bound(second['traces'][0]['trace'])['requests']
    assert len(high) > len(low)
    assert [(r['prompt'], r['max_tokens']) for r in low] == [(r['prompt'], r['max_tokens']) for r in high[:len(low)]]


def test_calibration_split_is_selected_without_reading_evaluation_results(tmp_path):
    prepared = family(tmp_path, split='calibration')
    rate = workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=.25)['workloads']
    trace = workloads.read_bound(rate['traces'][0]['trace'])
    assert trace['selection_split'] == 'calibration'
    assert all(request['prompt'] in ([1, 2], [3]) for request in trace['requests'])
    assert not trace['evaluation_used_for_selection']


def test_below_minimum_requires_new_family_and_never_creates_a_changed_window(tmp_path):
    prepared = family(tmp_path)
    before = Path(prepared['manifest']['path']).read_bytes()
    with pytest.raises(ValueError, match='new family'):
        workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=.125)
    assert not (tmp_path/'rate').exists()
    assert Path(prepared['manifest']['path']).read_bytes() == before


@pytest.mark.parametrize('seeds', [(701, 8802, 8803), (9701, 8802, 8803), (9702, 8802, 8803),
                                  (8801, 8801, 8802), (8801, 8802), (True, 8802, 8803)])
def test_evaluation_anchor_or_duplicate_seeds_cannot_be_independent_repetitions(tmp_path, seeds):
    corpus, anchor = inputs(tmp_path)
    with pytest.raises(ValueError, match='seed'):
        workloads.prepare_family(tmp_path/'family', corpus=corpus, dataset='alpaca', split='tuning',
                                 anchor=anchor, seeds=seeds, minimum_scale=.25)
    assert not (tmp_path/'family').exists()


def test_evaluation_split_or_evaluation_schema_cannot_be_relabelled(tmp_path):
    corpus, anchor = inputs(tmp_path)
    with pytest.raises(ValueError, match='never evaluation'):
        workloads.prepare_family(tmp_path/'family', corpus=corpus, dataset='alpaca', split='evaluation',
                                 anchor=anchor, seeds=SEEDS, minimum_scale=.25)
    evaluation_ref = put(tmp_path/'evaluation.json', dict(schema='five-system-evaluation-trace-v1',
                                                       selection_split='tuning', seed=701))
    with pytest.raises(ValueError, match='schema'):
        workloads.prepare_rate(evaluation_ref, tmp_path/'rate', rate_scale=1.)
    assert not (tmp_path/'family').exists() and not (tmp_path/'rate').exists()


def test_anchor_seed701_requests_cannot_be_relabelled_to_9702_even_with_rebound_hashes(tmp_path):
    corpus, anchor_ref = inputs(tmp_path)
    request_path = tmp_path/'anchor/alpaca-tuning-0/requests.json'
    request_ref = put(request_path, dict(seed=9702, duration_s=120., requests=[asdict(request)
        for request in poisson_trace(load_split(corpus, 'alpaca', 'evaluation'), 1., 120., 701, 'alpaca')]))
    confirmation_path = request_path.parent/'completion.json'
    confirmation = json.loads(confirmation_path.read_text())
    confirmation['trace_sha256'] = request_ref['sha256']
    confirmation_ref = put(confirmation_path, confirmation)
    anchor = workloads.read_bound(anchor_ref)
    anchor['anchors']['alpaca']['confirmation_sha256'] = confirmation_ref['sha256']
    anchor_ref = put(Path(anchor_ref['path']), anchor)
    with pytest.raises(ValueError, match='relabel forbidden'):
        workloads.prepare_family(tmp_path/'family', corpus=corpus, dataset='alpaca', split='tuning',
                                 anchor=anchor_ref, seeds=SEEDS, minimum_scale=.25)


@pytest.mark.parametrize('changed', ['corpus', 'anchor', 'confirmation', 'requests'])
def test_changed_input_bytes_cannot_enter_a_prepared_rate(tmp_path, changed):
    prepared = family(tmp_path)
    paths = dict(corpus=tmp_path/'corpus/alpaca.json', anchor=tmp_path/'anchor/completion.json',
        confirmation=tmp_path/'anchor/alpaca-tuning-0/completion.json',
        requests=tmp_path/'anchor/alpaca-tuning-0/requests.json')
    path = paths[changed]
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='checksum|changed'):
        workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=.25)
    assert not (tmp_path/'rate').exists()


def test_existing_family_and_rate_outputs_are_never_overwritten(tmp_path):
    prepared = family(tmp_path)
    workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=.25)
    with pytest.raises(ValueError, match='immutable'):
        workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=.25)
    corpus = tmp_path/'corpus'
    with pytest.raises(ValueError, match='immutable'):
        workloads.prepare_family(tmp_path/'family', corpus=corpus, dataset='alpaca', split='tuning',
            anchor=workloads.binding(tmp_path/'anchor/completion.json'), seeds=SEEDS, minimum_scale=.25)


@pytest.mark.parametrize('changes', [dict(minimum_scale=0), dict(minimum_scale=float('nan')),
    dict(base_duration_s=-1), dict(minimum_requests=99), dict(systems=['pdblend', 'pdblend'])])
def test_invalid_family_parameters_fail_before_writing(tmp_path, changes):
    corpus, anchor = inputs(tmp_path)
    arguments = dict(corpus=corpus, dataset='alpaca', split='tuning', anchor=anchor,
                     seeds=SEEDS, minimum_scale=.25)
    arguments.update(changes)
    with pytest.raises(ValueError):
        workloads.prepare_family(tmp_path/'family', **arguments)
    assert not (tmp_path/'family').exists()


def test_current_workspace_generator_change_does_not_invalidate_frozen_family(tmp_path, monkeypatch):
    prepared = family(tmp_path)
    def changed(*args, **kwargs):
        raise AssertionError('mutable workspace generator must not be used')
    monkeypatch.setattr(workloads, 'poisson_trace', changed)
    monkeypatch.setattr(workloads, 'load_split', changed)
    monkeypatch.setattr(workloads.client, '__file__', '/missing/future/client.py')
    result = workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=.25)
    validated = workloads.validate_trace(result['workloads']['traces'][0]['trace'])
    assert validated['family_ref'] == prepared['manifest']
    assert validated['family'] == prepared['family']
    assert validated['trace']['capacity_workload_family'] == prepared['manifest']


def test_approved_builtin_replayer_matches_original_load_split_and_poisson(tmp_path):
    corpus, _ = inputs(tmp_path)
    for split in ('calibration', 'tuning'):
        records = load_split(corpus, 'alpaca', split)
        assert workloads._v1_load_split(corpus, 'alpaca', split) == records
        for seed in (*SEEDS, 9702):
            expected = [asdict(r) for r in poisson_trace(records, .75, 400., seed, 'alpaca')]
            actual = [asdict(r) for r in workloads._v1_poisson_trace(records, .75, 400., seed, 'alpaca')]
            assert actual == expected


def test_rebound_malicious_replayer_never_executes_artifact_source(tmp_path):
    prepared = family(tmp_path)
    value = prepared['family']
    malicious = tmp_path/'malicious.py'
    sentinel = tmp_path/'executed'
    malicious.write_text('from pathlib import Path; Path('+repr(str(sentinel))+').touch()')
    value['identity']['generator']['replayer'] = workloads.binding(malicious)
    value['family_id'] = 'capacity-'+workloads.digest(value['identity'])
    rebound = put(tmp_path/'rebound-family.json', value)
    with pytest.raises(ValueError, match='approved v1'):
        workloads.prepare_rate(rebound, tmp_path/'rate', rate_scale=.25)
    assert not sentinel.exists() and not (tmp_path/'rate').exists()


def test_unbound_archive_change_is_rejected(tmp_path):
    prepared = family(tmp_path)
    path = Path(prepared['family']['identity']['generator']['replayer']['path'])
    path.write_text(path.read_text()+'# changed')
    with pytest.raises(ValueError, match='checksum'):
        workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=.25)


@pytest.mark.parametrize('changed', ['duration_s', 'base_rate_rps', 'model_hash', 'tokenizer_hash'])
def test_rebound_family_still_checks_independent_identity_and_window(tmp_path, changed):
    prepared = family(tmp_path)
    value = prepared['family']
    old = value['identity'][changed]
    value['identity'][changed] = old*2 if isinstance(old, (int, float)) else old+'-different'
    value['family_id'] = 'capacity-'+workloads.digest(value['identity'])
    rebound = put(tmp_path/'rebound-family.json', value)
    with pytest.raises(ValueError, match='duration|identity'):
        workloads.prepare_rate(rebound, tmp_path/'rate', rate_scale=.25)


@pytest.mark.parametrize('changed', ['duration_s', 'output_workload', 'repeat_id', 'prompt', 'arrival', 'max_tokens', 'bool-token', 'family'])
def test_trace_rebinding_does_not_hide_modified_cohort_or_metadata(tmp_path, changed):
    prepared = family(tmp_path)
    result = workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=.25)
    original = result['workloads']['traces'][0]['trace']
    trace = workloads.read_bound(original)
    if changed == 'prompt':
        trace['requests'][0]['prompt'] = [888]
    elif changed == 'arrival':
        trace['requests'][0]['arrival_s'] += .001
    elif changed == 'max_tokens':
        trace['requests'][0]['max_tokens'] += 1
    elif changed == 'bool-token':
        trace['requests'][0]['idx'] = False
    elif changed == 'family':
        trace['family'] = dict(path='/different', sha256='unbound')
    elif changed == 'duration_s':
        trace[changed] += 1
    else:
        trace[changed] = 'modified'
    trace['requests_sha256'] = workloads.digest(trace['requests'])
    rebound = put(tmp_path/'rebound-trace.json', trace)
    with pytest.raises(ValueError, match='reproduce|family bindings'):
        workloads.validate_trace(rebound)


def test_trace_seed701_and_below_minimum_cannot_be_rebound(tmp_path):
    prepared = family(tmp_path)
    result = workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=.25)
    trace = workloads.read_bound(result['workloads']['traces'][0]['trace'])
    trace['seed'] = 701
    with pytest.raises(ValueError, match='seed'):
        workloads.validate_trace(put(tmp_path/'seed701.json', trace))
    trace['seed'] = SEEDS[0]
    trace['scale'] = .125
    with pytest.raises(ValueError, match='new family'):
        workloads.validate_trace(put(tmp_path/'low.json', trace))


def test_python_version_change_requires_explicit_replay_support(tmp_path):
    prepared = family(tmp_path)
    value = prepared['family']
    value['identity']['generator']['python_version'] = [99, 0, 0]
    value['family_id'] = 'capacity-'+workloads.digest(value['identity'])
    with pytest.raises(ValueError, match='Python version'):
        workloads.prepare_rate(put(tmp_path/'rebound.json', value), tmp_path/'rate', rate_scale=.25)


def recovered_inputs(root):
    """Completed recovery retains exact old confirmation, with explicit raw ref."""
    corpus, old_ref = inputs(root)
    old = workloads.read_bound(old_ref)
    old.update(status='failed', complete=False,
               error='RuntimeError: longbench: no independent tuning confirmation passed')
    old_ref = put(Path(old_ref['path']), old)
    original_confirmation = root/'anchor/alpaca-tuning-0/completion.json'
    original_requests = original_confirmation.parent/'requests.json'
    recovered = dict(old, status='passed', complete=True, schema='longbench-anchor-recovery-v1',
                     scope='longbench_only_anchor_recovery')
    recovered.pop('error')
    recovered['prior_inputs'] = dict(receipts=dict(completion=old_ref), inherited=dict(alpaca=dict(
        completion=workloads.binding(original_confirmation),requests=workloads.binding(original_requests))))
    copied = root/'recovery/alpaca-tuning-0/completion.json'
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original_confirmation.read_bytes())
    return corpus, put(root/'recovery/completion.json', recovered)


def test_complete_recovery_explicit_inherited_requests_are_replayed_without_copying_history(tmp_path):
    corpus, anchor = recovered_inputs(tmp_path)
    prepared = workloads.prepare_family(tmp_path/'family', corpus=corpus,dataset='alpaca',split='tuning',
        anchor=anchor,seeds=SEEDS,minimum_scale=.25)
    bound = prepared['family']['identity']['anchor']
    assert bound['requests']['path'] == str(tmp_path/'anchor/alpaca-tuning-0/requests.json')
    assert bound['inherited_provenance']['completion']['path'] == str(tmp_path/'anchor/completion.json')
    rate = workloads.prepare_rate(prepared['manifest'], tmp_path/'rate', rate_scale=1.)
    assert workloads.validate_trace(rate['workloads']['traces'][0]['trace'])['family'] == prepared['family']
    assert not (tmp_path/'recovery/alpaca-tuning-0/requests.json').exists()


@pytest.mark.parametrize('changed', ['missing_link', 'request_hash', 'completion_hash', 'old_identity', 'schema', 'old_anchor', 'old_anchor_bool', 'old_hardware'])
def test_inherited_requests_need_exact_explicit_bound_provenance(tmp_path, changed):
    corpus, anchor = recovered_inputs(tmp_path)
    recovered = workloads.read_bound(anchor)
    inherited = recovered['prior_inputs']['inherited']['alpaca']
    if changed == 'missing_link':
        recovered['prior_inputs']['inherited'] = {}
    elif changed == 'request_hash':
        inherited['requests']['sha256'] = '0'*64
    elif changed == 'completion_hash':
        inherited['completion']['sha256'] = '0'*64
    elif changed == 'schema':
        recovered['schema'] = 'unknown'
    else:
        old_ref = recovered['prior_inputs']['receipts']['completion']
        old = workloads.read_bound(old_ref)
        if changed == 'old_identity':
            old['model_hash'] = 'other'
        elif changed == 'old_hardware':
            old['hardware_executed'] = False
        elif changed == 'old_anchor_bool':
            old['anchors']['alpaca']['base_rate_rps'] = True
        else:
            old['anchors']['alpaca']['base_rate_rps'] *= 2
        recovered['prior_inputs']['receipts']['completion'] = put(Path(old_ref['path']),old)
    anchor = put(Path(anchor['path']),recovered)
    with pytest.raises((ValueError,KeyError)):
        workloads.prepare_family(tmp_path/'family',corpus=corpus,dataset='alpaca',split='tuning',
            anchor=anchor,seeds=SEEDS,minimum_scale=.25)
    assert not (tmp_path/'family').exists()
