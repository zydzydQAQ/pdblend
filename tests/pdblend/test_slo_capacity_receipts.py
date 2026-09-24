"""Synthetic immutable receipt fixtures; these tests do not run experiments."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pdblend.bench.comparison_campaign import binding
from pdblend.bench import capacity_workloads as workloads
from pdblend.bench.client import load_split, poisson_trace
from pdblend.bench.resident_session import digest, engine_signature
from pdblend.bench.slo_capacity import CapacityConfig
from pdblend.bench.slo_capacity_receipts import read_capacity_ledger, trial_from_receipt
from test_capacity_workloads import inputs, SEEDS


CONFIG = CapacityConfig('7b/pdblend/alpaca', 1., .1)
SERIES = dict(model_id='Qwen2.5-7B-Instruct', system='pdblend', dataset='alpaca')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return binding(path)


def workload_family(root, *, base_rate=1.):
    directory = root/'workloads'/('anchor-'+str(base_rate))
    path = directory/'family/family.json'
    if not path.exists():
        corpus, anchor = inputs(directory)
        if base_rate != 1.:
            request_ref = put(directory/'anchor/alpaca-tuning-0/requests.json', dict(seed=9702, duration_s=120.,
                requests=[asdict(r) for r in poisson_trace(load_split(corpus, 'alpaca', 'tuning'),
                    base_rate, 120., 9702, 'alpaca')]))
            confirmation = workloads.read_bound(workloads.binding(directory/'anchor/alpaca-tuning-0/completion.json'))
            confirmation.update(rate_rps=base_rate, trace_sha256=request_ref['sha256'])
            confirmation_ref = put(directory/'anchor/alpaca-tuning-0/completion.json', confirmation)
            value = workloads.read_bound(anchor)
            value['anchors']['alpaca'].update(base_rate_rps=base_rate, confirmation_sha256=confirmation_ref['sha256'])
            anchor = put(Path(anchor['path']), value)
        workloads.prepare_family(directory/'family', corpus=corpus, dataset='alpaca', split='tuning',
            anchor=anchor, seeds=SEEDS, minimum_scale=1.)
    return binding(path)


def receipt(root, *, rate=1., repeat=0, split='tuning', passing=True,
            usable=True, canonical=True, raw_id=None, revision='source-v1', base_rate=1.,
            source_revision=None, dispatcher_revision=None, measured_revision=None,
            system='pdblend', family_ref=None, point_changes=None, trace_changes=None, gpu_prefix='GPU-'):
    name = f'x{rate}-r{repeat}'
    window = root/'windows'/name
    source_revision = revision if source_revision is None else source_revision
    source = put(root/'shared'/(source_revision+'.json'), {'source_sha256': source_revision})
    profile = put(root/'shared/profile.json', {'qualified': False})
    series = {**SERIES, 'system':system}
    config = put(root/'shared/config.json', {**series, 'profile': profile})
    family_ref = family_ref or workload_family(root, base_rate=base_rate)
    rate_directory = root/'rates'/('x'+str(rate)+'-'+family_ref['sha256'][:12])
    if not rate_directory.exists():
        workloads.prepare_rate(family_ref, rate_directory, rate_scale=rate)
    trace_ref = binding(rate_directory/('seed-'+str(SEEDS[repeat])+'.json'))
    trace = workloads.read_bound(trace_ref)
    if split != 'tuning' or trace_changes:
        trace.update(selection_split=split, **(trace_changes or {}))
        trace_ref = put(root/'traces'/(name+'.json'), trace)
    engine = dict(model_hash='model-hash', tokenizer_hash='model-tokenizer', image_digest='image',
        runtime_source_sha256='runtime', entrypoint='native', worker_extension='v1',
        dtype='bfloat16', environment={}, instances=[dict(instance_id='m'+str(i), tp=1,
            pp=1, gpu_uuids=[gpu_prefix+str(i)], launch_options={}) for i in range(8)])
    point = dict(**series, name=name, revision=revision, source_manifest=source,
        engine_identity=engine, seed=trace['seed'], duration_s=trace['duration_s'], rate_rps=trace['rate_rps'],
        scale=rate, slo=trace['slo'], measurement_protocol_version=trace['measurement_protocol_version'],
        family_id=trace['family_id'], repeat_id=trace['repeat_id'], output_workload=trace['output_workload'],
        capacity_workload_family=family_ref,
        trace=trace_ref, inputs=dict(trace=trace_ref, profiles=[profile], system_config=config,
            source_manifest=source, capacity_workload_family=family_ref))
    point.update(point_changes or {})
    if dispatcher_revision is not None:
        point['inputs']['source_manifest'] = put(root/'shared'/(dispatcher_revision+'.json'),
                                                {'source_sha256': dispatcher_revision})
    point_ref = put(window/'point.json', point)
    raw_id = system+'/'+name if raw_id is None else raw_id
    refs = dict(trace=trace_ref)
    for key in ('outcomes', 'canonical_requests', 'native_result', 'power', 'metering'):
        refs[key] = put(window/'run'/(key+'.json'), dict(synthetic=True, raw_id=raw_id, kind=key))
    for key in ('reset', 'drain'):
        refs[key] = put(window/(key+'.json'), dict(passed=True))
    count = len(trace['requests'])
    metrics = dict(offered_requests=count, successful_requests=count, failed_requests=0,
        joint_slo_requests=count if passing else int(count*.8), unresolved_requests=0,
        ttft_samples=count, tpot_samples=count, ttft_p99_s=.8 if passing else 1.1,
        tpot_p99_s=.08, slo_ttft_s=1., slo_tpot_s=.1, duration_s=trace['duration_s'],
        measurement_protocol_version=point['measurement_protocol_version'])
    audit = dict(evidence_valid=False, measurement_evidence_valid=usable,
        formal_eligible=False, point_sha256=digest(point), metrics_sha256=digest(metrics),
        raw_refs=refs, evidence_sha256=digest(refs), missing_gates=[], gate_failures={},
        checked_gates=['metering.raw_eight_gpu_window'] + (['pdblend.canonical_metrics'] if canonical else []))
    result = dict(metrics=metrics, acceptance=audit, measurement_evidence_valid=usable,
                  evidence_valid=False, formal_eligible=False, profile_qualified=False)
    if measured_revision is not None:
        result['identity'] = dict(source_sha256=measured_revision)
    put(window/'result.json', result)
    value = dict(point=name, point_sha256=digest(point), result=result, cleanup_passed=True,
        engine_signature=engine_signature(point['engine_identity']), session_id=name,
        artifacts={str(p.relative_to(window)): binding(p)['sha256'] for p in window.rglob('*')
                   if p.is_file() and p.name != 'receipt.json'})
    return dict(receipt=put(window/'receipt.json', value), point=point_ref, repeat_id=trace['repeat_id'])


def ledger(root, rows, *, system='pdblend'):
    path = root/'ledger.json'
    config = {**asdict(CONFIG), 'series_id':'7b/'+system+'/alpaca'}
    put(path, dict(schema='pdblend-slo-capacity-ledger/v1', config=config,
                  series={**SERIES, 'system':system}, trials=rows))
    return path


def load_trial(row):
    return trial_from_receipt(row['receipt'], row['point'], repeat_id=row['repeat_id'],
                              config=CONFIG, series=SERIES)


def test_real_receipt_schema_supplies_state_and_frozen_grid_without_qualification(tmp_path):
    rows = [receipt(tmp_path, rate=rate, repeat=repeat, passing=rate == 1.)
            for rate in (1., 1.04) for repeat in range(3)]
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    path = ledger(tmp_path, rows)
    report = read_capacity_ledger(path)
    assert report['state']['converged']
    assert report['state']['capacity_interval'] == {'passed_lower': 1., 'failed_upper': 1.04}
    assert report['frozen_evaluation_grid']['target_split'] == 'evaluation'
    assert not report['formal_eligible'] and not report['profile_qualification_promoted']
    assert not report['hardware_executed'] and not report['jobs_enqueued']
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('usable,canonical', [(False, True), (True, False)])
def test_missing_measurement_acceptance_abstains_even_when_metrics_look_good(tmp_path, usable, canonical):
    rows = [receipt(tmp_path, repeat=i, usable=usable, canonical=canonical) for i in range(3)]
    report = read_capacity_ledger(ledger(tmp_path, rows))
    assert report['state']['status'] == 'incomplete'
    assert report['state']['passed_lower'] is report['state']['failed_upper'] is None
    assert report['frozen_evaluation_grid'] is None


def test_evaluation_receipt_cannot_be_relabelled_as_tuning(tmp_path):
    row = receipt(tmp_path, split='evaluation')
    with pytest.raises(ValueError, match='evaluation'):
        load_trial(row)


def test_adapter_rejects_a_different_slo_without_waiting_for_aggregation(tmp_path):
    row = receipt(tmp_path)
    with pytest.raises(ValueError, match='frozen SLO'):
        trial_from_receipt(row['receipt'], row['point'], repeat_id=row['repeat_id'],
                          config=CapacityConfig(CONFIG.series_id, 2., .1), series=SERIES)


@pytest.mark.parametrize('target', ['receipt', 'point', 'raw', 'profile'])
def test_changed_bound_bytes_are_rejected(tmp_path, target):
    row = receipt(tmp_path)
    path = {'receipt': Path(row['receipt']['path']), 'point': Path(row['point']['path']),
            'raw': tmp_path/'windows/x1.0-r0/run/outcomes.json',
            'profile': tmp_path/'shared/profile.json'}[target]
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='checksum'):
        load_trial(row)


def test_relocated_duplicate_raw_window_cannot_become_independent_repeat(tmp_path):
    rows = [receipt(tmp_path, repeat=i, raw_id='same-measurement') for i in range(3)]
    with pytest.raises(ValueError, match='duplicate capacity receipt'):
        read_capacity_ledger(ledger(tmp_path, rows))


@pytest.mark.parametrize('change', [dict(scale=2.), dict(seed=701), dict(duration_s=151.),
    dict(measurement_protocol_version='old-150s'), dict(output_workload='shorter-output'),
    dict(family_id='foreign'), dict(repeat_id='seed-701')])
def test_point_cannot_disagree_with_replayed_workload_family(tmp_path, change):
    with pytest.raises(ValueError, match='trace identity differs'):
        load_trial(receipt(tmp_path, point_changes=change))


def test_fresh_raw_data_cannot_relabel_one_seed_as_another_repeat(tmp_path):
    row = receipt(tmp_path)
    row['repeat_id'] = 'seed-8802'
    with pytest.raises(ValueError, match='frozen seed identity'):
        load_trial(row)


def test_same_duration_and_anchor_cannot_hide_different_workload_family(tmp_path):
    alternate = workload_family(tmp_path/'alternative')
    rows = [receipt(tmp_path), receipt(tmp_path, rate=2., family_ref=alternate)]
    with pytest.raises(ValueError, match='workload family/rate anchor'):
        read_capacity_ledger(ledger(tmp_path, rows))


@pytest.mark.parametrize('target', ['point', 'dispatcher', 'engine'])
def test_family_binding_and_actual_model_identity_cannot_be_substituted(tmp_path, target):
    point = workloads.read_bound(receipt(tmp_path)['point'])
    if target == 'engine':
        point['engine_identity']['model_hash'] = 'different-weights'
        changes = dict(engine_identity=point['engine_identity'])
    else:
        alternate = workload_family(tmp_path/'alternative')
        if target == 'point':
            changes = dict(capacity_workload_family=alternate)
        else:
            changes = dict(inputs={**point['inputs'], 'capacity_workload_family':alternate})
    with pytest.raises(ValueError, match='family'):
        load_trial(receipt(tmp_path, point_changes=changes))


def test_system_must_belong_to_the_frozen_family(tmp_path):
    corpus, anchor = inputs(tmp_path/'source')
    family = workloads.prepare_family(tmp_path/'family', corpus=corpus, dataset='alpaca', split='tuning',
        anchor=anchor, seeds=SEEDS, minimum_scale=1., systems=['pdblend'])['manifest']
    row = receipt(tmp_path, system='mixed', family_ref=family)
    with pytest.raises(ValueError, match='system is outside workload family'):
        trial_from_receipt(row['receipt'], row['point'], repeat_id=row['repeat_id'],
                          config=CONFIG, series={**SERIES, 'system':'mixed'})


def test_eight_gpu_uuid_identity_is_explicit_and_cannot_change_between_rates(tmp_path):
    first = receipt(tmp_path)
    identity = load_trial(first).metrics['_capacity_evidence']['identity']
    assert identity['gpu_uuids'] == ['GPU-'+str(i) for i in range(8)]
    rows = [first, receipt(tmp_path, rate=2., gpu_prefix='foreign-GPU-')]
    with pytest.raises(ValueError, match='hardware'):
        read_capacity_ledger(ledger(tmp_path, rows))


def test_rebound_synthetic_requests_must_still_replay_from_the_frozen_corpus(tmp_path):
    row = receipt(tmp_path)
    point = workloads.read_bound(row['point'])
    trace = workloads.read_bound(point['trace'])
    requests = trace['requests']
    requests[0]['prompt'][0] = 999
    changes = dict(requests=requests, requests_sha256=digest(requests),
        cohort_sha256=digest([dict(prompt=r['prompt'], max_tokens=r['max_tokens']) for r in requests]))
    with pytest.raises(ValueError, match='replay|reproduc'):
        load_trial(receipt(tmp_path, trace_changes=changes))


@pytest.mark.parametrize('changes', [dict(revision='different'), dict(base_rate=2.)])
def test_different_execution_variant_or_rate_anchor_cannot_share_bracket(tmp_path, changes):
    rows = [receipt(tmp_path), receipt(tmp_path, rate=2., passing=False, **changes)]
    with pytest.raises(ValueError, match='changed source, profile, configuration, hardware or workload family/rate anchor'):
        read_capacity_ledger(ledger(tmp_path, rows))


def test_artifact_path_cannot_escape_window(tmp_path):
    row = receipt(tmp_path)
    value = json.loads(Path(row['receipt']['path']).read_text())
    external = put(tmp_path/'external.json', {})
    value['artifacts'][external['path']] = external['sha256']
    row['receipt'] = put(Path(row['receipt']['path']), value)
    with pytest.raises(ValueError, match='escapes'):
        load_trial(row)


@pytest.mark.parametrize('changes,reason', [
    (dict(source_revision='foreign'), 'source manifest revision'),
    (dict(dispatcher_revision='foreign'), 'dispatcher source manifest'),
    (dict(measured_revision='foreign'), 'measured source'),
])
def test_internally_inconsistent_source_bindings_are_rejected(tmp_path, changes, reason):
    with pytest.raises(ValueError, match=reason):
        load_trial(receipt(tmp_path, **changes))


def test_cli_is_read_only_and_reports_the_initial_rate(tmp_path):
    path = ledger(tmp_path, [])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-m', 'pdblend.bench.slo_capacity',
        '--ledger', str(path)], capture_output=True, text=True, timeout=10, env=env)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['state']['next_rate_scale'] == 1.
    assert report['state']['action'] == 'measure_rate'
    assert report['frozen_evaluation_grid'] is None
    assert list(tmp_path.iterdir()) == [path]


def test_cli_rejects_bad_binding_without_creating_a_report(tmp_path):
    row = receipt(tmp_path)
    Path(row['receipt']['path']).write_text('{}')
    path = ledger(tmp_path, [row])
    result = subprocess.run([sys.executable, '-m', 'pdblend.bench.slo_capacity',
        '--ledger', str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and 'checksum differs' in result.stderr
    assert result.stdout == ''
