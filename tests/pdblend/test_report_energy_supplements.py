"""Selection must not turn repeated baseline measurements into cherry picking."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from pdblend.bench.comparison_campaign import binding
from pdblend.bench.resident_session import digest


@pytest.fixture
def report(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('energy_repair_report', scripts / '2026-09-24_report_energy_repair.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(energy=None, *, path='historical/receipt.json', attempt=None):
    identity = dict(trace_sha256='trace', seed=701, duration_s=150., slo_ttft_s=1., slo_tpot_s=.1,
                    offered_requests=100, offered_rps=2., measurement_protocol_version='meter/v1',
                    model_hash='model', tokenizer_hash='tokenizer', image_digest='image',
                    runtime_source_sha256='runtime', measurement_source_sha256='meter',
                    gpu_uuids=['gpu' + str(i) for i in range(8)])
    row = dict(series='dynamollm', system='dynamollm', model='7B', dataset='alpaca', rate_scale=.25,
               offered_requests=100, offered_rps=2., slo_ttft_s=1., slo_tpot_s=.1,
               successful_requests=100, all_requests_successful=True, joint_slo_requests=100,
               total_energy_kj=energy, service_energy_kj=None if energy is None else energy-10,
               tail_energy_kj=10., energy_measurement_complete=energy is not None,
               revision='historical' if attempt is None else 'supplement',
               receipt_path=path, receipt_sha256=path+'-sha', comparison_identity=identity,
               measurement_evidence_valid=True, raw_eight_gpu_meter_qualified=True,
               formal_eligible=False, observation_kind='historical' if attempt is None else 'baseline_supplement')
    if attempt is not None:
        row.update(supplement_of=dict(path='historical/receipt.json', sha256='historical/receipt.json-sha'),
                   supplement_job_index=0, supplement_attempt=attempt)
    return row


def test_first_complete_wins_despite_failed_requests_and_lower_later_energy(report):
    original = observation()
    first = observation(200., path='attempt2', attempt=2)
    first.update(successful_requests=30, all_requests_successful=False, joint_slo_requests=20,
                 measurement_evidence_valid=False, raw_eight_gpu_meter_qualified=False)
    later = observation(80., path='attempt10', attempt=10)
    selected, ledger = report.select_baseline_observations([original], [later, first])
    assert selected == [first]
    assert ledger[0]['selected_attempt'] == 2
    assert ledger[0]['selected_measurement_qualified'] is False
    assert ledger[0]['selected_raw_meter_qualified'] is False
    assert ledger[0]['selected_all_requests_successful'] is False
    assert original['total_energy_kj'] is None  # The original is preserved.


def test_complete_history_is_never_replaced_by_cheaper_or_more_successful_supplement(report):
    original = observation(300.)
    original.update(all_requests_successful=False, measurement_evidence_valid=False)
    rerun = observation(50., path='new', attempt=1)
    selected, ledger = report.select_baseline_observations([original], [rerun])
    assert selected == [original]
    assert ledger[0]['selection_reason'] == 'historical_complete'


@pytest.mark.parametrize('damage', ['tail', 'sum', 'coverage_flag'])
def test_service_only_or_inconsistent_energy_does_not_fill_gap(report, damage):
    first = observation(90., path='first', attempt=1)
    if damage == 'tail':
        first['tail_energy_kj'] = None
    elif damage == 'sum':
        first['total_energy_kj'] += 1
    else:
        first['energy_measurement_complete'] = False
    second = observation(150., path='second', attempt=2)
    selected, _ = report.select_baseline_observations([observation()], [first, second])
    assert selected == [second]
    selected, ledger = report.select_baseline_observations([observation()], [first])
    assert selected[0]['total_energy_kj'] is None
    assert ledger[0]['selection_reason'] == 'no_complete_supplement'


def test_wrong_original_receipt_or_condition_and_ambiguous_attempt_fail_closed(report):
    row = observation(100., path='new', attempt=1)
    wrong = deepcopy(row)
    wrong['supplement_of']['sha256'] = 'different'
    with pytest.raises(ValueError, match='selected historical receipt'):
        report.select_baseline_observations([observation()], [wrong])
    wrong = deepcopy(row)
    wrong['comparison_identity']['trace_sha256'] = 'another-trace'
    with pytest.raises(ValueError, match='comparison identity'):
        report.select_baseline_observations([observation()], [wrong])
    with pytest.raises(ValueError, match='duplicate attempt'):
        report.select_baseline_observations([observation()], [row, deepcopy(row)])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return binding(path)


def source(root, *, changed=False):
    names = ['pdblend_baselines/runner.py', 'pdblend/measure/power.py', 'pdblend/measure/backends.py',
             'pdblend/bench/client.py', 'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py']
    files = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('value = 2\n' if changed and 'runner' in name else 'value = 1\n')
        files[name] = binding(path)['sha256']
    return write(root / 'manifest.json', dict(files=files, source_sha256=digest(files)))


@pytest.fixture
def declared(tmp_path):
    source_ref = source(tmp_path / 'source')
    original = dict(name='old', system='dynamollm', model_id='Qwen2.5-7B-Instruct', dataset='alpaca',
                    scale=.25, rate_rps=2., seed=701, duration_s=150., slo={'ttft_s': 1., 'tpot_s': .1},
                    trace={'path': 'trace', 'sha256': 'trace-sha'}, source_manifest=source_ref,
                    inputs={'source_manifest': source_ref, 'system_config': {'sha256': 'config'}},
                    engine_identity={'measurement_source_sha256': 'meter', 'metering_execution': 'thread'})
    old_dir = tmp_path / 'history'
    old_point = write(old_dir / 'point.json', original)
    old_receipt = write(old_dir / 'receipt.json', dict(artifacts={'point.json': old_point['sha256']}, point_sha256=digest(original)))
    supplement = dict(deepcopy(original), name='supplement', original_point_name='old',
                      energy_supplement_of=old_receipt, revision='source',
                      comparison_selection='first_predeclared_complete_service_and_tail_attempt')
    supplement['engine_identity']['metering_execution'] = 'isolated_process'
    campaign_path = tmp_path / 'campaign.json'
    group = dict(session_id='session', points=[supplement])
    campaign = write(campaign_path, dict(points=[supplement], groups=[group]))
    jobs = write(tmp_path / 'jobs.json', [dict(job_id='declared-job', payload=dict(
        comparison_campaign=str(campaign_path), session_id='session'))])
    preparation = tmp_path / 'preparation.json'
    write(preparation, dict(campaign=campaign, jobs=jobs))
    return dict(point=supplement, original=original, prep=preparation, attempts=tmp_path / 'attempts')


def record(declared, ordinal, *, suffix='fixed', point=None, metrics=True):
    point = point or declared['point']
    directory = declared['attempts'] / 'declared-job' / f'attempt-{ordinal:04d}-{suffix}' / 'session/windows/supplement'
    point_ref = write(directory / 'point.json', point)
    return Path(write(directory / 'receipt.json', dict(artifacts={'point.json': point_ref['sha256']},
        point_sha256=digest(point), result={'metrics': {}} if metrics else {}))['path'])


def test_bound_job_collection_uses_numeric_order_and_keeps_failed_attempt_audit(report, declared, monkeypatch):
    record(declared, 1, metrics=False)
    record(declared, 10)
    record(declared, 2)
    # An unrelated job must not be discovered by a broad glob.
    write(declared['attempts'] / 'not-declared/attempt-0001-x/session/windows/foreign/receipt.json', {})
    seen = []
    def analyze(path, *, supplement):
        ordinal = int(path.parents[3].name.split('-')[1])
        seen.append(ordinal)
        row = observation(200. if ordinal == 2 else 50., path=str(path), attempt=ordinal)
        return row, [], {'receipt_path': str(path)}, declared['point']
    monkeypatch.setattr(report, 'read_observation', analyze)
    rows, audits, _, _, proof = report.collect_supplements(declared['prep'], declared['attempts'])
    assert seen == [2, 10]
    assert len(rows) == 2 and len(audits) == 3
    assert audits[0]['status'] == 'no_complete_metrics'
    assert proof['declared_points'] == 1
    assert proof['source_reviews'][0]['baseline_core_and_meter_unchanged']


def test_exact_predeclared_point_and_unique_attempt_ordinal_are_required(report, declared):
    changed = deepcopy(declared['point'])
    changed['rate_rps'] = 3.
    path = record(declared, 1, point=changed)
    with pytest.raises(ValueError, match='predeclared point'):
        report.collect_supplements(declared['prep'], declared['attempts'])
    path.unlink()
    record(declared, 1, suffix='different')
    with pytest.raises(ValueError, match='ambiguous.*ordinal'):
        report.collect_supplements(declared['prep'], declared['attempts'])


def test_interrupted_raw_artifact_is_audited_but_corrupt_existing_artifact_is_not_ignored(report, declared, monkeypatch):
    record(declared, 1)
    record(declared, 2)
    def analyze(path, *, supplement):
        ordinal = int(path.parents[3].name.split('-')[1])
        if ordinal == 1:
            raise FileNotFoundError('comparison-metering.json')
        return observation(100., path=str(path), attempt=2), [], {}, declared['point']
    monkeypatch.setattr(report, 'read_observation', analyze)
    rows, audits, *_ = report.collect_supplements(declared['prep'], declared['attempts'])
    assert len(rows) == 1 and rows[0]['supplement_attempt'] == 2
    assert audits[0]['status'] == 'missing_raw_artifact'
    def corrupt(path, *, supplement):
        raise AssertionError('Receipt artifact hash mismatch')
    monkeypatch.setattr(report, 'read_observation', corrupt)
    with pytest.raises(AssertionError, match='hash mismatch'):
        report.collect_supplements(declared['prep'], declared['attempts'])


def test_bound_preparation_rejects_manifest_tampering(report, declared):
    jobs = declared['prep'].parent / 'jobs.json'
    jobs.write_text('[]')
    with pytest.raises(ValueError):
        report.collect_supplements(declared['prep'], declared['attempts'])


def test_source_or_configuration_drift_cannot_be_called_a_baseline_supplement(report, declared, tmp_path):
    changed = deepcopy(declared['point'])
    changed['source_manifest'] = source(tmp_path / 'different-source', changed=True)
    with pytest.raises(ValueError, match='core or energy'):
        report.baseline_core_review(declared['original'], changed)
    changed = deepcopy(declared['point'])
    changed['inputs']['system_config']['sha256'] = 'different-config'
    with pytest.raises(ValueError, match='settings'):
        report.baseline_core_review(declared['original'], changed)
    source_file = Path(declared['point']['source_manifest']['path']).parent / 'pdblend/measure/power.py'
    source_file.write_text('tampered = True\n')
    with pytest.raises(ValueError, match='checksum'):
        report.baseline_core_review(declared['original'], declared['point'])


def test_full_matrix_selection_keeps_36_conditions_and_144_baselines(report):
    rows = []
    for model in ['7B', '14B', '32B']:
        for dataset in ['alpaca', 'sharegpt', 'longbench']:
            for scale in [.25, .5, .75, 1.]:
                for system in ['mixed', 'distserve', 'dynamollm', 'ecoserve']:
                    row = observation(100., path=f'{model}/{dataset}/{scale}/{system}')
                    row.update(model=model, dataset=dataset, rate_scale=scale, system=system, series=system)
                    rows.append(row)
    selected, ledger = report.select_baseline_observations(rows, [])
    assert len(selected) == len(ledger) == 144
    assert len({(r['model'], r['dataset'], r['rate_scale']) for r in selected}) == 36
    assert all(r['selection_reason'] == 'historical_complete' for r in ledger)


def test_watcher_expands_only_hash_bound_pd_campaign_jobs(report, tmp_path):
    campaign = write(tmp_path / 'campaign.json', dict(points=[dict(system='pdblend', repair_experiment={'arm': 'candidate'})]))
    jobs = write(tmp_path / 'jobs.json', [dict(job_id='fixed-pd-job', payload={'comparison_campaign': campaign['path']})])
    prep = tmp_path / 'preparation.json'
    write(prep, dict(campaign=campaign, jobs=jobs))
    campaigns, directories, refs = report.repair_preparation_inputs([prep], tmp_path / 'attempts')
    assert campaigns == [tmp_path / 'campaign.json']
    assert directories == [tmp_path / 'attempts/fixed-pd-job']
    assert refs[0]['preparation'] == binding(prep)
    (tmp_path / 'jobs.json').write_text('[]')
    with pytest.raises(ValueError):
        report.repair_preparation_inputs([prep], tmp_path / 'attempts')
