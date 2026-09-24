import asyncio
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from pdblend.bench.resident_session import ResidentGroupSession, engine_signature, write_new, digest, file_sha
from pdblend.bench.comparison_campaign import group_points, rank_rows, export, MODELS, SYSTEMS, PROTOCOL


def identity():
    return dict(model_hash='a'*64, tokenizer_hash='b'*64, image_digest='sha256:image',
        runtime_source_sha256='c'*64, entrypoint='native', worker_extension=None,
        dtype='bfloat16', environment={'chunked_prefill': True},
        instances=[dict(instance_id='i0', tp=1, pp=1, gpu_uuids=['GPU-0'],
                        launch_options=dict(kv_connector=None, max_num_seqs=32))])


def points():
    return [dict(name=f'p{i}', model_id=MODELS[0], system=system, dataset=ds, scale=scale,
                 engine_identity=identity(), blockers=[]) for i, (system, ds, scale) in enumerate([
        ('mixed', 'sharegpt', 1.), ('mixed', 'alpaca', .5), ('mixed', 'longbench', .25),
        ('pdblend', 'alpaca', .5)])]


def group():
    g = group_points(points())[0]
    return g


class Adapter:
    def __init__(self, fail_reset=0, measured_failure=False):
        self.starts = self.stops = self.resets = self.runs = 0
        self.fail_reset, self.measured_failure = fail_reset, measured_failure

    async def start(self, group):
        self.starts += 1
        self.started_points=[p['name'] for p in group['points']]
        return {'engine_loads': 1}

    async def reset(self, point):
        self.resets += 1
        return {'passed': self.resets != self.fail_reset}

    async def execute(self, point, out):
        self.runs += 1
        write_new(out/'raw.json', {'tokens': [1, 2]})
        return dict(evidence_valid=True, formal_eligible=False, metrics={'slo_pass': not self.measured_failure})

    async def drain(self, point):
        return {'passed': True}

    async def close(self):
        self.stops += 1
        return {'passed': True}


def test_group_order_and_independence():
    groups = group_points(points())
    assert len(groups) == 2
    assert [p['scale'] for p in groups[0]['points']] == [.5, .25, 1.]
    assert groups[1]['points'][0]['system'] == 'pdblend'


@pytest.mark.parametrize('field,value', [('kv_connector','P2pNcclConnector'), ('max_num_seqs',16)])
def test_launch_changes_cannot_reuse(field, value):
    a, b = identity(), identity()
    b['instances'][0]['launch_options'][field] = value
    assert engine_signature(a) != engine_signature(b)


def test_missing_identity_rejected():
    value = identity(); del value['worker_extension']
    with pytest.raises(ValueError):
        engine_signature(value)


def test_load_once_freeze_measured_slo_failure(tmp_path):
    adapter = Adapter(measured_failure=True)
    result = asyncio.run(ResidentGroupSession(group(), adapter, tmp_path/'session').run())
    assert result['complete'] and adapter.starts == adapter.stops == 1 and adapter.runs == 3
    path = tmp_path/'session/windows/p1/receipt.json'
    row = json.loads(path.read_text())
    assert row['baseline_frozen'] and not row['result']['metrics']['slo_pass']


def test_stop_after_window_commits_receipt_and_cleans_up(tmp_path):
    stop_file = tmp_path/'stop'
    class StoppingAdapter(Adapter):
        async def drain(self, point):
            stop_file.touch()
            return {'passed': True}
    adapter = StoppingAdapter()
    out = tmp_path/'session'
    result = asyncio.run(ResidentGroupSession(group(), adapter, out, stop_after_window=stop_file).run())
    assert result['status'] == 'interrupted' and not result['complete']
    assert result['continuation_required'] and result['cleanup']['passed']
    assert adapter.runs == adapter.stops == 1
    assert len(result['windows']) == 1 and len(result['pending_points']) == 2
    frozen = result['windows'][0]
    assert file_sha(frozen['path']) == frozen['sha256']
    resumed = asyncio.run(ResidentGroupSession(group(), Adapter(), tmp_path/'resume', previous=[out]).run())
    assert resumed['complete'] and len(resumed['windows']) == 2


def test_reset_failure_quarantines_and_stops(tmp_path):
    adapter = Adapter(fail_reset=2)
    result = asyncio.run(ResidentGroupSession(group(), adapter, tmp_path/'session').run())
    assert not result['complete'] and result['quarantined']
    assert adapter.runs == 1 and adapter.stops == 1


def test_resume_skips_frozen_windows(tmp_path):
    original = tmp_path/'first'
    first = Adapter(fail_reset=2)
    asyncio.run(ResidentGroupSession(group(), first, original).run())
    second = Adapter()
    result = asyncio.run(ResidentGroupSession(group(), second, tmp_path/'second', previous=[original]).run())
    assert result['complete'] and second.runs == 2 and len(result['skipped']) == 1
    assert second.started_points==['p2','p0']


def clock_failure_result():
    gate='eco.observed_active_frequency'
    return dict(evidence_valid=False,formal_eligible=False,missing_gates=[gate],metrics=dict(slo_pass=False),
        acceptance=dict(evidence_valid=False,formal_eligible=False,missing_gates=[gate],
            gate_failures={gate:'actual active GPU frequency unproven or different'},
            preflight=dict(preflight_ready=True,missing_gates=[],gate_failures={}),checked_gates=['eco.observation_boundary','eco.fixed_fleet',
                'eco.startup','eco.reset','eco.all_rank_drain','metering.raw_eight_gpu_window',
                'metering.isolated_process_method','eco.raw_protocol_and_canonical_metrics']+
                ['raw.'+n for n in ('trace','native_result','events','power','canonical_requests','metering','startup_qualification','reset','drain')]+
                ['binding.'+n for n in ('native_result','startup_qualification','reset','metering','drain','trace')]))


@pytest.mark.parametrize('fault',[None,'unapproved','protocol','meter','source','reset','drain','execute','raw','binding','inputs'])
def test_only_isolated_clock_rejection_can_advance_after_clean_drain(tmp_path,fault):
    g=group()
    for p in g['points']:
        p.update(system='ecoserve',qualification_mode='ecoserve_native_bootstrap',metering_execution='isolated_process',
                 observation_failure_policy='continue_after_verified_frequency_rejection')
    if fault=='unapproved':g['points'][0].pop('observation_failure_policy')
    class ClockAdapter(Adapter):
        async def execute(self,point,out):
            await super().execute(point,out)
            if fault=='execute':raise RuntimeError('HTTP failed before result')
            value=clock_failure_result()
            if fault in ('protocol','meter'):
                gate='eco.raw_protocol_and_canonical_metrics' if fault=='protocol' else 'metering.raw_eight_gpu_window'
                value['missing_gates'].append(gate);value['acceptance']['missing_gates'].append(gate)
                value['acceptance']['gate_failures'][gate]='real second failure'
            elif fault=='source':value['acceptance']['preflight']['preflight_ready']=False
            elif fault in ('raw','binding'):value['acceptance']['checked_gates'].remove(fault+'.trace')
            elif fault=='inputs':value['acceptance']['preflight']['gate_failures']={'config':'different'}
            return value
        async def drain(self,point):return dict(passed=fault!='drain')
    adapter=ClockAdapter(fail_reset=1 if fault=='reset' else 0)
    report=asyncio.run(ResidentGroupSession(g,adapter,tmp_path/'session').run())
    if fault is None:
        assert report['complete'] and not report['all_observations_valid'] and report['invalid_observations']==3
        assert adapter.runs==adapter.resets==3 and adapter.starts==adapter.stops==1
        for window in report['windows']:
            receipt=json.loads(Path(window['path']).read_text())
            assert not receipt['evidence_valid'] and not receipt['baseline_frozen'] and receipt['cleanup_passed']
            assert receipt['continuation'].startswith('next_window_requires_fresh_reset')
    else:
        assert not report['complete'] and report['quarantined'] and adapter.runs<=1


def test_resume_rejects_changed_raw(tmp_path):
    original = tmp_path/'first'
    asyncio.run(ResidentGroupSession(group(), Adapter(), original).run())
    (original/'windows/p1/run/raw.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        asyncio.run(ResidentGroupSession(group(), Adapter(), tmp_path/'second', previous=[original]).run())


def ranking_fixture():
    return [dict(model_id=MODELS[0], dataset='alpaca', rate_scale=.5, seed=701, duration_s=150,
                 trace_sha256='d'*64, measurement_protocol_version='v1', system=s,
                 evidence_valid=True, formal_eligible=True, slo_pass=True, energy_service_j=10+i,
                 common_clock_evidence='pass',common_clock_scope='observed_requested_active_clock/v1',
                 baseline_frozen=s != 'pdblend', revision='initial',
                 receipt_path='/evidence/' + s, receipt_sha256=str(i) * 64,
                 image_digest='image', runtime_source_sha256='runtime', measurement_source_sha256='meter',
                 model_hash='model', tokenizer_hash='tokenizer', gpu_uuids=['GPU-'+str(i) for i in range(8)],
                 slo_ttft_s=1., slo_tpot_s=.1) for i, s in enumerate(SYSTEMS)]


def test_infeasible_baseline_excluded():
    rows = ranking_fixture(); rows[0]['slo_pass'] = False
    rank_rows(rows)
    assert not rows[0]['rank_eligible']
    pd = next(r for r in rows if r['system'] == 'pdblend')
    assert pd['best_feasible_baseline'] == 'distserve'


@pytest.mark.parametrize('evidence',['unknown','fail',None])
def test_unproven_realized_clock_blocks_full_ranking_without_rewriting_frozen_values(evidence):
    rows=ranking_fixture();baseline=rows[0];before=dict(baseline)
    baseline['common_clock_evidence']=evidence
    rank_rows(rows)
    assert all(not r['rank_eligible'] and r['energy_rank']=='' for r in rows)
    candidate=next(r for r in rows if r['system']=='pdblend')
    assert candidate['comparison_status']=='clock_evidence_unqualified'
    assert baseline['evidence_valid']==before['evidence_valid'] and baseline['baseline_frozen']
    assert baseline['energy_service_j']==before['energy_service_j'] and baseline['slo_pass']


def test_missing_system_and_ambiguous_attempt_never_select_best():
    rows = ranking_fixture()[:-1]
    rank_rows(rows)
    assert all(not r['rank_eligible'] for r in rows)
    rows = ranking_fixture(); rows.append(deepcopy(rows[-1]))
    rank_rows(rows)
    assert all(r['comparison_status'] == 'ambiguous_attempts' for r in rows)


def test_mismatched_runtime_not_ranked():
    rows = ranking_fixture(); rows[-1]['runtime_source_sha256'] = 'different'
    rank_rows(rows)
    assert all(not r['rank_eligible'] for r in rows)


def test_each_revision_uses_same_baselines_and_reports_tail_reversal():
    rows = ranking_fixture()
    original = next(r for r in rows if r['system'] == 'pdblend')
    original.update(energy_service_j=9., energy_tail_j=0.)
    revised = dict(deepcopy(original), revision='optimized', energy_service_j=8., energy_tail_j=5.)
    rows.append(revised)
    for row in rows:
        if row['system'] != 'pdblend':
            row['energy_tail_j'] = 1.
    rank_rows(rows)
    assert original['energy_rank'] == revised['energy_rank'] == 1
    assert original['comparison_baseline_receipts'] == revised['comparison_baseline_receipts']
    assert len(original['comparison_baseline_receipts']) == 4
    assert original['tail_reverses_saving'] is False
    assert revised['tail_reverses_saving'] is True
    assert revised['pdblend_saving_vs_best_feasible_baseline'] == pytest.approx(.2)
    assert revised['pdblend_saving_with_tail_vs_best_feasible_baseline'] == pytest.approx(1-13/11)
    mixed = next(r for r in rows if r['system'] == 'mixed')
    assert mixed['energy_rank'] == ''
    assert mixed['energy_rank_by_revision'] == {'initial': 2, 'optimized': 2}
    assert mixed['baseline_energy_rank'] == 1


def test_ambiguous_revision_does_not_discard_other_candidate():
    rows = ranking_fixture()
    original = next(r for r in rows if r['system'] == 'pdblend')
    duplicate = deepcopy(original)
    revised = dict(deepcopy(original), revision='optimized', energy_service_j=8.)
    rows += [duplicate, revised]
    rank_rows(rows)
    assert original['comparison_status'] == duplicate['comparison_status'] == 'ambiguous_attempts'
    assert not original['rank_eligible'] and not duplicate['rank_eligible']
    assert revised['energy_rank'] == 1 and revised['comparison_status'] == 'single_observation'
    assert revised['tail_reverses_saving'] is None


def test_baselines_must_be_frozen_and_second_baseline_attempt_is_ambiguous():
    rows = ranking_fixture(); rows[0]['baseline_frozen'] = False
    rank_rows(rows)
    assert all(r['comparison_status'] == 'baseline_not_frozen' for r in rows)
    assert not any(r['rank_eligible'] for r in rows)
    rows = ranking_fixture(); rows.append(deepcopy(rows[0]))
    rank_rows(rows)
    assert all(r['comparison_status'] == 'ambiguous_attempts' for r in rows)


def test_bad_revision_identity_does_not_poison_valid_revision():
    rows = ranking_fixture()
    revised = dict(deepcopy(rows[-1]), revision='different-runtime', runtime_source_sha256='other')
    rows.append(revised)
    rank_rows(rows)
    assert rows[-2]['comparison_status'] == 'single_observation'
    assert rows[-2]['rank_eligible']
    assert revised['comparison_status'] == 'identity_mismatch' and not revised['rank_eligible']


def export_point(revision='initial'):
    return dict(name='7b-pdblend-alpaca-x0.5-seed701', model_id=MODELS[0], dataset='alpaca',
                system='pdblend', scale=.5, rate_rps=1., seed=701, duration_s=150.,
                revision=revision, status='prepared', blockers=[], trace={'sha256': 'd'*64},
                measurement_protocol_version=PROTOCOL, slo={'ttft_s': 1., 'tpot_s': .1})


def export_receipt(root, point, *, bind_point=True):
    window = root/'windows'/point['name']
    window.mkdir(parents=True)
    result = dict(evidence_valid=True, formal_eligible=True, identity={},
                  metrics=dict(slo_pass=True, energy_service_j=10.))
    write_new(window/'result.json', result)
    if bind_point:
        write_new(window/'point.json', point)
    receipt = dict(point=point['name'], point_sha256=digest(point), result=result,
                   session_id=root.name, engine_signature='engine',
                   evidence_valid=True, cleanup_passed=True, baseline_frozen=False,
                   artifacts={p.name: file_sha(p) for p in window.iterdir()})
    write_new(window/'receipt.json', receipt)
    return window/'receipt.json'


def export_csv(tmp_path, current, roots):
    campaign = tmp_path/'campaign.json'
    campaign.write_text(json.dumps(dict(campaign_id='test', points=[current])))
    target = tmp_path/'compare.csv'
    summary = export(campaign, target, session_roots=roots)
    with target.open() as stream:
        return summary, list(csv.DictReader(stream))


def test_startup_failure_exposes_bound_blocker_for_unstarted_windows(tmp_path):
    point = export_point()
    point.update(status='prepared', blockers=[])
    session = tmp_path/'session'
    report = dict(schema='resident-group-session/v1', session_id='failed',
                  engine_signature='engine', planned_points={point['name']:digest(point)},
                  complete=False, error='native startup failed')
    write_new(session/'completion.json', report)
    campaign = tmp_path/'campaign.json'
    write_new(campaign, dict(campaign_id='test', points=[point], groups=[dict(
        session_id='failed', engine_signature='engine', points=[point])]))
    target = tmp_path/'compare.csv'
    export(campaign, target, session_roots=[session])
    row = next(csv.DictReader(target.open()))
    assert row['status'] == 'blocked' and 'native startup failed' in row['failure_reason']
    assert row['session_completion_sha256'] == file_sha(session/'completion.json')
    assert row['evidence_valid'] == 'False' and row['energy_service_j'] == ''


def test_old_startup_failure_does_not_block_revised_point_using_same_engine(tmp_path):
    old=export_point('old');current=export_point('fixed')
    current.update(status='prepared',blockers=[])
    session=tmp_path/'session'
    write_new(session/'completion.json',dict(schema='resident-group-session/v1',session_id='shared',
        engine_signature='same-engine',planned_points={old['name']:digest(old)},
        complete=False,error='old startup bug'))
    campaign=tmp_path/'campaign.json'
    write_new(campaign,dict(campaign_id='new',points=[current],groups=[dict(
        session_id='shared',engine_signature='same-engine',points=[current])]))
    target=tmp_path/'compare.csv';export(campaign,target,session_roots=[session])
    row=next(csv.DictReader(target.open()))
    assert row['status']=='prepared' and row['failure_reason']==''
    previous=json.loads(row['prior_session_failures'])
    assert previous[0]['error']=='old startup bug' and previous[0]['matches_current_revision'] is False


def test_export_keeps_bound_historical_revisions_and_deduplicates_roots(tmp_path):
    old, current = export_point(), export_point('optimized')
    export_receipt(tmp_path/'old', old)
    export_receipt(tmp_path/'new', current)
    summary, rows = export_csv(tmp_path, current, [tmp_path, tmp_path/'old'])
    assert summary['rows'] == summary['measured'] == 2
    assert {r['revision'] for r in rows} == {'initial', 'optimized'}
    assert len({r['receipt_path'] for r in rows}) == 2


def test_stale_same_name_receipt_does_not_remove_current_placeholder(tmp_path):
    old, current = export_point(), export_point('optimized')
    export_receipt(tmp_path/'old', old, bind_point=False)
    summary, rows = export_csv(tmp_path, current, [tmp_path/'old'])
    assert summary['rows'] == 2 and summary['measured'] == 0
    placeholder = next(r for r in rows if r['status'] == 'prepared')
    assert placeholder['revision'] == 'optimized' and placeholder['point_sha256'] == digest(current)
    unresolved = next(r for r in rows if r['status'] == 'unresolved_historical_receipt')
    assert unresolved['revision'] == unresolved['trace_sha256'] == ''
    assert unresolved['formal_eligible'] == unresolved['rank_eligible'] == 'False'


@pytest.mark.parametrize('mutation,match', [
    ('embedded_result', 'embedded window result'),
    ('empty_artifacts', 'lacks bound artifacts'),
    ('point_digest', 'bound historical point'),
    ('artifact_bytes', 'artifact changed'),
])
def test_export_rejects_inconsistent_evidence(tmp_path, mutation, match):
    point = export_point()
    path = export_receipt(tmp_path/'session', point)
    receipt = json.loads(path.read_text())
    if mutation == 'embedded_result':
        receipt['result']['metrics']['energy_service_j'] = .001
    elif mutation == 'empty_artifacts':
        receipt['artifacts'] = {}
    elif mutation == 'point_digest':
        receipt['point_sha256'] = 'wrong'
    else:
        (path.parent/'result.json').write_text('{}')
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=match):
        export_csv(tmp_path, point, [tmp_path/'session'])


def test_export_cannot_override_frozen_duration_even_with_matching_result_hash(tmp_path):
    point = export_point()
    path = export_receipt(tmp_path/'session', point)
    receipt = json.loads(path.read_text())
    receipt['result']['metrics']['duration_s'] = 100.
    (path.parent/'result.json').write_text(json.dumps(receipt['result']))
    receipt['artifacts']['result.json'] = file_sha(path.parent/'result.json')
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='overrides frozen point'):
        export_csv(tmp_path, point, [tmp_path/'session'])


def test_export_requires_cleanup_even_when_metric_receipt_claims_valid(tmp_path):
    point = export_point()
    path = export_receipt(tmp_path/'session', point)
    receipt = json.loads(path.read_text()); receipt['cleanup_passed'] = False
    path.write_text(json.dumps(receipt))
    summary, rows = export_csv(tmp_path, point, [tmp_path/'session'])
    assert summary['measured'] == 0
    assert rows[0]['status'] == 'invalid_measurement' and rows[0]['formal_eligible'] == 'False'


def completed_session(root, receipts):
    report = dict(session_id=root.name, engine_signature='engine', complete=True, status='passed',
                  started_s=100., finished_s=500., cleanup_s=1.5, skipped=[],
                  startup=dict(engine_loads=8, engine_load_cycles=1, engine_load_s=60., load_lock_wait_s=4.),
                  windows=[dict(point=json.loads(p.read_text())['point'], path=str(p), sha256=file_sha(p),
                                evidence_valid=True) for p in receipts])
    write_new(root/'completion.json', report)
    return report


def test_session_costs_are_shared_once_and_window_warmup_is_part_of_reset(tmp_path):
    root = tmp_path/'session'
    first, second = export_point(), dict(export_point(), name='other-point', dataset='sharegpt')
    receipts = [export_receipt(root, p) for p in (first, second)]
    for path in receipts:
        write_new(path.parent/'reset.json', dict(reset_s=3., warmup_s=2.5))
        receipt = json.loads(path.read_text())
        receipt['artifacts']['reset.json'] = file_sha(path.parent/'reset.json')
        path.write_text(json.dumps(receipt))
    completed_session(root, receipts)
    campaign = tmp_path/'campaign.json'
    campaign.write_text(json.dumps(dict(campaign_id='costs', points=[first, second])))
    target = tmp_path/'compare.csv'
    export(campaign, target, session_roots=[root])
    with target.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for row in rows:
        assert row['session_cost_scope'] == 'session_attempt_shared_do_not_sum_rows'
        assert row['session_cost_scope_id'] == str(root)
        assert row['session_cost_evidence_status'] == 'bound_completion'
        assert row['session_engine_loads'] == '8' and row['session_engine_load_cycles'] == '1'
        assert row['session_engine_load_s'] == '60.0' and row['session_load_lock_wait_s'] == '4.0'
        assert row['session_total_s'] == '400.0' and row['session_cleanup_s'] == '1.5'
        assert row['session_windows_attempted'] == row['session_windows_measured'] == '2'
        assert row['session_completion_sha256'] == file_sha(root/'completion.json')
        assert row['window_reset_s'] == '3.0' and row['window_warmup_s'] == '2.5'
        assert row['window_warmup_included_in_reset'] == 'True'


def test_running_session_does_not_invent_load_or_cleanup_costs(tmp_path):
    point = export_point(); root = tmp_path/'running'
    export_receipt(root, point)
    _, rows = export_csv(tmp_path, point, [root])
    assert rows[0]['session_cost_evidence_status'] == 'pending_completion'
    for key in ('session_engine_loads', 'session_engine_load_s', 'session_total_s', 'session_cleanup_s'):
        assert rows[0][key] == ''


def test_completion_must_bind_exact_window_receipt_before_costs_are_used(tmp_path):
    point = export_point(); root = tmp_path/'completed'
    path = export_receipt(root, point)
    completed_session(root, [path])
    receipt = json.loads(path.read_text()); receipt['window_index'] = 99
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='does not bind window receipt'):
        export_csv(tmp_path, point, [root])


def bind_utilization_meter(path, *, mismatched_uuid=False):
    uuids = ['GPU-'+str(i) for i in range(8)]
    per_gpu = {uuid: dict(coverage_fraction=1., status='complete', samples=1500,
                         missing_samples=0, max_gap_s=.15, max_missing_gap_s=0.) for uuid in uuids}
    per_gpu['GPU-2'].update(coverage_fraction=.9, status='missing', samples=1350,
                            missing_samples=2, max_gap_s=1.2, max_missing_gap_s=.8)
    write_new(path.parent/'run/comparison-metering.json',
              dict(gpu_uuids=uuids, service=dict(utilization=dict(per_gpu=per_gpu))))
    receipt = json.loads(path.read_text())
    receipt['artifacts']['run/comparison-metering.json'] = file_sha(path.parent/'run/comparison-metering.json')
    receipt['result']['identity']['gpu_uuids'] = list(reversed(uuids)) if mismatched_uuid else uuids
    (path.parent/'result.json').write_text(json.dumps(receipt['result']))
    receipt['artifacts']['result.json'] = file_sha(path.parent/'result.json')
    path.write_text(json.dumps(receipt))


def test_export_utilization_coverage_and_counts_are_per_physical_gpu(tmp_path):
    point = export_point(); root = tmp_path/'session'
    path = export_receipt(root, point)
    bind_utilization_meter(path)
    before = (path.parent/'result.json').read_bytes()
    _, rows = export_csv(tmp_path, point, [root])
    row = rows[0]
    assert row['gpu0_util_coverage_fraction'] == '1.0' and row['gpu0_util_status'] == 'complete'
    assert row['gpu0_util_samples'] == '1500' and row['gpu0_util_missing_samples'] == '0'
    assert row['gpu2_util_coverage_fraction'] == '0.9' and row['gpu2_util_status'] == 'missing'
    assert row['gpu2_util_samples'] == '1350' and row['gpu2_util_missing_samples'] == '2'
    assert row['gpu2_util_max_gap_s'] == '1.2' and row['gpu2_util_max_missing_gap_s'] == '0.8'
    assert (path.parent/'result.json').read_bytes() == before


def test_export_missing_or_unbound_meter_keeps_per_gpu_fields_empty(tmp_path):
    point = export_point(); root = tmp_path/'session'
    path = export_receipt(root, point)
    # Mere file existence cannot make data trusted. This file is intentionally
    # outside the immutable artifact map and must never be opened or parsed.
    (path.parent/'run').mkdir()
    (path.parent/'run/comparison-metering.json').write_text('not bound, not JSON')
    _, rows = export_csv(tmp_path, point, [root])
    assert all(rows[0][f'gpu{i}_util_{field}'] == '' for i in range(8)
               for field in ('coverage_fraction', 'status', 'samples', 'max_gap_s'))


def test_export_refuses_utilization_from_different_physical_gpu_order(tmp_path):
    point = export_point(); root = tmp_path/'session'
    path = export_receipt(root, point)
    bind_utilization_meter(path, mismatched_uuid=True)
    with pytest.raises(ValueError, match='UUID order differs'):
        export_csv(tmp_path, point, [root])


def watch_fixture(root, *, frozen=True):
    from pdblend.bench.comparison_campaign import binding, SCALES
    points = []
    for model in MODELS:
        for dataset in ('alpaca', 'sharegpt', 'longbench'):
            for system in SYSTEMS:
                for scale in SCALES:
                    missing = model == MODELS[2] and dataset == 'longbench'
                    points.append(dict(export_point(), name=f'{model}-{system}-{dataset}-{scale}',
                        model_id=model, dataset=dataset, system=system, scale=scale,
                        trace=None if missing else {'sha256': 'd'*64}, rate_rps=None if missing else 1.,
                        status='blocked' if missing else 'prepared', blockers=['missing_anchor'] if missing else []))
    ancestor = root/'original/campaign.json'
    original = dict(campaign_id='original', points=points, groups=[], schema='resident-comparison-campaign/v1',
                    seed=701, duration_s=150., scales=list(SCALES), measurement_protocol_version=PROTOCOL)
    write_new(ancestor, original)
    parent = root/'eco/campaign.json'
    write_new(parent, dict(original, campaign_id='eco', parent_campaign=binding(ancestor),
                           execution_campaigns=[str(ancestor)]))
    source = root/'source/manifest.json'; write_new(source, {'source_sha256': 'source-sha'})
    plan = root/'plan.json'
    write_new(plan, dict(schema='longbench-mixed-combined/v1', parent_campaign=binding(parent),
                         source_manifest=binding(source)))
    attempt = root/'arbitrary-attempt-directory'
    payload = dict(comparison_campaign=str(parent), combined_plan=binding(plan),
                   overlay_campaign_output='combined/campaign.json', session_output='combined/session',
                   source_sha256='source-sha', source_snapshot=str(source.parent))
    job = dict(job_id='combined32', status='running', payload=payload)
    lease = dict(job_id='combined32', lease_id='lease', attempt_dir=str(attempt))
    write_new(attempt/'manifest.json', dict(immutable=True, job_id=job['job_id'],
                                           lease_id=lease['lease_id'], payload=payload))
    overlay = deepcopy(json.loads(parent.read_text()))
    overlay.update(campaign_id='combined', parent_campaign=binding(parent), combined_plan=binding(plan),
                   execution_source_manifest=binding(source), anchor_outcome='confirmed')
    for point in overlay['points']:
        if point['trace'] is None:
            point.update(trace={'sha256': 'e'*64}, rate_rps=.125*point['scale'], status='prepared', blockers=[])
    group = dict(session_id='session', engine_signature='engine', points=[p for p in overlay['points']
                 if p['model_id'] == MODELS[2] and p['system'] == 'mixed'])
    overlay['groups'] = [group]
    output = attempt/'combined'
    write_new(output/'campaign.json', overlay); write_new(output/'group.json', group)
    write_new(output/'anchor/completion.json', dict(hardware_executed=True, cleanup_errors=[],
                                                   status='passed', complete=True))
    freeze = dict(schema='longbench-mixed-combined/v1', parent_campaign=binding(parent),
        campaign=binding(output/'campaign.json'), group=binding(output/'group.json'),
        anchor=binding(output/'anchor/completion.json'), anchor_outcome='confirmed', evaluation_has_started=False,
        point_count=12, inherited_points=[dict(name=p['name'], sha256=digest(p), trace=p['trace'])
            for p in original['points'] if p['model_id'] == MODELS[2] and p['system'] == 'mixed' and p['trace']])
    if frozen:write_new(output/'freeze.json', freeze)
    old_attempt = root/'older-lease'
    old_job = dict(job_id='original-job', status='running', payload={'comparison_campaign':str(ancestor)})
    queue = dict(jobs={j['job_id']:j for j in [old_job, job]}, leases={'lease':lease,
        'old':dict(job_id='original-job', lease_id='old', attempt_dir=str(old_attempt))})
    return dict(parent=parent, output=output, job=job, queue=queue, freeze=freeze,
                old_session=old_attempt/'session', group=group)


def test_watch_adopts_only_frozen_overlay_and_preserves_all_180_rows(tmp_path):
    from pdblend.bench.comparison_campaign import comparison_watch_inputs
    fixture = watch_fixture(tmp_path, frozen=False)
    parent, queue, output = (fixture[k] for k in ('parent', 'queue', 'output'))
    old_point = json.loads(parent.read_text())['points'][0]
    old_receipt = export_receipt(fixture['old_session'], old_point)
    old_sha = file_sha(old_receipt)
    # Even an invalid, half-written campaign is ignored before its freeze.
    contents = (output/'campaign.json').read_text()
    (output/'campaign.json').write_text('{')
    pending = comparison_watch_inputs(parent, queue)
    assert pending['campaign'] == parent and output/'session' not in pending['session_roots']
    assert not pending['terminal']
    (output/'campaign.json').write_text(contents)
    write_new(output/'freeze.json', fixture['freeze'])
    new_point = next(p for p in fixture['group']['points'] if p['dataset'] == 'longbench')
    export_receipt(output/'session', new_point)
    adopted = comparison_watch_inputs(parent, queue, session_roots=[fixture['old_session'].parent])
    assert adopted['campaign'] == output/'campaign.json'
    assert output/'session' in adopted['session_roots']
    assert set(adopted['relevant_jobs']) == {'combined32', 'original-job'}
    target = tmp_path/'compare.csv'
    summary = export(adopted['campaign'], target, session_roots=adopted['session_roots'])
    rows = list(csv.DictReader(target.open()))
    assert summary['rows'] == 180 and summary['measured'] == 2
    assert len({r['point_id'] for r in rows}) == 180
    assert next(r for r in rows if r['point_id'] == old_point['name'])['receipt_sha256'] == old_sha
    assert next(r for r in rows if r['point_id'] == new_point['name'])['status'] == 'measured'


def test_watch_waits_for_ancestor_jobs_and_checks_final_freeze_binding(tmp_path):
    from pdblend.bench.comparison_campaign import comparison_watch_inputs, binding
    fixture = watch_fixture(tmp_path)
    fixture['job']['status'] = 'passed'
    output = fixture['output']
    report = dict(parent_campaign=binding(fixture['parent']), campaign=binding(output/'campaign.json'),
                  freeze=binding(output/'freeze.json'), anchor_outcome='confirmed', complete=True)
    write_new(output/'completion.json', report)
    assert not comparison_watch_inputs(fixture['parent'], fixture['queue'])['terminal']
    fixture['queue']['jobs']['original-job']['status'] = 'passed'
    assert comparison_watch_inputs(fixture['parent'], fixture['queue'])['terminal']
    report['freeze']['sha256'] = 'wrong'
    (output/'completion.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='completion differs'):
        comparison_watch_inputs(fixture['parent'], fixture['queue'])


@pytest.mark.parametrize('mutation,match', [
    ('plan_bytes', 'binding checksum'), ('campaign_bytes', 'binding checksum'),
    ('manifest_payload', 'queue attempt'), ('inventory', 'fixed comparison inventory'),
    ('inherited_point', 'inherited comparison point'), ('freeze_boundary', 'parent boundary'),
    ('escaped_output', 'escapes queue attempt'), ('source', 'source differs'),
])
def test_watch_rejects_changed_overlay_evidence(tmp_path, mutation, match):
    from pdblend.bench.comparison_campaign import comparison_watch_inputs, binding
    fixture = watch_fixture(tmp_path); output = fixture['output']; job = fixture['job']
    if mutation == 'plan_bytes':
        (tmp_path/'plan.json').write_text('{}')
    elif mutation == 'campaign_bytes':
        (output/'campaign.json').write_text('{}')
    elif mutation == 'manifest_payload':
        manifest = json.loads((output.parent/'manifest.json').read_text())
        manifest['payload']['combined_plan']['sha256'] = 'wrong'
        (output.parent/'manifest.json').write_text(json.dumps(manifest))
    elif mutation in ('inventory', 'inherited_point'):
        campaign = json.loads((output/'campaign.json').read_text())
        if mutation == 'inventory':campaign['points'].pop()
        else:campaign['points'][0]['revision'] = 'changed'
        (output/'campaign.json').write_text(json.dumps(campaign))
        fixture['freeze']['campaign'] = binding(output/'campaign.json')
        (output/'freeze.json').write_text(json.dumps(fixture['freeze']))
    elif mutation == 'freeze_boundary':
        fixture['freeze']['evaluation_has_started'] = True
        (output/'freeze.json').write_text(json.dumps(fixture['freeze']))
    elif mutation == 'escaped_output':job['payload']['overlay_campaign_output'] = '../campaign.json'
    else:
        job['payload']['source_sha256'] = 'wrong'
        manifest = json.loads((output.parent/'manifest.json').read_text())
        manifest['payload'] = job['payload']
        (output.parent/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        comparison_watch_inputs(fixture['parent'], fixture['queue'])


def test_watch_multiple_frozen_attempts_are_ambiguous(tmp_path):
    from pdblend.bench.comparison_campaign import comparison_watch_inputs
    fixture = watch_fixture(tmp_path)
    # A second lease for the same frozen artifact is still not best-selected.
    fixture['queue']['leases']['duplicate'] = deepcopy(fixture['queue']['leases']['lease'])
    with pytest.raises(ValueError, match='ambiguous frozen'):
        comparison_watch_inputs(fixture['parent'], fixture['queue'])


def test_watch_partial_freeze_waits_while_running_but_terminal_corruption_fails(tmp_path):
    from pdblend.bench.comparison_campaign import comparison_watch_inputs
    fixture = watch_fixture(tmp_path)
    (fixture['output']/'freeze.json').write_text('{')
    assert comparison_watch_inputs(fixture['parent'], fixture['queue'])['campaign'] == fixture['parent']
    fixture['job']['status'] = 'failed'
    with pytest.raises(json.JSONDecodeError):
        comparison_watch_inputs(fixture['parent'], fixture['queue'])


def test_watch_main_refreshes_on_freeze_without_new_window_receipts(tmp_path, monkeypatch):
    import sys
    from pdblend.bench import comparison_campaign as module
    fixture = watch_fixture(tmp_path, frozen=False)
    queue_path, target = tmp_path/'queue.json', tmp_path/'compare.csv'
    write_new(queue_path, fixture['queue'])
    polls = []

    def publish_freeze(_):
        polls.append(1)
        before = list(csv.DictReader(target.open()))
        assert len(before) == 180
        assert all(row['trace_sha256'] == '' for row in before
                   if row['model_id'] == MODELS[2] and row['dataset'] == 'longbench')
        write_new(fixture['output']/'freeze.json', fixture['freeze'])
        for job in fixture['queue']['jobs'].values():job['status'] = 'passed'
        queue_path.write_text(json.dumps(fixture['queue']))

    monkeypatch.setattr(module.time, 'sleep', publish_freeze)
    monkeypatch.setattr(sys, 'argv', ['comparison_campaign', 'export', '--campaign', str(fixture['parent']),
                                    '--out', str(target), '--watch', '--queue', str(queue_path)])
    module.main()
    assert len(polls) == 1
    after = list(csv.DictReader(target.open()))
    assert len(after) == len({row['point_id'] for row in after}) == 180
    assert all(row['trace_sha256'] == 'e'*64 for row in after
               if row['model_id'] == MODELS[2] and row['dataset'] == 'longbench')


def bind_pdblend_load_counts(receipt_path, *, reset_added=2, reset_cumulative=10,
                           window_added=1, window_cumulative=11):
    reset = dict(reset_s=4., warmup_s=1., pdblend_inventory_reset=dict(engine_loads=reset_added,
        cumulative_engine_loads=reset_cumulative, before=dict(engine_load_accounting=dict(engine_loads=8)),
        allocated_to_previous_service=False))
    drain = dict(passed=True, window_engine_loads=window_added,
                 engine_load_accounting=dict(engine_loads=window_cumulative))
    receipt = json.loads(receipt_path.read_text())
    for name, data in (('reset.json', reset), ('drain.json', drain)):
        write_new(receipt_path.parent/name, data)
        receipt['artifacts'][name] = file_sha(receipt_path.parent/name)
    receipt_path.write_text(json.dumps(receipt))


def test_export_uses_final_actual_starts_and_separates_pd_reset_and_window_loads(tmp_path):
    point = export_point(); root = tmp_path/'session'
    receipt = export_receipt(root, point)
    bind_pdblend_load_counts(receipt)
    report = completed_session(root, [receipt])
    report['cleanup'] = dict(engine_loads=11)
    (root/'completion.json').write_text(json.dumps(report))
    _, rows = export_csv(tmp_path, point, [root])
    row = rows[0]
    assert row['session_engine_loads'] == '11' and row['session_initial_engine_loads'] == '8'
    assert row['session_engine_loads_source'] == 'cleanup_cumulative'
    assert row['pdblend_reset_engine_loads'] == '2' and row['pdblend_reset_cumulative_engine_loads'] == '10'
    assert row['pdblend_window_engine_loads'] == '1' and row['pdblend_window_cumulative_engine_loads'] == '11'


@pytest.mark.parametrize('cleanup_count', [7, 9, None, -1, True])
def test_export_never_hides_regressed_or_invalid_cleanup_counts_with_startup(tmp_path, cleanup_count):
    point = export_point(); root = tmp_path/'session'
    receipt = export_receipt(root, point)
    bind_pdblend_load_counts(receipt)
    report = completed_session(root, [receipt]); report['cleanup'] = dict(engine_loads=cleanup_count)
    (root/'completion.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        export_csv(tmp_path, point, [root])


@pytest.mark.parametrize('system', ['mixed', 'ecoserve'])
def test_existing_baseline_initial_and_cleanup_load_counts_remain_unchanged(tmp_path, system):
    point = dict(export_point(), system=system); root = tmp_path/'session'
    receipt = export_receipt(root, point)
    report = completed_session(root, [receipt]); report['cleanup'] = dict(engine_loads=8)
    (root/'completion.json').write_text(json.dumps(report))
    _, rows = export_csv(tmp_path, point, [root])
    assert rows[0]['session_engine_loads'] == rows[0]['session_initial_engine_loads'] == '8'
    assert rows[0]['pdblend_reset_engine_loads'] == rows[0]['pdblend_window_engine_loads'] == ''


@pytest.mark.parametrize('counts', [dict(reset_cumulative=9), dict(window_cumulative=12)])
def test_export_rejects_inconsistent_bound_pd_lifecycle_count_deltas(tmp_path, counts):
    point = export_point(); root = tmp_path/'session'
    receipt = export_receipt(root, point)
    bind_pdblend_load_counts(receipt, **counts)
    with pytest.raises(ValueError, match='engine-load accounting differs'):
        export_csv(tmp_path, point, [root])


def test_pd_window_counts_available_while_running_but_missing_final_total_stays_unknown(tmp_path):
    point = export_point(); root = tmp_path/'session'
    receipt = export_receipt(root, point)
    bind_pdblend_load_counts(receipt)
    _, rows = export_csv(tmp_path, point, [root])
    assert rows[0]['pdblend_window_engine_loads'] == '1' and rows[0]['session_engine_loads'] == ''
    completed_session(root, [receipt])
    _, rows = export_csv(tmp_path, point, [root])
    assert rows[0]['session_initial_engine_loads'] == '8'
    assert rows[0]['session_engine_loads'] == ''
    assert rows[0]['session_engine_loads_source'] == 'missing_cleanup_cumulative'


def test_unbound_pd_count_files_are_not_exported(tmp_path):
    point = export_point(); root = tmp_path/'session'
    receipt = export_receipt(root, point)
    (receipt.parent/'drain.json').write_text('{"window_engine_loads":99}')
    (receipt.parent/'reset.json').write_text('{"pdblend_inventory_reset":{"engine_loads":99}}')
    _, rows = export_csv(tmp_path, point, [root])
    assert rows[0]['pdblend_window_engine_loads'] == rows[0]['pdblend_reset_engine_loads'] == ''
