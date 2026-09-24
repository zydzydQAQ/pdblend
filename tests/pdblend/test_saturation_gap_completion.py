"""Synthetic, CPU-only fixtures for the twenty-gap final publication contract."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def complete_round(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'scripts/2026-09-24_audit_saturation_completion.py'
    spec = importlib.util.spec_from_file_location('gap_completion_auditor', path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    real_read = auditor.read
    # Queue state is irrelevant to completed fixtures and must never come from
    # a live GPU environment. The boundary proof reader is covered elsewhere;
    # this fixture supplies its already-verified manifest interface.
    monkeypatch.setattr(auditor, 'read', lambda p: {} if Path(p).name == 'queue.json' else real_read(p))
    monkeypatch.setattr(auditor, 'read_extension_manifest',
                        lambda ref: {'manifest': auditor.load_bound(ref)})

    def make(alias=False):
        package = tmp_path / ('alias-round' if alias else 'same-name-round')
        package.mkdir()
        run_id, revision = 'synthetic-gap-contract', 'synthetic-pd-revision'
        rows, gap_rows, gaps, pd_points, baseline_paths = [], [], [], [], {}

        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True))
            return auditor.binding(path)

        def point(name, model, dataset, system, scale):
            return dict(name=name, model_id=model, dataset=dataset, system=system,
                scale=scale, seed=701, duration_s=150., slo={'ttft_s': 1., 'tpot_s': .1},
                trace={'path': 'synthetic-trace', 'sha256': 't'*64, 'requests': 1},
                engine_identity={'runtime_source_sha256': 'synthetic-runtime',
                                 'measurement_source_sha256': 'synthetic-meter'},
                inputs={}, revision=revision, run_id=run_id)

        def observation(point, tag, *, energy=True, publish=True):
            directory = package / 'evidence' / tag
            pref = write(directory/'point.json', point)
            receipt = dict(point=point['name'], point_sha256=auditor.digest(point),
                recorded_window_complete=True, cleanup_passed=True, measurement_evidence_valid=energy,
                artifacts={'point.json': pref['sha256']},
                result={'metrics': {'energy_service_j': 100. if energy else None,
                                    'energy_tail_j': 10. if energy else None}})
            rref = write(directory/'receipt.json', receipt)
            if publish:
                rows.append(dict(point_id=point['name'], point_sha256=auditor.digest(point),
                    receipt_path=rref['path'], receipt_sha256=rref['sha256'],
                    analysis_slo_pass='False', analysis_energy_usable=str(energy),
                    energy_service_j='100' if energy else '', energy_tail_j='10' if energy else ''))
            return pref, rref

        policy = write(package/'policy.json', {'fixture': 'already verified policy'})
        active_policies = [policy]
        for size in ('7B', '14B', '32B'):
            model = 'Qwen2.5-'+size+'-Instruct'
            selected, boundaries, baselines = {}, {}, []
            for dataset in auditor.DATASETS:
                refs = {}
                for scale in (.25, .5, .75, 1.):
                    p = point(f'{size}-pdblend-{dataset}-x{scale}', model, dataset, 'pdblend', scale)
                    p['comparison_scope'] = 'original_matrix_full_group'
                    pd_points.append(p)
                    pref, _ = observation(p, p['name'])
                    if scale in (.5, 1.):
                        side = 'lower' if scale == .5 else 'upper'
                        refs[side] = pref
                        for system in auditor.BASELINES:
                            b = dict(p, name=p['name'].replace('-pdblend-', '-'+system+'-'), system=system)
                            baselines.append(b)
                            observation(b, b['name'])
                selected[dataset] = refs
                boundaries[dataset] = dict(status='bracketed', nonmonotonic=False,
                    passed_lower_point=refs['lower'], failed_upper_point=refs['upper'])
            manifest = write(package/(size+'-manifest.json'), dict(run_id=run_id, model_id=model,
                revision=revision, policy=policy, boundaries=boundaries))
            write(package/'orchestration'/('boundary-selection-'+size.lower()+'.json'),
                  dict(run_id=run_id, model_id=model, manifest=manifest, datasets=selected))
            baseline_path = package/(size+'-baseline-campaign.json')
            write(baseline_path, dict(run_id=run_id, groups=[{'points': baselines}]))
            baseline_paths[model] = str(baseline_path)

        for index in range(20):
            p = point(f'authorized-gap-{index}', 'Qwen2.5-7B-Instruct',
                      auditor.DATASETS[index % 3], auditor.BASELINES[index % 4], .25*(1+index//4))
            oldpoint, oldreceipt = observation(p, f'gap-{index}-original', energy=False, publish=False)
            gaps.append(dict(point_id=p['name'], point=oldpoint, receipt=oldreceipt,
                             point_sha256=auditor.digest(p)))
            actual = deepcopy(p)
            if alias:
                actual.update(name=p['name']+'-energy-supplement', original_point_name=p['name'],
                              energy_supplement_of=oldreceipt)
            _, receipt = observation(actual, f'gap-{index}-completion')
            gap_rows.append(dict(point_id=p['name'], original_receipt=oldreceipt,
                                 status='completed', completion_receipt=receipt))
        inventory = write(package/'authorized-gaps.json', dict(
            schema='pdblend-baseline-energy-gap-inventory/v1', frozen_baselines=[], gaps=gaps))
        cref = write(package/'campaign.json', dict(run_id=run_id, points=pd_points,
            active_extension_policy_refs=active_policies, authorized_baseline_energy_gaps=inventory,
            baseline_energy_gaps=inventory))
        write(package/'orchestration/status.json', dict(baseline_campaigns=baseline_paths))
        report = dict(schema='authorized-baseline-completion/v1', campaign=cref,
                      authorized_count=20, completed_count=20, all_complete=True, rows=gap_rows)
        csv_path = package/'compare.csv'

        def evaluate():
            with csv_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            gref = write(package/'gap-review.json', report)
            return auditor.audit(package, csv_path, gap_review=gref)

        return dict(evaluate=evaluate, report=report, rows=rows, gap_rows=gap_rows, auditor=auditor)
    return make


@pytest.mark.parametrize('alias', [False, True])
def test_all_twenty_completed_gaps_require_exact_published_receipts(complete_round, alias):
    fixture = complete_round(alias)
    report = fixture['evaluate']()
    assert report['all_scope_accounted'] and not report['scope_pending']
    assert not report['has_obstructions']
    assert report['counts']['baseline_service_gaps']['observed'] == 20
    assert all(row['status'] == 'observed' and row['complete_energy'] for row in report['baseline_gap_outcomes'])
    # Every synthetic request observation is an SLO failure. That must not
    # disqualify a recorded complete baseline energy observation.
    assert report['counts']['pd_originals']['observed'] == 36


def test_completed_gap_receipt_missing_from_csv_remains_pending(complete_round):
    fixture = complete_round(True)
    receipt = fixture['gap_rows'][0]['completion_receipt']
    fixture['rows'][:] = [r for r in fixture['rows'] if r['receipt_sha256'] != receipt['sha256']]
    report = fixture['evaluate']()
    gap = next(r for r in report['baseline_gap_outcomes']
               if r['point_id'] == fixture['gap_rows'][0]['point_id'])
    assert gap['status'] == 'pending'
    assert gap['reason'] == 'complete baseline receipt absent from CSV'
    assert report['counts']['baseline_service_gaps']['pending'] == 1
    assert report['scope_pending'] and not report['all_scope_accounted']


def test_completed_gap_csv_wrong_point_sha_is_rejected(complete_round):
    fixture = complete_round(True)
    receipt = fixture['gap_rows'][0]['completion_receipt']
    row = next(r for r in fixture['rows'] if r['receipt_sha256'] == receipt['sha256'])
    row['point_sha256'] = '0'*64
    with pytest.raises(ValueError):
        fixture['evaluate']()


def test_one_pending_gap_keeps_overall_scope_pending(complete_round):
    fixture = complete_round()
    entry = fixture['gap_rows'][0]
    receipt = entry.pop('completion_receipt')
    entry['status'] = 'pending'
    fixture['rows'][:] = [r for r in fixture['rows'] if r['receipt_sha256'] != receipt['sha256']]
    fixture['report'].update(completed_count=19, all_complete=False)
    report = fixture['evaluate']()
    assert report['counts']['pd_originals']['observed'] == 36
    assert report['counts']['boundaries']['bracketed'] == 9
    assert report['counts']['baseline_endpoints']['observed'] == 72
    assert report['counts']['baseline_service_gaps']['pending'] == 1
    assert report['scope_pending'] and not report['all_scope_accounted']
    assert not report['has_obstructions']
