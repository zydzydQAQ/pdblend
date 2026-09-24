"""Synthetic repeated boundaries; no execution or formal capacity assertions."""
import json
import subprocess
import sys

import pytest

from pdblend.bench.capacity_comparison import BASELINES, compare_capacity_ledgers
from test_slo_capacity_receipts import ledger, receipt, workload_family


def campaign(root, *, pd=(2., 2.08), baseline=(1., 1.04), systems=None, overrides=None, raw_aliases=None):
    family = workload_family(root)
    ledgers = {}
    for system in (systems or (*BASELINES, 'pdblend')):
        directory = root/system
        lower, upper = pd if system == 'pdblend' else baseline
        rows = [receipt(directory, rate=rate, repeat=repeat, passing=rate == lower,
                        system=system, family_ref=family,
                        raw_id=((raw_aliases or {})[system]+f'/x{rate}-r{repeat}'
                                if system in (raw_aliases or {}) else None),
                        **(overrides or {}).get(system, {}))
                for rate in (lower, upper) if rate is not None for repeat in range(3)]
        ledgers[system] = ledger(directory, rows, system=system)
    return ledgers


def test_only_strictly_separated_five_system_brackets_report_an_observed_lead(tmp_path):
    ledgers = campaign(tmp_path)
    report = compare_capacity_ledgers(ledgers)
    assert report['observed_boundary_lead']
    assert report['status'] == 'observed_pdblend_boundary_lead'
    assert report['reason_codes'] == []
    assert set(report['pairwise_bounds']) == set(BASELINES)
    assert all(pair['pdblend_passed_lower'] > pair['baseline_failed_upper']
               for pair in report['pairwise_bounds'].values())
    assert report['shared_identity']['gpu_uuids'] == ['GPU-'+str(i) for i in range(8)]
    assert not report['formal_eligible'] and not report['profile_qualification_promoted']
    assert not report['exact_capacity_established']
    assert not report['hardware_executed'] and not report['jobs_enqueued']


@pytest.mark.parametrize('pd', [(1., 1.04), (1.04, 1.08), (2., None)])
def test_overlapping_equal_or_unbounded_intervals_never_announce_a_lead(tmp_path, pd):
    report = compare_capacity_ledgers(campaign(tmp_path, pd=pd))
    assert not report['observed_boundary_lead']
    if pd[1] is None:
        assert 'missing_failed_upper:pdblend' in report['reason_codes']
    else:
        assert all('not_strictly_separated:'+system in report['reason_codes'] for system in BASELINES)


def test_missing_baseline_or_unconverged_baseline_never_announce_a_lead(tmp_path):
    ledgers = campaign(tmp_path, baseline=(1., 2.))
    report = compare_capacity_ledgers(ledgers)
    assert not report['observed_boundary_lead']
    assert any(reason == 'unconverged:mixed:bracketing' for reason in report['reason_codes'])
    ledgers.pop('mixed')
    assert 'missing_system:mixed' in compare_capacity_ledgers(ledgers)['reason_codes']


@pytest.mark.parametrize('variant', ['hardware', 'family', 'policy', 'assignment', 'duplicate_raw'])
def test_cross_system_identity_or_protocol_substitution_is_rejected(tmp_path, variant):
    overrides = {}
    if variant == 'hardware': overrides['mixed'] = dict(gpu_prefix='other-GPU-')
    ledgers = campaign(tmp_path, overrides=overrides,
                       raw_aliases={'distserve':'mixed'} if variant == 'duplicate_raw' else None)
    if variant == 'family':
        foreign = campaign(tmp_path/'other', systems=['mixed'])
        ledgers['mixed'] = foreign['mixed']
    elif variant == 'policy':
        value = json.loads(ledgers['mixed'].read_text())
        value['config']['required_repeats'] = 4
        ledgers['mixed'].write_text(json.dumps(value))
    elif variant == 'assignment':
        ledgers['mixed'] = ledgers['distserve']
    with pytest.raises(ValueError, match='family|hardware|protocol|assignment|same raw measurement'):
        compare_capacity_ledgers(ledgers)


def test_comparison_cli_reads_all_bound_ledgers_and_writes_nothing(tmp_path):
    ledgers = campaign(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    args = [sys.executable, '-m', 'pdblend.bench.capacity_comparison']
    for system, path in ledgers.items():
        args += ['--ledger', system+'='+str(path)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['observed_boundary_lead']
    assert {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()} == before
