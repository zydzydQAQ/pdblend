"""Missing reset evidence must not masquerade as a hardware clock failure."""
from collections import Counter
import pytest

from pdblend.bench import comparison_pdblend_acceptance as audit
from test_comparison_pdblend_acceptance import fixture, raw, write
from test_comparison_acceptance import save_change


def remove_reset(args):
    events = [r for r in raw(args, 'controller') if r.get('operation') != 'clock_reset']
    phases = [{k: v for k, v in r.items() if k not in ('kind', 't')}
              for r in events if r.get('kind') == 'transition_phase']
    write(args, 'controller', events)
    args['native_result']['controller'].update(
        events=dict(Counter(r['kind'] for r in events)), transition_phases=phases)
    save_change(args, 'native_result', args['native_result'])
    transitions = raw(args, 'transition_measurements')
    transitions['phases'] = [r for r in transitions['phases'] if r['operation'] != 'clock_reset']
    write(args, 'transition_measurements', transitions)


def test_missing_reset_is_not_evidence_of_early_parking_or_bad_clocks(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch, parked=True)
    remove_reset(args)
    result = audit.audit_pdblend_window(**args)
    assert not result['formal_eligible']
    assert result['gate_failures']['pdblend.controller_actions'] == (
        'parking is missing required operations: clock_reset')
    assert result['blocked_gates']['pdblend.physical_clocks'] == ['pdblend.controller_actions']
    assert 'pdblend.physical_clocks' in result['missing_gates']
    assert 'pdblend.physical_clocks' not in result['gate_failures']
    assert 'pdblend.frequency_samples' in result['checked_gates']


def test_raw_frequency_defect_remains_visible_when_controller_evidence_fails(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch, parked=True)
    remove_reset(args)
    write(args, 'frequencies', [[100., [1500] * 7]])
    result = audit.audit_pdblend_window(**args)
    assert 'pdblend.frequency_samples' in result['gate_failures']
    assert result['blocked_gates']['pdblend.physical_clocks'] == ['pdblend.controller_actions']


def test_real_clock_mismatch_still_fails_directly(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    samples = raw(args, 'frequencies')
    samples[20][1][0] = 900
    write(args, 'frequencies', samples)
    result = audit.audit_pdblend_window(**args)
    assert result['blocked_gates'] == {}
    assert result['gate_failures']['pdblend.physical_clocks'] == (
        'actual active/parked clock differs from executed plan')


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -1, True])
def test_invalid_raw_clock_value_is_visible_without_control_evidence(tmp_path, monkeypatch, value):
    args = fixture(tmp_path, monkeypatch, parked=True)
    remove_reset(args)
    samples = raw(args, 'frequencies')
    samples[20][1][0] = value
    write(args, 'frequencies', samples)
    result = audit.audit_pdblend_window(**args)
    assert 'pdblend.frequency_samples' in result['gate_failures']
    assert result['blocked_gates']['pdblend.physical_clocks'] == ['pdblend.controller_actions']
    assert not result['formal_eligible']


def test_zero_clock_is_structurally_valid_for_stopped_gpu():
    audit._frequency_samples([[100., [0] * 8], [100.1, [0] * 8]])
