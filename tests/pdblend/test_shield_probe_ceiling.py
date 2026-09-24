"""Physical-clock evidence may suppress a no-op short-gap probe, never real risk."""
import asyncio
import time
from dataclasses import replace

import pytest

from pdblend.online.shield import Pressure, Shield
from pdblend.planner.pool import SLO
from test_controller import make_controller
from test_controller_capacity_reserve import collective_controller, run_ticks


def collective(paths=('M',)):
    return Pressure(decode=True, decode_paths=paths, decode_stalled=2,
                    decode_stalled_fraction=.5, mode='budget_aware')


def test_repeated_verified_ceiling_edges_never_open_frequency_episode_or_capacity():
    shield = Shield(SLO(5., .15), mode='budget_aware')
    for now in (100., 105., 110.):
        assert shield.needs_collective_clock_probe(collective(), now)
        assert shield.update(collective(), now, clock_probe_available=False) == 0
        assert not shield.needs_collective_clock_probe(collective(), now+1)
        shield.update(Pressure(), now+1)
    assert len(shield.events) == 3
    assert all(row['clock_probe_noop'] for row in shield.events)
    assert shield.escalation_sequence == shield.floor_active == shield.capacity_target_active == 0
    # The same event must still raise clocks once the deployment has headroom.
    assert shield.update(collective(), 115., clock_probe_available=True) == 1
    assert shield.events[-1]['collective_clock_probe'] is True


@pytest.mark.parametrize('change', [dict(prefill=True, prefill_paths=('M',)),
    dict(decode_sustained_stalls=1), dict(decode_budget_risks=1),
    dict(decode_short_output_risks=1), dict(tpot_p90=.13)])
def test_real_pressure_keeps_clock_and_capacity_recovery_at_measured_ceiling(change):
    shield = Shield(SLO(5., .15), mode='budget_aware')
    pressure = replace(collective(), **change)
    assert not shield.needs_collective_clock_probe(pressure, 100.)
    assert shield.update(pressure, 100., clock_probe_available=False) == 1
    assert shield.update(pressure, 102., clock_probe_available=False) == 2
    assert shield.events[-1]['capacity_escalation'] is True


def ready_controller(paths=('M',)):
    ctl, _, _ = make_controller(n=4)
    ctl.shield = Shield(SLO(5., .15), mode='budget_aware')
    ctl.max_freq = 2100
    ctl.roles = {'i0':'M', 'i1':'M', 'i2':'D', 'i3':'L1'}
    ctl.freqs = {iid:2100 if role != 'L1' else 210 for iid,role in ctl.roles.items()}
    ctl._clock_known = set(ctl.roles)
    for iid, role in ctl.roles.items():
        ctl.router.set_roles({iid:'parked' if role == 'L1' else role})
        ctl.router.set_accepting(iid, role != 'L1')
    def capture(**kwargs):
        now = time.time()
        return [dict(gpu=g, requested_mhz=2100, observed_mhz=2100,
                     read_started_s=now, read_finished_s=now, error=None) for g in (0,1,2)]
    ctl.frequency_snapshot = capture
    return ctl


@pytest.mark.parametrize('paths', [('M',), ('PD',), ('M','PD')])
def test_physical_ceiling_covers_exact_pressure_roles_and_all_their_gpus(paths):
    ctl = ready_controller()
    assert asyncio.run(ctl._collective_probe_available(collective(paths))) is False
    assert ctl._log[-1]['reason'] == 'collective_probe_headroom'


@pytest.mark.parametrize('fault', ['underclock','overclock','nan','missing','duplicate','stale',
                                  'reversed','wrong_target','error','in_transition',
                                  'unknown_clock','low_target','not_accepting','owner_changed'])
def test_missing_stale_throttled_or_changed_ownership_evidence_keeps_probe(fault):
    ctl = ready_controller()
    actual = ctl.frequency_snapshot
    def capture(**kwargs):
        rows = actual(**kwargs)
        if fault == 'underclock': rows[0]['observed_mhz'] = 1900
        if fault == 'overclock': rows[0]['observed_mhz'] = 2520
        if fault == 'nan': rows[0]['observed_mhz'] = float('nan')
        if fault == 'missing': rows = rows[1:]
        if fault == 'duplicate': rows.append(dict(rows[0]))
        if fault == 'stale':
            rows[0]['read_started_s'] -= 2
            rows[0]['read_finished_s'] -= 2
        if fault == 'reversed': rows[0]['read_finished_s'] -= 1
        if fault == 'wrong_target': rows[0]['requested_mhz'] = 1800
        if fault == 'error': rows[0]['error'] = 'unavailable'
        if fault == 'owner_changed': ctl.roles['i0'] = 'L1'
        return rows
    ctl.frequency_snapshot = capture
    if fault == 'in_transition': ctl._transition_open = True
    if fault == 'unknown_clock': ctl._clock_known.discard('i0')
    if fault == 'low_target': ctl.freqs['i0'] = 1800
    if fault == 'not_accepting': ctl.router.set_accepting('i0',False)
    assert asyncio.run(ctl._collective_probe_available(collective())) is None


def test_requested_clock_without_physical_reader_cannot_suppress_probe():
    ctl = ready_controller()
    ctl.frequency_snapshot = None
    assert asyncio.run(ctl._collective_probe_available(collective())) is None


def test_measured_ceiling_noop_does_not_starve_two_independent_capacity_votes(monkeypatch):
    ctl = collective_controller('tuning')
    target = ctl.initial_plan
    ctl.planner.plan = lambda demand, current: target
    ctl.min_plan_hold_s = 0
    ctl.initial_plan = replace(ctl.initial_plan, counts={'M':4,'L1':4})
    ctl.down_plan_votes = 2
    # A physical-reader substitute isolates gating in the real decision loop;
    # the real per-GPU evidence guard is checked by the cases above.
    async def at_ceiling(pressure):
        return False
    ctl._collective_probe_available = at_ceiling
    ctl.shield.observe = lambda rows, now: collective() if int(now) % 2 else Pressure()
    executed = run_ticks(ctl, monkeypatch, seconds=16)
    assert [plan.counts['M'] for plan in executed] == [4,2]
    assert any(row.get('clock_probe_noop') for row in ctl.shield.events)
    forecasts = [r for r in ctl._log if r['kind']=='forecast']
    assert forecasts[0]['down_votes'] == 1
    assert forecasts[1]['decision_reason'] == 'confirmed_downshift'
