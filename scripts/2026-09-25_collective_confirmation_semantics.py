#!/usr/bin/env python3
"""CPU semantic prototype only. This is not wired into a serving controller."""
from dataclasses import dataclass, replace
import json
import sys

sys.path.insert(0, '/home/pdblend4/src')
from pdblend.online.shield import Pressure, Shield
from pdblend.planner.pool import SLO


@dataclass
class ConfirmationGate:
    """A proposed caller-side gate, separate from the existing no-op fix.

    A real integration must supply a fresh independent observation id, stable
    ownership epoch, and verified physical headroom. Unknown evidence falls
    back to the current immediate probe. No duration threshold is fitted here.
    """
    pending: tuple | None = None

    def decide(self, shield, pressure, *, observation, epoch, now,
               headroom='unknown', stable=True, deadline=False):
        if deadline:
            self.pending = None
            return 'deadline_immediate'
        eligible = (shield.mode == 'budget_aware' and shield._collective_only(pressure)
            and shield.level == 0 and shield.floor_active == 0
            and shield.capacity_target_active == 0 and not shield._pending_capacity)
        if not eligible or not stable or headroom == 'unknown':
            self.pending = None
            return 'existing_update'
        if headroom == 'at_ceiling':
            self.pending = None
            return 'verified_noop'
        assert headroom == 'available'
        key = (epoch, tuple(sorted(shield._pressure_roles(pressure))))
        if self.pending is not None:
            pending_key, previous_observation = self.pending
            if pending_key == key and observation != previous_observation:
                self.pending = None
                return 'confirmed_update'
            if pending_key == key:
                return 'pending_same_observation'
        self.pending = (key, observation)
        return 'pending_first_observation'


def exercise():
    cases = []
    def collective():
        return Pressure(decode=True, decode_paths=('M',), decode_stalled=3,
            decode_active=8, decode_stalled_fraction=3/8, longest_token_gap_s=.3,
            tpot_p90=.035, mode='budget_aware')
    def setup(): return Shield(SLO(5.,.15),mode='budget_aware'), ConfirmationGate()
    def decide(g,s,p,obs,**kw):
        return g.decide(s,p,observation=obs,epoch=1,now=100.+obs,headroom='available',**kw)
    s,g=setup(); p=collective()
    assert decide(g,s,p,1)=='pending_first_observation'
    assert s.level == s.escalation_sequence == 0
    assert decide(g,s,p,1)=='pending_same_observation'
    assert decide(g,s,p,2)=='confirmed_update'
    assert s.update(p,102.)==1
    cases.append('two distinct observations confirm; duplicate observation never confirms')
    s,g=setup(); decide(g,s,p,1)
    assert decide(g,s,Pressure(),2)=='existing_update'
    s.update(Pressure(),102.)
    assert g.pending is None and s.level == 0
    assert decide(g,s,p,3)=='pending_first_observation'
    cases.append('transient gap clears; later isolated episode starts fresh')
    for change in [dict(prefill=True),dict(decode_sustained_stalls=1),
                   dict(decode_budget_risks=1),dict(decode_short_output_risks=1),dict(tpot_p90=.13)]:
        s,g=setup(); decide(g,s,p,1); strong=replace(p,**change)
        assert decide(g,s,strong,2)=='existing_update'
        assert s.update(strong,102.)==1 and g.pending is None
        cases.append('no added confirmation delay for '+next(iter(change)))
    s,g=setup(); decide(g,s,p,1)
    assert decide(g,s,p,2,deadline=True)=='deadline_immediate' and g.pending is None
    cases.append('deadline remains independently immediate and cancels confirmation')
    for headroom,expected in [('unknown','existing_update'),('at_ceiling','verified_noop')]:
        s,g=setup()
        assert g.decide(s,p,observation=1,epoch=1,now=101.,headroom=headroom)==expected
        if headroom=='at_ceiling':
            assert s.update(p,101.,clock_probe_available=False)==0
            assert s.escalation_sequence==0 and not s.collective_clock_probe_only
        else:
            assert s.update(p,101.)==1
        cases.append(headroom+' evidence retains existing safety/no-op behavior')
    s,g=setup(); decide(g,s,p,1)
    assert g.decide(s,p,observation=2,epoch=2,now=102.,headroom='available')=='pending_first_observation'
    assert decide(g,s,p,3,stable=False)=='existing_update'
    cases.append('ownership epoch change cannot confirm; active transition keeps existing response')
    for field,value in [('level',1),('floor_active',4),('capacity_target_active',4)]:
        s,g=setup(); setattr(s,field,value)
        assert decide(g,s,p,1)=='existing_update'
        cases.append('existing '+field+' is preserved rather than hidden by confirmation')
    return dict(schema='collective-confirmation-semantic-prototype/v1',passed=True,
        cases=cases,case_count=len(cases),production_implementation=False,
        gpu_executed=False,current_nine_point_frequency_retest_affected=False,
        existing_noop_code_reused=True,thresholds_fitted_to_historical_data=False,
        limitations=['CPU state transitions do not demonstrate physical latency or energy improvement.',
            'The current physical helper proves at-ceiling or unknown; it does not provide verified headroom for this proposed gate.',
            'A production integration needs fresh observation/ownership identities and failure behavior; this prototype is not deployable as-is.',
            'Confirmation can add one observation interval for collective-only risk; strong-pressure throttle and deadline worker remain the existing mechanisms.'])


if __name__=='__main__': print(json.dumps(exercise(),indent=2))
