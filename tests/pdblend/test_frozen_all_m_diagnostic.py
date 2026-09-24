from collections import Counter

import pytest

from pdblend.bench import comparison_pdblend_acceptance as audit
from test_comparison_pdblend_acceptance import fixture, raw


def test_only_explicit_bound_all_m_diagnostic_can_omit_periodic_planning(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    selected = audit._inputs()
    events = [r for r in raw(args, 'controller') if r['kind'] != 'forecast']
    plan = next(r for r in events if r['kind'] == 'plan')
    plan.update(tau=0, f_M=max(selected['frequencies']))
    selected['choice']['plan'].update(tau=plan['tau'], f_M=plan['f_M'])
    native = args['native_result']
    native['controller']['events'] = dict(Counter(r['kind'] for r in events))
    identity = args['engine_identity']
    instances = {r['instance_id']: r for r in identity['instances']}
    def check():
        return audit._controller(events, native, instances, identity, args['reset'],
                                 selected, raw(args, 'transition_measurements'))
    with pytest.raises(ValueError, match='periodic'):
        check()
    selected['experiment_mode'] = 'freeze_initial_all_m'
    assert len(check()[0]) == 1
    plan['f_M'] = selected['choice']['plan']['f_M'] = min(selected['frequencies'])
    with pytest.raises(ValueError, match='fixed all-M'):
        check()
