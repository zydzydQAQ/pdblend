"""Scientific validity gates reject timing corruption and unknown failures."""
import copy
from types import SimpleNamespace
import pytest
import audit_eco_observations_v1 as audit


def need(ok, why):
    if not ok:
        raise ValueError(why)


P = SimpleNamespace(need=need)


def example(n=20):
    bench, events = [], []
    for i in range(n):
        planned = 1000. + i
        bench.append(dict(request_id=str(i), planned_arrival_s=str(planned),
            actual_dispatch_s=str(planned+.001), request_deadline_s=str(planned+120),
            open_loop_independent='True'))
        events.append(dict(kind='request_timing', client_request_id=str(i),
            planned_arrival_s=planned, actual_dispatch_s=planned+.001,
            hard_deadline_s=planned+120, handler_arrival_s=planned+.002))
    return bench, events


def test_exact_original_arrivals():
    bench, events = example()
    result = audit.timing(P, bench, events)
    assert result['dispatch_lateness_p99_s'] == pytest.approx(.001)
    assert result['request_budget_s'] == 120


@pytest.mark.parametrize('change', [
    'missing_controller', 'duplicate_controller', 'duplicate_client',
    'wrong_client', 'nan_dispatch', 'extended_deadline', 'closed_loop',
    'controller_changed_dispatch', 'starved_handler', 'negative_handler',
    'max_dispatch', 'p99_dispatch',
])
def test_invalid_arrivals(change):
    b, e = example()
    if change == 'missing_controller': e.pop()
    elif change == 'duplicate_controller': e.append(copy.deepcopy(e[0]))
    elif change == 'duplicate_client': b.append(copy.deepcopy(b[0]))
    elif change == 'wrong_client': e[0]['client_request_id'] = 'unknown'
    elif change == 'nan_dispatch': b[0]['actual_dispatch_s'] = 'nan'
    elif change == 'extended_deadline': b[0]['request_deadline_s'] = '1240'
    elif change == 'closed_loop': b[0]['open_loop_independent'] = 'False'
    elif change == 'controller_changed_dispatch': e[0]['actual_dispatch_s'] += .001
    elif change == 'starved_handler': e[0]['handler_arrival_s'] += 2
    elif change == 'negative_handler': e[0]['handler_arrival_s'] -= 1
    else:
        delay = 1.1 if change == 'max_dispatch' else .2
        for i in range(len(b)):
            at = float(b[i]['planned_arrival_s']) + delay
            b[i]['actual_dispatch_s'] = str(at)
            e[i]['actual_dispatch_s'] = at
            e[i]['handler_arrival_s'] = at + .001
    with pytest.raises(ValueError):
        audit.timing(P, b, e)


def test_b_legacy_mismatch_cannot_be_erased():
    import final_selected_collect_v4 as collect
    p = collect.load_audit().p
    binding = p.read(audit.B_QUALIFIED_BINDING)
    binding['legacy_single_vs_pair_exact'] = True
    with pytest.raises(ValueError, match='mismatch must remain'):
        audit.b_qualification(p, binding)


def test_b_qualification_cannot_change_owner():
    import final_selected_collect_v4 as collect
    p = collect.load_audit().p
    binding = p.read(audit.B_QUALIFIED_BINDING)
    binding['instances'][0]['container']['StartedAt'] = 'other-process'
    with pytest.raises(ValueError, match='changed its fresh qualification'):
        audit.b_qualification(p, binding)


def test_c_continuation_cannot_change_configuration():
    import final_selected_collect_v4 as collect
    p = collect.load_audit().p
    binding = p.read(audit.C_QUALIFIED_BINDING)
    binding['host_release'] = '/another-release'
    with pytest.raises(ValueError, match='changed its measured native qualification'):
        audit.c_qualification(p, binding)
