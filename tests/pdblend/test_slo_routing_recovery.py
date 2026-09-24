"""New-request recovery must not rewrite existing PD/KV ownership."""
import pytest

from pdblend.online.router import Router, DuplicateRequestError
from pdblend.planner.pool import SLO


class TimingModel:
    freqs = (2520,)
    kv_capacity_tokens = 100000
    profile_key = {'test': 'explicit-timing'}

    def prefill_seconds(self, n, f):
        return n * .001

    def step_seconds(self, batch, context, f):
        return .02

    def decode_supported(self, batch, context, f):
        return True

    def transfer_seconds(self, n):
        return .30


def router():
    result = Router(['p', 'd', 'm'], pd_threshold_tokens=50)
    result.set_roles({'p': 'P', 'd': 'D', 'm': 'M'})
    return result


def configure(result, model=None, **kwargs):
    result.configure_slo_routing(model=model or TimingModel(), slo=SLO(.5, .1),
        frequency_provider=lambda iid: 2520, max_model_len=10000, safety=1., **kwargs)


def test_congested_prefill_spills_new_request_without_rewriting_old_ownership():
    r = router()
    old = r.dispatch('old', 1000, 100)
    old.engine_instances.update({'p', 'd'})
    before = (old.path, old.prefill_instance, old.decode_instance, old.generation,
              set(old.engine_instances), r.loads['p'].inflight_prefill_tokens)
    configure(r)
    new = r.dispatch('new', 100, 100)
    assert (new.path, new.decode_instance) == ('M', 'm')
    assert new.route_reason == 'pd_capacity_or_ttft_spillover'
    assert before == (old.path, old.prefill_instance, old.decode_instance, old.generation,
                      set(old.engine_instances), r.loads['p'].inflight_prefill_tokens)
    assert r._active_ids['old'] is old and old in r.active['d']
    with pytest.raises(DuplicateRequestError):
        r.dispatch('old', 100, 100)


def test_two_token_handoff_risk_changes_route_but_long_budget_can_amortise():
    class MeasuredGap(TimingModel):
        def pd_first_gap_seconds(self, **query):
            assert query['output_tokens'] == 2
            return .32
    r = router()
    configure(r, MeasuredGap())
    short = r.dispatch('short', 100, 2)
    assert short.path == 'M' and short.route_reason == 'pd_handoff_tpot_risk'
    pd = next(p for p in short.route_estimate['slo_routing']['predictions'] if p['path'] == 'PD')
    assert pd['ttft_s'] == pytest.approx(.1)
    assert pd['tpot_s'] == pytest.approx(.32)
    r.finish(short, 2)
    long = r.dispatch('long', 100, 100)
    assert long.path == 'PD'
    assert long.route_estimate['slo_routing']['selected']['tpot_s'] < .1


@pytest.mark.parametrize('constraint', ['sequence', 'kv', 'not_accepting', 'incumbent', 'waiting_ttft'])
def test_spillover_does_not_use_unavailable_mixed_capacity(constraint):
    r = router()
    existing = r.dispatch('mixed-owned', 10, 100)
    model = TimingModel()
    configure(r, model, max_num_seqs=1 if constraint == 'sequence' else 32)
    if constraint == 'kv':
        model.kv_capacity_tokens = 200
    elif constraint == 'not_accepting':
        r.set_accepting('m', False)
    elif constraint == 'incumbent':
        import time
        existing.max_tokens = 3
        r.token(existing, at_s=time.time() - .18, count=2)
    elif constraint == 'waiting_ttft':
        import time
        existing.submitted_s = time.time() - .45
    new = r.dispatch('new', 100, 2)
    assert new.path == 'PD'
    assert new.route_estimate['slo_routing']['fallback']
    assert existing in r.active['m']


def test_missing_mixed_prediction_keeps_capacity_admitted_legacy_path_and_audits():
    class Uncovered(TimingModel):
        def step_seconds(self, *args):
            raise ValueError('outside measured coverage')
    r = router()
    configure(r, Uncovered())
    record = r.dispatch('new', 100, 2)
    assert record.path == 'PD'
    audit = record.route_estimate['slo_routing']
    assert audit['fallback'] and audit['selected'] is None
    assert all(e['reason'] == 'prediction_unavailable' for e in audit['exclusions'])
    assert r.slo_routing_summary()['decisions'][audit['reason']] == 1


def test_unavailable_transfer_can_only_fall_back_to_a_predicted_safe_mixed_route():
    class UncoveredTransfer(TimingModel):
        def transfer_seconds(self, *args):
            raise ValueError('transfer outside measured coverage')
    r = router()
    configure(r, UncoveredTransfer())
    record = r.dispatch('new', 100, 3)
    assert record.path == 'M'
    assert record.route_estimate['slo_routing']['selected']['tpot_s'] == .02
    assert any(e['reason'] == 'prediction_unavailable' for e in record.route_estimate['slo_routing']['exclusions'])


def test_unknown_clock_is_not_silently_replaced_with_max_frequency():
    r = router()
    configure(r)
    r._slo_route_frequency = lambda iid: None
    record = r.dispatch('new', 100, 2)
    assert record.path == 'PD' and record.route_estimate['slo_routing']['fallback']
    assert record.route_estimate['slo_routing']['predictions'] == []


def test_extrapolated_decode_value_cannot_qualify_a_mixed_spillover():
    class ExtrapolatingModel(TimingModel):
        def decode_supported(self, batch, context, f):
            return False
    r = router()
    configure(r, ExtrapolatingModel())
    record = r.dispatch('new', 100, 2)
    assert record.path == 'PD'
    assert record.route_estimate['slo_routing']['predictions'] == []
    assert record.route_estimate['slo_routing']['fallback']


def test_engine_running_batch_limit_is_not_a_hard_admission_queue_limit():
    r = router()
    old = r.dispatch('old', 100, 100)
    r.first_token(old)
    configure(r, max_num_seqs=1)
    r.set_accepting('m', False)
    new = r.dispatch('new', 100, 100)
    assert new.path == 'PD' and r.rejected == 0
    assert old in r.active['d'] and r._active_ids['old'] is old
    assert new.route_estimate['slo_routing']['fallback']


def test_capacity_exhaustion_rejects_before_reserving_and_disabled_control_is_unchanged():
    r = router()
    configure(r)
    assert r.dispatch('oversized', 9999, 2) is None
    assert not r._active_ids and r.rejected == 1
    control = router()
    configure(control, enabled=False)
    record = control.dispatch('short', 100, 2)
    assert record.path == 'PD' and record.route_estimate == {}


def test_observed_first_decode_token_records_the_actual_handoff_interval():
    r = router()
    record = r.dispatch('new', 100, 2)
    r.first_token(record, at_s=100.)
    record.pd_handoff_started_s = record.first_token_s
    r.token(record, at_s=100.4)
    r.token(record, at_s=100.5)
    assert record.first_decode_token_s == 100.4
    assert record.route_estimate['observed_handoff_s'] == pytest.approx(.4)


@pytest.mark.parametrize('floor', [0., .05])
def test_small_legacy_transfer_and_manual_floor_cannot_prove_two_token_pd_safe(floor):
    class LegacyCopy(TimingModel):
        def transfer_seconds(self, n):
            return .001
    r = router()
    configure(r, LegacyCopy(), handoff_floor_s=floor)
    record = r.dispatch('new', 100, 2)
    assert record.path == 'M'
    assert record.route_reason == 'pd_first_gap_unavailable_spillover'
    audit = record.route_estimate['slo_routing']
    assert all(p['path'] == 'M' for p in audit['predictions'])
    assert any(e['reason'] == 'prediction_unavailable' and 'first-gap coverage' in e['detail']
               for e in audit['exclusions'])
    assert audit['selected']['tpot_s'] == .02


def test_two_token_measured_first_gap_includes_first_decode_without_legacy_double_count():
    class MeasuredGap(TimingModel):
        def pd_first_gap_seconds(self, **query):
            assert query == dict(input_tokens=100, output_tokens=2, f_P_mhz=2520, f_D_mhz=2520,
                                 batch=1, context_tokens=102, prefill_instance='p', decode_instance='d')
            return .09

        def transfer_seconds(self, n):
            pytest.fail('signed/copy transfer must not be added to a measured first gap')
    r = router()
    configure(r, MeasuredGap())
    record = r.dispatch('new', 100, 2)
    assert record.path == 'PD'
    prediction = record.route_estimate['slo_routing']['selected']
    assert prediction['tpot_s'] == pytest.approx(.09)
    assert prediction['transfer_s'] is None
    assert prediction['handoff_source'] == 'measured_first_gap_including_first_decode'


@pytest.mark.parametrize('gap', [None, float('nan'), -.01, True])
def test_invalid_explicit_first_gap_does_not_qualify_two_token_pd(gap):
    class InvalidGap(TimingModel):
        def pd_first_gap_seconds(self, **query):
            return gap
    r = router()
    configure(r, InvalidGap())
    record = r.dispatch('new', 100, 2)
    assert record.path == 'M'
    assert all(p['path'] == 'M' for p in record.route_estimate['slo_routing']['predictions'])


def test_missing_first_gap_keeps_unproven_capacity_fallback_when_mixed_cannot_serve():
    r = router()
    configure(r)
    r.set_accepting('m', False)
    record = r.dispatch('new', 100, 2)
    audit = record.route_estimate['slo_routing']
    assert record.path == 'PD' and audit['fallback']
    assert audit['reason'] == 'legacy_capacity_fallback_unproven_slo'
    assert audit['selected'] is None
    assert record in r.active['d'] and r._active_ids['new'] is record
