import json
import numpy as np
import pytest

from pdblend.profile.calibration import holdout_plan, evaluate_holdout, _checkpoint_points
from pdblend.profile.decode_fit import fit_split_b1, predict, supported


def test_holdout_keeps_legal_shapes_and_contains_unseen_interpolation():
    rows = [dict(freq_mhz=f, batch=b, context_tokens=c)
            for f in (900, 1200, 1500, 1800, 2100, 2520)
            for c in (256, 1024, 4096) for b in (1, 4, 8, 16, 32, 64)
            if b*(c+64) <= 65536*.9]
    plan = holdout_plan(dict(freqs=[900, 1200, 1500, 1800, 2100, 2520], decode=rows, kv_capacity_tokens=65536))
    assert {p['freq_mhz'] for p in plan['decode']} == {900, 1200, 1500, 1800, 2100, 2520}
    assert all(p['batch']*(p['context_tokens']+64) <= 65536*.9 for p in plan['decode'])
    assert any(p['unseen_shape'] and p['batch'] == 2 for p in plan['decode'])
    assert len(plan['decode']) <= 48
    assert plan['repeats'] == 3 and plan['measure_s'] == 5


def test_split_fit_learns_small_batch_overhead_without_high_batch_distortion():
    rows = []
    for b in (1, 4, 8, 16, 32, 64):
        for c in (256, 1024, 4096):
            value = .01+.0000004*c if b == 1 else .013+.0001*b+.00000002*b*c+.0000002*b*b
            rows.append(dict(batch=b, context_tokens=c, effective_context_tokens=c, step_seconds=value))
    spec = fit_split_b1(rows, 1000000)
    assert max(abs(predict(spec, r['batch'], r['context_tokens'])/r['step_seconds']-1) for r in rows) < 1e-8
    assert not supported(spec, 128, 1024)
    assert not supported(spec, 4, 8192)
    assert predict(spec, 1, 1024) < predict(spec, 2, 1024) < predict(spec, 4, 1024)
    assert spec['validation_status'] == 'training_only'


def test_split_fit_rejects_insufficient_low_batch_data():
    with pytest.raises(ValueError, match='three B1'):
        fit_split_b1([], 1000)


def holdout_fixture(root):
    import hashlib

    def sample(name, payload):
        path = root / name
        path.write_text(json.dumps(payload))
        return dict(samples_file=name, samples_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    reps = []
    for index, context in enumerate((100, 200, 300)):
        reps.append(dict(effective_context_tokens=context, step_seconds=context/1000, power_w=200,
                         steady_window_s=5, min_steps=10, power_samples=2, frequency_samples=1,
                         **sample(f'decode-{index}.json', dict(power=[[1, [200]], [6, [200]]], frequency=[[1, [1500]]]))))
    point = dict(freq_mhz=1500, batch=4, context_tokens=64, effective_context_tokens=200,
                 step_seconds=.2, power_w=200, repeats=reps)
    mixed = dict(valid=True, base_step_s=.1, alone_prefill_s=.1, probe_ttft_s=.2, **sample('mixed.json', {}))
    return dict(prefill=[], decode=[point], mixed=[mixed.copy() for _ in range(12)])


class HoldoutModel:
    freqs = (1500,)
    upper = 500

    def step_seconds(self, batch, context, frequency):
        if not self.decode_supported(batch, context, frequency):
            raise ValueError('outside frozen domain')
        return context / 1000

    def decode_supported(self, batch, context, frequency):
        return frequency == 1500 and 0 < context <= self.upper

    def decode_power_w(self, batch, frequency, *, ctx=None):
        return 200


def test_holdout_evaluates_every_repetition_not_only_passing_median(tmp_path):
    data = holdout_fixture(tmp_path)
    assert evaluate_holdout(data, HoldoutModel(), tmp_path)['passed']
    data['decode'][0]['repeats'][2]['step_seconds'] = .4
    # Deliberately dishonest stored prediction is ignored by the evaluator.
    data['decode'][0]['repeats'][2]['prediction'] = dict(relative_error=0)
    audit = evaluate_holdout(data, HoldoutModel(), tmp_path)
    assert not audit['passed']
    bad = [x for x in audit['failures'] if x['metric'] == 'decode_time_repeat']
    assert len(bad) == 1 and bad[0]['repeat'] == 2 and bad[0]['relative_error'] == pytest.approx(.25)


def test_outside_coverage_is_sampling_domain_error_not_fit_residual(tmp_path):
    data = holdout_fixture(tmp_path)
    model = HoldoutModel(); model.upper = 250
    audit = evaluate_holdout(data, model, tmp_path)
    assert not audit['passed']
    invalid = audit['invalid_sampling_domain']
    assert len(invalid) == 1 and invalid[0]['repeat'] == 2
    assert invalid[0]['error_class'] == 'measurement_domain_error'
    assert invalid[0]['relative_error'] is None


def test_checkpoint_retains_nested_window_fields_and_rejects_corrupt_evidence(tmp_path):
    data = holdout_fixture(tmp_path)
    data['decode'][0]['sampling_method'] = 'consecutive_windows_shared_prefill'
    data['decode'][0]['repeats'][0]['prediction'] = dict(context_tokens=100, relative_error=0)
    before = json.dumps(data)
    assert _checkpoint_points(data, tmp_path)['decode'] == {(1500, 4, 64)}
    assert json.dumps(data) == before
    (tmp_path / 'decode-0.json').write_text('{}')
    with pytest.raises(ValueError, match='corrupt'):
        _checkpoint_points(data, tmp_path)


def test_holdout_matrix_cannot_pass_with_missing_planned_points(tmp_path):
    data = holdout_fixture(tmp_path)
    plan = dict(prefill=[], decode=[dict(freq_mhz=1500, batch=4, context_tokens=64),
                                   dict(freq_mhz=1500, batch=8, context_tokens=64)])
    audit = evaluate_holdout(data, HoldoutModel(), tmp_path, expected_plan=plan)
    assert not audit['passed']
    assert any(x['metric'] == 'decode_matrix_coverage' for x in audit['failures'])


def test_shared_prefill_cli_is_opt_in(monkeypatch, tmp_path):
    import sys
    from pdblend.profile import calibration
    received = []
    monkeypatch.setattr(calibration, 'run_holdout', lambda **kw: (received.append(kw) or {'complete': True}))
    base = ['calibration', '--candidate-dir', str(tmp_path), '--model', 'm', '--gpus', '0', '1',
            '--base-port', '15000', '--out', str(tmp_path)]
    for enabled in (False, True):
        monkeypatch.setattr(sys, 'argv', base + (['--shared-prefill-windows'] if enabled else []))
        with pytest.raises(SystemExit) as done:
            calibration.main()
        assert done.value.code == 0
        assert received[-1]['shared_prefill_windows'] is enabled
