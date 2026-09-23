import json

from pdblend.bench import tp_smoke
from tests.pdblend.test_tp_runtime import profiled_model


def test_trace_keeps_long_inputs_and_seed_701():
    one, two = tp_smoke.build_trace(), tp_smoke.build_trace()
    assert one == two
    assert {request.input_tokens for request in one} == {512, 2048, 7168}
    assert all(request.input_tokens + request.max_tokens <= 8192 for request in one)


def test_mechanism_smoke_needs_real_pd_outcomes_and_does_not_claim_golden(monkeypatch, tmp_path):
    profile = tmp_path / 'profile.json'
    profiled_model().save(profile)

    def execute(*args, **kwargs):
        trace, out = args[5], args[7]
        assert kwargs['fixed_plan'].detail['mechanism_forced_roles']
        (out / 'outcomes.jsonl').write_text(''.join(json.dumps(dict(
            idx=request.idx, error=None, completion_tokens=16, sampling_seed=701,
            first_token_s=1, finished_s=2, path='PD' if request.input_tokens >= 1024 else 'M')) + '\n'
            for request in trace))
        (out / 'power.jsonl').write_text('[1, [100]]\n')
        (out / 'routes.jsonl').write_text('{}\n')
        (out / 'metering.json').write_text('{"error": null}')
        return {'quarantined_instances': []}

    monkeypatch.setattr(tp_smoke, 'run_point', execute)
    result = tp_smoke.run_smoke(model='Qwen2.5-7B-Instruct', gpus=[0, 1, 2, 3], tp=1,
                                profile=profile, out=tmp_path / 'out', duration_s=30)
    assert result['status'] == 'passed' and result['scope']['pd_path']
    assert not result['scope']['complete_kv'] and not result['scope']['output_golden']
    assert not result['formal_eligible'] and not result['policy_decision']


def test_impossible_32b_tp1_rejects_before_launcher(monkeypatch, tmp_path):
    monkeypatch.setattr(tp_smoke, 'run_point', lambda *a, **k: (_ for _ in ()).throw(AssertionError('launch')))
    result = tp_smoke.run_smoke(model='Qwen2.5-32B-Instruct', gpus=[0, 1], tp=1,
                                profile=tmp_path / 'missing.json', out=tmp_path / 'out')
    assert result['status'] == 'failed'
    assert 'unsupported topology' in result['errors'][0]
