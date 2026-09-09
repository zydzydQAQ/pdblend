import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ecopadg.serving.eco_validation import validation_config, verify_execution, restore, validate


def evidence():
    routes = {'request-a': 'custom-a', 'request-b': 'custom-b'}
    events = [dict(instance=instance, request_ids=[rid], tokens=1,
                   role='mixed', mode='temporal', prefill=int(phase == 'prefill'),
                   decode=int(phase == 'decode'))
              for rid, instance in routes.items() for phase in ('prefill', 'decode')]
    owners = [dict(at_s=1, owners={'custom-a': ['request-a'], 'custom-b': ['request-b']})]
    return events, routes, owners


def test_real_phase_steps_and_physical_kv_are_required_for_each_request():
    events, routes, owners = evidence()
    result = verify_execution(events, routes, owners)
    assert result == dict(actual_steps=4, overlap_steps=0, requests_with_stationary_kv=2)
    with pytest.raises(RuntimeError, match='phase exclusion'):
        verify_execution([], routes, owners)  # Proxy-only window logs prove nothing.
    with pytest.raises(RuntimeError, match='actual prefill and decode'):
        verify_execution(events[:-1], routes, owners)
    with pytest.raises(RuntimeError, match='physical KV observation'):
        verify_execution(events, routes, [])


@pytest.mark.parametrize('change,error', [
    ({'decode': 1}, 'phase exclusion'),
    ({'mode': 'continuous'}, 'temporal engine mode'),
    ({'role': 'decode'}, 'temporal engine mode'),
    ({'instance': 'custom-b'}, 'unassigned instance'),
    ({'request_ids': ['unrelated-work']}, 'unassigned instance'),
])
def test_invalid_engine_evidence_cannot_pass(change, error):
    events, routes, owners = evidence()
    events[0].update(change)
    with pytest.raises(RuntimeError, match=error):
        verify_execution(events, routes, owners)


def test_kv_migration_is_rejected_even_when_final_routes_look_correct():
    events, routes, owners = evidence()
    owners.append(dict(at_s=2, owners={'custom-b': ['request-a']}))
    with pytest.raises(RuntimeError, match='KV appeared'):
        verify_execution(events, routes, owners)


def runtime_config(tmp_path):
    profile = tmp_path / 'measured.json'
    profile.write_text(json.dumps(dict(schema=2, measurement='hardware', points=[dict(
        role='mixed', tp=1, frequency_mhz=2520, input_tokens=2048, context_tokens=2560,
        batch=1, prefill_s=.1, iteration_s=.04, power_w=200, residency_w=30,
        error_fraction=.1, samples=1, source_sha256='measured-fixture')])) )
    return dict(profiles=str(profile), strategy='mixed', slo_ttft_s=5, slo_tpot_s=.1,
        topology={'must_not_start': True}, instances=[dict(id=f'custom-{i}', tp=1,
        gpus=[i], url=f'http://127.0.0.1:{19000 + i}', role='decode') for i in range(5)])


def test_configuration_uses_supplied_endpoints_and_profiles_without_physical_changes(tmp_path):
    config = runtime_config(tmp_path); original = deepcopy(config)
    result = validation_config(config, tmp_path / 'out', 128, 1024, 128)
    assert config == original
    assert [i['id'] for i in result['instances']] == ['custom-0', 'custom-1', 'custom-2', 'custom-3']
    assert result['instances'][1]['port'] == 19001
    assert result['profiles'] == config['profiles'] and 'topology' not in result
    assert result['strategy'] == 'ecoserve' and result['eco_initial_instances'] == 3
    assert result['eco_scale_period_s'] == 3600 and not result['park_idle']
    with pytest.raises(ValueError, match='at least four'):
        validation_config(dict(config, instances=config['instances'][:3]), tmp_path, 128, 1024, 128)
    with pytest.raises(ValueError, match='profile does not cover'):
        validation_config(config, tmp_path, 128, 4096, 128)
    duplicate = deepcopy(config); duplicate['instances'][1]['gpus'] = [0]
    with pytest.raises(ValueError, match='disjoint GPUs'):
        validation_config(duplicate, tmp_path, 128, 1024, 128)
    with pytest.raises(ValueError, match='fit the engine'):
        validation_config(config, tmp_path, 128, 8000, 512)


def test_restoration_uses_current_generation_and_continues_after_a_failed_member():
    async def run():
        original = {identifier: dict(role='decode', mode='continuous', admit_prefill=True, admit_decode=True)
                    for identifier in ('a', 'b')}
        class Backend:
            def __init__(self):
                self.calls = []
                self.states = {i: dict(role='mixed', mode='temporal', admit_prefill=False,
                    admit_decode=True, generation=8, acknowledged_generation=8) for i in original}
            async def json(self, identifier, path, payload=None):
                if path == '/control':
                    self.calls.append((identifier, payload))
                    if identifier == 'a': raise RuntimeError('injected restore failure')
                    self.states[identifier].update(payload, acknowledged_generation=payload['generation'])
                return self.states[identifier].copy()
        backend = Backend()
        errors = await restore(backend, original)
        assert len(errors) == 1 and errors[0]['instance'] == 'a'
        assert 'injected restore failure' in errors[0]['traceback']
        assert [i for i, _ in backend.calls] == ['a', 'b']
        assert backend.calls[1][1]['generation'] == 9
        assert backend.states['b']['role'] == 'decode'
    asyncio.run(run())


def test_failure_records_full_error_in_new_output_and_refuses_overwrite(tmp_path):
    async def run():
        config = tmp_path / 'bad.json'; config.write_text('{}')
        args = SimpleNamespace(config=config, out=tmp_path / 'out', runtime_dir=tmp_path,
            short_input=128, long_input=1024, output_tokens=128, timeout=1, port=0)
        with pytest.raises(RuntimeError, match='see raw.json'):
            await validate(args)
        raw = json.loads((args.out / 'raw.json').read_text())
        assert not raw['passed'] and not raw['complete']
        assert 'at least four explicitly configured' in raw['errors'][0]
        assert 'Traceback' in raw['errors'][0]
        assert not raw['cleanup_errors']
        with pytest.raises(FileExistsError): await validate(args)
    asyncio.run(run())
