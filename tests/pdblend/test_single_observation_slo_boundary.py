import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pdblend.bench import single_observation_slo_boundary as boundary
from pdblend.bench.resident_session import ResidentGroupSession, digest, engine_signature, write_new
from test_resident_comparison import identity


def put(path, value):
    write_new(path, value)
    return boundary.binding(path)


def fixture(tmp_path, *, steps=12):
    corpus = put(tmp_path/'corpus.json', dict(evaluation=[dict(prompt=[1,2], output_tokens=4)],
                                             tuning=[dict(prompt=[8,9], output_tokens=4)]))
    trace = put(tmp_path/'trace.json', dict(corpus_sha256=corpus['sha256']))
    prior = put(tmp_path/'prior.json', dict(selection_split='tuning'))
    engine = identity()
    points = [dict(name=f'7b-pdblend-alpaca-x{scale:g}-seed701', model_id='Qwen2.5-7B-Instruct',
        system='pdblend', dataset='alpaca', revision='new-profile', seed=701, duration_s=150.,
        scale=scale, rate_rps=scale*10, slo=dict(ttft_s=1., tpot_s=.1), blockers=[],
        result_policy='all_recorded_windows/v1', trace=trace, engine_identity=engine,
        inputs=dict(trace=trace, planning_trace=prior)) for scale in (.5,.25,.75,1.)]
    group = dict(session_id='test-resident', model_id=points[0]['model_id'], points=points,
                 engine_identity=engine, engine_signature=engine_signature(engine))
    group['extension_policy'] = boundary.build_policy(group, tmp_path/'policy', run_id='round-new',
        corpus_refs=dict(alpaca=corpus), base_rates=dict(alpaca=10.), max_steps_per_dataset=steps)
    return group


def metrics(passed=True):
    return dict(duration_s=150., service_start_s=1., service_end_s=151.,
        service_started_s=1., service_finished_s=151., offered_requests=100,
        successful_requests=100, failed_requests=0, joint_slo_requests=100,
        ttft_samples=100, tpot_samples=100, ttft_p99_s=.5 if passed else 2., tpot_p99_s=.02,
        unresolved_requests=0, invalid_timing_requests=0, energy_service_j=None)


def fake_factory(ref, dataset, scale, out):
    policy = boundary.read_bound(ref)
    point = deepcopy(boundary.read_bound(policy['datasets'][dataset]['template_point']))
    point.update(name=f'7b-pdblend-{dataset}-x{scale:.12g}-seed701', scale=scale,
                 rate_rps=scale*10, boundary_policy=ref)
    return point, put(Path(out)/'point.json', point)


class Adapter:
    def __init__(self, *, fail_above=1.25, fail_reset=None, incomplete_above=None):
        self.starts = self.closes = 0
        self.scales = []; self.registered = []; self.reset_count = 0
        self.fail_above, self.fail_reset, self.incomplete_above = fail_above, fail_reset, incomplete_above

    async def start(self, group):
        self.starts += 1
        return dict(engine_loads=1)

    async def register_point(self, point):
        self.registered.append(point['scale'])

    async def reset(self, point):
        self.reset_count += 1
        return dict(passed=self.reset_count != self.fail_reset)

    async def execute(self, point, out):
        self.scales.append(point['scale'])
        write_new(out/'raw.json', dict(scale=point['scale']))
        value = metrics(point['scale'] <= self.fail_above)
        if self.incomplete_above is not None and point['scale'] > self.incomplete_above:
            value['invalid_timing_requests'] = 1
        return dict(evidence_valid=False, formal_eligible=False, metrics=value)

    async def drain(self, point):
        return dict(passed=True)

    async def close(self):
        self.closes += 1
        return dict(passed=True)


@pytest.mark.parametrize('records,status,next_rate,lower,upper', [
    ([(1.,'pass')], 'expanding', 1.25, 1., None),
    ([(1.,'fail')], 'searching_lower', .8, None, 1.),
    ([(.5,'pass'),(1.,'fail')], 'narrowing', .625, .5, 1.),
    ([(1.,'pass'),(1.25,'fail')], 'bracketed', None, 1., 1.25),
    ([(1.,'incomplete')], 'incomplete_observation', None, None, None),
])
def test_rate_decision(records, status, next_rate, lower, upper):
    state = boundary.boundary_state([dict(scale=r, verdict=v) for r,v in records])
    assert (state['status'],state['next_rate_scale'],state['passed_lower'],state['failed_upper']) == (
        status,next_rate,lower,upper)


def test_nonmonotonic_is_explicit_not_hidden():
    state = boundary.boundary_state([dict(scale=.5,verdict='fail'),dict(scale=1.,verdict='pass')])
    assert state['nonmonotonic'] and state['next_rate_scale'] == 1.25
    assert not state['saturation_observed']


def test_slo_verdict_ignores_energy_but_not_timing(tmp_path):
    point = fixture(tmp_path)['points'][0]
    receipt = dict(cleanup_passed=True, result=dict(metrics=metrics()))
    assert boundary.observation_verdict(point, receipt)['verdict'] == 'pass'
    receipt['result']['metrics']['invalid_timing_requests'] = 1
    assert boundary.observation_verdict(point, receipt)['verdict'] == 'incomplete'


def test_all_explicit_failed_requests_establish_upper(tmp_path):
    point = fixture(tmp_path)['points'][0]
    value = metrics()
    value.update(successful_requests=0,failed_requests=100,joint_slo_requests=0,
                 ttft_samples=0,tpot_samples=0,ttft_p99_s=None,tpot_p99_s=None)
    assert boundary.observation_verdict(point,dict(cleanup_passed=True,result=dict(metrics=value)))['verdict']=='fail'


def test_dynamic_windows_share_load_and_have_replayable_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(boundary, 'make_extension_point', fake_factory)
    group = fixture(tmp_path)
    adapter = Adapter()
    report = asyncio.run(ResidentGroupSession(group,adapter,tmp_path/'session').run())
    assert report['complete'] and adapter.starts == adapter.closes == 1
    assert adapter.scales == [.5,.25,.75,1.,1.25,1.5625]
    assert adapter.registered == [1.25,1.5625]
    loaded = boundary.read_extension_manifest(tmp_path/'session/extensions/latest.json')
    assert len(loaded['points']) == 2 and len(loaded['receipts']) == 6
    state = loaded['manifest']['boundaries']['alpaca']
    assert state['status'] == 'bracketed' and (state['passed_lower'],state['failed_upper']) == (1.25,1.5625)
    assert state['passed_lower_point'] and state['failed_upper_receipt']
    assert not loaded['manifest']['continuation_required']


def test_budget_is_not_saturation_and_resume_does_not_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(boundary, 'make_extension_point', fake_factory)
    group = fixture(tmp_path,steps=1)
    first = Adapter()
    report = asyncio.run(ResidentGroupSession(group,first,tmp_path/'first').run())
    assert report['continuation_required'] and first.scales[-1] == 1.25
    state = boundary.read_extension_manifest(report['extension_manifest'])['manifest']['boundaries']['alpaca']
    assert not state['saturation_observed'] and state['failed_upper'] is None
    second = Adapter()
    report = asyncio.run(ResidentGroupSession(group,second,tmp_path/'second',previous=[tmp_path/'first']).run())
    assert report['complete'] and not report['continuation_required']
    assert second.scales == [1.5625] and second.starts == 1


def test_reset_failure_quarantines_dynamic_session(tmp_path, monkeypatch):
    monkeypatch.setattr(boundary,'make_extension_point',fake_factory)
    adapter = Adapter(fail_reset=5)
    report = asyncio.run(ResidentGroupSession(fixture(tmp_path),adapter,tmp_path/'session').run())
    assert not report['complete'] and report['quarantined'] and adapter.closes == 1
    assert adapter.scales == [.5,.25,.75,1.]
    state = boundary.read_extension_manifest(report['extension_manifest'])['manifest']['boundaries']['alpaca']
    assert state['status'] == 'incomplete_observation' and not state['saturation_observed']


def test_changed_manifest_boundary_or_generated_config_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(boundary,'make_extension_point',fake_factory)
    report=asyncio.run(ResidentGroupSession(fixture(tmp_path),Adapter(),tmp_path/'session').run())
    manifest=boundary.read_bound(report['extension_manifest'])
    manifest['boundaries']['alpaca']['passed_lower']=99
    forged=put(tmp_path/'forged.json',manifest)
    with pytest.raises(ValueError,match='endpoints'):
        boundary.read_extension_manifest(forged)
    policy=boundary.read_bound(manifest['policy'])
    point=boundary.read_bound(manifest['points'][0]['point'])
    point['inputs']['system_config']={'path':'/forged','sha256':'0'*64}
    with pytest.raises(ValueError,match='frozen inputs'):
        boundary.validate_generated_point(policy,point)


def test_registered_adapter_checks_identity_before_gpu_work(tmp_path):
    from pdblend.bench.comparison_runtime import NativeResidentAdapter
    group=fixture(tmp_path)
    point,_=fake_factory(group['extension_policy'],'alpaca',1.25,tmp_path/'generated')
    adapter=NativeResidentAdapter(tmp_path/'native',base_port=18000)
    adapter.group=group
    point['engine_identity']['instances'][0]['launch_options']['kv_connector']='different'
    with pytest.raises(ValueError,match='frozen configuration'):
        asyncio.run(adapter.register_point(point))
