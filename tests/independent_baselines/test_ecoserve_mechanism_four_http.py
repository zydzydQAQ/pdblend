"""Real four-endpoint HTTP execution of the bounded manual EcoServe probe."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip('fastapi')
_spec = importlib.util.spec_from_file_location('eco_http_fixture', Path(__file__).with_name('test_ecoserve_native_http.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
Engine, native_services = _fixture.Engine, _fixture.native_services
from pdblend_baselines.ecoserve.mechanism_four import run


class SlowEngine(Engine):
    def __init__(self, identifier, scheduler):
        super().__init__(identifier, scheduler)
        self.blocks = {}
        def block_ids(rid):
            if rid not in self.blocks:
                number = len(self.blocks)*10+1
                self.blocks[rid] = [number, number+1]
            return (self.blocks[rid],)
        scheduler.kv_cache_manager.get_block_ids = block_ids

    async def generate(self, *args):
        async for event in super().generate(*args):
            yield event
            await asyncio.sleep(.015)


def config(tmp_path, identifiers, *, pressure=True):
    csv = tmp_path/'own-profile.csv'
    # Explicit CPU fixture observations exercise the unchanged author rule.
    # The test never publishes these numbers as measured GPU profile data.
    csv.write_text('Length,Prefill Time\n16,10\n4096,'+('16000' if pressure else '10')+'\n')
    return dict(instances=[dict(id=iid, tp=1, gpus=[gpu]) for gpu, iid in enumerate(identifiers)],
        eco_prefill_csv=str(csv), slo_ttft_s=5., slo_tpot_s=.15,
        eco_initial_instances=3, eco_macro_lower=2, eco_macro_upper=3,
        eco_scale_period_s=5., eco_state_poll_s=.005, eco_probe_phase_timeout_s=3.,
        request_timeout_s=10., eco_drain_timeout_s=3., eco_probe_output_tokens=128)


@pytest.mark.asyncio
async def test_four_member_probe_records_real_live_kv_cancel_and_unchanged_streams(tmp_path, monkeypatch):
    identifiers = ('first', 'second', 'third', 'idle-fourth')
    async with native_services(monkeypatch, identifiers, SlowEngine) as (engines, endpoints):
        cfg = config(tmp_path, identifiers)
        cfg['eco_probe_pressure_requests'] = 8
        value = await run(cfg, endpoints, tmp_path/'four.json')
        assert value['status'] == 'passed', (value['errors'], value['missing_required_actions'],
            [(row['kind'], row.get('instance_id'), row.get('controls')) for row in value['journal']
             if row['kind'] in ('eco_admission', 'eco_observed_output_hold')],
            {row.get('instance_id') for row in value['journal'] if row['kind'] == 'eco_output_flush'})
        assert value['checks']['live_kv_continuity'] and value['checks']['complete_output']
        assert value['checks']['policy_rotation'] and value['checks']['actual_hold']
        assert value['checks']['held_output_flush'] and value['checks']['cancel_release']
        assert value['cancellation']['before']['native_tokens'] >= 2
        assert value['cancellation']['released']
        assert value['background_scaling_suspended_for_manual_probe']
        assert value['original_scale_period_s'] == 5.
        assert value['manualprimitive'] and not value['automatic_policy_triggered']
        assert not value['formal_eligible'] and not value['energy_comparable'] and not value['complete_reproduction']
        commits = [row for row in value['journal'] if row['kind'] == 'eco_membership_commit']
        assert [row['instance_id'] for row in commits] == ['idle-fourth', 'idle-fourth']
        assert commits[0]['split'] and commits[1]['merge']
        for stage in ('before_split', 'after_split', 'after_merge'):
            assert len(value['live_kv_snapshots'][stage]) == 3
        for rid, outcome in value['requests'].items():
            if 'main' in rid:
                assert outcome['continuous_complete_output'] and outcome['events'][-1]['finished']
                assert len(outcome['native_events']) == len(outcome['events']) == 128
        assert any(operation == 'abort' and row['request_id'] == 'eco-four-701-cancel'
                   for engine in engines.values() for operation, row in engine.calls)
        assert all(not engine.scheduler.requests for engine in engines.values())
        assert json.loads((tmp_path/'four.json').read_text())['status'] == 'passed'
        assert not value['pressure']['executed']
        assert value['pressure']['reason'] == 'required_policy_evidence_already_observed'
        assert not any('pressure' in rid for rid in value['requests'])


@pytest.mark.asyncio
async def test_no_real_policy_rotation_or_buffering_is_inconclusive(tmp_path, monkeypatch):
    identifiers = ('one', 'two', 'three', 'four')
    async with native_services(monkeypatch, identifiers, SlowEngine) as (_, endpoints):
        value = await run(config(tmp_path, identifiers, pressure=False), endpoints, tmp_path/'quiet.json')
        assert value['status'] == 'inconclusive'
        assert 'policy_rotation' in value['missing_required_actions']
        assert 'actual_hold' in value['missing_required_actions']
        assert value['checks']['cancel_release'] and value['checks']['live_kv_continuity']
        assert value['checks']['complete_output']
        assert value['pressure']['requested_count'] == 0
        assert value['pressure']['reason'] == 'disabled'


@pytest.mark.asyncio
async def test_optional_real_pressure_runs_after_membership_and_delivers_held_packets(tmp_path, monkeypatch):
    identifiers = ('p0', 'p1', 'p2', 'p3')
    cfg = config(tmp_path, identifiers, pressure=False)
    # Roughly 1.1s per long input: original four prompts fit within the SLO,
    # while actual simultaneous long requests exercise the unchanged policy.
    Path(cfg['eco_prefill_csv']).write_text('Length,Prefill Time\n16,3\n4096,625\n')
    cfg.update(eco_probe_pressure_requests=8, request_timeout_s=25.)
    async with native_services(monkeypatch, identifiers, SlowEngine) as (engines, endpoints):
        value = await run(cfg, endpoints, tmp_path/'pressure.json')
        assert value['status'] == 'passed', (value['errors'], value['missing_required_actions'])
        pressure = value['pressure']
        assert pressure['executed'] and pressure['requested_count'] == 8
        assert pressure['reason'] == 'missing_policy_evidence'
        assert not all(pressure['before_checks'].values()) and all(pressure['after_checks'].values())
        assert len(pressure['request_ids']) == 8 and value['checks']['pressure_complete_output']
        started = next(row['at_s'] for row in value['journal'] if row['kind'] == 'eco_pressure_start')
        assert all(row['at_s'] < started for row in value['journal']
                   if row['kind'] == 'eco_client_sse' and 'main' in row['request_id'])
        held = [packet for row in value['journal'] if row['kind'] == 'eco_observed_output_hold'
                for packet in row['packets'] if packet['request_id'] in pressure['request_ids']]
        assert held
        for rid in pressure['request_ids']:
            outcome = value['requests'][rid]
            assert len(outcome['payload']['prompt']) == 7168 and outcome['payload']['max_tokens'] == 512
            assert outcome['continuous_complete_output'] and outcome['events'][-1]['finished']
            assert outcome['native_events'][-1]['token_index'] == outcome['events'][-1]['token_index'] == 512
        assert any(row['kind'] == 'eco_client_sse' and row['request_id'] == packet['request_id']
                   and row['payload']['token_index'] == packet['token_index']
                   and row['payload']['token_ids'] == packet['token_ids']
                   for row in value['journal'] for packet in held)
        assert all(not engine.scheduler.requests for engine in engines.values())


@pytest.mark.asyncio
async def test_incomplete_pressure_output_cannot_pass(tmp_path, monkeypatch):
    from types import SimpleNamespace
    class ShortPressureEngine(SlowEngine):
        async def generate(self, prompt, params, rid):
            if rid == 'eco-four-701-pressure-0':
                params = SimpleNamespace(**dict(vars(params), max_tokens=4))
            async for event in super().generate(prompt, params, rid):
                yield event
    identifiers = ('x0', 'x1', 'x2', 'x3')
    cfg = config(tmp_path, identifiers, pressure=False)
    cfg.update(eco_probe_pressure_requests=1, request_timeout_s=25.)
    async with native_services(monkeypatch, identifiers, ShortPressureEngine) as (engines, endpoints):
        value = await run(cfg, endpoints, tmp_path/'short-pressure.json')
        assert value['status'] == 'inconclusive' and value['pressure']['executed']
        assert not value['checks']['pressure_complete_output']
        assert any('continuous complete' in error for error in value['errors'])
        assert all(not engine.scheduler.requests for engine in engines.values())


@pytest.mark.asyncio
async def test_kv_prefix_change_cannot_become_a_successful_live_transition(tmp_path, monkeypatch):
    from pdblend_baselines.ecoserve.controller import EcoServeController
    original = EcoServeController.add_member
    identifiers = ('a', 'b', 'c', 'd')
    async with native_services(monkeypatch, identifiers, SlowEngine) as (engines, endpoints):
        async def changed(self, *args, **kwargs):
            result = await original(self, *args, **kwargs)
            for engine in engines.values():
                for rid, blocks in engine.blocks.items():
                    if 'main' in rid:
                        blocks[0] += 1000
            return result
        monkeypatch.setattr(EcoServeController, 'add_member', changed)
        value = await run(config(tmp_path, identifiers), endpoints, tmp_path/'changed.json')
        assert value['status'] == 'inconclusive'
        assert any('block prefix changed' in error for error in value['errors'])
        assert not value['checks']['live_kv_continuity']
        assert all(not engine.scheduler.requests for engine in engines.values())
