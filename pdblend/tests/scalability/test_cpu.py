import asyncio
from dataclasses import asdict, replace
import json
import time

import pytest

from ecopadg.scalability.cpu import concurrent_cell, parser, planner_cell, run, validate_args
from ecopadg.scalability.fixtures import (coverage, layout_roles, load_inputs,
    make_planner, make_snapshot, request, synthetic_inputs)
from ecopadg.scalability.telemetry import ObservedPlanner, TimedLock, distribution
from ecopadg.serving.state import StateManager, StalePlan


def arguments(tmp_path):
    args = parser().parse_args(['--smoke', '--output', str(tmp_path / 'out')])
    args.instances = [4]
    args.seeds = [101]
    args.layouts = ['selective']
    args.active_modes = ['total32']
    args.planner_calls = 2
    args.concurrent_seconds = .5
    args.frequency_period = args.role_period = .1
    args.arrivals_per_instance = 20.
    return args


@pytest.mark.parametrize('n', [4, 8, 128])
@pytest.mark.parametrize('layout', ['mixed', 'selective', 'pd'])
@pytest.mark.parametrize('active_mode', ['per_instance4', 'total32'])
def test_virtual_snapshot_preserves_unique_ids_and_decode_ownership(n, layout, active_mode):
    snapshot = make_snapshot(n, layout, active_mode, 101, 100.)
    ids = [r.request_id for i in snapshot.instances for r in i.requests]
    assert len(ids) == len(set(ids))
    assert all(not i.requests and not i.running for i in snapshot.instances if i.role == 'prefill')
    if active_mode == 'total32':
        assert len(ids) == 32
    else:
        assert all(len(i.requests) == 4 for i in snapshot.instances if i.role != 'prefill')


def test_virtual_pd_enumerates_every_pair_without_claiming_physical_links():
    profiles, transfers, costs, _ = synthetic_inputs()
    # Pretend a fixture covers one exact pair. The virtual adapter explicitly
    # reuses costs for every pair, preserving the measured source value.
    transfers = [replace(transfers[0], source_gpus=(0,), target_gpus=(1,))]
    planner = make_planner(profiles, transfers, costs)
    snapshot = make_snapshot(16, 'pd', 'total32', 101, 100.)
    result = coverage(planner, snapshot, (request('new', 100.),), 100.)
    assert result['all_paths_covered']
    assert result['feasible_path_count'] == 64
    assert result['candidate_count'] == 64 * 3
    assert planner.transfers[0].source_gpus == (0,)
    assert planner.transfer_store.links[0].source_sha256 == transfers[0].source_sha256


def test_uncovered_pd_rejected_even_with_feasible_mixed_path():
    profiles, _, costs, _ = synthetic_inputs()
    planner = make_planner(profiles, [], costs)
    snapshot = make_snapshot(4, 'selective', 'total32', 101, 100.)
    check = coverage(planner, snapshot, (request('new', 100.),), 100.)
    assert check['candidate_count'] > 0
    assert not check['all_paths_covered']
    assert check['feasible_path_count'] == 2
    assert check['expected_path_count'] == 3


def test_first_enumeration_measured_outside_nominal_budget():
    profiles, transfers, costs, _ = synthetic_inputs()
    class SlowEnumeration(ObservedPlanner):
        def candidates(self, *args):
            time.sleep(.015)
            return super().candidates(*args)
    planner = SlowEnumeration(profiles, transfers, frequency_costs=costs)
    snapshot = make_snapshot(4, 'mixed', 'per_instance4', 101, 100.)
    plan, sample = planner.measured_plan(snapshot, (request('new', 100.),), now=100.)
    assert plan.feasible
    assert sample['latency_ms'] >= 15
    assert sample['budget_exceeded'] and sample['budget_fallback']
    assert sample['first_candidate_count'] > 0


def test_missing_inputs_cannot_become_formal_evidence(tmp_path):
    args = arguments(tmp_path)
    args.smoke = False
    with pytest.raises(ValueError, match='Formal replay requires'):
        load_inputs(args)


def test_planner_only_does_not_require_unused_role_costs(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.smoke = False
    args.modes = ['planner']
    profiles, _, frequencies, _ = synthetic_inputs()
    args.profiles = tmp_path / 'profile-codec-fixture.json'
    args.profiles.write_text('{}')
    args.frequency_costs = tmp_path / 'frequency-codec-fixture.json'
    args.frequency_costs.write_text(json.dumps([asdict(c) for c in frequencies]))
    monkeypatch.setattr('ecopadg.scalability.fixtures.ProfileStore.load', lambda _: profiles)
    loaded = load_inputs(args)
    assert loaded[3] == []
    args.modes = ['concurrent']
    with pytest.raises(ValueError, match='Concurrent replay requires --role-costs'):
        load_inputs(args)


def test_default_telemetry_matches_production(tmp_path):
    assert parser().parse_args(['--output', str(tmp_path)]).telemetry_period == .1


@pytest.mark.parametrize('field,value', [('concurrent_seconds', float('nan')),
                                        ('arrivals_per_instance', 0), ('instances', [6])])
def test_invalid_cli_bounds(tmp_path, field, value):
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_args(args)


def test_real_replay_runs_commit_token_and_periodic_paths(tmp_path):
    async def exercise():
        args = arguments(tmp_path)
        profiles, transfers, costs, role_costs = synthetic_inputs()
        planner = make_planner(profiles, transfers, costs)
        samples = []
        result = await concurrent_cell(planner, role_costs, 4, 'selective', 'total32', 101, args, samples)
        assert result['measurement_valid'], result['invalid_reason']
        assert result['counts']['successful_commits'] > 0
        assert result['counts']['token_updates'] > 0
        assert result['periodic']['frequency']['latency']['count'] > 0
        assert result['periodic']['role']['latency']['count'] > 0
        assert any(r['kind'] == 'state_lock' for r in samples)
        assert any(r['kind'] == 'commit_lock' for r in samples)
        assert result['control_latency']['p99_ms'] is not None
        assert result['planner']['latency']['p99_ms'] is not None
    asyncio.run(exercise())


def test_failed_reservations_cannot_count_as_commits(tmp_path, monkeypatch):
    async def reject(*args, **kwargs):
        raise StalePlan('forced version race')
    monkeypatch.setattr(StateManager, 'reserve', reject)
    async def exercise():
        args = arguments(tmp_path)
        profiles, transfers, costs, role_costs = synthetic_inputs()
        samples = []
        result = await concurrent_cell(make_planner(profiles, transfers, costs), role_costs,
                                       4, 'pd', 'total32', 101, args, samples)
        assert not result['measurement_valid']
        assert result['successful_commit_throughput_rps'] == 0
        assert result['counts']['stale_retries'] > 0
        assert result['counts']['rejected'] > 0
    asyncio.run(exercise())


def test_smoke_results_have_scope_hashes_and_cannot_be_formal(tmp_path):
    args = arguments(tmp_path)
    args.modes = ['planner']
    result = asyncio.run(run(args, {'physical_core_count': 2, 'planning_workers': 1}))
    assert result == 0
    summary_path = next(args.output.glob('*/summary.json'))
    summary = json.loads(summary_path.read_text())
    assert summary['scope'] == 'control_plane_replay'
    assert summary['measurement_valid']
    assert not summary['formal_evidence']
    assert summary['provenance']['synthetic']
    assert summary['source_hashes']
    assert not summary['slow_topology']['measured']
    with pytest.raises(FileExistsError):
        asyncio.run(run(args, {}))


def test_distribution_reports_milliseconds_and_empty_is_not_zero():
    assert distribution([])['p99_ms'] is None
    assert distribution([1, 2, 3])['p50_ms'] == 2
    assert distribution([1, 2, 3])['p99_ms'] == pytest.approx(2.98)
