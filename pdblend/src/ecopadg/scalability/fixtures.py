"""Explicit virtual CPU work; hardware measurements are never manufactured.

Every virtual P/D route uses the upper envelope of measured TP1 transfer costs.
This expands algorithmic work only: GPU IDs here name virtual resources, and no
latency, energy or connectivity claim is made about an actual larger machine.
"""
from dataclasses import replace
import json
from pathlib import Path
import random
import math

from ecopadg.serving.frequency import FrequencyCost
from ecopadg.serving.profiles import ProfilePoint, ProfileStore
from ecopadg.serving.reconfigure import RoleCost
from ecopadg.serving.transfers import TransferCost, TransferStore
from ecopadg.serving.types import InstanceState, RequestBudget, RuntimeSnapshot

from .telemetry import ObservedPlanner, sha256


ROLES = ('mixed', 'prefill', 'decode')


def read_rows(path, key):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    if key not in data:
        raise ValueError(f'{path}: expected a list or a {key!r} collection')
    return data[key]


def synthetic_inputs():
    """Software smoke only; never accepted by the formal CLI path."""
    source = 'synthetic-software-fixture-not-hardware'
    points = [ProfilePoint(role, 1, frequency, 4096, 8192, 64,
                .01, .01, power, 30., .05, 1, source, .01)
              for role in ROLES for frequency, power in ((900, 110), (1500, 130), (2520, 220))]
    profiles = ProfileStore(points, fingerprint=source, gpu_count=8, idle_unallocated_gpu_w=20)
    transfers = [TransferCost(1, 1, 4096, .005, 1., source, True,
                              import_seconds_upper=.003, profile_batch=64)]
    frequencies = [FrequencyCost(1, a, b, .0001, .01, source)
                   for a in (900, 1500, 2520) for b in (900, 1500, 2520) if a != b]
    roles = [RoleCost(1, a, b, .001, .01, source) for a in ROLES for b in ROLES if a != b]
    return profiles, transfers, frequencies, roles


def load_inputs(args):
    synthetic = args.smoke and args.profiles is None
    if synthetic:
        profiles, transfers, frequency_costs, role_costs = synthetic_inputs()
        hashes = {"fixture": "synthetic-software-fixture-not-hardware"}
    else:
        if not args.profiles or not args.frequency_costs:
            raise ValueError('Formal replay requires --profiles and --frequency-costs')
        if 'concurrent' in args.modes and not args.role_costs:
            raise ValueError('Concurrent replay requires --role-costs for actual periodic role search')
        profiles = ProfileStore.load(args.profiles)
        transfers = [TransferCost(**r) for r in read_rows(args.transfers, 'links')] if args.transfers else []
        frequency_costs = [FrequencyCost(**r) for r in read_rows(args.frequency_costs, 'frequency_costs')]
        role_costs = [RoleCost(**r) for r in read_rows(args.role_costs, 'role_costs')] if args.role_costs else []
        hashes = {name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
                  for name, path in (("profiles", args.profiles), ("transfers", args.transfers),
                    ("frequency_costs", args.frequency_costs), ("role_costs", args.role_costs)) if path}
        if any(not c.source_sha256 or c.tp != 1 or c.source_role not in ROLES
               or c.target_role not in ROLES or c.source_role == c.target_role
               or not all(math.isfinite(v) and v >= 0 for v in (c.time_upper_s, c.energy_upper_j))
               for c in role_costs if c.tp == 1):
            raise ValueError('TP1 role costs must retain source hashes and valid measured bounds')
    available = set.intersection(*(set(profiles.frequencies(role, 1)) for role in ROLES))
    selected = set(args.frequencies) if args.frequencies else available
    if not selected or not selected <= available or 2520 not in selected:
        raise ValueError('Require common TP1 role profiles including 2520 MHz; no frequency extrapolation')
    reachable = {(c.source_mhz, c.target_mhz) for c in frequency_costs if c.tp == 1}
    if any((a, b) not in reachable for a in selected for b in selected if a != b):
        raise ValueError('Frequency costs do not cover every reachable TP1 transition')
    role_pairs = {(c.source_role, c.target_role) for c in role_costs if c.tp == 1}
    if 'concurrent' in args.modes and any((a, b) not in role_pairs for a in ROLES for b in ROLES if a != b):
        raise ValueError('Role costs do not cover all TP1 role transitions')
    # Restrict the tested frequency set; preserve measured values and provenance.
    profiles = ProfileStore([p for p in profiles.points if p.frequency_mhz in selected],
        fingerprint=profiles.fingerprint, node_residency_w=profiles.node_residency_w,
        idle_unallocated_gpu_w=profiles.idle_unallocated_gpu_w, gpu_count=profiles.gpu_count,
        parked_residency_w_by_tp=profiles.parked_residency_w_by_tp,
        interference_points=profiles.interference_points)
    return profiles, transfers, frequency_costs, role_costs, {
        "synthetic": synthetic, "input_hashes": hashes, "frequencies_mhz": sorted(selected),
        "hardware_performance_measurement": False,
        "virtual_resource_assumption": 'TP1 profile replay; every virtual P/D pair uses the '
          'conservative measured transfer envelope. No physical connectivity or GPU performance is inferred.'}


def make_planner(profiles, transfers, frequency_costs):
    planner = ObservedPlanner(profiles, transfers, frequency_costs=frequency_costs,
                             depth=3, width=8, decision_budget_s=.01)
    # Remove placement constraints only in this explicitly virtual workload.
    # Preserve each original time, energy, clock, bucket and source hash.
    virtual_links = [replace(t, source_gpus=(), target_gpus=(),
                             interconnect_class='', topology_sha256='') for t in transfers
                     if t.source_tp == t.target_tp == 1 and t.validated and t.source_sha256]
    planner.transfer_store = TransferStore(virtual_links)
    return planner


def layout_roles(n, layout):
    if n < 4 or n % 4:
        raise ValueError('CPU matrix instance counts must be positive multiples of 4')
    if layout == 'mixed':
        return ['mixed'] * n
    if layout == 'selective':
        return ['mixed'] * (n // 2) + ['prefill'] * (n // 4) + ['decode'] * (n // 4)
    if layout == 'pd':
        return ['prefill'] * (n // 2) + ['decode'] * (n // 2)
    raise ValueError(f'Unknown layout {layout}')


def request(request_id, now, input_tokens=128, output_tokens=128):
    return RequestBudget(request_id, now, input_tokens, output_tokens, 5., .15,
                         output_limit=output_tokens)


def make_snapshot(n, layout, active_mode, seed, now, *, input_tokens=128, output_tokens=128):
    roles = layout_roles(n, layout)
    consumers = [i for i, role in enumerate(roles) if role != 'prefill']
    counts = {i: 0 for i in range(n)}
    if active_mode == 'per_instance4':
        for i in consumers:
            counts[i] = 4
    elif active_mode == 'total32':
        for index in range(32):
            counts[consumers[index % len(consumers)]] += 1
    else:
        raise ValueError(f'Unknown active mode {active_mode}')
    rng = random.Random(seed)
    instances = []
    for index, role in enumerate(roles):
        budgets = tuple(replace(request(f'background-{seed}-{index}-{j}', now - 1.,
                    input_tokens, output_tokens), emitted=rng.randint(16, min(31, output_tokens - 1)),
                    first_token_s=now - .1, last_token_s=now)
                        for j in range(counts[index]))
        instances.append(InstanceState(f'virtual-{index}', role, 1, (index,), now, 0,
            2520, 1_000_000, len(budgets), 0, requests=budgets,
            free_transfer_bytes=16 * 1024**3, transfer_bytes_per_token=206848))
    return RuntimeSnapshot(1, now, tuple(instances))


def coverage(planner, snapshot, pending, now):
    """Validate all layout paths, not just a surviving mixed fallback."""
    paths = planner.candidates(snapshot, pending[0], now)
    actual = {(p.routes[0].prefill_id, p.routes[0].decode_id) for p in paths}
    mixed = [i for i in snapshot.instances if i.role == 'mixed']
    prefill = [i for i in snapshot.instances if i.role == 'prefill']
    decode = [i for i in snapshot.instances if i.role == 'decode']
    expected = {(i.instance_id, i.instance_id) for i in mixed}
    expected.update((p.instance_id, d.instance_id) for p in prefill for d in decode)
    return {"expected_path_count": len(expected), "feasible_path_count": len(actual),
            "candidate_count": len(paths), "all_paths_covered": bool(expected) and actual == expected,
            "missing_paths": sorted(expected - actual)[:20]}
