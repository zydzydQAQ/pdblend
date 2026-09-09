"""Measured, amortized local TP/replica changes for PDBlend's slow control.

The bounded search replays admission-visible shape strata through the same
joint estimator. A change must also leave a feasible routing layout during
preparation. This is an uncertain demand model, never a formal energy result.
"""
from dataclasses import dataclass, replace
import itertools
import math
import time
from .types import InstanceState

@dataclass(frozen=True)
class MeasuredCapacity:
    tp: int
    kv_tokens: int
    transfer_buffer_bytes: int
    transfer_bytes_per_token: int
    source_sha256: str

    def __post_init__(self):
        if self.tp not in (1, 2, 4, 8) or min(self.kv_tokens, self.transfer_buffer_bytes, self.transfer_bytes_per_token) <= 0 or (not self.source_sha256):
            raise ValueError('measured per-TP memory and staging capacities required')

def idle(instance, now):
    return instance.accepting and 0 <= now - instance.timestamp_s <= 1 and (not instance.requests) and (not instance.running) and (not instance.waiting) and (not instance.reserved_kv_tokens) and (not instance.kv_allocations) and (not instance.transfer_allocations) and (not instance.reserved_transfer_bytes)

class PDBlendTopologyPlanner:

    def __init__(self, planner, costs, capacities, *, node_gpus=tuple(range(8)), error_fraction=0.3, budget_s=0.25):
        self.planner = planner
        self.costs = tuple(costs)
        self.capacities = {c.tp: c for c in capacities}
        self.node_gpus = tuple(node_gpus)
        self.error_fraction = error_fraction
        self.budget_s = budget_s
        if not self.costs or not self.capacities:
            raise ValueError('PDBlend slow topology requires measured switch and memory costs')
        if not 0 <= error_fraction < 1 or budget_s <= 0:
            raise ValueError('invalid forecast error or search budget')

    def score(self, snapshot, requests, now):
        energy = 0.0
        for request in requests:
            plans = self.planner.candidates(snapshot, request, now)
            if not plans:
                return None
            plan = plans[0]
            energy += plan.routes[0].incremental_j
            snapshot = self.planner.advance(snapshot, plan, request)
        return energy

    def allocations(self, tps, available):
        if not tps:
            yield ()
            return
        tp = tps[0]
        for first in range(0, 8 - tp + 1, tp):
            group = tuple(range(first, first + tp))
            if not set(group) <= available:
                continue
            for rest in self.allocations(tps[1:], available - set(group)):
                if rest and tp == len(rest[0]) and (group > rest[0]):
                    continue
                yield ((group,) + rest)

    def choose(self, snapshot, forecast, now, *, cached_weights=False):
        began = time.perf_counter()
        requests = forecast.requests
        original = self.score(snapshot, requests, now)
        if original is None:
            return None
        best = None
        for cost in self.costs:
            source = tuple(sorted(cost.source_tps))
            target = tuple(sorted(cost.target_tps))
            if not cost.source_sha256 or (cost.cached_weights and (not cached_weights)) or source == target or (not target) or (len(source) > 2) or (len(target) > 2) or any((tp not in self.capacities for tp in target)) or (not all((math.isfinite(v) and v >= 0 for v in (cost.duration_upper_s, cost.energy_upper_j)))) or (cost.duration_upper_s >= forecast.horizon_s):
                continue
            for removed in itertools.combinations(snapshot.instances, len(source)):
                if tuple(sorted((i.tp for i in removed))) != source or not all((idle(i, now) for i in removed)):
                    continue
                ids = {i.instance_id for i in removed}
                others = tuple((i for i in snapshot.instances if i.instance_id not in ids))
                temporary = replace(snapshot, instances=others)
                interim = self.score(temporary, requests, now)
                if interim is None:
                    continue
                occupied = {g for i in others for g in i.gpus}
                for groups in self.allocations(target, set(self.node_gpus) - occupied):
                    for roles in itertools.product(('mixed', 'prefill', 'decode'), repeat=len(target)):
                        if time.perf_counter() - began > self.budget_s:
                            return best
                        added = []
                        for (index, (tp, gpus, role)) in enumerate(zip(target, groups, roles)):
                            capacity = self.capacities[tp]
                            added.append(InstanceState('pdb-forecast-' + str(index), role, tp, gpus, now, 0, getattr(self.planner, 'max_frequency', 2520), capacity.kv_tokens, 0, 0, free_transfer_bytes=capacity.transfer_buffer_bytes, transfer_bytes_per_token=capacity.transfer_bytes_per_token))
                        changed = self.score(replace(snapshot, instances=others + tuple(added)), requests, now)
                        if changed is None or changed >= original:
                            continue
                        useful = max(0.0, forecast.horizon_s - cost.duration_upper_s)
                        gain = (original - changed) * forecast.rate_lower_rps * useful / len(requests)
                        gain *= 1 - self.error_fraction
                        lost = max(0.0, interim - original) * forecast.rate_upper_rps * cost.duration_upper_s / len(requests)
                        upper = cost.energy_upper_j + lost * (1 + self.error_fraction)
                        if gain <= upper:
                            continue
                        proposal = dict(remove_ids=tuple((i.instance_id for i in removed)), source_generations={i.instance_id: i.generation for i in removed}, replacements=tuple((dict(tp=i.tp, gpus=i.gpus, role=i.role) for i in added)), savings_lower_j=gain, cost_upper_j=upper, source_cost=cost, capacity_loss_upper_j=lost * (1 + self.error_fraction), snapshot_version=snapshot.version, created_s=now, expires_s=now + 1, demand_source='historical admission shapes and conservative arrival rates')
                        if best is None or gain - upper > best['savings_lower_j'] - best['cost_upper_j']:
                            best = proposal
        return best
