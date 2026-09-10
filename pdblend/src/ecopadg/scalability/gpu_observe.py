"""Measurement wrappers; scheduling choices and the serving API are unchanged."""
import asyncio
import time

from ecopadg.serving.observe import PlanningStats


class TimedLock:
    def __init__(self, records):
        self.lock = asyncio.Lock()
        self.records = records
        self.entered = None

    async def __aenter__(self):
        begin = time.perf_counter()
        await self.lock.acquire()
        self.entered = time.perf_counter()
        self.records.append(dict(kind='action_lock_wait', at_s=time.time(),
                                 elapsed_ms=1000 * (self.entered - begin)))
        return self

    async def __aexit__(self, *exc):
        elapsed = time.perf_counter() - self.entered
        self.lock.release()
        self.records.append(dict(kind='action_lock_hold', at_s=time.time(), elapsed_ms=1000 * elapsed))

    def locked(self):
        return self.lock.locked()


class SampledPlanningStats(PlanningStats):
    def __init__(self, records):
        super().__init__()
        self.records = records

    def finish(self, plan, elapsed_s):
        super().finish(plan, elapsed_s)
        self.records.append(dict(kind='admission_planning', at_s=time.time(),
            elapsed_ms=elapsed_s * 1000, feasible=bool(plan and plan.feasible),
            no_candidate=plan is None, reason=plan.reason if plan else 'no_candidate',
            fallback=bool(plan and 'decision budget exceeded' in plan.reason)))

    def discard(self, reason):
        super().discard(reason)
        self.records.append(dict(kind='discard', at_s=time.time(), reason=reason))


def instrument(controller):
    records = []
    controller.action_lock = TimedLock(records)
    controller.planning_stats = SampledPlanningStats(records)
    # The physical budget is the allocated subset. The input measured table is
    # untouched; no TP/latency/power operating point is synthesized.
    controller.profiles.gpu_count = len(controller.config['allocated_gpu_ids'])
    estimators = [controller.planner]
    if controller.distserve_scheduler:
        estimators.append(controller.distserve_scheduler.estimator)
    for planner in estimators:
        original = planner.candidates

        def candidates(*args, _original=original, _planner=planner, **kwargs):
            start = time.perf_counter()
            result = _original(*args, **kwargs)
            snapshot = args[0]
            roles = {role: sum(i.role == role for i in snapshot.instances)
                     for role in ('mixed', 'prefill', 'decode')}
            paths = roles['mixed'] + (roles['prefill'] * roles['decode'] if _planner.allow_pd else 0)
            records.append(dict(kind='candidate_generation', at_s=time.time(),
                elapsed_ms=(time.perf_counter()-start)*1000, feasible_candidates=len(result),
                possible_routes=paths))
            return result

        planner.candidates = candidates
    return records
