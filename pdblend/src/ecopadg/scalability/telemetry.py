"""Benchmark-only observation; production request interfaces are unchanged."""
import asyncio
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import platform
import resource
import time

from ecopadg.serving.planner import JointPlanner


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    def percentile(q):
        position = (len(values) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {"count": len(values), "p50_ms": percentile(.5), "p95_ms": percentile(.95),
            "p99_ms": percentile(.99), "max_ms": values[-1]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory():
    """Linux current RSS and process-lifetime peak, separately labelled."""
    rss = None
    try:
        pages = int(Path('/proc/self/statm').read_text().split()[1])
        rss = pages * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        pass
    return {"rss_bytes": rss, "process_lifetime_peak_rss_bytes":
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)}


@contextmanager
def physical_core_affinity():
    """Pin before creating the planning worker; restore caller affinity after."""
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('CPU benchmark requires Linux CPU affinity support')
    original = os.sched_getaffinity(0)
    physical = {}
    for cpu in sorted(original):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            key = (int((root / 'physical_package_id').read_text()),
                   int((root / 'core_id').read_text()))
        except (OSError, ValueError) as exc:
            raise RuntimeError('Cannot verify physical CPU topology') from exc
        physical.setdefault(key, cpu)
    if len(physical) < 2:
        raise RuntimeError('Benchmark requires two available distinct physical cores')
    selected = list(physical.items())[:2]
    cpus = [cpu for _, cpu in selected]
    model = platform.processor()
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            model = line.split(':', 1)[1].strip()
            break
    os.sched_setaffinity(0, cpus)
    try:
        yield {"model": model, "logical_cpu_ids": cpus,
               "physical_cores": [{"package": k[0], "core": k[1]} for k, _ in selected],
               "planning_workers": 1, "physical_core_count": 2}
    finally:
        os.sched_setaffinity(0, original)


class TimedLock:
    """Drop-in context-manager wrapper around the actual asyncio lock."""
    def __init__(self, lock=None):
        self.lock = lock or asyncio.Lock()
        self.wait_ms = []
        self.hold_ms = []

    async def __aenter__(self):
        started = time.perf_counter()
        await self.lock.acquire()
        self.wait_ms.append((time.perf_counter() - started) * 1000)
        self.acquired = time.perf_counter()
        return self

    async def __aexit__(self, *exc):
        self.hold_ms.append((time.perf_counter() - self.acquired) * 1000)
        self.lock.release()

    def samples(self, kind):
        return ({"kind": kind, "operation": index, "wait_ms": wait, "hold_ms": hold}
                for index, (wait, hold) in enumerate(zip(self.wait_ms, self.hold_ms)))


class ObservedPlanner(JointPlanner):
    """Time the full existing algorithm, including its first unbounded enumeration."""
    def candidates(self, snapshot, request, now):
        started = time.perf_counter()
        result = super().candidates(snapshot, request, now)
        if getattr(self, '_observations', None) is not None:
            self._observations.append({"latency_ms": (time.perf_counter() - started) * 1000,
                                       "candidate_count": len(result)})
        return result

    def measured_plan(self, snapshot, pending, *, now, submitted_s=None):
        began = time.perf_counter()
        cpu = time.thread_time()
        self._observations = []
        try:
            result = self.plan(snapshot, pending, now=now)
            enumerations = self._observations
        finally:
            self._observations = None
        elapsed = (time.perf_counter() - began) * 1000
        first = enumerations[0]['candidate_count'] if enumerations else 0
        return result, {"kind": "planner", "latency_ms": elapsed,
            "cpu_ms": (time.thread_time() - cpu) * 1000,
            "worker_wait_ms": 0 if submitted_s is None else max(0, (began - submitted_s) * 1000),
            "candidate_count": sum(x['candidate_count'] for x in enumerations),
            "first_candidate_count": first, "candidate_calls": len(enumerations),
            "candidate_latency_ms": sum(x['latency_ms'] for x in enumerations),
            "first_candidate_latency_ms": enumerations[0]['latency_ms'] if enumerations else 0,
            "feasible": result.feasible, "no_candidate": first == 0,
            "active_rejection": not result.feasible and first > 0,
            "budget_fallback": 'decision budget exceeded' in result.reason,
            "reason": result.reason, "nominal_budget_ms": self.decision_budget_s * 1000,
            "budget_exceeded": elapsed > self.decision_budget_s * 1000}


def summarize_plans(samples):
    rows = [s for s in samples if s.get('kind') == 'planner']
    denominator = len(rows)
    return {"latency": distribution(r['latency_ms'] for r in rows),
        "candidate_latency": distribution(r['candidate_latency_ms'] for r in rows),
        "first_candidate_latency": distribution(r['first_candidate_latency_ms'] for r in rows),
        "worker_wait": distribution(r['worker_wait_ms'] for r in rows),
        "cpu_ms_total": sum(r['cpu_ms'] for r in rows),
        "first_candidate_count_min": min((r['first_candidate_count'] for r in rows), default=0),
        "candidate_count_total": sum(r['candidate_count'] for r in rows),
        **{key + '_ratio': sum(bool(r[key]) for r in rows) / denominator if denominator else None
           for key in ('budget_fallback', 'no_candidate', 'active_rejection', 'budget_exceeded')}}
