"""Coordinate isolated/concurrent probes across GPU-disjoint profile jobs."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import time

from ..engine.client import EngineClient
from .parallel import common_window_overlap, evaluate_interference


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')
    os.replace(temp, path)


class ProfileWave:
    """A fixed cohort keeps engines resident through qualification and sampling.

    The coordinator directory is shared between containers, not an output
    profile. Every member has its own profile and exclusively owned UUIDs.
    Failed qualification serializes that cohort without accepting bad samples.
    """

    def __init__(self, root: Path, member: str, *, timeout_s: float = 3600):
        self.root, self.member, self.timeout_s = Path(root), member, timeout_s
        self.spec = json.loads((self.root / 'wave.json').read_text())
        self.members = self.spec['members']
        if not self.members or len(set(self.members)) != len(self.members) or member not in self.members:
            raise ValueError('invalid profile cohort membership')
        self.parallel = False
        # Formal cross-job qualification requires a coordinator-issued cohort
        # identity.  A local members list alone is useful for diagnostics but
        # cannot establish that other jobs were held at the barrier.
        self.cohort_id = self.spec.get('cohort_id')
        self.coordinator = bool(self.spec.get('coordinator'))

    @classmethod
    def from_environment(cls):
        root, member = os.environ.get('PDBLEND_PROFILE_WAVE'), os.environ.get('PDBLEND_PROFILE_MEMBER')
        if not root and not member:
            return None
        if not root or not member:
            raise ValueError('both profile wave and member are required')
        return cls(Path(root), member)

    def write(self, phase: str, value: dict) -> None:
        atomic_json(self.root / f'{self.member}.{phase}.json', value)

    async def wait(self, phase: str, members=None) -> list[dict]:
        members = self.members if members is None else members
        start = time.monotonic()
        paths = [self.root / f'{m}.{phase}.json' for m in members]
        while True:
            for member in self.members:
                error = self.root / f'{member}.error.json'
                if error.exists():
                    raise RuntimeError(f'profile cohort peer failed: {error.read_text()}')
            if all(p.is_file() for p in paths):
                return [json.loads(p.read_text()) for p in paths]
            if time.monotonic() - start > self.timeout_s:
                raise TimeoutError(f'profile cohort waiting for {phase}: {members}')
            await asyncio.sleep(.25)

    async def probe(self, profiler, phase: str) -> dict:
        local_ready = {}
        async def before_measure(instance, repeat):
            ready = local_ready.setdefault(repeat, set())
            if instance in ready:
                raise RuntimeError('duplicate parallel qualification window')
            ready.add(instance)
            marker = f'parallel-window-{repeat}-ready'
            if len(ready) == len(profiler.specs):
                self.write(marker, dict(time=time.time(), instances=sorted(ready)))
            # The last local instance publishes ready; every member is already
            # continuously decoding and settled before starting any sampler.
            await self.wait(marker)

        async with AsyncExitStack() as stack:
            clients = [await stack.enter_async_context(EngineClient(s.instance_id, s.base_url))
                       for s in profiler.specs]
            for spec in profiler.specs:
                profiler._lock(2100, spec.gpus)
            rows = await asyncio.gather(*[
                profiler._decode_batch(c, s.gpus, 8, 1024, 64, f'wave-{phase}-{s.instance_id}',
                    **({'before_measure': lambda repeat, iid=s.instance_id: before_measure(iid, repeat)}
                       if phase == 'parallel' and self.spec.get('synchronize_parallel_windows') is True else {}))
                for c, s in zip(clients, profiler.specs)])
        return {'instances': rows, 'point': {'frequency': 2100, 'batch': 8, 'context': 1024},
                'gpu_uuids': profiler.raw['environment']['gpu_uuids']}

    async def qualify(self, profiler, fleet=None) -> None:
        uuids = profiler.raw['environment']['gpu_uuids']
        self.write('ready', {'gpu_uuids': uuids, 'profile_key': profiler.profile_key.as_dict()})
        ready = await self.wait('ready')
        all_uuids = [u for r in ready for u in r['gpu_uuids']]
        if len(all_uuids) != len(set(all_uuids)):
            raise RuntimeError('profile cohort overlaps physical GPUs')
        # Every non-current cohort member remains idle for the whole probe.
        for member in self.members:
            if member == self.member:
                self.write('isolated', await self.probe(profiler, 'isolated'))
            await self.wait('isolated', [member])
        self.write('parallel_ready', {'time': time.time()})
        await self.wait('parallel_ready')
        self.write('parallel', await self.probe(profiler, 'parallel'))
        isolated, concurrent = await self.wait('isolated'), await self.wait('parallel')
        comparisons = []
        for name, alone, together in zip(self.members, isolated, concurrent):
            if len(alone['instances']) != len(together['instances']):
                raise RuntimeError('profile cohort changed instance count')
            for index, (a, b) in enumerate(zip(alone['instances'], together['instances'])):
                comparisons.append(dict(member=name, instance=index, **evaluate_interference(a, b)))
        window_check = common_window_overlap([inst for p in concurrent for inst in p['instances']])
        overlap = window_check['passed']
        self.parallel = overlap and all(row['passed'] for row in comparisons)
        cross_job = bool(self.cohort_id and self.coordinator)
        evidence = dict(complete=True, passed=self.parallel, cross_job=cross_job,
                        cohort_id=self.cohort_id, coordinator=self.coordinator, limit=.05,
                        member=self.member, members=self.members, isolated=isolated,
                        parallel=concurrent, comparisons=comparisons, overlapping_windows=overlap,
                        common_measurement_windows=window_check,
                        synchronized_parallel_windows=self.spec.get('synchronize_parallel_windows') is True,
                        fallback=None if self.parallel and cross_job else 'serial_cohort',
                        energy_comparable=False)
        path = profiler.out_dir / 'samples' / 'external-interference.json'
        atomic_json(path, evidence)
        profiler.raw['external_interference'] = dict(
            complete=True, passed=self.parallel,
            measured_mode='parallel' if self.parallel else 'serial_cohort',
            fallback=evidence['fallback'], cross_job=cross_job, cohort_id=self.cohort_id,
            samples_file=str(path.relative_to(profiler.out_dir)),
            samples_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        profiler._checkpoint()
        print(f'cohort {self.member}: parallel={self.parallel}; {comparisons}', flush=True)

    async def qualify_external(self, profiler, fleet=None) -> dict:
        """Run the cross-job barrier and return the persisted qualification.

        This named entry point is the integration contract for the profiler
        driver.  It keeps cohort coordination outside the profiler's local
        isolated/concurrent probe and returns the raw receipt for callers that
        need to gate subsequent sampling.
        """
        await self.qualify(profiler, fleet)
        return dict(profiler.raw.get('external_interference', {}))

    @asynccontextmanager
    async def measurement(self):
        if not self.parallel:
            index = self.members.index(self.member)
            await self.wait('done', self.members[:index])
        try:
            yield
            self.write('done', {'time': time.time()})
            if self.spec.get('keep_peers_resident_until_all_done') is True:
                # Publish completion before waiting so serial fallback can
                # admit the next member. Keep the caller's resident Fleet and
                # GPU lease until every peer has finished sampling.
                await self.wait('done')
        except BaseException as exc:
            self.write('error', {'error': repr(exc)})
            raise
