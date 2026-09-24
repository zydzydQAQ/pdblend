"""Window-boundary qualification and early retirement of profile groups.

Initial members come from a frozen cohort specification. Every change waits
for in-flight windows to finish, then for the departing member's actual engine
cleanup, and finally repeats the isolated/parallel measurement. No old window
is relabelled with a new qualification. New jobs must join a separately frozen
cohort; the queue prevents unqualified peers from loading during this cohort.
"""
from __future__ import annotations

import asyncio
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from pdblend.profile.collection.wave import ProfileWave, atomic_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class SamplingEpochs:
    def __init__(self, root: Path, member: str, profiler, *, timeout_s=3600., poll_s=.25,
                 probe_callback=None):
        self.root, self.member, self.profiler = Path(root), member, profiler
        self.timeout_s, self.poll_s = timeout_s, poll_s
        spec = json.loads((self.root/'cohort.json').read_text())
        if (not spec.get('cohort_id') or not spec.get('members')
                or len(set(spec['members'])) != len(spec['members']) or member not in spec['members']):
            raise ValueError('invalid frozen sampling cohort')
        self.spec = spec
        self.probe_callback = probe_callback
        self.binding = None
        with self._state() as state:
            if not state:
                state.update(cohort_sha256=digest(spec), epoch=0, phase='loading',
                    active=list(spec['members']), ready={}, qualified={}, boundaries=[], retiring=[], cleaned=[],
                    parallel=False, errors={})
            if state['cohort_sha256'] != digest(spec):
                raise ValueError('sampling cohort specification changed')

    @contextmanager
    def _state(self):
        with (self.root/'epoch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root/'epoch-state.json'
            value = json.loads(path.read_text()) if path.exists() else {}
            try:
                yield value
                atomic_json(path, value)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _check(self, state):
        if state.get('errors'):
            # Embedding all peer exception strings recursively grows on each
            # broadcast. The immutable first cause remains in the state file.
            raise RuntimeError('sampling cohort failed for '+', '.join(sorted(state['errors']))
                               +'; see '+str(self.root/'epoch-state.json'))

    def _layout(self, state):
        return {m:state['ready'][m]['gpu_uuids'] for m in state['active']}

    async def ready(self):
        """Register only after the resident engines have finished loading."""
        uuids = list(self.profiler.raw['environment']['gpu_uuids'])
        with self._state() as state:
            self._check(state)
            if state['phase'] != 'loading' or self.member in state['ready']:
                raise RuntimeError('cohort readiness may only be registered once')
            occupied = {u for row in state['ready'].values() for u in row['gpu_uuids']}
            if not uuids or len(set(uuids)) != len(uuids) or occupied.intersection(uuids):
                raise RuntimeError('profile groups overlap physical GPUs')
            state['ready'][self.member] = dict(gpu_uuids=uuids, at_s=time.time())
            if set(state['ready']) == set(state['active']):
                state['phase'] = 'qualifying'
        await self.window_boundary(None, None, 'ready')

    async def _qualify(self, epoch, members, layout):
        wave_dir = self.root/f'qualification-{epoch:04d}'
        wave_dir.mkdir(exist_ok=True)
        spec = dict(cohort_id=f'{self.spec["cohort_id"]}:epoch:{epoch}', coordinator=True,
            members=members, synchronize_parallel_windows=True,
            qualification_measure_s=self.spec.get('qualification_measure_s', 5.),
            keep_peers_resident_until_all_done=False, layout_sha256=digest(layout))
        with self._state():
            path = wave_dir/'wave.json'
            if path.exists() and json.loads(path.read_text()) != spec:
                raise RuntimeError('epoch layout changed during qualification')
            if not path.exists(): atomic_json(path, spec)
        # ProfileWave writes fixed relative sample paths. Give every epoch a
        # separate directory so earlier qualification bytes remain immutable.
        child = copy.copy(self.profiler)
        child.raw = copy.deepcopy(self.profiler.raw)
        child.out_dir = self.profiler.out_dir/'qualification-epochs'/f'{epoch:04d}'
        (child.out_dir/'samples').mkdir(parents=True, exist_ok=True)
        wave = ProfileWave(wave_dir, self.member, timeout_s=self.timeout_s)
        if self.probe_callback is not None:
            async def independent_probe(profile, phase):
                return await self.probe_callback(profile, phase, wave)
            wave.probe = independent_probe
        await wave.qualify_external(child)
        path = child.out_dir/'samples/external-interference.json'
        receipt = json.loads(path.read_text())
        if receipt.get('complete') is not True or receipt.get('cross_job') is not True:
            raise RuntimeError('epoch has no complete coordinator qualification')
        binding = dict(epoch_id=spec['cohort_id'], epoch=epoch,
            qualification_path=str(path), qualification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            layout_sha256=digest(layout), layout=layout,
            measured_mode='parallel' if wave.parallel else 'serial_cohort')
        self.binding = binding
        if epoch == 0:
            # The measurement child takes an immutable copy after ready().
            # Later epochs travel through the explicit per-window guard;
            # changing the parent's raw document would mix evidence streams.
            self.profiler.raw['external_interference'] = dict(child.raw['external_interference'],
                samples_file=str(path.relative_to(self.profiler.out_dir)), **binding)
            self.profiler._checkpoint()
        with self._state() as state:
            self._check(state)
            if state['epoch'] != epoch or state['phase'] != 'qualifying':
                raise RuntimeError('cohort changed while measuring qualification')
            state['qualified'][self.member] = dict(passed=wave.parallel, binding=binding)
            if set(state['qualified']) == set(members):
                state['parallel'] = all(v['passed'] for v in state['qualified'].values())
                state['phase'] = 'measuring'

    async def window_boundary(self, point=None, repeat=None, phase=None):
        """Called before every repeat; blocks during cleanup/requalification."""
        started = time.monotonic()
        while True:
            qualify = None
            with self._state() as state:
                self._check(state)
                if self.member not in state['active']:
                    raise RuntimeError('retired member cannot start another measurement')
                if state['phase'] == 'quiescing':
                    if self.member not in state['boundaries']: state['boundaries'].append(self.member)
                    self._advance_quiescence(state)
                if state['phase'] == 'qualifying' and self.member not in state['qualified']:
                    qualify = (state['epoch'], list(state['active']), self._layout(state))
                elif state['phase'] == 'measuring':
                    # Failed parallel qualification preserves the original
                    # serial fallback. Other engines stay idle until retirement.
                    if state['parallel'] or self.member == state['active'][0]:
                        if self.binding is None or self.binding['epoch'] != state['epoch']:
                            raise RuntimeError('measurement lacks current qualification')
                        return dict(self.binding)
            if qualify:
                try:
                    await self._qualify(*qualify)
                except BaseException as exc:
                    self.fail(exc); raise
            else:
                if time.monotonic()-started > self.timeout_s:
                    error = TimeoutError('profile epoch boundary wait exceeded')
                    self.fail(error); raise error
                await asyncio.sleep(self.poll_s)

    def qualification_guard(self):
        with self._state() as state:
            self._check(state)
            # A retirement request is safe during a window: departing engines
            # cannot clean up until every member has reached the next boundary.
            if (state['phase'] not in ('measuring', 'quiescing') or not self.binding
                    or self.binding['epoch'] != state['epoch'] or self.member not in state['active']):
                raise RuntimeError('measurement crossed an unqualified layout boundary')
            path = Path(self.binding['qualification_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.binding['qualification_sha256']:
                raise RuntimeError('qualification receipt bytes changed')
            return dict(self.binding)

    @staticmethod
    def _advance_quiescence(state):
        if set(state['boundaries']) >= set(state['active']):
            state['phase'] = 'cleanup'

    async def retire(self):
        """Wait for a safe boundary, then permit this group's engine cleanup.

        Caller MUST subsequently call ``released`` after stopping its Fleet.
        Peers remain paused until that physical cleanup has finished.
        """
        start = time.monotonic()
        while True:
            with self._state() as state:
                self._check(state)
                if self.member not in state['active']:
                    raise RuntimeError('member already retired')
                if self.member not in state['retiring']: state['retiring'].append(self.member)
                if self.member not in state['boundaries']: state['boundaries'].append(self.member)
                if state['phase'] == 'measuring': state['phase'] = 'quiescing'
                if state['phase'] == 'quiescing': self._advance_quiescence(state)
                if state['phase'] == 'cleanup': return
            if time.monotonic()-start > self.timeout_s:
                error = TimeoutError('profile retirement could not reach a safe boundary')
                self.fail(error); raise error
            await asyncio.sleep(self.poll_s)

    def released(self):
        """Call only after native drain and owned engine cleanup completed."""
        with self._state() as state:
            self._check(state)
            if state['phase'] != 'cleanup' or self.member not in state['retiring']:
                raise RuntimeError('cleanup was not permitted by the cohort')
            if self.member in state['cleaned']:
                raise RuntimeError('duplicate cleanup receipt')
            state['cleaned'].append(self.member)
            if set(state['cleaned']) == set(state['retiring']):
                state['active'] = [m for m in state['active'] if m not in state['retiring']]
                state.update(epoch=state['epoch']+1, qualified={}, boundaries=[], retiring=[], cleaned=[])
                state['phase'] = 'qualifying' if state['active'] else 'complete'

    def fail(self, exc):
        with self._state() as state:
            failure = state.setdefault('errors', {}).setdefault(self.member,
                dict(error=repr(exc), at_s=time.time()))
            # Peers inside ProfileWave.wait() are waiting on the qualification
            # directory, not this state file. Wake that barrier as well so a
            # failed rank cannot keep the other leased groups for an hour.
            atomic_json(self.root/f'qualification-{state["epoch"]:04d}'/
                        f'{self.member}.error.json', failure)
