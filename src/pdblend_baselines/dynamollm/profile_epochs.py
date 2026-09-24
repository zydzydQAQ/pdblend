"""Dynamo measurements behind the shared GPU-group sampling barrier.

Only coordination and interference comparison are shared. Representative
probes use this system's SSE/native-rank collector and NVML telemetry; no
PDBlend fitted model, workload estimator, or profile query is imported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import time

from .deployment import save, sha


class IdentityKey:
    def __init__(self, value):
        self.value = value

    def as_dict(self):
        return dict(self.value)


class QualificationFacade:
    """Minimal data/checkpoint protocol consumed by the shared barrier."""
    def __init__(self, out, metadata):
        self.out_dir = Path(out)
        self.profile_key = IdentityKey(metadata)
        self.raw = dict(system='dynamollm', environment=dict(
            gpu_uuids=list(metadata['gpu_uuids'].values())), identity=metadata)

    def _checkpoint(self):
        save(self.out_dir/'qualification-state.json', self.raw)


class DynamoEpochs:
    def __init__(self, root, member, output, metadata, transport, telemetry, iid, ownership):
        # The only PD package import is the common coordinator. Its callback
        # avoids constructing the PDBlend profiler or using its estimators.
        from pdblend.profile.sampling_epochs import SamplingEpochs
        self.output, self.transport, self.telemetry = Path(output), transport, telemetry
        self.iid, self.ownership = iid, ownership
        self.facade = QualificationFacade(output, metadata)
        self.epochs = SamplingEpochs(Path(root), member, self.facade,
                                     probe_callback=self.probe)
        self.retirement_permitted = False

    async def probe(self, facade, phase, wave):
        from .profile_v1 import measure_window, reduce_window
        point = dict(frequency_mhz=2100, batch=1, input_tokens=1024, output_tokens=64)
        self.telemetry.clock(self.transport.instances[self.iid]['gpus'], 2100)
        windows, reduced, artifacts = [], [], {}
        for repeat in range(3):
            async def before_measure():
                if phase == 'parallel':
                    marker = f'parallel-window-{repeat}-ready'
                    wave.write(marker, dict(time=time.time(), instances=[self.iid]))
                    await wave.wait(marker)
            window = await measure_window(self.transport, self.telemetry, self.iid, point,
                repeat=repeat, settle_s=2., measure_s=wave.qualification_measure_s,
                ownership=self.ownership,
                before_measure=before_measure)
            metric = reduce_window(window, tp=self.transport.instances[self.iid]['tp'],
                                   batch=point['batch'], frequency=point['frequency_mhz'])
            path = facade.out_dir/'samples'/f'dynamo-{phase}-{repeat}.json'
            save(path, window)
            artifacts[str(path.relative_to(facade.out_dir))] = sha(path)
            windows.append(dict(start_s=window['started_s'], end_s=window['finished_s']))
            reduced.append(metric)
        return dict(instances=[dict(instance_id=self.iid,
            step_seconds=statistics.median(row['iteration_s'] for row in reduced),
            power_w=statistics.median(row['power_w'] for row in reduced), repeats=windows)],
            point=point, gpu_uuids=facade.raw['environment']['gpu_uuids'],
            estimator='Dynamo own SSE iteration and native-rank/NVML reduction',
            artifacts=artifacts)

    async def ready(self):
        await self.epochs.ready()

    async def measure(self, point, repeat, *, settle_s, measure_s):
        from .profile_v1 import measure_window
        await self.epochs.window_boundary(point, repeat, 'holdout' if repeat == 3 else 'training')
        # Qualification changes clocks. Restore the requested bin only after
        # the boundary has admitted this member in the newly qualified epoch.
        self.telemetry.clock(self.transport.instances[self.iid]['gpus'], point['frequency_mhz'])
        before = self.epochs.qualification_guard()
        before_at = time.time()
        value = await measure_window(self.transport, self.telemetry, self.iid, point,
            repeat=repeat, settle_s=settle_s, measure_s=measure_s, ownership=self.ownership)
        after = self.epochs.qualification_guard()
        if before != after:
            raise RuntimeError('Dynamo repeat crossed a sampling qualification epoch')
        receipt = Path(before['qualification_path'])
        if not receipt.is_relative_to(self.output):
            raise RuntimeError('Dynamo epoch receipt lies outside the owned artifact directory')
        value['sampling_epoch'] = dict(before,
            qualification_path=str(receipt.relative_to(self.output)),
            guard_before_at_s=before_at, guard_after_at_s=time.time())
        return value

    async def retire(self):
        await self.epochs.retire()
        self.retirement_permitted = True

    def released(self):
        self.epochs.released()

    def fail(self, error):
        self.epochs.fail(error)


def validate_window_qualification(window, root):
    """Recheck the exact receipt associated with a retained training window."""
    binding = window.get('sampling_epoch')
    if binding is None:
        return None
    root = Path(root).resolve()
    relative = Path(binding['qualification_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('sampling qualification reference escapes profile')
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or sha(path) != binding['qualification_sha256']:
        raise ValueError('sampling qualification checksum differs')
    evidence = json.loads(path.read_text())
    if (evidence.get('complete') is not True or evidence.get('cross_job') is not True
            or evidence.get('cohort_id') != binding['epoch_id']
            or evidence.get('measured_mode', binding['measured_mode']) != binding['measured_mode']
            or binding['measured_mode'] not in ('parallel', 'serial_cohort')
            or (binding['measured_mode'] == 'parallel' and evidence.get('passed') is not True)
            or (binding['measured_mode'] == 'serial_cohort' and evidence.get('fallback') != 'serial_cohort')
            or binding['guard_before_at_s'] > window['started_s']
            or binding['guard_after_at_s'] < window['finished_s']):
        raise ValueError('sampling qualification chronology or coordinator identity differs')
    if set(binding['layout']) != set(evidence['members']):
        raise ValueError('sampling qualification layout membership differs')
    layout_digest = hashlib.sha256(json.dumps(binding['layout'], sort_keys=True,
                                              separators=(',', ':')).encode()).hexdigest()
    if layout_digest != binding['layout_sha256']:
        raise ValueError('sampling qualification layout checksum differs')
    # Probe artifacts are owned by each peer. Our own raw probe files are
    # resolvable and must remain unchanged; other peers retain theirs in their
    # separate immutable output directories and coordinator receipt copies.
    member = evidence['member']
    index = evidence['members'].index(member)
    epoch_root = path.parent.parent
    for phase in ('isolated', 'parallel'):
        probe = evidence[phase][index]
        if sorted(probe['gpu_uuids']) != sorted(binding['layout'][member]):
            raise ValueError('sampling qualification physical UUID layout differs')
        for name, expected in probe.get('artifacts', {}).items():
            raw = (epoch_root/name).resolve()
            if not raw.is_relative_to(epoch_root) or sha(raw) != expected:
                raise ValueError('independent Dynamo qualification probe checksum differs')
    return dict(path=str(relative), sha256=binding['qualification_sha256'])
