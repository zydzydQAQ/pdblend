"""PDBlend residency energy over each instance's own remaining active window."""
import math


class ResidencyHorizon:
    def __init__(self, estimator, snapshot, tails, park_grace_s):
        if not math.isfinite(park_grace_s) or park_grace_s < 0:
            raise ValueError('finite nonnegative parking grace required')
        self.estimator, self.snapshot = estimator, snapshot
        self.park_grace_s = park_grace_s
        # P work is already represented in a consumer's completion horizon,
        # but the source clock still consumes residency during that phase.
        self.prefill_windows = {}
        for instance in snapshot.instances:
            if instance.role == 'prefill':
                windows = [tails.source_ready(request)[0] for request in instance.requests]
                if any(window is None for window in windows):
                    raise ValueError('unmeasured prefill residency window')
                self.prefill_windows[instance.instance_id] = max(windows, default=0.)
        self.before_j = self.energy(tails.tails)

    def energy(self, tails, frequencies=None, source_windows=None):
        frequencies = frequencies or {}
        source_windows = dict(self.prefill_windows, **(source_windows or {}))
        horizon = max((*tails.values(), *source_windows.values()), default=0.)
        occupied = {gpu for instance in self.snapshot.instances for gpu in instance.gpus}
        profiles = self.estimator.profiles
        energy = (profiles.gpu_count - len(occupied)) * profiles.idle_unallocated_gpu_w * horizon
        for instance in self.snapshot.instances:
            name = instance.instance_id
            work = source_windows.get(name, tails[name])
            active = (min(horizon, work + self.park_grace_s)
                if work > 0 or not instance.parked or name in frequencies else 0.)
            frequency = frequencies.get(name, instance.frequency_mhz)
            resident = self.estimator.instance_residency(instance, frequency)
            parked = profiles.parked_residency(instance.tp)
            energy += resident * active + parked * (horizon - active)
        return energy
