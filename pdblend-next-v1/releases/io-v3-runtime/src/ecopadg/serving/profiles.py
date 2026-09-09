"""Measured profile lookup with explicit provenance and conservative envelopes.

A point at TP=2 is never relabeled as TP=1/4/8. Querying an uncovered region
fails closed. Power is whole-instance power, residency is charged once by the
campaign integrator, not again for every queued request.
"""
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ProfilePoint:
    role: str
    tp: int
    frequency_mhz: int
    input_tokens: int
    context_tokens: int
    batch: int
    prefill_s: float
    iteration_s: float
    power_w: float
    residency_w: float
    error_fraction: float
    samples: int
    source_sha256: str
    interference_s: float = 0
    prefill_power_w: float = 0
    decode_power_w: float = 0
    energy_error_fraction: float = 0
    prefill_power_upper_w: float = 0
    prefill_duration_upper_s: float = 0

    def batch_service_s(self,output_tokens):
        # Mixed profiles measure one newly admitted prefill with an existing
        # decode batch. Filling that batch requires one prefill per request.
        prefill_work=self.phase_time_bound('prefill')*(self.batch if self.role=='mixed' else 1)
        return prefill_work+max(output_tokens-1,0)*self.phase_time_bound('decode')

    def phase_power(self,phase):
        measured=self.prefill_power_w if phase=="prefill" else self.decode_power_w
        return measured or self.power_w

    def phase_power_bound(self, phase):
        """Power uncertainty for this phase, separate from its central estimate."""
        if phase not in ('prefill', 'decode'):
            raise ValueError('unknown execution phase')
        measured = self.phase_power(phase) * (1 + self.energy_error_fraction)
        return max(measured, self.prefill_power_upper_w) if phase == 'prefill' else measured

    def phase_time_bound(self, phase):
        """Keep prefill history uncertainty out of the decode latency bound."""
        if phase == 'prefill':
            return max(self.prefill_s * self.bound, self.prefill_duration_upper_s)
        if phase == 'decode':
            return self.iteration_s * self.bound
        raise ValueError('unknown execution phase')

    @property
    def bound(self):
        return 1 + self.error_fraction


def validate_profile_observations(points):
    """Validate observations without treating an idle reference as a lower bound.

    A separately measured maximum idle power is a modeling reference. Valid
    phase observations may be lower. The online store additionally enforces
    nonnegative modeled incremental power after training-model reconciliation.
    """
    for p in points:
        if (p.role not in ('mixed', 'prefill', 'decode') or p.tp not in (1, 2, 4, 8)
                or p.samples < 1 or not p.source_sha256
                or any(not math.isfinite(v) or v < 0 for v in
                       (p.prefill_s, p.iteration_s, p.power_w, p.residency_w,
                        p.error_fraction, p.interference_s, p.prefill_power_w, p.decode_power_w,
                        p.energy_error_fraction, p.prefill_power_upper_w, p.prefill_duration_upper_s))):
            raise ValueError('invalid or unmeasured profile point')


class ProfileStore:
    def __init__(self, points, *, fingerprint="", node_residency_w=0,
                 idle_unallocated_gpu_w=0, gpu_count=8,parked_residency_w_by_tp=None,interference_points=()):
        self.points = tuple(points)
        validate_profile_observations(self.points)
        self.fingerprint = fingerprint
        self.node_residency_w = float(node_residency_w)
        self.idle_unallocated_gpu_w=float(idle_unallocated_gpu_w)
        self.gpu_count=int(gpu_count)
        self.interference_points=tuple(interference_points)
        if any(not math.isfinite(p['delay_s']) or p['delay_s']<0 or not p.get('source_sha256')
               for p in self.interference_points):
            raise ValueError('invalid measured mixed interference')
        interference_groups=defaultdict(dict)
        for p in self.interference_points:
            group=interference_groups[(p['tp'],p['frequency_mhz'])]
            key=(p['background_batch'],p['context_tokens'],p['input_tokens'])
            group[key]=max(group.get(key,0.),p['delay_s'])
        self._interference={k:tuple(sorted(v.items())) for k,v in interference_groups.items()}
        self.parked_residency_w_by_tp={int(k):float(v) for k,v in (parked_residency_w_by_tp or {}).items()}
        if any(k not in (1,2,4,8) or not math.isfinite(v) or v<0
               for k,v in self.parked_residency_w_by_tp.items()):
            raise ValueError('invalid measured parked residency')
        groups=defaultdict(list)
        for p in self.points:
            if (p.power_w < p.residency_w
                    or any(0<v<p.residency_w for v in (p.prefill_power_w,p.decode_power_w))):
                raise ValueError("invalid or unmeasured profile point")
            groups[(p.role,p.tp,p.frequency_mhz)].append(p)
        # Immutable indexes keep online admission from rescanning every TP and
        # frequency point. Sorting is equivalent to the dominating-bucket key.
        self._groups={key:tuple(sorted(points,key=lambda p:(p.batch,p.context_tokens,p.input_tokens)))
                      for key,points in groups.items()}
        self._residency={key:max(p.residency_w for p in points) for key,points in groups.items()}
        self._frequencies={pair:tuple(sorted(key[2] for key in groups if key[:2]==pair))
                           for pair in {key[:2] for key in groups}}

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if data.get("schema") != 2 or data.get("measurement") != "hardware":
            raise ValueError("requires measured schema 2 profiles")
        return cls([ProfilePoint(**p) for p in data["points"]],
                   fingerprint=hashlib.sha256(raw).hexdigest(),
                   node_residency_w=data.get("node_residency_w",0),
                   idle_unallocated_gpu_w=data.get("idle_unallocated_gpu_w",0),
                   gpu_count=data.get("gpu_count",8),
                   parked_residency_w_by_tp=data.get('parked_residency_w_by_tp',{}),
                   interference_points=data.get('interference_points',()))

    def interference(self,tp,frequency,input_tokens,context,background_batch):
        if background_batch==0: return 0.
        return next((delay for (batch,ctx,n),delay in self._interference.get((tp,frequency),())
                     if batch>=background_batch and ctx>=context and n>=input_tokens),None)

    def lookup(self, role, tp, frequency, input_tokens, context, batch):
        # The closest dominating measured bucket; no TP scaling/extrapolation.
        return next((p for p in self._groups.get((role,tp,frequency),())
                     if p.input_tokens>=input_tokens and p.context_tokens>=context and p.batch>=batch),None)

    def frequencies(self, role, tp):
        return self._frequencies.get((role,tp),())

    def residency(self, role, tp, frequency):
        return self._residency.get((role,tp,frequency))

    def parked_residency(self,tp):
        if tp in self.parked_residency_w_by_tp:
            return self.parked_residency_w_by_tp[tp]
        # Development fallback keeps resident model cost. Never assume that
        # releasing the clock lock is equivalent to an unallocated GPU.
        measured=[value for (_,degree,_),value in self._residency.items() if degree==tp]
        return max(tp*self.idle_unallocated_gpu_w,min(measured,default=0.))


class OutputPredictor:
    """Historical 90th-percentile output length by visible input bucket."""
    def __init__(self, prior=256, history_limit=512):
        self.prior = prior
        self.history = defaultdict(lambda: deque(maxlen=history_limit))

    @staticmethod
    def bucket(input_tokens):
        return max(0, int(input_tokens).bit_length() - 1)

    def predict(self, input_tokens):
        values = sorted(self.history[self.bucket(input_tokens)])
        # In fixed-work experiments max_tokens is also the output-work label.
        # Keep it out of prediction, even though the executor and KV capacity
        # checks still need that public stopping/space bound.
        return values[min(len(values)-1, math.ceil(.9*len(values))-1)] if values else self.prior

    def observe_completed(self, input_tokens, actual_output):
        if actual_output > 0:
            self.history[self.bucket(input_tokens)].append(int(actual_output))
