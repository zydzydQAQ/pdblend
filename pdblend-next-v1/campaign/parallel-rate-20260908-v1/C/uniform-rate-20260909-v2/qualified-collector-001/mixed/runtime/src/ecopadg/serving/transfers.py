"""Measured transport lookup, preserving the destination clock dimension."""
from dataclasses import dataclass, replace
import hashlib
import json
import math


@dataclass(frozen=True)
class TransferCost:
    source_tp: int
    target_tp: int
    max_input_tokens: int
    seconds_upper: float
    incremental_j: float
    source_sha256: str
    validated: bool = False
    import_seconds_upper: float = 0
    profile_batch: int = 1
    source_gpus: tuple[int,...] = ()
    target_gpus: tuple[int,...] = ()
    interconnect_class: str = ''
    topology_sha256: str = ''
    # None means a legacy or derived worst-observed-clock envelope, never an
    # exact measurement of the requested frequency.
    decode_frequency_mhz: int | None = None

    def __post_init__(self):
        if (self.decode_frequency_mhz is not None and
                (type(self.decode_frequency_mhz) is not int or self.decode_frequency_mhz<=0)):
            raise ValueError('positive measured decode frequency or an explicit unlabeled envelope required')
        if any(not math.isfinite(v) or v<0 for v in
               (self.seconds_upper,self.import_seconds_upper,self.incremental_j)):
            raise ValueError('finite nonnegative measured transfer costs required')

    def matches_placement(self,source,target,topology=None):
        if (self.source_tp,self.target_tp)!=(len(source),len(target)):
            return False
        if not self.source_gpus or not self.target_gpus:
            # Legacy development tables have no placement evidence. Strict
            # topology-aware searches do not promote them to measured links.
            return topology is None
        if tuple(source)==tuple(self.source_gpus) and tuple(target)==tuple(self.target_gpus):
            return True
        return bool(topology and self.interconnect_class and
            self.topology_sha256==topology.source_sha256 and
            self.interconnect_class==topology.link_class(source,target) and
            topology.intra_class(source)==topology.intra_class(self.source_gpus) and
            topology.intra_class(target)==topology.intra_class(self.target_gpus))


def upper_envelope(costs,frequency=None):
    """Take independent maxima; energy and time need not peak at one clock."""
    costs=tuple(costs)
    if not costs: return None
    if len(costs)==1 and costs[0].decode_frequency_mhz==frequency:
        return costs[0]
    return replace(costs[0],decode_frequency_mhz=frequency,
        seconds_upper=max(c.seconds_upper for c in costs),
        import_seconds_upper=max(c.import_seconds_upper for c in costs),
        incremental_j=max(c.incremental_j for c in costs),
        source_sha256=hashlib.sha256(json.dumps(sorted({c.source_sha256 for c in costs})).encode()).hexdigest())


class TransferStore:
    def __init__(self,links,topology=None):
        self.links=tuple(links);self.topology=topology

    def lookup(self,source_tp,target_tp,source_gpus,target_gpus,input_tokens,batch,decode_frequency_mhz=None):
        """Closest measured bucket, exact clock or its conservative envelope.

        An unmeasured destination clock uses independent maxima over every
        measured clock in this bucket. No monotonicity or interpolation is
        assumed. Missing shape, TP or placement coverage still rejects a path.
        """
        links=[t for t in self.links if (t.source_tp,t.target_tp)==(source_tp,target_tp)
               and t.max_input_tokens>=input_tokens and t.profile_batch>=batch
               and t.source_sha256 and t.validated
               and t.matches_placement(source_gpus,target_gpus,self.topology)]
        if not links: return None
        bucket=min((t.max_input_tokens,t.profile_batch) for t in links)
        links=[t for t in links if (t.max_input_tokens,t.profile_batch)==bucket]
        exact=[t for t in links if decode_frequency_mhz is not None
               and t.decode_frequency_mhz==decode_frequency_mhz]
        return upper_envelope(exact,decode_frequency_mhz) if exact else upper_envelope(links)
