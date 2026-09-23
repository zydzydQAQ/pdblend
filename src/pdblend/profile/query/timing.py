"""Bounded timing overlay, independent of collection and fitting."""
import math
from pathlib import Path
from pdblend.profile.long_context_plan import FREQUENCIES
KIND = "bounded_batch_knee_residual_v1"

def weight(batch):
    if not math.isfinite(batch):
        raise ValueError('non-finite timing batch')
    if batch <= 16 or batch >= 64:
        return 0.
    return (batch-16)/16 if batch <= 32 else (64-batch)/32


def validate_candidate(candidate, base):
    if (candidate.get('kind') != KIND or candidate.get('system') != 'pdblend' or
            candidate.get('model_id') != 'Qwen2.5-32B-Instruct' or candidate.get('tp') != 4 or
            candidate.get('pp') != 1 or candidate.get('batch_knots') != [16, 32, 64] or
            candidate.get('context_degree') != 0 or candidate.get('formal_eligible') is not False or
            (base.system, Path(base.model).name, base.tp, base.pp) != ('pdblend', 'Qwen2.5-32B-Instruct', 4, 1)):
        raise ValueError('wrong timing overlay identity/family')
    if set(candidate['residual_seconds']) != {str(f) for f in FREQUENCIES}:
        raise ValueError('timing overlay requires all six frequencies')
    for f in FREQUENCIES:
        a = candidate['residual_seconds'][str(f)]
        if not isinstance(a, (int, float)) or not math.isfinite(a) or a < 0:
            raise ValueError('timing residual must be finite and nonnegative')
        if candidate['domains'][str(f)] != base.decode_overrides[f]['domain']:
            raise ValueError('timing overlay cannot enlarge or shrink original coverage')


class TimingOverlay:
    def __init__(self, base, candidate):
        validate_candidate(candidate, base)
        self.base, self.candidate = base, candidate

    def __getattr__(self, name):
        return getattr(self.base, name)

    def step_seconds(self, batch, ctx, f):
        if f not in FREQUENCIES or not self.base.decode_supported(batch, ctx, f):
            raise ValueError('timing overlay outside unchanged frozen coverage')
        return self.base.step_seconds(batch, ctx, f) + weight(batch)*self.candidate['residual_seconds'][str(f)]
