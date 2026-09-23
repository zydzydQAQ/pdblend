"""Opt-in monotone decode-time candidates; raw measurements remain immutable."""
from __future__ import annotations
import numpy as np

KINDS = ('quadratic_relative', 'hinge64', 'hinges4_64', 'hinges4_64_ctx', 'segments4_64_80_ctx', 'segments4_64_80_96_ctx', 'segments4_64_80_96_ctxcap')


def features(batch, context, kind):
    b, c = np.broadcast_arrays(np.asarray(batch, float), np.asarray(context, float))
    if kind == 'quadratic_relative':
        values = [b * 0 + 1, b / 64, b * c / 65536, (b / 64) ** 2]
    elif kind == 'hinge64':
        values = [b * 0 + 1, b / 64, b * c / 65536, np.maximum(b - 64, 0) / 64]
    elif kind == 'hinges4_64':
        values = [b * 0 + 1, np.minimum(b, 4) / 4, np.maximum(b - 4, 0) / 64,
                  b * c / 65536, np.maximum(b - 64, 0) / 64]
    elif kind == 'hinges4_64_ctx':
        # Keep the low-batch hinge while allowing the high-batch decode slope
        # to change with effective context.  All features are non-negative,
        # so non-negative coefficients preserve monotonicity.
        values = [b * 0 + 1, np.minimum(b, 4) / 4, np.maximum(b - 4, 0) / 64,
                  b * c / 65536, np.maximum(b - 64, 0) / 64,
                  np.maximum(b - 64, 0) * c / (64 * 4096)]
    elif kind == 'segments4_64_80_ctx':
        # Independent non-negative slopes on 4-64, 64-80 and >80.  Unlike a
        # cumulative hinge this can represent a measured knee at B=80 while
        # preserving a non-decreasing step-time curve.
        values = [b * 0 + 1, np.minimum(b, 4) / 4,
                  np.minimum(np.maximum(b - 4, 0), 60) / 60,
                  np.minimum(np.maximum(b - 64, 0), 16) / 16,
                  np.maximum(b - 80, 0) / 176,
                  b * c / 65536,
                  np.maximum(b - 64, 0) * c / (64 * 4096)]
    elif kind == 'segments4_64_80_96_ctx':
        values = [b * 0 + 1, np.minimum(b, 4) / 4,
                  np.minimum(np.maximum(b - 4, 0), 60) / 60,
                  np.minimum(np.maximum(b - 64, 0), 16) / 16,
                  np.minimum(np.maximum(b - 80, 0), 16) / 16,
                  np.maximum(b - 96, 0) / 160,
                  b * c / 65536,
                  np.maximum(b - 64, 0) * c / (64 * 4096)]
    elif kind == 'segments4_64_80_96_ctxcap':
        values = [b * 0 + 1, np.minimum(b, 4) / 4,
                  np.minimum(np.maximum(b - 4, 0), 60) / 60,
                  np.minimum(np.maximum(b - 64, 0), 16) / 16,
                  np.minimum(np.maximum(b - 80, 0), 16) / 16,
                  np.maximum(b - 96, 0) / 160,
                  np.minimum(b, 80) * c / (80 * 4096),
                  np.minimum(np.maximum(b - 64, 0), 32) * c / (32 * 4096)]
    else:
        raise ValueError(f'unknown decode model: {kind}')
    return np.stack(values, axis=-1)


def predict(spec, batch, context):
    # Scalar hot path, no SciPy dependency in planner queries.
    a = spec['coefficients']; b, c = batch, context; kind = spec['kind']
    if kind == 'split_b1_relative':
        low = a[4] + a[5] * c / 4096
        high = lambda n: a[0] + a[1]*n/64 + a[2]*n*c/65536 + a[3]*(n/64)**2
        if b <= 1:
            return low
        if b < 4:
            return low + (high(4) - low) * (b - 1) / 3
        return high(b)
    if kind == 'quadratic_relative':
        return a[0] + a[1]*b/64 + a[2]*b*c/65536 + a[3]*(b/64)**2
    if kind == 'hinge64':
        return a[0] + a[1]*b/64 + a[2]*b*c/65536 + a[3]*max(b-64,0)/64
    if kind == 'hinges4_64':
        return a[0]+a[1]*min(b,4)/4+a[2]*max(b-4,0)/64+a[3]*b*c/65536+a[4]*max(b-64,0)/64
    if kind == 'hinges4_64_ctx':
        return (a[0]+a[1]*min(b,4)/4+a[2]*max(b-4,0)/64+
                a[3]*b*c/65536+a[4]*max(b-64,0)/64+
                a[5]*max(b-64,0)*c/(64*4096))
    if kind == 'segments4_64_80_ctx':
        return (a[0]+a[1]*min(b,4)/4+a[2]*min(max(b-4,0),60)/60+
                a[3]*min(max(b-64,0),16)/16+a[4]*max(b-80,0)/176+
                a[5]*b*c/65536+a[6]*max(b-64,0)*c/(64*4096))
    if kind == 'segments4_64_80_96_ctx':
        return (a[0]+a[1]*min(b,4)/4+a[2]*min(max(b-4,0),60)/60+
                a[3]*min(max(b-64,0),16)/16+a[4]*min(max(b-80,0),16)/16+
                a[5]*max(b-96,0)/160+a[6]*b*c/65536+
                a[7]*max(b-64,0)*c/(64*4096))
    if kind == 'segments4_64_80_96_ctxcap':
        return (a[0]+a[1]*min(b,4)/4+a[2]*min(max(b-4,0),60)/60+
                a[3]*min(max(b-64,0),16)/16+a[4]*min(max(b-80,0),16)/16+
                a[5]*max(b-96,0)/160+a[6]*min(b,80)*c/(80*4096)+
                a[7]*min(max(b-64,0),32)*c/(32*4096))
    raise ValueError(f'unknown decode model: {kind}')


def supported(spec, batch, context):
    d = spec['domain']
    return (d['batch'][0] <= batch <= d['batch'][1] and
            d['context'][0] <= context <= d['context'][1] and
            batch * context <= d['max_batch_context'])


def boundary(spec, batch, context):
    d = spec['domain']
    b = min(max(batch, d['batch'][0]), d['batch'][1])
    c = min(max(context, d['context'][0]), d['context'][1])
    if b*c > d['max_batch_context']:
        c = d['max_batch_context']/b
    return b, c
