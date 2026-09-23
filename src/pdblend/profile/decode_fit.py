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


def fit_candidate(rows, kind, capacity):
    from scipy.optimize import lsq_linear
    b = np.array([x['batch'] for x in rows], float)
    c = np.array([x.get('effective_context_tokens', x['context_tokens']) for x in rows])
    y = np.array([x['step_seconds'] for x in rows])
    result = lsq_linear(features(b, c, kind)/y[:,None], np.ones(len(y)), bounds=(0,np.inf), tol=1e-12)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError('constrained decode fit failed')
    return dict(kind=kind, coefficients=result.x.tolist(),
                scales=dict(batch=64, context=1024, first_segment=4),
                domain=dict(batch=[float(b.min()),float(b.max())],
                            context=[float(c.min()),float(c.max())], max_batch_context=capacity*.9),
                objective='squared_relative_error', constraints='nonnegative_coefficients',
                validation_status='training_only')


def fit_split_b1(rows, capacity):
    """Freeze a six-parameter relative-error fit before independent sampling.

    Small-batch launch overhead is measured separately. The B>=4 polynomial
    retains the existing four terms, without the previous absolute-error
    objective which disproportionately weighted long high-batch steps.
    Signed coefficients are allowed only when the fitted curve is positive
    throughout the measured domain; a holdout must still qualify the model.
    """
    low = [r for r in rows if r['batch'] == 1]
    high = [r for r in rows if r['batch'] >= 4]
    if len(low) < 3 or len(high) < 4:
        raise ValueError('split decode fit needs three B1 and four higher-batch shapes')
    def solve(data, small):
        b = np.asarray([r['batch'] for r in data], float)
        c = np.asarray([r.get('effective_context_tokens', r['context_tokens']) for r in data], float)
        y = np.asarray([r['step_seconds'] for r in data], float)
        x = (np.stack([np.ones_like(c), c / 4096], 1) if small else
             np.stack([np.ones_like(b), b / 64, b*c / 65536, (b / 64)**2], 1))
        if np.any(y <= 0) or not np.all(np.isfinite(y)):
            raise ValueError('invalid training timing')
        return np.linalg.lstsq(x / y[:, None], np.ones(len(y)), rcond=None)[0].tolist()
    # The observations cover continuous decode windows, not only their
    # midpoint contexts. Keep the actual covered token interval in the domain.
    upper = max(r.get('effective_context_tokens', r['context_tokens']) for r in rows)
    for r in rows:
        for rep in r.get('repeats', []):
            upper = max(upper, rep.get('effective_context_tokens', upper) + rep.get('steps', 0) / 2)
    spec = dict(kind='split_b1_relative', coefficients=solve(high, False)+solve(low, True),
                domain=dict(batch=[1, max(r['batch'] for r in rows)],
                            context=[min(r['context_tokens'] for r in rows), upper],
                            max_batch_context=capacity*.9),
                objective='squared_relative_error', validation_status='training_only')
    for b in np.linspace(1, spec['domain']['batch'][1], 65):
        for c in np.linspace(*spec['domain']['context'], 33):
            if supported(spec, b, c) and (not np.isfinite(predict(spec, b, c)) or predict(spec, b, c) <= 0):
                raise ValueError('non-positive decode curve inside measured domain')
    return spec


def errors(observed, predicted):
    y, p = np.asarray(observed), np.asarray(predicted)
    valid = np.isfinite(p)
    e = np.abs(p[valid]/y[valid]-1)
    return dict(samples=len(y), supported=int(sum(valid)), unsupported=int(sum(~valid)),
                mape=float(e.mean()) if len(e) else None, max=float(e.max()) if len(e) else None,
                max_underestimate=float(np.maximum(1-p[valid]/y[valid],0).max()) if len(e) else None)


def compare(rows, old_model, capacity):
    from scipy.interpolate import LinearNDInterpolator
    b=np.array([r['batch'] for r in rows]); c=np.array([r['effective_context_tokens'] for r in rows])
    ctx=np.array([r['context_tokens'] for r in rows]); y=np.array([r['step_seconds'] for r in rows])
    xy=np.stack([b/64,c/1024],axis=1)
    report={}
    for kind in ('legacy', *KINDS, 'linear_2d'):
        def predictions(train, test):
            if kind == 'legacy':
                from .model import DecodePoint, PrefillPoint, fit
                ds=[DecodePoint(r['freq_mhz'],r['batch'],r['effective_context_tokens'],r['step_seconds'],r['power_w']) for r in train]
                m=fit([],ds,{})
                return [m.step_seconds(r['batch'],r['effective_context_tokens'],r['freq_mhz']) for r in test]
            if kind == 'linear_2d':
                interp=LinearNDInterpolator([[r['batch']/64,r['effective_context_tokens']/1024] for r in train], [r['step_seconds'] for r in train])
                return interp([[r['batch']/64,r['effective_context_tokens']/1024] for r in test])
            s=fit_candidate(train,kind,capacity)
            return [predict(s,r['batch'],r['effective_context_tokens']) for r in test]
        groups=[]
        for axis, values in (('batch',b),('context',ctx)):
            for v in sorted(set(values)):
                mask=values==v
                pred=predictions([r for i,r in enumerate(rows) if not mask[i]], [r for i,r in enumerate(rows) if mask[i]])
                groups.append(dict(axis=axis,value=int(v),boundary=v in (min(values),max(values)), **errors(y[mask],pred)))
        report[kind]=dict(training=errors(y,predictions(rows,rows)), group_holdouts=groups)
    return report
