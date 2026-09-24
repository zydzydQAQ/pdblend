"""Offline decode fitting; scalar querying lives in query.decode."""
import numpy as np
from pdblend.profile.query.decode import KINDS, features, predict, supported, boundary

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
                from pdblend.profile.query.model import DecodePoint, PrefillPoint, fit
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
