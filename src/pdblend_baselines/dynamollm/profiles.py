"""DynamoLLM-owned measured model; interpolation follows paper section IV-E.

Only measured mixed-instance observations are used. Interpolation requires every
surrounding corner; extrapolation and substitution from another TP are forbidden.
This deliberately does not use PDblend's dominating bucket or query index.
"""
from dataclasses import dataclass
from itertools import product
import hashlib
import json
import math
from pathlib import Path


class CoverageError(ValueError):
    pass


@dataclass(frozen=True)
class Estimate:
    prefill_s: float
    decode_s: float
    prefill_power_w: float
    decode_power_w: float
    source_sha256: tuple[str, ...]

    def duration(self, batch, output):
        return batch*self.prefill_s + max(0, output-1)*self.decode_s

    def energy(self, batch, output):
        return batch*self.prefill_s*self.prefill_power_w + max(0,output-1)*self.decode_s*self.decode_power_w


class PaperProfiles:
    def __init__(self, rows, *, fingerprint='',coordinate_system='input_context_batch'):
        self.fingerprint=fingerprint
        if coordinate_system not in ('input_context_batch','input_output_batch'):
            raise ValueError('explicit supported Dynamo profile coordinate system required')
        self.coordinate_system=coordinate_system
        self.groups={}
        for row in rows:
            if row.get('role')!='mixed':continue
            key=(row['tp'],row['frequency_mhz'])
            length=row['context_tokens']-(row['input_tokens'] if coordinate_system=='input_output_batch' else 0)
            shape=(row['input_tokens'],length,row['batch'])
            if (key[0] not in (1,2,4,8) or any(type(v) is not int or v<1 for v in (*key,*shape))
                    or row.get('samples',0)<1 or not row.get('source_sha256')):
                raise ValueError('measured Dynamo profile identity and positive geometry required')
            values=(row['prefill_s'],row['iteration_s'],
                    row.get('prefill_power_w') or row['power_w'],row.get('decode_power_w') or row['power_w'])
            if any(not math.isfinite(v) or v<=0 for v in values):
                raise ValueError('positive measured time and power required')
            previous=self.groups.setdefault(key,{}).get(shape)
            point=Estimate(*values,(row['source_sha256'],))
            if previous is not None:
                # Duplicate measurements remain conservative, deterministically.
                point=Estimate(*(max(getattr(previous,n),getattr(point,n)) for n in
                    ('prefill_s','decode_s','prefill_power_w','decode_power_w')),
                    tuple(sorted(set(previous.source_sha256+point.source_sha256))))
            self.groups[key][shape]=point
        if not self.groups:raise CoverageError('Dynamo requires measured mixed profiles')

    @classmethod
    def load(cls,path):
        raw=Path(path).read_bytes();value=json.loads(raw)
        if value.get('schema')!=2 or value.get('measurement')!='hardware':
            raise ValueError('Dynamo requires measured schema 2 observations')
        return cls(value['points'],fingerprint=hashlib.sha256(raw).hexdigest(),
            coordinate_system=value.get('coordinate_system','input_context_batch'))

    def frequencies(self,tp):
        return tuple(sorted(f for t,f in self.groups if t==tp))

    def query(self,tp,frequency,input_tokens,context_tokens,batch):
        from scipy.interpolate import interp1d
        length=context_tokens-(input_tokens if self.coordinate_system=='input_output_batch' else 0)
        targets=(input_tokens,length,batch)
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in targets):
            raise ValueError('finite positive Dynamo query required')
        rows=self.groups.get((tp,frequency))
        if not rows:raise CoverageError('unmeasured Dynamo TP/frequency')
        bounds=[]
        for axis,value in enumerate(targets):
            coordinates=sorted({shape[axis] for shape in rows})
            lower=max((x for x in coordinates if x<=value),default=None)
            upper=min((x for x in coordinates if x>=value),default=None)
            if lower is None or upper is None:raise CoverageError('Dynamo interpolation cannot extrapolate')
            bounds.append((lower,) if lower==upper else (lower,upper))
        corners=list(product(*bounds))
        if self.coordinate_system=='input_output_batch' and any(shape not in rows for shape in corners):
            # Merged hardware campaigns may have different batch grids. An
            # extra point in just one batch must not hide a complete measured
            # rectangle a little farther away. Try bounding rectangles in
            # increasing normalized width; every corner remains actual data.
            alternatives=[]
            for axis,value in enumerate(targets):
                xs=sorted({shape[axis] for shape in rows});scale=max(xs[-1]-xs[0],1)
                pairs=[]
                for lower in xs:
                    if lower>value:break
                    for upper in xs:
                        if upper<value:continue
                        pair=(lower,) if lower==upper else (lower,upper)
                        pairs.append(((upper-lower)/scale,pair))
                alternatives.append(pairs)
            candidates=sorted(product(*alternatives),key=lambda p:(sum(x[0] for x in p),tuple(x[1] for x in p)))
            for candidate in candidates:
                candidate_bounds=[x[1] for x in candidate];candidate_corners=list(product(*candidate_bounds))
                if all(shape in rows for shape in candidate_corners):
                    bounds=candidate_bounds;corners=candidate_corners;break
        if any(shape not in rows for shape in corners):
            raise CoverageError('Dynamo interpolation lacks measured surrounding corners')
        names=('prefill_s','decode_s','prefill_power_w','decode_power_w')
        def interpolate(axis,prefix,name):
            if axis==3:return getattr(rows[prefix],name)
            xs=bounds[axis];ys=[interpolate(axis+1,(*prefix,x),name) for x in xs]
            if len(xs)==1:return ys[0]
            return float(interp1d(xs,ys,kind='linear',bounds_error=True)(targets[axis]))
        return Estimate(*(interpolate(0,(),name) for name in names),
                        tuple(sorted({s for c in corners for s in rows[c].source_sha256})))

    def configurations(self,input_tokens,output_tokens,ttft_s,tpot_s,*,max_frequency=None):
        from .policy import Configuration
        result=[]
        for tp in sorted({key[0] for key in self.groups}):
            frequency=max(self.frequencies(tp)) if max_frequency is None else max_frequency
            for batch in sorted({shape[2] for shape in self.groups.get((tp,frequency),{})}):
                try:p=self.query(tp,frequency,input_tokens,input_tokens+output_tokens,batch)
                except CoverageError:continue
                if batch*p.prefill_s+p.decode_s>ttft_s or p.decode_s>tpot_s:continue
                duration=p.duration(batch,output_tokens)
                result.append(Configuration(tp,batch,batch/duration,p.energy(batch,output_tokens)/duration))
        return result
