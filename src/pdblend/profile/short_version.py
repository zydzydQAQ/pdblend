"""Explicit short-request component versions, with mixed energy kept separate.

These versions describe serial back-to-back B1 requests and their complete
metered window, including prefill and inter-request gaps. They are not a
pure-decode power model or a full-profile qualification. Native CUDA timing
is still a required missing gate for formal consumption.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

from . import short_domain as sd, short_domain_collect as collector
from .identity import sha256_value
from .power_calibration import write_immutable

KIND='pdblend_short_complete_request_components_v1'
EXECUTION='serial_back_to_back_complete_requests'


class MissingShortProfile(ValueError):
    pass


def mixed_observation(evidence):
    """Time and energy have exactly the same measured complete-request scope."""
    summary=sd.summarize(evidence)
    if evidence['point']['role']!='short_mixed':raise ValueError('complete mixed request evidence required')
    start,end=evidence['start_s'],evidence['end_s'];samples=evidence['power']
    # NVML samples are instantaneous group values. Constant endpoints retain
    # the full declared window; no prefill or host gap is subtracted.
    nodes=[(start,sum(samples[0][1]))]+[(t,sum(values)) for t,values in samples]+[(end,sum(samples[-1][1]))]
    energy=sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(nodes,nodes[1:]))
    count=len(evidence['requests'])
    return dict(seconds=(end-start)/count,energy_j=energy/count,power_w=energy/(end-start),
        requests=count,window_s=end-start,power_scope=EXECUTION,
        original_mixed_mean_power_w=summary['mixed_power_w'])


def verify_epoch(evidence, saved_receipt):
    binding=evidence['epoch_binding'];receipt=json.loads(Path(saved_receipt).read_text())
    layout=binding.get('layout')
    if (receipt.get('cohort_id')!=binding['epoch_id'] or not isinstance(layout,dict)
            or sha256_value(layout)!=binding['layout_sha256'] or set(layout)!=set(receipt.get('members',()))):
        raise ValueError('short component epoch layout/coordinator differs')
    for index,member in enumerate(receipt['members']):
        for phase in ('isolated','parallel'):
            if sorted(receipt[phase][index]['gpu_uuids'])!=sorted(layout[member]):
                raise ValueError('short component physical group qualification differs')


def observations(raw, archive, plan, manifest):
    archive=Path(archive);values={}
    for phase in ('training','holdout'):
        expected={sd.point_key(p):p for p in plan[phase]}
        if set(raw[phase])!=set(expected):raise ValueError('complete exact short panel required')
        values[phase]={}
        for key,point in expected.items():
            row=raw[phase][key]
            if row['point']!=point or len(row['repeats'])!=point['repeats']:
                raise ValueError('short component repeat matrix differs')
            measured=[]
            for index,repeat in enumerate(row['repeats']):
                collector.verify_repeat(archive,repeat,point,index,manifest['plan_sha256'])
                evidence=json.loads((archive/repeat['samples_file']).read_text())
                verify_epoch(evidence,archive/repeat['qualification']['samples_file'])
                value=(mixed_observation(evidence) if point['role']=='short_mixed' else
                    dict(seconds=sd.summarize(evidence)['ttft_seconds']))
                measured.append(value)
            values[phase][key]=dict(point=point,repeats=measured)
    return values


def fit_direct(plan, training):
    """Only the separately designated training windows determine parameters."""
    expected={sd.point_key(p) for p in plan['training']}
    if set(training)!=expected:raise ValueError('exact short training matrix required')
    nodes={}
    for f in sd.FREQUENCIES:
        nodes[str(f)]={}
        for role in ('prefill_timing','short_mixed'):
            rows=sorted((row for row in training.values() if row['point']['freq_mhz']==f and row['point']['role']==role),
                key=lambda row:row['point']['input_tokens'])
            fields=('seconds',) if role=='prefill_timing' else ('seconds','power_w','energy_j')
            nodes[str(f)][role]=[dict(input_tokens=row['point']['input_tokens'],
                **{field:statistics.fmean(r[field] for r in row['repeats']) for field in fields}) for row in rows]
    return dict(kind=KIND,**{k:plan[k] for k in sd.IDENTITY},nodes=nodes,exact_batch=1,
        output_tokens={'prefill_timing':1,'mixed':64},execution=EXECUTION,
        timing_source='HTTP/SSE and complete measured wall-clock window',
        energy_scope='complete mixed request window including prefill and inter-request gaps',
        holdout_used=False,native_cuda_timing_qualified=False,pure_decode_power_qualified=False,
        formal_eligible=False,energy_comparable=False)


def audit_direct(candidate, plan, holdout):
    expected={sd.point_key(p):p for p in plan['holdout']};errors=[];failures=[]
    if set(holdout)!=set(expected):raise ValueError('exact independent short holdout matrix required')
    for key,point in expected.items():
        row=holdout[key]
        if row['point']!=point or len(row['repeats'])!=point['repeats']:
            raise ValueError('independent short holdout repeats missing')
        nodes=candidate['nodes'][str(point['freq_mhz'])][point['role']]
        fields=('seconds',) if point['role']=='prefill_timing' else ('seconds','power_w','energy_j')
        for repeat,observed in enumerate(row['repeats']):
            for metric in fields:
                predicted=sd.linear(nodes,point['input_tokens'],metric)
                if not math.isfinite(observed[metric]) or observed[metric]<=0:raise ValueError('nonpositive short observation')
                error=abs(predicted/observed[metric]-1)
                item=dict(point=key,repeat=repeat,metric=metric,relative_error=error)
                errors.append(item)
                if not math.isfinite(error) or error>.10:failures.append(item)
    return dict(complete=True,passed=not failures,maximum_error=max(x['relative_error'] for x in errors),
        holdout_maximum_limit=.10,errors=errors,failures=failures,formal_eligible=False,
        missing_gates=['native_cuda_timing_crosscheck','full_profile_quality_audit','formal_workload_and_energy_qualification'],
        pure_decode_power_qualified=False,original_continuous_decode_power_protocol_passed=False)


class ShortComponentModel:
    def __init__(self,candidate,version_id):
        self.candidate=deepcopy(candidate);self.version_id=version_id

    def query(self, *, role, frequency, input_tokens, output_tokens, batch,
              execution, usage):
        if usage!='experimental':
            raise MissingShortProfile('formal short component use is blocked by missing native timing/full-profile gates')
        if (execution!=EXECUTION or batch!=1 or role not in ('prefill_timing','mixed') or
                type(frequency) is not int or frequency not in sd.FREQUENCIES or
                output_tokens!=self.candidate['output_tokens'][role] or
                not isinstance(input_tokens,(int,float)) or not math.isfinite(input_tokens)):
            raise MissingShortProfile('missing_profile: role/execution/frequency/batch/output outside measured short matrix')
        family='short_mixed' if role=='mixed' else role
        nodes=self.candidate['nodes'][str(frequency)][family]
        metrics=('seconds',) if role=='prefill_timing' else ('seconds','power_w','energy_j')
        try:values={key:sd.linear(nodes,input_tokens,key) for key in metrics}
        except ValueError as exc:raise MissingShortProfile('missing_profile: input outside measured short domain') from exc
        return dict(version_id=self.version_id,role=role,execution=execution,**values,
            exact_batch=1,output_tokens=output_tokens,formal_eligible=False,
            pure_decode_power_qualified=False,original_continuous_decode_power_protocol_passed=False,
            timing_source=self.candidate['timing_source'],energy_scope=self.candidate['energy_scope'] if role=='mixed' else None)


def verified_archive(package,archive):
    package,archive=map(Path,(package,archive));manifest,plan=collector.load_package(package)
    completion=json.loads((archive/'completion.json').read_text());raw=json.loads((archive/'raw.json').read_text())
    if completion.get('complete') is not True or completion['raw_sha256']!=collector.digest(archive/'raw.json'):
        raise ValueError('completed hash-bound short measurement required')
    if (raw['binding']['package_sha256']!=collector.digest(package/'manifest.json')
            or raw['binding']['plan_sha256']!=manifest['plan_sha256']
            or raw['binding']['identity']!={k:manifest[k] for k in sd.IDENTITY}):
        raise ValueError('short measurement archive belongs to a different package')
    rebuilt=sd.fit(plan,raw['training']);rebuilt['training_rows_sha256']=sha256_value(raw['training'])
    if json.loads((archive/'candidate.json').read_text())!=rebuilt:raise ValueError('short timing candidate differs from training-only fit')
    original_audit=sd.audit(rebuilt,plan,raw['holdout'])
    if original_audit!=json.loads((archive/'audit.json').read_text()) or not original_audit['experimental_components_passed']:
        raise ValueError('original short experimental holdout gates did not pass')
    measured=observations(raw,archive,plan,manifest)
    candidate=fit_direct(plan,measured['training']);audit=audit_direct(candidate,plan,measured['holdout'])
    if not audit['passed']:raise ValueError('direct mixed independent holdout exceeds 10%')
    return manifest,candidate,audit


def publish(*,package,archive,out):
    package,archive,out=map(Path,(package,archive,out))
    manifest,candidate,audit=verified_archive(package,archive)
    bindings={name:dict(path=str(path.resolve()),sha256=collector.digest(path)) for name,path in (
        ('package_manifest',package/'manifest.json'),('raw',archive/'raw.json'),('completion',archive/'completion.json'),
        ('original_candidate',archive/'candidate.json'),('original_audit',archive/'audit.json'))}
    body=dict(kind=KIND,candidate=candidate,audit=audit,evidence=bindings,
        implementation_sha256={n:collector.digest(Path(__file__).with_name(n)) for n in ('short_version.py','short_domain.py','short_domain_collect.py')},
        profile_key={**{k:manifest[k] for k in sd.IDENTITY},'role':'mixed','scope':EXECUTION},
        formal_eligible=False,energy_comparable=False,usage='experimental')
    version=dict(body,version_id=KIND+'-'+sha256_value(body)[:20])
    if out.exists():raise FileExistsError('new immutable short version required')
    out.mkdir(parents=True);write_immutable(out/'version.json',version)
    return version


def load(path, *,system,model_id,tp,pp,usage):
    if usage!='experimental':raise MissingShortProfile('short version formal use is not qualified')
    path=Path(path);version=json.loads(path.read_text());body={k:v for k,v in version.items() if k!='version_id'}
    if version.get('kind')!=KIND or version['version_id']!=KIND+'-'+sha256_value(body)[:20]:
        raise ValueError('short version content checksum differs')
    if any(version['profile_key'][k]!=v for k,v in dict(system=system,model_id=model_id,tp=tp,pp=pp).items()):
        raise ValueError('short version system/model/topology identity differs')
    for name,sha in version['implementation_sha256'].items():
        if collector.digest(Path(__file__).with_name(name))!=sha:raise ValueError('short numerical implementation changed')
    for bound in version['evidence'].values():
        if collector.digest(bound['path'])!=bound['sha256']:raise ValueError('short version bound evidence changed')
    package=Path(version['evidence']['package_manifest']['path']).parent
    archive=Path(version['evidence']['raw']['path']).parent
    _,candidate,audit=verified_archive(package,archive)
    if candidate!=version['candidate'] or audit!=version['audit'] or not audit['passed']:
        raise ValueError('short version does not reconstruct from raw windows')
    return ShortComponentModel(candidate,version['version_id'])
