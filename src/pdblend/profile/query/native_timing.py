"""Development-only native CUDA timing on exact frequencies and measured hulls."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path

from pdblend.profile.collection.native_timing_audit import finite,need
from pdblend.profile.collection.native_timing_replay import replay_evidence
from pdblend.profile.collection.native_frequency_domain import identity_frequencies,require_same_domain


class NativeTimingOverlay:
    def __init__(self,base,replay):
        from scipy.spatial import ConvexHull
        component=replay['component']
        need(isinstance(component,dict) and component.get('component_qualified') is True and replay.get('formal_eligible') is False,
             'native timing independent replay/holdout is unqualified')
        identity=component['identity']
        need(tuple(identity[k] for k in ('system','model_id','tp','pp'))==
             (base.system,Path(base.model).name,base.tp,base.pp),'native timing base model identity differs')
        self.base=base;self.native_timing_replay=deepcopy(replay)
        self.freqs=identity_frequencies(identity);self._models={}
        require_same_domain(identity,getattr(base,'calibration_identity',getattr(base,'identity',{})))
        for row in component['models']:
            key=(row['role'],row['frequency_mhz']);coef=tuple(row['coefficients']);vertices=row['coverage_vertices']
            need(key not in self._models and len(coef)==3 and all(finite(v) and v>=0 for v in coef), 'invalid replayed timing coefficients')
            hull=tuple(tuple(x) for x in ConvexHull(vertices).equations) if row['role']=='decode' else None
            domain=(min(x[0] for x in vertices),max(x[0] for x in vertices)) if hull is None else hull
            self._models[key]=(coef,domain)
        need(set(self._models)=={(role,f) for role in ('prefill','decode') for f in self.freqs},'native timing role/frequency inventory incomplete')

    def __getattr__(self,name):return getattr(self.base,name)

    def nearest_freq(self,f):
        if not finite(f) or f not in self.freqs:raise ValueError('missing_profile: exact native timing frequency required')
        return f

    def _prefill(self,n,f):
        self.nearest_freq(f)
        coef,(lo,hi)=self._models[('prefill',f)]
        if not finite(n) or n<=0 or not lo<=n/8192.<=hi:
            raise ValueError('missing_profile: prefill outside actual native timing hull')
        return coef,n/8192.

    def prefill_seconds(self,n,f):
        (a,b,c),x=self._prefill(n,f)
        return (a+b*x+c*x*x)/1000.

    def prefill_marginal_seconds(self,n,f):
        (_,b,c),x=self._prefill(n,f)
        return (b*x+c*x*x)/1000.

    def decode_supported(self,batch,ctx,f):
        if not finite(f) or f not in self.freqs or not finite(batch) or not finite(ctx) or batch<=0 or ctx<=0:return False
        _,equations=self._models[('decode',f)]
        return all(a*batch+b*(ctx/8192.)+c<=1e-9 for a,b,c in equations)

    def step_seconds(self,batch,ctx,f):
        if not self.decode_supported(batch,ctx,f):raise ValueError('missing_profile: decode outside actual native timing hull')
        a,b,c=self._models[('decode',f)][0]
        return (a+b*batch+c*batch*ctx/8192.)/1000.

    def prefill_energy_j(self,n,f):
        return self.prefill_seconds(n,f)*self.base.prefill_power_w(n,f)

    def token_energy_j(self,batch,ctx,f):
        return self.step_seconds(batch,ctx,f)*self.base.decode_power_w(batch,f,ctx=ctx)/batch

    @property
    def query_qualification(self):
        return dict(self.base.query_qualification,native_timing_evidence=self.native_timing_replay['evidence'],
            native_timing_scope='exact_frequency_actual_hull_development_only',
            native_timing_hull_query='finite_halfspace_scan',native_timing_units='seconds',
            native_timing_formal_eligible=False)


def attach_native_timing(loaded,evidence_ref,*,path_map=()):
    """Explicit evidence path/SHA required; no registry mutation or fallback."""
    from .versions import LoadedVersion,VersionError
    if loaded.qualification.get('usage')!='development':
        raise VersionError('native timing bridge is development-only')
    from pdblend.profile.collection.native_timing_replay import Resolver
    evidence=Resolver(path_map).read(evidence_ref)
    if evidence.get('schema')=='pdblend-native-terminal-timing-stage-evidence/v1':
        from pdblend.profile.collection.native_timing_stage import replay_terminal_evidence
        replay=replay_terminal_evidence(evidence_ref,path_map=path_map)
    else:
        replay=replay_evidence(evidence_ref,path_map=path_map)
    component=replay['component']
    if not isinstance(component,dict) or component.get('component_qualified') is not True:
        raise VersionError('native timing measured domain/independent holdout remains unqualified')
    for key in ('system','model_id','tp','pp','model_hash','tokenizer_hash'):
        if loaded.identity.get(key)!=component['identity'].get(key):
            raise VersionError('native timing differs from explicit base identity: '+key)
    if loaded.identity.get('image_digest') is not None and loaded.identity['image_digest']!=component['identity']['image_digest']:
        raise VersionError('native timing image differs from explicit base')
    model=NativeTimingOverlay(loaded.model,replay)
    identity=dict(loaded.identity,native_timing_evidence_sha256=replay['evidence']['sha256'])
    key=dict(loaded.profile_key,native_timing_evidence_sha256=replay['evidence']['sha256'])
    coverage=dict(deepcopy(loaded.coverage),native_timing_actual_hulls=deepcopy(component['models']),
                  timing_query_scope='native_component_only_no_historical_fallback')
    qualification=dict(deepcopy(loaded.qualification),usage='development',formal_eligible=False,
        full_profile_qualified=False,energy_comparable=False,native_timing_replay_passed=True,
        native_timing_component_qualified=True,auxiliary_power_qualifies_power_component=False,
        query_index=model.query_qualification)
    model.profile_key=deepcopy(key);model.calibration_identity=deepcopy(identity)
    model.calibration_coverage=deepcopy(coverage);model.calibration_qualification=deepcopy(qualification)
    return LoadedVersion(model,identity,coverage,qualification,key)
