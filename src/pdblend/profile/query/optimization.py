"""Explicitly attach independently audited power components to one base profile."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from pdblend.profile.calibration import optimization_profiles as cal
from pdblend.profile.query.index import CurveIndex
from pdblend.profile.query.power_table import PowerCoverageError


class OptimizationPowerOverlay:
    """Read-only compiled component; original timing methods stay on the base."""
    def __init__(self, base, candidate, version_id, validation=None):
        self.base = base
        self.candidate = deepcopy(candidate)
        self.version_id = version_id
        self.serving_entrypoint = candidate['serving_entrypoint']
        self.power_validation = deepcopy(validation or {})
        self._curves = {}
        for name,nodes in candidate['nodes'].items():
            coordinates, values = [], []
            for node in nodes:
                for context in (node['context_min'],node['context_max']):
                    if coordinates and context == coordinates[-1]:
                        continue
                    coordinates.append(context); values.append(node['power_w'])
            self._curves[name] = CurveIndex(coordinates,values)

    def __getattr__(self, name):
        return getattr(self.base,name)

    def _power(self, family, batch, ctx, f, chunk, rate):
        if (not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (batch,ctx,f,chunk,rate)) or
                int(batch) != batch or int(f) != f or int(chunk) != chunk):
            raise PowerCoverageError('missing_profile: exact optimization shape required')
        point = dict(family=family,batch=int(batch),freq_mhz=int(f),chunk_tokens=int(chunk),
                     prefill_rate_rps=float(rate))
        curve = self._curves.get(cal.family_key(point))
        if curve is None:
            raise PowerCoverageError('missing_profile: optimization shape/rate not measured')
        try:
            return curve.predict(ctx)
        except ValueError as exc:
            raise PowerCoverageError('missing_profile: optimization context not measured') from exc

    def decode_power_supported(self, batch, ctx, f):
        if batch not in (2,3):
            return self.base.decode_power_supported(batch,ctx,f)
        try:
            self._power('decode',batch,ctx,f,0,0)
            return True
        except PowerCoverageError:
            return False

    def decode_power_w(self, batch, f, *, ctx=None):
        if batch in (2,3):
            return self._power('decode',batch,ctx,f,0,0)
        return self.base.decode_power_w(batch,f,ctx=ctx)

    def token_energy_j(self,batch,ctx,f):
        return self.base.step_seconds(batch,ctx,f)*self.decode_power_w(batch,f,ctx=ctx)/batch

    def mixed_power_supported(self,batch,ctx,f,*,chunk_tokens,prefill_rate_rps):
        try:
            self.mixed_power_w(batch,ctx,f,chunk_tokens=chunk_tokens,prefill_rate_rps=prefill_rate_rps)
            return True
        except PowerCoverageError:
            return False

    def mixed_power_w(self,batch,ctx,f,*,chunk_tokens,prefill_rate_rps):
        return self._power('mixed',batch,ctx,f,chunk_tokens,prefill_rate_rps)

    def mixed_power_residual_bound_w(self,batch,ctx,f,*,chunk_tokens,prefill_rate_rps):
        self.mixed_power_w(batch,ctx,f,chunk_tokens=chunk_tokens,prefill_rate_rps=prefill_rate_rps)
        name=cal.family_key(dict(family='mixed',batch=int(batch),freq_mhz=int(f),
                                chunk_tokens=int(chunk_tokens),prefill_rate_rps=float(prefill_rate_rps)))
        return self.power_validation['residuals'][name]['max_abs_residual_w']

    @property
    def query_qualification(self):
        old = self.base.query_qualification
        return dict(old,index_bytes=old['index_bytes']+sum(c.bytes for c in self._curves.values()),
                    optimization_component=self.version_id)


def attach_component(loaded, root):
    """Re-audit immutable samples; no unverified passed flag is sufficient."""
    from pdblend.profile.query.versions import LoadedVersion, VersionError
    root = Path(root).resolve()
    if root.is_file():
        union = json.loads(root.read_text())
        if union.get('kind') != 'pdblend_optimization_power_union_v1' or union.get('refit_performed') is not False:
            raise VersionError('unknown optimization component union')
        nodes, candidates, residuals = {}, [], {}
        for panel in union['components']:
            directory = (root.parent/panel['path']).resolve()
            if not directory.is_dir() or cal.digest(directory/'completion.json') != panel['completion_sha256']:
                raise VersionError('optimization union panel changed')
            verified = attach_component(loaded,directory)
            candidate = verified.model.candidate
            if any(candidate[k] != union[k] for k in (*cal.IDENTITY,'base_profile_sha256','serving_entrypoint')):
                raise VersionError('optimization union identity differs')
            if nodes.keys() & candidate['nodes'].keys():
                raise VersionError('optimization union has overlapping domains')
            nodes.update(deepcopy(candidate['nodes'])); candidates.append(candidate)
            residuals.update(verified.model.power_validation['residuals'])
        if len(candidates) < 2:
            raise VersionError('optimization union needs at least two independent panels')
        combined = dict(candidates[0],nodes=nodes,union_of_independent_panels=True)
        return _receipt(loaded,combined,cal.digest(root),dict(independent_holdout=True,residuals=residuals))
    completion = json.loads((root/'completion.json').read_text())
    if loaded.qualification['usage'] != 'development':
        raise VersionError('optimization components are development-only; formal gates remain missing')
    if not completion.get('complete') or not completion.get('components_passed'):
        raise VersionError('optimization component independent holdout did not pass')
    for name in ('raw','candidate','audit'):
        if cal.digest(root/(name+'.json')) != completion.get(name+'_sha256'):
            raise VersionError('optimization component checksum mismatch: '+name)
    raw = json.loads((root/'raw.json').read_text())
    candidate = json.loads((root/'candidate.json').read_text())
    audit = json.loads((root/'audit.json').read_text())
    if cal.digest(root/'package-manifest.json') != raw['binding']['package_sha256']:
        raise VersionError('optimization package manifest changed')
    base_hash = loaded.profile_key.get('power_candidate_sha256',loaded.profile_key.get('profile_sha256'))
    if candidate.get('base_profile_sha256') != base_hash:
        raise VersionError('optimization component belongs to another explicit base profile')
    if (candidate.get('kind') != cal.KIND or candidate.get('holdout_used') is not False or
            candidate.get('serving_entrypoint') != cal.SERVING_ENTRYPOINT or
            candidate.get('formal_eligible') is not False or candidate.get('batch_interpolation_qualified') is not False or
            any(candidate.get(k) != loaded.identity.get(k) for k in ('system','model_id','tp','pp'))):
        raise VersionError('optimization component identity/family mismatch')
    plan = json.loads((root/'plan.json').read_text())
    if cal.digest(root/'plan.json') != raw['binding']['plan_sha256']:
        raise VersionError('optimization component plan changed')
    observations = cal.observations(raw,root,plan)
    if (candidate != cal.fit_component(plan,observations['training'],base_hash) or
            audit != cal.audit_component(candidate,plan,observations['holdout']) or not audit['passed']):
        raise VersionError('optimization component cannot reproduce independent audit')
    return _receipt(loaded,candidate,completion['candidate_sha256'],audit)


def _receipt(loaded,candidate,version_id,validation):
    from pdblend.profile.query.versions import LoadedVersion
    model = OptimizationPowerOverlay(loaded.model,candidate,version_id,validation)
    identity = dict(loaded.identity,optimization_component_sha256=version_id)
    key = dict(loaded.profile_key,optimization_component_sha256=version_id)
    coverage = dict(loaded.coverage,optimization_power=deepcopy(candidate['nodes']))
    qualification = dict(loaded.qualification,optimization_power_independent_holdout_passed=True,
        optimization_serving_entrypoint=candidate['serving_entrypoint'],native_timing_crosscheck_passed=False,
        historical_base_timing_is_native_qualified=False,
        actual_batch_distribution_verified=True,batch_interpolation_qualified=False,
        optimization_formal_eligible=False,mixed_energy_is_pure_decode=False,query_index=model.query_qualification)
    model.profile_key = deepcopy(key)
    model.calibration_identity = deepcopy(identity)
    model.calibration_coverage = deepcopy(coverage)
    model.calibration_qualification = deepcopy(qualification)
    return LoadedVersion(model,identity,coverage,qualification,key)
