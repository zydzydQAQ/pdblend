"""Explicit, checksum-bound consumption of component calibration versions.

This does not select a latest version, change a planner, or promote component
evidence into formal qualification. Query composition is bounded by the frozen
candidate domains; full-campaign eligibility is a separate, mandatory check.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .model import PerfModel
from .timing_calibration import TimingOverlay


class VersionError(ValueError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _bound_file(info):
    if not isinstance(info, dict) or not info.get('path') or not info.get('sha256'):
        raise VersionError('missing checksum-bound evidence')
    path = Path(info['path']).resolve()
    if not path.is_file() or _sha(path) != info['sha256']:
        raise VersionError('calibration evidence checksum mismatch: '+str(path))
    return path


def _verify_samples(value, root, inherited_root=None, evidence_roots=None):
    if isinstance(value, dict):
        if value.get('evidence_source') is not None and ('samples_file' in value or 'repeats' in value):
            source=value['evidence_source']
            if evidence_roots is None or source not in evidence_roots:
                raise VersionError('raw row has unbound evidence source: '+str(source))
            root=evidence_roots[source]
        if 'samples_file' in value and 'samples_sha256' in value:
            path = (root/value['samples_file']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or _sha(path) != value['samples_sha256']:
                raise VersionError('raw calibration sample checksum mismatch: '+str(path))
        for key, item in value.items():
            selected = inherited_root if inherited_root is not None and key in (
                'external_interference', 'parallel_interference', 'concurrency_environment') else root
            _verify_samples(item, selected, evidence_roots=evidence_roots)
    elif isinstance(value, list):
        for item in value:
            _verify_samples(item, root, evidence_roots=evidence_roots)


class BoundedVersionModel:
    """Only calibrated timing/power queries; no unqualified transition costs."""
    def __init__(self, core, profile_key):
        self._core = core
        self.profile_key = deepcopy(profile_key)
        self.freqs = tuple(core.freqs)
        self.model, self.system, self.tp, self.pp = core.model, core.system, core.tp, core.pp
        self.bounded_coverage = deepcopy(core.bounded_coverage)

    def _frequency(self, frequency):
        if frequency not in self.freqs:
            raise VersionError('query outside exact calibrated frequencies')

    def _prefill(self, tokens, frequency):
        self._frequency(frequency)
        low, high = self.bounded_coverage['prefill_tokens']
        if not math.isfinite(tokens) or not low <= tokens <= high:
            raise VersionError('prefill query outside candidate domain')

    def decode_supported(self, batch, ctx, f):
        return (f in self.freqs and math.isfinite(batch) and math.isfinite(ctx)
                and batch > 0 and ctx > 0 and self._core.decode_supported(batch, ctx, f))

    def decode_power_supported(self, batch, ctx, f):
        return (f in self.freqs and math.isfinite(batch) and math.isfinite(ctx)
                and batch > 0 and ctx > 0 and self._core.decode_power_supported(batch, ctx, f))

    def prefill_seconds(self, n, f):
        self._prefill(n, f)
        return self._core.prefill_seconds(n, f)

    def prefill_power_w(self, n, f):
        self._prefill(n, f)
        return self._core.prefill_power_w(n, f)

    def prefill_energy_j(self, n, f):
        return self.prefill_seconds(n, f) * self.prefill_power_w(n, f)

    def prefill_marginal_seconds(self, n, f):
        self._prefill(n, f)
        return self._core.prefill_marginal_seconds(n, f)

    def step_seconds(self, batch, ctx, f):
        if not self.decode_supported(batch, ctx, f):
            raise VersionError('decode timing query outside candidate domain')
        return self._core.step_seconds(batch, ctx, f)

    def decode_power_w(self, batch, f, *, ctx=None):
        if ctx is None or not self.decode_power_supported(batch, ctx, f):
            raise VersionError('decode power query outside candidate domain; exact context required')
        return self._core.decode_power_w(batch, f, ctx=ctx)

    def token_energy_j(self, batch, ctx, f):
        # TimingOverlay.__getattr__ would otherwise return base.token_energy_j,
        # whose internal step_seconds misses the calibrated timing overlay.
        return self.step_seconds(batch, ctx, f) * self.decode_power_w(batch, f, ctx=ctx) / batch


@dataclass(frozen=True)
class LoadedVersion:
    model: BoundedVersionModel
    identity: dict
    coverage: dict
    qualification: dict
    profile_key: dict

    def manifest_fields(self):
        return deepcopy(dict(profile_key=self.profile_key, calibration_identity=self.identity,
                             calibration_coverage=self.coverage, calibration_qualification=self.qualification))


def load_version(registry, version_id, *, system, model_id, tp, pp, usage):
    """Load one explicit version for ``development`` or reject formal use.

``usage`` is required. Hash-bound registry values are preserved; this loader's
receipt records consumer availability without editing the immutable registry.
"""
    if not version_id or usage not in ('development', 'formal'):
        raise VersionError('explicit version_id and development/formal usage required')
    registry = Path(registry).resolve()
    data = _read(registry)
    if data.get('registry_kind') != 'bounded_component_calibration_versions':
        raise VersionError('unsupported calibration registry schema')
    matches = [row for row in data['versions'] if row.get('version_id') == version_id]
    if len(matches) != 1:
        raise VersionError('explicit version_id missing or ambiguous')
    row = matches[0]
    expected = dict(system=system, model_id=model_id, tp=tp, pp=pp)
    if any(row.get(k) != value for k, value in expected.items()):
        raise VersionError('calibration system/model/TP/PP identity mismatch')
    basis = {k:v for k,v in row.items() if k not in ('version_id', 'execution')}
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:20]
    expected_id = f'{model_id}-tp{tp}-pp{pp}-{digest}'
    if version_id != expected_id:
        raise VersionError('version content identity checksum mismatch')
    if any(row.get(k) is not True for k in ('sampling_complete', 'power_passed', 'effective_timing_passed', 'calibration_components_passed')):
        raise VersionError('required calibration components did not pass')
    if usage == 'formal' and (row.get('missing_gates') or row.get('formal_eligible') is not True
            or row.get('full_profile_qualified') is not True or row.get('energy_comparable') is not True):
        raise VersionError('formal calibration use blocked by missing gates: '+', '.join(row.get('missing_gates', [])))
    _bound_file(data['source_binding'])
    evidence = {key:_bound_file(value) for key,value in row['evidence'].items()}
    inputs = {key:_bound_file(value) for key,value in row['original_inputs'].items()}
    for key, path in {**inputs, **evidence}.items():
        if key in ('training_raw', 'original_raw', 'raw.json', 'timing/raw.json'):
            raw = _read(path)
            inherited_root = None
            if key=='timing/raw.json':
                if (raw.get('parent_evidence_paths_are_relative_to')!='parent_power_artifact_root'
                        or not raw.get('parent_power_artifact_root')):
                    raise VersionError('timing parent power artifact binding absent')
                # Container /output is not a host path. The bound power raw
                # identifies its actual host root; only the three explicitly
                # inherited qualification fields resolve there.
                inherited_root = evidence['raw.json'].parent
            _verify_samples(raw, path.parent, inherited_root)
    for archive in row.get('raw_evidence',[]):
        path=_bound_file(archive)
        roots={key:_bound_file(info).parent for key,info in archive.get('evidence_sources',{}).items()}
        samples_root=_bound_file(archive['samples_root_binding']).parent if archive.get('samples_root_binding') else path.parent
        raw=_read(path)
        if tuple(raw.get(k) for k in ('system','model_id','tp','pp'))!=(system,model_id,tp,pp):
            raise VersionError('raw evidence source belongs to another model/system/topology')
        _verify_samples(raw,samples_root,evidence_roots=roots)
    frozen = _read(evidence['frozen_source'])['files']
    names = ['model.py', 'decode_fit.py', 'power_table.py']
    if 'timing_candidate' in evidence:
        names.append('timing_calibration.py')
    if 'long_candidate' in evidence:
        names.append('long_context_followup.py')
    for name in names:
        if _sha(Path(__file__).with_name(name)) != frozen['pdblend/profile/'+name]:
            raise VersionError('numerical implementation differs from validated version: '+name)
    base = PerfModel.load(evidence['power_candidate'])
    if (base.system, Path(base.model).name, base.tp, base.pp) != (system, model_id, tp, pp):
        raise VersionError('power candidate model identity mismatch')
    if (base.bounded_coverage != row['bounded_coverage'] or
            {str(f):v['domain'] for f,v in base.decode_overrides.items()} != row['decode_timing_domains'] or
            {str(f):dict(kind=v['kind'], nodes=v['nodes']) for f,v in base.decode_power_overrides.items()} != row['decode_power_domains']):
        raise VersionError('candidate domain differs from registered coverage')
    if 'long_candidate' in evidence and 'timing_candidate' in evidence:
        raise VersionError('one version cannot silently compose two timing families')
    core = TimingOverlay(base, _read(evidence['timing_candidate'])) if 'timing_candidate' in evidence else base
    if 'long_candidate' in evidence:
        from .long_domain import LongDomainUnion
        long=_read(evidence['long_candidate'])
        declared=row.get('long_domain')
        actual=dict(exact_batches=long['exact_batches'],batch_interpolation_qualified=False,
            context_intervals={key:[nodes[0]['context'],nodes[-1]['context']] for key,nodes in long['nodes'].items()},
            composition='union_without_gap_interpolation')
        if declared!=actual:raise VersionError('registered long union domain differs')
        core=LongDomainUnion(base,long)
    identity = dict(expected, model_hash=row['model_hash'], tokenizer_hash=row['tokenizer_hash'], version_id=version_id)
    profile_key = dict(identity, registry_sha256=_sha(registry),
                       power_candidate_sha256=row['evidence']['power_candidate']['sha256'],
                       timing_candidate_sha256=row['evidence'].get('timing_candidate', {}).get('sha256'))
    if 'long_candidate' in evidence:profile_key['long_candidate_sha256']=row['evidence']['long_candidate']['sha256']
    qualification = dict(usage=usage, calibration_components_passed=True, consumer_loader_available=True,
        planner_automatically_wired=False, full_profile_qualified=row['full_profile_qualified'],
        formal_eligible=row['formal_eligible'], energy_comparable=row['energy_comparable'],
        missing_gates=deepcopy(row['missing_gates']), immutable_registry_unchanged=True,
        validated_timing_scope=deepcopy(row['timing']), limits=deepcopy(row['limits']),
        consumer_implementation_sha256=_sha(__file__))
    coverage = {key:deepcopy(row[key]) for key in ('bounded_coverage','decode_timing_domains','decode_power_domains')}
    if row.get('long_domain'):coverage['long_domain']=deepcopy(row['long_domain'])
    return LoadedVersion(BoundedVersionModel(core, profile_key), identity, coverage, qualification, profile_key)
