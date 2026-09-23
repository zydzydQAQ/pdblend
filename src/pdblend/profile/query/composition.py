"""Join qualified timing/power to explicitly audited runtime measurements."""
from copy import deepcopy
import math
from pathlib import Path
from types import MappingProxyType


class RuntimeComposition:
    def __init__(self, calibrated, runtime, components):
        self.calibrated, self.runtime = calibrated, runtime
        self.runtime_components = MappingProxyType(dict(components))

    def __getattr__(self, name):
        return getattr(self.calibrated,name)

    def require_runtime_components(self,*names):
        from .runtime import RuntimeQualificationError
        missing = [name for name in names if self.runtime_components.get(name) is not True]
        if missing:
            raise RuntimeQualificationError('missing_profile: unqualified runtime components: '+', '.join(missing))

    @property
    def kv_capacity_tokens(self):
        self.require_runtime_components('capacity')
        return self.runtime.kv_capacity_tokens

    @property
    def kv_bytes_per_token(self):
        self.require_runtime_components('capacity')
        return self.runtime.kv_bytes_per_token

    @property
    def freq_switch_s(self):
        self.require_runtime_components('clock_transition')
        return self.runtime.freq_switch_s

    def static_power_w(self,state,f=None):
        self.require_runtime_components('static')
        return self.runtime.static_power_w(state,f)

    def wake_seconds(self,state):
        self.require_runtime_components('static')
        return self.runtime.wake_seconds(state)

    def transfer_seconds(self,tokens):
        self.require_runtime_components('transfer')
        return self.runtime.transfer_seconds(tokens)


def compose_runtime(loaded, binding, root):
    """Require distinct runtime audit; a component version cannot self-promote.

    Binding entries ``profile``, ``raw``, ``audit`` are path/sha256 pairs. The
    audit must bind each passed component to raw sample hashes. Missing checks
    stay unavailable. No historical estimates become measured transition energy.
    """
    from .versions import LoadedVersion, VersionError, _bound_file, _read, _sha, _verify_samples
    from .model import PerfModel
    if set(binding) != {'profile','raw','audit'}:
        raise VersionError('runtime_base requires profile/raw/audit checksum bindings')
    files = {name:_bound_file(dict(value,path=str(Path(root)/value['path'])))
             for name,value in binding.items()}
    runtime = PerfModel.load(files['profile'])
    raw, audit = _read(files['raw']), _read(files['audit'])
    expected = {k:loaded.identity[k] for k in ('system','model_id','tp','pp')}
    if ((runtime.system,Path(runtime.model).name,runtime.tp,runtime.pp) != tuple(expected.values()) or
            any(raw.get(k) != v or audit.get(k) != v for k,v in expected.items()) or
            tuple(runtime.freqs) != tuple(loaded.model.freqs)):
        raise VersionError('runtime_base identity/frequencies differ from calibrated version')
    if (audit.get('kind') != 'pdblend_runtime_components_v1' or
            audit.get('profile_sha256') != _sha(files['profile']) or
            audit.get('raw_sha256') != _sha(files['raw'])):
        raise VersionError('runtime component audit binding mismatch')
    _verify_samples(raw,files['raw'].parent)
    from pdblend.profile.calibration.runtime_components import audit_runtime
    if audit != audit_runtime(files['profile'],files['raw']):
        raise VersionError('runtime measurement audit cannot be reproduced')
    samples = {_sha(files['raw'])}
    def collect(value):
        if isinstance(value,dict):
            if 'samples_sha256' in value: samples.add(value['samples_sha256'])
            for item in value.values(): collect(item)
        elif isinstance(value,list):
            for item in value: collect(item)
    collect(raw)
    components = {}
    for name in ('capacity','static','transfer','clock_transition'):
        check = audit.get('components',{}).get(name,{})
        evidence = check.get('evidence_sha256',[])
        components[name] = (check.get('passed') is True and bool(evidence) and
                            set(evidence) <= samples and check.get('failures') == [])
    if components['capacity'] and (runtime.kv_capacity_tokens <= 0 or runtime.kv_bytes_per_token <= 0 or
            runtime.kv_capacity_tokens != raw.get('kv_capacity_tokens') or
            runtime.kv_bytes_per_token != raw.get('kv_bytes_per_token')):
        raise VersionError('runtime measured capacity differs from base profile')
    if components['static']:
        required = {f'active_idle@{f}' for f in runtime.freqs} | {'active_idle_reset','parked','off'}
        if not required <= set(runtime.static) or not required <= set(raw.get('static',{})):
            raise VersionError('runtime static measurements incomplete')
        for state in required:
            value, measured = runtime.static[state], raw['static'][state]
            if (not math.isfinite(value.power_w) or value.power_w <= 0 or value.wake_s < 0 or
                    value.power_w != measured.get('power_w') or value.wake_s != measured.get('wake_s',0)):
                raise VersionError('runtime static model differs from measured state')
    if components['clock_transition']:
        import statistics
        values = raw.get('freq_switch_s',[])
        if (len(values) < 3 or any(not math.isfinite(v) or v < 0 for v in values) or
                runtime.freq_switch_s != statistics.median(values)):
            raise VersionError('runtime frequency transition differs from measured repetitions')
    if components['transfer'] and (len(raw.get('transfer',[])) < 3 or
            not math.isfinite(runtime.transfer[0]) or runtime.transfer[0] < 0 or
            not math.isfinite(runtime.transfer[1]) or runtime.transfer[1] <= 0):
        raise VersionError('runtime transfer evidence incomplete')
    if loaded.qualification['usage'] == 'formal' and audit.get('formal_eligible') is not True:
        raise VersionError('runtime audit is not qualified for formal use')
    model = RuntimeComposition(loaded.model,runtime,components)
    identity = dict(loaded.identity,runtime_profile_sha256=_sha(files['profile']))
    key = dict(loaded.profile_key,runtime_profile_sha256=_sha(files['profile']),runtime_audit_sha256=_sha(files['audit']))
    qualification = dict(loaded.qualification,planner_automatically_wired=all(components.values()),
        planner_blocked_runtime_components=[k for k,v in components.items() if not v],
        runtime_components=components,runtime_audit_sha256=_sha(files['audit']),
        runtime_measurement_scope=audit['measurement_scope'],
        independent_runtime_holdout_passed=False,transition_energy_qualified=False)
    model.profile_key = deepcopy(key)
    model.calibration_identity = deepcopy(identity)
    model.calibration_coverage = deepcopy(loaded.coverage)
    model.calibration_qualification = deepcopy(qualification)
    return LoadedVersion(model,identity,deepcopy(loaded.coverage),qualification,key)
