"""Audit the measured runtime inputs independently of timing/power fit gates.

This checks historical raw measurement summaries and their immutable samples.
It does not imply a fresh runtime holdout, native cancellation/KV correctness,
or measured transition energy. Such a receipt is for development consumption.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from pdblend.profile.query.model import PerfModel
from pdblend.profile.identity import sha256_value
from pdblend.profile.calibration.optimization_profiles import digest
from pdblend.profile.calibration.power_calibration import write_immutable


def audit_runtime(base_profile, raw_path, *, out=None):
    from pdblend.profile.query.versions import _verify_samples
    base_profile, raw_path = Path(base_profile), Path(raw_path)
    model = PerfModel.load(base_profile)
    raw = json.loads(raw_path.read_text())
    identity = dict(system=model.system,model_id=Path(model.model).name,tp=model.tp,pp=model.pp)
    if any(raw.get(k) != v for k,v in identity.items()):
        raise ValueError('runtime raw/base model identity differs')
    if raw.get('identity_sha256') != sha256_value({k:v for k,v in raw.items() if k != 'identity_sha256'}):
        raise ValueError('runtime raw identity checksum differs')
    if any(not raw.get(k) for k in ('model_hash','tokenizer_hash')):
        raise ValueError('runtime raw needs model and tokenizer identities')
    if any(not raw.get('environment',{}).get(k) for k in ('hardware_id','source_hash','gpu_uuids','image_digest')):
        raise ValueError('runtime raw needs hardware, source and engine identities')
    _verify_samples(raw,raw_path.resolve().parent)
    sha = digest(raw_path)
    result = dict(kind='pdblend_runtime_components_v1',**identity,profile_sha256=digest(base_profile),
        raw_sha256=sha,model_hash=raw['model_hash'],tokenizer_hash=raw['tokenizer_hash'],
        environment=raw['environment'],components={},measurement_backed=True,
        measurement_scope='historical_raw_measurement_summaries_and_checksum_bound_samples',
        independent_runtime_holdout_passed=False,native_correctness_qualified=False,
        transition_energy_qualified=False,formal_eligible=False)
    def record(name, failures):
        result['components'][name] = dict(passed=not failures,failures=failures,
            evidence_sha256=[sha],evidence_scope='verified_raw_measurement_summary')
    record('capacity', [] if (type(model.kv_capacity_tokens) is int and model.kv_capacity_tokens > 0 and
        type(model.kv_bytes_per_token) is int and model.kv_bytes_per_token > 0 and
        model.kv_capacity_tokens == raw.get('kv_capacity_tokens') and
        model.kv_bytes_per_token == raw.get('kv_bytes_per_token')) else ['capacity_missing_or_model_differs'])
    static_failures = []
    required = {f'active_idle@{f}' for f in model.freqs} | {'active_idle_reset','parked','off'}
    for state in sorted(required):
        measured = raw.get('static',{}).get(state)
        value = model.static.get(state)
        if (not measured or value is None or not math.isfinite(value.power_w) or value.power_w <= 0 or
                not math.isfinite(value.wake_s) or value.wake_s < 0 or
                value.power_w != measured.get('power_w') or value.wake_s != measured.get('wake_s',0)):
            static_failures.append('missing_or_inconsistent_measured_state:'+state)
    record('static',static_failures)
    switches = raw.get('freq_switch_s',[])
    record('clock_transition', [] if (len(switches) >= 3 and all(isinstance(x,(int,float)) and
        math.isfinite(x) and x >= 0 for x in switches) and model.freq_switch_s == statistics.median(switches))
        else ['missing_or_inconsistent_three_clock_transition_samples'])
    points = raw.get('transfer',[])
    transfer_failures = []
    if (len(points) < 3 or len({r.get('input_tokens') for r in points}) < 3 or
            any(not isinstance(r.get('overhead_s'),(int,float)) or not math.isfinite(r['overhead_s']) or
                r['overhead_s'] <= 0 or r.get('runs',0) < 3 for r in points)):
        transfer_failures.append('missing_repeated_three_shape_transfer_measurements')
    else:
        import numpy as np
        tokens = np.array([r['input_tokens']*model.kv_bytes_per_token for r in points],float)
        seconds = np.array([r['overhead_s'] for r in points],float)
        coefficients, *_ = np.linalg.lstsq(np.stack([np.ones_like(tokens),tokens],1),seconds,rcond=None)
        fitted = (max(float(coefficients[0]),0),1/max(float(coefficients[1]),1e-12))
        if any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(model.transfer,fitted)):
            transfer_failures.append('transfer_model_differs_from_measured_fit')
    record('transfer',transfer_failures)
    result['passed'] = all(x['passed'] for x in result['components'].values())
    if out is not None:
        write_immutable(Path(out),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-profile',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit_runtime(args.base_profile,args.raw,out=args.out),indent=2))


if __name__ == '__main__':main()
