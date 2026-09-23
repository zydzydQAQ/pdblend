#!/usr/bin/env python3
"""Read-only component revalidation and immutable, bounded calibration versions.

No fit, GPU operation, queue change, or promotion into a formal campaign occurs.
Audits execute against the exact frozen implementation that collected the data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/2026-09-23'
SOURCE = RESULTS / 'power-holdout-sources/f7f443a115ccee96bdf455dbe4e4fb8b0f3e8b2efbdbd3d9e70f9c9573f4a49b'
QUEUE = ROOT / 'results/2026-09-22/three-model/queue.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def immutable(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError('immutable calibration version already exists with different evidence: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(text)


def verify_source(source):
    source = Path(source).resolve()
    manifest = read(source/'manifest.json')
    if manifest['source_sha256'] != source.name:
        raise ValueError('frozen source identity differs')
    for name, expected in manifest['files'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source) or sha(path) != expected:
            raise ValueError('frozen implementation changed: '+name)
    return manifest['source_sha256']


def require_checksum(path, expected):
    if sha(path) != expected:
        raise ValueError('evidence checksum mismatch: '+str(path))


def require_complete(receipt, scope):
    if receipt.get('complete') is not True or receipt.get('status') not in ('completed', 'passed'):
        raise ValueError(scope+' measurement is incomplete')


def _audit_entry(entry):
    """Executed in a fresh process with only the frozen source on PYTHONPATH."""
    from pdblend.profile import power_calibration as pc
    from pdblend.profile import timing_calibration as tc
    from pdblend.profile.calibration import evaluate_holdout
    from pdblend.profile.model import PerfModel

    source = Path(entry['source'])
    expected_source = verify_source(source)
    if not Path(pc.__file__).resolve().is_relative_to(source):
        raise ValueError('audit imported a different implementation')
    package, out = Path(entry['power_package']), Path(entry['power_output'])
    manifest, plan, model = pc.load_package(package)
    raw, completion = read(out/'raw.json'), read(out/'completion.json')
    require_complete(completion, 'power')
    require_checksum(out/'raw.json', completion['raw_sha256'])
    require_checksum(out/'composite-audit.json', completion['composite_audit_sha256'])
    identities = ('system', 'model_id', 'model_hash', 'tokenizer_hash', 'tp', 'pp')
    if any(raw.get(k) != manifest.get(k) for k in identities):
        raise ValueError('power raw model/topology identity mismatch')
    if (raw['environment']['source_hash'] != expected_source or completion.get('independent_holdout') is not True
            or completion.get('fit_performed') is not False or raw.get('config',{}).get('power_only_holdout') is not True):
        raise ValueError('power raw implementation/independence mismatch')
    expected = dict(candidate_sha256=manifest['candidate_sha256'], plan_sha256=manifest['plan_sha256'],
                    package_manifest_sha256=sha(package/'manifest.json'))
    if raw['power_holdout_binding'] != expected:
        raise ValueError('power package binding mismatch')
    power = pc.audit_power(raw, out, plan['points'], model, expected)
    if power != read(out/'power-only-audit.json'):
        raise ValueError('fresh power audit differs from archived receipt')
    original = Path(manifest['inputs']['original_raw']['path'])
    base = PerfModel.load(manifest['inputs']['base_candidate']['path'])
    original_plan = read(manifest['inputs']['original_manifest']['path'])['plan']
    timing = pc.timing_component(evaluate_holdout(read(original), base, original.parent, expected_plan=original_plan))
    if timing != read(out/'reused-timing-audit.json'):
        raise ValueError('reused timing audit differs from archived receipt')
    proofs = {name: binding(out/name) for name in ('raw.json', 'completion.json', 'composite-audit.json',
              'power-only-audit.json', 'reused-timing-audit.json')}
    proofs.update(power_package=binding(package/'manifest.json'), power_candidate=binding(package/'candidate.json'),
                  frozen_source=binding(source/'manifest.json'))
    timing_scope = dict(kind='original_prefill_decode_and_mixed_timing', original_timing_passed=timing['passed'],
                        timing_max=timing.get('timing_max'), mixed_timing_median=timing.get('mixed_timing_median'))
    effective_timing_passed = timing['passed']
    overlay = None
    if entry.get('timing_package'):
        tpkg, tout = Path(entry['timing_package']), out/'timing-overlay'
        tmanifest, tplan, tmodel = tc.load_package(tpkg)
        if any(tmanifest[k] != manifest[k] for k in identities):
            raise ValueError('power/timing component identity mismatch')
        if tmanifest['inputs']['base_candidate'] != manifest['inputs']['base_candidate']:
            raise ValueError('power/timing components have different base models')
        traw, tcompletion = read(tout/'raw.json'), read(tout/'completion.json')
        require_complete(tcompletion, 'timing')
        require_checksum(tout/'raw.json', tcompletion['raw_sha256'])
        require_checksum(tout/'timing-composite-audit.json', tcompletion['composite_receipt_sha256'])
        if any(traw.get(k) != manifest.get(k) for k in identities) or traw['environment'] != raw['environment']:
            raise ValueError('timing raw identity differs from resident power instance')
        tbinding = dict(candidate_sha256=tmanifest['candidate_sha256'], plan_sha256=tmanifest['plan_sha256'],
                        manifest_sha256=sha(tpkg/'manifest.json'))
        if traw['binding'] != tbinding or traw.get('independent_holdout') is not True:
            raise ValueError('timing raw binding/independence mismatch')
        fresh, retained = tc.audit_fresh(traw, tout, tplan['points'], tmodel, tbinding), tc.reuse_original(tmanifest, tmodel)
        if fresh != read(tout/'fresh-timing-audit.json') or retained != read(tout/'retained-timing-audit.json'):
            raise ValueError('timing revalidation differs from archived receipts')
        # Construct the real composite to prove the existing numerical API can consume it.
        overlay = tc.TimingOverlay(model, read(tpkg/'candidate.json'))
        for row in traw['decode']:
            for rep in row['repeats']:
                args = row['batch'], rep['effective_context_tokens'], row['freq_mhz']
                if overlay.step_seconds(*args) != tmodel.step_seconds(*args):
                    raise ValueError('composed power/timing model changed timing prediction')
        effective_timing_passed = fresh['passed'] and retained['passed']
        timing_scope.update(kind='retained_timing_plus_independent_overlay', fresh_points=len(traw['decode']),
            fresh_windows=len(fresh['points']), fresh_timing_max=fresh['timing_max'],
            fresh_batches=[24, 32, 48], fresh_nominal_context_tokens=1024,
            fresh_observed_context_bounds={str(f): [min(r['observed_context_min'] for row in traw['decode'] if row['freq_mhz']==f for r in row['repeats']),
                max(r['observed_context_max'] for row in traw['decode'] if row['freq_mhz']==f for r in row['repeats'])] for f in model.freqs},
            retained_decode_points=retained['retained_decode_points'], replaced_decode_points=retained['replaced_decode_points'],
            retained_timing_max=retained['timing_max'], fresh_long_context_qualified=False)
        proofs.update(timing_package=binding(tpkg/'manifest.json'), timing_candidate=binding(tpkg/'candidate.json'))
        proofs.update({'timing/'+name: binding(tout/name) for name in ('raw.json', 'completion.json',
                       'timing-composite-audit.json', 'fresh-timing-audit.json', 'retained-timing-audit.json')})
    candidate = read(package/'candidate.json')
    result = dict(schema=1, system='pdblend', model_id=manifest['model_id'], tp=4, pp=1,
        model_hash=manifest['model_hash'], tokenizer_hash=manifest['tokenizer_hash'],
        sampling_complete=True, power_passed=power['passed'], effective_timing_passed=effective_timing_passed,
        metadata_discrepancies=([dict(field='raw.holdout_independent',recorded=raw.get('holdout_independent'),
            interpretation='legacy Profiler default; independent purpose/immutable candidate/plan and per-window evidence revalidated',
            original_raw_unchanged=True)] if raw.get('holdout_independent') is not True else []),
        calibration_components_passed=power['passed'] and effective_timing_passed,
        original_completion_unchanged=True, original_calibration_status=read(manifest['inputs']['original_completion']['path']).get('calibration_status'),
        power=dict(fresh_points=len(raw['decode']), fresh_windows=len(power['points']),
                   max_error=max(x['max_error'] for x in power['by_frequency'].values()), by_frequency=power['by_frequency']),
        timing=timing_scope, environment=raw['environment'], evidence=proofs,
        original_inputs=manifest['inputs'], bounded_coverage=candidate['bounded_coverage'],
        decode_timing_domains={f: spec.get('domain') for f, spec in candidate['decode_overrides'].items()},
        decode_power_domains={f: dict(kind=spec['kind'], nodes=spec['nodes']) for f, spec in candidate['decode_power_overrides'].items()},
        consumer=dict(power_api='PerfModel.load(power_candidate)', timing_api='TimingOverlay(power_model, timing_candidate)' if overlay else 'PerfModel.step_seconds',
            numerical_composition_verified=True, planner_registry_wired=False,
            required_adapter='checksum-bound version loader with strict domain checks and action/profile-key propagation',
            eligibility='component_validation_only'),
        full_profile_qualified=False, formal_eligible=False, energy_comparable=False, seeds=[701], single_seed=True,
        missing_gates=['full_profile_provenance_revalidation', 'mixed_power_if_required', 'transfer_protocol_revalidation',
                       'native_system_mechanisms', 'campaign_acceptance', 'planner_version_loader'],
        limits=['No extrapolation or frequency interpolation.', 'Independent baseline profiles cannot consume this PDBlend version.',
                'Fresh timing overlay checks do not qualify unmeasured long-context or batch regimes.'])
    version_basis = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    result['version_id'] = manifest['model_id']+'-tp4-pp1-'+hashlib.sha256(version_basis).hexdigest()[:20]
    return result


def revalidate(entry):
    verify_source(entry['source'])
    env = dict(os.environ, PYTHONPATH=entry['source'], PYTHONDONTWRITEBYTECODE='1')
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--audit-entry'],
                             input=json.dumps(entry), text=True, capture_output=True, env=env, check=False)
    if process.returncode:
        raise ValueError('frozen calibration revalidation failed: '+process.stderr[-6000:])
    return json.loads(process.stdout)


def build_registry(queue, source=SOURCE):
    rows = []
    for short in ('7b', '32b'):
        jid = 'power-holdout-'+short+'-tp4-power-4c21d9296a11efb78dcb'
        leases = [x for x in queue['leases'].values() if x['job_id']==jid]
        if queue['jobs'][jid]['status'] != 'succeeded' or not leases:
            raise ValueError('required completed power job absent: '+jid)
        lease = max(leases, key=lambda x:x['claimed_at'])
        entry = dict(source=str(source), power_package=str(RESULTS/'power-pair-resident-v2'/(short+'-package')),
                     power_output=lease['attempt_dir'])
        if short=='32b':
            entry['timing_package'] = str(RESULTS/'32b-tp4-timing-overlay-package-final')
        result = revalidate(entry)
        if sorted(result['environment']['gpu_uuids']) != sorted(lease['gpu_uuids']):
            raise ValueError('power receipt GPU UUIDs differ from lease')
        result['execution'] = dict(job_id=jid, lease_id=lease.get('lease_id'), gpu_uuids=lease['gpu_uuids'])
        rows.append(result)
    return dict(schema=1, registry_kind='bounded_component_calibration_versions', versions=rows,
                source_binding=binding(Path(__file__)), seeds=[701], formal_eligible=False, energy_comparable=False)


def report(registry):
    lines = ['# 校准组件版本（seed 701）', '', '旧失败记录保持原样；以下是新证据版本，不是五系统正式排名。', '',
             '|模型|新功率最大误差|有效 timing|版本|', '|---|---:|---|---|']
    for row in registry['versions']:
        timing = row['timing']
        detail = ('原 timing 保留，max %.2f%%' % (100*timing['timing_max']) if row['model_id'].startswith('Qwen2.5-7B') else
                  '42 点保留 + 18 点新验证，新 max %.2f%%' % (100*timing['fresh_timing_max']))
        lines.append('|%s|%.2f%%|%s|`%s`|' % (row['model_id'], 100*row['power']['max_error'], detail, row['version_id']))
    lines += ['', '32B 新 timing 检验限定在 B24/32/48、名义 context 1024；没有据此扩大长输入资格。',
              '候选原有 bounded coverage 和实测窗口范围均保存在 registry；不允许外推或跨系统借用。',
              '数值组合 API 已验证，但 planner 仍需显式版本加载器。完整 provenance、mixed power、transfer、自动机制和正式矩阵门槛仍独立保留。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--out', type=Path, default=RESULTS/'calibration-versions-v1')
    parser.add_argument('--audit-entry', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.audit_entry:
        print(json.dumps(_audit_entry(json.load(sys.stdin)), allow_nan=False)); return
    registry = build_registry(read(args.queue))
    immutable(args.out/'registry.json', registry)
    text = report(registry)
    path = args.out/'report.md'
    if path.exists() and path.read_text()!=text:
        raise ValueError('immutable report changed')
    path.write_text(text)
    print(json.dumps(dict(registry=str(args.out/'registry.json'), versions=[r['version_id'] for r in registry['versions']])))


if __name__=='__main__':
    main()
