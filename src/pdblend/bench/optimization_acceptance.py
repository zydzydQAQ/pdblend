"""Fail-closed acceptance of one controlled PDBlend optimization stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from pdblend.online.energy_routing import qualified_power_source

SCHEMA = 'pdblend.optimization-ablation/v1'
MODELS = {'Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Qwen2.5-32B-Instruct'}
DATASETS = {'alpaca', 'sharegpt', 'longbench'}
BASELINES = {'mixed', 'distserve', 'dynamollm', 'ecoserve'}
CONDITION_FIELDS = {'model_id', 'dataset', 'tp', 'pp', 'seed', 'trace_sha256', 'rate_rps',
                    'slo', 'engine_sha256', 'weights_sha256'}


def _finite(value, positive=False):
    return type(value) in (float, int) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _read(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('every evidence file requires explicit path and SHA256')
    payload = (root / ref['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref['sha256']:
        raise ValueError('evidence file digest mismatch: ' + ref['path'])
    return json.loads(payload)


def _conditions(value):
    if (not isinstance(value, dict) or set(value) != CONDITION_FIELDS or value['model_id'] not in MODELS
            or value['dataset'] not in DATASETS or type(value['tp']) is not int or value['tp'] not in (1, 2, 4)
            or type(value['pp']) is not int or value['pp'] != 1
            or type(value['seed']) is not int or value['seed'] != 701 or not _finite(value['rate_rps'], True)
            or not isinstance(value['slo'], dict) or set(value['slo']) != {'ttft_s', 'tpot_s'}
            or any(not _finite(x, True) for x in value['slo'].values())):
        raise ValueError('comparison requires registered three-model/dataset domains, PP1, seed701 and positive rate/SLO')
    for key in ('trace_sha256', 'engine_sha256', 'weights_sha256'):
        digest = value[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('comparison identity requires immutable trace/engine/weight SHA256')
    return value


def _run(root, value, side, stage):
    if not isinstance(value, dict):
        raise ValueError('missing comparison run')
    conditions = _conditions(value.get('conditions'))
    summary = _read(root, value.get('summary'))
    execution = _read(root, value.get('execution'))
    profile = _read(root, value.get('profile'))
    qualification = _read(root, value.get('qualification'))
    uncertainty = _read(root, value.get('uncertainty'))
    if (execution.get('system') != 'pdblend' or execution.get('stage') != stage
            or execution.get('arm') != side or execution.get('conditions') != conditions
            or execution.get('summary_sha256') != value['summary']['sha256']
            or execution.get('profile_sha256') != value['profile']['sha256']
            or not isinstance(execution.get('gpu_uuids'), list)
            or any(not isinstance(g, str) or not g for g in execution.get('gpu_uuids', []))
            or execution.get('exclusive') is not True or len(set(execution.get('gpu_uuids', []))) != 8
            or len(execution.get('gpu_uuids', [])) != 8
            or not _finite(execution.get('start_s')) or not _finite(execution.get('end_s'))
            or execution['end_s'] <= execution['start_s'] or execution.get('lease_verified') is not True):
        raise ValueError('stage execution requires a verified exclusive eight-GPU lease and bound conditions')
    if (profile.get('system') != 'pdblend' or profile.get('model_id') != conditions['model_id']
            or profile.get('tp') != conditions['tp'] or profile.get('pp') != 1
            or profile.get('accepted') is not True):
        raise ValueError('run requires its accepted PDBlend model/TP profile')
    if (qualification.get('system') != 'pdblend' or qualification.get('model_id') != conditions['model_id']
            or qualification.get('tp') != conditions['tp'] or qualification.get('pp') != 1
            or qualification.get('functional_passed') is not True
            or qualification.get('hardware_qualified') is not True
            or qualification.get('native_cleanup_complete') is not True):
        raise ValueError('CPU tests or incomplete native cleanup cannot qualify a GPU comparison')
    slo = summary.get('slo', {})
    joint = slo.get('joint_slo_rate')
    tokens = slo.get('joint_output_tokens')
    requests = slo.get('joint_slo_requests')
    offered = slo.get('offered')
    if (not _finite(joint) or not .9 <= joint <= 1 or type(tokens) is not int or tokens < 1
            or type(requests) is not int or requests < 1
            or type(offered) is not int or offered < requests
            or not math.isclose(joint, requests / max(offered, 1), rel_tol=1e-6)
            or any(not _finite(summary.get(k), True) for k in
                   ('energy_j', 'window_s', 'goodput_request_s', 'goodput_token_s', 'j_per_goodput_token'))
            or summary.get('quarantined_instances')
            or summary.get('metering', {}).get('error') is not None
            or not qualified_power_source(summary.get('metering', {}).get('source'))):
        raise ValueError('run lacks qualified energy/goodput metrics or misses joint SLO >= 0.9')
    if (not math.isclose(summary['j_per_goodput_token'], summary['energy_j']/tokens, rel_tol=1e-6)
            or not math.isclose(summary['goodput_token_s'], tokens/summary['window_s'], rel_tol=1e-6)
            or not math.isclose(summary['goodput_request_s'], requests/summary['window_s'], rel_tol=1e-6)):
        raise ValueError('reported energy/goodput disagrees with measured totals')
    if (uncertainty.get('summary_sha256') != value['summary']['sha256']
            or not _finite(uncertainty.get('absolute_energy_j'), True)
            or not isinstance(uncertainty.get('method'), str) or not uncertainty['method']):
        raise ValueError('comparison requires a positive measured instrumentation uncertainty bound')
    return dict(conditions=conditions, summary=summary, execution=execution,
                uncertainty_per_token=uncertainty['absolute_energy_j']/tokens)


def evaluate_stage(manifest_path):
    """Return an auditable stage-only verdict; malformed or missing evidence fails."""
    path = Path(manifest_path).resolve()
    root = path.parent
    result = dict(schema=SCHEMA, accepted=False, formal_eligible=False, stage=None, cells=[], errors=[])
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload)
        result['manifest_sha256'] = hashlib.sha256(payload).hexdigest()
        if (manifest.get('schema') != SCHEMA or not isinstance(manifest.get('stage'), str)
                or not manifest['stage'] or not isinstance(manifest.get('pairs'), list)):
            raise ValueError('explicit optimization stage and paired runs are required')
        result['stage'] = manifest['stage']
        traces = {}
        for ref in manifest.get('traces', []):
            trace = _read(root, ref)
            if (not isinstance(trace, dict) or trace.get('seed') != 701
                    or not isinstance(trace.get('requests'), list) or not trace['requests']):
                raise ValueError('frozen evaluation traces must carry seed701')
            traces[ref['sha256']] = trace
        baseline_profiles = manifest.get('independent_baselines', [])
        baseline_keys = set()
        hashes = set()
        for ref in baseline_profiles:
            profile = _read(root, ref)
            system = profile.get('system')
            key = (system, profile.get('model_id'), profile.get('tp'))
            if (system not in BASELINES or profile.get('model_id') not in MODELS or profile.get('pp') != 1
                    or profile.get('accepted') is not True or ref['sha256'] in hashes or key in baseline_keys):
                raise ValueError('independent baselines require distinct accepted per-system model/TP profiles')
            hashes.add(ref['sha256'])
            baseline_keys.add(key)
        groups = defaultdict(list)
        grids, summaries, executions = set(), set(), []
        for pair in manifest['pairs']:
            control = _run(root, pair.get('control'), 'control', manifest['stage'])
            candidate = _run(root, pair.get('candidate'), 'candidate', manifest['stage'])
            c = control['conditions']
            if c['trace_sha256'] not in traces:
                raise ValueError('comparison trace hash lacks an immutable frozen trace artifact')
            if any(run['summary']['slo']['offered'] != len(traces[c['trace_sha256']]['requests'])
                   for run in (control, candidate)):
                raise ValueError('reported offered requests differ from the frozen trace')
            if c != candidate['conditions']:
                raise ValueError('control/candidate trace, rate, SLO, seed, model, TP, engine or weights differ')
            for system in BASELINES:
                if (system, c['model_id'], c['tp']) not in baseline_keys:
                    raise ValueError('missing independent baseline profile for a tested model/TP')
            if (set(control['execution']['gpu_uuids']) != set(candidate['execution']['gpu_uuids'])
                    or not math.isclose(control['summary']['window_s'], candidate['summary']['window_s'],
                                        rel_tol=0, abs_tol=1e-6)):
                raise ValueError('paired runs require the same GPU fleet and load window')
            for side, run in [('control', control), ('candidate', candidate)]:
                digest = pair[side]['summary']['sha256']
                if digest in summaries:
                    raise ValueError('repeated comparison cannot reuse a summary artifact')
                summaries.add(digest)
                executions.append(run['execution'])
            repeat = pair.get('repeat')
            if type(repeat) is not int or repeat < 0:
                raise ValueError('every paired repetition requires a nonnegative index')
            key = json.dumps(c, sort_keys=True)
            groups[key].append((repeat, control, candidate))
            grids.add((c['model_id'], c['dataset']))
        if grids != {(model, dataset) for model in MODELS for dataset in DATASETS}:
            raise ValueError('stage acceptance requires the complete three-model by three-dataset grid')
        executions.sort(key=lambda run: run['start_s'])
        if any(a['end_s'] > b['start_s'] and set(a['gpu_uuids']) & set(b['gpu_uuids'])
               for i, a in enumerate(executions) for b in executions[i+1:]):
            raise ValueError('exclusive comparison windows overlap on a physical GPU')
        for key, pairs in groups.items():
            if len(pairs) < 3 or len({p[0] for p in pairs}) != len(pairs):
                raise ValueError('each exact condition needs at least three independent paired repetitions')
            metrics = {}
            for side, index in [('control', 1), ('candidate', 2)]:
                runs = [p[index] for p in pairs]
                energies = [r['summary']['j_per_goodput_token'] for r in runs]
                center = mean(energies)
                metrics[side] = dict(j_per_goodput_token=center,
                    goodput_request_s=mean(r['summary']['goodput_request_s'] for r in runs),
                    goodput_token_s=mean(r['summary']['goodput_token_s'] for r in runs),
                    uncertainty=max(abs(e-center) for e in energies) + max(r['uncertainty_per_token'] for r in runs))
            control, candidate = metrics['control'], metrics['candidate']
            improvement = control['j_per_goodput_token'] - candidate['j_per_goodput_token']
            bound = control['uncertainty'] + candidate['uncertainty']
            goodput_ok = all(candidate[k] >= control[k] for k in ('goodput_request_s', 'goodput_token_s'))
            passed = goodput_ok and improvement > bound
            result['cells'].append(dict(conditions=json.loads(key), repetitions=len(pairs), accepted=passed,
                metrics=metrics, goodput_not_regressed=goodput_ok, energy_improvement=improvement,
                combined_uncertainty=bound, improvement_exceeds_uncertainty=improvement > bound))
        result['accepted'] = bool(result['cells']) and all(c['accepted'] for c in result['cells'])
        result['claim'] = ('controlled stage accepted' if result['accepted'] else 'stage optimization unproven')
        result['claim_scope'] = manifest['stage']
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        result['errors'].append(str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = evaluate_stage(args.manifest)
    payload = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(payload)
    else:
        print(payload, end='')
    raise SystemExit(0 if result['accepted'] else 1)


if __name__ == '__main__':
    main()
