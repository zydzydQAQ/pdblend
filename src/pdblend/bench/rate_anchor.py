"""Measure model-owned Mixed rate anchors before freezing evaluation traces.

Calibration and tuning corpora alone select the rate. A passing measured rate
is a lower bound, not an assertion that a sparse search found exact capacity.
All datasets reuse one fixed-TP eight-GPU fleet; no other system profile is read.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

from pdblend.engine.launcher import Fleet
from pdblend.model_registry import ModelRegistry
from pdblend_runtime.probe import NativeSpec
from pdblend_runtime.cleanup import cleanup_owned
from pdblend_baselines.resident_campaign import (verify_endpoints, warmup_endpoints,
                                                drain_endpoints, model_load_lock)
from .client import load_split, poisson_trace, SLOS
from .metering import Gpus
from .mixed_smoke import gpu_manifest
from .native_mixed import execute
from pdblend.results.power_archive import write_power_archive

START_RATES = {'alpaca': 4., 'sharegpt': 1., 'longbench': .5}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def next_rate(observations):
    """Bracket then bisect; failed requests count against capacity."""
    if not observations:
        raise ValueError('first rate must be explicit')
    good = [r['rate_rps'] for r in observations if r['metrics']['passed']]
    bad = [r['rate_rps'] for r in observations if not r['metrics']['passed']]
    if not good:
        return min(bad)/2
    lower = max(good)
    higher = [r for r in bad if r > lower]
    if higher:
        return (lower+min(higher))/2
    return lower*2


def preflight(args):
    registry = ModelRegistry(os.environ.get('PDBLEND_MODELS_DIR', '/models'),
        verification_receipt=os.environ.get('PDBLEND_MODEL_VERIFICATION_RECEIPT'))
    model = registry.get(Path(args.model).name); model.validate_config()
    expected_tp = 2 if '32B' in model.model_id else 1
    if args.tp != expected_tp or args.gpus != list(range(8)):
        raise ValueError('rate anchors require the declared fixed TP and an exclusive eight-GPU lease')
    model.validate_topology(args.tp, 1, available_gpus=8)
    if (not math.isfinite(args.window) or args.window < 60
            or not math.isfinite(args.confirm_window) or args.confirm_window < 120
            or not 2 <= args.search_points <= 10):
        raise ValueError('invalid calibration search budget')
    manifest = json.loads((args.corpus/'manifest.json').read_text())
    if manifest['model_name'] != model.model_id:
        raise ValueError('model-owned corpus identity mismatch')
    # Corpus fingerprints use relative tokenizer filenames; registry receipts
    # additionally bind mounted paths and sizes. Validate their actual bytes
    # instead of comparing those deliberately different digest formats.
    tokenizer_files = manifest['tokenizer_files_sha256']
    canonical = json.dumps(tokenizer_files, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != manifest['tokenizer_sha256']:
        raise ValueError('corpus tokenizer fingerprint corrupt')
    for name, digest in tokenizer_files.items():
        if Path(name).name != name or sha(Path(args.model)/name) != digest:
            raise ValueError('corpus tokenizer bytes differ: '+name)
    if sha(Path(args.model)/'config.json') != manifest['model_config_sha256']:
        raise ValueError('corpus model configuration differs')
    files = {}
    for dataset in SLOS:
        path = args.corpus/f'{dataset}.json'
        files[dataset] = sha(path)
        if files[dataset] != manifest['dataset_sha256'][dataset]:
            raise ValueError('corpus dataset checksum differs: '+dataset)
        for split in ('calibration', 'tuning'):
            if not load_split(args.corpus, dataset, split):
                raise ValueError('empty calibration/tuning split')
    source_manifest = Path(os.environ['PDBLEND_SOURCE_MANIFEST'])
    source = json.loads(source_manifest.read_text())
    for name, digest in source['files'].items():
        if sha(source_manifest.parent/name) != digest:
            raise ValueError('frozen source checksum differs: '+name)
    return dict(status='cpu_preflight_passed', hardware_executed=False, model_id=model.model_id,
        model_hash=model.model_hash, tokenizer_hash=model.tokenizer_hash, corpus_sha256=files,
        corpus_tokenizer_sha256=manifest['tokenizer_sha256'], tokenizer_files_sha256=tokenizer_files,
        corpus_manifest_sha256=sha(args.corpus/'manifest.json'), tp=args.tp, pp=1,
        source_sha256=os.environ['PDBLEND_SOURCE_SHA256'], image_digest=os.environ['PDBLEND_IMAGE_ID'],
        search_points=args.search_points, search_window_s=args.window,
        tuning_window_s=args.confirm_window, calibration_seed=9701, tuning_seed=9702,
        evaluation_used_for_selection=False, formal_eligible=False)


async def run(args):
    audit = preflight(args)
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out/name).exists() for name in ('completion.json', 'rate-anchor.json')):
        raise FileExistsError('rate anchor outputs already exist')
    write(args.out/'preflight.json', audit)
    specs = [NativeSpec(f'mixed-anchor-{i}', tuple(range(i*args.tp, (i+1)*args.tp)),
        args.base_port+i*4, args.model, tp=args.tp, kv_connector=None, max_num_seqs=32,
        extra_args=('--enforce-eager',)) for i in range(8//args.tp)]
    fleet, meter, sampler = None, None, None
    result = dict(audit, status='failed', complete=False, hardware_executed=True,
        scope='mixed_rate_anchor_calibration', capacity_exact=False, energy_comparable=False,
        selection_splits=['calibration', 'tuning'], anchors={}, windows=[], cleanup_errors=[])
    try:
        meter = Gpus(args.gpus, power_mode='instant')
        result['hardware'] = gpu_manifest(meter, args.gpus)
        if len(result['hardware']) != 8:
            raise RuntimeError('full-host rate calibration requires eight physical GPUs')
        for gpu in args.gpus:
            meter.unpark(gpu); meter.set_clock(gpu, 2520)
        sampler = meter.sampler(interval_s=.1); sampler.start()
        fleet = Fleet(specs, args.out/'logs')
        result['startup_started_s'] = time.time()
        with model_load_lock():
            for spec in specs:
                fleet[spec.instance_id].start()
                fleet[spec.instance_id].wait_ready(timeout_s=600)
        result['startup_finished_s'] = time.time()
        result['capabilities'] = await verify_endpoints(specs)
        for dataset in SLOS:
            observations = []
            rate = START_RATES[dataset]
            for index in range(args.search_points):
                label = f'{dataset}-calibration-{index}'
                warm = await warmup_endpoints(specs, label)
                trace = poisson_trace(load_split(args.corpus, dataset, 'calibration'),
                    rate, args.window, 9701, source=dataset)
                measured = await execute(specs, trace, args.out/label,
                    duration_s=args.window, slo=SLOS[dataset], seed=9701)
                measured.update(rate_rps=rate, split='calibration', dataset=dataset,
                    trace_sha256=sha(args.out/label/'requests.json'), warmup=warm,
                    drain=await drain_endpoints(specs))
                write(args.out/label/'completion.json', measured)
                observations.append(measured); result['windows'].append(measured)
                write(args.out/'progress.json', result)
                rate = next_rate(observations)
            passing = sorted((r['rate_rps'] for r in observations if r['metrics']['passed']), reverse=True)
            if not passing:
                raise RuntimeError(dataset+': no calibrated rate satisfies SLO')
            # Confirm on a separate split. Failed confirmations can only move
            # to a lower already-measured passing calibration rate.
            confirmation = None
            for index, rate in enumerate(passing):
                label = f'{dataset}-tuning-{index}'
                warm = await warmup_endpoints(specs, label)
                trace = poisson_trace(load_split(args.corpus, dataset, 'tuning'),
                    rate, args.confirm_window, 9702, source=dataset)
                measured = await execute(specs, trace, args.out/label,
                    duration_s=args.confirm_window, slo=SLOS[dataset], seed=9702)
                measured.update(rate_rps=rate, split='tuning', dataset=dataset,
                    trace_sha256=sha(args.out/label/'requests.json'), warmup=warm,
                    drain=await drain_endpoints(specs))
                write(args.out/label/'completion.json', measured)
                result['windows'].append(measured); write(args.out/'progress.json', result)
                if measured['metrics']['passed']:
                    confirmation = measured; break
            if confirmation is None:
                raise RuntimeError(dataset+': no independent tuning confirmation passed')
            result['anchors'][dataset] = dict(base_rate_rps=confirmation['rate_rps'],
                x05_rate_rps=.5*confirmation['rate_rps'], model_id=audit['model_id'],
                corpus_sha256=audit['corpus_sha256'][dataset],
                confirmation_path=str(args.out/label/'completion.json'),
                confirmation_sha256=sha(args.out/label/'completion.json'),
                capacity_exact=False, scope='highest_tested_and_confirmed_passing_rate',
                clock_mhz=2520, tp=args.tp, replicas=len(specs), pp=1)
            write(args.out/'rate-anchor.json', dict(audit, anchors=result['anchors']))
        result.update(complete=True, status='passed')
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if fleet is not None:
            try:
                result['final_drain'] = await drain_endpoints(specs)
            except Exception as exc:
                result['cleanup_errors'].append('final drain: '+repr(exc))
            # cleanup_owned closes only this Fleet and resets only leased GPUs.
            try:
                result['cleanup'] = cleanup_owned(fleet, meter, sampler)
                result['cleanup_errors'].extend(result['cleanup'])
            except Exception as exc:
                result['cleanup_errors'].append('cleanup: '+repr(exc))
        elif meter is not None:
            meter.reset_all()
        if sampler is not None:
            sampler.stop()
            write_power_archive(args.out/'power.json', dict(samples=sampler.samples,
                frequency_samples=sampler.frequency_samples, power_metadata=sampler.power_metadata,
                power_source=sampler.power_source, error=sampler.error))
            result['total_calibration_energy_j'] = sampler.total_energy_j()
            if sampler.error:
                result['cleanup_errors'].append('sampler: '+sampler.error)
        result['finished_s'] = time.time()
        if result['cleanup_errors']:
            result.update(complete=False, status='failed')
        result['artifact_sha256'] = {str(p.relative_to(args.out)):sha(p)
            for p in args.out.rglob('*.json') if p.name != 'completion.json'}
        write(args.out/'completion.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True); p.add_argument('--tp', type=int, required=True)
    p.add_argument('--gpus', type=lambda v:[int(x) for x in v.split(',')], required=True)
    p.add_argument('--corpus', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base-port', type=int, default=12000)
    p.add_argument('--search-points', type=int, default=5)
    p.add_argument('--window', type=float, default=60.)
    p.add_argument('--confirm-window', type=float, default=120.)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    if a.preflight_only:
        result = preflight(a); write(a.out/'preflight.json', result)
    else:
        result = asyncio.run(run(a))
    print(json.dumps({k:result.get(k) for k in ('status','complete','error','anchors')}))
    return 0 if a.preflight_only or result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
