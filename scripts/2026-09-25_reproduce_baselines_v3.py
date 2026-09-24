#!/usr/bin/env python3
"""Fresh standard baseline observations from each accepted independent source.

This preserves original algorithms, inputs and prior qualification boundaries.
GPU UUIDs and queue ownership belong to the target server. Historical profiles
are explicitly reused for development and confer no target-machine calibration.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduce_v3', Path(__file__).with_name('2026-09-25_reproduce_v3.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
read, write, bound, binding = common.read, common.write, common.bound, common.binding
SYSTEMS = ('mixed', 'distserve', 'ecoserve', 'dynamollm')


def groups_for(campaign, sizes, systems):
    from pdblend.bench.resident_session import engine_signature
    points = []
    for size in sizes:
        for system in systems:
            names = {f'{size}-{system}-{dataset}-x{scale:g}-seed701'
                     for dataset in ('alpaca', 'sharegpt', 'longbench') for scale in (.25, .5, .75, 1.)}
            selected = [p for p in campaign['points'] if p['name'] in names]
            if len(selected) != 12 or {p['name'] for p in selected} != names:
                raise ValueError('missing or ambiguous accepted standard baseline: ' + size + '/' + system)
            points.extend(deepcopy(selected))
    groups = {}
    for point in points:
        if point.get('blockers') or not point.get('engine_identity'):
            raise ValueError('baseline inventory has blocked points or missing engine identity')
        signature = engine_signature(point['engine_identity'])
        key = (point['model_id'], point['system'], signature, point['source_manifest']['sha256'])
        group = groups.setdefault(key, dict(model_id=point['model_id'], engine_identity=point['engine_identity'],
            engine_signature=signature, points=[], gpu_count=8, exclusive=True, reserve_host=True))
        group['points'].append(point)
    return list(groups.values())


def runtime_assets(campaign_path, sizes, systems):
    campaign, files = read(campaign_path), {}

    def add(ref):
        path = Path(ref['path']).absolute()
        if str(path) in files:
            if files[str(path)]['sha256'] != ref['sha256']:
                raise ValueError('conflicting asset checksums: ' + str(path))
            return read(path) if path.suffix == '.json' else None
        if common.sha(path) != ref['sha256']:
            raise ValueError('baseline asset checksum differs: ' + str(path))
        files[str(path)] = dict(path=str(path), sha256=ref['sha256'], size=path.stat().st_size)
        data = read(path) if path.suffix == '.json' else None
        if isinstance(data, dict) and isinstance(data.get('files'), dict):
            for name, checksum in data['files'].items():
                if isinstance(checksum, str) and len(checksum) == 64:
                    add(dict(path=str(path.parent / name), sha256=checksum))
        return data

    def refs(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                add(value)
            else:
                for key, item in value.items():
                    if key != 'original_input_preparation':
                        refs(item)
        elif isinstance(value, list):
            for item in value:
                refs(item)

    add(binding(campaign_path))
    execution = add(campaign['execution_inputs'])
    add(execution['model_verification'])
    for group in groups_for(campaign, sizes, systems):
        for point in group['points']:
            add(point['trace'])
            add(point['source_manifest'])
            inputs = point.get('inputs', {})
            refs(inputs)
            if point['system'] == 'dynamollm':
                config = add(inputs['system_config'])
                history = add(inputs['history'])
                summary = add(inputs['history_summary'])
                add(dict(path=history['source_path'], sha256=history['source_sha256']))
                history_dir = Path(inputs['history']['path']).parent
                for name, checksum in summary['implementation_files'].items():
                    add(dict(path=str(history_dir / 'implementation' / name), sha256=checksum))
                provenance = history['provenance']
                receipt_key = 'source_receipt' if provenance.get('source_receipt_kind') == 'existing_file_revalidation' else 'download_receipt'
                add(dict(path=provenance[receipt_key], sha256=provenance[receipt_key + '_sha256']))
                for name, checksum in provenance['official_reference_files'].items():
                    add(dict(path=name, sha256=checksum))
                for row in config.get('dynamo_transition_costs', []):
                    add(dict(path=row['audit_path'], sha256=row['audit_sha256']))
                for row in config.get('goldens', {}).values():
                    add(dict(path=row['source_path'], sha256=row['source_sha256']))
    return dict(schema='pdblend-v3-baseline-runtime-assets/v1', campaign=binding(campaign_path),
        source_root=str(ROOT), files=sorted(files.values(), key=lambda row: row['path']),
        total_bytes=sum(row['size'] for row in files.values()), includes_queue_state=False,
        includes_model_weights=False, includes_historical_audit_closure=False,
        systems=list(systems), models=list(sizes), formal_eligible=False)


def prepare(args):
    assets = runtime_assets(args.campaign, args.models, args.systems)
    groups = groups_for(read(args.campaign), args.models, args.systems)
    if args.dry_run:
        return dict(dry_run=True, writes_files=False, hardware_executed=False,
            points=sum(len(g['points']) for g in groups), resident_sessions=len(groups),
            runtime_files=len(assets['files']), runtime_bytes=assets['total_bytes'],
            queue=str(args.out.resolve() / 'queue.json'), source_queue_used=False)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('fresh baseline output required: ' + str(out))
    uuids = common.target_gpus()
    environment = common.module(ROOT / 'scripts/2026-09-25_environment.py', 'baseline_environment')
    image_check = environment.verify(args.image, environment.IDENTITY)
    actual_image = image_check['image_id']
    original = read(args.campaign)
    execution = bound(original['execution_inputs'])
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'image-verification.json', image_check)
    write(out / 'runtime-assets.json', assets)
    for index, group in enumerate(groups):
        group['engine_identity'] = common.rebind_identity(group['engine_identity'], uuids)
        group['session_id'] = out.name + '-' + str(index)
        for point in group['points']:
            point.update(engine_identity=deepcopy(group['engine_identity']), run_id=out.name,
                formal_eligible=False, evidence_valid=False, baseline_frozen=False, status='prepared')
            point['reproduction'] = dict(source_mode='historical', target_machine_calibrated=False,
                source_hardware_qualification_inherited=False, historical_profile_reused=True,
                actual_image_id=actual_image, image_verification=binding(out / 'image-verification.json'))
        source_ref = group['points'][0]['source_manifest']
        if any(p['source_manifest'] != source_ref for p in group['points']):
            raise ValueError('one baseline group cannot mix independent execution sources')
        source = Path(source_ref['path']).parent
        write(out / 'draft-groups' / (str(index) + '.json'), group)
        argv = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '2', '--memory', '6g',
            '--entrypoint', '/opt/venv/bin/python', '-v', str(ROOT) + ':' + str(ROOT) + ':ro',
            '-v', str(source) + ':/opt/pdblend-src:ro', '-v', str(out) + ':' + str(out) + ':rw',
            '-e', 'PYTHONPATH=/opt/pdblend-src', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1']
        if group['points'][0]['system'] == 'dynamollm':
            history = bound(group['points'][0]['inputs']['history'])
            parent = str(Path(history['source_path']).parent)
            argv += ['-v', parent + ':' + parent + ':ro']
            # Official source receipts/references can live outside the project.
            for item in assets['files']:
                path = Path(item['path'])
                if not path.is_relative_to(ROOT) and path.parent != Path(parent):
                    argv += ['-v', str(path) + ':' + str(path) + ':ro']
        argv += [actual_image, '-B', str(Path(__file__).resolve()), '_image-check', '--out', str(out), '--index', str(index)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        write(out / 'preflight' / (str(index) + '-execution.json'), dict(argv=argv,
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, hardware_executed=False))
        if result.returncode:
            raise RuntimeError('baseline CPU preflight failed: ' + result.stderr[-5000:])
    from pdblend.bench.comparison_jobs import resident_job
    from pdblend.bench.resident_session import engine_signature
    jobs = []
    for index, group in enumerate(groups):
        group['engine_signature'] = engine_signature(group['engine_identity'])
        path = out / 'groups' / (str(index) + '.json')
        write(path, group)
        system = group['points'][0]['system']
        source = Path(group['points'][0]['source_manifest']['path']).parent
        logical_image = group['engine_identity']['image_digest']
        job = resident_job(group, path, root=ROOT, source=source, image=logical_image,
            verification=execution['model_verification']['path'], campaign=out / 'campaign.json', priority=300-index)
        argv = job['payload']['argv']
        image_index = argv.index(logical_image)
        argv[image_index] = actual_image
        argv[:] = [str(args.models_dir.resolve()) + ':/models:ro' if value == '/home/models:/models:ro' else value for value in argv]
        if system == 'dynamollm':
            mounts = []
            for item in assets['files']:
                path = Path(item['path'])
                if not path.is_relative_to(ROOT):
                    mounts += ['-v', str(path) + ':' + str(path) + ':ro']
            argv[image_index:image_index] = mounts
        job['payload'].update(run_id=out.name, system=system, model_id=group['model_id'],
            image_digest=actual_image, after_terminal=[jobs[-1]['job_id']] if jobs else [], formal_eligible=False)
        jobs.append(job)
    execution = dict(execution, gpu_uuids=uuids, image_digest=actual_image)
    campaign = dict(schema='pdblend-v3-baseline-reproduction/v1', campaign_id=out.name,
        run_id=out.name, groups=groups, points=[p for g in groups for p in g['points']],
        execution=execution, formal_eligible=False, parent_campaign=binding(args.campaign))
    write(out / 'campaign.json', campaign)
    write(out / 'jobs.json', jobs)
    result = dict(schema='pdblend-v3-preparation/v1', campaign=binding(out / 'campaign.json'),
        jobs=binding(out / 'jobs.json'), points=len(campaign['points']), jobs_count=len(jobs),
        cpu_image_preflight_passed=True, hardware_executed=False, enqueued=False,
        formal_eligible=False, historical_profile_reused=True, target_machine_calibrated=False,
        queue=str(out / 'queue.json'))
    write(out / 'preparation.json', result)
    return result


def image_check(out, index):
    import pdblend
    if not Path(pdblend.__file__).resolve().is_relative_to('/opt/pdblend-src'):
        raise ValueError('must validate each independent frozen source')
    from pdblend.bench.resident_session import engine_signature
    group = read(out / 'draft-groups' / (str(index) + '.json'))
    group['engine_signature'] = engine_signature(group['engine_identity'])
    source_ref = group['points'][0]['source_manifest']
    for name, checksum in bound(source_ref)['files'].items():
        if common.sha(Path(source_ref['path']).parent / name) != checksum:
            raise ValueError('frozen baseline source changed: ' + name)
    for point in group['points']:
        if point['system'] in ('distserve', 'dynamollm'):
            from pdblend.bench.comparison_baseline_observation import validate_observation_inputs
            checked = validate_observation_inputs(point, group['engine_identity'])
            if point['system'] == 'dynamollm':
                from pdblend_baselines.dynamollm.validation import verified_history
                from pdblend_baselines.dynamollm.predictor import verify_checkpoint
                verified_history(checked['config']['dynamo_weekly_history'])
                verify_checkpoint(checked['config']['dynamo_predictor_dir'], expected_model=point['model_id'])
        elif point['system'] == 'ecoserve':
            target = 'pdblend.bench.comparison_ecoserve32_inputs' if '32B' in point['model_id'] else 'pdblend.bench.comparison_ecoserve_inputs'
            checker = importlib.import_module(target).validate_ecoserve_inputs
            checked = checker(point, group['engine_identity'], source_manifest=source_ref)
            if not checked['preflight_ready']:
                raise ValueError(str(checked['gate_failures']))
        else:
            from pdblend.bench.comparison_acceptance import _topology
            _topology(point, group['engine_identity'])
    write(out / 'preflight' / (str(index) + '.json'), dict(passed=True, points=len(group['points']),
        source_manifest=source_ref, hardware_executed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inventory', 'prepare'):
        command = commands.add_parser(name)
        command.add_argument('--campaign', type=Path, default=common.DEFAULT_CAMPAIGN)
        command.add_argument('--models', choices=common.SIZES, nargs='+', default=list(common.SIZES))
        command.add_argument('--systems', choices=SYSTEMS, nargs='+', default=list(SYSTEMS))
        command.add_argument('--out', type=Path, required=True)
        if name == 'prepare':
            command.add_argument('--models-dir', type=Path, default=Path('/home/models'))
            command.add_argument('--image', default='pdblend:l20-cu128-vllm-v1')
            command.add_argument('--dry-run', action='store_true')
    check = commands.add_parser('_image-check')
    check.add_argument('--out', type=Path, required=True)
    check.add_argument('--index', type=int, required=True)
    start = commands.add_parser('start')
    start.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'inventory':
        assets = runtime_assets(args.campaign, args.models, args.systems)
        write(args.out, assets)
        result = dict(inventory=str(args.out.resolve()), files=len(assets['files']), bytes=assets['total_bytes'])
    elif args.command == 'prepare':
        result = prepare(args)
    elif args.command == 'start':
        result = common.start(args.out.resolve())
    else:
        image_check(args.out.resolve(), args.index)
        result = dict(passed=True, hardware_executed=False)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
