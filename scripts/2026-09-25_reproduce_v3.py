#!/usr/bin/env python3
"""Prepare fresh three-model PDblend observations; never resume a source lease.

The default snapshots the checkout's current implementation and replays the
36 standard workloads using explicitly historical development profiles. The
historical mode instead uses the exact accepted implementation/configuration.
Neither mode transfers calibration or formal qualification to another server.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / 'results/2026-09-24/profile-saturation-repaired-v1/campaign.json'
SIZES = ('7b', '14b', '32b')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def bound(ref):
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('input checksum differs: ' + ref['path'])
    return read(ref['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def selected_groups(campaign, sizes=SIZES):
    """Use the accepted active groups, excluding historical points and branches."""
    result = []
    for size in sizes:
        model = 'Qwen2.5-' + size.upper() + '-Instruct'
        candidates = [g for g in campaign['groups'] if g['model_id'] == model
                      and {p['system'] for p in g['points']} == {'pdblend'}]
        if len(candidates) != 1:
            raise ValueError('need one accepted PDblend group per model: ' + model)
        group = candidates[0]
        expected = {f'{size}-pdblend-{dataset}-x{scale:g}-seed701'
                    for dataset in ('alpaca', 'sharegpt', 'longbench') for scale in (.25, .5, .75, 1.)}
        if {p['name'] for p in group['points']} != expected or len(group['points']) != 12:
            raise ValueError('accepted group is not the twelve standard seed701 points')
        if any(p['duration_s'] != 150 or p['seed'] != 701 for p in group['points']):
            raise ValueError('standard trace duration/seed differs')
        result.append(deepcopy(group))
    return result


def runtime_assets(campaign_path, sizes=SIZES):
    """Inventory files the current PD observation loader actually consumes.

    Preserve profile source bindings byte-for-byte. Their complete historical
    provenance is a separate optional archive, not a new-machine calibration.
    """
    campaign = read(campaign_path)
    files = {}

    def add(ref):
        path = Path(ref['path']).resolve()
        if not path.is_file() or sha(path) != ref['sha256']:
            raise ValueError('missing or changed runtime asset: ' + str(path))
        files[str(path)] = dict(path=str(path), sha256=ref['sha256'], size=path.stat().st_size)
        return read(path) if path.suffix == '.json' else None

    add(binding(campaign_path))
    execution = add(campaign['execution_inputs'])
    add(execution['model_verification'])
    for group in selected_groups(campaign, sizes):
        for point in group['points']:
            inputs = point['inputs']
            for key in ('trace', 'planning_trace', 'offline_choice', 'system_config'):
                data = add(inputs[key])
                for name in ('startup_helper', 'runtime_options_source'):
                    if isinstance(data.get(name), dict):
                        add(data[name])
            for ref in inputs['profiles']:
                profile = add(ref)
                if profile.get('kind') != 'pdblend_development_composite_profile_v1':
                    raise ValueError('this restart inventory requires the accepted development composite profile')
                add(profile['base_profile'])
                compiled = add(profile['compiled'])
                for source in compiled['source_bindings']:
                    add(source)
            source_ref = inputs['source_manifest']
            source = add(source_ref)
            for name, checksum in source['files'].items():
                add(dict(path=str(Path(source_ref['path']).parent / name), sha256=checksum))
    return dict(schema='pdblend-v3-runtime-assets/v1', campaign=binding(campaign_path),
                source_root=str(ROOT), files=sorted(files.values(), key=lambda x: x['path']),
                total_bytes=sum(x['size'] for x in files.values()),
                models=[str(Path('/home/models') / ('Qwen2.5-' + s.upper() + '-Instruct')) for s in sizes],
                includes_model_weights=False, includes_historical_audit_closure=False,
                includes_queue_state=False, formal_eligible=False)


def target_gpus():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=15)
    rows = [line.split(',') for line in result.stdout.splitlines() if line.strip()]
    rows = sorted(rows, key=lambda row: int(row[0]))
    if len(rows) != 8 or any(row[2].strip() not in ('NVIDIA L20', 'L20') for row in rows):
        raise ValueError('reproduction requires exactly eight NVIDIA L20 GPUs')
    return [row[1].strip() for row in rows]


def rebind_identity(identity, uuids):
    if len(uuids) != 8 or len(set(uuids)) != 8:
        raise ValueError('eight distinct target GPU UUIDs required')
    result = deepcopy(identity)
    previous = identity['fleet_gpu_uuids']
    if len(previous) != 8 or len(set(previous)) != 8:
        raise ValueError('source fleet identity is invalid')
    mapping = dict(zip(previous, uuids))
    result['fleet_gpu_uuids'] = list(uuids)
    for row in result['instances']:
        row['gpu_uuids'] = [mapping[uuid] for uuid in row['gpu_uuids']]
    return result


def prepare(args):
    assets = runtime_assets(args.campaign, args.models)
    if args.dry_run:
        return dict(dry_run=True, writes_files=False, hardware_executed=False,
                    points=12 * len(args.models), runtime_files=len(assets['files']),
                    runtime_bytes=assets['total_bytes'], source_mode=args.source_mode,
                    output=str(args.out.resolve()), queue=str(args.out.resolve() / 'queue.json'),
                    formal_eligible=False, source_queue_used=False)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('fresh output directory required: ' + str(out))
    campaign = read(args.campaign)
    execution = bound(campaign['execution_inputs'])
    uuids = target_gpus()
    environment = module(ROOT / 'scripts/2026-09-25_environment.py', 'v3_environment')
    image_check = environment.verify(args.image, environment.IDENTITY)
    image_id = image_check['image_id']
    execution = dict(execution, image_digest=image_id)
    for size in args.models:
        if not (args.models_dir / ('Qwen2.5-' + size.upper() + '-Instruct') / 'config.json').is_file():
            raise ValueError('model directory missing; restore and fully verify weights first')
    groups = selected_groups(campaign, args.models)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'image-verification.json', image_check)
    write(out / 'runtime-assets.json', assets)
    if args.source_mode == 'workspace':
        freezer = module(ROOT / 'scripts/2026-09-22_enqueue_parallel_profiles.py', 'v3_source_freezer')
        source, _ = freezer.freeze_source(ROOT / 'src', out / 'sources')
    else:
        refs = {p['inputs']['source_manifest']['path'] for g in groups for p in g['points']}
        if len(refs) != 1:
            raise ValueError('historical groups must use one explicit accepted source')
        source = Path(next(iter(refs))).parent
    source_ref = binding(source / 'manifest.json')
    source_data = bound(source_ref)
    from_source = source_data['source_sha256']
    for group in groups:
        identity = rebind_identity(group['engine_identity'], uuids)
        identity['image_digest'] = image_id
        file_groups = dict(runtime_source_sha256={k: v for k, v in source_data['files'].items()
            if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))},
            measurement_source_sha256={k: v for k, v in source_data['files'].items()
            if k.startswith('pdblend/measure/') or k in ('pdblend/bench/comparison_metrics.py',
                'pdblend/bench/comparison_metering.py', 'pdblend/bench/client.py')})
        for name, files in file_groups.items():
            identity[name] = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        group['engine_identity'] = identity
        for point in group['points']:
            original_revision = point['revision']
            point.update(engine_identity=deepcopy(identity), source_manifest=source_ref,
                         revision=from_source, run_id=out.name, status='prepared',
                         formal_eligible=False, profile_qualified=False, baseline_frozen=False,
                         evidence_valid=False)
            point['inputs']['source_manifest'] = source_ref
            point.pop('measurement_compatibility', None)
            point['optimization_version']['source_manifest'] = source_ref
            point['reproduction'] = dict(source_mode=args.source_mode, original_revision=original_revision,
                historical_profile_reused=True, target_machine_calibrated=False,
                source_hardware_qualification_inherited=False, formal_eligible=False)
    draft = dict(schema='pdblend-v3-fresh-reproduction/v1', campaign_id=out.name, run_id=out.name,
        parent_campaign=binding(args.campaign), groups=groups, source_manifest=source_ref,
        execution=dict(execution, source=str(source), source_sha256=from_source, gpu_uuids=uuids),
        source_mode=args.source_mode, models_dir=str(args.models_dir.resolve()), formal_eligible=False)
    write(out / 'draft.json', draft)
    argv = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '2', '--memory', '6g',
            '--entrypoint', '/opt/venv/bin/python', '-v', str(ROOT) + ':' + str(ROOT) + ':ro',
            '-v', str(source) + ':/opt/pdblend-src:ro', '-v', str(out) + ':' + str(out) + ':rw',
            '-e', 'PYTHONPATH=/opt/pdblend-src', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', image_id,
            '-B', str(Path(__file__).resolve()), '_image-prepare', '--out', str(out)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    write(out / 'container-preflight.json', dict(argv=argv, returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, hardware_executed=False))
    if result.returncode:
        raise RuntimeError('CPU image preflight failed; no queue created: ' + result.stderr[-5000:])
    return read(out / 'preparation.json')


def image_prepare(out):
    """CPU-only validation inside the exact execution image and frozen source."""
    from types import SimpleNamespace
    import pdblend
    from pdblend.bench.comparison_jobs import resident_job
    from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
    from pdblend.bench.comparison_runtime import pdblend_window_resources
    from pdblend.bench.independent_dispatch import request_rows
    from pdblend.bench.pdblend_runtime_options import comparison_options
    from pdblend.bench.resident_session import engine_signature
    if not Path(pdblend.__file__).resolve().is_relative_to('/opt/pdblend-src'):
        raise ValueError('CPU validation must use the frozen execution source')
    draft = read(out / 'draft.json')
    source_ref = draft['source_manifest']
    source = bound(source_ref)
    for name, checksum in source['files'].items():
        if sha(Path(source_ref['path']).parent / name) != checksum:
            raise ValueError('frozen source changed: ' + name)
    groups, jobs, checks = [], [], []
    for index, group in enumerate(draft['groups']):
        group['session_id'] = out.name + '-' + group['model_id'].split('-')[1].lower()
        group['engine_signature'] = engine_signature(group['engine_identity'])
        # A new output has no continuation, resume manifest, or old windows.
        group = {key: group[key] for key in ('session_id', 'model_id', 'engine_identity',
                 'engine_signature', 'points', 'gpu_count', 'exclusive', 'reserve_host')}
        specs = [SimpleNamespace(tp=row['tp'], pp=row['pp'], generation=0)
                 for row in group['engine_identity']['instances']]
        for point in group['points']:
            config = bound(point['inputs']['system_config'])
            choice = bound(point['inputs']['offline_choice'])
            helper_ref = config['startup_helper']
            if sha(helper_ref['path']) != helper_ref['sha256']:
                raise ValueError('startup helper checksum differs')
            helper = module(helper_ref['path'], 'pdblend.bench.comparison_startup')
            loaded, plan = pdblend_window_resources(point, specs)
            options = comparison_options(config, point['inputs']['system_config']['path'], point=point)['values']
            rows = request_rows(bound(point['inputs']['planning_trace']))
            contract = helper.startup_contract(point, loaded.model, plan, options, rows,
                profile=config['profile'], planning_trace=point['inputs']['planning_trace'])
            if draft['source_mode'] == 'historical':
                if contract['expected_first_plan'] != choice['startup_contract']['expected_first_plan']:
                    raise ValueError('historical startup plan changed')
            else:
                choice['startup_contract'] = contract
            config_path, choice_path = out / 'configs' / (point['name'] + '.json'), out / 'choices' / (point['name'] + '.json')
            write(config_path, config)
            write(choice_path, choice)
            point['inputs'].update(system_config=binding(config_path), offline_choice=binding(choice_path))
            validate_observation_inputs(point, point['inputs'])
            checks.append(dict(point=point['name'], actual_first_plan=contract['expected_first_plan'], passed=True))
        group_path = out / 'groups' / (group['session_id'] + '.json')
        write(group_path, group)
        job = resident_job(group, group_path, root=ROOT, source=Path(source_ref['path']).parent,
            image=draft['execution']['image_digest'], verification=draft['execution']['model_verification']['path'],
            campaign=out / 'campaign.json', priority=300 - index)
        # The original factory uses a fixed host mount. Bind the target location
        # explicitly without rewriting historical model identities or receipts.
        job['payload']['argv'] = [draft['models_dir'] + ':/models:ro' if value == '/home/models:/models:ro' else value
                                  for value in job['payload']['argv']]
        job['payload'].update(run_id=out.name, system='pdblend', model_id=group['model_id'],
            after_terminal=[jobs[-1]['job_id']] if jobs else [], formal_eligible=False)
        jobs.append(job)
        groups.append(group)
    write(out / 'execution-inputs.json', draft['execution'])
    campaign = dict(draft, groups=groups, points=[p for g in groups for p in g['points']],
                    execution_inputs=binding(out / 'execution-inputs.json'),
                    execution_source_manifest=source_ref, duration_s=150, seed=701)
    write(out / 'campaign.json', campaign)
    write(out / 'jobs.json', jobs)
    write(out / 'input-preflight.json', checks)
    write(out / 'preparation.json', dict(schema='pdblend-v3-preparation/v1',
        campaign=binding(out / 'campaign.json'), jobs=binding(out / 'jobs.json'),
        source_manifest=source_ref, points=len(checks), jobs_count=len(jobs),
        cpu_image_preflight_passed=True, hardware_executed=False,
        queue=str(out / 'queue.json'), enqueued=False, formal_eligible=False,
        historical_profile_reused=True, target_machine_calibrated=False))


def start(out):
    from pdblend.experimentation.lease import GPULeaseQueue
    from pdblend.experimentation.worker import run_one
    preparation = read(out / 'preparation.json')
    campaign = bound(preparation['campaign'])
    jobs = bound(preparation['jobs'])
    if not preparation.get('cpu_image_preflight_passed'):
        raise ValueError('CPU image preflight is required')
    if target_gpus() != campaign['execution']['gpu_uuids']:
        raise ValueError('target hardware changed after preparation; prepare a new output')
    # Fresh queue creation is deliberate. Resume uses the normal queue worker
    # with this explicit path, never an imported source-server database.
    queue_path = out / 'queue.json'
    if queue_path.exists():
        raise FileExistsError('queue exists; use the documented explicit worker command to resume')
    queue = GPULeaseQueue(queue_path)
    for job in jobs:
        queue.enqueue(**job)
    while not (out / 'worker.stop').exists():
        statuses = [job.status for job in queue.list_jobs()]
        if all(status in ('succeeded', 'failed', 'cancelled', 'blocked') for status in statuses):
            break
        if not run_one(queue):
            import time
            time.sleep(5)
    return dict(queue=str(queue_path), statuses={job.job_id: job.status for job in queue.list_jobs()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inventory = commands.add_parser('inventory', help='verify and list exact restart runtime files, without GPU work')
    inventory.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    inventory.add_argument('--models', nargs='+', choices=SIZES, default=list(SIZES))
    inventory.add_argument('--out', type=Path, required=True)
    prep = commands.add_parser('prepare', help='fresh immutable jobs and CPU image preflight; does not enqueue')
    prep.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--models', nargs='+', choices=SIZES, default=list(SIZES))
    prep.add_argument('--models-dir', type=Path, default=Path('/home/models'))
    prep.add_argument('--image', default='pdblend:l20-cu128-vllm-v1')
    prep.add_argument('--source-mode', choices=('workspace', 'historical'), default='workspace')
    prep.add_argument('--dry-run', action='store_true')
    for name in ('start', '_image-prepare'):
        command = commands.add_parser(name)
        command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if hasattr(args, 'models') and len(args.models) != len(set(args.models)):
        parser.error('--models must be unique')
    if args.command == 'inventory':
        result = runtime_assets(args.campaign, args.models)
        write(args.out, result)
        result = dict(inventory=str(args.out.resolve()), files=len(result['files']), bytes=result['total_bytes'])
    elif args.command == 'prepare':
        result = prepare(args)
    elif args.command == 'start':
        result = start(args.out.resolve())
    else:
        image_prepare(args.out.resolve())
        result = dict(cpu_image_preflight_passed=True, hardware_executed=False)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
