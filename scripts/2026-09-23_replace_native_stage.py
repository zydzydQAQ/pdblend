#!/usr/bin/env python3
"""Replace failed/unstarted native jobs with a narrowly patched frozen source."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pdblend.experimentation.lease import GPULeaseQueue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--kind', choices=['eco4','dynamo','distserve_pipeline'], required=True)
    p.add_argument('--models', nargs='+', choices=['7B','14B','32B'])
    p.add_argument('--files', nargs='+', required=True, help='reviewed paths relative to src')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    result = deepcopy(json.loads(args.spec.read_text()))
    q = GPULeaseQueue(ROOT/'results/2026-09-22/three-model/queue.json')
    state = q.snapshot()
    requested = {'Qwen2.5-'+size+'-Instruct' for size in (args.models or ['7B','14B','32B'])}
    chosen = [j for j in result['jobs'] if j['payload'].get('kind') == args.kind
              and j['payload'].get('model_id') in requested]
    if len(chosen) != len(requested):
        raise ValueError('expected exactly one job per selected model')
    for job in chosen:
        current = state['jobs'][job['job_id']]
        if current['status'] not in ('queued','failed') or (current['status']=='queued' and current['attempts']):
            raise ValueError('cannot replace a running, completed or previously attempted queued job')
    helper_spec = importlib.util.spec_from_file_location('freeze', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    # No partially edited core files enter this bundle. Copy the prior frozen
    # source and replace only the explicitly reviewed implementation files.
    bases = {next(x.split(':/opt/pdblend-src:ro')[0] for x in j['payload']['argv']
                  if x.endswith(':/opt/pdblend-src:ro')) for j in chosen}
    if len(bases) != 1:
        raise ValueError('one previous source bundle required')
    with tempfile.TemporaryDirectory(prefix='pdblend-stage-overlay-') as tmp:
        staging = Path(tmp)/'src'
        shutil.copytree(next(iter(bases)), staging)
        for name in args.files:
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe implementation path')
            shutil.copy2(ROOT/'src'/relative, staging/relative)
        source, sha = helper.freeze_source(staging, ROOT/'results/2026-09-22/three-model/native-acceptance-sources')
    replacements, changes = {}, []
    for job in chosen:
        old = job['job_id']; new = old+'-'+sha[:12]
        payload = job['payload']; argv = payload['argv']
        argv[argv.index('--name')+1] = new
        argv[:] = [str(source)+':/opt/pdblend-src:ro' if x.endswith(':/opt/pdblend-src:ro') else
                   'PDBLEND_SOURCE_SHA256='+sha if x.startswith('PDBLEND_SOURCE_SHA256=') else x for x in argv]
        payload.update(source_sha256=sha, container_name=new)
        job['job_id'] = new; replacements[old] = new; changes.append((old, job))
    for job in result['jobs']:
        deps = job['payload'].get('depends_on', [])
        if not any(dep in replacements for dep in deps):
            continue
        old = job['job_id']; current = state['jobs'][old]
        if current['status'] != 'queued' or current['attempts']:
            raise ValueError('dependent profile wave already started')
        new = old+'-'+sha[:12]; job['job_id'] = new
        payload = job['payload']; payload['container_name'] = new
        payload['depends_on'] = [replacements.get(x,x) for x in deps]
        argv = payload['argv']; argv[argv.index('--name')+1] = new
        changes.append((old, job))
    result['parent_spec'] = str(args.spec.resolve())
    result.setdefault('source_variants', {})[args.kind] = dict(path=str(source), sha256=sha, files=args.files)
    output = args.spec.with_name('native-stage-'+args.kind+'-'+sha[:16]+'.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    if not args.prepare_only:
        fresh = q.snapshot()
        for old, _ in changes:
            if fresh['jobs'][old]['status'] != state['jobs'][old]['status']:
                raise RuntimeError('job state changed; review current leases before replacing')
        for old, _ in changes:
            if fresh['jobs'][old]['status'] == 'queued':
                q.block(old, reason='Replaced before execution by reviewed native-stage patch '+sha[:12])
        for _, job in changes:
            q.enqueue(**job)
    print(json.dumps(dict(spec=str(output), source_sha256=sha, prepared_only=args.prepare_only,
                         replacements={old:job['job_id'] for old,job in changes}), indent=2))


if __name__ == '__main__':
    main()
