"""Freeze a new measurement binding to already validated, idle PDB engines."""
import argparse
import copy
import json
from pathlib import Path
import socket
import subprocess
import urllib.request

from run import GLOBAL_DEADLINE, PROTOCOL, read, require, sha, write, stat_identity

ROOT = Path('/root/workspace/pdblend-next-v1')
MODELS = {
    '14b': dict(label='A14B', old='A14B-deadline-matrix-v1', names=['pdb-v2-nextv3a6', 'pdb-v2-nextv3a7'], native='v3', budget=2048),
    '32b': dict(label='B32B', old='B32B-main-scale-fixed-window-v1', names=['pdb-v2-nextv3b0', 'pdb-v2-nextv3b1'], native='v3', budget=None),
    '7b': dict(label='C7B', old='C7B-deadline-matrix-v1', names=['pdb-next-c7quick7', 'pdb-next-pdbcap-12aee0f43ea3429bb4ca3f4d3f3570bb'], native='legacy_sync_put', budget=None),
}


def build(model, host, manifest, out):
    settings = MODELS[model]
    old = ROOT / 'campaign' / settings['old']
    require((old / 'STOP').exists(), 'old queue stop not requested')
    invocations = sorted((old / 'invocations').glob('*.json'))
    latest = read(invocations[-1])
    require(latest.get('complete') is True and latest.get('phase') == 'stopped_by_request', 'old queue not stopped at complete cell boundary')
    original = read(old / 'inputs/controller.fixed.json')
    require(read(manifest)['model'] == model, 'wrong model trace manifest')
    require(not out.exists(), 'new binding destination required')
    out.mkdir(parents=True)
    files = {}
    def freeze(path):
        path = Path(path).resolve()
        require(path.is_file(), 'missing binding input: ' + str(path))
        files[str(path)] = sha(path)
    old_freeze = read(old / 'freeze.json')
    large_inputs = old_freeze['dependencies'].get('large_inputs', {})
    for path, digest in old_freeze['dependencies']['files'].items():
        if path in large_inputs:
            require(stat_identity(path) == large_inputs[path]['stat'], 'previously verified model input stat changed')
            require(sha(path) == digest == large_inputs[path]['sha256'], 'previous model/large evidence bytes changed')
        else:
            freeze(path)
            require(files[path] == digest, 'previous validated input changed: ' + path)
    for path in (manifest, host / 'manifest.json', old / 'package-manifest.json', old / 'freeze.json', invocations[-1]):
        freeze(path)
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('test_'):
            freeze(path)
    for name, digest in read(host / 'manifest.json')['files'].items():
        freeze(host / name)
        require(files[str(host / name)] == digest, 'new host source differs')
    for row in read(manifest)['workloads']:
        freeze(row['trace'])
        require(files[row['trace']] == row['trace_sha256'], 'new trace differs')
    def dependencies(value, key=''):
        if isinstance(value, dict):
            for k, v in value.items(): dependencies(v, k)
        elif isinstance(value, list):
            for v in value: dependencies(v, key)
        elif isinstance(value, str) and value.startswith('/') and Path(value).is_file() and key != 'journal':
            freeze(value)
    dependencies(original)
    configs = {}
    for dataset in ('alpaca', 'sharegpt', 'longbench'):
        cfg = copy.deepcopy(original)
        cfg.update(measurement_window_protocol=PROTOCOL, arrival_window_s=100.,
            slo_attainment_target=.9, comparison_system='pdblend', controller_source_release=str(host))
        for key in ('host_source_release',):
            if key in cfg: cfg[key] = str(host)
        path = out / 'configs' / (dataset + '.json')
        write(path, cfg); freeze(path); configs[dataset] = str(path)
    names = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).split()
    require(set(names) == set(settings['names']), 'unexpected live deployment')
    rows = json.loads(subprocess.check_output(['docker', 'inspect', *names], text=True))
    by_name = {r['Name'].lstrip('/'): r for r in rows}
    prior_inventory = {r['Name'].lstrip('/'): r for r in read(old / 'freeze.json')['inventory']}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    instances = []
    for instance in original['instances']:
        name = next(n for n, r in by_name.items() if any(
            f'--config' == arg and read(r['Args'][index+1]).get('id') == instance['id']
            for index, arg in enumerate(r['Args'][:-1])))
        c = by_name[name]
        prior = prior_inventory[name]
        require(c['Id'] == prior['Id'] and c['Image'] == prior['Image']
            and c['State']['StartedAt'] == prior['StartedAt'], 'validated PDB process changed')
        p = json.load(opener.open(instance['url'] + '/provenance', timeout=10))
        raw = json.load(opener.open(instance['url'] + '/runtime', timeout=10))
        require(not any(raw.get(k) for k in ('active', 'running', 'waiting', 'kv_allocations', 'transfer_allocations')),
            'old engine still executing')
        require(p.get('instance_id') == instance['id'] and p.get('tp') == instance['tp']
            and p.get('model') == '/models/Qwen2.5-' + model.upper() + '-Instruct'
            and p.get('cuda_visible_devices') == ','.join(str(g) for g in instance['gpus']), 'actual model/TP/GPU differs')
        cfg_path = c['Args'][c['Args'].index('--config') + 1]; freeze(cfg_path)
        provenance = {k: p[k] for k in ('instance_id', 'tp', 'model', 'cuda_visible_devices', 'source_files_at_import')}
        i = dict(instance, port=int(instance['url'].rsplit(':', 1)[-1]), native_kind=settings['native'],
            scheduler_cache_observed=True, scheduler_cache_count=1,
            container=dict(name=name, id=c['Id'], image=c['Image'], StartedAt=c['State']['StartedAt']),
            provenance=provenance)
        if settings['budget']:
            i.update(service_budget_tokens=settings['budget'], restore_budget_tokens=8192)
        instances.append(i)
    write(out / 'identity-at-binding.json', rows)
    freeze(out / 'identity-at-binding.json')
    binding = dict(schema=1, protocol_id=PROTOCOL, model=model, system='pdblend', hostname=socket.gethostname(),
        deadline_s=GLOBAL_DEADLINE, host_release=str(host), output=str(out / 'results'),
        configs=configs, instances=instances, files=files, large_inputs=large_inputs, prior_package=str(old),
        old_completed_main_cells=latest.get('checkpointed_cells'), unchanged_pdb_policy=True,
        window_s=100, seeds=[701], formal_eligible=False)
    write(out / 'binding.json', binding)
    print(json.dumps(dict(binding=str(out / 'binding.json'), sha256=sha(out / 'binding.json'), inputs=len(files))))
    return binding


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--host', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    build(a.model, a.host.resolve(), a.manifest.resolve(), a.out.resolve())
