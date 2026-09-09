"""Freeze a model-wide fixed-two policy, preserving the actual engine binding."""
import argparse
import copy
from pathlib import Path
import time
import protocol as p

FLAGS = {'admission_round_fairness': True, 'pending_admission_capacity_guard': True,
         'measured_frequency_write_guard_v1': True, 'observed_first_admission_frequency_v1': False, 'observed_idle_admission_frequency_v2': True}

def prepare(model, binding_path, host, qualification, cpu, out, profile=None):
    binding_path, host, out = map(lambda x: Path(x).resolve(), (binding_path, host, out))
    base = p.read(binding_path)
    p.need(base['model'] == model and base['system'] == 'pdblend' and len(base['instances']) == 2,
           'actual original two-PDB binding required')
    p.need(p.read(qualification).get('passed') is True, 'actual ordinary/profile qualification is not passed')
    cpu_result = p.read(cpu)
    measured_cpu = cpu_result['per_model'][model]
    p.need(cpu_result['passed'] is True and measured_cpu['manifest'] == str(host / 'manifest.json')
           and measured_cpu['manifest_sha256'] == p.sha(host / 'manifest.json'), 'CPU validation belongs to another runtime')
    host_manifest = p.read(host / 'manifest.json')
    runtime_files = {str(host / name): digest for name, digest in host_manifest['files'].items()}
    for path, digest in runtime_files.items():
        p.need(p.sha(path) == digest, 'new runtime changed before release')
    p.need(not out.exists(), 'new release directory required')
    configs = {}
    profile_refs = {}
    changes = {}
    for dataset in p.DATASETS:
        old = p.read(base['configs'][dataset])
        p.need(p.sha(base['configs'][dataset]) == base['files'][base['configs'][dataset]], 'original config changed')
        cfg = copy.deepcopy(old)
        cfg.update(FLAGS)
        if profile:
            cfg['profiles'] = str(Path(profile).resolve())
        p.need(cfg.get('allow_pd') is False and cfg.get('dynamic_pools') is False,
               'fixed mixed configuration only')
        p.need({i['id'] for i in cfg['instances']} == {i['id'] for i in base['instances']},
               'actual config and binding instance IDs differ')
        changes[dataset] = {k: {'before': old.get(k), 'after': v} for k, v in cfg.items() if old.get(k) != v}
        p.need(set(changes[dataset]) <= {*FLAGS, 'profiles'}, 'unapproved policy change')
        p.write(out / 'configs' / (dataset + '.json'), cfg, exclusive=True)
        configs[dataset] = p.ref(out / 'configs' / (dataset + '.json'))
        profile_refs[dataset] = p.ref(cfg['profiles'])
    p.need(len({v['sha256'] for v in profile_refs.values()}) == 1, 'one profile across all datasets required')
    files = dict(runtime_files)
    for path in (host / 'manifest.json', p.ROOT / 'protocol.py', p.ROOT / 'runner.py',
                 p.ROOT / 'prepare_release.py', Path(cpu), Path(qualification),
                 p.REPO / 'campaign/five-system-execution-v3/run.py',
                 p.REPO / 'campaign/five-system-execution-v3/child.py',
                 p.REPO / 'campaign/pdblend-ablation-20260908-v1/execution.py'):
        files[str(path.resolve())] = p.sha(path)
    for reference in [*configs.values(), *profile_refs.values()]:
        files[reference['path']] = reference['sha256']
    release = dict(schema='main-slo-improvement-release-v1', approved=True, model=model,
        created_s=time.time(), deadline_s=p.DEADLINE, implementation_id=host.name,
        binding=p.ref(binding_path), host_release=str(host), host_manifest=p.ref(host / 'manifest.json'),
        qualification=p.ref(qualification), cpu_validation=dict(passed=True, evidence=p.ref(cpu)),
        declaration=p.ref(p.ROOT / 'work-declaration.json'), configs={'fixed2': configs},
        profile_refs=profile_refs, policy_diff=changes, files=files, dynamic_qualified=False,
        original_engine_source_preserved=True, serving_validation_pending=True,
        serving_validation='fresh measured ordinary gate followed by declared actual fixed2 screen',
        no_baseline_rerun=True, original_results_unchanged=True)
    p.write(out / 'release.json', release, exclusive=True)
    return p.ref(out / 'release.json')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=p.MODELS)
    parser.add_argument('--binding', required=True, type=Path)
    parser.add_argument('--host', required=True, type=Path)
    parser.add_argument('--qualification', required=True, type=Path)
    parser.add_argument('--cpu', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    print(prepare(args.model, args.binding, args.host, args.qualification, args.cpu, args.out, args.profile))

if __name__ == '__main__':
    main()
