"""Freeze one model-wide dynamic policy only after actual measured calibration."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import sys
import time
import protocol as p

def certificate_inputs(reference):
    """Freeze the actual source groups, traces, requests and final inventories."""
    files, visited = {}, set()
    def retain(path, digest):
        path = str(Path(path).resolve())
        p.need(path not in files or files[path] == digest, 'conflicting calibration source identity: '+path)
        p.need(p.sha(path) == digest, 'calibration source changed: '+path)
        files[path] = digest
        return path
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                path = retain(value['path'], value['sha256'])
                if path not in visited:
                    visited.add(path)
                    if Path(path).suffix == '.json':
                        walk(p.read(path))
            for key, item in value.items():
                if key in ('files', 'artifacts') and isinstance(item, dict):
                    for path, digest in item.items():
                        if Path(path).is_absolute() and isinstance(digest, str) and len(digest) == 64:
                            retain(path, digest)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(reference)
    return files

def prepare(args):
    original_release = p.read(args.fixed_release)
    p.need(original_release['model'] == args.model and original_release['approved'] is True,
           'same-model frozen fixed strategy required')
    p.need(not args.out.exists(), 'new dynamic release directory required')
    host = args.host.resolve()
    manifest = p.read(host/'manifest.json')
    p.need(manifest['parent_release'] == original_release['host_release']
           and manifest['parent_manifest_sha256'] == original_release['host_manifest']['sha256'],
           'dynamic arm must extend exactly the measured fixed strategy')
    cpu = p.read(args.cpu)
    p.need(cpu.get('passed') is True and cpu['per_model'][args.model]['manifest_sha256'] == p.sha(host/'manifest.json')
           and cpu['per_model'][args.model]['manifest'] == str(host/'manifest.json'), 'dynamic lifecycle tests do not match host')
    capacity = p.read(args.capacity_binding)
    p.need(capacity['deadline_s'] == p.DEADLINE and capacity['identity']['tp'] == (2 if args.model == '32b' else 1),
           'dynamic engine TP/deadline changed')
    # Pure validation: no Controller, NVML or physical action is instantiated.
    sys.path[:0] = [str(host),str(host/'src'),'/root/workspace/pdblend/.runtime-deps']
    import capacity_runtime
    planner_module = capacity_runtime.load_planner(capacity['planner_source'])
    capacity_runtime.calibration_model(planner_module,capacity)
    certificate = p.checked(capacity['calibration'])
    p.need(certificate.get('three_independent_repetitions_per_item') is True,
           'three actual repetitions of every bound have not qualified')
    domains = capacity.get('demand_domains', [])
    p.need(domains and len({d['sha256'] for d in domains}) == len(domains),
           'explicit distinct causal demand domains required')
    coverage = []
    for domain in domains:
        layouts = [v for v in certificate['layouts'] if v['demand_domain_sha256'] == domain['sha256']]
        qualified = ({2, 3} <= {len(v['resident_groups']) for v in layouts}
                     and any(v['demand_domain_sha256'] == domain['sha256'] for v in certificate['savings']))
        coverage.append(dict(domain_sha256=domain['sha256'], two_three_calibrated=qualified))
    p.need(any(v['two_three_calibrated'] for v in coverage), 'no actual two/three-instance domain calibrated')
    files = dict(original_release['files'])
    files.update({str(host/name):digest for name,digest in manifest['files'].items()})
    for path,digest in files.items():
        p.need(p.sha(path) == digest, 'dynamic host source changed')
    configs = {}
    for dataset,reference in original_release['configs']['fixed2'].items():
        cfg = copy.deepcopy(p.checked(reference))
        # The fixed configs can retain unused PD transfer evidence. The dynamic
        # independent-mixed executor requires no transfer candidates at all.
        p.need(cfg.get('allow_pd') is False, 'dynamic parent must disable PD')
        cfg.pop('transfers', None)
        cfg.update(capacity_integration_v1=True,capacity_binding_path=str(args.capacity_binding.resolve()),
                   capacity_binding_sha256=p.sha(args.capacity_binding))
        config_path = args.out/'configs'/(dataset+'.json')
        p.write(config_path,cfg,exclusive=True)
        configs[dataset] = p.ref(config_path)
        files[str(config_path.resolve())] = p.sha(config_path)
    dependencies = [host/'manifest.json', args.fixed_release, args.cpu, args.capacity_binding,
                    Path(capacity['calibration']['path']), Path(capacity['planner_source']['path'])]
    dependencies += [p.ROOT/name for name in ('protocol.py','runner.py','runner_dynamic_v1.py',
        'dynamic_measurement.py','dynamic_measurement.manifest.json','build_dynamic_measurement.py',
        'dynamic_child.py','dynamic_ownership.py','prepare_dynamic_release.py')]
    adaptation = p.read(p.ROOT/'dynamic_measurement.manifest.json')
    p.need(adaptation['sha256'] == p.sha(p.ROOT/'dynamic_measurement.py')
           and adaptation['builder_sha256'] == p.sha(p.ROOT/'build_dynamic_measurement.py')
           and p.sha(adaptation['original_source']['path']) == adaptation['original_source']['sha256'],
           'frozen original-measurement adaptation differs')
    for reference in [capacity['engine_template'],capacity['correctness_oracles']]:
        p.checked(reference); files[reference['path']]=reference['sha256']
    files.update(capacity['files'])
    files.update(certificate_inputs(capacity['calibration']))
    for reference in certificate['raw_measurements']:
        raw=p.checked(reference);files[reference['path']]=reference['sha256'];files.update(raw['artifacts'])
    for path in dependencies:
        files[str(Path(path).resolve())]=p.sha(path)
    release_path=args.out.resolve()/'release.json'
    result=copy.deepcopy(original_release)
    result.update(release_path=str(release_path),created_s=time.time(),implementation_id=host.name,
        host_release=str(host),host_manifest=p.ref(host/'manifest.json'),
        cpu_validation=dict(passed=True,evidence=p.ref(args.cpu)),
        configs={'dynamic':configs},files=files,dynamic_qualified=True,
        dynamic_qualification_scope='actual empirical capacity/cost qualification; original-rate performance not yet measured',
        capacity_binding=p.ref(args.capacity_binding),fixed_comparison_release=p.ref(args.fixed_release),
        calibrated_domain_coverage=coverage,unknown_domain_capacity_action='hold',
        no_mid_declaration_certificate_changes=True,
        serving_validation_pending=True)
    p.write(release_path,result,exclusive=True)
    return p.ref(release_path)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,choices=p.MODELS)
    for name in ('fixed-release','host','cpu','capacity-binding','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    print(prepare(parser.parse_args()))
