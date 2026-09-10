"""Freeze a current uniform-rate dispatch against an independently qualified binding."""
import argparse
import json
from pathlib import Path
import time

import support as p


def prepare(*, declaration, qualification, qualification_validator, node, model, dataset,
            rate, system, out, scheduling_observations=(), predecessors=(), repeats=(1, 2),
            measurement_executor=None, extra_files=(), stop_paths=(), dynamic_reuse_observations=()):
    """References are {path,sha256}; out must be a new directory.

    The validator's verify(qualification) returns passed=True,
    independently_recomputed=True, and binding={path,sha256}. That binding must
    contain the exact qualified instances/configs/host_release/system. It is
    used unchanged. Optional measurement_executor preserves a qualified
    dynamic meter's run_one and identity entrypoints. Operational per-cell
    dynamic binding changes require their own explicit qualifier wrapper.
    """
    destination = Path(out).resolve()
    p.need(not destination.exists(), 'immutable fresh release directory required')
    contract_ref = p.ref(p.ROOT / 'common/uniform-rate-20260909-v1/contract.py')
    contract = p.load(contract_ref, 'uniform_prepare_contract')
    rows = [contract.lookup(declaration, model, dataset, rate, system, rep) for rep in repeats]
    validator = p.load(qualification_validator, 'uniform_prepare_qualification')
    q = validator.verify(qualification)
    p.need(q['passed'] and q['independently_recomputed'], 'qualification did not pass independently')
    binding_ref = q['binding']
    binding = p.checked(binding_ref)
    host = Path(binding['host_release'])
    p.need(binding['model'] == model and binding['system'] == system, 'qualified model/system differs')
    host_manifest = p.ref(host / 'manifest.json')
    references = dict(declaration=declaration, declaration_contract=contract_ref,
        qualification=qualification, qualification_validator=qualification_validator,
        binding=binding_ref, host_manifest=host_manifest,
        original_executor=p.ref(p.ROOT / 'B/baseline-return-after-external-source-v1/execution.py'),
        raw_auditor=p.ref(p.ROOT.parent / 'main-slo-improvement-v7/report.py'),
        additional_metrics=p.ref(p.ROOT / 'raw_metrics_v3.py'),
        arrival_auditor=p.ref(p.ROOT / 'audit_cooperative_arrivals_v1.py'))
    files = dict(binding['files'])
    files.update({str(host / path): digest for path, digest in p.read(host / 'manifest.json')['files'].items()})
    saved_qualification = p.checked(qualification)
    files.update(saved_qualification.get('files', {}))
    files.update(saved_qualification.get('source_files', {}))
    files.update(q.get('files', {}))
    own_sources = [p.HERE / name for name in ('support.py', 'prepare_release.py', 'run_cells.py', 'audit_cell.py',
                                           'pipeline.py', 'prepare_request.py', 'validate_handoff.py')]
    own_sources += [p.ROOT / 'common/execution-until-complete-v1/run.py',
        p.ROOT / 'common/execution-until-complete-v1/child.py', p.ROOT / 'raw_metrics_v2.py',
        p.ROOT.parent / 'main-slo-improvement-v7/protocol.py',
        p.ROOT.parent / 'main-slo-improvement-v7/raw_metrics.py']
    for path in own_sources:
        files[str(path)] = p.sha(path)
    for reference in [*references.values(), *scheduling_observations, *dynamic_reuse_observations, *predecessors, *extra_files]:
        p.need(p.sha(reference['path']) == reference['sha256'], 'release dependency changed')
        files[reference['path']] = reference['sha256']
    if measurement_executor:
        files[measurement_executor['path']] = measurement_executor['sha256']
    for row in rows:
        files[row['trace']] = row['trace_sha256']
        if row.get('source_300s_trace'):
            ref = row['source_300s_trace']
            files[ref['path']] = ref['sha256']
    release = dict(schema='uniform-rate-cells-release-v1', created_s=time.time(), node=node,
        model=model, system=system, dataset=dataset, rate_rps=float(rate), rows=rows,
        expected_hostname=binding['hostname'], scheduling_observations=list(scheduling_observations),
        dynamic_reuse_observations=list(dynamic_reuse_observations),
        predecessors=list(predecessors), stop_paths=[str(x) for x in stop_paths], files=files,
        measurement_executor=measurement_executor, **references)
    import run_cells
    run_cells.validate_rows(release, binding, contract)
    destination.mkdir(parents=True)
    p.save(destination / 'release.json', release)
    result = p.ref(destination / 'release.json')
    run_cells.load_release(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('declaration', 'qualification', 'qualification-validator', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('node', 'model', 'dataset', 'system', 'rate'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--scheduling-observation', action='append', type=Path, default=[])
    parser.add_argument('--dynamic-reuse-observation', action='append', type=Path, default=[])
    parser.add_argument('--predecessor', action='append', type=Path, default=[])
    parser.add_argument('--stop-path', action='append', type=Path, default=[])
    parser.add_argument('--measurement-executor', type=Path)
    parser.add_argument('--repeat', action='append', type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(declaration=p.ref(args.declaration), qualification=p.ref(args.qualification),
        qualification_validator=p.ref(args.qualification_validator), out=args.out,
        node=args.node, model=args.model, dataset=args.dataset, rate=args.rate, system=args.system,
        scheduling_observations=[p.ref(x) for x in args.scheduling_observation],
        dynamic_reuse_observations=[p.ref(x) for x in args.dynamic_reuse_observation],
        predecessors=[p.ref(x) for x in args.predecessor], stop_paths=args.stop_path,
        repeats=tuple(args.repeat or (1, 2)),
        measurement_executor=p.ref(args.measurement_executor) if args.measurement_executor else None)))


if __name__ == '__main__':
    main()
