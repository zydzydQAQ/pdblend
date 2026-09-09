"""Freeze formal P8 initial2 only after actual P8 autonomous gate and full900."""
import copy
import json
from pathlib import Path
import socket
import sys
import time
from run import checked, need, read, ref, sha, validate_candidate_qualification

HERE = Path(__file__).resolve().parent
A = HERE.parent
ROOT = A.parent
HOST = ROOT / 'hosts/14b-capacity-p8'
OUT = A / 'final-p8-release-001'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    need(not OUT.exists(), 'new final release required')
    runner_cpu = read(HERE / 'cpu-validation.json')
    need(runner_cpu['passed'] and runner_cpu['gpu_qualified'] is False
         and all(sha(file) == digest for file, digest in runner_cpu['files'].items()),
         'final runner CPU checks do not match its actual source')
    qref = ref(A / 'final-p8-continuation-001/qualification.json')
    q = checked(qref)
    validate_candidate_qualification(q)
    need(q['passed'] and q['source'] == ref(HOST / 'manifest.json'), 'actual P8 qualification required')
    cpu = ref(ROOT / 'cpu-validation-p8.json')
    need(cpu['sha256'] == 'ca892d296f7b49c58d328a099ca30cd9ff9e6c36ed58a31c451c0bea05fd8271'
         and checked(cpu)['passed'], 'frozen CPU evidence changed')
    need(q['source']['sha256'] == '54f3e3b609dde6452127ae9d4d040ff1f18715baaa9d220c652147bc8f5ca40f',
         'frozen actual P8 source changed')
    cap = checked(q['capacity_binding'])
    certificate = checked(q['certificate'])
    need(cap['calibration'] == q['certificate'] and cap['identity'] == certificate['identity'],
         'certificate does not belong to actual qualified binding')
    sys.path[:0] = [str(HOST), str(HOST / 'src'), str(ROOT), '/root/workspace/pdblend/.runtime-deps']
    from capacity_runtime import calibration_model, load_planner
    from capacity_calibration_compatibility_v3 import verify
    verify(q['controller_calibration_compatibility'], q['source'], cap)
    calibration_model(load_planner(cap['planner_source']), cap)
    fullspec = read(A / 'load-p6-full-inputs-002/spec.json')
    base = checked(fullspec['original_binding'])
    need(q['profile'] == fullspec['profiles'], 'actual measured profile differs')
    cap.update(owner_id='ap8final', http_port_base=32300, kv_port_base=62000, max_creations=8,
               runtime_dir=str(A / 'final-p8-001/capacity-runtime'),
               production_ready=False, calibration_only=False)
    cap['files'].update(q['files'])
    cap['files'][qref['path']] = qref['sha256']
    cap['files'][str(HERE / 'prepare.py')] = sha(HERE / 'prepare.py')
    write(OUT / 'capacity-binding.json', cap)
    capref = ref(OUT / 'capacity-binding.json')
    config = read(A / 'p4-minimal/fixed-release-001/configs/alpaca.json')
    config.update(instances=copy.deepcopy(base['instances']), prepare_peers=False, transfers=[],
                  capacity_integration_v1=True, capacity_binding_path=capref['path'],
                  capacity_binding_sha256=capref['sha256'])
    need(config['profiles'] == q['profile']['path'] and config['arrival_window_s'] == 100
         and config['request_timeout_s'] == 120, 'unchanged serving measurement protocol required')
    configs = {}
    for dataset in ('alpaca', 'sharegpt', 'longbench'):
        write(OUT / 'configs' / (dataset + '.json'), config)
        configs[dataset] = ref(OUT / 'configs' / (dataset + '.json'))
    manifest = checked(q['source'])
    files = {**base['files'], **runner_cpu['files'], **cap['files'], **q['files']}
    files.update({str(HOST / file): digest for file, digest in manifest['files'].items()})
    executor_dir = A / 'dynamic-execution-until-complete-001'
    for file in (HERE / 'run.py', HERE / 'prepare.py', HERE / 'continue.py', HERE / 'qualification_contract.py',
                 HERE / 'work-declaration.json', HERE / 'cpu-validation.json', HERE / 'manifest.json',
                 A / 'p4-minimal/runner.py', A / 'p4-minimal/protocol.py',
                 executor_dir / 'manifest.json', executor_dir / 'dynamic_measurement.manifest.json',
                 executor_dir / 'dynamic_measurement.py', executor_dir / 'dynamic_child.py',
                 executor_dir / 'dynamic_ownership.py', executor_dir / 'protocol.py'):
        files[str(file)] = sha(file)
    for reference in (qref, q['source'], cpu, q['profile'], q['certificate'], capref,
                      q['qualification_auditor'], q['stage_auditor'], q['controller_calibration_compatibility'],
                      ref(ROOT / 'common/capacity-off-reuse-declaration-v3.json'), *configs.values()):
        files[reference['path']] = reference['sha256']
    need(all(sha(file) == digest for file, digest in files.items()), 'frozen source/raw evidence changed')
    base.update(host_release=str(HOST), deadline_s=None, campaign_lifecycle='until_declared_complete_v1',
        configs={dataset: value['path'] for dataset, value in configs.items()}, files=files,
        controller_calibration_compatibility=q['controller_calibration_compatibility'],
        experiment_scope='final-common-p8-explicit-P6-calibration-single-domain-dynamic-initial2',
        unchanged_pdb_policy=False, formal_eligible=False, output=str(A / 'final-p8-001/results'))
    write(OUT / 'binding.json', base)
    release = dict(schema='parallel-rate-p8-qualified-dynamic-release-v1', approved=True,
        dynamic_qualified=True, production_ready=False, created_s=time.time(), model='14b',
        hostname=socket.gethostname(), deadline_s=None, campaign_lifecycle='until_declared_complete_v1',
        implementation_id='14b-capacity-p8-final-single-alpaca-domain', host_release=str(HOST),
        host_manifest=q['source'], profile=q['profile'], cpu_validation=cpu,
        qualification900=qref, qualification_auditor=q['qualification_auditor'], stage_auditor=q['stage_auditor'],
        autonomous_gate=q['autonomous_gate'], actual900=q['arms']['dynamic'],
        controller_calibration_compatibility=q['controller_calibration_compatibility'],
        measured_controller_manifest=q['measured_controller_manifest'],
        actual_transition_controller_manifest=q['actual_transition_controller_manifest'],
        empirical_cost_prediction_misses=q['empirical_cost_prediction_misses'],
        transition_prediction_errors=q['transition_prediction_errors'],
        empirical_estimates_are_not_guarantees=True,
        capacity_off_reuse=ref(ROOT / 'common/capacity-off-reuse-declaration-v3.json'),
        actual_controller_manifest=q['actual_controller_manifest'],
        capacity_binding=capref, binding=ref(OUT / 'binding.json'), configs=configs, files=files,
        declaration=ref(HERE / 'work-declaration.json'),
        dynamic_executor=ref(executor_dir / 'dynamic_measurement.py'),
        ordinary_helper=ref(A / 'p4-minimal/runner.py'), stop_path=str(A / 'STOP_FINAL_P8'),
        source_scope='actual common P8; P6 layout/savings and actual P8 three-transition calibration; A12 domain only',
        cold_start_slo_limitation_preserved=True, initial_instances_each_cell=2,
        formal100_performance_not_inferred_from900=True,
        candidate900_full_reference900_capacity_negative=True,
        qualification_reference_equal_work_energy_comparison_eligible=False)
    write(OUT / 'release.json', release)
    print(json.dumps(ref(OUT / 'release.json')))


if __name__ == '__main__':
    main()
