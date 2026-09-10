"""Build fresh new-A capacity evidence and a qualified Alpaca handoff."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import fresh_support as f

PENDING_SIGNAL = None
CHILD_UPDATE = None


def request_stop(signum, _frame):
    global PENDING_SIGNAL
    PENDING_SIGNAL = signum


def stop_requested():
    return PENDING_SIGNAL is not None or (f.U / 'STOP').exists() or (f.HERE / 'STOP').exists()


def startticks(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def input_files():
    files = f.frozen_sources()
    for path in [f.U / 'bootstrap-spec.json', f.U / 'bootstrap-001/status.json',
                 f.U / 'profiles.domain2100.json', f.U / 'model-manifest.json',
                 f.A / 'load-p6-full-inputs-002/capacity-binding.json',
                 f.A / 'load-alpaca-inputs-001/input-declaration.json']:
        f.add(files, f.ref(path))
    bootstrap = f.read(f.U / 'bootstrap-001/status.json')
    f.add(files, bootstrap['ordinary'])
    # Native control journals are still written by the retained engines.
    # Bootstrap receipts and sampler records are terminal immutable evidence.
    files.update({p: h for p, h in f.tree_files(f.U / 'bootstrap-001').items()
                  if 'native' not in Path(p).relative_to(f.U / 'bootstrap-001').parts})
    boot_spec = f.checked(bootstrap['spec'])
    files.update(boot_spec['files'])
    files.update(f.tree_files(f.U / 'power-test-001'))
    for name in ('templates.json', 'domain.json', 'qualification900-trace.json'):
        f.add(files, f.ref(f.A / 'load-alpaca-inputs-001' / name))
    for cycle in (1, 2, 3):
        for phase in ('low', 'low40', 'high2', 'high3', 'under_load'):
            f.add(files, f.ref(f.A / f'load-alpaca-inputs-001/cycle-{cycle}-{phase}.json'))
    files.update(f.tree_files(f.A / 'dynamic-execution-isolated-power-002'))
    return files


def command(argv, log):
    f.need(not stop_requested(), 'stop requested before child launch')
    with Path(log).open('xb') as stream:
        child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        evidence = dict(pid=child.pid, startticks=startticks(child.pid), argv=argv, started_s=time.time())
        if CHILD_UPDATE:
            CHILD_UPDATE(child=evidence)
        forwarded = False
        while True:
            if stop_requested() and not forwarded and child.poll() is None:
                child.send_signal(PENDING_SIGNAL or signal.SIGTERM)
                evidence.update(stop_forwarded_s=time.time(), signal=int(PENDING_SIGNAL or signal.SIGTERM))
                forwarded = True
                if CHILD_UPDATE:
                    CHILD_UPDATE(child=evidence)
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
        evidence.update(exitcode=code, finished_s=time.time())
        f.save(Path(log).with_suffix('.process.json'), evidence)
        if CHILD_UPDATE:
            CHILD_UPDATE(child=evidence)
    f.need(not stop_requested(), 'stop requested; child has exited after its bounded cleanup')
    f.need(code == 0, 'stage failed, preserve output and diagnose: ' + str(log))


def fixed_proof(qpath, validator, out):
    result = out / 'fixed-verification.json'
    command([sys.executable, '-B', str(f.HERE / 'verify.py'), '--fixed-only',
             '--qualification', str(qpath), '--validator', str(validator), '--out', str(result)],
            out / 'fixed-verification.log')
    proof = f.read(result)
    f.need(proof['passed'] and proof['independently_recomputed'], 'fixed frequency/native qualification missing')
    return proof


def prepare_root(args, out):
    proof = fixed_proof(args.fixed_qualification, args.fixed_validator, out)
    source_binding = f.checked(proof['binding'])
    bootstrap_ref = f.ref(f.U / 'bootstrap-001/status.json')
    bootstrap = f.checked(bootstrap_ref)
    boot_spec = f.checked(bootstrap['spec'])
    f.need(source_binding['instances'] == bootstrap['instances'] and source_binding['hostname'] == bootstrap['hostname'],
           'fixed and bootstrap original engines differ')
    files = input_files()
    qualification = f.read(args.fixed_qualification)
    files.update(qualification.get('files', {}))
    files.update(qualification.get('source_files', {}))
    files.update(source_binding['files'])
    files.update(proof.get('files', {}))
    for reference in (f.ref(args.fixed_qualification), f.ref(args.fixed_validator), proof['binding'],
                      f.ref(out / 'fixed-verification.json')):
        f.add(files, reference)
    profile = f.ref(f.U / 'profiles.domain2100.json')
    templates = [f.read(path) for path in source_binding['configs'].values()]
    f.need(all(t['profiles'] == profile['path'] and t['max_service_frequency_mhz'] == 2100 for t in templates),
           'actual fixed qualification frequency domain differs')
    f.need(all(t.get('idle_domain_reacquire_v1') is True for t in templates),
           'fresh native idle-wakeup qualification must precede dynamic calibration')
    semantics = dict(schema='new-A-capacity-source-semantics-v1',
                     host_manifest=f.ref(f.HOST / 'manifest.json'), profile=profile,
                     max_service_frequency_mhz=2100, engine_template=boot_spec['engine_template'],
                     idle_domain_reacquire_v1=True, qualified_fixed_binding=proof['binding'],
                     engine_source_files=boot_spec['expected_provenance']['source_files_at_import'],
                     capacity_modules={name: f.sha(f.DRIVER / name) for name in
                                       ('capacity_runtime.py', 'capacity_backend.py', 'capacity_executor.py',
                                        'capacity_certificate.py', 'capacity_planner.py')},
                     demand_domain=f.read(f.A / 'load-alpaca-inputs-001/domain.json'),
                     old_capacity_evidence_inherited=False)
    semantic_ref = f.save(out / 'source-semantics.json', semantics)
    identity = dict(node_sha256=boot_spec['node_identity']['sha256'], model_sha256=boot_spec['model_manifest']['sha256'],
                    engine_image=boot_spec['image'], tp=1, source_sha256=semantic_ref['sha256'])
    replies = f.checked(bootstrap['ordinary'])
    cases = []
    for length in (128, 7168):
        matched = [r for r in replies if r['prompt_length'] == length]
        f.need(len(matched) == 2 and matched[0]['response']['token_ids'] == matched[1]['response']['token_ids'],
               'new-node ordinary numerical output differs')
        cases.append(dict(prompt_length=length, prompt=([9707, 1879, 13] * (length // 3 + 1))[:length],
                          token_ids=matched[0]['response']['token_ids'], max_tokens=64))
    oracle_ref = f.save(out / 'fresh-oracles.json', dict(schema='capacity-correctness-oracles-v1', measured=True,
                       model_source_identity=identity, source=bootstrap['ordinary'], bootstrap=bootstrap_ref,
                       cases=cases, old_outputs_used_only_for_numerical_crosscheck=True))
    for reference in (semantic_ref, oracle_ref):
        f.add(files, reference)
    compatibility = dict(schema='new-A-fresh-capacity-source-identity-v1', node='Anew20260909',
                         old_capacity_evidence_inherited=False, host_manifest=semantics['host_manifest'],
                         profile=profile, bootstrap=bootstrap_ref, node_identity=boot_spec['node_identity'],
                         model_manifest=boot_spec['model_manifest'], engine_image=boot_spec['image'],
                         semantics=semantic_ref, fixed_qualification=f.ref(args.fixed_qualification),
                         fixed_validator=f.ref(args.fixed_validator), files=dict(files))
    compatibility_ref = f.save(out / 'fresh-source-identity.json', compatibility)
    f.add(files, compatibility_ref)
    cap = f.read(f.A / 'load-p6-full-inputs-002/capacity-binding.json')
    cap.update(identity=identity, owner_id='uniformcap', engine_template=boot_spec['engine_template'],
               environment=boot_spec['environment'], expected_provenance=boot_spec['expected_provenance'],
               correctness_oracles=oracle_ref, http_port_base=34800, kv_port_base=62000,
               max_creations=3, runtime_dir=str(out / 'layout/runtime'), files=dict(files),
               calibrated_source_semantics=semantics, planner_source=f.ref(f.HOST / 'capacity_planner.py'),
               controller_calibration_compatibility=compatibility_ref,
               calibration_only=True, production_ready=False, physical_operation_timeout_s=360)
    cap.pop('calibration', None)
    config = copy.deepcopy(templates[0])
    config.update(instances=bootstrap['instances'], port=34750, profiles=profile['path'],
                  journal=str(out / 'unused.jsonl'), max_service_frequency_mhz=2100,
                  slo_ttft_s=1.0, slo_tpot_s=.1, slo_scale=1., prepare_peers=False,
                  allow_pd=False, transfers=[], dynamic_pools=False, slow_topology=False,
                  capacity_integration_v1=False, controller_source_release=str(f.HOST), host_source_release=str(f.HOST))
    config.pop('measurement_window_protocol', None)
    config.pop('arrival_window_s', None)
    config['frequency_costs'] = [v for v in config.get('frequency_costs', [])
                               if v['source_mhz'] <= 2100 and v['target_mhz'] <= 2100]
    source_binding = dict(source_binding, files=dict(files), configs={}, independent_capacity_qualification_granted=False)
    return dict(files=files, capacity=cap, config=config, binding=source_binding, profile=profile,
                compatibility=compatibility_ref, fixed_qualification=f.ref(args.fixed_qualification),
                fixed_validator=f.ref(args.fixed_validator))


def prepare_stage(root, out, mode, *, certificate=None, paired=None):
    inputs = out / (mode + '-inputs')
    run = out / mode
    files = dict(root['files'])
    cap = copy.deepcopy(root['capacity'])
    cap.update(runtime_dir=str(run / 'runtime'), owner_id='uniformcap',
               max_creations=3 if mode == 'layout_calibration' else 8,
               calibration_only=mode == 'layout_calibration')
    config = copy.deepcopy(root['config'])
    if certificate:
        cap['calibration'] = certificate
        cap.update(rate_observation_window_s=60., arrival_count_margin=2., policy={})
        f.add(files, certificate)
        files.update(f.tree_files(Path(certificate['path']).parent))
        config['capacity_integration_v1'] = True
    cap['files'] = dict(files)
    capref = f.save(inputs / 'capacity-binding.json', cap)
    if certificate:
        config.update(capacity_binding_path=capref['path'], capacity_binding_sha256=capref['sha256'])
    cfgref = f.save(inputs / 'config.json', config)
    for reference in (capref, cfgref):
        f.add(files, reference)
    binding = dict(root['binding'], files=dict(files), configs=dict(alpaca=cfgref['path']))
    binding_ref = f.save(inputs / 'execution-binding.json', binding)
    f.add(files, binding_ref)
    hooks_manifest = f.save(inputs / 'hooks-manifest.json', dict(files={str(f.DRIVER / name): f.sha(f.DRIVER / name)
                                             for name in ('sampler_hooks.py', 'meter_evidence.py')}))
    f.add(files, hooks_manifest)
    spec = dict(schema='capacity-load-calibration-spec-v1', authorized=True, automatic_retries=False, mode=mode,
                deadline_s=None, campaign_lifecycle='until_declared_complete_v1', original_binding=binding_ref,
                capacity_binding=capref, config=cfgref, profiles=root['profile'],
                common_executor=f.ref(f.ROOT / 'common/execution-until-complete-v1/run.py'),
                host_release=str(f.HOST), input_declaration=f.ref(f.A / 'load-alpaca-inputs-001/input-declaration.json'),
                demand_domain_sha256=cap['demand_domain']['sha256'], gpus=[5], matched_idle_duration_s=60,
                cold_start_after_s=10., api_base='http://127.0.0.1:34750', served_model='pdblend',
                stop_path=str(f.U / 'STOP'), controller_calibration_compatibility=root['compatibility'],
                measurement_adapter=f.ref(f.U / 'isolated-power/manifest.json'),
                measurement_hooks=f.ref(f.DRIVER / 'sampler_hooks.py'), measurement_hooks_manifest=hooks_manifest,
                files=files, fresh_node=True, historical_results_not_inherited=True)
    if mode == 'layout_calibration':
        cycles = []
        for n in (1, 2, 3):
            cycle = {key: f.ref(f.A / f'load-alpaca-inputs-001/cycle-{n}-{key}.json')
                     for key in ('low', 'low40', 'high2', 'high3', 'under_load')}
            for operation in ('restore', 'remove'):
                cycle[operation] = f.save(inputs / f'cycle-{n}-{operation}.json',
                    dict(schema='capacity-calibration-action-v1', authorized=True, operation=operation, gpus=[5],
                         deadline_s=None, campaign_lifecycle='until_declared_complete_v1',
                         automatic_retries=False, work_budget_s=360, repeat=n,
                         purpose='fresh new-A empirical calibration with exact unchanged runtime'))
            cycles.append(cycle)
            for reference in cycle.values():
                f.add(files, reference)
        spec['cycles'] = cycles
    else:
        spec.update(arm='dynamic', actual_certificate=certificate)
        if mode == 'automatic_underload_gate':
            spec.update(trace=f.ref(f.A / 'load-alpaca-inputs-001/cycle-1-under_load.json'), paired_trace_result=paired)
        else:
            spec['trace'] = f.ref(f.A / 'load-alpaca-inputs-001/qualification900-trace.json')
        f.add(files, spec['trace'])
        if paired:
            f.add(files, paired)
    reference = f.save(inputs / 'spec.json', spec)
    from calibration_compatibility import validate_compatibility
    validate_compatibility(spec, cap)
    return reference, run


def main():
    global CHILD_UPDATE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--handoff', type=Path)
    ap.add_argument('--fixed-qualification', type=Path)
    ap.add_argument('--fixed-validator', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    if args.manifest:
        f.save(args.manifest, dict(schema='new-A-dynamic-producer-source-closure-v1', files=input_files(),
                                  control_algorithm_changed=False, fresh_node_calibration_required=True))
        return
    f.need(args.out and args.handoff and args.fixed_qualification and args.fixed_validator, 'complete producer paths required')
    f.need(not args.out.exists() and not args.handoff.exists(), 'fresh immutable producer output required')
    f.need('PDBLEND_NODE_LOCK_FD' not in os.environ, 'each original driver needs its own fresh node lock')
    if not args.run and not args.prepare_only:
        print(json.dumps(dict(cpu_only=True, source_files=len(input_files()), hardware_actions=False)))
        return
    args.out.mkdir(parents=True)
    status = dict(schema='new-A-dynamic-producer-status-v1', started_s=time.time(), pid=os.getpid(), startticks=startticks(os.getpid()),
                  complete=False, node='Anew20260909', stages=[], node_lease_held=False)
    def update(**values):
        status.update(values, updated_s=time.time())
        path = args.out / 'status.json'; temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2, allow_nan=False)+'\n'); temp.replace(path)
    CHILD_UPDATE = update
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    update(phase='verify_fresh_native_frequency')
    try:
        root = prepare_root(args, args.out)
        if args.prepare_only:
            spec, run = prepare_stage(root, args.out, 'layout_calibration')
            command([sys.executable, '-B', str(f.DRIVER / 'capacity_load_calibrate.py'), '--spec', spec['path'],
                     '--spec-sha256', spec['sha256'], '--out', str(run)], args.out / 'driver-cpu-validation.log')
            update(phase='cpu_prepared_only', complete=True, hardware_actions=False, spec=spec, finished_s=time.time())
            return
        import audit
        certificate = None
        paired = None
        for mode in ('layout_calibration', 'automatic_underload_gate', 'qualification900'):
            f.need(not stop_requested(), 'stop requested at dynamic qualification boundary')
            spec, run = prepare_stage(root, args.out, mode, certificate=certificate, paired=paired)
            update(phase=mode, current_spec=spec, current_output=str(run))
            command([sys.executable, '-B', str(f.DRIVER / 'capacity_load_calibrate.py'), '--spec', spec['path'],
                     '--spec-sha256', spec['sha256'], '--out', str(run), '--run'], args.out / (mode + '.log'))
            proof = audit.audit_stage(run, spec, mode)
            proof_ref = f.save(args.out / (mode + '-audit.json'), proof)
            status['stages'].append(dict(mode=mode, spec=spec, output=str(run), audit=proof_ref))
            update()
            if mode == 'layout_calibration':
                certificate = audit.build_certificate(run, spec, args.out / 'certificate')
                paired = f.ref(run / 'cycle-1-under_load-layout2to3/result.json')
                update(certificate=certificate)
        base = f.checked(f.checked(status['stages'][-1]['spec'])['original_binding'])
        f.need(not stop_requested(), 'stop requested before publishing dynamic qualification')
        base.update(independent_capacity_qualification_granted=True, isolated_power_adapter=f.ref(f.U / 'isolated-power/manifest.json'))
        base_ref = f.save(args.out / 'qualified-base-binding.json', base)
        update(complete=True, phase='qualified', binding=base_ref, finished_s=time.time())
        qualification = dict(schema='new-A-fresh-dynamic-capacity-qualification-v1', node='Anew20260909', model='14b',
            binding=base_ref, status=f.ref(args.out / 'status.json'), stages=status['stages'], certificate=certificate,
            fixed_qualification=root['fixed_qualification'], fixed_validator=root['fixed_validator'],
            source_identity=root['compatibility'], files=f.tree_files(args.out), source_files=input_files())
        qualification_ref = f.save(args.out / 'qualified.json', qualification)
        f.save(args.handoff, dict(node='Anew20260909', model='14b', system='pdblend',
            qualification=qualification_ref, qualification_validator=f.ref(f.HERE / 'verify.py'),
            prepare_adapter=f.ref(f.HERE / 'prepare_adapter.py'), per_cell_release=True,
            measurement_executor=f.ref(f.A / 'dynamic-execution-isolated-power-002/dynamic_measurement.py'),
            extra_files=[f.ref(args.out / 'status.json')], predecessors=[f.ref(args.out / 'status.json')]))
    except BaseException as exc:
        update(complete=False, phase='stopped_failure', error=repr(exc), finished_s=time.time())
        raise


if __name__ == '__main__':
    main()
