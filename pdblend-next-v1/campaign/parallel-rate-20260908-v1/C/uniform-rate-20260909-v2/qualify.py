"""Fresh native qualification around the unchanged B/C legacy gates."""
import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0, str(ROOT / 'C/uniform-rate-20260909-v1'))
import support as p


def tree(path):
    return {str(q): p.sha(q) for q in Path(path).rglob('*') if q.is_file() and '__pycache__' not in q.parts}


def control(out):
    module = p.load(ROOT / 'B/ascending-rate-v1/baseline_control_v2.py', 'v2_original_B_control')
    module.HERE = out.parent
    module.RESTORE = out.parent / 'baseline-restoration-001'
    module.QUAL = out
    return module


def predecessor(out):
    owner = out.parent / 'baseline-restoration-001-owner.json'
    status = p.checked(p.ref(owner))
    p.need(status['complete'] and status['finished_s'] and not status['node_lease_held']
           and not p.active_owner(status), 'restoration must finish and release its node lease')
    p.need(p.checked(status['restoration'])['correctness']['passed'], 'fresh ordinary restoration check failed')
    return p.ref(owner), status['binding']


def gate_C(out, parent_ref, binding_ref, state, state_path):
    p.need(not out.exists(), 'fresh native gate directory required')
    boot = copy.deepcopy(p.checked(binding_ref))
    host = ROOT.parents[1] / 'releases/five-system100-C7B-baseline-v1-runtime'
    source = ROOT / 'C/baseline-until-complete-v2/validate.py'
    boot.update(host_release=str(host), system='mixed', configs={}, output=str(out / 'gate-work'),
                output_correctness_verified=False, correctness_gate_required_before_performance=True,
                old_correctness_is_historical_only=True, mechanism_proof=None,
                fresh_restart=parent_ref)
    manifest = p.read(host / 'manifest.json')
    boot['files'].update({str(host / name): digest for name, digest in manifest['files'].items()})
    for ref in (p.ref(host / 'manifest.json'), p.ref(source), parent_ref, binding_ref, p.ref(__file__)):
        boot['files'][ref['path']] = ref['sha256']
    p.save(out / 'bootstrap.json', boot)
    runtime = {p.read(i['engine_config'])['runtime_dir'] for i in boot['instances']}
    p.need(len(runtime) == 1, 'native runtime paths differ')
    argv = [sys.executable, '-B', str(source), '--binding', str(out / 'bootstrap.json'),
            '--runtime-dir', runtime.pop(), '--out', str(out / 'original27'), '--run']
    run_child(argv, out / 'original27.log', state, state_path)
    status = p.read(out / 'original27/status.json')
    p.need(status['complete'] and status['passed'] and status['measurement_valid']
           and status['native_cleanup_complete'] and status['clock_restore_complete']
           and not status['cleanup_errors'], 'fresh native mechanisms did not qualify')


def run_child(argv, log, state, state_path):
    with log.open('xb') as stream:
        child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        state['child'] = dict(pid=child.pid, argv=argv, started_s=time.time(),
                              startticks=p.process_identity(child.pid)['startticks'])
        p.save(state_path, state)
        stopped = False
        def stop(signum, _frame):
            nonlocal stopped
            if not stopped and child.poll() is None:
                stopped = True
                child.send_signal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, stop)
        code = child.wait()
        state['child'].update(exitcode=code, stopped=stopped, finished_s=time.time())
        p.save(state_path, state)
    p.need(code == 0 and not stopped, 'original native gate stopped/failed after cleanup')


def identity_policy(instance):
    result = copy.deepcopy(instance)
    result.pop('host_pid', None)
    result['provenance'].pop('pid', None)
    result['container'].pop('StartedAt', None)
    return result


def derive_C(out):
    parent_ref, binding_ref = predecessor(out)
    fresh = p.checked(binding_ref)
    paths = {system: ROOT / f'C/boundary-baseline-bindings-p4v2/{system}/binding.json'
             for system in ('mixed', 'distserve', 'dynamollm')}
    paths['ecoserve'] = ROOT / 'C/eco-drain37-v1/qualified/binding.json'
    audit = p.load(ROOT.parents[1] / 'campaign/AC-baseline-binding-v2/gate_evidence.py', 'v2_original_C_gate_audit')
    helper = p.load(ROOT / 'B/baseline-return-after-external-source-v1/execution.py', 'v2_C_source_paths')
    helper.load_common(fresh['host_release'])
    from ecopadg.serving.measurement import power_evidence
    refs = {}
    for system, path in paths.items():
        old_ref = p.ref(path)
        old = p.checked(old_ref)
        p.need([identity_policy(i) for i in old['instances']] ==
               [identity_policy(i) for i in fresh['instances']], 'retained native source/policy changed')
        strategy = 'dynamollm-resident' if system == 'dynamollm' else system
        proof, raw_files = audit.audit(out / 'original27', fresh['instances'], strategy, power_evidence)
        binding = copy.deepcopy(old)
        binding.update(instances=copy.deepcopy(fresh['instances']), deadline_s=None,
                       campaign_lifecycle='until_declared_complete_v1', output=str(out / system / 'results'),
                       correctness_evidence=str(out / 'original27'),
                       output_correctness_verified=True, correctness_gate_required_before_performance=False,
                       mechanism_proof=proof, fresh_restart=parent_ref,
                       fresh_native_qualification=dict(gate=p.ref(out / 'original27/status.json'),
                                                       selected_source=old_ref, original_policy_unchanged=True))
        binding['files'].update(raw_files)
        binding['files'].update(tree(out.parent / 'baseline-restoration-001'))
        for ref in (old_ref, parent_ref, binding_ref, p.ref(__file__), p.ref(out / 'bootstrap.json')):
            binding['files'][ref['path']] = ref['sha256']
        destination = out / system / 'binding.json'
        p.need(not destination.exists(), 'fresh system binding required')
        p.save(destination, binding)
        refs[system] = p.ref(destination)
    p.save(out / 'bindings.json', refs)


def derive_B(out):
    c = control(out)
    q = p.load(ROOT / 'B/ascending-resume-20260909-v2/qualify_eco_drained_baseline_v2.py',
               'v2_original_B_Eco_qualification')
    eco = q.qualify(p.ref(out / 'bootstrap.json'), out / 'original27', p.ref(c.ORIGINAL),
                    p.ref(c.RESTORE / 'containers.after.json'), out / 'ecoserve')
    q.audit_binding(eco)
    refs = {'ecoserve': eco}
    for system in ('mixed', 'distserve', 'dynamollm'):
        binding = c.derive_other(system)
        p.save(out / system / 'binding.json', binding)
        refs[system] = p.ref(out / system / 'binding.json')
    p.save(out / 'bindings.json', refs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', choices=('B', 'C'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('gate', 'derive'), required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    parent_ref, binding_ref = predecessor(args.out)
    p.need(not (args.out.parent / 'STOP').exists(), 'stop requested')
    p.need(args.run, 'explicit phase invocation required')
    path = args.out.parent / ('native-' + args.phase + '-owner.json')
    p.need(not path.exists(), 'fresh phase owner required')
    state = dict(schema='uniform-v2-native-qualification-owner-v1', node=args.node,
                 model='7b' if args.node == 'C' else '32b', pid=os.getpid(),
                 startticks=p.process_identity(os.getpid())['startticks'],
                 started_s=time.time(), complete=False, node_lease_held=False, predecessor=parent_ref)
    p.save(path, state)
    try:
        if args.phase == 'gate':
            if args.node == 'C':
                gate_C(args.out, parent_ref, binding_ref, state, path)
            else:
                boundary = args.out.parent / 'baseline-boundary-001.json'
                p.need(not boundary.exists(), 'fresh predecessor boundary required')
                p.save(boundary, dict(restoration=parent_ref, original_PDB_scope=p.read(args.out.parent / 'restore-spec-001.json')['predecessor']))
                control(args.out).gate()
        elif args.node == 'C':
            derive_C(args.out)
        else:
            derive_B(args.out)
        state['complete'] = True
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        state['finished_s'] = time.time()
        p.save(path, state)


if __name__ == '__main__':
    main()
