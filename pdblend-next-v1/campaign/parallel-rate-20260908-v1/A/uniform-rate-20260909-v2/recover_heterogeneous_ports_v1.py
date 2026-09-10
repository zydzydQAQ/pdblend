"""Resume only a terminal, pre-hardware heterogeneous port-collision failure."""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common/uniform-rate-20260909-v2'))
import support as p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    job_ref = p.ref(a.job)
    job = p.checked(job_ref)
    p.need(not a.out.exists(), 'fresh recovery supervisor output required')
    a.out.mkdir(parents=True)
    state = dict(schema='new-A-conditional-port-recovery-v1', pid=os.getpid(),
                 startticks=p.process_identity(os.getpid())['startticks'], job=job_ref,
                 started_s=time.time(), phase='waiting', complete=False, node_lease_held=False)
    p.save(a.out / 'status.json', state)
    try:
        for path, digest in job['files'].items():
            p.need(p.sha(path) == digest, 'recovery source changed: ' + path)
        predecessor = Path(job['predecessor_pipeline'])
        while True:
            prior = p.read(predecessor / 'status.json')
            if prior.get('finished_s') and not p.active_owner(prior):
                break
            time.sleep(30)
        if prior.get('complete'):
            state.update(complete=True, phase='predecessor_completed_without_recovery')
            return
        p.need(not prior.get('node_lease_held') and '8tp1' in prior.get('completed_baseline_stages', []),
               'failure is outside completed resident-stage boundary')
        failed_stage = Path(job['failed_stage'])
        stage = p.read(failed_stage / 'status.json')
        bootstrap = p.read(failed_stage / 'bootstrap-stage/status.json')
        for owner in (stage, bootstrap):
            p.need(owner.get('finished_s') and not p.active_owner(owner) and not owner.get('node_lease_held'),
                   'failed stage still active')
        log = failed_stage / 'create.log'
        collisions = [int(port) for port in re.findall(r'RuntimeError: new port is occupied: (\d+)', log.read_text())]
        old_ports = set(range(38200, 38206)) | set(range(58000, 58192))
        p.need(collisions and set(collisions) <= old_ports, 'failure is not the diagnosed original port preflight')
        p.need(bootstrap.get('error') in {repr(RuntimeError('new port is occupied: ' + str(port)))
                                        for port in collisions}, 'bootstrap terminal error differs')
        deployment = failed_stage / 'bootstrap-stage/deployment'
        p.need(not (deployment / 'deployment-receipt.json').exists() and not (failed_stage / 'qualification').exists(),
               'hardware/qualification already advanced; automatic port recovery forbidden')
        names = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'],
                               check=True, text=True, capture_output=True).stdout.splitlines()
        p.need(not any(name.startswith('pdb-v2-uniforma2l') for name in names),
               'heterogeneous creation already occurred')
        old_plan = p.checked(prior['plan'])
        p.need(not any(Path(path).exists() for path in old_plan['stop_paths']), 'explicit stop remains active')
        plan = copy.deepcopy(old_plan)
        plan.update(initial_observations=prior['observations'], declaration=prior['declaration'],
                    resume_resident_pipeline_terminal=p.ref(predecessor / 'status.json'))
        plan['baseline_stages'][-1]['ready'] = job['ready']['path']
        plan['files'].update(job['files'])
        plan['files'][job_ref['path']] = job_ref['sha256']
        p.need(plan['baseline_stages'][-1]['name'] == 'lb-distserve', 'unexpected final stage')
        module = p.load(job['pipeline_source'], 'newA_terminal_port_recovery_pipeline')
        module.restore_completed_resident_stage(plan, dict(declaration=plan['declaration'],
                                                           observations=plan['initial_observations']))
        ready = p.checked(job['ready'])
        for source in ready['sources']:
            p.need(p.sha(source['path']) == source['sha256'], 'final stage source changed')
        plan_path = Path(job['successor_plan'])
        p.need(not plan_path.exists() and not Path(job['successor_pipeline']).exists(), 'successor already exists')
        p.save(plan_path, plan)
        subprocess.run([sys.executable, '-B', job['pipeline_source']['path'], '--plan', str(plan_path),
                        '--out', job['successor_pipeline']], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.save(a.out / 'diagnosis.json', dict(
            predecessor=p.ref(predecessor / 'status.json'), failed_stage=p.ref(failed_stage / 'status.json'),
            bootstrap=p.ref(failed_stage / 'bootstrap-stage/status.json'), port_error_log=p.ref(log),
            original_colliding_ports=collisions, heterogeneous_containers_absent=True,
            original_deployment_receipt_absent=True, preserved_observations=prior['observations'],
            successor_plan=p.ref(plan_path), source_equivalence=job['port_source_equivalence']))
        with (a.out / 'successor.log').open('xb') as output:
            child = subprocess.Popen([sys.executable, '-B', job['pipeline_source']['path'],
                '--plan', str(plan_path), '--out', job['successor_pipeline'], '--run'],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        state.update(phase='successor_started', complete=True, successor_pid=child.pid,
                     successor_pipeline=job['successor_pipeline'], diagnosis=p.ref(a.out / 'diagnosis.json'))
        with (a.out / 'setup-index-watcher.log').open('xb') as output:
            watcher = subprocess.Popen([sys.executable, '-B', str(HERE / 'setup_index_watcher_v1.py'),
                '--stage', job['successor_stage'], '--out', job['successor_setup_metadata']],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        state['setup_index_watcher_pid'] = watcher.pid
    except BaseException as exc:
        state.update(error=repr(exc), phase='stopped_failure')
        raise
    finally:
        state['finished_s'] = time.time()
        p.save(a.out / 'status.json', state)


if __name__ == '__main__':
    main()
