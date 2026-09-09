"""Claim an actually idle node and execute the whole paired campaign there."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import dispatcher
import protocol
import runner
import trace_source

HERE = Path(__file__).resolve().parent


def verify_package():
    package = runner.read(HERE / 'package.json')
    for path, digest in package['files'].items():
        protocol.require(trace_source.file_sha(path) == digest, 'frozen campaign file changed: ' + path)
    release = package['release']
    protocol.require(trace_source.file_sha(release['path']) == release['sha256'], 'release changed')
    return package, runner.read(release['path'])


def stage(release_path, name, proof_path, lease_fd, previous_binding=None):
    release = runner.read(release_path)
    result_path = Path(release['deployment_root']) / name / 'stage-result.json'
    protocol.require(not (Path(release['deployment_root']) / 'restore-result.json').exists(),
                     'host restoration was attempted; old stage bindings cannot be reused')
    if result_path.exists():
        result = runner.read(result_path)
        validate_stage_result(result, release_path, name)
        return result
    command = [sys.executable, '-u', str(HERE / 'stage_worker.py'), 'stage', '--stage', name,
               '--release', str(release_path), '--idle-proof', str(proof_path)]
    if previous_binding:
        command += ['--previous-binding', str(previous_binding)]
    logpath = HERE / 'execution' / (name + '-stage.log')
    with logpath.open('a') as log:
        process = subprocess.run(command, env=dict(os.environ, PDBLEND_NODE_LOCK_FD=str(lease_fd)),
                                 pass_fds=(lease_fd,), stdout=log, stderr=subprocess.STDOUT)
    protocol.require(result_path.exists(), 'stage produced no sealed result; exit ' + str(process.returncode))
    result = runner.read(result_path)
    protocol.require(process.returncode == 0 and result.get('measurement_valid') is True,
                     'fresh deployment/correctness stage failed: ' + str(result.get('error')))
    validate_stage_result(result, release_path, name)
    return result


def validate_stage_result(result, release_path, name):
    protocol.require(result.get('complete') is True and result.get('measurement_valid') is True,
                     'prior stage failed or is incomplete; diagnosis required')
    protocol.require(result.get('protocol_id') == protocol.PROTOCOL and result.get('stage') == name
                     and result.get('hostname') == socket.gethostname(), 'stage belongs to another protocol/host')
    reference = result.get('release', {})
    protocol.require(Path(reference.get('path', '')).resolve() == Path(release_path).resolve()
                     and reference.get('sha256') == trace_source.file_sha(release_path), 'stage release reference differs')
    expected = {'pdblend'} if name == 'pdblend' else set(protocol.BASELINES)
    protocol.require(set(result.get('bindings', {})) == expected, 'qualified stage has incomplete system bindings')
    for system, reference in result['bindings'].items():
        protocol.require(trace_source.file_sha(reference['path']) == reference['sha256'], 'qualified binding changed')
        binding = runner.read(reference['path'])
        protocol.require(binding.get('system') == system and binding.get('hostname') == socket.gethostname()
                         and binding.get('protocol_id') == protocol.PROTOCOL, 'actual binding host/system differs')


def run(claim_id, resume=False):
    claim_dir = HERE / 'claims' / claim_id
    claim_dir.mkdir(parents=True, exist_ok=True)
    status_path = claim_dir / 'status.json'
    status = dict(claim_id=claim_id, pid=os.getpid(), hostname=socket.gethostname(),
                  started_s=time.time(), status='checking', gpu_actions=False)

    def update(**values):
        status.update(values, updated_s=time.time())
        runner.save(status_path, status)

    update()
    lease = dispatcher.NODE_LOCK.open('a+')
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            update(status='declined', reason='node lease was claimed by another task')
            return 3
        proof = dispatcher.probe_local(owned_lock_fd=lease.fileno(), ignore_identities=dispatcher.current_ancestry())
        if not proof.get('eligible_for_owned_start'):
            update(status='declined', reason='node no longer idle after atomic lease acquisition', evidence=proof)
            return 3
        package, release = verify_package()
        execution = HERE / 'execution'
        execution.mkdir(parents=True, exist_ok=True)
        state_path, manifest_path = execution / 'state.json', execution / 'manifest.json'
        if state_path.exists():
            protocol.require(resume, 'existing campaign requires explicit resume, not a fresh dispatch')
            state = runner.reconcile(state_path)
            protocol.require(state['executing_host'] == socket.gethostname(), 'campaign cannot move hosts')
            protocol.require(state['status'] in ('ready', 'complete'), 'blocked attempt requires diagnosis')
        else:
            protocol.require(not manifest_path.exists(), 'incomplete manifest initialization requires diagnosis')
            manifest = protocol.declaration()
            manifest.update(created_s=time.time(), versions=release['versions'],
                executing_host=socket.gethostname(), execution_host=socket.gethostname(),
                release=package['release'], package_sha256=trace_source.file_sha(HERE / 'package.json'),
                dispatch_delay_max_limit_s=1., dispatch_delay_p99_limit_s=.1,
                dispatch_p99_method='linear interpolation at (n-1)*0.99',
                qualification_before_first_point=True)
            runner.save(manifest_path, manifest)
            state = runner.new_state(manifest_path)
            runner.save(state_path, state)
        proof_path = claim_dir / 'idle-proof.json'
        runner.save(proof_path, proof)
        update(status='claimed', idle_proof=str(proof_path), state_path=str(state_path),
               manifest_path=str(manifest_path), claimed_s=time.time())
        bindings = {}
        try:
            while state['status'] != 'complete':
                pending = runner.next_point(state)
                protocol.require(pending is not None, 'campaign blocked by invalid or incomplete evidence')
                system = pending[0]
                stage_name = 'pdblend' if system == 'pdblend' else 'baselines'
                if system not in bindings:
                    update(status='qualifying_' + stage_name, gpu_actions=True)
                    prior_pdb = bindings.get('pdblend') or state.get('stages', {}).get('pdblend', {}).get('bindings', {}).get('pdblend')
                    previous = prior_pdb.get('path') if prior_pdb and stage_name == 'baselines' else None
                    result = stage(package['release']['path'], stage_name, proof_path, lease.fileno(), previous)
                    bindings.update(result['bindings'])
                    state['stages'][stage_name] = result
                    runner.save(state_path, state)
                update(status='measuring', current=dict(system=system, scale=pending[1], rate=pending[2]))
                state = runner.execute_one(manifest_path, state_path, execution, bindings[system]['path'], lease.fileno())
                report = runner.make_report(manifest_path, state_path, execution,
                                            figures=state['status'] in ('blocked', 'complete'))
                protocol.require(report['totals']['valid_measurements'] == sum(r.get('status') == 'valid' for r in state['records']),
                                 'independent report rejects one or more sealed measurements')
                update(valid_points=sum(r.get('status') == 'valid' for r in state['records']),
                       scan=state['scan'], queue_status=state['status'])
                protocol.require(state['status'] != 'blocked', 'technical invalid point; retained for diagnosis')
            update(status='restoring', current=None)
        finally:
            # Restoration operates only on this task's recorded containers and
            # the exact original idle container IDs captured on first deploy.
            current_state = runner.read(state_path)
            for stage_name in ('pdblend', 'baselines'):
                retained_stage = Path(release['deployment_root']) / stage_name / 'stage-result.json'
                if retained_stage.exists():
                    current_state['stages'][stage_name] = runner.read(retained_stage)
            runner.save(state_path, current_state)
            first = Path(release['deployment_root']) / 'pdblend/deployment-receipt.json'
            if first.exists():
                command = [sys.executable, '-u', str(HERE / 'stage_worker.py'), 'restore',
                           '--release', package['release']['path']]
                with (execution / 'restore.log').open('a') as log:
                    restored = subprocess.run(command, env=dict(os.environ, PDBLEND_NODE_LOCK_FD=str(lease.fileno())),
                        pass_fds=(lease.fileno(),), stdout=log, stderr=subprocess.STDOUT)
                restoration_path = Path(release['deployment_root']) / 'restore-result.json'
                if restoration_path.exists():
                    current_state = runner.read(state_path)
                    current_state['restoration'] = dict(path=str(restoration_path),
                        sha256=trace_source.file_sha(restoration_path))
                    runner.save(state_path, current_state)
                protocol.require(restored.returncode == 0, 'original idle host restoration requires diagnosis')
        report = runner.make_report(manifest_path, state_path, execution, figures=True)
        protocol.require(report['totals']['campaign_complete'] is True,
                         'independent report has missing endpoints or same-trace baseline pairs')
        update(status='complete', finished_s=time.time(), completion=runner.read(state_path)['stop_reason'])
        return 0
    except BaseException as exc:
        if 'state_path' in locals() and state_path.exists() and manifest_path.exists():
            try:
                current_state = runner.read(state_path)
                current_state.update(execution_status='failed', execution_error=str(exc))
                runner.save(state_path, current_state)
                runner.make_report(manifest_path, state_path, execution, figures=True)
            except Exception as report_error:
                status['failure_report_error'] = str(report_error)
        update(status='failed', error=str(exc), traceback=traceback.format_exc(), finished_s=time.time())
        return 2
    finally:
        lease.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--claim-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    protocol.require(args.claim_id.replace('-', '').isalnum(), 'simple unique claim identity required')
    raise SystemExit(run(args.claim_id, args.resume))
