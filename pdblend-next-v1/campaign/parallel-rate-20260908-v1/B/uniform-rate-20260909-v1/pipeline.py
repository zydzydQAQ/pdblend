"""Durable B32B continuation: old tail, qualified PDB grid, then paired baselines."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMON = ROOT / 'common/uniform-rate-20260909-v1'
RUNNER = ROOT / 'C/uniform-rate-20260909-v1'
sys.path.insert(0, str(RUNNER))
import support as p
sys.path.insert(0, str(COMMON))
import contract
import metrics
RESUME = HERE.parent / 'ascending-resume-20260909-v2'


def dynamic_reuse():
    report_path = HERE / 'predecessor-reuse-audit.json'
    index_path = HERE / 'predecessor-reuse-index.json'
    if index_path.exists():
        values = p.read(index_path)
        for reference in values:
            observation = p.checked(reference)
            p.checked(observation['audit_reference'])
            p.checked(observation['checkpoint'])
        return values
    sources = [HERE.parent / 'ascending-rate-v1/pdb-performance-001', RESUME / 'baseline-mixed-performance-001',
               RESUME / 'baseline-distserve-performance-001']
    sources += [HERE / ('predecessor-' + system + '-001') for system in ('distserve', 'dynamollm', 'ecoserve')]
    observations = []
    for directory in sources:
        state = p.read(directory / 'status.json')
        assert state.get('finished_s') and not state.get('node_lease_held') and not p.active_owner(state)
        for reference in state['observed_checkpoints']:
            cp = p.checked(reference)
            verifier = p.load(cp['qualification_validator'], 'uniform_B_reuse_qualification')
            qualified = verifier.verify(cp['qualification'])
            assert qualified['passed'] and qualified['independently_recomputed']
            assert p.checked(cp['binding'])['instances'] == p.checked(qualified['binding'])['instances']
            observation = metrics.audit_checkpoint(reference['path'])
            observation['independently_recomputed'] = True
            if not observation['work_complete']:
                tail = p.load(HERE / 'finish_predecessor.py', 'uniform_B_reuse_timeout')
                proof = tail.diagnose(reference['path'])
                diagnosis_path = HERE / 'reuse-diagnoses' / (observation['cell_id'] + '.json')
                assert not diagnosis_path.exists()
                p.save(diagnosis_path, proof)
                observation.update(failure_class='independently_diagnosed_capacity_deadline',
                    diagnosis_reference=p.ref(diagnosis_path))
            observations.append(observation)
    assert len(observations) == 10 and len({(r['system'], r['repeat']) for r in observations}) == 10
    assert all(r['model'] == '32b' and r['dataset'] == 'alpaca' and r['rate_rps'] == 4.5 for r in observations)
    p.save(report_path, dict(schema='uniform-B-dynamic-reuse-audit-v1', independently_recomputed=True,
        slo_threshold_comparison='strict_lt', observations=observations))
    references = []
    for observation in observations:
        observation = dict(observation, audit_reference=p.ref(report_path))
        path = HERE / 'reuse-observations' / (observation['cell_id'] + '.json')
        p.save(path, observation)
        references.append(p.ref(path))
    p.save(index_path, references)
    return references


class Pipeline:
    def __init__(self):
        self.out = HERE / 'pipeline-001'
        self.path = self.out / 'status.json'
        self.child = None
        self.stopping = False
        if self.path.exists():
            self.state = p.read(self.path)
            assert not p.active_owner(self.state) and not self.state.get('error'), 'prior pipeline needs diagnosis'
        else:
            self.out.mkdir(exist_ok=False)
            self.state = dict(schema='uniform-B-pipeline-v1', complete=False, phase='initializing',
                stages={}, observations=[], boundaries={},
                declaration=p.ref(COMMON / 'release-001/declaration.json'), extensions=[])
        self.state.update(pid=os.getpid(), startticks=p.process_identity(os.getpid())['startticks'],
                          started_s=time.time(), node_lease_held=False)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.stop)
        self.save()

    def save(self):
        self.state['updated_s'] = time.time()
        p.save(self.path, self.state)

    def stop(self, *_):
        self.stopping = True
        if self.child and self.child.poll() is None:
            self.child.terminate()

    def guard(self):
        assert not self.stopping and not (HERE / 'STOP').exists(), 'stop requested'

    def run(self, name, argv):
        self.guard()
        prior = self.state['stages'].get(name)
        if prior:
            assert prior.get('exitcode') == 0 and prior.get('finished_s'), 'partial stage requires diagnosis: ' + name
            return
        self.state['phase'] = name
        with (self.out / (name + '.log')).open('xb') as log:
            self.child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            record = dict(pid=self.child.pid, argv=argv, started_s=time.time())
            self.state['stages'][name] = record
            self.save()
            record.update(exitcode=self.child.wait(), finished_s=time.time())
        self.child = None
        self.save()
        self.guard()
        assert record['exitcode'] == 0, 'stage failed; preserve evidence: ' + name

    def finish_old_tail(self):
        for system in ('distserve', 'dynamollm', 'ecoserve'):
            status = HERE / ('predecessor-' + system + '-001/status.json')
            if status.exists():
                while True:
                    self.guard()
                    old = p.read(status)
                    if not p.active_owner(old):
                        assert old.get('complete') and not old.get('error') and not old.get('failed'), 'old tail stopped: ' + system
                        break
                    self.state['phase'] = 'waiting_predecessor_' + system
                    self.save()
                    time.sleep(5)
            else:
                self.run('predecessor_' + system, [sys.executable, '-B', str(HERE / 'finish_predecessor.py'), '--system', system, '--run'])
        self.state['dynamic_reuse_observations'] = dynamic_reuse()
        self.save()

    def phase(self, name):
        self.run(name.replace('-', '_'), [sys.executable, '-B', str(HERE / 'prepare_node.py'), name, '--run'])

    def refs_for(self, dataset, key='observations'):
        return [r for r in self.state.get(key, []) if p.checked(r)['dataset'] == dataset]

    def select(self, dataset):
        group = contract.resolve_group(self.state['declaration'], '32b', dataset, actual_host='B')
        extra = [p.checked(r) for r in self.refs_for(dataset, 'dynamic_reuse_observations')]
        if extra:
            group = contract.apply_audited_reuse(group, extra)
        observations = [p.checked(r) for r in self.refs_for(dataset)]
        return contract.select_group(group, observations)

    def measure(self, dataset, rate, system, tasks):
        key = dataset + '-r' + contract.number(rate).replace('.', 'p') + '-' + system
        entry = p.read(HERE / 'pdb-entry.json') if system == 'pdblend' else p.read(HERE / 'baseline-entries.json')[system]
        release_dir = HERE / 'releases' / key
        out = HERE / 'measurements' / key
        argv = [sys.executable, '-B', str(RUNNER / 'prepare_release.py'), '--declaration', self.state['declaration']['path'],
            '--qualification', entry['qualification']['path'], '--qualification-validator', entry['qualification_validator']['path'],
            '--node', 'B', '--model', '32b', '--dataset', dataset, '--system', system, '--rate', str(rate),
            '--out', str(release_dir), '--stop-path', str(HERE / 'STOP')]
        for task in tasks:
            argv += ['--repeat', str(task['repeat'])]
        for reference in self.refs_for(dataset):
            argv += ['--scheduling-observation', reference['path']]
        for reference in self.refs_for(dataset, 'dynamic_reuse_observations'):
            argv += ['--dynamic-reuse-observation', reference['path']]
        for reference in entry.get('predecessors', []):
            argv += ['--predecessor', reference['path']]
        self.run('prepare_' + key, argv)
        self.run('measure_' + key, [sys.executable, '-B', str(RUNNER / 'run_cells.py'), '--release',
            str(release_dir / 'release.json'), '--out', str(out), '--run'])
        status = p.read(out / 'status.json')
        assert status['complete'] and not status['failed'] and not status['node_lease_held']
        for reference in status['observations']:
            p.checked(reference)
            if reference not in self.state['observations']:
                self.state['observations'].append(reference)
        self.save()

    def extend(self, dataset, rate):
        generator = p.load(COMMON / 'generate.py', 'uniform_B_append')
        out = HERE / 'extensions' / (dataset + '-r' + contract.number(rate).replace('.', 'p'))
        generator.append_point(self.state['declaration'], '32b', dataset, rate, out)
        self.state['declaration'] = p.ref(out / 'declaration.json')
        self.state['extensions'].append(self.state['declaration'])
        self.save()

    def execute(self):
        self.finish_old_tail()
        for phase in ('pdb-restore', 'pdb-spec'):
            self.phase(phase)
        self.run('pdb_qualification', [sys.executable, '-B', str(HERE.parent / 'ascending-rate-v1/qualify_pdb_v1.py'),
            '--spec', str(HERE / 'pdb-qualification-spec.json'), '--out', str(HERE / 'pdb-qualification-001'), '--run'])
        self.phase('pdb-freeze')
        for dataset in contract.DATASETS:
            while True:
                self.guard()
                selected = self.select(dataset)
                self.state['current_group'] = dataset
                self.state['decision'] = selected
                self.save()
                if selected['phase'] in ('baselines', 'complete'):
                    self.state['boundaries'][dataset] = selected['cap_rate_rps']
                    self.save()
                    break
                if selected['phase'] == 'extension_declaration_required':
                    self.extend(dataset, selected['next_rate_rps_decimal'])
                    continue
                assert selected['phase'] == 'pdblend', 'PDB observation requires diagnosis'
                self.measure(dataset, selected['rate_rps'], 'pdblend', selected['next_tasks'])
        p.save(HERE / 'pdb-boundaries.json', dict(complete=True, groups=self.state['boundaries'], pipeline_status=str(self.path)))
        for phase in ('baseline-restore', 'baseline-gate', 'baseline-qualify', 'baseline-freeze'):
            self.phase(phase)
        for system in contract.SYSTEMS[1:]:
            for dataset in contract.DATASETS:
                while True:
                    selected = self.select(dataset)
                    assert selected['phase'] in ('baselines', 'complete'), 'baseline group requires diagnosis'
                    pending = [t for t in selected.get('baseline_tasks', []) if t['action'] == 'execute' and t['row']['system'] == system]
                    if not pending:
                        break
                    rate = pending[0]['row']['rate_rps']
                    tasks = [t for t in pending if t['row']['rate_rps'] == rate]
                    self.measure(dataset, rate, system, tasks)
        assert all(self.select(ds)['phase'] == 'complete' for ds in contract.DATASETS)
        self.state.update(complete=True, phase='complete')
        self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    assert args.run
    with (HERE / 'pipeline-owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = Pipeline()
        try:
            pipeline.execute()
        except BaseException as exc:
            pipeline.state.update(error=repr(exc), phase='stopped_for_diagnosis')
            raise
        finally:
            pipeline.state['finished_s'] = time.time()
            pipeline.save()
