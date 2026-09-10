"""Only the authorized PDB grids; retain all baseline outcomes and stop markers."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'uniform-rate-20260909-v1'
spec = importlib.util.spec_from_file_location('uniform_B_priority_pipeline', OLD / 'pipeline.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.HERE = HERE
p, c, m = base.p, base.contract, base.metrics


def pdb_reuse():
    destination = HERE / 'pdb-reuse-audit.json'
    index = HERE / 'pdb-reuse-index.json'
    if index.exists():
        values = p.read(index)
        for reference in values:
            v = p.checked(reference)
            p.checked(v['audit_reference'])
            p.checked(v['checkpoint'])
        return values
    source = HERE.parent / 'ascending-rate-v1/pdb-performance-001/status.json'
    state = p.read(source)
    assert state['complete'] and not state['failed'] and not p.active_owner(state)
    values = []
    for reference in state['observed_checkpoints']:
        cp = p.checked(reference)
        verifier = p.load(cp['qualification_validator'], 'uniform_priority_PDB_saved')
        result = verifier.verify(cp['qualification'])
        assert result['passed'] and result['independently_recomputed']
        value = m.audit_checkpoint(reference['path'])
        assert value['work_complete'] and value['measurement_valid']
        value.update(completed_work_throughput_rps=value['request_throughput_rps'],
                     generated_token_throughput_tps=value['token_throughput_tps'], independently_recomputed=True)
        values.append(value)
    assert len(values) == 2 and any(v['slo_attainment'] < .9 for v in values)
    p.save(destination, dict(schema='uniform-PDB-priority-reuse-audit-v1', slo_threshold_comparison='strict_lt', observations=values))
    references = []
    for value in values:
        path = HERE / 'reuse-observations' / (value['cell_id'] + '.json')
        p.save(path, dict(value, audit_reference=p.ref(destination)))
        references.append(p.ref(path))
    p.save(index, references)
    return references


class Pipeline(base.Pipeline):
    def execute(self):
        self.state.update(scope='pdblend_only_priority', baseline_queue_not_resumed=True,
                          dynamic_reuse_observations=pdb_reuse())
        self.save()
        for phase in ('pdb-restore', 'pdb-spec'):
            self.phase(phase)
        self.run('pdb_qualification', [sys.executable, '-B', str(HERE.parent / 'ascending-rate-v1/qualify_pdb_v1.py'),
            '--spec', str(HERE / 'pdb-qualification-spec.json'), '--out', str(HERE / 'pdb-qualification-001'), '--run'])
        self.phase('pdb-freeze')
        for dataset in c.DATASETS:
            while True:
                self.guard()
                selected = self.select(dataset)
                self.state.update(current_group=dataset, decision=selected)
                self.save()
                if selected['phase'] in ('baselines', 'complete'):
                    self.state['boundaries'][dataset] = selected['cap_rate_rps']
                    self.save()
                    break
                if selected['phase'] == 'extension_declaration_required':
                    self.extend(dataset, selected['next_rate_rps_decimal'])
                    continue
                assert selected['phase'] == 'pdblend', 'PDB measurement requires diagnosis'
                self.measure(dataset, selected['rate_rps'], 'pdblend', selected['next_tasks'])
        assert all(self.select(ds)['phase'] in ('baselines', 'complete') for ds in c.DATASETS)
        self.state.update(complete=True, pdb_complete=True, five_system_complete=False, phase='pdb_complete')
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
