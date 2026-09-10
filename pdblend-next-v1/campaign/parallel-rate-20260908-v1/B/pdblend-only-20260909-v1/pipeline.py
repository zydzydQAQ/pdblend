"""Only B32B PDBlend fixed-rate groups; baseline queue remains retired."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
from pdb_only_support import HERE, OLD, UNIFORM, p, contract, pdb_reuse, source_contract, validate, retired_evidence

base = p.load(UNIFORM / 'pipeline.py', 'B_pdb_only_original_pipeline')
base.HERE = HERE


class Pipeline(base.Pipeline):
    def __init__(self):
        assert not (HERE / 'pipeline-001').exists(), 'one fresh attempt only; never auto-resume unknown failure'
        super().__init__()
        self.state.update(schema='B32B-pdblend-only-pipeline-v1', manifest=p.ref(HERE / 'manifest.json'),
            baseline_policy='preserve existing outcomes; no additional baseline cells',
            user_priority=source_contract()['user_priority'])
        self.save()

    def guard(self):
        super().guard()
        source_contract()

    def measure(self, dataset, rate, system, tasks):
        assert system == 'pdblend' and all(t['row']['system'] == 'pdblend' for t in tasks)
        return super().measure(dataset, rate, system, tasks)

    def execute(self):
        retired_evidence(probe_processes=True)
        self.state['dynamic_reuse_observations'] = pdb_reuse()
        self.save()
        for phase in ('pdb-restore', 'pdb-spec'):
            self.phase(phase)
        self.run('pdb_qualification', [sys.executable, '-B', str(OLD / 'qualify_pdb_v1.py'),
            '--spec', str(HERE / 'pdb-qualification-spec.json'), '--out', str(HERE / 'pdb-qualification-001'), '--run'])
        self.phase('pdb-freeze')
        for dataset in contract.DATASETS:
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
                assert selected['phase'] == 'pdblend', 'PDB observation requires diagnosis'
                self.measure(dataset, selected['rate_rps'], 'pdblend', selected['next_tasks'])
        assert set(self.state['boundaries']) == set(contract.DATASETS)
        p.save(HERE / 'pdb-boundaries.json', dict(complete=True, systems=['pdblend'],
            groups=self.state['boundaries'], baseline_runs_skipped_by_user=True,
            pipeline_status=str(self.path), user_priority=source_contract()['user_priority']))
        self.state.update(complete=True, phase='all_B_PDB_groups_capped', baseline_suite_complete=False)
        self.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    report = validate()
    if not args.run:
        print(json.dumps(report, indent=2))
        return
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


if __name__ == '__main__':
    main()
