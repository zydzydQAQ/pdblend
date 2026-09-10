"""Fresh PDB continuation after the separate baseline stop; old STOP is retained."""
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'uniform-rate-20260909-v1'
spec = importlib.util.spec_from_file_location('uniform_B_priority_preparation', OLD / 'prepare_node.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.HERE = HERE
p = previous.p


def predecessor():
    status = p.read(OLD / 'pipeline-001/status.json')
    tail = p.read(OLD / 'predecessor-ecoserve-001/status.json')
    assert (OLD / 'STOP').exists(), 'the explicit baseline stop must remain preserved'
    for state in (status, tail):
        assert state.get('finished_s') and not state.get('node_lease_held') and not previous.alive(state['pid'])
    assert status['error'] == "AssertionError('stop requested')"
    assert not tail['failed'] and len(tail['completed']) == len(tail['observed_checkpoints']) == 1
    assert len(tail['attempted']) == 1, 'unexpected additional baseline attempted'
    sys.path.insert(0, str(previous.ROOT / 'common/uniform-rate-20260909-v1'))
    import metrics
    observation = metrics.audit_checkpoint(tail['observed_checkpoints'][0]['path'])
    assert observation['measurement_valid'] and observation['work_complete']
    value = dict(schema='uniform-PDB-priority-predecessor-v1', baseline_stop_preserved=p.ref(OLD / 'STOP'),
        prior_pipeline=p.ref(OLD / 'pipeline-001/status.json'), prior_last_cell=p.ref(OLD / 'predecessor-ecoserve-001/status.json'),
        last_cell_raw_audit=observation, all_prior_gpu_owners_stopped=True,
        incomplete_baseline_queue_not_claimed_complete=True, scope='continue shared authorized PDB work')
    path = HERE / 'predecessor.json'
    if path.exists():
        assert p.read(path) == value
    else:
        p.save(path, value)
    return p.ref(path)


previous.predecessor = predecessor

if __name__ == '__main__':
    assert len(sys.argv) > 1 and sys.argv[1].startswith('pdb-'), 'this continuation cannot start a baseline stage'
    previous.main()
