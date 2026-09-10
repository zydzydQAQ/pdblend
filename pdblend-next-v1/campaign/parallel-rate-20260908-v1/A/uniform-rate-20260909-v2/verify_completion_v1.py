"""Check terminal A assignment and already raw-audited observation completeness."""
import argparse
import fcntl
import math
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common/uniform-rate-20260909-v2'))
import support as p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    p.need(not a.out.exists(), 'fresh completion audit required')
    terminal = p.ref(a.pipeline / 'status.json')
    state = p.checked(terminal)
    plan = p.checked(state['plan'])
    p.need(socket.gethostname() == plan['expected_hostname'], 'wrong physical host')
    p.need(state['complete'] and state['phase'] == 'complete' and state.get('finished_s')
           and not state.get('error') and not state['node_lease_held'] and not p.active_owner(state),
           'pipeline has not terminated successfully')
    for path, digest in plan['files'].items():
        p.need(p.sha(path) == digest, 'frozen pipeline dependency changed: ' + path)
    last = p.checked(state['last_cell_status'])
    p.need(last['complete'] and not last['failed'] and not last.get('error')
           and last.get('finished_s') and not last['node_lease_held'] and not p.active_owner(last),
           'last measurement did not terminate cleanly')
    rows = [p.checked(ref) for ref in state['observations']]
    p.need(len(rows) == len({row['cell_id'] for row in rows}) == 61, 'unexpected or duplicate new observations')
    baselines = [row for row in rows if row['system'] != 'pdblend']
    p.need(len(baselines) == 52 and all(row['repeat'] == 1 for row in baselines), 'baseline count/repetition changed')
    metrics = ('energy_j', 'slo_attainment', 'ttft_avg_s', 'tpot_avg_s', 'token_throughput_tps', 'gpu_util')
    for row in rows:
        p.need(row['work_complete'] and row['token_throughput_is_exact'], 'incomplete work or unknown token evidence')
        p.need(all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) and row[key] >= 0
                   for key in metrics), 'missing or nonfinite metric')
        p.need(0 <= row['slo_attainment'] <= 1 and 0 <= row['gpu_util'] <= 1, 'invalid ratio')
        p.need(row['energy_measured_gpu_count'] == 8 and len(row['energy_per_gpu_j']) == 8
               and len(row['gpu_util_per_gpu']) == 8, 'incomplete eight-GPU metric coverage')
    pipeline = p.load(p.ref(HERE / 'pipeline_v4.py'), 'completion_pipeline_rules')
    contract = p.load(plan['contract'], 'completion_contract')
    decisions = {}
    expected = dict(alpaca=[1.5 * n for n in range(1, 9)], longbench=[.25 * n for n in range(1, 6)])
    for dataset, rates in expected.items():
        group = pipeline.resolve_group(plan, state, contract, dataset)
        decision = contract.select_group(group, [row for row in rows if row['dataset'] == dataset])
        p.need(decision['phase'] == 'complete', 'declared group remains incomplete: ' + dataset)
        for system in ('mixed', 'distserve', 'dynamollm', 'ecoserve'):
            observed = sorted(row['rate_rps'] for row in baselines if row['dataset'] == dataset and row['system'] == system)
            p.need(observed == rates, 'baseline grid mismatch: ' + dataset + '/' + system)
        decisions[dataset] = decision
    lock_path = Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')
    p.need(lock_path.is_file(), 'expected node lock is missing')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    setup_metadata = p.checked(p.ref(HERE / 'setup-energy-heterogeneous-002/metadata.json'))
    outcome = p.checked(p.ref(HERE / 'setup-energy-heterogeneous-001-outcome/metadata.json'))
    p.need(not outcome['measurement_occurred'] and not outcome['old_waiter']['active'], 'old unexecuted setup not resolved')
    p.save(a.out, dict(
        schema='new-A-uniform-v2-completion-audit-v1', passed=True, audited_s=time.time(),
        audit_scope='Terminal contract, all frozen pipeline dependencies, observation hashes, complete metrics and grid. Each observation was independently audited from raw evidence by its frozen measurement runner.',
        auditor=p.ref(__file__), terminal=terminal, plan=state['plan'], node=state['node'], model=state['model'],
        new_observation_count=len(rows), new_baseline_count=len(baselines), new_pdblend_count=len(rows)-len(baselines),
        baseline_grid=expected, normal_baselines_each_once=True, complete_eight_gpu_metrics=True,
        token_throughput_exact_for_all_new_observations=True, groups=decisions,
        observations=state['observations'], last_cell_status=state['last_cell_status'], node_lock_available_at_check=True,
        heterogeneous_setup_index=setup_metadata['setup_energy_index'],
        unexecuted_setup_outcome=p.ref(HERE / 'setup-energy-heterogeneous-001-outcome/metadata.json')))


if __name__ == '__main__':
    main()
