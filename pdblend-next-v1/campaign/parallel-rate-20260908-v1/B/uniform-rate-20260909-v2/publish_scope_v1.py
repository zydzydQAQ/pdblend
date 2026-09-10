"""Publish complete 32B scope only after normal tail and independent metric repair."""
import argparse
import copy
import os
from pathlib import Path
import signal
import socket
import sys
import time

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p
import contract


def repaired(plan):
    old = p.checked(plan['original_observation'])
    new = p.checked(plan['replacement_observation'])
    p.need(old['checkpoint'] == new['checkpoint'] and old['cell_id'] == new['cell_id'], 'replacement changed physical observation')
    p.need(old['measurement_purpose'] == new['measurement_purpose'] == 'metric_supplement'
           and old['token_throughput_is_exact'] is False and new['token_throughput_is_exact'] is True,
           'not a closure of the declared metric gap')
    auditor = p.load(plan['metric_auditor'], 'scope_metric_independent_auditor')
    result = auditor.audit_checkpoint(old['checkpoint']['path'])
    p.need(result['checkpoint'] == old['checkpoint'] and result['token_throughput_is_exact']
           and result['zero_output_diagnosis']['passed'], 'independent metric diagnosis failed')
    for key, value in result.items():
        p.need(new.get(key) == value, 'replacement differs from independent raw recomputation: ' + key)
    unchanged = ('energy_j', 'slo_attainment', 'ttft_avg_s', 'tpot_avg_s', 'gpu_util',
        'request_throughput_rps', 'measurement_duration_s', 'work_complete', 'completed_work_requests',
        'good_requests', 'n_expected', 'failure_class', 'diagnosis_reference')
    for key in unchanged:
        p.need(old.get(key) == new.get(key), 'metric closure changed an existing result: ' + key)
    p.need(contract.acceptable_baseline(new), 'baseline qualification lost')
    return result


def finish(plan, tail, tail_ref):
    p.need(tail.get('measurements_complete') is True and tail['complete'] is False
           and tail['phase'] == 'awaiting_metric_closure' and tail['finished_s']
           and not tail.get('error') and not tail['node_lease_held'] and not p.active_owner(tail), 'normal tail is not terminal')
    p.need(tail['node'] == 'B' and tail['model'] == '32b' and tail['scope'] == 'five_systems'
           and tail['plan'] == plan['normal_tail_plan'], 'wrong normal tail')
    p.need(len(tail['observations']) == 24 and len(tail['attempts']) == 5
           and all(a.get('complete') for a in tail['attempts']), 'normal measurements missing')
    p.need(tail['metric_gap_observations'] == [plan['original_observation']], 'different pending metric gap')
    last = p.checked(tail['last_cell_status'])
    p.need(last['complete'] and not last.get('error') and not last['failed'] and last['finished_s']
           and not last['node_lease_held'] and not p.active_owner(last), 'last measurement not clean and exited')
    observations = [plan['replacement_observation'] if r == plan['original_observation'] else r for r in tail['observations']]
    p.need(sum(r == plan['replacement_observation'] for r in observations) == 1, 'repair must substitute exactly one derived audit')
    decisions = {}
    for dataset in contract.DATASETS:
        group = contract.resolve_group(tail['declaration'], '32b', dataset, actual_host='B')
        decision = contract.select_group(group, [p.checked(r) for r in observations if p.checked(r)['dataset'] == dataset])
        p.need(decision['phase'] == 'complete' and decision['five_system_complete'], 'five-system metrics/work scope incomplete: ' + dataset)
        decisions[dataset] = decision
    binding = p.checked(last['release'])['binding']
    value = copy.deepcopy(tail)
    value.update(schema='uniform-v2-five-system-pipeline-status', phase='complete', complete=True,
        five_system_complete=True, group_decisions=decisions, observations=observations, binding=binding,
        normal_tail_terminal=tail_ref, metric_repair=dict(original_observation=plan['original_observation'],
            replacement_observation=plan['replacement_observation'], metric_auditor=plan['metric_auditor'],
            original_checkpoint_preserved=True, no_new_measurement=True), published_s=time.time(),
        completion_note='All normal cells and the independently repaired exact token metric are complete; original raw checkpoints retained.')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reference = p.ref(args.plan)
    plan = p.checked(reference)
    p.need(socket.gethostname() == 'iZwz9i5bte3xkpmcoes3t2Z', 'wrong physical host')
    p.need(not args.out.exists() and not Path(plan['terminal_path']).exists(), 'new publisher and terminal required')
    args.out.mkdir(parents=True)
    state = dict(schema='uniform-v2-scope-publisher-status', plan=reference, pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], node_lease_held=False, complete=False,
        started_s=time.time(), phase='independent_metric_reaudit')
    def stop(*_):
        raise KeyboardInterrupt('scope publisher stopped')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        state['metric_reaudit'] = repaired(plan)
        state['phase'] = 'awaiting_normal_tail_terminal'
        p.save(args.out / 'status.json', state)
        while True:
            p.need(not any(Path(path).exists() for path in plan['stop_paths']), 'publisher stop requested')
            if Path(plan['normal_tail_status']).exists():
                tail = p.read(plan['normal_tail_status'])
                p.need(not tail.get('error'), 'normal tail failed; no complete scope publication')
                if tail.get('finished_s') and not p.active_owner(tail):
                    tail_ref = p.ref(plan['normal_tail_status'])
                    terminal = finish(plan, p.checked(tail_ref), tail_ref)
                    terminal['scope_publication_plan'] = reference
                    p.save(Path(plan['terminal_path']), terminal)
                    state.update(complete=True, phase='published', terminal=p.ref(plan['terminal_path']))
                    break
            time.sleep(10)
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        state['finished_s'] = time.time()
        p.save(args.out / 'status.json', state)


if __name__ == '__main__':
    main()
