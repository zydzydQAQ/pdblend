"""Finish the authorized PDB-only scope without changing the frozen pipeline."""
import argparse
import json
import time
from pathlib import Path
import support as p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pipeline', type=Path, required=True)
    ap.add_argument('--priority', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    priority = p.ref(args.priority)
    p.need(p.checked(priority)['effective_systems'] == ['pdblend'], 'scope differs')
    p.need(not args.out.exists(), 'fresh guard directory required')
    args.out.mkdir(parents=True)
    report = dict(schema='uniform-pdblend-only-scope-completion-v1', priority=priority,
                  pipeline=str(args.pipeline), started_s=time.time(), complete=False)
    try:
        while True:
            state = p.read(args.pipeline / 'status.json')
            p.need(not state.get('error'), 'pipeline stopped before PDB scope completion')
            p.need(not state.get('finished_s'), 'unexpected prior pipeline terminal state')
            if state['phase'] == 'waiting_for_mixed_stage':
                break
            time.sleep(5)
        plan = p.checked(state['plan'])
        p.need(not Path(plan['system_handoffs']['mixed']).exists()
               and not Path(plan['stage_requests']['mixed']).exists(), 'baseline producer exists')
        contract = p.load(plan['contract'], 'pdb_scope_contract')
        observations = [p.checked(r) for r in state['observations']]
        decisions = {}
        for dataset in plan['dataset_order']:
            group = contract.resolve_group(state['declaration'], plan['model'], dataset,
                                           actual_host=plan['node'])
            dynamic = [p.checked(r) for r in plan.get('dynamic_reuse_observations', [])
                       if p.checked(r)['dataset'] == dataset]
            if dynamic:
                group = contract.apply_audited_reuse(group, dynamic)
            decision = contract.select_group(group, [o for o in observations if o['dataset'] == dataset])
            p.need(decision['phase'] in ('baselines', 'complete') and decision['decision']['cap_observed'],
                   'PDB has not reached a valid complete cap: ' + dataset)
            decisions[dataset] = decision
        terminal = p.checked(state['last_cell_status'])
        p.need(terminal['complete'] and not terminal['node_lease_held'] and terminal['finished_s']
               and not p.active_owner(terminal), 'measurement still active')
        p.save(args.out / 'pipeline-at-pdb-completion.json', state)
        report.update(declaration=state['declaration'], observations=state['observations'],
                      group_decisions=decisions, pipeline_at_pdb_completion=p.ref(args.out / 'pipeline-at-pdb-completion.json'),
                      pdb_complete=True, baseline_complete=False,
                      baseline_disposition='not scheduled per explicit user PDB-only scope',
                      last_measurement=state['last_cell_status'])
        p.save(args.out / 'completion-before-stop.json', report)
        stop = Path(plan['stop_paths'][0])
        p.need(not stop.exists(), 'another stop owner intervened')
        with stop.open('x') as stream:
            json.dump(dict(reason='PDB-only user scope completed; baseline stage withheld',
                           completion=p.ref(args.out / 'completion-before-stop.json'),
                           priority=priority, created_s=time.time()), stream, indent=2)
        for _ in range(30):
            state = p.read(args.pipeline / 'status.json')
            if state.get('finished_s') and not p.active_owner(state):
                break
            time.sleep(1)
        p.need(state.get('finished_s') and not p.active_owner(state), 'supervisor did not terminate')
        p.need('stop requested at stage boundary' in state.get('error', ''), 'unexpected supervisor exit')
        report.update(complete=True, supervisor_terminal=p.ref(args.pipeline / 'status.json'),
                      stop_reference=p.ref(stop))
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['finished_s'] = time.time()
        p.save(args.out / 'status.json', report)


if __name__ == '__main__':
    main()
