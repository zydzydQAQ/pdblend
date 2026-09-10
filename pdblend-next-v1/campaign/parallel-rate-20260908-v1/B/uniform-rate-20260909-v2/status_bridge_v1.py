"""Publish the verified full-scope terminal at the report's status.json index."""
import copy
import os
from pathlib import Path
import sys
import time

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
U = ROOT / 'B/uniform-rate-20260909-v2'
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p
import contract


def validate(value):
    p.need(value['complete'] and value['five_system_complete'] and not value.get('error')
           and value['finished_s'] and not value['node_lease_held'] and not p.active_owner(value), 'full scope owner not clean')
    p.need(value['node'] == 'B' and value['model'] == '32b' and value['scope'] == 'five_systems'
           and set(value['datasets']) == set(contract.DATASETS), 'wrong scope/model/datasets')
    tail = p.checked(value['normal_tail_terminal'])
    p.need(tail['measurements_complete'] and not tail.get('error') and tail['finished_s']
           and not tail['node_lease_held'] and not p.active_owner(tail), 'normal tail remains active')
    last = p.checked(value['last_cell_status'])
    p.need(last['complete'] and not last.get('error') and not last['failed'] and last['finished_s']
           and not last['node_lease_held'] and not p.active_owner(last), 'last measurement not clean')
    for dataset in contract.DATASETS:
        group = contract.resolve_group(value['declaration'], '32b', dataset, actual_host='B')
        result = contract.select_group(group, [p.checked(r) for r in value['observations'] if p.checked(r)['dataset'] == dataset])
        p.need(result['phase'] == 'complete' and result['five_system_complete'], 'incomplete dataset: ' + dataset)


def main():
    source = U / 'full-scope-terminal-002.json'
    destination = U / 'full-scope-complete-002/status.json'
    p.need(not destination.exists(), 'status publication must be fresh')
    while not source.exists():
        p.need(not (U / 'STOP-scope002').exists(), 'scope status publication stopped')
        time.sleep(10)
    reference = p.ref(source)
    value = p.checked(reference)
    validate(value)
    result = copy.deepcopy(value)
    result.update(full_scope_terminal=reference, status_index_publisher=p.ref(__file__), status_index_published_s=time.time())
    p.save(destination, result)


if __name__ == '__main__':
    main()
