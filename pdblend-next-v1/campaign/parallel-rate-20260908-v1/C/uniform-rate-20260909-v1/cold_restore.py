"""Retained C restart; only Mounts list order is additionally normalized."""
import copy

import support as p


def mount_equivalence(historical, actual):
    p.need(isinstance(historical, list) and isinstance(actual, list), 'mount list required')
    def keyed(rows):
        p.need(all(isinstance(x, dict) and isinstance(x.get('Destination'), str) for x in rows),
               'mount destination required')
        result = {x['Destination']: x for x in rows}
        p.need(len(result) == len(rows), 'duplicate mount destination is not equivalent')
        return result
    p.need(keyed(historical) == keyed(actual), 'mount fields changed')
    return dict(equivalent=True, rule='permutation_only_unique_Destination_all_fields_equal',
                historical=historical, actual=actual,
                array_order_changed=historical != actual)


def main():
    original = p.load(p.ROOT / 'C/ascending-resume-20260909-v1/cold_restore.py', 'uniform_C_cold_original')
    original_equivalence = original.container_equivalence
    def equivalent(historical, actual, dns):
        mounts = mount_equivalence(historical['Mounts'], actual['Mounts'])
        normalized = copy.deepcopy(actual)
        normalized['Mounts'] = copy.deepcopy(historical['Mounts'])
        hostconfig = original_equivalence(historical, normalized, dns)
        return dict(hostconfig=hostconfig, mounts=mounts)
    original.container_equivalence = equivalent
    original.main()


if __name__ == '__main__':
    main()
