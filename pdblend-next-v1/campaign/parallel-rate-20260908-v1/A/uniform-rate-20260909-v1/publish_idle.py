"""Publish the independently qualified, unchanged P12 idle-recovery configuration."""
import argparse
import json
from pathlib import Path

import bootstrap as b
import power_selftest as p
import verify_idle_budget2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    args = parser.parse_args()
    assert not args.handoff.exists()
    reference = p.ref(args.qualification)
    result = verify_idle_budget2.verify(reference)
    assert result['passed'] and result['independently_recomputed']
    b.save(args.handoff, dict(node='Anew20260909', model='14b', system='pdblend',
        qualification=reference, qualification_validator=p.ref(p.HERE / 'verify_idle_budget2.py'),
        frequency_domain_mhz=[900,1500,2100], idle_domain_reacquire_v1=True,
        idle_domain_reacquire_timeout_s=2.0, active_settle_timeout_s=.3,
        dynamic_capacity_qualification=False, extra_files=[p.ref(Path(__file__))], predecessors=[]))
    print(json.dumps(dict(handoff=p.ref(args.handoff), passed=True)))


if __name__ == '__main__':
    main()
