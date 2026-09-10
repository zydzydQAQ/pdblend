"""Publish the fixed-stage handoff only after independent completed evidence replay."""
import argparse
import json
from pathlib import Path
import time

import bootstrap as b
import power_selftest as p
import verify_domain2100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification-dir', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    args = parser.parse_args()
    assert not args.handoff.exists()
    qualified = args.qualification_dir / 'qualified.json'
    while not qualified.exists():
        status = args.qualification_dir / 'status.json'
        if status.exists():
            state = p.read(status)
            assert not state.get('error') and state['phase'] != 'stopped_failure', 'qualification stopped with failure'
            assert not (state.get('finished_s') and not state['passed']), 'qualification finished without passing'
        time.sleep(5)
    reference = p.ref(qualified)
    result = verify_domain2100.verify(reference)
    assert result['passed'] and result['independently_recomputed']
    b.save(args.handoff, dict(node='Anew20260909', model='14b', system='pdblend',
        qualification=reference, qualification_validator=p.ref(p.HERE / 'verify_domain2100.py'),
        frequency_domain_mhz=[900, 1500, 2100], native_tp1_instances=2, dynamic_capacity_qualification=False,
        extra_files=[p.ref(Path(__file__))], predecessors=[]))
    print(json.dumps(dict(handoff=p.ref(args.handoff), verification=result)))


if __name__ == '__main__':
    main()
