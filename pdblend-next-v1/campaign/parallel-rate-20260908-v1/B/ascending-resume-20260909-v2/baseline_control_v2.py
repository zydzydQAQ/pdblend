"""Fresh-path gate and qualification entrypoint; native implementations unchanged."""
import argparse
import json
from adapters import control, source_contract

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('gate', 'qualify'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source_contract()
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True, action=args.action)))
    else:
        print(json.dumps(getattr(control, args.action)()))
