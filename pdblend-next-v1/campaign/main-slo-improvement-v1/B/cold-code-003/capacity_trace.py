"""Freeze a declared tokenized development capacity trace, never a formal trace."""
import argparse
from pathlib import Path
from capacity_executor import durable, fixed, require, sha
from capacity_load_calibrate import generate_trace, ref


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--declaration',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    spec=fixed(dict(path=str(args.declaration),sha256=args.sha256))
    require(spec['schema']=='capacity-trace-declaration-v1' and spec['split']=='development',
            'declared development prompt shapes/phases required')
    templates=fixed(spec['templates'])['templates']
    trace=generate_trace(templates,spec['phases'],spec['seed'],spec['demand_domain_sha256'])
    trace['declaration']=ref(args.declaration)
    require(not args.out.exists(),'trace output must be fresh')
    durable(args.out,trace)
    print(sha(args.out))


if __name__=='__main__':
    main()
