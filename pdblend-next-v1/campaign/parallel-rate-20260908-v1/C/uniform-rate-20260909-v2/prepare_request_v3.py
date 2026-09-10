"""New immutable preparation entry for native reference diagnosis compatibility."""
import argparse
from pathlib import Path
import sys
ROOT=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
import prepare_dispatch_v3
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--request',required=True);a.add_argument('--out',required=True);args=a.parse_args()
 request=p.read(args.request);request['kwargs'].setdefault('extra_files',[]).append(p.ref(__file__))
 p.save(args.out,prepare_dispatch_v3.prepare(**request['kwargs']))
