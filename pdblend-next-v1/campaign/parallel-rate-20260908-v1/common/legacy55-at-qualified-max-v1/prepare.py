import argparse,json
from pathlib import Path
from run import prepare
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--runtime-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--maximum',type=int,default=2400);p.add_argument('--heterogeneous',action='store_true');a=p.parse_args();print(json.dumps(prepare(a.binding,a.runtime_dir,a.out,heterogeneous=a.heterogeneous,maximum=a.maximum)))
