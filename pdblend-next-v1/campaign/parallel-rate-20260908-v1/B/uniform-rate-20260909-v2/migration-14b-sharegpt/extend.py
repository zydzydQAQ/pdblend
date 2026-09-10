"""Isolated exact .25 ShareGPT declaration extension."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
import generate
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--request',type=Path,required=True);a=ap.parse_args();r=p.read(a.request)
    generate.append_point(r['declaration'],'14b','sharegpt',r['next_rate'],Path(r['out']))
