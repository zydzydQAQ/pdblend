"""Bounded invocation of the unchanged frozen B Cell; max_cells is a clean boundary."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'B32B-load-matrix-user-slo-v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def require(ok,reason):
    if not ok:raise RuntimeError(reason)

def verify():
    m=json.loads((ROOT/'manifest.json').read_text())
    for rel,digest in m['files'].items():require(sha(ROOT/rel)==digest,'adapter source changed: '+rel)
    for path,digest in m['frozen_references'].items():require(sha(path)==digest,'frozen B reference changed: '+path)
    require(m['first_invocation_max_cells']==1 and m['baseline_execution'] is False,'wrong scope')
    return m

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

async def execute(b,cell,queue,max_cells):
    task=asyncio.current_task();interrupted=False
    def stop():
        nonlocal interrupted
        if not interrupted:interrupted=True;task.cancel()
    loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,stop)
    return await queue.sweep(b,cell,max_cells=max_cells)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--max-cells',type=int,default=1);p.add_argument('--check',action='store_true');args=p.parse_args()
    require(args.max_cells>0,'max-cells must be positive');verify()
    if not any((BASE/'checkpoints').glob('*.json')):
        require(args.max_cells==1,'first invocation is authorized for exactly one cell')
    sys.path.insert(0,str(BASE));b=load('frozen_b_matrix',BASE/'run.py');queue=load('b_checkpoint_queue',ROOT/'queue_runner.py')
    if args.check:
        b.package_check();print(json.dumps(dict(adapter_valid=True,queue=queue.inspect_queue(b),baseline_execution=False)));return
    from execution import Cell
    require(Path(sys.modules[Cell.__module__].__file__).resolve()==BASE/'execution.py','wrong Cell import')
    from ecopadg.serving.campaign import node_lease
    with node_lease():asyncio.run(execute(b,Cell,queue,args.max_cells))
if __name__=='__main__':main()
