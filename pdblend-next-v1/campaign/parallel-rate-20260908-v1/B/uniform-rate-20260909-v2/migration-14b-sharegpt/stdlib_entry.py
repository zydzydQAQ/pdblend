"""Run an unchanged experiment entry with Python's queue module preloaded.

The migration directory contains a frozen historical queue declaration script;
this entry prevents it from shadowing the standard library thread queue.
"""
import os
import sys
from pathlib import Path

saved_path=list(sys.path)
stdlib=Path(os.__file__).resolve().parent
sys.path.insert(0,str(stdlib))
import queue
assert Path(queue.__file__).resolve()==stdlib/'queue.py'
assert hasattr(queue,'SimpleQueue')
sys.path[:]=saved_path
import runpy

def main():
    assert len(sys.argv)>=3 and sys.argv[1]=='--source'
    source=Path(sys.argv[2]).resolve()
    sys.argv=[str(source),*sys.argv[3:]]
    sys.path[0]=str(source.parent)
    runpy.run_path(str(source),run_name='__main__')

if __name__=='__main__':main()
