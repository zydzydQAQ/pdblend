from pathlib import Path
import ast
U=Path(__file__).resolve().parent;ROOT=U.parents[1]
source=(ROOT/'C/uniform-rate-20260909-v1/pipeline.py').read_text();tree=ast.parse(source)
functions=[]
for node in tree.body:
 if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['group_observations','resolve_group','child_command','run_rate']:
  functions.append(ast.get_source_segment(source,node))
s='''"""New-A uniform-v2: original dynamic PDB gate, one normal run, then declared baselines."""
import argparse,asyncio,fcntl,json,os,signal,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
'''+ '\n\n'.join(functions)+'\n'
s=s.replace("extra_files=handoff.get('extra_files', []), stop_paths=plan['stop_paths'])", "extra_files=handoff.get('extra_files', []), stop_paths=plan['stop_paths'],\n        measurement_purpose=rows[0].get('measurement_purpose','normal'))")
with (U/'pipeline.py').open('x') as stream:stream.write(s)
