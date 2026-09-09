"""Exact frozen pure power-provenance function; no backend import or hardware."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

HOST=Path('/root/workspace/pdblend-next-v1/releases/five-system100-B32B-v1-runtime/src/ecopadg')
SOURCES={str(HOST/'serving/measurement.py'):'4633928d62f9d17e22573452f6931d7b337aeb1bdd47aa0c4b25e8f093211db8',
    str(HOST/'measure/backends.py'):'6934ac5ca53e23994ee3c32d9bf356baa3d27fdc66ee1e6e789faf4a1225417b'}

def load():
    for p,h in SOURCES.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('frozen power evidence source changed')
    tree=ast.parse((HOST/'serving/measurement.py').read_text())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='power_evidence')
    bt=ast.parse((HOST/'measure/backends.py').read_text())
    constant=next(x.value for x in bt.body if isinstance(x,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='INSTANT_POWER_SOURCE_ID' for t in x.targets))
    ns={'math':math,'INSTANT_POWER_SOURCE_ID':ast.literal_eval(constant)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(HOST/'serving/measurement.py'),'exec'),ns)
    return ns['power_evidence']

def audit_raw(directory,files):
    directory=Path(directory)
    def raw(p):
        data=p.read_bytes();files[str(p.resolve())]=hashlib.sha256(data).hexdigest();return data
    rows=list(csv.DictReader(raw(directory/'power.csv').decode().splitlines()))
    values=[(float(r['t_s']),[float(r[f'gpu{i}_w']) for i in range(8)]) for r in rows]
    result=load()(values,json.loads(raw(directory/'power_source.json')),
        [json.loads(x) for x in raw(directory/'power_metadata.jsonl').splitlines()])
    if not result['power_source_verified']:raise RuntimeError('raw all-eight instant field provenance failed')
    files.update(SOURCES);return result
