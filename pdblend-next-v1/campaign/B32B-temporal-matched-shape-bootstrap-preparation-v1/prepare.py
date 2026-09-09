"""Call the reviewed pure-CPU derivation only, with exact source/actual inputs.

The full binder's unfrozen package_check/performance path is not invoked.
"""
import hashlib,importlib.util,json,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent;s=json.loads((ROOT/'inputs.json').read_text());p=ROOT/'derive.frozen.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()==s['derive_source_sha256']
sp=importlib.util.spec_from_file_location('sameflag_derive_frozen',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
a=SimpleNamespace(**{k:Path(v) if k in ('bootstrap','identity','out') else v for k,v in s.items() if k not in ('schema','derive_source_sha256','derive_source_original')})
r=m.derive_bootstrap(a)
(ROOT/'receipt.json').write_text(json.dumps(dict(cpu_only=True,source_sha256=s['derive_source_sha256'],no_package_performance_code_invoked=True,result=r,finished_s=time.time()),indent=2)+'\n');print(json.dumps(r))
