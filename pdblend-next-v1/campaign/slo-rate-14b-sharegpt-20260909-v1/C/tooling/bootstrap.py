from pathlib import Path
import json,hashlib,importlib.util
def checked(r):
 p=Path(r["path"]);assert hashlib.sha256(p.read_bytes()).hexdigest()==r["sha256"],str(p);return json.loads(p.read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n");t.replace(p)
def load(p,name):
 if isinstance(p,dict):checked_source=p;p=p["path"]
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
