"""New immutable Eco execution adapters, original measurement engine unchanged."""
import ast,copy,hashlib,json
from pathlib import Path
C=Path(__file__).resolve().parent;OUT=C/'eco-drain37-v1';REPO=C.parents[2]
HOST=REPO/'releases/five-system100-C7B-baseline-eco-drain-v2-runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def source(p,s):ast.parse(s);p.write_text(s)
def main():
    code=OUT/'code';assert not code.exists();code.mkdir()
    old=C/'baseline-until-complete-v2/bind.py';s=old.read_text()
    s=s.replace("system='mixed' if not a.strategy", "system='ecoserve' if not a.strategy")
    s=s.replace("'ecoserve': '/root/workspace/pdblend-next-v1/releases/five-system100-C7B-baseline-v1-runtime'", "'ecoserve': "+repr(str(HOST)))
    s=s.replace("system='ecoserve' if not a.strategy else", "system='ecoserve' if not a.strategy else")
    source(code/'bind.py',s)
    base=read(C/'baseline-strategy-specs-after-gamma-002/ecoserve.json');base['host_release']=str(HOST)
    manifest=read(HOST/'manifest.json');base['files'].update({str(HOST/p):h for p,h in manifest['files'].items()});base['files'][str(HOST/'manifest.json')]=sha(HOST/'manifest.json')
    base['files'][str(OUT/'declaration.json')]=sha(OUT/'declaration.json')
    base['files'][str(code/'bind.py')]=sha(code/'bind.py')
    base['baseline_source_strategy']='ecoserve';write(OUT/'strategy-spec.json',base)
    bootstrap=copy.deepcopy(base);bootstrap['baseline_source_strategy']='correctness-only';write(OUT/'bootstrap-spec.json',bootstrap)
    s=(C/'run_boundary_baseline_after_gamma_v2.py').read_text()
    s=s.replace('HERE=Path(__file__).resolve().parent','HERE=Path('+repr(str(C))+')')
    s=s.replace("DECLARATION=HERE/'boundary-p4v2/declaration.json'","DECLARATION=HERE/'eco-drain37-v1/declaration.json'")
    s=s.replace("PACKAGE=HERE/'boundary-baseline-package-p4v2.json'","PACKAGE=HERE/'eco-drain37-v1/package.json'")
    s=s.replace("HERE/'boundary-baseline-gate-after-gamma-002'","HERE/'eco-drain37-v1/gate'")
    s=s.replace("require(cells and len({c['cell_id'] for c in cells})==len(cells)","require(len(cells)==37 and len({c['cell_id'] for c in cells})==37")
    s=s.replace("dict(row=row,binding=str(bp)","dict(row=row,declaration=dict(path=str(DECLARATION),sha256=DECLARATION_SHA),binding=str(bp)")
    source(code/'run.py',s)
    files={str(p):sha(p) for p in [__file__,OUT/'declaration.json',OUT/'cpu-validation.json',OUT/'quarantine.json',OUT/'strategy-spec.json',OUT/'bootstrap-spec.json',code/'bind.py',code/'run.py',C/'baseline-until-complete-v1/validate.py']}
    files.update(read(OUT/'declaration.json')['files']);files.update(base['files'])
    write(OUT/'package.json',dict(schema='eco37-execution-package-v1',files=files,measurement_executor_unchanged=True,source_only_drain_guard=True))
    print(json.dumps(dict(package_sha256=sha(OUT/'package.json'),files=len(files),binder_sha256=sha(code/'bind.py'),runner_sha256=sha(code/'run.py'))))
if __name__=='__main__':main()
