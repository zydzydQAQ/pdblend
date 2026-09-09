"""Freeze lifecycle-only wrappers for original A/C baseline binding and gate."""
from pathlib import Path
import ast,hashlib,json,time
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
OUT=HERE/'baseline-until-complete-v1';assert not OUT.exists();OUT.mkdir()
COMMON=REPO/'campaign/parallel-rate-20260908-v1/common/execution-until-complete-v1'
BINDER=REPO/'campaign/AC-baseline-binding-v2';GATE=REPO/'campaign/AC-legacy-resident-correctness-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=(BINDER/'bind.py').read_text()
s=s.replace('from gate_evidence import audit,files,read,require,sha',"sys.path.insert(0,"+repr(str(BINDER))+")\nfrom gate_evidence import audit,files,read,require,sha")
s=s.replace('ROOT=Path(__file__).resolve().parent','ROOT=Path('+repr(str(BINDER))+')')
s=s.replace("hostname=spec['hostname'],deadline_s=spec['deadline_s'],host_release=spec['host_release']", "hostname=spec['hostname'],deadline_s=spec['deadline_s'],campaign_lifecycle=spec['campaign_lifecycle'],host_release=spec['host_release']")
(OUT/'bind.py').write_text(s)
s=(GATE/'validate.py').read_text()
s=s.replace('from checks import Checks,require,write',"sys.path.insert(0,"+repr(str(GATE))+')\nfrom checks import Checks,require,write')
s=s.replace('ROOT=Path(__file__).resolve().parent','ROOT=Path('+repr(str(GATE))+')')
s=s.replace("COMMON=ROOT.parent/'five-system-execution-v2/run.py'",'COMMON=Path('+repr(str(COMMON/'run.py'))+')')
s=s.replace("COMMON_SHA='ddc634e0b826d1873ed0bb7e3bd9088ba1412476725d8ec1e371414ccce54ad2'",'COMMON_SHA='+repr(sha(COMMON/'run.py')))
s=s.replace("    require(time.time()+510<m.GLOBAL_DEADLINE,'insufficient global time for gate plus cleanup')\n",'')
s=s.replace('cleanup_end=time.monotonic()+max(0,min(90,m.GLOBAL_DEADLINE-time.time()))','cleanup_end=time.monotonic()+90')
(OUT/'validate.py').write_text(s)
for n in ('bind.py','validate.py'):compile((OUT/n).read_text(),str(OUT/n),'exec')
gateold=ast.parse((GATE/'validate.py').read_text());gatenew=ast.parse((OUT/'validate.py').read_text())
functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=functions(gateold),functions(gatenew)
assert all(a[n]==b[n] for n in a if n!='execute')
assert 'wait_for(checker.run(),390)' in (OUT/'validate.py').read_text()
assert 'cleanup_end=time.monotonic()+90' in (OUT/'validate.py').read_text()
assert 'GLOBAL_DEADLINE' not in (OUT/'validate.py').read_text()
old=functions(ast.parse((BINDER/'bind.py').read_text()));new=functions(ast.parse((OUT/'bind.py').read_text()))
assert all(old[n]==new[n] for n in old if n!='build')
manifest=dict(schema=1,created_s=time.time(),campaign_lifecycle='until_declared_complete_v1',deadline_s=None,
 files={str(OUT/n):sha(OUT/n) for n in ('bind.py','validate.py')},references={str(p):sha(p) for p in [BINDER/'bind.py',BINDER/'gate_evidence.py',GATE/'validate.py',GATE/'checks.py',COMMON/'run.py',COMMON/'manifest.json',COMMON/'cpu-validation.json',Path(__file__).resolve()]},
 baseline_policies_unchanged=True,mechanism_checks_byte_identical=True,correctness_work_budget_s=390,cleanup_budget_s=90,
 changes=['propagate required lifecycle into new bindings','remove task-wide reserve checks','retain full local gate cleanup budget'])
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(passed=True,manifest=str(OUT/'manifest.json'),sha256=sha(OUT/'manifest.json'),binder=str(OUT/'bind.py'),gate=str(OUT/'validate.py'))))
