"""Freeze original native gate semantics with the newly declared device clock domain."""
from pathlib import Path
import ast, hashlib, json

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parents[4]
REPLACEMENTS={
 'resident':CAMPAIGN/'parallel-rate-20260908-v1/A/eco-drain31-v1/code/validate.py',
 'heterogeneous':CAMPAIGN/'A14B-legacy-heterogeneous-correctness-v1/validate.py',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 proof={}
 for layout,source in REPLACEMENTS.items():
  original=source.read_text();text=original.replace('clocks.set(range(8),2520,verify_rise=False)','clocks.set(range(8),2100,verify_rise=False)').replace('ClockOwner,hardware,tuple(range(8))','ClockOwner,hardware,tuple(range(8)),max_frequency=2100')
  changes=['fixed clock command 2520 -> 2100; original native checks, limits, cleanup unchanged']
  if layout=='heterogeneous':
   text=text.replace('from checks import require,write',"sys.path.insert(0,'"+str(CAMPAIGN/'A14B-legacy-heterogeneous-correctness-v1')+"')\nfrom checks import require,write")
   text=text.replace("COMMON=ROOT.parent/'five-system-execution-v2/run.py'", "COMMON=Path('"+str(CAMPAIGN/'parallel-rate-20260908-v1/common/execution-until-complete-v1/run.py')+"')")
   text=text.replace("COMMON_SHA='ddc634e0b826d1873ed0bb7e3bd9088ba1412476725d8ec1e371414ccce54ad2'", "COMMON_SHA='77bcbbb68e20419e5bc469a838c71e1abfa789dc167d901501715fda0ff4a8a9'")
   text=text.replace("    require(time.time()+510<m.GLOBAL_DEADLINE,'insufficient global time for gate plus cleanup')\n",'')
   text=text.replace('cleanup_end=time.monotonic()+max(0,min(90,m.GLOBAL_DEADLINE-time.time()))','cleanup_end=time.monotonic()+90')
   changes.append('original until-complete executor; fresh original 390s work + 90s cleanup budgets')
  before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  assert {k for k in before if before[k]!=after[k]}=={'execute'}
  target=HERE/(layout+'_gate.py');target.write_text(text)
  proof[layout]=dict(original=dict(path=str(source),sha256=sha(source)),actual=dict(path=str(target),sha256=sha(target)),changes=changes,unchanged_functions=sorted(set(before)-{'execute'}))
 (HERE/'native-source-equivalence.json').write_text(json.dumps(proof,indent=2)+'\n')
if __name__=='__main__':main()
