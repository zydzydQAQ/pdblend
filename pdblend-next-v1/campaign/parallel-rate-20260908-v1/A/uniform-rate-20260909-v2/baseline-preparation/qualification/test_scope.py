"""CPU rejection controls for actual loaded clocks and exact original native gate scope."""
import ast,copy,json
from pathlib import Path
import frequency as q

def rejected(call):
 try:call()
 except (ValueError,AssertionError):return
 raise AssertionError('invalid evidence accepted')

def main():
 rows=[(i*.05,[2100]*8) for i in range(21)]
 good=q.clock_window(rows,[6,7],2100,.1,.9);assert good['samples']>=15
 bad=copy.deepcopy(rows);bad[8][1][7]=2070;rejected(lambda:q.clock_window(bad,[6,7],2100,.1,.9))
 rejected(lambda:q.clock_window([r for r in rows if not .25<r[0]<.7],[6],2100,.1,.9))
 rejected(lambda:q.clock_window(rows[:5],[6],2100,.1,.9))
 proof=json.loads((Path(__file__).parent/'native-source-equivalence.json').read_text())
 for item in proof.values():
  before=ast.parse(Path(item['original']['path']).read_text());after=ast.parse(Path(item['actual']['path']).read_text())
  funcs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  b,a=funcs(before),funcs(after);assert {k for k in b if b[k]!=a[k]}=={'execute'}
  for k in ('validate_scope','mechanism_gates'):assert b[k]==a[k]
 print('CPU native scope/source and rank-clock/gap/truncated-window rejection checks passed')
if __name__=='__main__':main()
