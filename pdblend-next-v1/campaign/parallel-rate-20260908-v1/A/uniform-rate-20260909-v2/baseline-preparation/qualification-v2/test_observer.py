"""CPU-only observer equivalence and joined cancellation checks."""
import ast,asyncio,json,time
from pathlib import Path
import frequency

async def cancellation():
 completed=[]
 def work():time.sleep(.1);completed.append(True)
 task=asyncio.create_task(frequency.joined_thread(work));await asyncio.sleep(.01);task.cancel()
 try:await task
 except asyncio.CancelledError:pass
 else:raise AssertionError('cancellation lost')
 assert completed==[True]

def main():
 root=Path(__file__).parent;old=root.parent/'qualification/frequency.py'
 funcs=lambda p:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 previous,actual=funcs(old),funcs(root/'frequency.py')
 assert all(previous[k]==actual[k] for k in ('shapes','clock_window','check','paths'))
 case=dict(instance_id='cpu',frequency=2100,input_length=7168,batch=1,requests=[dict(request_id='r1',prompt_token_ids=[9707]*7168)],native_after={},loaded_clocks=[])
 record=frequency.case_record(case);assert record['request_ids']==['r1'] and 'requests' not in record
 assert len(json.dumps(record))<1024
 asyncio.run(cancellation())
 print('unchanged shape/numerical/clock gate; compact exact request identity; cancelled writer joined')
if __name__=='__main__':main()
