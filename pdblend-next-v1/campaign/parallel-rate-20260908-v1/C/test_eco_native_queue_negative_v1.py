"""Run the actual native admission method without constructing a GPU engine."""
import ast,asyncio,copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'/root/workspace/pdblend/.runtime-deps')
from aiohttp import web
C=Path(__file__).resolve().parent;REPO=C.parents[2]
ENGINE=REPO/'campaign/AC-baseline-deployment-v1/engines/C/engine.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class BodyReached(Exception):pass
class Request:
    async def json(self):raise BodyReached()
async def native():
    tree=ast.parse(ENGINE.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='completions')
    mod=ast.Module(body=[node],type_ignores=[]);space={'web':web};exec(compile(ast.fix_missing_locations(mod),str(ENGINE),'exec'),space)
    passed=[]
    for count,rejected in [(128,True),(129,True),(127,False),(0,False)]:
        obj=SimpleNamespace(accepting=True,error=None,streams={str(k):None for k in range(count)},config={})
        try:await space['completions'](obj,Request())
        except web.HTTPTooManyRequests as e:assert rejected and e.text=='bounded admission queue full';passed.append(f'native_{count}_refuses_exact_original_429')
        except BodyReached:assert not rejected;passed.append(f'native_{count}_proceeds_to_original_body')
    return passed
def main():
    results=asyncio.run(native());spec=importlib.util.spec_from_file_location('eco_negative',C/'audit_eco_native_queue_negative_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    bench=[dict(request_id='0',success='1'),dict(request_id='1',success='0',http_status='503',error='RuntimeError: HTTP 503: decode failed: bounded admission queue full',request_timeout='False',admission_rejection='',n_text_chunks='0',first_token_s='')]
    events=[]
    for k in ['0','1']:
        events.append(dict(kind='admission',client_request_id=k,plan=dict(routes=[dict(decode_id='x')])))
        events.append(dict(kind='request_timing',client_request_id=k,request_id=k,forward_started_s=10 if k=='0' else 11,stream_end_s=12,cleanup_end_s=11.1,hard_deadline_s=130))
    m.queue_proof(bench,events,128);results.append('unique_native_refusal_with_active_work_accepted')
    for key,value in [('error','frequency failure'),('http_status','429'),('request_timeout','True'),('admission_rejection','unknown'),('n_text_chunks','1')]:
        changed=copy.deepcopy(bench);changed[1][key]=value
        try:m.queue_proof(changed,events,128)
        except AssertionError:results.append('reject_'+key)
        else:raise AssertionError(key)
    print(json.dumps(dict(passed=True,count=len(results),checks=results,actual_engine_method_sha256=hashlib.sha256(ast.dump(next(n for n in ast.walk(ast.parse(ENGINE.read_text())) if isinstance(n,ast.AsyncFunctionDef) and n.name=='completions'),include_attributes=False).encode()).hexdigest(),sources={str(p):sha(p) for p in [Path(__file__),ENGINE,C/'audit_eco_native_queue_negative_v1.py']})))
if __name__=='__main__':main()
